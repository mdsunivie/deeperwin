"""
Logic for Markov chain Monte Carlo (MCMC) steps.
"""

import copy
import functools
from typing import Dict
import jax
import jax.numpy as jnp
import numpy as np
import chex
from dataclasses import field

from deeperwin.configuration import MCMCConfig, MCMCLangevinProposalConfig, PhysicalConfig, LocalStepsizeProposalConfig
from deeperwin.utils.periodic import project_into_first_unit_cell
from deeperwin.utils.utils import get_el_ion_distance_matrix, pmap, pmean, batch_rng_split, merge_from_devices
from deeperwin.orbitals import initialize_walkers_with_exponential_radial_pdf


@chex.dataclass
class MCMCState:
    """
    Dataclasss that holds an electronic configuration and metadata required for MCMC.
    """

    r: jnp.array  # [batch-size x n_el x 3]
    R: jnp.array  # [n_ions x 3 ]
    Z: jnp.array  # [n_ions]
    log_psi_sqr: jnp.array = None
    walker_age: jnp.array = None  # [batch-size]; dtype=int
    rng_state: jnp.array = None
    stepsize: jnp.array = field(default_factory=lambda: jnp.array(1e-2))
    step_nr: jnp.array = field(default_factory=lambda: jnp.array(0, dtype=int))
    acc_rate: jnp.array = field(default_factory=lambda: jnp.array(0.0))

    def build_batch(self, fixed_params: Dict):
        return self.r, self.R, self.Z, fixed_params

    @classmethod
    def initialize_around_nuclei(
        cls, n_walkers, physical_config: PhysicalConfig, init_method, spin_initialization, rng
    ):
        rng_r, rng_spin, rng = jax.random.split(rng, 3)

        # Assign electrons to nuclei, according to the spin_initialization method
        if spin_initialization == "el_ion_mapping":
            el_ion_mapping = np.array(physical_config.el_ion_mapping, int)
        elif spin_initialization == "random":
            el_ion_mapping = np.concatenate([[i] * Z for i, Z in enumerate(physical_config.Z)])
            rng_spin = jax.random.split(rng_spin, n_walkers)
            el_ion_mapping = jax.vmap(lambda k: jax.random.permutation(k, el_ion_mapping))(rng_spin)
        elif spin_initialization == "ordered":
            el_ion_mapping = np.concatenate([[i] * Z for i, Z in enumerate(physical_config.Z)])
        elif spin_initialization == "mixed":
            el_ion_mapping = np.array(physical_config.el_ion_mapping, int)
            el_ion_mapping = np.tile(el_ion_mapping, (n_walkers // 2, 1))
            el_ion_map_ordered = np.concatenate([[i] * Z for i, Z in enumerate(physical_config.Z)])
            el_ion_map_ordered = np.tile(el_ion_map_ordered, (n_walkers - len(el_ion_mapping), 1))
            el_ion_mapping = np.concatenate([el_ion_mapping, el_ion_map_ordered], axis=0)
        else:
            raise NotImplementedError(f"Unknown spin initialization: {spin_initialization}")

        # Place electrons around the assigned nuclei, based on the initialization method
        if init_method == "gaussian":
            r0 = jax.random.normal(rng_r, [n_walkers, physical_config.n_electrons, 3])
            offset = np.array(physical_config.R)[el_ion_mapping]
            r0 += offset
        elif init_method == "exponential":
            assert (
                spin_initialization == "el_ion_mapping"
            ), "Spin initialization '{spin_initialization}' currently not implemented for exponential initialization"
            r0 = initialize_walkers_with_exponential_radial_pdf(
                rng_r,
                physical_config.R,
                physical_config.Z,
                n_walkers,
                physical_config.n_electrons,
                physical_config.n_up,
                physical_config.el_ion_mapping,
            )
        else:
            raise NotImplementedError(f"Unknown initialization: {init_method}")
        return cls(
            r=jnp.array(r0),
            R=jnp.array(physical_config.R),
            Z=jnp.array(physical_config.Z),
            log_psi_sqr=-jnp.ones(n_walkers)
            * 1000,  # initialize walkers with very low probability; will always be accepted in first MCMC move,
            walker_age=jnp.zeros(n_walkers, dtype=int),
            rng_state=jax.random.split(rng, n_walkers),
        )

    @classmethod
    def resize_or_init(cls, mcmc_state, mcmc_config: MCMCConfig, physical_config: PhysicalConfig, rng):
        if mcmc_state:
            if mcmc_state.r.ndim == 4:
                mcmc_state = mcmc_state.merge_devices()
            mcmc_state = resize_nr_of_walkers(mcmc_state, mcmc_config.n_walkers)
        else:
            mcmc_state = cls.initialize_around_nuclei(
                mcmc_config.n_walkers, physical_config, mcmc_config.initialization, mcmc_config.spin_initialization, rng
            )
        return mcmc_state

    def split_across_devices(self):
        assert self.r.ndim == 3, "State is already split across devices"
        assert (
            len(self.r) % jax.device_count() == 0
        ), f"Number of samples ({len(self.r)}) is not evenly divisible across devices ({jax.device_count()})"
        n_samples_per_device = len(self.r) // jax.device_count()

        def _split(x):
            x = x.reshape((jax.process_count(), jax.local_device_count(), n_samples_per_device, *x.shape[1:]))
            return x[jax.process_index()]

        def _tile(x):
            return jnp.tile(x, [jax.local_device_count()] + [1] * x.ndim)

        return MCMCState(
            r=_split(self.r),
            log_psi_sqr=_split(self.log_psi_sqr),
            walker_age=_split(self.walker_age),
            rng_state=_split(self.rng_state),
            R=_tile(self.R),
            Z=_tile(self.Z),
            stepsize=_tile(self.stepsize),
            step_nr=_tile(self.step_nr),
            acc_rate=_tile(self.acc_rate),
        )

    def merge_devices(self):
        if self.r.ndim == 3:
            return self  # already merged
        assert self.r.ndim == 4, "State is not split across devices"

        return MCMCState(
            r=merge_from_devices(self.r),
            log_psi_sqr=merge_from_devices(self.log_psi_sqr),
            walker_age=merge_from_devices(self.walker_age),
            rng_state=merge_from_devices(self.rng_state),
            R=self.R[0],
            Z=self.Z[0],
            stepsize=self.stepsize[0],
            step_nr=self.step_nr[0],
            acc_rate=self.acc_rate[0],
        )


MCMC_BATCH_AXES = MCMCState(
    r=0, R=None, Z=None, log_psi_sqr=0, walker_age=0, rng_state=0, stepsize=None, step_nr=None, acc_rate=None
)


def _resize_array(x, new_length):
    old_length = x.shape[0]
    if new_length < old_length:
        return x[:new_length]
    else:
        n_replicas = new_length // old_length
        n_remainder = new_length % old_length
        return jnp.concatenate([x for _ in range(n_replicas)] + [x[:n_remainder, ...]], axis=0)


def resize_nr_of_walkers(state: MCMCState, n_walkers_new):
    if n_walkers_new == len(state.r):
        return state
    new_state = copy.deepcopy(state)
    new_state.r = _resize_array(state.r, n_walkers_new)
    new_state.log_psi_sqr = _resize_array(state.log_psi_sqr, n_walkers_new)
    new_state.walker_age = _resize_array(state.walker_age, n_walkers_new)
    new_state.rng_state = jax.random.split(state.rng_state[0], n_walkers_new)
    return new_state


@functools.partial(jax.vmap, in_axes=(MCMC_BATCH_AXES,), out_axes=(MCMC_BATCH_AXES, 0))
def _propose_normal(state: MCMCState):
    new_state = copy.copy(state)
    new_state.rng_state, subkey = jax.random.split(state.rng_state)
    new_state.r += jax.random.normal(subkey, state.r.shape) * state.stepsize
    return new_state, 0.0


@functools.partial(jax.vmap, in_axes=(MCMC_BATCH_AXES,), out_axes=(MCMC_BATCH_AXES, 0))
def _propose_normal_one_el(state: MCMCState):
    n_el = state.r.shape[-2]
    index = state.step_nr % n_el

    new_state = copy.copy(state)
    new_state.rng_state, subkey = jax.random.split(state.rng_state)
    new_state.r = new_state.r.at[..., index, :].add(
        jax.random.normal(subkey, state.r[..., index, :].shape) * state.stepsize
    )
    return new_state, 0.0


@functools.partial(jax.vmap, in_axes=(MCMC_BATCH_AXES,), out_axes=(MCMC_BATCH_AXES, 0))
def _propose_cauchy(state: MCMCState):
    new_state = copy.copy(state)
    new_state.rng_state, subkey = jax.random.split(state.rng_state)
    new_state.r += jax.random.cauchy(subkey, state.r.shape) * state.stepsize
    return new_state, 0.0


def _calculate_local_stepsize(d_to_closest_ion, stepsize, r_min=0.1, r_max=2.0):
    return jnp.clip(d_to_closest_ion, r_min, r_max) * stepsize


def _calculate_stepsize_and_langevin_bias(r, R, Z, stepsize, mcmc_langevin_scale, mcmc_r_min, mcmc_r_max):
    el_ion_diff, el_ion_dist = get_el_ion_distance_matrix(r, R)
    dist_closest = jnp.min(el_ion_dist, axis=-1, keepdims=True)
    g_r = -mcmc_langevin_scale * jnp.sum(
        el_ion_diff * Z[..., np.newaxis] / el_ion_dist[..., np.newaxis], axis=-2
    )  # sum over ions
    s = stepsize * jnp.clip(dist_closest, mcmc_r_min, mcmc_r_max)
    return s, g_r


@functools.partial(jax.vmap, in_axes=(MCMC_BATCH_AXES,), out_axes=(MCMC_BATCH_AXES, 0))
def _propose_step_local_stepsize(state: MCMCState, config: LocalStepsizeProposalConfig):
    new_state = copy.copy(state)
    new_state.rng_state, subkey = jax.random.split(state.rng_state)

    dist_closest = jnp.min(get_el_ion_distance_matrix(state.r, state.R)[1], axis=-1)
    s = state.stepsize * jnp.clip(dist_closest, config.r_min, config.r_max)
    new_state.r += jax.random.normal(subkey, state.r.shape) * s[..., None]

    dist_closest = jnp.min(get_el_ion_distance_matrix(new_state.r, state.R)[1], axis=-1)
    s_new = state.stepsize * jnp.clip(dist_closest, config.r_min, config.r_max)

    dist_sqr = jnp.sum((new_state.r - state.r) ** 2, axis=-1)
    log_q_ratio = 3 * (jnp.log(s) - jnp.log(s_new))
    log_q_ratio += 0.5 * dist_sqr * (1 / s**2 - 1 / s_new**2)
    log_q_ratio = jnp.sum(log_q_ratio, axis=-1)  # sum over electrons
    return new_state, log_q_ratio


@functools.partial(jax.vmap, in_axes=(MCMC_BATCH_AXES,), out_axes=(MCMC_BATCH_AXES, 0))
def _propose_step_local_stepsize_one_el(state: MCMCState, config: LocalStepsizeProposalConfig):
    new_state = copy.copy(state)
    new_state.rng_state, subkey = jax.random.split(state.rng_state)
    n_el = state.r.shape[-2]
    index = state.step_nr % n_el

    # new_state.r += jax.random.normal(subkey, state.r.shape) * s[..., None]
    dist_closest = jnp.min(get_el_ion_distance_matrix(state.r, state.R)[1], axis=-1).at[..., index].get()
    s = state.stepsize * jnp.clip(dist_closest, config.r_min, config.r_max)
    new_state.r = new_state.r.at[..., index, :].add(jax.random.normal(subkey, state.r.shape[:-2] + (3,)) * s[..., None])

    dist_closest = jnp.min(get_el_ion_distance_matrix(new_state.r, state.R)[1], axis=-1).at[..., index].get()
    s_new = state.stepsize * jnp.clip(dist_closest, config.r_min, config.r_max)

    dist_sqr = jnp.sum((new_state.r.at[..., index, :].get() - state.r.at[..., index, :].get()) ** 2, axis=-1)
    log_q_ratio = 3 * (jnp.log(s) - jnp.log(s_new))
    log_q_ratio += (
        0.5 * dist_sqr * (1 / s**2 - 1 / s_new**2)
    )  # log_q_ratio = jnp.sum(log_q_ratio, axis=-1) # sum over electrons
    return new_state, log_q_ratio


@functools.partial(jax.vmap, in_axes=(MCMC_BATCH_AXES,), out_axes=(MCMC_BATCH_AXES, 0))
def _propose_step_local_approximated_langevin(state: MCMCState, config: MCMCLangevinProposalConfig):
    """
    Proposes a new electron configuration for metropolis hastings, using 2 techniques: Local stepsize and an approximated langevin-dynamics.

    The stepsise grows linearly with the distance from the nuclei, to encourage large steps further away from the nucleus where the electron density varies more slowly.
    The proposal direction is biased towards the nuclei (=approximated langevin dynamics).

    Returns:
        Proposed new state, log of ratio of proposal likelihoods
    """
    new_state = copy.copy(state)
    new_state.rng_state, subkey = jax.random.split(state.rng_state)
    s, g_r = _calculate_stepsize_and_langevin_bias(
        state.r, state.R, state.Z, state.stepsize, config.langevin_scale, config.r_min, config.r_max
    )
    new_state.r += jax.random.normal(subkey, state.r.shape) * s + g_r * s**2
    s_new, g_r_new = _calculate_stepsize_and_langevin_bias(
        new_state.r, state.R, state.Z, state.stepsize, config.langevin_scale, config.r_min, config.r_max
    )

    d_fwd = jnp.sum((new_state.r - state.r - g_r * s**2) ** 2, axis=-1)
    d_rev = jnp.sum((state.r - new_state.r - g_r_new * s_new**2) ** 2, axis=-1)
    s = jnp.squeeze(s)
    s_new = jnp.squeeze(s_new)
    log_q_ratio = 3 * (jnp.log(s) - jnp.log(s_new))
    log_q_ratio += 0.5 * (d_fwd / s**2 - d_rev / s_new**2)
    log_q_ratio = jnp.sum(log_q_ratio, axis=-1)
    return new_state, log_q_ratio


@functools.partial(jax.vmap, in_axes=(0, 0, None, None))
def spin_swap(r, rng_state, n_up, p_swap):
    n_el = r.shape[-2]
    rng_state, key_select, key_up, key_dn = jax.random.split(rng_state, 4)
    do_spin_swap = jax.random.uniform(key_select, ()) < p_swap
    ind_up = jax.random.randint(key_up, (), 0, n_up)
    ind_dn = jax.random.randint(key_dn, (), n_up, n_el)
    ind_dn = jnp.where(do_spin_swap, ind_dn, ind_up)
    r_up = r[..., ind_up, :]
    r = r.at[..., ind_up, :].set(r[..., ind_dn, :])
    r = r.at[..., ind_dn, :].set(r_up)
    return r, rng_state


@functools.partial(jax.vmap, in_axes=(0, 0, None))
def spin_flip(r, rng_state, p_flip):
    rng_state, subkey = jax.random.split(rng_state, 2)
    do_spin_flip = jax.random.uniform(subkey, ()) < p_flip
    # By reversing order of electrons, effectively all spins are flipped
    r = jnp.where(do_spin_flip, r[..., ::-1, :], r)
    return r, rng_state


class MetropolisHastingsMonteCarlo:
    """
    Class that performs monte carlo steps.

    This class holds the MCMC logic and configuration, but does not hold the actual state (e.g. electron positions, psi², etc).
    The actual state is stored in an MCMCState object.
    """

    def __init__(self, mcmc_config: MCMCConfig):
        self.config: MCMCConfig = mcmc_config
        self._build_proposal_function()
        self.run_mcmc_steps = jax.pmap(
            self._run_mcmc_steps, axis_name="devices", static_broadcasted_argnums=(0, 3, 4, 6)
        )

    def _build_proposal_function(self):
        if self.config.proposal.name == "normal":
            self.propose = _propose_normal
        elif self.config.proposal.name == "cauchy":
            self.propose = _propose_cauchy
        elif self.config.proposal.name == "langevin":
            self.propose = functools.partial(_propose_step_local_approximated_langevin, config=self.config.proposal)
        elif self.config.proposal.name == "local":
            self.propose = functools.partial(_propose_step_local_stepsize, config=self.config.proposal)
        elif self.config.proposal.name == "normal_one_el":
            self.propose = _propose_normal_one_el
        elif self.config.proposal.name == "local_one_el":
            self.propose = functools.partial(_propose_step_local_stepsize_one_el, config=self.config.proposal)
        else:
            raise NotImplementedError("Unknown MCMC proposal type")

    def make_mcmc_step(self, func, state: MCMCState):
        # Propose a new state
        state_new, log_q_ratio = self.propose(state)
        rng_state = state.rng_state
        if self.config.p_spin_swap > 0:
            # TODO: this assumes n_up == n_dn; if it is not, nothing bad should happen,
            # we would just swap electrons with the same spin. This is useless, but not harmful.
            n_up = state_new.r.shape[-2] // 2
            state_new.r, rng_state = spin_swap(state_new.r, rng_state, n_up, self.config.p_spin_swap)
        if self.config.p_spin_flip > 0:
            state_new.r, rng_state = spin_flip(state_new.r, rng_state, self.config.p_spin_flip)

        state_new.log_psi_sqr = func(state_new)
        # Decide which samples to accept and which ones to reject
        p_accept = jnp.exp(state_new.log_psi_sqr - state.log_psi_sqr + log_q_ratio)
        rng_state, subkeys = batch_rng_split(rng_state, 2)
        thr_accept = jax.vmap(lambda k: jax.random.uniform(k, ()))(subkeys)
        do_accept = p_accept > thr_accept
        do_accept = jnp.logical_or(do_accept, state_new.walker_age >= self.config.max_age)
        state_new.walker_age = jnp.where(do_accept, 0, state_new.walker_age + 1)
        state_new.log_psi_sqr = jnp.where(do_accept, state_new.log_psi_sqr, state.log_psi_sqr)
        state_new.r = jnp.where(do_accept[..., np.newaxis, np.newaxis], state_new.r, state.r)
        acceptance_rate = pmean(jnp.mean(do_accept))

        # Update running acceptance rate and stepsize
        state_new.step_nr += 1
        state_new.acc_rate = 0.9 * state.acc_rate + 0.1 * acceptance_rate  # exp running average of acc_rate
        state_new.stepsize = jax.lax.cond(
            state_new.step_nr % self.config.stepsize_update_interval == 0,
            self._adjust_stepsize,
            lambda x: x[0],
            (state.stepsize, state.acc_rate),
        )
        state_new.rng_state = rng_state
        return state_new

    def _adjust_stepsize(self, args):
        stepsize, acceptance_rate = args
        stepsize = jax.lax.cond(
            acceptance_rate < self.config.target_acceptance_rate, lambda s: s / 1.05, lambda s: s * 1.05, stepsize
        )
        stepsize = jnp.clip(stepsize, self.config.min_stepsize_scale, self.config.max_stepsize_scale)
        return stepsize

    def _run_mcmc_steps(self, func, state, params, n_up, n_dn, fixed_params, n_steps):
        def partial_func(s):
            return func(params, n_up, n_dn, *s.build_batch(fixed_params))[
                1
            ]  # only use log_psi_squared and remove phase

        # TODO: technically we don't have to always reevaluate log-psi-squared, but we do it for simplicity; when updating params it is required, but during evaluation we could skip it
        state.log_psi_sqr = partial_func(state)

        def _loop_body(i, _state):
            s = self.make_mcmc_step(partial_func, _state)
            if "periodic" in fixed_params:
                lattice = fixed_params["periodic"].lattice
                inv_lattice = fixed_params["periodic"].rec / (2 * np.pi)
                s.r = project_into_first_unit_cell(s.r, lattice, inv_lattice)
            return s

        return jax.lax.fori_loop(0, n_steps, _loop_body, state)

    def run_inter_steps(self, func, state: MCMCState, params, n_up, n_dn, fixed_params):
        return self.run_mcmc_steps(func, state, params, n_up, n_dn, fixed_params, self.config.n_inter_steps)

    def run_burn_in(self, func, state: MCMCState, params, n_up, n_dn, fixed_params):
        return self.run_mcmc_steps(func, state, params, n_up, n_dn, fixed_params, self.config.n_burn_in)
