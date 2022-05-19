"""
Logic for Markov chain Monte Carlo (MCMC) steps.
"""

import copy
import functools
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from deeperwin.configuration import MCMCConfig, MCMCLangevinProposalConfig, PhysicalConfig, LocalStepsizeProposalConfig
from deeperwin.utils import get_el_ion_distance_matrix


@jax.tree_util.register_pytree_node_class
@dataclass
class MCMCState:
    """
    Dataclasss that holds an electronic configuration and metadata required for MCMC.
    """
    r: jnp.array  # [batch-size x n_el x 3]
    R: jnp.array  # [n_ions x 3 ]
    Z: jnp.array  # [n_ions]
    log_psi_sqr: jnp.array = None
    walker_age: jnp.array = None  # [batch-size]; dtype=int
    rng_state: Tuple[jnp.array] = jax.random.PRNGKey(0)
    stepsize: jnp.array = jnp.array(1e-2)
    step_nr: jnp.array = jnp.array(0, dtype=int)
    acc_rate: jnp.array = jnp.array(0.0)

    def tree_flatten(self):
        children = (self.r, self.R, self.Z, self.log_psi_sqr, self.walker_age, self.rng_state, self.stepsize, self.step_nr, self.acc_rate)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @property
    def model_args(self):
        return self.r, self.R, self.Z

    @classmethod
    def initialize_around_nuclei(cls, n_walkers, physical_config: PhysicalConfig):
        r0 = np.random.normal(size=[n_walkers, physical_config.n_electrons, 3])
        for i_el, i_nuc in enumerate(physical_config.el_ion_mapping):
            r0[:, i_el, :] += np.array(physical_config.R[i_nuc])
        r0 = jnp.array(r0)
        return cls(
            r=r0, R=jnp.array(physical_config.R), Z=jnp.array(physical_config.Z),
            walker_age=jnp.zeros(n_walkers, dtype=int)
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
    new_state = copy.deepcopy(state)
    new_state.r = _resize_array(state.r, n_walkers_new)
    new_state.log_psi_sqr = _resize_array(state.log_psi_sqr, n_walkers_new)
    new_state.walker_age = _resize_array(state.walker_age, n_walkers_new)
    return new_state


def _propose_normal(state: MCMCState):
    new_state = copy.copy(state)
    new_state.rng_state, subkey = jax.random.split(state.rng_state)
    new_state.r += jax.random.normal(subkey, state.r.shape) * state.stepsize
    return new_state, jnp.zeros(state.r.shape[0])


def _propose_cauchy(state: MCMCState):
    new_state = copy.copy(state)
    new_state.rng_state, subkey = jax.random.split(state.rng_state)
    new_state.r += jax.random.cauchy(subkey, state.r.shape) * state.stepsize
    return new_state, jnp.zeros(state.r.shape[0])


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
    log_q_ratio += 0.5 * dist_sqr * (1 / s ** 2 - 1 / s_new ** 2)
    log_q_ratio = jnp.sum(log_q_ratio, axis=-1) # sum over electrons
    return new_state, log_q_ratio

def _propose_step_local_stepsize_one_el(state: MCMCState, config: LocalStepsizeProposalConfig):
    new_state = copy.copy(state)
    new_state.rng_state, subkey = jax.random.split(state.rng_state)
    n_el = state.r.shape[-2]
    index = state.step_nr % n_el

    #new_state.r += jax.random.normal(subkey, state.r.shape) * s[..., None]
    dist_closest = jnp.min(get_el_ion_distance_matrix(state.r, state.R)[1], axis=-1).at[..., index].get()
    s = state.stepsize * jnp.clip(dist_closest, config.r_min, config.r_max)
    new_state.r = new_state.r.at[..., index, :].add(jax.random.normal(subkey, state.r.shape[:-2] + (3,)) * s[..., None])

    dist_closest = jnp.min(get_el_ion_distance_matrix(new_state.r, state.R)[1], axis=-1).at[..., index].get()
    s_new = state.stepsize * jnp.clip(dist_closest, config.r_min, config.r_max)

    dist_sqr = jnp.sum((new_state.r.at[..., index, :].get() - state.r.at[..., index, :].get()) ** 2, axis=-1)
    log_q_ratio = 3 * (jnp.log(s) - jnp.log(s_new))
    log_q_ratio += 0.5 * dist_sqr * (1 / s ** 2 - 1 / s_new ** 2)    #log_q_ratio = jnp.sum(log_q_ratio, axis=-1) # sum over electrons
    return new_state, log_q_ratio

def _propose_normal_one_el(state: MCMCState):
    n_el = state.r.shape[-2]
    index = state.step_nr % n_el

    new_state = copy.copy(state)
    new_state.rng_state, subkey = jax.random.split(state.rng_state)
    new_state.r = new_state.r.at[..., index, :].add(jax.random.normal(subkey, state.r[..., index, :].shape) * state.stepsize)
    return new_state, jnp.zeros(state.r.shape[0])

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
    new_state.r += jax.random.normal(subkey, state.r.shape) * s + g_r * s ** 2
    s_new, g_r_new = _calculate_stepsize_and_langevin_bias(
        new_state.r, state.R, state.Z, state.stepsize, config.langevin_scale, config.r_min, config.r_max
    )

    d_fwd = jnp.sum((new_state.r - state.r - g_r * s ** 2) ** 2, axis=-1)
    d_rev = jnp.sum((state.r - new_state.r - g_r_new * s_new ** 2) ** 2, axis=-1)
    s = jnp.squeeze(s)
    s_new = jnp.squeeze(s_new)
    log_q_ratio = 3 * (jnp.log(s) - jnp.log(s_new))
    log_q_ratio += 0.5 * (d_fwd / s ** 2 - d_rev / s_new ** 2)
    log_q_ratio = jnp.sum(log_q_ratio, axis=-1)
    return new_state, log_q_ratio


class MetropolisHastingsMonteCarlo:
    """
    Class that performs monte carlo steps.

    This class holds the MCMC logic and configuration, but does not hold the actual state (e.g. electron positions, psi², etc).
    The actual state is stored in an MCMCState object.
    """

    def __init__(self, mcmc_config: MCMCConfig):
        self.config: MCMCConfig = mcmc_config
        self._build_proposal_function()

    def _build_proposal_function(self):
        if self.config.proposal.name == "normal":
            self.propose = _propose_normal
        elif self.config.proposal.name == "cauchy":
            self.propose = _propose_cauchy
        elif self.config.proposal.name == "langevin":
            self.propose = functools.partial(_propose_step_local_approximated_langevin, config=self.config.proposal)
        elif self.config.proposal.name == 'local':
            self.propose = functools.partial(_propose_step_local_stepsize, config=self.config.proposal)
        elif self.config.proposal.name == 'normal_one_el':
            self.propose = _propose_normal_one_el
        elif self.config.proposal.name == "local_one_el":
            self.propose = functools.partial(_propose_step_local_stepsize_one_el, config=self.config.proposal)
        else:
            raise NotImplementedError("Unknown MCMC proposal type")

    def make_mcmc_step(self, func, func_params, state: MCMCState, method, stepsize_update_interval):
        # Propose a new state
        state_new, log_q_ratio = self.propose(state)
        state_new.log_psi_sqr = func(*state_new.model_args, *func_params)

        # Decide which samples to accept and which ones to reject
        p_accept = jnp.exp(state_new.log_psi_sqr - state.log_psi_sqr + log_q_ratio)
        state_new.rng_state, subkey = jax.random.split(state.rng_state)
        thr_accept = jax.random.uniform(subkey, p_accept.shape)
        do_accept = p_accept > thr_accept

        if method == 'opt':
            is_too_old = state_new.walker_age >= self.config.max_age_opt
        else:
            is_too_old = state_new.walker_age >= self.config.max_age_eval
        do_accept = jnp.logical_or(do_accept, is_too_old)
        state_new.walker_age = jnp.where(do_accept, 0, state_new.walker_age + 1)

        state_new.log_psi_sqr = jnp.where(do_accept, state_new.log_psi_sqr, state.log_psi_sqr)
        state_new.r = jnp.where(do_accept[..., np.newaxis, np.newaxis], state_new.r, state.r)
        acceptance_rate = jnp.mean(do_accept)

        state_new.step_nr += 1

        if self.config.debug_mcmc_stepsize:
            state_new.stepsize = jax.lax.cond(state_new.step_nr % stepsize_update_interval == 0,
                                              lambda x: x*1.25,
                                              lambda x: x,
                                              state.stepsize)
        else:
            state_new.acc_rate = jax.lax.cond(state_new.step_nr % stepsize_update_interval == 0,
                                              lambda x: jnp.array(0.0),
                                              lambda x: x[0] + x[1],
                                              (state.acc_rate,
                                               acceptance_rate))
            state_new.stepsize = jax.lax.cond(state_new.step_nr % stepsize_update_interval == 0,
                                              self._adjust_stepsize,
                                              lambda x: x[0],
                                              (state.stepsize,
                                              state.acc_rate/(stepsize_update_interval-1)))
        return state_new


    def _adjust_stepsize(self, args):
        stepsize, acceptance_rate = args
        stepsize = jax.lax.cond(
            acceptance_rate < self.config.target_acceptance_rate, lambda s: s / 1.05, lambda s: s * 1.05, stepsize
        )
        stepsize = jnp.clip(stepsize, self.config.min_stepsize_scale, self.config.max_stepsize_scale)
        return stepsize

    @functools.partial(jax.jit, static_argnums=(0, 1, 4, 5, 6))
    def _run_mcmc_steps(self, func, func_params, state, n_steps, method, stepsize_update_interval):
        def _loop_body(i, _state):
            return self.make_mcmc_step(func, func_params, _state, method, stepsize_update_interval)
        # if self.config.proposal.name == 'normal_one_el':
        #     return jax.lax.fori_loop(0, n_steps*state.r.shape[-2], _loop_body, state)
        # else:
        return jax.lax.fori_loop(0, n_steps, _loop_body, state)


    def run_inter_steps(self, func, func_params, state: MCMCState, method='opt'):
        return self._run_mcmc_steps(func, func_params, state, self.config.n_inter_steps, method, self.config.stepsize_update_interval)

    def run_burn_in_opt(self, func, func_params, state: MCMCState):
        return self._run_mcmc_steps(func, func_params, state, self.config.n_burn_in_opt, 'opt', self.config.stepsize_update_interval)

    def run_burn_in_eval(self, func, func_params, state: MCMCState):
        return self._run_mcmc_steps(func, func_params, state, self.config.n_burn_in_eval, 'eval', self.config.stepsize_update_interval)


