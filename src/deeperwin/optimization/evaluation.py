"""
Logic for wavefunction evaluation.
"""
import logging
from typing import Tuple, Optional, Dict
import jax
import functools
from deeperwin.configuration import EvaluationConfig, PhysicalConfig
from deeperwin.hamiltonian import get_local_energy, calculate_forces
from deeperwin.loggers import DataLogger, LoggerCollection, WavefunctionLogger
from deeperwin.mcmc import MCMCState, MetropolisHastingsMonteCarlo
from deeperwin.utils.utils import pmap, pmean, replicate_across_devices, merge_from_devices
import jax.numpy as jnp

LOGGER = logging.getLogger("dpe")

def evaluate_wavefunction(
        log_psi_sqr,
        cache_func,
        params,
        fixed_params,
        mcmc_state: MCMCState,
        config: EvaluationConfig,
        phys_config: PhysicalConfig,
        rng_seed: int,
        loggers: LoggerCollection = None,
        opt_epoch_nr: int = None,
        extra_summary_metrics: Optional[Dict] = None
):
    # Burn-in MCMC
    rng = jax.random.PRNGKey(rng_seed)
    LOGGER.debug(f"Starting burn-in for evaluation: {config.mcmc.n_burn_in} steps")
    # static_args are n_up, n_down
    log_psi_squared_pmapped = jax.pmap(log_psi_sqr, axis_name="devices", static_broadcasted_argnums=(1, 2))
    cache_func_pmapped = jax.pmap(cache_func, axis_name="devices", static_broadcasted_argnums=(1, 2))

    mcmc = MetropolisHastingsMonteCarlo(config.mcmc)
    mcmc_state = MCMCState.resize_or_init(mcmc_state, config.mcmc.n_walkers, phys_config, config.mcmc.initialization, rng)
    mcmc_state = mcmc_state.split_across_devices()

    spin_state = (phys_config.n_up, phys_config.n_dn)
    params, fixed_params = replicate_across_devices((params, fixed_params))
    fixed_params['cache'] = cache_func_pmapped(params, *spin_state, *mcmc_state.build_batch(fixed_params))
    mcmc_state.log_psi_sqr = log_psi_squared_pmapped(params, *spin_state, *mcmc_state.build_batch(fixed_params))
    mcmc_state = mcmc.run_burn_in(log_psi_sqr, mcmc_state, params, *spin_state, fixed_params)

    @functools.partial(jax.pmap, axis_name="devices", static_broadcasted_argnums=(2,))
    def get_observables(params, fixed_params, spin_state: Tuple[int], mcmc_state: MCMCState):
        metrics = dict()
        if config.calculate_energies:
            energies = get_local_energy(log_psi_sqr, params, spin_state, *mcmc_state.build_batch(fixed_params))
            metrics['E_mean'] = pmean(jnp.nanmean(energies))
            metrics['E_var'] = pmean(jnp.nanmean((energies - metrics['E_mean'])**2))
        if config.forces:
            forces = calculate_forces(log_psi_sqr, params, *mcmc_state.build_batch(fixed_params),
                                      mcmc_state.log_psi_sqr, config.forces)
            metrics['forces_mean'] = pmean(jnp.nanmean(forces, axis=0))
            metrics['forces_var'] = pmean(jnp.nanmean((forces - metrics['forces_mean'])**2, axis=0))
        return metrics

    # Evaluation loop
    wf_logger = WavefunctionLogger(loggers, prefix="eval", smoothing=1.0)
    for n_epoch in range(config.n_epochs):
        mcmc_state = mcmc.run_inter_steps(log_psi_sqr, mcmc_state, params, *spin_state, fixed_params)
        metrics = get_observables(params, fixed_params, spin_state, mcmc_state)
        metrics = {k: v[0] for k, v in metrics.items()}

        mcmc_state_merged = mcmc_state.merge_devices()
        wf_logger.log_step(metrics, E_ref=phys_config.E_ref, mcmc_state=mcmc_state_merged,
                           extra_metrics={'opt_epoch': opt_epoch_nr})
    wf_logger.log_summary(phys_config.E_ref, opt_epoch_nr, extra_summary_metrics)
    return wf_logger.history, mcmc_state_merged
