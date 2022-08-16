"""
Logic for wavefunction evaluation.
"""
import logging
import jax
from deeperwin.configuration import EvaluationConfig, PhysicalConfig
from deeperwin.hamiltonian import get_local_energy, calculate_forces
from deeperwin.loggers import DataLogger, WavefunctionLogger
from deeperwin.mcmc import MCMCState, MetropolisHastingsMonteCarlo
from deeperwin.utils import pmap, pmean, replicate_across_devices, merge_from_devices
import jax.numpy as jnp

LOGGER = logging.getLogger("dpe")

def evaluate_wavefunction(
        log_psi_sqr,
        params,
        fixed_params,
        mcmc_state: MCMCState,
        config: EvaluationConfig,
        phys_config: PhysicalConfig,
        rng_seed: int,
        logger: DataLogger = None,
        opt_epoch_nr=None,
):
    # Burn-in MCMC
    rng = jax.random.PRNGKey(rng_seed)
    LOGGER.debug(f"Starting burn-in for evaluation: {config.mcmc.n_burn_in} steps")
    mcmc = MetropolisHastingsMonteCarlo(config.mcmc)
    mcmc_state = MCMCState.resize_or_init(mcmc_state, config.mcmc.n_walkers, phys_config, rng)
    mcmc_state = mcmc_state.split_across_devices()
    params, fixed_params = replicate_across_devices((params, fixed_params))
    mcmc_state.log_psi_sqr = pmap(log_psi_sqr)(params, *mcmc_state.build_batch(fixed_params))
    mcmc_state = mcmc.run_burn_in(log_psi_sqr, mcmc_state, params, fixed_params)

    @pmap
    def get_observables(params, fixed_params, mcmc_state: MCMCState):
        metrics = dict()
        if config.calculate_energies:
            energies = get_local_energy(log_psi_sqr, params, *mcmc_state.build_batch(fixed_params))
            metrics['E_mean'] = pmean(jnp.nanmean(energies))
            metrics['E_var'] = pmean(jnp.nanmean(energies - metrics['E_mean'])**2)
        if config.forces:
            forces = calculate_forces(log_psi_sqr, params, *mcmc_state.build_batch(fixed_params),
                                      mcmc_state.log_psi_sqr, config.forces)
            metrics['forces_mean'] = pmean(jnp.nanmean(forces, axis=0))
            metrics['forces_var'] = pmean(jnp.nanmean((forces - metrics['forces_mean'])**2, axis=0))
        return metrics

    # Evaluation loop
    wf_logger = WavefunctionLogger(logger, prefix="eval", smoothing=1.0)
    for n_epoch in range(config.n_epochs):
        mcmc_state = mcmc.run_inter_steps(log_psi_sqr, mcmc_state, params, fixed_params)
        metrics = get_observables(params, fixed_params, mcmc_state)
        metrics = {k: v[0] for k, v in metrics.items()}

        mcmc_state_merged = mcmc_state.merge_devices()
        wf_logger.log_step(metrics, E_ref=phys_config.E_ref, mcmc_state=mcmc_state_merged,
                           extra_metrics={'opt_epoch': opt_epoch_nr})
    wf_logger.log_summary(E_ref=phys_config.E_ref, epoch_nr=opt_epoch_nr)
    return wf_logger.history, mcmc_state_merged
