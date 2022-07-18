"""
Logic for wavefunction evaluation.
"""
import logging
import jax
from deeperwin.configuration import EvaluationConfig, PhysicalConfig
from deeperwin.hamiltonian import get_local_energy, calculate_forces
from deeperwin.loggers import DataLogger, WavefunctionLogger
from deeperwin.mcmc import MCMCState, MetropolisHastingsMonteCarlo
from deeperwin.utils import pmap, replicate_across_devices

LOGGER = logging.getLogger("dpe")

def evaluate_wavefunction(
        log_psi_sqr,
        params,
        fixed_params,
        mcmc_state: MCMCState,
        config: EvaluationConfig,
        phys_config: PhysicalConfig,
        logger: DataLogger = None,
        opt_epoch_nr=None,
):
    # Burn-in MCMC
    LOGGER.debug(f"Starting burn-in for evaluation: {config.mcmc.n_burn_in} steps")
    n_devices = jax.device_count()
    mcmc = MetropolisHastingsMonteCarlo(config.mcmc)
    mcmc_state = MCMCState.resize_or_init(mcmc_state, config.mcmc.n_walkers, phys_config, n_devices)
    params, fixed_params = replicate_across_devices((params, fixed_params), n_devices)
    mcmc_state.log_psi_sqr = pmap(log_psi_sqr)(params, *mcmc_state.build_batch(fixed_params))
    mcmc_state = mcmc.run_burn_in(log_psi_sqr, mcmc_state, params, fixed_params)

    @pmap
    def get_observables(params, fixed_params, mcmc_state: MCMCState):
        energies, forces = None, None
        if config.calculate_energies:
            energies = get_local_energy(log_psi_sqr, params, *mcmc_state.build_batch(fixed_params))
        if config.forces:
            forces = calculate_forces(log_psi_sqr, params, *mcmc_state.build_batch(fixed_params),
                                      mcmc_state.log_psi_sqr, config.forces)
        return energies, forces

    # Evaluation loop
    wf_logger = WavefunctionLogger(logger, prefix="eval", smoothing=1.0)
    for n_epoch in range(config.n_epochs):
        mcmc_state = mcmc.run_inter_steps(log_psi_sqr, mcmc_state, params, fixed_params)
        E_loc, forces = get_observables(params, fixed_params, mcmc_state)
        wf_logger.log_step(E_loc_unclipped=E_loc, forces=forces, E_ref=phys_config.E_ref, mcmc_state=mcmc_state,
                           extra_metrics={'opt_epoch': opt_epoch_nr})
    wf_logger.log_summary(E_ref=phys_config.E_ref, epoch_nr=opt_epoch_nr)
    return wf_logger.history, mcmc_state
