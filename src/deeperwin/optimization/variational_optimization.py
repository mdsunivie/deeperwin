import logging
from typing import List, Dict, Tuple, Optional, Any, Callable
import jax
import jax.numpy as jnp
import numpy as np

from deeperwin.configuration import Configuration, OptimizationConfig, EvaluationConfig, PhysicalConfig
from deeperwin.optimization.evaluation import evaluate_wavefunction
from deeperwin.geometries import GeometryDataStore, distort_geometry
from deeperwin.hamiltonian import build_local_energy
from deeperwin.checkpoints import is_checkpoint_required, delete_obsolete_checkpoints
from deeperwin.loggers import DataLogger, WavefunctionLogger
from deeperwin.optimization.loss_function import build_value_and_grad_func, init_clipping_state
from deeperwin.mcmc import MetropolisHastingsMonteCarlo, MCMCState
from deeperwin.optimization.opt_utils import _run_mcmc_with_cache
from deeperwin.optimizers import build_optimizer
from deeperwin.utils.utils import replicate_across_devices, get_from_devices, get_next_geometry_index, replicate_across_processes
from deeperwin.model import init_model_fixed_params


LOGGER = logging.getLogger("dpe")

# TODO: Is there substantial common functionality between shared/non-shared that can be extracted into sub-functions?
def optimize_wavefunction(
        log_psi_squared,
        cache_func,
        params,
        fixed_params,
        mcmc_state: MCMCState,
        opt_config: OptimizationConfig,
        eval_config: EvaluationConfig,
        phys_config: PhysicalConfig,
        is_complex: bool,
        kfac_register_complex: bool,
        rng_seed: int,
        logger: DataLogger = None,
        initial_opt_state=None,
        initial_clipping_state=None,
):
    """
    Minimizes the energy of the wavefunction defined by the callable `log_psi_squared` by adjusting the trainable parameters.

    Args:
        log_psi_func (callable): A function representing the wavefunction model
        params (dict): Trainable paramters of the model defined by `log_psi_func`
        fixed_params (dict): Fixed paramters of the model defined by `log_psi_func`
        mcmc (MetropolisHastingsMonteCarlo): Object that implements the MCMC algorithm
        mcmc_state (MCMCState): Initial state of the MCMC walkers
        opt_config (OptimizationConfig): Optimization hyperparameters
        checkpoints (dict): Dictionary with items of the form {n_epochs: path}. A checkpoint is saved for each item after optimization epoch n_epochs in the folder path.
        logger (DataLogger): A logger that is used to log information about the optimization process
        log_config (LoggingConfig): Logging configuration for checkpoints

    Returns:
        A tuple (mcmc_state, trainable_paramters, opt_state), where mcmc_state is the final MCMC state and trainable_parameters contains the optimized parameters.

    """
    LOGGER.debug("Starting optimize_wavefunction")

    # Run burn-in of monte carlo chain
    LOGGER.debug(f"Starting burn-in for optimization: {opt_config.mcmc.n_burn_in} steps")

    rng_mcmc, rng_opt = jax.random.split(jax.random.PRNGKey(rng_seed), 2)
    mcmc = MetropolisHastingsMonteCarlo(opt_config.mcmc)
    mcmc_state = MCMCState.resize_or_init(mcmc_state, opt_config.mcmc, phys_config, rng_mcmc)
    clipping_state = initial_clipping_state or init_clipping_state(is_complex)
    spin_state = (phys_config.n_up, phys_config.n_dn)

    params, fixed_params, initial_opt_state, clipping_state, rng_opt = replicate_across_devices(
        (params, fixed_params, initial_opt_state, clipping_state, rng_opt))

    mcmc_state, fixed_params = _run_mcmc_with_cache(log_psi_squared, cache_func, mcmc, params, spin_state, mcmc_state,
                                                    fixed_params, split_mcmc=True, merge_mcmc=False, mode="burnin")
    
    # Initialize loss and optimizer
    get_local_energy = build_local_energy(log_psi_squared,
                                            is_complex,
                                            phys_config.periodic is not None,
                                            phys_config.periodic is not None and phys_config.periodic.include_heg_background,
                                            opt_config.max_batch_size
                                            )
    value_and_grad_func = build_value_and_grad_func(log_psi_squared, get_local_energy, opt_config.clipping, is_complex, kfac_register_complex)
    optimizer = build_optimizer(value_and_grad_func=value_and_grad_func, 
                                opt_config=opt_config.optimizer, 
                                value_func_has_aux=True, 
                                value_func_has_state=True,
                                log_psi_squared_func=log_psi_squared)
    opt_state = initial_opt_state or optimizer.init(params=params, 
                                                    rng=rng_opt, 
                                                    batch=mcmc_state.build_batch(fixed_params),
                                                    static_args=spin_state, 
                                                    func_state=clipping_state)

    # Set-up check-points
    wf_logger = WavefunctionLogger(logger, prefix="opt", n_step=opt_config.n_epochs_prev, smoothing=0.05)
    for n_epoch in range(opt_config.n_epochs_prev, opt_config.n_epochs_prev+opt_config.n_epochs+1):
        if is_checkpoint_required(n_epoch, opt_config.checkpoints) and (logger is not None):
            LOGGER.debug(f"Saving checkpoint n_epoch={n_epoch}")
            params_merged, fixed_params_merged, opt_state_merged, clipping_state_merged = get_from_devices(
                (params, fixed_params, opt_state, clipping_state))
            mcmc_state_merged = mcmc_state.merge_devices()
            if jax.process_index() == 0:
                logger.log_checkpoint(n_epoch, params_merged, fixed_params_merged, mcmc_state_merged, opt_state_merged, clipping_state_merged)
                delete_obsolete_checkpoints(n_epoch, opt_config.checkpoints)

        if n_epoch in eval_config.opt_epochs:
            LOGGER.debug(f"opt epoch {n_epoch:5d}: Running intermediate evaluation...")
            params_merged, fixed_params_merged = get_from_devices((params, fixed_params))
            mcmc_state_merged = mcmc_state.merge_devices()
            evaluate_wavefunction(
                log_psi_squared, cache_func, get_local_energy, params_merged, fixed_params_merged, mcmc_state_merged, eval_config, phys_config, is_complex,
                rng_seed, logger, n_epoch,
            )
        if n_epoch == (opt_config.n_epochs_prev + opt_config.n_epochs):
            break
        mcmc_state, fixed_params = _run_mcmc_with_cache(log_psi_squared, cache_func, mcmc, params, spin_state,
                                                        mcmc_state,
                                                        fixed_params, split_mcmc=False, merge_mcmc=False,
                                                        mode="intersteps")
        params, opt_state, clipping_state, stats = optimizer.step(params=params,
                                                                  state=opt_state,
                                                                  static_args=spin_state,
                                                                  rng=rng_opt,
                                                                  batch=mcmc_state.build_batch(fixed_params),
                                                                  func_state=clipping_state)
        mcmc_state_merged = mcmc_state.merge_devices()

        metrics = {k: float(v[0].real) for k,v in stats['aux'].items() if not k.startswith('E_loc')}
        wf_logger.log_step(metrics,
                           E_ref=phys_config.E_ref,
                           mcmc_state=mcmc_state_merged,
                           opt_stats=get_from_devices(stats))

        if opt_config.stop_on_nan:
            E = metrics.get('E_mean', 0.0)
            if not np.isfinite(E):
                LOGGER.warning(f"opt epoch {n_epoch:5d}: Hit non-finite optimization energy opt_E_mean={E}. Dumping checkpoint.")
                params_merged, fixed_params_merged, opt_state_merged, clipping_state_merged = get_from_devices(
                    (params, fixed_params, opt_state, clipping_state))
                logger.log_checkpoint(n_epoch, params_merged, fixed_params_merged, mcmc_state_merged, opt_state_merged,
                                      clipping_state_merged)
                raise ValueError("Aborting due to nan-energy")

    LOGGER.debug("Finished wavefunction optimization...")
    params, opt_state, clipping_state = get_from_devices((params, opt_state, clipping_state))
    return mcmc_state, params, opt_state, clipping_state


def optimize_shared_wavefunction(
    log_psi_squared: Callable,
    cache_func: Callable,
    geometries_data_stores: List[GeometryDataStore],
    config: Configuration,
    params: Dict[str, Dict[str, jax.Array]],
    rng_seed: int,
    initial_opt_state: Optional[Any] = None,
    initial_clipping_state: Optional[Any] = None,
    phisnet_model = None,
    N_ions_max = None,
) -> Tuple[Dict, Any, List[GeometryDataStore], Dict]:
    """
    Minimizes the energy of the wavefunction defined by the callable `log_psi_squared` by adjusting the trainable parameters for
    multiple geometries at once sharing all trainable parameters.

    Args:
        log_psi_func (callable): A function representing the wavefunction model
        geometries_data_stores (List[GeometryDataStores]): A list containing all geometries
        params (dict): Trainable paramters of the model defined by `log_psi_func`
        fixed_params (dict): Fixed paramters of the model defined by `log_psi_func`
        mcmc (MetropolisHastingsMonteCarlo): Object that implements the MCMC algorithm
        mcmc_state (MCMCState): Initial state of the MCMC walkers
        opt_config (OptimizationConfig): Optimization hyperparameters
        checkpoints (dict): Dictionary with items of the form {n_epochs: path}. A checkpoint is saved for each item after optimization epoch n_epochs in the folder path.
        logger (DataLogger): A logger that is used to log information about the optimization process
        log_config (LoggingConfig): Logging configuration for checkpoints

    Returns:
        A tuple (mcmc_state, trainable_paramters, opt_state), where mcmc_state is the final MCMC state and trainable_parameters contains the optimized parameters.

    """
    LOGGER.debug("Starting optimize_wavefunction")
    opt_config = config.optimization


    # Run burn-in of monte carlo chain
    LOGGER.debug(f"Starting burn-in for optimization: {opt_config.mcmc.n_burn_in} steps")

    # init MCMC
    rng_opt = jax.random.PRNGKey(rng_seed)
    mcmc = MetropolisHastingsMonteCarlo(opt_config.mcmc)
    params, initial_opt_state, rng_opt = replicate_across_devices((params, initial_opt_state, rng_opt))

    # init ema params as a copy of params
    ema_params = jax.tree_map(lambda x: x, params)

    # create MCMC state & run burn in for each goemetry
    for idx, g in enumerate(geometries_data_stores):
        logging.debug(f"Running burn-in before variational optimization for geom {idx}")
        g.spin_state = (g.physical_config.n_up, g.physical_config.n_dn)
        g.mcmc_state = MCMCState.resize_or_init(g.mcmc_state, 
                                                opt_config.mcmc,
                                                g.physical_config,
                                                jax.random.PRNGKey(rng_seed + idx))
        if g.clipping_state is None:
            g.clipping_state = initial_clipping_state or init_clipping_state(config.model.complex_wf)
        g.fixed_params, g.clipping_state = replicate_across_devices((g.fixed_params, g.clipping_state))
        g.mcmc_state, g.fixed_params = _run_mcmc_with_cache(log_psi_squared,
                                                            cache_func,
                                                            mcmc,
                                                            params,
                                                            g.spin_state,
                                                            g.mcmc_state,
                                                            g.fixed_params,
                                                            mode="burnin")

    # Initialize loss and optimizer
    is_periodic = geometries_data_stores[0].physical_config.periodic is not None
    include_heg_background = is_periodic and geometries_data_stores[0].physical_config.periodic.include_heg_background
    get_local_energy = build_local_energy(log_psi_squared,
                                          config.model.complex_wf,
                                          is_periodic,
                                          include_heg_background,
                                          opt_config.max_batch_size
                                          )
    value_and_grad_func = build_value_and_grad_func(log_psi_squared, get_local_energy, opt_config.clipping, config.model.complex_wf, config.model.kfac_register_complex)
    optimizer = build_optimizer(value_and_grad_func=value_and_grad_func, 
                                opt_config=opt_config.optimizer, 
                                value_func_has_aux=True, 
                                value_func_has_state=True,
                                log_psi_squared_func=log_psi_squared)
    opt_state = initial_opt_state or optimizer.init(params=params, 
                                                    rng=rng_opt, 
                                                    batch=geometries_data_stores[0].mcmc_state.split_across_devices().build_batch(
                                                            geometries_data_stores[0].fixed_params
                                                    ), 
                                                    static_args=geometries_data_stores[0].spin_state,
                                                    func_state=geometries_data_stores[0].clipping_state)

    # Set-up check-points
    eval_checkpoints = set(config.evaluation.opt_epochs * len(geometries_data_stores))
    geometry_permutation = np.asarray(jax.random.permutation(jax.random.PRNGKey(rng_seed), len(geometries_data_stores)))

    # Main optimization loop
    for n_epoch in range(opt_config.n_epochs_prev, opt_config.n_epochs_prev+opt_config.n_epochs+1):
        if is_checkpoint_required(n_epoch, opt_config.checkpoints):
            LOGGER.debug(f"Saving checkpoint n_epoch={n_epoch}")
            params_merged, opt_state_merged, ema_params_merged = get_from_devices((params, opt_state, ema_params))
            for idx_geom, g in enumerate(geometries_data_stores):
                if opt_config.checkpoints.log_only_zero_geom and idx_geom != 0:
                    continue

                fixed_params, clipping_state = get_from_devices((g.fixed_params, g.clipping_state))
                g.mcmc_state = g.mcmc_state.merge_devices()
                if jax.process_index() == 0:
                    g.wavefunction_logger.loggers.log_checkpoint(n_epoch,
                                                                params_merged,
                                                                fixed_params,
                                                                g.mcmc_state,
                                                                opt_state_merged if idx_geom == 0 else None,
                                                                clipping_state,
                                                                ema_params_merged)
                    delete_obsolete_checkpoints(n_epoch, opt_config.checkpoints, directory=f"{idx_geom:04d}")

        if n_epoch in eval_checkpoints:
            LOGGER.debug(f"opt epoch {n_epoch:5d}: Running intermediate evaluation...")
            params_merged = get_from_devices(params)
            for idx_geom, g in enumerate(geometries_data_stores):
                fixed_params = get_from_devices(g.fixed_params)
                g.mcmc_state = g.mcmc_state.merge_devices()
                evaluate_wavefunction(
                    log_psi_squared, cache_func, get_local_energy, params_merged, fixed_params, g.mcmc_state, config.evaluation, g.physical_config, config.model.complex_wf,
                    rng_seed, g.wavefunction_logger.loggers, g.n_opt_epochs, dict(opt_n_epoch=n_epoch, geom_id=idx_geom)
                )
        if n_epoch == (opt_config.n_epochs_prev + opt_config.n_epochs):
            break

        # Step 1. get next geometry for optimization
        next_geometry_index = get_next_geometry_index(n_epoch,
                                                      geometries_data_stores,
                                                      opt_config.shared_optimization.scheduling_method,
                                                      opt_config.shared_optimization.max_age,
                                                      opt_config.shared_optimization.n_initial_round_robin_per_geom,
                                                      geometry_permutation
                                                      )
        next_geometry_index = replicate_across_processes(next_geometry_index)
        g = geometries_data_stores[next_geometry_index]

        # Optionally distort geometry to get more diverse PES
        if opt_config.shared_optimization.distortion and g.n_opt_epochs_last_dist >= opt_config.shared_optimization.distortion.max_age:
            g.fixed_params, g.clipping_state = get_from_devices((g.fixed_params, g.clipping_state))
            if jax.process_index() == 0:
                E_old = g.fixed_params["baseline_energies"].get("E_hf", np.nan)
                g = distort_geometry(g, opt_config.shared_optimization.distortion)
                g.fixed_params = init_model_fixed_params(config.model, 
                                                         g.physical_config, 
                                                         phisnet_model,
                                                         N_ions_max, 
                                                         g.fixed_params['transferable_atomic_orbitals']["orbitals"].atomic_orbitals)
                E_new = g.fixed_params["baseline_energies"].get("E_hf", np.nan)
                LOGGER.debug(f"New geometry: geom_id={g.idx}; R_new={g.physical_config.R}; U_new={g.rotation.tolist()}, delta_E={E_new-E_old:.6f}")
            g.fixed_params, g.clipping_state = replicate_across_devices((g.fixed_params, g.clipping_state))


        # Step 2. Split MCMC and do MCMC + optimization step
        g.mcmc_state, g.fixed_params = _run_mcmc_with_cache(log_psi_squared,
                                                            cache_func,
                                                            mcmc,
                                                            params,
                                                            g.spin_state,
                                                            g.mcmc_state,
                                                            g.fixed_params,
                                                            merge_mcmc=False,
                                                            mode="intersteps")

        params, opt_state, g.clipping_state, stats = optimizer.step(params=params,
                                                                    state=opt_state,
                                                                    static_args=g.spin_state,
                                                                    rng=rng_opt,
                                                                    batch=g.mcmc_state.build_batch(g.fixed_params),
                                                                    func_state=g.clipping_state)

        # Update ema params with updated params
        ema_params = jax.tree_map(lambda old, new: opt_config.params_ema_factor * old + (1 - opt_config.params_ema_factor) * new, ema_params, params)

        # Step 3. gather states across devices again
        g.mcmc_state = g.mcmc_state.merge_devices()

        # Step 4. update & log metrics
        g.current_metrics = {k: float(v[0].real) for k,v in stats['aux'].items() if not k.startswith('E_loc')}
        g.current_metrics["damping"] = stats.get("damping")
        g.n_opt_epochs += 1
        g.n_opt_epochs_last_dist += 1
        g.last_epoch_optimized = n_epoch
        g.current_metrics['n_epoch'] = n_epoch
        g.wavefunction_logger.log_step(metrics=g.current_metrics,
                                        E_ref=g.physical_config.E_ref,
                                        mcmc_state=g.mcmc_state,
                                        opt_stats=get_from_devices(stats),
                                        epoch=g.n_opt_epochs,
                                        extra_metrics=dict(geom_id=next_geometry_index))

        if opt_config.stop_on_nan:
            E = g.current_metrics.get('E_mean', 0.0)
            if not np.isfinite(E):
                LOGGER.warning(f"opt epoch {n_epoch:5d}: Hit non-finite optimization energy opt_E_mean={E}. Dumping checkpoint.")
                params_merged, fixed_params_merged, opt_state_merged, clipping_state_merged, ema_params_merged = get_from_devices(
                    (params, g.fixed_params, opt_state, g.clipping_state, ema_params))
                g.wavefunction_logger.loggers.log_checkpoint(n_epoch, params_merged, fixed_params_merged, g.mcmc_state, opt_state_merged,
                                      clipping_state_merged, ema_params_merged)
                raise ValueError("Aborting due to nan-energy")

    # Step 5. gather all states across devices again for final evaluation
    LOGGER.debug("Finished wavefunction optimization...")
    params, opt_state = get_from_devices((params, opt_state))
    for g in geometries_data_stores:
        g.clipping_state, g.fixed_params = get_from_devices((g.clipping_state, g.fixed_params))
        g.mcmc_state = g.mcmc_state.merge_devices()

    return params, opt_state, geometries_data_stores, ema_params
