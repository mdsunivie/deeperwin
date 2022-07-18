"""
Logic for wavefunction optimization.
"""
import logging
import jax
import jax.numpy as jnp
import kfac_jax
from deeperwin.configuration import OptimizationConfig, EvaluationConfig, ClippingConfig, LoggingConfig, PreTrainingConfig, ModelConfig, PhysicalConfig
from deeperwin.evaluation import evaluate_wavefunction
from deeperwin.hamiltonian import get_local_energy
from deeperwin.checkpoints import is_checkpoint_required, delete_obsolete_checkpoints
from deeperwin.loggers import DataLogger, WavefunctionLogger
from deeperwin.mcmc import MetropolisHastingsMonteCarlo, MCMCState
from deeperwin.utils import get_el_ion_distance_matrix
from deeperwin.model import evaluate_sum_of_determinants, get_baseline_slater_matrices
from deeperwin.orbitals import get_baseline_solution
from deeperwin.optimizers import build_optimizer
from deeperwin.utils import pmap, replicate_across_devices, merge_from_devices, pmean
import functools

LOGGER = logging.getLogger("dpe")

def _update_clipping_state(E, clipping_state, clipping_config: ClippingConfig):
    del clipping_state
    center = dict(mean=jnp.nanmean,
                  median=jnp.nanmedian,
                  )[clipping_config.center](E)
    center = pmean(center)
    width = dict(std=jnp.nanstd,
                 mae=lambda x: jnp.nanmean(jnp.abs(x-center)),
                 )[clipping_config.width_metric](E) * clipping_config.clip_by
    width = pmean(width)
    return center, width

def _clip_energies(E, clipping_state, clipping_config: ClippingConfig):
    if clipping_config.from_previous_step:
        center, width = clipping_state
    else:
        center, width = _update_clipping_state(E, clipping_state, clipping_config)

    if clipping_config.name == "hard":
        clipped_energies = jnp.clip(E, center - width, center + width)
    elif clipping_config.name == "tanh":
        clipped_energies = center + jnp.tanh((E - center) / width) * width
    else:
        raise ValueError(f"Unsupported config-value for optimization.clipping.name: {clipping_config.name}")
    new_clipping_state = _update_clipping_state(clipped_energies, clipping_state, clipping_config)
    return clipped_energies, new_clipping_state

def build_value_and_grad_func(log_psi_sqr_func, clipping_config: ClippingConfig):
    """
    Returns a callable that computes the gradient of the mean local energy for a given set of MCMC walkers with respect to the model defined by `log_psi_func`.

    Args:
        log_psi_sqr_func (callable): A function representing the wavefunction model
        clipping_config (ClippingConfig): Clipping hyperparameters
        use_fwd_fwd_hessian (bool): If true, the second partial derivatives required for computing the local energy are obtained with a forward-forward scheme.

    """

    # Build custom total energy jvp. Based on https://github.com/deepmind/ferminet/blob/jax/ferminet/train.py
    @jax.custom_jvp
    def total_energy(params, state, batch):
        clipping_state = state
        E_loc_unclipped = get_local_energy(log_psi_sqr_func, params, *batch)
        E_loc_clipped, clipping_state = _clip_energies(E_loc_unclipped, clipping_state, clipping_config)
        E_mean_clipped = pmean(jnp.nanmean(E_loc_clipped))
        stats = dict(E_mean_clipped=E_mean_clipped,
                     E_loc_clipped=E_loc_clipped,
                     E_loc_unclipped=E_loc_unclipped)
        loss = E_mean_clipped
        return loss, (clipping_state, stats)

    @total_energy.defjvp
    def total_energy_jvp(primals, tangents):
        params, state, batch = primals
        batch_size = batch[0].shape[0]

        loss, (state, stats) = total_energy(*primals)
        diff = stats["E_loc_clipped"] - stats["E_mean_clipped"]

        log_psi_sqr, tangents_log_psi_sqr = jax.jvp(lambda p: log_psi_sqr_func(p, *batch), (primals[0],), (tangents[0],))
        kfac_jax.register_normal_predictive_distribution(log_psi_sqr[:, None])  # Register loss for kfac optimizer

        primals_out = loss, (state, stats)
        tangents_out = jnp.dot(tangents_log_psi_sqr, diff) / batch_size, (state, stats)
        return primals_out, tangents_out

    return jax.value_and_grad(total_energy, has_aux=True)


def optimize_wavefunction(
        log_psi_squared,
        params,
        fixed_params,
        mcmc_state: MCMCState,
        opt_config: OptimizationConfig,
        phys_config: PhysicalConfig,
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
    n_devices = jax.device_count()
    log_psi_squared_pmapped = jax.pmap(log_psi_squared)

    mcmc = MetropolisHastingsMonteCarlo(opt_config.mcmc)
    mcmc_state = MCMCState.resize_or_init(mcmc_state, opt_config.mcmc.n_walkers, phys_config, n_devices)
    clipping_state = initial_clipping_state or (jnp.array([0.0]).squeeze(), jnp.array([1e5]).squeeze())  # do not clip at first epoch, then adjust
    params, fixed_params, initial_opt_state, clipping_state = replicate_across_devices(
        (params, fixed_params, initial_opt_state, clipping_state), n_devices)
    mcmc_state.log_psi_sqr = log_psi_squared_pmapped(params, *mcmc_state.build_batch(fixed_params))
    mcmc_state = mcmc.run_burn_in(log_psi_squared, mcmc_state, params, fixed_params)

    # Initialize loss and optimizer
    rng = jax.random.split(jax.random.PRNGKey(0), n_devices)
    value_and_grad_func = build_value_and_grad_func(log_psi_squared, opt_config.clipping)
    optimizer = build_optimizer(value_and_grad_func, opt_config.optimizer, True, True)
    opt_state = optimizer.init(params, rng, mcmc_state.build_batch(fixed_params), clipping_state)
    opt_state = initial_opt_state or opt_state

    # Set-up check-points
    eval_checkpoints = set(opt_config.intermediate_eval.opt_epochs) if opt_config.intermediate_eval else set()

    wf_logger = WavefunctionLogger(logger, prefix="opt", n_step=opt_config.n_epochs_prev, smoothing=0.05)
    for n_epoch in range(opt_config.n_epochs_prev, opt_config.n_epochs_prev+opt_config.n_epochs):
        mcmc_state = mcmc.run_inter_steps(log_psi_squared, mcmc_state, params, fixed_params)
        params, opt_state, clipping_state, stats = optimizer.step(params,
                                                                  opt_state,
                                                                  rng=None,
                                                                  batch=mcmc_state.build_batch(fixed_params),
                                                                  func_state=clipping_state)
        mcmc_state.log_psi_sqr = log_psi_squared_pmapped(params, *mcmc_state.build_batch(fixed_params))
        wf_logger.log_step(E_loc_unclipped=stats["aux"]["E_loc_unclipped"], E_loc_clipped=stats["aux"]["E_loc_clipped"],
                           E_ref=phys_config.E_ref, mcmc_state=mcmc_state, opt_stats=merge_from_devices(stats))

        if is_checkpoint_required(n_epoch, opt_config.checkpoints) and (logger is not None):
            LOGGER.debug(f"Saving checkpoint n_epoch={n_epoch}")
            params_merged, fixed_params_merged, opt_state_merged, clipping_state_merged = merge_from_devices(
                (params, fixed_params, opt_state, clipping_state))
            mcmc_state_merged = mcmc_state.merge_devices()
            logger.log_checkpoint(n_epoch, params_merged, fixed_params_merged, mcmc_state_merged, opt_state_merged, clipping_state_merged)
            delete_obsolete_checkpoints(n_epoch, opt_config.checkpoints)

        if (n_epoch+1) in eval_checkpoints:
            LOGGER.debug(f"opt epoch {n_epoch:5d}: Running intermediate evaluation...")
            eval_config = EvaluationConfig(n_epochs=opt_config.intermediate_eval.n_epochs)
            params_merged, fixed_params_merged = merge_from_devices((params_merged, fixed_params_merged))
            mcmc_state_merged = mcmc_state.merge_devices()
            evaluate_wavefunction(
                log_psi_squared, params_merged, fixed_params_merged, mcmc_state_merged, eval_config, phys_config, logger, n_epoch,
            )

    LOGGER.debug("Finished wavefunction optimization...")
    params, opt_state, clipping_state = merge_from_devices((params, opt_state, clipping_state))
    return mcmc_state, params, opt_state, clipping_state


def pretrain_orbitals(orbital_func, mcmc_state: MCMCState, params, fixed_params,
                      config: PreTrainingConfig,
                      phys_config: PhysicalConfig,
                      model_config: ModelConfig,
                      loggers: DataLogger = None, opt_state=None):
    if model_config.orbitals.baseline_orbitals and (model_config.orbitals.baseline_orbitals.baseline == config.baseline):
        baseline_orbitals = fixed_params['orbitals']
        LOGGER.debug("Identical CASSCF-config for pre-training and baseline model: Reusing baseline calculation")
    else:
        LOGGER.warning("Using different baseline pySCF settings for pre-training and the baked-in baseline model. Calculating new orbitals...")
        n_determinants = 1 if config.use_only_leading_determinant else model_config.orbitals.n_determinants
        baseline_orbitals, (E_hf, E_casscf) = get_baseline_solution(phys_config, config.baseline, n_determinants)
        LOGGER.debug(f"Finished baseline calculation for pretraining: E_HF = {E_hf:.6f}, E_casscf={E_casscf:.6f}")

    def loss_func(params, batch):
        r, R, Z, fixed_params = batch
        # Calculate HF / CASSCF reference orbitals
        diff_el_ion, dist_el_ion = get_el_ion_distance_matrix(r, R)
        mo_up_ref, mo_dn_ref = get_baseline_slater_matrices(diff_el_ion, dist_el_ion, baseline_orbitals,
                                                            model_config.orbitals.use_full_det)

        if config.use_only_leading_determinant:
            mo_up_ref = mo_up_ref[...,:1,:,:]
            mo_dn_ref = mo_dn_ref[...,:1,:,:]

        # Calculate neural net orbitals
        mo_up, mo_dn = orbital_func(params, *batch)
        residual_up = mo_up - mo_up_ref
        residual_dn = mo_dn - mo_dn_ref
        return jnp.mean(residual_up**2) + jnp.mean(residual_dn**2)

    def log_psi_squared_func(params, r, R, Z, fixed_params):
        mo_up, mo_dn = orbital_func(params, r, R, Z, fixed_params)
        return evaluate_sum_of_determinants(mo_up, mo_dn, model_config.orbitals.use_full_det)
    log_psi_sqr_pmapped = pmap(log_psi_squared_func)


    # Init MCMC
    logging.debug(f"Starting pretraining...")
    n_devices = jax.device_count()
    mcmc = MetropolisHastingsMonteCarlo(config.mcmc)
    mcmc_state = MCMCState.resize_or_init(mcmc_state, config.mcmc.n_walkers, phys_config, n_devices)

    # Split(Replicate data across devices
    params, fixed_params, opt_state = replicate_across_devices((params, fixed_params, opt_state), n_devices)
    mcmc_state.log_psi_sqr = log_psi_sqr_pmapped(params, *mcmc_state.build_batch(fixed_params))

    # MCMC burn-in
    mcmc_state = mcmc.run_burn_in(log_psi_squared_func, mcmc_state, params, fixed_params)

    # Init optimizer
    rng = jax.random.PRNGKey(0)
    optimizer = build_optimizer(jax.value_and_grad(loss_func), config.optimizer, False, False)
    opt_state = opt_state or optimizer.init(params, rng, mcmc_state.build_batch(fixed_params))

    # Pre-training optimization loop
    for n in range(config.n_epochs):
        mcmc_state = mcmc.run_inter_steps(log_psi_squared_func, mcmc_state, params, fixed_params)
        params, opt_state, stats = optimizer.step(params, opt_state, None, batch=mcmc_state.build_batch(fixed_params))
        mcmc_state.log_psi_sqr = log_psi_sqr_pmapped(params, *mcmc_state.build_batch(fixed_params))

        if loggers is not None:
            loggers.log_metrics(dict(loss=float(stats['loss'].mean()),
                                     mcmc_stepsize=float(mcmc_state.stepsize.mean()),
                                     mcmc_step_nr=int(mcmc_state.step_nr.mean())
                                     ),
                                epoch=n,
                                metric_type="pre")

    params, opt_state = merge_from_devices((params, opt_state))
    return params, opt_state, mcmc_state

