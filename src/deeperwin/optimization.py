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
from deeperwin.utils import pmap, replicate_across_devices, get_from_devices, pmean, merge_from_devices, is_equal_across_devices
import functools

LOGGER = logging.getLogger("dpe")

def init_clipping_state():
    return jnp.array([0.0]).squeeze(), jnp.array([1e5]).squeeze()

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
        E_loc = get_local_energy(log_psi_sqr_func, params, *batch)
        E_mean = pmean(jnp.nanmean(E_loc))
        E_var = pmean(jnp.nanmean((E_loc - E_mean) ** 2))

        E_loc_clipped, clipping_state = _clip_energies(E_loc, clipping_state, clipping_config)
        E_mean_clipped = pmean(jnp.nanmean(E_loc_clipped))
        E_var_clipped = pmean(jnp.nanmean((E_loc_clipped - E_mean_clipped) ** 2))
        aux = dict(E_mean=E_mean,
                   E_var=E_var,
                   E_mean_clipped=E_mean_clipped,
                   E_var_clipped=E_var_clipped,
                   E_loc_clipped=E_loc_clipped)
        loss = E_mean_clipped
        return loss, (clipping_state, aux)

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
    log_psi_squared_pmapped = jax.pmap(log_psi_squared)

    rng_mcmc, rng_opt = jax.random.split(jax.random.PRNGKey(rng_seed), 2)
    mcmc = MetropolisHastingsMonteCarlo(opt_config.mcmc)
    mcmc_state = MCMCState.resize_or_init(mcmc_state, opt_config.mcmc.n_walkers, phys_config, rng_mcmc)
    clipping_state = initial_clipping_state or init_clipping_state() # do not clip at first epoch, then adjust

    params, fixed_params, initial_opt_state, clipping_state, rng_opt = replicate_across_devices(
        (params, fixed_params, initial_opt_state, clipping_state, rng_opt))
    mcmc_state = mcmc_state.split_across_devices()
    mcmc_state.log_psi_sqr = log_psi_squared_pmapped(params, *mcmc_state.build_batch(fixed_params))
    mcmc_state = mcmc.run_burn_in(log_psi_squared, mcmc_state, params, fixed_params)

    # Initialize loss and optimizer
    value_and_grad_func = build_value_and_grad_func(log_psi_squared, opt_config.clipping)
    optimizer = build_optimizer(value_and_grad_func, opt_config.optimizer, True, True)

    opt_state = optimizer.init(params, rng_opt, mcmc_state.build_batch(fixed_params), clipping_state)
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
        mcmc_state_merged = mcmc_state.merge_devices()

        metrics = {k: float(v[0]) for k,v in stats['aux'].items() if not k.startswith('E_loc')}
        wf_logger.log_step(metrics,
                           E_ref=phys_config.E_ref,
                           mcmc_state=mcmc_state_merged,
                           opt_stats=get_from_devices(stats))

        if is_checkpoint_required(n_epoch, opt_config.checkpoints) and (logger is not None):
            LOGGER.debug(f"Saving checkpoint n_epoch={n_epoch}")
            params_merged, fixed_params_merged, opt_state_merged, clipping_state_merged = get_from_devices(
                (params, fixed_params, opt_state, clipping_state))
            logger.log_checkpoint(n_epoch, params_merged, fixed_params_merged, mcmc_state_merged, opt_state_merged, clipping_state_merged)
            delete_obsolete_checkpoints(n_epoch, opt_config.checkpoints)

        if (n_epoch+1) in eval_checkpoints:
            LOGGER.debug(f"opt epoch {n_epoch:5d}: Running intermediate evaluation...")
            eval_config = EvaluationConfig(n_epochs=opt_config.intermediate_eval.n_epochs)
            params_merged, fixed_params_merged = get_from_devices((params, fixed_params))
            mcmc_state_merged = mcmc_state.merge_devices()
            evaluate_wavefunction(
                log_psi_squared, params_merged, fixed_params_merged, mcmc_state_merged, eval_config, phys_config,
                rng_seed, logger, n_epoch,
            )

    LOGGER.debug("Finished wavefunction optimization...")
    params, opt_state, clipping_state = get_from_devices((params, opt_state, clipping_state))
    return mcmc_state, params, opt_state, clipping_state


def pretrain_orbitals(orbital_func, mcmc_state: MCMCState, params, fixed_params,
                      config: PreTrainingConfig,
                      phys_config: PhysicalConfig,
                      model_config: ModelConfig,
                      rng_seed: int,
                      loggers: DataLogger = None, opt_state=None):
    if model_config.orbitals.baseline_orbitals and (model_config.orbitals.baseline_orbitals.baseline == config.baseline):
        fixed_params['pretrain_orbitals'] = fixed_params['orbitals']
        LOGGER.debug("Identical CASSCF-config for pre-training and baseline model: Reusing baseline calculation")
    else:
        LOGGER.warning("Using different baseline pySCF settings for pre-training and the baked-in baseline model. Calculating new orbitals...")
        n_determinants = 1 if config.use_only_leading_determinant else model_config.orbitals.n_determinants
        fixed_params['pretrain_orbitals'] , (E_hf, E_casscf) = get_baseline_solution(phys_config, config.baseline, n_determinants)
        LOGGER.debug(f"Finished baseline calculation for pretraining: E_HF = {E_hf:.6f}, E_casscf={E_casscf:.6f}")

    def loss_func(params, batch):
        r, R, Z, fixed_params = batch
        # Calculate HF / CASSCF reference orbitals
        diff_el_ion, dist_el_ion = get_el_ion_distance_matrix(r, R)
        mo_up_ref, mo_dn_ref = get_baseline_slater_matrices(diff_el_ion, dist_el_ion,
                                                            fixed_params["pretrain_orbitals"],
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
    rng_mcmc, rng_opt = jax.random.split(jax.random.PRNGKey(rng_seed),2)
    logging.debug(f"Starting pretraining...")
    mcmc = MetropolisHastingsMonteCarlo(config.mcmc)
    mcmc_state = MCMCState.resize_or_init(mcmc_state, config.mcmc.n_walkers, phys_config, rng_mcmc)

    # Split/Replicate data across devices
    mcmc_state = mcmc_state.split_across_devices()
    params, fixed_params, opt_state = replicate_across_devices((params, fixed_params, opt_state))
    mcmc_state.log_psi_sqr = log_psi_sqr_pmapped(params, *mcmc_state.build_batch(fixed_params))

    # MCMC burn-in
    mcmc_state = mcmc.run_burn_in(log_psi_squared_func, mcmc_state, params, fixed_params)

    # Init optimizer
    optimizer = build_optimizer(jax.value_and_grad(loss_func), config.optimizer, False, False)
    opt_state = opt_state or optimizer.init(params, rng_opt, mcmc_state.build_batch(fixed_params))

    # Pre-training optimization loop
    for n in range(config.n_epochs):
        mcmc_state = mcmc.run_inter_steps(log_psi_squared_func, mcmc_state, params, fixed_params)
        params, opt_state, stats = optimizer.step(params, opt_state, None, batch=mcmc_state.build_batch(fixed_params))
        mcmc_state.log_psi_sqr = log_psi_sqr_pmapped(params, *mcmc_state.build_batch(fixed_params))
        mcmc_state_merged = mcmc_state.merge_devices()

        if loggers is not None:
            loggers.log_metrics(dict(loss=float(stats['loss'].mean()),
                                     mcmc_stepsize=float(mcmc_state_merged.stepsize.mean()),
                                     mcmc_step_nr=int(mcmc_state_merged.step_nr.mean())
                                     ),
                                epoch=n,
                                metric_type="pre")

    params, opt_state = get_from_devices((params, opt_state))
    return params, opt_state, mcmc_state_merged

