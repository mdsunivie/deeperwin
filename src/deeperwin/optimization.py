"""
Logic for wavefunction optimization.
"""
import time
from jax.flatten_util import ravel_pytree
import jax.profiler
from deeperwin.bfgs import build_bfgs_optimizer, calculate_metrics_bfgs
from deeperwin.configuration import OptimizationConfig, EvaluationConfig, ClippingConfig, LoggingConfig, PreTrainingConfig, DeepErwinModelConfig, PhysicalConfig
from deeperwin.evaluation import evaluate_wavefunction, build_evaluation_step
from deeperwin.hamiltonian import *
from deeperwin.kfac import build_grad_loss_kfac, build_optimize_epoch_kfac_test
from deeperwin.loggers import DataLogger, prepare_checkpoint_directory
from deeperwin.mcmc import MetropolisHastingsMonteCarlo, MCMCState
from deeperwin.utils import get_builtin_optimizer, calculate_clipping_state, prepare_data_for_logging, _update_adam_opt_state, \
    _update_adam_scaled_opt_state, set_scaled_optimizer_lr, calculate_metrics
from deeperwin.model import evaluate_sum_of_determinants, get_baseline_slater_matrices
from deeperwin.orbitals import get_baseline_solution

LOGGER = logging.getLogger("dpe")

def build_optimizer(
    log_psi_squared_func, grad_loss_func, mcmc, initial_params, fixed_params, opt_config: OptimizationConfig, n_walkers, mcmc_state=None
):
    """
    Builds an optimizer for optimizing the wavefunction model defined by `log_psi_squared_func`.

    Args:
        log_psi_squared_func (callable): A function representing the wavefunction model
        grad_loss_func (callable): A function that yields gradients for minimizing the local energies
        mcmc (MetropolisHastingsMonteCarlo): Object that implements the MCMC algorithm
        initial_params (array): Initial trainable parameters for the model defined by `log_psi_squared_func`
        fixed_params (array): Fixed paraemters of the model defined by `log_psi_squared_func`
        opt_config (OptimizationConfig): Optimization hyperparameters
        n_walkers (int): Number of MCMC walkers
        mcmc_state (MCMCState): Initial state of the MCMC walkers

    Returns:
        A tuple (get_params, optimize_epoch, opt_state) where get_params is a callable that extracts the (optimized) trainable parameters from and optimization state, optimize_epoch is a callable that performs one epoch of wavefunction optimization and opt_state is the initial optimization state.

    """
    if opt_config.optimizer.order == 1:
        opt_init, opt_update, opt_get_params = get_builtin_optimizer(opt_config.optimizer, opt_config.schedule, opt_config.learning_rate)
        opt_state = opt_init(initial_params)
        optimize_epoch = build_optimize_epoch_first_order(
            log_psi_squared_func, grad_loss_func, mcmc, opt_get_params, opt_update, opt_config, n_walkers
        )

        if opt_config.optimizer.name == "adam":
            change_parameters_fn = _update_adam_opt_state
        elif opt_config.optimizer.name == "adam_scaled":
            change_parameters_fn = _update_adam_scaled_opt_state
        else:
            change_parameters_fn = None
        return opt_get_params, optimize_epoch, opt_state, change_parameters_fn
    elif opt_config.optimizer.name == "slbfgs":
        return build_bfgs_optimizer(log_psi_squared_func, grad_loss_func, mcmc, initial_params, opt_config, n_walkers)
    elif opt_config.optimizer.name == "kfac":
        if opt_config.optimizer.internal_optimizer.name == "adam":
            change_parameters_fn = _update_adam_opt_state
        elif opt_config.optimizer.internal_optimizer.name == "adam_scaled":
            change_parameters_fn = _update_adam_scaled_opt_state

        def _update_kfac_opt_state(opt_state_old, params):
            adam_state = change_parameters_fn(opt_state_old[0], params)
            opt_state = (adam_state, params, opt_state_old[-2], opt_state_old[-1])
            return opt_state

        n_batches = n_walkers // opt_config.batch_size
        func_state = (mcmc_state.R, mcmc_state.Z, fixed_params, (jnp.array([0.0]).squeeze(), jnp.array([1000.0]).squeeze()))
        r_batch = mcmc_state.r.reshape([n_batches, opt_config.batch_size, -1, 3])[0]
        # opt_get_params, optimizer, opt_state = build_kfac_optimizer(grad_loss_func, initial_params, opt_config, r_batch, func_state)
        # optimize_epoch = build_optimize_epoch_kfac(log_psi_squared_func, mcmc, optimizer, opt_config, n_walkers, n_batches)
        optimize_epoch, opt_get_params, opt_state = build_optimize_epoch_kfac_test(
            grad_loss_func, initial_params, opt_config, r_batch, func_state, log_psi_squared_func, mcmc, n_walkers, n_batches
        )
        return opt_get_params, optimize_epoch, opt_state, _update_kfac_opt_state
    else:
        raise ValueError("Unsupported optimizer")


def build_optimize_epoch_first_order(log_psi_squared_func, grad_loss_func, mcmc, opt_get_params, opt_update_params, opt_config, n_walkers):
    """
    Returns a callable that optimizes the model defined by `log_psi_squared_func` for a single epoch with a first-order method.

    Args:
        log_psi_squared_func (callable): A function representing the wavefunction model
        grad_loss_func (callable): A function that yields gradients for minimizing the local energies
        mcmc (MetropolisHastingsMonteCarlo): Object that implements the MCMC algorithm
        opt_get_params (callable): A function to extract the trainable parameters from an optimization state object
        opt_update_params (callable): Function that implements an update rule for the model weights given the gradient of the loss function
        opt_config (OptimizationConfig): Optimization hyperparameters
        n_walkers (int): Number of MCMC walkers

    """
    n_batches_per_epoch = n_walkers // opt_config.batch_size

    @jax.jit
    def _optimize_epoch(epoch_nr, mcmc_state: MCMCState, opt_state, clipping_state, fixed_params):
        mcmc_state = mcmc.run_inter_steps(log_psi_squared_func, (opt_get_params(opt_state), fixed_params), mcmc_state)
        r_batches = mcmc_state.r.reshape([n_batches_per_epoch, opt_config.batch_size, -1, 3])

        def _batch_step(opt_state, r_batch):
            grads, E_batch, E_batch_unclipped = grad_loss_func(r_batch, mcmc_state.R, mcmc_state.Z, opt_get_params(opt_state), fixed_params, clipping_state)
            opt_state = opt_update_params(epoch_nr, grads, opt_state)
            return opt_state, (E_batch, E_batch_unclipped)

        opt_state, (E_epoch, E_epoch_unclipped) = jax.lax.scan(_batch_step, opt_state, r_batches)
        E_epoch = E_epoch.flatten()
        E_epoch_unclipped = E_epoch_unclipped.flatten()

        clipping_state = calculate_clipping_state(E_epoch, opt_config.clipping)
        mcmc_state.log_psi_sqr = log_psi_squared_func(*mcmc_state.model_args, opt_get_params(opt_state), fixed_params)
        return E_epoch, E_epoch_unclipped, mcmc_state, opt_state, clipping_state

    return _optimize_epoch


def build_grad_loss(log_psi_func, clipping_config: ClippingConfig, use_fwd_fwd_hessian=False):
    """
    Returns a callable that computes the gradient of the mean local energy for a given set of MCMC walkers with respect to the model defined by `log_psi_func`.

    Args:
        log_psi_func (callable): A function representing the wavefunction model
        clipping_config (ClippingConfig): Clipping hyperparameters
        use_fwd_fwd_hessian (bool): If true, the second partial derivatives required for computing the local energy are obtained with a forward-forward scheme.

    """

    @jax.custom_jvp
    def total_energy(trainable_params, aux_params):
        r, R, Z, fixed_params, clipping_state = aux_params
        E_loc_unclipped = get_local_energy(log_psi_func, r, R, Z, trainable_params, fixed_params, use_fwd_fwd_hessian)
        E_loc = clip_energies(E_loc_unclipped, *clipping_state, clipping_config)
        E_loc = jnp.where(jnp.isnan(E_loc), jnp.nanmean(E_loc) * jnp.ones_like(E_loc), E_loc)
        return jnp.mean(E_loc), (E_loc, E_loc_unclipped)

    @total_energy.defjvp
    def total_energy_jvp(primals, tangents):
        trainable_params, (r, R, Z, fixed_params, clipping_state) = primals
        E_mean, (E_loc, E_loc_unclipped) = total_energy(*primals)
        log_psi_func_simplified = lambda p: log_psi_func(r, R, Z, p, fixed_params)
        psi_primal, psi_tan = jax.jvp(log_psi_func_simplified, primals[:1], tangents[:1])
        primals_out = (E_mean, (E_loc, E_loc_unclipped))
        tangents_out = (jnp.dot(psi_tan, E_loc - E_mean) / len(E_loc), (E_loc, E_loc_unclipped))
        return primals_out, tangents_out

    def grad_loss_func(r, R, Z, trainable_params, fixed_params, clipping_state):
        grads, (E_batch, E_batch_unclipped) = jax.grad(total_energy, argnums=0, has_aux=True)(trainable_params, (r, R, Z, fixed_params, clipping_state))
        grads_flat, unravel_func = ravel_pytree(grads)
        grads_flat = jnp.nan_to_num(grads_flat, nan=0.0)
        grads = unravel_func(grads_flat)
        return grads, E_batch, E_batch_unclipped

    return grad_loss_func


def optimize_wavefunction(
        log_psi_squared,
        initial_trainable_params,
        fixed_params,
        mcmc: MetropolisHastingsMonteCarlo,
        mcmc_state: MCMCState,
        opt_config: OptimizationConfig,
        checkpoints=None,
        logger: DataLogger = None,
        initial_opt_state=None,
        initial_clipping_state=None,
        E_ref=None,
        use_profiler= False
):
    """
    Minimizes the energy of the wavefunction defined by the callable `log_psi_squared` by adjusting the trainable parameters.

    Args:
        log_psi_func (callable): A function representing the wavefunction model
        initial_trainable_params (dict): Trainable paramters of the model defined by `log_psi_func`
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
    checkpoints = checkpoints or []
    n_walkers = mcmc_state.r.shape[0]
    clipping_state = initial_clipping_state or (jnp.array([0.0]).squeeze(), jnp.array([1e5]).squeeze())  # do not clip at first epoch, then adjust

    if opt_config.optimizer.name == "kfac":
        grad_loss_func = build_grad_loss_kfac(log_psi_squared, opt_config.clipping)
    else:
        grad_loss_func = build_grad_loss(log_psi_squared, opt_config.clipping)

    opt_get_params, optimize_epoch, opt_state, _ = build_optimizer(
            log_psi_squared, grad_loss_func, mcmc, initial_trainable_params, fixed_params, opt_config, n_walkers, mcmc_state
        )
    if initial_opt_state is not None:
        opt_state = initial_opt_state

    if mcmc.config.n_burn_in_opt > 0:
        LOGGER.debug("Starting burn-in for optimization...")
        mcmc_state = mcmc.run_burn_in_opt(log_psi_squared, (initial_trainable_params, fixed_params), mcmc_state)

    if opt_config.intermediate_eval is not None:
        eval_checkpoints = set(opt_config.intermediate_eval.opt_epochs)
    else:
        eval_checkpoints = set()
    intermediate_eval_step_func = None

    t_start = time.time()
    E_mean_unclipped_history = []
    for n_epoch in range(opt_config.n_epochs_prev, opt_config.n_epochs_prev+opt_config.n_epochs):
        if use_profiler and (n_epoch == 2):
            jax.profiler.start_trace('profiler')
        mcmc_state_old = mcmc_state
        E_epoch, E_epoch_unclipped, mcmc_state, opt_state, clipping_state = optimize_epoch(n_epoch, mcmc_state, opt_state, clipping_state, fixed_params)
        if use_profiler and (n_epoch == 2):
            jax.profiler.stop_trace()
        t_end = time.time()

        if (n_epoch in checkpoints) and (logger is not None):
            checkpoint_dir = prepare_checkpoint_directory(n_epoch)
            full_data = prepare_data_for_logging(opt_get_params(opt_state), fixed_params, mcmc_state, opt_state, clipping_state)
            logger.log_weights(full_data)
            LOGGER.info(f"Logging checkpoint to folder {checkpoint_dir}")
            logger.log_checkpoint(checkpoint_dir, n_epoch)

        if logger is not None:
            E_ref = fixed_params['baseline_energies']['E_ref']
            E_mean_unclipped_history.append(np.nanmean(np.array(E_epoch_unclipped, dtype=float)))
            logger.log_metrics(*calculate_metrics(n_epoch, E_epoch, E_epoch_unclipped, E_mean_unclipped_history, mcmc_state, mcmc_state_old, t_end - t_start, "opt", E_ref=E_ref))
            if opt_config.optimizer.name == "slbfgs":
                logger.log_metrics(calculate_metrics_bfgs(opt_state), n_epoch, "opt")

        if n_epoch in eval_checkpoints:
            LOGGER.debug(f"opt epoch {n_epoch:5d}: Running intermediate evaluation...")
            eval_config = EvaluationConfig(n_epochs=opt_config.intermediate_eval.n_epochs)
            if intermediate_eval_step_func is None:
                intermediate_eval_step_func = build_evaluation_step(log_psi_squared, mcmc, eval_config)
            E_eval, _, _ = evaluate_wavefunction(
                log_psi_squared, opt_get_params(opt_state), fixed_params, mcmc, mcmc_state, eval_config, logger=logger,
                evaluation_step_func=intermediate_eval_step_func, evaluation_type="Intermed_eval"
            )
            intermed_metrics = dict(E_intermed_eval_mean=jnp.nanmean(E_eval), E_intermed_eval_std=jnp.nanstd(E_eval))
            if E_ref is not None:
                intermed_metrics['error_intermed_eval'] = 1e3 * (intermed_metrics['E_intermed_eval_mean'] - E_ref)
                intermed_metrics['sigma_intermed_eval'] = 1e3 * intermed_metrics['E_intermed_eval_std'] / np.sqrt(eval_config.n_epochs)
            logger.log_metrics(intermed_metrics, n_epoch, "opt", force_log=True)
        t_start = t_end
    LOGGER.debug("Finished wavefunction optimization...")
    return mcmc_state, opt_get_params(opt_state), opt_state, clipping_state


def pretrain_orbitals(orbital_func, mcmc, mcmc_state: MCMCState, initial_trainable_params, fixed_params,
                      config: PreTrainingConfig,
                      physical_config: PhysicalConfig,
                      model_config: DeepErwinModelConfig,
                      loggers: DataLogger = None, opt_state=None):

    if model_config.orbitals.baseline_orbitals and (model_config.orbitals.baseline_orbitals.baseline == config.baseline):
        baseline_orbitals = fixed_params['orbitals']
        LOGGER.debug("Identical CASSCF-config for pre-training and baseline model: Reusing baseline calculation")
    else:
        LOGGER.warning("Using different baseline pySCF settings for pre-training and the baked-in baseline model. Calculating new orbitals...")
        n_determinants = 1 if config.use_only_leading_determinant else model_config.orbitals.n_determinants
        baseline_orbitals, (E_hf, E_casscf) = get_baseline_solution(physical_config, config.baseline, n_determinants)
        LOGGER.debug(f"Finished baseline calculation for pretraining: E_HF = {E_hf:.6f}, E_casscf={E_casscf:.6f}")

    def loss_func(r, R, Z, params, fixed_params):
        # Calculate HF / CASSCF reference orbitals
        diff_el_ion, dist_el_ion = get_el_ion_distance_matrix(r, R)
        mo_up_ref, mo_dn_ref = get_baseline_slater_matrices(diff_el_ion, dist_el_ion, baseline_orbitals, model_config.orbitals.use_full_det)

        if config.use_only_leading_determinant:
            mo_up_ref = mo_up_ref[...,:1,:,:]
            mo_dn_ref = mo_dn_ref[...,:1,:,:]

        # Calculate neural net orbitals
        mo_up, mo_dn = orbital_func(r, R, Z, params, fixed_params)
        residual_up = mo_up - mo_up_ref
        residual_dn = mo_dn - mo_dn_ref
        return jnp.mean(residual_up**2) + jnp.mean(residual_dn**2)

    def log_psi_squared_func(r, R, Z, params, fixed_params):
        mo_up, mo_dn = orbital_func(r, R, Z, params, fixed_params)
        return evaluate_sum_of_determinants(mo_up, mo_dn, model_config.orbitals.use_full_det)

    opt_init, opt_update_params, opt_get_params = get_builtin_optimizer(config.optimizer, config.schedule, config.learning_rate)

    @jax.jit
    def pretrain_step(epoch_nr, mcmc_state: MCMCState, opt_state):
        loss, grads = jax.value_and_grad(loss_func, argnums=3)(*mcmc_state.model_args, opt_get_params(opt_state), fixed_params)
        opt_state = opt_update_params(epoch_nr, grads, opt_state)

        mcmc_state.log_psi_sqr = log_psi_squared_func(*mcmc_state.model_args, opt_get_params(opt_state), fixed_params)
        # mcmc_state = mcmc.run_inter_steps(log_psi_squared_func, params, mcmc_state)
        mcmc_state = mcmc.make_mcmc_step(log_psi_squared_func, (opt_get_params(opt_state), fixed_params), mcmc_state, 'opt', config.n_epochs)
        return opt_state, mcmc_state, loss

    logging.debug(f"Starting pretraining...")
    opt_state = opt_state or opt_init(initial_trainable_params)
    mcmc_state.log_psi_sqr = log_psi_squared_func(*mcmc_state.model_args, opt_get_params(opt_state), fixed_params)
    for n in range(config.n_epochs):
        opt_state, mcmc_state, loss = pretrain_step(n, mcmc_state, opt_state)
        if loggers is not None:
            loggers.log_metrics(dict(loss=float(loss), mcmc_stepsize=float(mcmc_state.stepsize), mcmc_step_nr=int(mcmc_state.step_nr)), epoch=n, metric_type="pre")

        if (n in config.checkpoints) and (loggers is not None):
            checkpoint_dir = prepare_checkpoint_directory(n, pretraining_checkpoint=True)
            full_data = prepare_data_for_logging(opt_get_params(opt_state), fixed_params, mcmc_state, opt_state)
            loggers.log_weights(full_data)
            LOGGER.info(f"Logging checkpoint to folder {checkpoint_dir}")
            loggers.log_checkpoint(checkpoint_dir, n, pretraining=True)
    return opt_get_params(opt_state), opt_state, mcmc_state

