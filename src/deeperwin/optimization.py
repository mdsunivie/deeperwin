import time

from deeperwin.bfgs import build_bfgs_optimizer, calculate_metrics_bfgs
from deeperwin.configuration import OptimizationConfig, EvaluationConfig
from deeperwin.evaluation import evaluate_wavefunction
from deeperwin.hamiltonian import *
from deeperwin.kfac import build_grad_loss_kfac, build_optimize_epoch_kfac_test
from deeperwin.loggers import DataLogger
from deeperwin.mcmc import MetropolisHastingsMonteCarlo, MCMCState, calculate_metrics
from deeperwin.utils import get_builtin_optimizer, calculate_clipping_state, make_opt_state_picklable


def build_optimizer(log_psi_squared_func, grad_loss_func, mcmc, initial_params, fixed_params,
                    opt_config: OptimizationConfig, n_walkers, mcmc_state=None):
    if opt_config.optimizer.order == 1:
        opt_init, opt_update, opt_get_params = get_builtin_optimizer(opt_config.optimizer, opt_config.schedule,
                                                                     opt_config.learning_rate)
        opt_state = opt_init(initial_params)
        optimize_epoch = build_optimize_epoch_first_order(log_psi_squared_func, grad_loss_func, mcmc, opt_get_params,
                                                          opt_update, opt_config, n_walkers)
        return opt_get_params, optimize_epoch, opt_state
    elif opt_config.optimizer.name == 'slbfgs':
        return build_bfgs_optimizer(log_psi_squared_func, grad_loss_func, mcmc, initial_params, opt_config, n_walkers)
    elif opt_config.optimizer.name == 'kfac':
        n_batches = n_walkers // opt_config.batch_size
        func_state = (
            mcmc_state.R, mcmc_state.Z, fixed_params, (jnp.array([0.0]).squeeze(), jnp.array([1000.0]).squeeze()))
        r_batch = mcmc_state.r.reshape([n_batches, opt_config.batch_size, -1, 3])[0]
        # opt_get_params, optimizer, opt_state = build_kfac_optimizer(grad_loss_func, initial_params, opt_config, r_batch, func_state)
        # optimize_epoch = build_optimize_epoch_kfac(log_psi_squared_func, mcmc, optimizer, opt_config, n_walkers, n_batches)
        optimize_epoch, opt_get_params, opt_state = build_optimize_epoch_kfac_test(grad_loss_func, initial_params,
                                                                                   opt_config, r_batch, func_state,
                                                                                   log_psi_squared_func,
                                                                                   mcmc, n_walkers, n_batches)
        return opt_get_params, optimize_epoch, opt_state
    else:
        raise ValueError("Unsupported optimizer")


def build_optimize_epoch_first_order(log_psi_squared_func, grad_loss_func, mcmc, opt_get_params, opt_update_params,
                                     opt_config, n_walkers):
    n_batches_per_epoch = n_walkers // opt_config.batch_size

    @jax.jit
    def _optimize_epoch(epoch_nr, mcmc_state: MCMCState, opt_state, clipping_state, fixed_params):
        mcmc_state = mcmc.run_inter_steps(log_psi_squared_func, (opt_get_params(opt_state), fixed_params), mcmc_state)
        r_batches = mcmc_state.r.reshape([n_batches_per_epoch, opt_config.batch_size, -1, 3])

        def _batch_step(opt_state, r_batch):
            grads, E_batch = grad_loss_func(r_batch, mcmc_state.R, mcmc_state.Z, opt_get_params(opt_state),
                                            fixed_params, clipping_state)
            opt_state = opt_update_params(epoch_nr, grads, opt_state)
            return opt_state, E_batch

        opt_state, E_epoch = jax.lax.scan(_batch_step, opt_state, r_batches)
        E_epoch = E_epoch.flatten()

        clipping_state = calculate_clipping_state(E_epoch, opt_config.clipping)
        mcmc_state.log_psi_sqr = log_psi_squared_func(*mcmc_state.model_args, opt_get_params(opt_state), fixed_params)
        return E_epoch, mcmc_state, opt_state, clipping_state

    return _optimize_epoch


def build_grad_loss(log_psi_func, clipping_config: ClippingConfig, use_fwd_fwd_hessian=False):
    @jax.custom_jvp
    def total_energy(trainable_params, aux_params):
        r, R, Z, fixed_params, clipping_state = aux_params
        E_loc_unclipped = get_local_energy(log_psi_func, r, R, Z, trainable_params, fixed_params, use_fwd_fwd_hessian)
        E_loc = clip_energies(E_loc_unclipped, *clipping_state, clipping_config)
        return jnp.mean(E_loc), E_loc

    @total_energy.defjvp
    def total_energy_jvp(primals, tangents):
        trainable_params, (r, R, Z, fixed_params, clipping_state) = primals
        E_mean, E_loc = total_energy(*primals)
        log_psi_func_simplified = lambda p: log_psi_func(r, R, Z, p, fixed_params)
        psi_primal, psi_tan = jax.jvp(log_psi_func_simplified, primals[:1], tangents[:1])
        primals_out = (E_mean, E_loc)
        tangents_out = (jnp.dot(psi_tan, E_loc - E_mean) / len(E_loc), E_loc)
        return primals_out, tangents_out

    def grad_loss_func(r, R, Z, trainable_params, fixed_params, clipping_state):
        return jax.grad(total_energy, argnums=0, has_aux=True)(trainable_params,
                                                               (r, R, Z, fixed_params, clipping_state))

    return grad_loss_func


def optimize_wavefunction(
        log_psi_squared,
        initial_trainable_params,
        fixed_params,
        mcmc: MetropolisHastingsMonteCarlo,
        mcmc_state: MCMCState,
        opt_config: OptimizationConfig,
        checkpoints = {},
        logger: DataLogger = None
):
    n_walkers = mcmc_state.r.shape[0]
    clipping_params = (
        jnp.array([0.0]).squeeze(), jnp.array([1000.0]).squeeze())  # do not clip at first epoch, then adjust

    if opt_config.optimizer.name == 'kfac':
        grad_loss_func = build_grad_loss_kfac(log_psi_squared, opt_config.clipping)
    else:
        grad_loss_func = build_grad_loss(log_psi_squared, opt_config.clipping)
    opt_get_params, optimize_epoch, opt_state = build_optimizer(log_psi_squared, grad_loss_func, mcmc,
                                                                initial_trainable_params, fixed_params, opt_config,
                                                                n_walkers, mcmc_state)

    logging.debug("Starting burn-in for optimization...")
    mcmc_state = mcmc.run_burn_in_opt(log_psi_squared, (initial_trainable_params, fixed_params), mcmc_state)

    if opt_config.intermediate_eval is not None:
        eval_checkpoints = set(opt_config.intermediate_eval.opt_epochs)
    else:
        eval_checkpoints = set()
    t_start = time.time()
    for n_epoch in range(opt_config.n_epochs):
        E_epoch, mcmc_state, opt_state, clipping_params = optimize_epoch(n_epoch, mcmc_state, opt_state,
                                                                         clipping_params, fixed_params)
        t_end = time.time()

        if n_epoch in checkpoints and logger is not None:
            full_data = dict(trainable=opt_get_params(opt_state), fixed=fixed_params, mcmc=mcmc_state, opt = make_opt_state_picklable(opt_state))
            logger.log_weights(full_data)
            logging.info(f"Logging checkpoint to folder {checkpoints[n_epoch]}")
            logger.log_checkpoint(checkpoints[n_epoch])

        if logger is not None:
            logger.log_metrics(*calculate_metrics(n_epoch, 1, E_epoch, mcmc_state, t_end - t_start, "opt"))
            if opt_config.optimizer.name == 'slbfgs':
                logger.log_metrics(calculate_metrics_bfgs(opt_state), n_epoch, "opt")
        if n_epoch in eval_checkpoints:
            logging.debug(f"opt epoch {n_epoch:5d}: Running intermediate evaluation...")
            E_eval, _, _ = evaluate_wavefunction(log_psi_squared, opt_get_params(opt_state), fixed_params, mcmc,
                                              mcmc_state, EvaluationConfig(n_epochs=opt_config.intermediate_eval.n_epochs))
            logger.log_metrics(dict(E_intermed_eval_mean=jnp.mean(E_eval), E_intermed_eval_std=jnp.std(E_eval)),
                               n_epoch, "opt")
        t_start = t_end
    logging.debug("Finished wavefunction optimization...")
    return mcmc_state, opt_get_params(opt_state), opt_state
