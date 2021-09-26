"""
Helper functions for K-FAC 2nd-order optimization.
"""

from deeperwin.configuration import OptimizationConfig
from deeperwin.hamiltonian import *
from deeperwin.kfac_ferminet_alpha import loss_functions
from deeperwin.kfac_ferminet_alpha import optimizer as kfac_optim
from deeperwin.utils import build_inverse_schedule, get_builtin_optimizer, calculate_clipping_state


def build_grad_loss_kfac(log_psi_func, clipping_config: ClippingConfig, use_fwd_fwd_hessian=False):
    # Build custom total energy jvp. Based on https://github.com/deepmind/ferminet/blob/jax/ferminet/train.py
    @jax.custom_jvp
    def total_energy(params, state, r):
        R, Z, fixed_params, clipping = state
        E_loc = get_local_energy(log_psi_func, r, R, Z, params, fixed_params, use_fwd_fwd_hessian)
        if clipping_config.unclipped_center:
            center = jnp.nanmean(E_loc)
            width = jnp.nanmean(jnp.abs(E_loc - center)) * 5.0
            E_loc = clip_energies(E_loc, center, width, clipping_config)
        else:
            E_loc = clip_energies(E_loc, *clipping, clipping_config)
        loss = jnp.mean(E_loc)
        return loss, ((R, Z, fixed_params, clipping), E_loc)

    @total_energy.defjvp
    def total_energy_jvp(primals, tangents):
        params, (R, Z, fixed_params, clipping), r = primals
        loss, ((_, _, _, _), E_loc) = total_energy(params, (R, Z, fixed_params, clipping), r)
        diff = E_loc - loss
        func = lambda p, r: log_psi_func(r, R, Z, p, fixed_params)
        psi_primal, psi_tangent = jax.jvp(func, (primals[0], primals[-1]), (tangents[0], tangents[-1]))
        loss_functions.register_normal_predictive_distribution(psi_primal[:, None])  # Register loss for kfac optimizer
        primals_out = loss, ((R, Z, fixed_params, clipping), E_loc)
        tangents_out = jnp.dot(psi_tangent, diff), ((R, Z, fixed_params, clipping), E_loc)

        return primals_out, tangents_out

    grad_loss_func = jax.value_and_grad(total_energy, argnums=0, has_aux=True)

    return grad_loss_func


def build_optimize_epoch_kfac_test(grad_loss_func, initial_params, opt_config: OptimizationConfig, r_batch, func_state,
                                   log_psi_squared, mcmc, n_walkers, n_batches):
    learning_rate_scheduler = build_inverse_schedule(opt_config.learning_rate, opt_config.optimizer.decay_time)
    optimizer = kfac_optim.Optimizer(
        grad_loss_func,
        l2_reg=0.0,
        value_func_has_aux=True,
        value_func_has_state=True,
        inverse_update_period=opt_config.optimizer.update_inverse_period,
        value_func_has_rng=False,
        learning_rate_schedule=learning_rate_scheduler,
        num_burnin_steps=0,
        register_only_generic=opt_config.optimizer.register_generic,
        norm_constraint=opt_config.optimizer.norm_constraint,
        estimation_mode=opt_config.optimizer.estimation_mode,
        min_damping=opt_config.optimizer.min_damping,
        multi_device=False,
        curvature_ema=1.0 - opt_config.optimizer.curvature_ema
        # debug=True
    )

    key_init = jax.random.PRNGKey(1245123)
    opt_key, subkeys = jax.random.split(key_init, 2)

    kfac_opt_state = optimizer.init(initial_params, subkeys, r_batch, func_state)

    if opt_config.optimizer.internal_optimizer is not None:
        opt_init, opt_update, get_params_adam = get_builtin_optimizer(opt_config.optimizer.internal_optimizer,
                                                                      opt_config.schedule,
                                                                      opt_config.learning_rate)
        adam_state = opt_init(initial_params)

        opt_state = (adam_state, get_params_adam(adam_state), kfac_opt_state, opt_key)

        def _get_params(state):
            return get_params_adam(state[0])

        @jax.jit
        def update_with_adam(epoch_nr, grads, adam_state):
            adam_state = opt_update(epoch_nr, grads, adam_state)
            return get_params_adam(adam_state), adam_state

    else:
        opt_state = (None, initial_params, kfac_opt_state, opt_key)

        def _get_params(state):
            return state[1]

    if opt_config.optimizer.damping_scheduler:
        damping_scheduler = build_inverse_schedule(opt_config.optimizer.damping,
                                                   opt_config.optimizer.decay_time_damping)
    else:
        damping_scheduler = lambda x: opt_config.optimizer.damping

    def _optimize_epoch_with_kfac(epoch, mcmc_state, opt_state, clipping_params, fixed_params):
        adam_state, params, kfac_opt_state, key = opt_state

        func_state = (mcmc_state.R, mcmc_state.Z, fixed_params, clipping_params)
        mcmc_state = mcmc.run_inter_steps(log_psi_squared, (params, fixed_params), mcmc_state)
        r_batches = mcmc_state.r.reshape([n_batches, opt_config.batch_size, -1, 3])

        damping = damping_scheduler(epoch)
        momentum = opt_config.optimizer.momentum

        E_epoch = jnp.zeros(n_walkers)
        for i in range(n_batches):
            r_batch = r_batches[i]
            key, key_batch = jax.random.split(key, 2)
            params, kfac_opt_state, func_state, stats, grads = optimizer.step(
                params,
                kfac_opt_state,
                key_batch,
                iter([r_batch]),
                func_state=func_state,
                momentum=momentum,
                damping=damping)

            E_batch = stats['aux']
            E_epoch = jax.lax.dynamic_update_slice(E_epoch, E_batch, (i * opt_config.batch_size,))
            if opt_config.optimizer.internal_optimizer is not None:
                params, adam_state = update_with_adam(epoch, grads, adam_state)

        mcmc_state.log_psi_sqr = log_psi_squared(*mcmc_state.model_args, params, fixed_params)
        clipping_params = calculate_clipping_state(E_epoch, opt_config.clipping)

        opt_state = (adam_state, params, kfac_opt_state, key)

        return E_epoch, mcmc_state, opt_state, clipping_params

    return _optimize_epoch_with_kfac, _get_params, opt_state
