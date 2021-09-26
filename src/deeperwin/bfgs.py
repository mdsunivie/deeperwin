"""
Implementation of the BFGS 2nd-order optimizer.
"""

import jax
import numpy as np
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree

from deeperwin.configuration import BFGSOptimizerConfig, OptimizationConfig
from deeperwin.mcmc import MCMCState, MetropolisHastingsMonteCarlo
from deeperwin.utils import get_builtin_optimizer, calculate_clipping_state


def calculate_hvp_by_2loop_recursion(inv_hessian, v):
    """
    Calculate the hessian vector product Hv, where H is implicitly defined by the vectors s and y.
    """
    s, y, rho = inv_hessian

    # Loop 1: recurse into the iterated matrix product
    def _loop_inwards(v, xs):
        s, y, rho = xs
        alpha = rho * jnp.dot(s, v)
        v -= alpha * y
        return v, alpha

    # Loop 2: Recurse back out
    def _loop_outwards(v, xs):
        s, y, rho, alpha = xs
        beta = rho * jnp.dot(y, v)
        v += s * (alpha - beta)
        return v, beta

    v, alpha = jax.lax.scan(_loop_inwards, v, (s, y, rho), reverse=True)
    # Apply no internal update to v <==> identity as initial hessian guess
    v, _ = jax.lax.scan(_loop_outwards, v, (s, y, rho, alpha))
    return v


def update_hessian_representation(inv_hessian, s_new, y_new):
    EPSILON_RHO = 1e-10
    s, y, rho = inv_hessian
    rho_new = 1.0 / (EPSILON_RHO + jnp.dot(s_new, y_new))
    s = jnp.concatenate([s[1:], s_new[np.newaxis, :]], axis=0)
    y = jnp.concatenate([y[1:], y_new[np.newaxis, :]], axis=0)
    rho = jnp.concatenate([rho[1:], rho_new.reshape((1,))], axis=0)
    return s, y, rho


def _calculate_search_direction(grad, inv_hessian, norm_constraint):
    """
    Calculates the search direction x = H^-1 g and scales the step, so that the norm of xHx does not exceed the norm_constraint
    """
    step = calculate_hvp_by_2loop_recursion(inv_hessian, grad)  # apply inverse Hessiang: g = H^-1g
    norm = jnp.dot(grad, step)  # calculate step H step (= grad H^-1 grad)
    scaling = jnp.minimum(jnp.sqrt(norm_constraint / norm), 1.0)
    return scaling * step


def calculate_metrics_bfgs(opt_state):
    rho = opt_state[1][2]
    sy = 1.0 / rho
    return dict(sy_min=float(jnp.min(sy)), sy_max=float(jnp.max(sy)), sy_mean=float(jnp.mean(sy)))


def _regularize_curvature_pair(s, y, fixed_regularization):
    reg = -jnp.dot(s, y) / jnp.dot(s, s)
    reg = jnp.maximum(reg, 0.0) + fixed_regularization
    return s, y + reg * s


def build_bfgs_optimizer(
        log_psi_squared_func,
        grad_loss_func,
        mcmc: MetropolisHastingsMonteCarlo,
        initial_params,
        opt_config: OptimizationConfig,
        n_walkers
):
    initial_params_flat, unravel_pytree = ravel_pytree(initial_params)
    n_params = initial_params_flat.shape[0]

    def grad_loss_flat(r, R, Z, params_flat, fixed_params, clipping_state):
        g, l = grad_loss_func(r, R, Z, unravel_pytree(params_flat), fixed_params, clipping_state)
        return ravel_pytree(g)[0], l

    bfgs_config: BFGSOptimizerConfig = opt_config.optimizer
    n_batches = n_walkers // opt_config.batch_size
    opt_init, opt_update, get_params = get_builtin_optimizer(bfgs_config.internal_optimizer, opt_config.schedule,
                                                             opt_config.learning_rate)

    @jax.jit
    def optimize_epoch_bfgs(epoch, mcmc_state: MCMCState, opt_state, clipping_state, fixed_params):
        inner_opt_state, inv_hessian = opt_state

        mcmc_state = mcmc.run_inter_steps(log_psi_squared_func,
                                          (unravel_pytree(get_params(inner_opt_state)), fixed_params), mcmc_state)
        r_batches = mcmc_state.r.reshape([n_batches, opt_config.batch_size, -1, 3])

        def _batch_step(carry, r_batch):
            params_before_step, inner_opt_state = carry
            params_before_step = get_params(inner_opt_state)
            raw_grad, E_loc = grad_loss_flat(r_batch, mcmc_state.R, mcmc_state.Z, params_before_step, fixed_params,
                                             clipping_state)
            g = _calculate_search_direction(raw_grad, inv_hessian, bfgs_config.norm_constraint)
            inner_opt_state = opt_update(epoch, g, inner_opt_state)
            return (params_before_step, inner_opt_state), (E_loc, raw_grad)

        params_old = get_params(inner_opt_state)
        (params_new, inner_opt_state), (E_epoch, gradients) = jax.lax.scan(_batch_step,
                                                                           (jnp.zeros([n_params]), inner_opt_state),
                                                                           r_batches)
        E_epoch = E_epoch.flatten()

        def _update_hessian(inv_hessian):
            s_new = params_new - params_old
            y_new = gradients[-1] - gradients[0]
            if bfgs_config.use_variance_reduction:
                # Calculate 2 additional batch-gradients for variance reduction
                y_new += \
                grad_loss_flat(r_batches[0], mcmc_state.R, mcmc_state.Z, params_new, fixed_params, clipping_state)[0]
                y_new -= \
                grad_loss_flat(r_batches[-1], mcmc_state.R, mcmc_state.Z, params_old, fixed_params, clipping_state)[0]
                y_new *= 0.5  # Gradient is the average of 2 batches => divide by 2
            # Update inverse hessian
            s_new, y_new = _regularize_curvature_pair(s_new, y_new, bfgs_config.hessian_regularization)
            return update_hessian_representation(inv_hessian, s_new, y_new)

        inv_hessian = jax.lax.cond(epoch % bfgs_config.update_hessian_every_n_epochs == 0, _update_hessian, lambda x: x,
                                   inv_hessian)

        # Update clipping and wavefunction values
        clipping_state = calculate_clipping_state(E_epoch, opt_config.clipping)
        mcmc_state.log_psi_sqr = log_psi_squared_func(*mcmc_state.model_args,
                                                      unravel_pytree(get_params(inner_opt_state)), fixed_params)
        return E_epoch, mcmc_state, (inner_opt_state, inv_hessian), clipping_state

    def _get_params(opt_state):
        return unravel_pytree(get_params(opt_state[0]))

    s_init = jnp.zeros([bfgs_config.memory_length, n_params])
    y_init = jnp.zeros([bfgs_config.memory_length, n_params])
    rho_init = jnp.zeros([bfgs_config.memory_length])
    initial_state = opt_init(initial_params_flat), (s_init, y_init, rho_init)

    return _get_params, optimize_epoch_bfgs, initial_state
