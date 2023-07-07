import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg
from typing import Any, Callable, Optional, Tuple, Union
from deeperwin.configuration import SRCGOptimizerConfig
from deeperwin.optimization.opt_utils import build_optax_optimizer, build_lr_schedule
from deeperwin.utils.utils import pmap, pmean, tree_dot, tree_norm
import optax

class SRCGOptimizer():
    def __init__(
        self,
        log_psi_squared: Callable,
        value_and_grad_func: Callable,
        config: SRCGOptimizerConfig,
    ):
        self.config = config
        self.log_psi_squared = log_psi_squared
        self.value_and_grad_func = value_and_grad_func
        self.internal_optimizer = build_optax_optimizer(config.internal_optimizer)
        self.damping = build_lr_schedule(config.damping, config.damping_schedule)
        self.learning_rate = build_lr_schedule(config.learning_rate, config.lr_schedule)
        self._jit_step = jax.pmap(self._step, axis_name="devices", static_broadcasted_argnums=(2,))

    def init(
            self,
            params,
            rng: jnp.ndarray,
            batch,
            static_args,
            func_state = None
    ):
        """Initializes the optimizer and returns the appropriate optimizer state."""
        del rng, func_state, static_args

        n_devices = batch[0].shape[0]
        internal_opt_state = pmap(self.internal_optimizer.init)(params)
        previous_nat_grad = jax.tree_map(jnp.zeros_like, params)
        step_count = jnp.zeros((n_devices,), dtype=jnp.int32)
        preconditioner = None
        return internal_opt_state, previous_nat_grad, preconditioner, step_count

    def step(
            self,
            params,
            state,
            static_args: Any,
            rng: jnp.ndarray,
            batch,
            func_state,
    ):
        return self._jit_step(params, state, static_args, rng, batch, func_state)
    
    def _get_gradient_variance_preconditioner(self, params, mean_grads, last_variance, damping, static_args, batch):
        """Compute the diagonal of S (which corresponds to the variance of the gradients) and use its inverse as a preconditioner."""
        r, R, Z, fixed_params = batch
        batch_size = r.shape[0]
        n_subbatches = batch_size // self.config.preconditioner_batch_size
        r_batches = r.reshape((n_subbatches, self.config.preconditioner_batch_size, -1, 3))

        def aggregate_grad_psi_sqr(grad_psi_sqr, r):
            grad_func = jax.grad(self.log_psi_squared, argnums=0)
            grad_psi = jax.vmap(grad_func, in_axes=(None, None, None, 0, None, None, None), out_axes=0)(params, *static_args, r, R, Z, fixed_params)
            grad_psi_sqr = jax.tree_util.tree_map(lambda old, g: old + jnp.sum((g/2) ** 2, axis=0), grad_psi_sqr, grad_psi)
            return grad_psi_sqr, None
        
        grad_psi_sqr = jax.tree_util.tree_map(jnp.zeros_like, params)
        grad_psi_sqr, _ = jax.lax.scan(aggregate_grad_psi_sqr, grad_psi_sqr, r_batches)
        grad_psi_sqr = jax.tree_util.tree_map(lambda x: x / batch_size, grad_psi_sqr)
        grad_psi_sqr = pmean(grad_psi_sqr)
        variance = jax.tree_util.tree_map(lambda g2, g: g2 - g**2, grad_psi_sqr, mean_grads)

        if last_variance is not None and self.config.preconditioner_ema:
            ema = self.config.preconditioner_ema
            variance = jax.tree_util.tree_map(lambda old, new: old * ema + new * (1-ema), last_variance, variance)

        def precond(x):
            return jax.tree_util.tree_map(lambda x_, v: x_ / (v + damping), x, variance)
        return precond, variance
        

    def _step(
            self,
            params,
            opt_state,
            static_args: Any,
            rng: jnp.ndarray,
            batch,
            func_state,
    ):
        """
        Calculates natural gradients & calculates updates in model parameters + optimizer state
        using the internal optax optmizer.
        """
        del rng

        internal_opt_state, previous_nat_grad, preconditioner_state, step_count = opt_state
        damping = self.damping(step_count)
        batch_size = batch[0].shape[0]

        def log_psi(params): 
            return self.log_psi_squared(params, *static_args, *batch) / 2

        if self.config.center_gradients or self.config.preconditioner:
            mean_grads = jax.grad(lambda p: jnp.mean(log_psi(p)))(params)
            mean_grads = pmean(mean_grads)

        if self.config.linearize_jvp:
            jvp_func = jax.linearize(log_psi, params)[1]
        else:
            jvp_func = lambda x: jax.jvp(log_psi, (params,), (x,))[1]
        def fisher_matmul(x):
            # Compute 1/batch_size * VJP(log_psi, JVP(log_psi, x))
            log_psi_jac_x = jvp_func(x)
            update, = jax.vjp(log_psi, params)[1](log_psi_jac_x / batch_size)
            if self.config.center_gradients:
                # update = update - g * <g, x>
                innerprod = tree_dot(mean_grads, x)
                update = jax.tree_util.tree_map(lambda u, g: u - g * innerprod, update, mean_grads)
            # update = update + damping * x
            update = jax.tree_util.tree_map(lambda u, x_: u + damping * x_, update, x)
            update = pmean(update)
            return update


        # Compute raw (= non-preconditioned) energy gradient
        (loss, (new_func_state, aux_metrics)), loss_grads = self.value_and_grad_func(params, func_state, static_args, batch)
        loss_grads = pmean(loss_grads)

        if self.config.preconditioner == "variance":
            preconditioner, preconditioner_state = self._get_gradient_variance_preconditioner(params, mean_grads, preconditioner_state, damping, static_args, batch)
        else:
            preconditioner = None

        if self.config.initial_guess == "zero":
            x0 = jax.tree_util.tree_map(jnp.zeros_like, params)
        elif self.config.initial_guess == "previous":
            x0 = previous_nat_grad
        elif self.config.initial_guess == "grad":
            x0 = loss_grads
            
        # Actually solve linear system S * nat_grad = loss_grads
        nat_grad, _ = jax.scipy.sparse.linalg.cg(fisher_matmul, loss_grads, x0=x0, maxiter=self.config.maxiter, M=preconditioner)
        nat_grad = pmean(nat_grad)

        # Compute norms for logging
        grad_norm = tree_norm(loss_grads)
        precon_grad_norm = tree_norm(nat_grad)

        # Apply norm constraint
        lr = self.learning_rate(step_count)
        norm_constraint_factor = self.config.max_update_norm / (precon_grad_norm * lr)
        lr *= jnp.minimum(1.0, norm_constraint_factor)
        update = jax.tree_util.tree_map(lambda x: lr * x, nat_grad)

        # Apply gradient update, and update optimizer state
        update, internal_opt_state = self.internal_optimizer.update(update, internal_opt_state, params)
        params = optax.apply_updates(params, update)

        new_opt_state = (internal_opt_state, nat_grad, preconditioner_state, step_count + 1)
        stats = dict(damping=damping, 
                     grad_norm = grad_norm,
                     precon_grad_norm=precon_grad_norm,
                     norm_constraint_factor=norm_constraint_factor,
                     aux=aux_metrics)
        return params, new_opt_state, new_func_state, stats