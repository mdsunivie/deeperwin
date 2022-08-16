# Modified from DeepMind repo https://github.com/deepmind/kfac-jax/2022
# Copyright DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import kfac_jax
import optax
from typing import Optional, Any, Callable, Tuple, Mapping, Iterator, Union
import jax
import jax.numpy as jnp
from deeperwin.configuration import OptimizerConfigKFAC, BFGSOptimizerConfig, OptimizationConfig, StandardOptimizerConfig
from deeperwin import curvature_tags_and_blocks
import haiku as hk
import re

OptimizerConfigType = Union[StandardOptimizerConfig, OptimizerConfigKFAC, BFGSOptimizerConfig]
OptaxState = Any

class OptaxWrapper:
    """Wrapper class for Optax optimizers to have the same interface as KFAC.
    """
    def __init__(
            self,
            value_and_grad_func: kfac_jax.optimizer.ValueAndGradFunc,
            value_func_has_aux: bool,
            value_func_has_state: bool,
            value_func_has_rng: bool,
            optax_optimizer: optax.GradientTransformation,
            multi_device: bool = False,
            pmap_axis_name="devices",
           batch_process_func: Optional[Callable[[Any], Any]] = lambda x: x,
    ):
        """Initializes the Optax wrapper.

        Args:
          value_and_grad_func: Python callable. The function should return the value
            of the loss to be optimized and its gradients. If the argument
            `value_func_has_aux` is `False` then the interface should be:
              loss, loss_grads = value_and_grad_func(params, batch)
            If `value_func_has_aux` is `True` then the interface should be:
              (loss, aux), loss_grads = value_and_grad_func(params, batch)
          value_func_has_aux: Boolean. Specifies whether the provided callable
            `value_and_grad_func` returns the loss value only, or also some
            auxiliary data. (Default: `False`)
          value_func_has_state: Boolean. Specifies whether the provided callable
            `value_and_grad_func` has a persistent state that is inputted and it
            also outputs an update version of it. (Default: `False`)
          value_func_has_rng: Boolean. Specifies whether the provided callable
            `value_and_grad_func` additionally takes as input an rng key. (Default:
            `False`)
          optax_optimizer: The optax optimizer to be wrapped.
          batch_process_func: Callable. A function which to be called on each batch
            before feeding to the KFAC on device. This could be useful for specific
            device input optimizations. (Default: `lambda x: x`)
        """
        self._value_and_grad_func = value_and_grad_func
        self._value_func_has_aux = value_func_has_aux
        self._value_func_has_state = value_func_has_state
        self._value_func_has_rng = value_func_has_rng
        self._optax_optimizer = optax_optimizer
        self._batch_process_func = batch_process_func or (lambda x: x)
        self._multi_device = multi_device
        self._pmap_axis_name = pmap_axis_name
        if self._multi_device:
            self._jit_step = jax.pmap(self._step, axis_name=self._pmap_axis_name, donate_argnums=[0, 1, 2, 3, 5])
        else:
            self._jit_step = jax.jit(self._step)

    def init(
            self,
            params: kfac_jax.utils.Params,
            rng: jnp.ndarray,
            batch: kfac_jax.utils.Batch,
            func_state: Optional[kfac_jax.utils.FuncState] = None
    ) -> OptaxState:
        """Initializes the optimizer and returns the appropriate optimizer state."""
        del rng, batch, func_state
        if self._multi_device:
            return jax.pmap(self._optax_optimizer.init)(params)
        else:
            return self._optax_optimizer.init(params)

    def _step(
            self,
            params: kfac_jax.utils.Params,
            state: OptaxState,
            rng,
            batch: kfac_jax.utils.Batch,
            func_state: Optional[kfac_jax.utils.FuncState] = None,
    ) -> kfac_jax.optimizer.FuncOutputs:
        """A single step of optax."""
        batch = self._batch_process_func(batch)
        func_args = kfac_jax.optimizer.make_func_args(
            params, func_state, rng, batch,
            has_state=self._value_func_has_state,
            has_rng=self._value_func_has_rng
        )
        out, grads = self._value_and_grad_func(*func_args)

        if not self._value_func_has_aux and not self._value_func_has_state:
            loss, new_func_state, aux = out, None, {}
        else:
            loss, other = out
            if self._value_func_has_aux and self._value_func_has_state:
                new_func_state, aux = other
            elif self._value_func_has_aux:
                new_func_state, aux = None, other
            else:
                new_func_state, aux = other, {}
        stats = dict(loss=loss, aux=aux)
        if self._multi_device:
            stats, grads = jax.lax.pmean((stats, grads), axis_name=self._pmap_axis_name)
        # Compute and apply updates via our optimizer.
        updates, new_state = self._optax_optimizer.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)

        # Add batch size
        batch_size = jax.tree_leaves(batch)[0].shape[0]
        stats["batch_size"] = batch_size * jax.device_count()

        if self._value_func_has_state:
            return new_params, new_state, new_func_state, stats
        else:
            return new_params, new_state, stats

    def step(
            self,
            params: kfac_jax.utils.Params,
            state: OptaxState,
            rng: jnp.ndarray,
            batch: kfac_jax.utils.Batch,
            func_state: Optional[kfac_jax.utils.FuncState] = None,
    ) -> Union[Tuple[kfac_jax.utils.Params, Any, kfac_jax.utils.FuncState,
                     Mapping[str, jnp.ndarray]],
               Tuple[kfac_jax.utils.Params, Any,
                     Mapping[str, jnp.ndarray]]]:
        """A step with similar interface to KFAC."""
        result = self._jit_step(
            params=params,
            state=state,
            rng=rng,
            batch=batch,
            func_state=func_state,
        )
        return result

def build_lr_schedule(base_lr, schedule_config):
    if schedule_config.name == "inverse":
        return lambda t: base_lr / (1 + t / schedule_config.decay_time)
    elif schedule_config.name == "fixed":
        return lambda t: base_lr
    else:
        raise ValueError(f"Unsupported config-value for optimization.schedule.name: {schedule_config.name}")

def build_optax_optimizer(config: StandardOptimizerConfig):
    lr_schedule = build_lr_schedule(config.learning_rate, config.lr_schedule)
    if config.name in ['adam', 'sgd', 'rmsprop']:
        optimizer = getattr(optax, config.name)(lr_schedule)
    else:
        raise ValueError(f"Unknown optimizer: {config.name}")

    # Scale gradients for selected modules
    if hasattr(config, "scaled_modules") and config.scaled_modules:
        regex = re.compile("(" + "|".join(config.scaled_modules) + ")")
        def leaf_filter_func(module_name, name, v):
            return len(regex.findall(f"{module_name}/{name}")) > 0
        optimizer = optax.chain(
            optimizer,
            optax.masked(optax.scale(config.scale_lr), lambda params: hk.data_structures.map(leaf_filter_func, params))
        )
    return optimizer


def build_optimizer(value_and_grad_func,
                    opt_config: OptimizerConfigType,
                    value_func_has_aux=False,
                    value_func_has_state=False):
    if opt_config.name in ['kfac', 'kfac_adam']:
        schedule = build_lr_schedule(opt_config.learning_rate, opt_config.lr_schedule)
        internal_optimizer = build_optax_optimizer(opt_config.internal_optimizer)
        return kfac_jax.Optimizer(value_and_grad_func,
                                  l2_reg=0.0,
                                  value_func_has_aux=value_func_has_aux,
                                  value_func_has_state=value_func_has_state,
                                  value_func_has_rng=False,
                                  multi_device=True,
                                  pmap_axis_name="devices",
                                  momentum_schedule = lambda t: opt_config.momentum,
                                  damping_schedule = build_lr_schedule(opt_config.damping, opt_config.lr_schedule),
                                  internal_optimizer=internal_optimizer,
                                  learning_rate_schedule=schedule,
                                  inverse_update_period=opt_config.update_inverse_period,
                                  num_burnin_steps=opt_config.n_burn_in,
                                  register_only_generic=opt_config.register_generic,
                                  norm_constraint=opt_config.norm_constraint,
                                  estimation_mode=opt_config.estimation_mode,
                                  min_damping=opt_config.min_damping,
                                  curvature_ema=opt_config.curvature_ema,
                                  auto_register_kwargs=dict(
                                        graph_patterns=curvature_tags_and_blocks.GRAPH_PATTERNS),
                                  include_norms_in_stats=True
                                  )
    elif opt_config.name in ['adam', 'sgd', 'rmsprop']:
        return OptaxWrapper(value_and_grad_func,
                            value_func_has_aux=value_func_has_aux,
                            value_func_has_state=value_func_has_state,
                            value_func_has_rng=False,
                            optax_optimizer=build_optax_optimizer(opt_config),
                            multi_device=True,
                            pmap_axis_name="devices")
    elif opt_config.name == 'slbfgs':
        raise NotImplementedError("BFGS currently not yet implemented")
    else:
        raise ValueError(f"Unknown optimizer: {opt_config.name}")


