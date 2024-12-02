# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Curvature blocks for FermiNet."""

from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import kfac_jax
import numpy as np


Array = kfac_jax.utils.Array
Scalar = kfac_jax.utils.Scalar
Numeric = kfac_jax.utils.Numeric

vmap_matmul = jax.vmap(jnp.matmul, in_axes=(0, 0), out_axes=0)


repeated_dense_tag = kfac_jax.LayerTag("repeated_dense_tag", 1, 1)


def register_repeated_dense(y, x, w, b):
    if b is None:
        return repeated_dense_tag.bind(y, x, w)
    return repeated_dense_tag.bind(y, x, w, b)


class RepeatedDenseBlock(kfac_jax.DenseTwoKroneckerFactored):
    """Dense block that is repeatedly applied to multiple inputs (e.g. vmap)."""

    # @property
    def fixed_scale(self) -> Numeric:
        (x_shape,) = self.inputs_shapes
        return float(kfac_jax.utils.product(x_shape) // (x_shape[0] * x_shape[-1]))

    def update_curvature_matrix_estimate(
        self,
        state: kfac_jax.TwoKroneckerFactored.State,
        estimation_data: Mapping[str, Sequence[Array]],
        ema_old: Numeric,
        ema_new: Numeric,
        batch_size: int,
    ) -> kfac_jax.TwoKroneckerFactored.State:
        estimation_data = dict(**estimation_data)
        (x,) = estimation_data["inputs"]
        (dy,) = estimation_data["outputs_tangent"]
        assert x.shape[0] == batch_size
        estimation_data["inputs"] = (x.reshape([-1, x.shape[-1]]),)
        estimation_data["outputs_tangent"] = (dy.reshape([-1, dy.shape[-1]]),)
        batch_size = x.size // x.shape[-1]
        return super().update_curvature_matrix_estimate(state, estimation_data, ema_old, ema_new, batch_size)


def _dense(x: Array, params: Sequence[Array]) -> Array:
    """Example of a dense layer function."""
    w, *opt_b = params
    y = jnp.matmul(x, w)
    return y if not opt_b else y + opt_b[0]


def _dense_parameter_extractor(
    eqns: Sequence[jax.core.JaxprEqn],
) -> Mapping[str, Any]:
    """Extracts all parameters from the conv_general_dilated operator."""
    for eqn in eqns:
        if eqn.primitive.name == "dot_general":
            return dict(**eqn.params)
    assert False


# repeating a dense layer once
repeated_dense_patterns = []
_dense_func = _dense
_dense_func_no_bias = _dense
_example_args_x = np.zeros([11, 13])
_example_args_w = np.zeros([13, 7])
_example_args_b = np.zeros([7])
n_repeated_max = 4
for n_rep in range(1, n_repeated_max + 1):
    _dense_func = jax.vmap(_dense_func, in_axes=[0, [None, None]])
    _dense_func_no_bias = jax.vmap(_dense_func_no_bias, in_axes=[0, [None]])
    _example_args_x = np.tile(_example_args_x, [n_rep + 1] + [1] * _example_args_x.ndim)
    pattern_dense = kfac_jax.tag_graph_matcher.GraphPattern(
        name=f"repeated_dense{n_rep}_with_bias",
        tag_primitive=repeated_dense_tag,
        compute_func=_dense_func,
        parameters_extractor_func=_dense_parameter_extractor,
        example_args=[_example_args_x, [_example_args_w, _example_args_b]],
    )
    pattern_dense_no_bias = kfac_jax.tag_graph_matcher.GraphPattern(
        name=f"repeated_dense{n_rep}_no_bias",
        tag_primitive=repeated_dense_tag,
        compute_func=_dense_func_no_bias,
        parameters_extractor_func=_dense_parameter_extractor,
        example_args=[_example_args_x, [_example_args_w]],
    )
    repeated_dense_patterns.append(pattern_dense)
    repeated_dense_patterns.append(pattern_dense_no_bias)

GRAPH_PATTERNS = tuple(repeated_dense_patterns) + kfac_jax.tag_graph_matcher.DEFAULT_GRAPH_PATTERNS

kfac_jax.set_default_tag_to_block_ctor("repeated_dense_tag", RepeatedDenseBlock)
