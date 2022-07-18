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
from typing import Any, Mapping, Optional, Sequence, Union
import chex
import jax
import jax.numpy as jnp
import kfac_jax
import numpy as np


vmap_psd_inv_cholesky = jax.vmap(kfac_jax.utils.psd_inv_cholesky, (0, None), 0)
vmap_matmul = jax.vmap(jnp.matmul, in_axes=(0, 0), out_axes=0)


repeated_dense_tag = kfac_jax.LayerTag("repeated_dense_tag", 1, 1)


def register_repeated_dense(y, x, w, b):
    if b is None:
        return repeated_dense_tag.bind(y, x, w)
    return repeated_dense_tag.bind(y, x, w, b)


class RepeatedDenseBlock(kfac_jax.DenseTwoKroneckerFactored):
    """Dense block that is repeatedly applied to multiple inputs (e.g. vmap)."""

    @property
    def scale(self) -> Union[float, jnp.ndarray]:
        (x_shape,) = self.inputs_shapes
        return float(kfac_jax.utils.product(x_shape) // (x_shape[0] * x_shape[-1]))

    def update_curvature_matrix_estimate(
        self,
        state: kfac_jax.TwoKroneckerFactored.State,
        estimation_data: Mapping[str, Sequence[chex.Array]],
        ema_old: chex.Numeric,
        ema_new: chex.Numeric,
        batch_size: int,
        pmap_axis_name: Optional[str],
    ) -> kfac_jax.TwoKroneckerFactored.State:
        estimation_data = dict(**estimation_data)
        (x,) = estimation_data["inputs"]
        (dy,) = estimation_data["outputs_tangent"]
        assert x.shape[0] == batch_size
        estimation_data["inputs"] = (x.reshape([-1, x.shape[-1]]),)
        estimation_data["outputs_tangent"] = (dy.reshape([-1, dy.shape[-1]]),)
        batch_size = x.size // x.shape[-1]
        return super().update_curvature_matrix_estimate(state, estimation_data, ema_old, ema_new, batch_size, pmap_axis_name)


def _dense(x: chex.Array, params: Sequence[chex.Array]) -> chex.Array:
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
_repeated_dense1 = jax.vmap(_dense, in_axes=[0, [None, None]])
_repeated_dense2 = jax.vmap(_repeated_dense1, in_axes=[0, [None, None]])
_repeated_dense1_no_b = jax.vmap(_dense, in_axes=[0, [None]])
_repeated_dense2_no_b = jax.vmap(_repeated_dense1_no_b, in_axes=[0, [None]])

# Computation for repeated dense layer
repeated_dense1_with_bias_pattern = kfac_jax.tag_graph_matcher.GraphPattern(
    name="repeated_dense1_with_bias",
    tag_primitive=repeated_dense_tag,
    precedence=0,
    compute_func=_repeated_dense1,
    parameters_extractor_func=_dense_parameter_extractor,
    example_args=[np.zeros([9, 11, 13]), [np.zeros([13, 7]), np.zeros([7])]],
)

repeated_dense1_no_bias_pattern = kfac_jax.tag_graph_matcher.GraphPattern(
    name="repeated_dense1_no_bias",
    tag_primitive=repeated_dense_tag,
    precedence=0,
    compute_func=_repeated_dense1_no_b,
    parameters_extractor_func=_dense_parameter_extractor,
    example_args=[np.zeros([9, 11, 13]), [np.zeros([13, 7])]],
)

repeated_dense2_with_bias_pattern = kfac_jax.tag_graph_matcher.GraphPattern(
    name="repeated_dense2_with_bias",
    tag_primitive=repeated_dense_tag,
    precedence=0,
    compute_func=_repeated_dense2,
    parameters_extractor_func=_dense_parameter_extractor,
    example_args=[np.zeros([8, 9, 11, 13]), [np.zeros([13, 7]), np.zeros([7])]],
)

repeated_dense2_no_bias_pattern = kfac_jax.tag_graph_matcher.GraphPattern(
    name="repeated_dense2_no_bias",
    tag_primitive=repeated_dense_tag,
    precedence=0,
    compute_func=_repeated_dense2_no_b,
    parameters_extractor_func=_dense_parameter_extractor,
    example_args=[np.zeros([8, 9, 11, 13]), [np.zeros([13, 7])]],
)

GRAPH_PATTERNS = (
    repeated_dense1_with_bias_pattern,
    repeated_dense2_with_bias_pattern,
    repeated_dense1_no_bias_pattern,
    repeated_dense2_no_bias_pattern,
) + kfac_jax.tag_graph_matcher.DEFAULT_GRAPH_PATTERNS


kfac_jax.set_default_tag_to_block_ctor("repeated_dense_tag", RepeatedDenseBlock)
