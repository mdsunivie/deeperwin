"""
File containing a regular Multi-Layer Perceptron (MLP) implemented in haiku.
"""


from typing import Iterable, Optional, Callable
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
from deeperwin.configuration import MLPConfig

def residual_update(update, old=None):
    if (old is None) or (old.shape != update.shape):
        return update
    return (old + update) / np.sqrt(2.0)

class MLP(hk.Module):
    def __init__(
        self,
        output_sizes: Iterable[int],
        config: MLPConfig = None,
        output_bias: bool = True,
        ln_aft_act: bool = None,
        ln_bef_act: bool = None,
        linear_out: bool = None,
        residual=None,
        use_bias=True,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        config = config or MLPConfig()
        self.output_sizes = output_sizes
        self.use_bias = use_bias
        self.output_bias = output_bias if self.use_bias else False
        self.ln_aft_act = ln_aft_act if (ln_aft_act is not None) else False
        self.ln_bef_act = ln_bef_act if (ln_bef_act is not None) else (config.use_layer_norm)
        self.residual = residual if (residual is not None) else (config.use_residual)
        self.linear_out = linear_out
        self.activation = get_activation(config.activation)
        self.init_w = hk.initializers.VarianceScaling(1.0, config.init_weights_scale, config.init_weights_distribution)
        self.init_b = hk.initializers.TruncatedNormal(config.init_bias_scale)

    def __call__(self, x):
        for i, output_size in enumerate(self.output_sizes):
            is_output_layer = i == (len(self.output_sizes) - 1)

            # LayerNorm
            if self.ln_bef_act:
                y = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            else:
                y = x

            # Linear
            if is_output_layer:
                y = hk.Linear(output_size, self.output_bias, self.init_w, self.init_b, f"linear_{i}")(y)
            else:
                y = hk.Linear(output_size, self.use_bias, self.init_w, self.init_b, f"linear_{i}")(y)

            # Activation
            if not (is_output_layer and self.linear_out):
                y = self.activation(y)
            if self.ln_aft_act: # TODO: eventually remove this option
                y = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(y)

            # Residual
            x = residual_update(y, x) if self.residual else y
        return x


def get_activation(activation: str) -> Callable:
    if isinstance(activation, Callable):
        return activation
    else:
        return dict(tanh=jnp.tanh,
                    silu=jax.nn.silu,
                    elu=jax.nn.elu,
                    relu=jax.nn.relu,
                    gelu=jax.nn.gelu)[activation]


#TODO: Move this function somewhere else, or rename the file from MLP to model_utils? building_blocks?
def _get_normalization_factors(neighbor_normalization, n_el, n_ion):
    if neighbor_normalization == "sum":
        return 1.0, 1.0
    if neighbor_normalization == 'sqrt':
        return 1.0 / np.sqrt(n_el - 1), 1.0 / np.sqrt(n_ion)
    if neighbor_normalization == 'mean':
        return 1.0 / n_el, 1.0 / n_ion

def symmetrize(f, tmp_axis=-2):
    """
    Take an arbitrary function f and return a new function g(x) = f(x) + f(-x).

    For any continous f, the new function g is now symmetric, i.e. g(x) = g(-x)
    """
    def symm_f(x):
        x = jnp.stack([x, -x], axis=tmp_axis)
        y = f(x)
        return jnp.sum(y, axis=tmp_axis)
    return symm_f

def antisymmetrize(f, tmp_axis=-2):
    """
    Take an arbitrary function f and return a new function g(x) = f(x) - f(-x).

    For any continous f, the new function g is now anti-symmetric, i.e. g(x) = -g(-x)
    """
    def asymm_f(x):
        x = jnp.stack([x, -x], axis=tmp_axis)
        y = f(x)
        y = jnp.moveaxis(y, tmp_axis, -1)
        return y[..., 0] - y[..., 1]
    return asymm_f


def get_rbf_features(dist, n_features, r_max=5.0):
    """
    Computes radial basis features based on Gaussians with different means from pairwise distances. This can be interpreted as a special type of "one-hot-encoding" for the distance between two particles.

    Args:
        dist (array): Pairwise particle distances

    Returns:
        array: Pairwise radial basis features

    """
    q = jnp.linspace(0, 1.0, n_features)
    mu = q ** 2 * r_max
    sigma = (1 / 7) * (1 + r_max * q)
    dist = dist[..., jnp.newaxis]  # add dimension for features
    return dist ** 2 * jnp.exp(-dist - ((dist - mu) / sigma) ** 2)


def get_gauss_env_features(dist, nb_features, max_scale=7.0):
    # if trainable_scale:
    #     sigma = hk.get_parameter("mu", [nb_features], init=jnp.ones)
    #     exp_env = dist[..., None] / sigma
    #     exp_env = register_scale_and_shift(exp_env, dist[..., None], scale=sigma, shift=None)

    sigma = jnp.linspace(1.0, max_scale, nb_features)
    exp_env = dist[..., None] / sigma
    gauss_env = jnp.exp(-(exp_env) ** 2)
    gauss_env = hk.Linear(nb_features, with_bias=False, name="gauss_env")(gauss_env)
    # gauss_env *= MLP(nb_layer * [nb_features],
    #                  linear_out=True,
    #                  name="gauss_env")(diff)

    return gauss_env