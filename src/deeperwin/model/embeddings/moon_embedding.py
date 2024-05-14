from typing import Iterable
import jax
import jax.numpy as jnp
import haiku as hk
from deeperwin.configuration import EmbeddingConfigMoon
from deeperwin.model.definitions import DiffAndDistances, Embeddings, InputFeatures
from deeperwin.model.mlp import get_activation
from deeperwin.utils.utils import split_axis, residual_update

def _split_into_same_diff(diff, n_up, n_dn):
    batch_dims = diff.shape[:-3]
    up_up = diff[..., :n_up, :n_up, :]
    up_dn = diff[..., :n_up, n_up:, :]
    dn_up = diff[..., n_up:, :n_up, :]
    dn_dn = diff[..., n_up:, n_up:, :]
    same = jnp.concatenate(
        [up_up.reshape((*batch_dims, n_up * n_up, -1)), dn_dn.reshape((*batch_dims, n_dn * n_dn, -1))], axis=-2
    )
    diff = jnp.concatenate(
        [up_dn.reshape((*batch_dims, n_up * n_dn, -1)), dn_up.reshape((*batch_dims, n_dn * n_up, -1))], axis=-2
    )
    return same, diff


class SpatialFilter(hk.Module):
    def __init__(
        self,
        n_hidden: Iterable[int],
        activation: str = "silu",
        use_envelopes: bool = True,
        envelope_power: int = 2,
        initial_scale: float = 10.0,
        name=None,
    ):
        super().__init__(name=name)
        self.mlp = hk.nets.MLP(n_hidden, activate_final=False, activation=get_activation(activation))
        self.use_envelopes = use_envelopes
        self.envelope_power = envelope_power
        self.scale_init = self._scale_initializer(initial_scale)
        self.n_hidden = n_hidden

    def _scale_initializer(self, initial_scale):
        if isinstance(initial_scale, Iterable):
            assert len(initial_scale) == 2
            scale_min, scale_max = initial_scale
        else:
            scale_min, scale_max = 0.8 * initial_scale, 1.2 * initial_scale

        def init(shape, dtype=None):
            u = jax.random.uniform(
                hk.next_rng_key(), minval=jnp.log(scale_min), maxval=jnp.log(scale_max), shape=shape, dtype=dtype
            )
            scale = jnp.exp(-u)
            param = jnp.log(jnp.exp(scale) - 1.0)  # inverse of softplus
            return param

        return init

    def __call__(self, diff, dist):
        y = self.mlp(diff)
        hidden_dim, output_dim = self.n_hidden[-2:]

        scale = hk.get_parameter("scale", (hidden_dim,), init=self.scale_init)
        scale = jax.nn.softplus(scale)
        exp = dist[..., None] * scale
        # The following registration is not fully correct, since it ignores the softmax
        # exp = register_scale_and_shift(exp, dist, scale=scale, shift=None)
        envelope = jnp.exp(-(exp**self.envelope_power))
        envelope = hk.Linear(output_dim, False, name="lin_env")(envelope)
        return y * envelope


class MoonEmbedding(hk.Module):
    def __init__(self, config: EmbeddingConfigMoon, name=None):
        super().__init__(name=name)
        self.config = config
        self.activation = get_activation(config.activation)

    def spatial_filter(self, diff, dist, name):
        return SpatialFilter(
            n_hidden=self.config.n_hidden_spatial,
            activation=self.config.activation,
            envelope_power=self.config.envelope_power_spatial,
            initial_scale=self.config.initial_scale_spatial,
            name="beta_" + name,
        )(diff, dist)

    def output_filter(self, diff, dist, name):
        return SpatialFilter(
            n_hidden=self.config.n_hidden_spatial,
            activation=self.config.activation,
            envelope_power=self.config.envelope_power_output,
            initial_scale=self.config.initial_scale_output,
            name="beta_" + name,
        )(diff, dist)

    def __call__(self, diff_dist: DiffAndDistances, features: InputFeatures, n_up: int):
        n_el = diff_dist.dist_el_el.shape[-1]
        n_dn = n_el - n_up

        beta_el_el = self.spatial_filter(diff_dist.diff_el_el, diff_dist.dist_el_el, "el_el")
        beta_el_ion = self.spatial_filter(diff_dist.diff_el_ion, diff_dist.dist_el_ion, "el_ion")

        # 0) (Optional) Initialize ion embedding with messages from all other ions
        h_ion_0 = hk.Linear(self.config.el_ion_dim_collect, True, name="ion_emb")(features.ion)
        if self.config.use_ion_ion_features:
            beta_ion_ion = self.spatial_filter(diff_dist.diff_ion_ion, diff_dist.dist_ion_ion, "ion_ion")
            gamma_ion_ion = hk.Linear(self.config.el_ion_dim_collect, False, name="Gamma_ion_ion")(beta_ion_ion)
            h_ion_0 = jnp.sum(h_ion_0[..., None, :, :] * gamma_ion_ion, axis=-2)

        # 1) Initialize electron embedding with messages from all other electrons
        diffdist_same, diffdist_diff = _split_into_same_diff(features.el_el, n_up, n_dn)
        beta_same, beta_diff = _split_into_same_diff(beta_el_el, n_up, n_dn)

        gamma_same = hk.Linear(self.config.el_el_dim, False, name="Gamma_el_el_same")(beta_same)
        gamma_diff = hk.Linear(self.config.el_el_dim, False, name="Gamma_el_el_diff")(beta_diff)
        h_el_el_same = self.activation(hk.Linear(self.config.el_el_dim, name="el_el_same")(diffdist_same)) * gamma_same
        h_el_el_diff = self.activation(hk.Linear(self.config.el_el_dim, name="el_el_diff")(diffdist_diff)) * gamma_diff

        h_up_up = split_axis(h_el_el_same[..., : n_up * n_up, :], -2, [n_up, n_up])
        h_up_dn = split_axis(h_el_el_diff[..., : n_up * n_dn, :], -2, [n_up, n_dn])
        h_dn_up = split_axis(h_el_el_diff[..., n_up * n_dn :, :], -2, [n_dn, n_up])
        h_dn_dn = split_axis(h_el_el_same[..., n_up * n_up :, :], -2, [n_dn, n_dn])

        h_for_up = jnp.concatenate([jnp.sum(h_up_up, axis=-2), jnp.sum(h_up_dn, axis=-2)], axis=-1)
        h_for_dn = jnp.concatenate([jnp.sum(h_dn_dn, axis=-2), jnp.sum(h_dn_up, axis=-2)], axis=-1)
        h_for_both = jnp.concatenate([h_for_up, h_for_dn], axis=-2)
        h_el_0 = self.activation(hk.Linear(self.config.el_ion_dim_collect, name="el_0")(h_for_both))

        # 2) Generate el-ion features by combining electron-, ion-, and diff-features
        g_el_ion = hk.Linear(self.config.el_ion_dim_collect, False, name="g_el_ion")(features.el_ion)
        h_el_ion_0 = self.activation(h_el_0[..., :, None, :] + h_ion_0[..., None, :, :] + g_el_ion)

        # 3) Initialize ion-features as filtered contraction over el-ion
        gamma_el_ion = hk.Linear(self.config.el_ion_dim_collect, False, name="Gamma_el_ion")(beta_el_ion)
        msg = h_el_ion_0 * gamma_el_ion
        h_ion_up = jnp.sum(msg[..., :n_up, :, :], axis=-3)  # sum over electrons
        h_ion_dn = jnp.sum(msg[..., n_up:, :, :], axis=-3)  # sum over electrons

        # 4) Deep and wide MLP, processing ions, differentiating between same/diff spins contracted to the ion
        for i in range(self.config.n_iterations):
            h_ion_for_up = jnp.stack([h_ion_up, h_ion_dn], axis=-2)
            h_ion_for_dn = jnp.stack([h_ion_dn, h_ion_up], axis=-2)
            h_ion_for_both = jnp.concatenate([h_ion_for_up, h_ion_for_dn], axis=-1)
            if i != (self.config.n_iterations - 1):
                update = self.activation(hk.Linear(self.config.ion_dim, name=f"ion_mlp_{i}")(h_ion_for_both))
                h_ion_up = residual_update(update[..., 0, :], h_ion_up)
                h_ion_dn = residual_update(update[..., 1, :], h_ion_dn)

        # 5) "Diffuse" ion information back to electron-ion pairs
        # Residual from el-ion embedding
        h_el_ion = hk.Linear(self.config.output_dim, name="lin_out_elion")(h_el_ion_0)

        # Ion embedding
        h_ion_for_both = hk.Linear(self.config.output_dim, False, name="lin_out_ion")(h_ion_for_both)
        h_el_ion = h_el_ion.at[..., :n_up, :, :].add(h_ion_for_both[..., None, :, 0, :])
        h_el_ion = h_el_ion.at[..., n_up:, :, :].add(h_ion_for_both[..., None, :, 1, :])

        # Electron embedding
        h_el_ion += hk.Linear(self.config.output_dim, False, name="lin_out_el")(h_el_0[..., None, :])

        # Spatial filtering
        beta_out = self.output_filter(diff_dist.diff_el_ion, diff_dist.dist_el_ion, "el_ion_out")
        gamma_out = hk.Linear(self.config.output_dim, False, name="Gamma_el_ion_out")(beta_out)
        h_el_ion = self.activation(h_el_ion) * gamma_out
        h_el = jnp.sum(h_el_ion, axis=-2)  # sum over ions
        return Embeddings(h_el, None, None, h_el_ion)
