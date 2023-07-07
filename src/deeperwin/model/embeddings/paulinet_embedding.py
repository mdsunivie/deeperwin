"""
File containing PauliNet-inspired embedding Layer.
"""

import jax.numpy as jnp
import numpy as np
import haiku as hk
from deeperwin.configuration import (
    EmbeddingConfigDeepErwin1,
    MLPConfig,
)
from deeperwin.model.definitions import *
from deeperwin.model.mlp import MLP


class PauliNetEmbedding(hk.Module):
    def __init__(self, config: EmbeddingConfigDeepErwin1, mlp_config: MLPConfig, name=None):
        super().__init__(name=name)
        self.config = config
        self.mlp_config = mlp_config

    def _split_into_same_diff(self, features_el_el: jnp.DeviceArray, n_up: int):
        n_el = features_el_el.shape[-3] # [batch x n_el x n_el x features]
        batch_dims = features_el_el.shape[:-3]
        n_dn = n_el - n_up
        h_uu = features_el_el[..., : n_up, : n_up, :].reshape(batch_dims + (n_up * n_up, -1))
        h_ud = features_el_el[..., : n_up, n_up :, :].reshape(batch_dims + (n_up * n_dn, -1))
        h_du = features_el_el[..., n_up :, : n_up, :].reshape(batch_dims + (n_dn * n_up, -1))
        h_dd = features_el_el[..., n_up :, n_up :, :].reshape(batch_dims + (n_dn * n_dn, -1))
        return [jnp.concatenate([h_uu, h_dd], axis=-2), jnp.concatenate([h_ud, h_du], axis=-2)]

    def __call__(self, features: InputFeatures, n_up: int):
        h_el_el = features.el_el
        h_el = features.el
        h_ion = features.ion
        h_el_ion = features.el_ion

        h_ion_mapped = MLP([self.config.embedding_dim], self.mlp_config, name="h_ion_map")(h_ion)

        f_pairs_same, f_pairs_diff = self._split_into_same_diff(h_el_el)

        n_el = h_el_el.shape[-3] # [batch x n_el x n_el x features]
        n_dn = n_el - n_up
        batch_dims = h_el_el.shape[:-3]
        for n in range(self.config.n_iterations):
            h_same = MLP(self.config.n_hidden_h + [self.config.embedding_dim], self.mlp_config, residual=False, name=f"h_same_{n}")(h_el)
            h_diff = MLP(self.config.n_hidden_h + [self.config.embedding_dim], self.mlp_config, residual=False, name=f"h_diff_{n}")(h_el)

            h_u_u = h_same[..., jnp.newaxis, :n_up, :]
            h_d_d = h_same[..., jnp.newaxis, n_up:, :]
            h_u_d = h_diff[..., jnp.newaxis, n_up:, :]
            h_d_u = h_diff[..., jnp.newaxis, :n_up, :]

            if n == 0 or not self.config.deep_w:
                w_same = MLP(self.config.n_hidden_w + [self.config.embedding_dim], self.mlp_config, residual=False,
                             linear_out=self.config.use_linear_layer_w,
                             name=f"w_same_{n}")(f_pairs_same)
                w_diff = MLP(self.config.n_hidden_w + [self.config.embedding_dim], self.mlp_config, residual=False,
                             linear_out=self.config.use_linear_layer_w,
                             name=f"w_diff_{n}")(f_pairs_diff)
            else:
                w_same = MLP(self.config.n_hidden_w + [self.config.embedding_dim], self.mlp_config, residual=False,
                             linear_out=self.config.use_linear_layer_w,
                             name=f"w_same_{n}")(w_same)
                w_diff = MLP(self.config.n_hidden_w + [self.config.embedding_dim], self.mlp_config, residual=False,
                             linear_out=self.config.use_linear_layer_w,
                             name=f"w_diff_{n}")(w_diff)

            w_u_u = w_same[..., :n_up * n_up, :].reshape(batch_dims + (n_up, n_up, -1))
            w_u_d = w_diff[..., :n_up * n_dn, :].reshape(batch_dims + (n_up, n_dn, -1))
            w_d_u = w_diff[..., n_up * n_dn:, :].reshape(batch_dims + (n_dn, n_up, -1))
            w_d_d = w_same[..., n_up * n_up:, :].reshape(batch_dims + (n_dn, n_dn, -1))

            if n == 0 or not self.config.deep_w:
                w_el_ions = MLP(self.config.n_hidden_w + [self.config.embedding_dim], self.mlp_config, residual=False, linear_out=self.config.use_linear_layer_w,
                                name=f"w_ion_{n}")(h_el_ion)

            else:
                w_el_ions = MLP(self.config.n_hidden_w + [self.config.embedding_dim], self.mlp_config, residual=False, linear_out=self.config.use_linear_layer_w,
                                name=f"w_ion_{n}")(w_el_ions)

            h_el_el = jnp.concatenate([
                jnp.concatenate([w_u_u * h_u_u, w_u_d * h_u_d], axis=-2),
                jnp.concatenate([w_d_u * h_d_u, w_d_d * h_d_d], axis=-2)
            ], axis=-3)
            h_el_ion_feat = w_el_ions * h_ion_mapped[..., None, :, :]

            h_el = jnp.sum(h_el_el, axis=-2) + jnp.sum(h_el_ion_feat, axis=-2)
            h_el = MLP(self.config.n_hidden_g + [self.config.embedding_dim], self.mlp_config, residual=False, name=f"g_{n}")(h_el)

        return Embeddings(h_el, h_ion, h_el_el, h_el_ion_feat)


    def apply_backflow_shift(self, diff_dist: DiffAndDistances, embeddings: Embeddings):
        shift_towards_electrons = self._calc_shift(embeddings.el, embeddings.el_el,
                                                   diff_dist.diff_el_el, diff_dist.dist_el_el, name="el")
        shift_towards_ions = self._calc_shift(embeddings.el, embeddings.el_ion, diff_dist.diff_el_ion,
                                              diff_dist.dist_el_ion, name="ion")

        # TODO: replace 1/Z scaling with an MLP of the ion embedding
        decay_lengthscale = hk.get_parameter("bf_shift_decay_scale", [1], init=jnp.ones)
        decay_lengthscale = decay_lengthscale / np.array(self.physical_config.Z)
        decay_factor = jnp.prod(jnp.tanh((diff_dist.dist_el_ion / decay_lengthscale) ** 2), axis=-1)
        shift = (shift_towards_electrons + shift_towards_ions) * decay_factor[..., jnp.newaxis]

        diff_el_el = diff_dist.diff_el_el + (shift[..., :, None, :] - shift[..., None, :, :])
        dist_el_el = jnp.linalg.norm(diff_el_el, axis=-1)
        diff_el_ion = diff_dist.diff_el_ion + shift[..., :, None, :]
        dist_el_ion = jnp.linalg.norm(diff_el_ion, axis=-1)
        return DiffAndDistances(diff_el_el, dist_el_el, diff_el_ion, dist_el_ion)


    def _calc_shift(self, x, pair_embedding, diff, dist, name="el"):
        n_particles = diff.shape[-2]
        x_tiled = jnp.tile(jnp.expand_dims(x, axis=-2), (n_particles, 1))
        features = jnp.concatenate([x_tiled, pair_embedding], axis=-1)
        shift = MLP(
            self.config.n_hidden_bf_shift + [1],
            self.mlp_config,
            output_bias=False,
            linear_out=True,
            name=f"shift_{name}",
        )(features)
        shift_weights = shift / (1 + dist[..., jnp.newaxis] ** 3)
        return jnp.sum(shift_weights * diff, axis=-2)