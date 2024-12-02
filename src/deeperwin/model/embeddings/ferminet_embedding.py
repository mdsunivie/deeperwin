"""
File containing FermiNet embedding layer, including necessary symmetrical/convolutional feature construction layers.

"""

import haiku as hk
import jax
import jax.numpy as jnp

from deeperwin.configuration import EmbeddingConfigFermiNet, MLPConfig
from deeperwin.model.definitions import Embeddings, InputFeatures
from deeperwin.model.mlp import MLP


class ScalarSymmetricProduct(hk.Module):
    """Product of linear mappings of input.

    output = (W1@x) + (W1@x) (W2@x) + (W1@x) (W2@x) (W3@x) ...
    """

    def __init__(self, correlation, symmetric=True, output_dim=None, name=None):
        self.correlation = correlation
        self.symmetric = symmetric
        self.output_dim = output_dim
        super().__init__(name=name)

    def __call__(self, x):
        output_dim = self.output_dim or x.shape[-1]

        if self.symmetric:
            z = hk.Linear(output_dim)(x)
            z = jnp.tile(z[..., None, :], (self.correlation, 1))
        else:
            z = hk.Linear(self.correlation * output_dim)(x)
            z = z.reshape(z.shape[:-1] + (self.correlation, output_dim))

        product = z[..., 0, :]
        output = product
        for n in range(1, self.correlation):
            product *= z[..., n, :]
            output += product
        return output


class SymmetricFeatures(hk.Module):
    def __init__(self, config: EmbeddingConfigFermiNet, mlp_config: MLPConfig, n_up, name="symm_features"):
        super().__init__(name=name)
        self.config = config
        self.mlp_config = mlp_config
        self.n_up = n_up

    def __call__(self, h_one, h_ion, h_el_el, h_el_ion):
        n_el = h_one.shape[-2]
        features = []
        if self.config.use_h_one:
            features.append(h_one)
        # TODO: Thoroughly test effect of neighbor_normalization (mean/sum/sqrt) here and in SchNet features, in particular for heavy atoms (Fe)

        # Average over all h_ones from 1-el-stream
        if self.config.use_average_h_one:
            avg_h_one_up = jnp.mean(h_one[..., : self.n_up, :], keepdims=True, axis=-2)
            avg_h_one_dn = jnp.mean(h_one[..., self.n_up :, :], keepdims=True, axis=-2)
            if self.config.use_h_one_same_diff:
                avg_h_one_same = jnp.concatenate(
                    [jnp.tile(avg_h_one_up, [self.n_up, 1]), jnp.tile(avg_h_one_dn, [n_el - self.n_up, 1])], axis=-2
                )
                avg_h_one_diff = jnp.concatenate(
                    [jnp.tile(avg_h_one_dn, [self.n_up, 1]), jnp.tile(avg_h_one_up, [n_el - self.n_up, 1])], axis=-2
                )
                features += [avg_h_one_same, avg_h_one_diff]
            else:
                features += [jnp.tile(avg_h_one_up, [n_el, 1]), jnp.tile(avg_h_one_dn, [n_el, 1])]

        # Average over 2-el-stream
        if self.config.use_average_h_two:
            assert (
                not self.config.use_h_two_same_diff
            ), "Averaging over 2-el-stream only implemented for use_h_two_same_diff==False"
            f_pairs_with_up = jnp.mean(h_el_el[..., : self.n_up, :], axis=-2)
            f_pairs_with_dn = jnp.mean(h_el_el[..., self.n_up :, :], axis=-2)
            features += [f_pairs_with_up, f_pairs_with_dn]

        # Average of el-ion-stream
        if self.config.use_el_ion_stream and not self.config.use_schnet_features:
            features.append(jnp.mean(h_el_ion, axis=-2))

        if self.config.use_schnet_features:
            el_el, el_ion, el_el_mult = ConvolutionalFeatures(self.config, self.mlp_config, self.n_up)(
                h_one, h_ion, h_el_el, h_el_ion
            )
            features += [el_el, el_ion]
        else:
            el_el_mult = None
        return jnp.concatenate(features, axis=-1), el_el_mult


class ConvolutionalFeatures(hk.Module):
    def __init__(self, config: EmbeddingConfigFermiNet, mlp_config: MLPConfig, n_up, name=None):
        super().__init__(name=name)
        self.config = config
        self.mlp_config = mlp_config
        self.n_up = n_up

        self.aggregation = dict(sum=jnp.sum, mean=jnp.mean)[self.config.schnet_aggregation]

    def __call__(self, h_one, h_ion, h_el_el, h_el_ion):
        batch_dims = h_one.shape[:-2]
        n_el = h_one.shape[-2]
        n_up, n_dn = self.n_up, n_el - self.n_up

        if (not self.config.use_w_mapping) and (not self.config.use_h_two_same_diff):
            # Simple case where we do not differentiate between same and different at all (not in input streams and not in the mappings)
            h_mapped = MLP(
                [h_el_el.shape[-1]], self.mlp_config, output_bias=self.config.use_schnet_bias_feat, name="h_map"
            )(h_one)
            embeddings_el_el = jnp.sum(h_el_el * h_mapped[..., None, :, :], axis=-2)
        else:
            if self.config.use_h_two_same_diff:
                w_same, w_diff = h_el_el
            else:
                w_same = jnp.concatenate(
                    [
                        h_el_el[..., :n_up, :n_up, :].reshape(batch_dims + (n_up * n_up, -1)),
                        h_el_el[..., n_up:, n_up:, :].reshape(batch_dims + (n_dn * n_dn, -1)),
                    ],
                    axis=-2,
                )
                w_diff = jnp.concatenate(
                    [
                        h_el_el[..., :n_up, n_up:, :].reshape(batch_dims + (n_up * n_dn, -1)),
                        h_el_el[..., n_up:, :n_up, :].reshape(batch_dims + (n_dn * n_up, -1)),
                    ],
                    axis=-2,
                )

            if self.config.use_w_mapping:
                w_same = MLP(
                    [self.config.emb_dim],
                    self.mlp_config,
                    output_bias=self.config.use_schnet_bias_feat,
                    linear_out=self.config.use_linear_out,
                    name="w_same",
                )(w_same)
                w_diff = MLP(
                    [self.config.emb_dim],
                    self.mlp_config,
                    output_bias=self.config.use_schnet_bias_feat,
                    linear_out=self.config.use_linear_out,
                    name="w_diff",
                )(w_diff)

            w_uu = w_same[..., : n_up * n_up, :].reshape(batch_dims + (n_up, n_up, -1))
            w_ud = w_diff[..., : n_up * n_dn, :].reshape(batch_dims + (n_up, n_dn, -1))
            w_du = w_diff[..., n_up * n_dn :, :].reshape(batch_dims + (n_dn, n_up, -1))
            w_dd = w_same[..., n_up * n_up :, :].reshape(batch_dims + (n_dn, n_dn, -1))
            h_mapped = MLP(
                [w_uu.shape[-1]], self.mlp_config, output_bias=self.config.use_schnet_bias_feat, name="h_map"
            )(h_one)
            h_u = h_mapped[..., None, :n_up, :]
            h_d = h_mapped[..., None, n_up:, :]

            mult_up_up = w_uu * h_u
            mult_up_dn = w_ud * h_d

            mult_dn_up = w_du * h_u
            mult_dn_dn = w_dd * h_d
            emb_up = self.aggregation(mult_up_up, axis=-2) + self.aggregation(mult_up_dn, axis=-2)
            emb_dn = self.aggregation(mult_dn_up, axis=-2) + self.aggregation(mult_dn_dn, axis=-2)
            embeddings_el_el = jnp.concatenate([emb_up, emb_dn], axis=-2)

        if self.config.use_el_ion_stream:
            h_ion_mapped = MLP(
                [h_el_ion.shape[-1]], self.mlp_config, output_bias=self.config.use_schnet_bias_feat, name="h_ion_map"
            )(h_ion)
            embeddings_el_ions = self.aggregation(h_el_ion * h_ion_mapped[..., None, :, :], axis=-2)
            return (
                embeddings_el_el,
                embeddings_el_ions,
                jnp.concatenate(
                    [
                        jnp.concatenate([mult_up_up, mult_up_dn], axis=-2),
                        jnp.concatenate([mult_dn_up, mult_dn_dn], axis=-2),
                    ],
                    axis=-3,
                ),
            )
        else:
            return embeddings_el_el


class FermiNetEmbedding(hk.Module):
    def __init__(self, config: EmbeddingConfigFermiNet, mlp_config: MLPConfig, name=None):
        super().__init__(name=name)
        self.config = config
        self.mlp_config = mlp_config

    def _split_into_same_diff(self, features_el_el: jax.Array, n_up: int):
        n_el = features_el_el.shape[-3]  # [batch x n_el x n_el x features]
        batch_dims = features_el_el.shape[:-3]
        n_dn = n_el - n_up
        h_uu = features_el_el[..., :n_up, :n_up, :].reshape(batch_dims + (n_up * n_up, -1))
        h_ud = features_el_el[..., :n_up, n_up:, :].reshape(batch_dims + (n_up * n_dn, -1))
        h_du = features_el_el[..., n_up:, :n_up, :].reshape(batch_dims + (n_up * n_dn, -1))
        h_dd = features_el_el[..., n_up:, n_up:, :].reshape(batch_dims + (n_dn * n_dn, -1))
        return [jnp.concatenate([h_uu, h_dd], axis=-2), jnp.concatenate([h_ud, h_du], axis=-2)]

    def __call__(self, features: InputFeatures, n_up: int):
        h_el = features.el
        h_ion = features.ion
        h_el_ion = features.el_ion

        if self.config.use_h_two_same_diff:
            h_el_el = self._split_into_same_diff(features.el_el, n_up)
        else:
            h_el_el = features.el_el

        for i in range(self.config.n_iterations):
            h_el, h_el_el_schnet = SymmetricFeatures(self.config, self.mlp_config, n_up, name=f"symm_features_{i}")(
                h_el, h_ion, h_el_el, h_el_ion
            )
            if self.config.use_deep_schnet_feat:
                h_el_el = self._split_into_same_diff(h_el_el_schnet, n_up)

            if self.config.h_one_correlation > 0:
                if self.config.downmap_during_product:
                    prod_output_dim = self.config.n_hidden_one_el[i]
                else:
                    prod_output_dim = h_el.shape[-1]
                h_el = ScalarSymmetricProduct(
                    self.config.h_one_correlation, self.config.use_symmetric_product, prod_output_dim
                )(h_el)

            if self.config.use_h_one_mlp:
                h_el = MLP(
                    [self.config.n_hidden_one_el[i]],
                    self.mlp_config,
                    ln_bef_act=self.config.use_ln_bef_act,
                    ln_aft_act=self.config.use_ln_aft_act,
                    residual=True,
                    name=f"h_el_{i}",
                )(h_el)

            if i == (self.config.n_iterations - 1):
                # We have one more 1-electron layer than 2-electron layers: Can skip last 2-particle layers
                break

            if self.config.use_h_two_same_diff:
                h_el_el[0] = MLP([self.config.n_hidden_two_el[i]], self.mlp_config, residual=True, name=f"h_same_{i}")(
                    h_el_el[0]
                )
                h_el_el[1] = MLP([self.config.n_hidden_two_el[i]], self.mlp_config, residual=True, name=f"h_diff_{i}")(
                    h_el_el[1]
                )
            else:
                h_el_el = MLP([self.config.n_hidden_two_el[i]], self.mlp_config, residual=True, name=f"h_el_el_{i}")(
                    h_el_el
                )

            if self.config.use_el_ion_stream:
                h_el_ion = MLP([self.config.n_hidden_two_el[i]], self.mlp_config, residual=True, name=f"h_el_ion_{i}")(
                    h_el_ion
                )

        if not self.config.use_el_ion_stream:
            h_el_ion = h_el[..., jnp.newaxis, :]

        return Embeddings(h_el, h_ion, h_el_el, h_el_ion)
