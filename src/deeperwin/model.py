from typing import Iterable, Optional, Callable, Literal
import haiku.experimental
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import logging
from deeperwin.configuration import (
    ModelConfig,
    JastrowConfig,
    InputFeatureConfig,
    EnvelopeOrbitalsConfig,
    BaselineOrbitalsConfig,
    OrbitalsConfig,
    EmbeddingConfigDeepErwin1,
    EmbeddingConfigFermiNet,
    PhysicalConfig,
    CASSCFConfig,
    MLPConfig,
)
from deeperwin.orbitals import get_baseline_solution, get_envelope_exponents_from_atomic_orbitals, evaluate_molecular_orbitals
from deeperwin.local_features import build_local_rotation_matrices
from deeperwin.utils import get_distance_matrix, get_el_ion_distance_matrix
from collections import namedtuple
from kfac_jax import register_scale_and_shift

from jax import numpy as jnp

DiffAndDistances = namedtuple("DiffAndDistances", "diff_el_el, dist_el_el, diff_el_ion, dist_el_ion")
InputFeatures = namedtuple("InputFeatures", "el, ion, el_el, el_ion")
Embeddings = namedtuple("Embeddings", "el, ion, el_el, el_ion")

LOGGER = logging.getLogger("dpe")

def _w_init(mlp_config: MLPConfig):
    return hk.initializers.VarianceScaling(1.0, mlp_config.init_weights_scale, mlp_config.init_weights_distribution)

def _b_init(mlp_config: MLPConfig):
    return hk.initializers.TruncatedNormal(mlp_config.init_bias_scale)

class MLP(hk.Module):
    def __init__(
        self,
        output_sizes: Iterable[int],
        config: MLPConfig = None,
        output_bias: bool = True,
        linear_out: bool = False,
        residual=False,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        config = config or MLPConfig()
        self.output_sizes = output_sizes
        self.output_bias = output_bias
        self.linear_out = linear_out
        self.residual = residual
        self.activation = dict(tanh=jnp.tanh, silu=jax.nn.silu, elu=jax.nn.elu, relu=jax.nn.relu)[config.activation]
        self.init_w = hk.initializers.VarianceScaling(1.0, config.init_weights_scale, config.init_weights_distribution)
        self.init_b = hk.initializers.TruncatedNormal(config.init_bias_scale)

    def __call__(self, x):
        for i, output_size in enumerate(self.output_sizes):
            is_output_layer = i == (len(self.output_sizes) - 1)
            y = hk.Linear(output_size, self.output_bias or not is_output_layer, self.init_w, self.init_b, f"linear_{i}")(x)
            if not (is_output_layer and self.linear_out):
                y = self.activation(y)
            if self.residual and (x.shape == y.shape):
                x = (y + x) / np.sqrt(2.0)
            else:
                x = y
        return x


class PairwiseFeatures(hk.Module):
    def __init__(self, config: InputFeatureConfig, physical_config: PhysicalConfig, pair_type: Literal["el_el", "el_ion"], name=None):
        super().__init__(name=name)
        self.config = config
        self.physical_config = physical_config
        self.pair_type = pair_type

    def __call__(self, differences, dist):
        """
        Computes pairwise features based on particle distances.

        Args:
            differences (array): Pairwise particle differences (i.e. 3 coordinates per pair)
            dist (array): Pairwise particle distances
        Returns:
            array: Pairwise distance features
        """
        features = []
        if self.config.use_rbf_features:
            features_rbf = self._get_rbf_features(dist)
            features.append(features_rbf)
        if self.config.use_distance_features:
            eps = self.config.eps_dist_feat
            if len(self.config.distance_feature_powers) > 0:
                features_dist = jnp.stack(
                    [dist ** n if n > 0 else 1 / (dist ** (-n) + eps) for n in self.config.distance_feature_powers], axis=-1
                )
            features.append(features_dist)

        if self.pair_type == "el_el":
            if self.config.use_el_el_differences:
                features.append(differences)
            if self.config.use_el_el_spin:
                el_shape = differences.shape[:-2]  # [batch-dims x n_el]
                spins = np.ones(el_shape)
                spins[..., : self.phys_config.n_up] = 0
                spin_diff = np.abs(spins[..., None, :] - spins[..., :, None]) * 2 - 1
                features.append(spin_diff[..., None])
        if (self.pair_type == "el_ion") and self.config.use_el_ion_differences:
            features.append(differences)
        return jnp.concatenate(features, axis=-1)

    def _get_rbf_features(self, dist):
        """
        Computes radial basis features based on Gaussians with different means from pairwise distances. This can be interpreted as a special type of "one-hot-encoding" for the distance between two particles.

        Args:
            dist (array): Pairwise particle distances

        Returns:
            array: Pairwise radial basis features

        """
        r_rbf_max = 5.0
        q = jnp.linspace(0, 1.0, self.config.n_rbf_features)
        mu = q ** 2 * r_rbf_max
        sigma = (1 / 7) * (1 + r_rbf_max * q)
        dist = dist[..., jnp.newaxis]  # add dimension for features
        return dist ** 2 * jnp.exp(-dist - ((dist - mu) / sigma) ** 2)



class InputPreprocessor(hk.Module):
    def __init__(self, config: InputFeatureConfig, physical_config: PhysicalConfig, mlp_config: MLPConfig, name=None):
        super().__init__(name=name)
        self.config = config
        self.physical_config = physical_config
        self.mlp_config = mlp_config

    def __call__(self, r, R, Z, fixed_params=None):
        Z = jnp.array(Z, jnp.float32)
        if Z.ndim == 1: # no batch-dim for Z => tile across batch
            Z = jnp.tile(Z, r.shape[:-2] + (1,))
        diff_el_el, dist_el_el = get_distance_matrix(r)
        diff_el_ion, dist_el_ion = get_el_ion_distance_matrix(r, R)

        if self.config.use_local_coordinates:
            diff_el_ion = jnp.sum(fixed_params["local_rotations"] * diff_el_ion[..., None, :], axis=-1)

        features_el_el = PairwiseFeatures(self.config, self.physical_config, "el_el")(diff_el_el, dist_el_el)
        features_el_ion = PairwiseFeatures(self.config, self.physical_config, "el_ion")(diff_el_ion, dist_el_ion)

        features_el = []
        if self.config.n_one_el_features > 0:
            mlp = MLP(self.config.n_hidden_one_el_features + [self.config.n_one_el_features], self.mlp_config, name="one_el_features")
            features_el.append(jnp.sum(mlp(features_el_ion), axis=-2))
        if self.config.concatenate_el_ion_features:
            # concatenate all el-ion features into a long h_one feature
            features_el.append(jnp.reshape(features_el_ion, features_el_ion.shape[:-2] + (-1,)))
        if len(features_el) == 0:
            features_el.append(jnp.ones(r.shape[:-1] + (1,)))
        features_el = jnp.concatenate(features_el, axis=-1)

        # TODO: Build better ion-encoding, e.g. one-hot
        features_ion = Z[..., None]
        if self.config.n_ion_features:
            features_ion = MLP([self.config.n_ion_features], self.mlp_config, name="h_ion")(features_ion)
        diff_dist = DiffAndDistances(diff_el_el, dist_el_el, diff_el_ion, dist_el_ion)
        features = InputFeatures(features_el, features_ion, features_el_el, features_el_ion)
        return diff_dist, features


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

        # Average over all h_ones from 1-el-stream
        if self.config.use_average_h_one:
            g_one = [
                jnp.mean(h_one[..., : self.n_up, :], keepdims=True, axis=-2),
                jnp.mean(h_one[..., self.n_up :, :], keepdims=True, axis=-2),
            ]
            features += [jnp.tile(el, [n_el, 1]) for el in g_one]

        # Average over 2-el-stream
        if self.config.use_average_h_two:
            assert not self.config.use_h_two_same_diff, "Averaging over 2-el-stream only implemented for use_h_two_same_diff==False"
            f_pairs_with_up = jnp.mean(h_el_el[..., : self.n_up, :], axis=-2)
            f_pairs_with_dn = jnp.mean(h_el_el[..., self.n_up :, :], axis=-2)
            features += [f_pairs_with_up, f_pairs_with_dn]

        # Average of el-ion-stream
        if self.config.use_el_ion_stream and not self.config.use_schnet_features:
            features.append(jnp.mean(h_el_ion, axis=-2))

        if self.config.use_schnet_features:
            features += list(ConvolutionalFeatures(self.config, self.mlp_config, self.n_up)(h_one, h_ion, h_el_el, h_el_ion))

        return jnp.concatenate(features, axis=-1)


class ConvolutionalFeatures(hk.Module):
    def __init__(self, config: EmbeddingConfigFermiNet, mlp_config: MLPConfig, n_up, name=None):
        super().__init__(name=name)
        self.config = config
        self.mlp_config = mlp_config
        self.n_up = n_up

    def __call__(self, h_one, h_ion, h_el_el, h_el_ion):
        batch_dims = h_one.shape[:-2]
        n_el = h_one.shape[-2]
        n_up, n_dn = self.n_up, n_el - self.n_up

        if (not self.config.use_w_mapping) and (not self.config.use_h_two_same_diff):
            # Simple case where we do not differentiate between same and different at all (not in input streams and not in the mappings)
            h_mapped = MLP([h_el_el.shape[-1]], self.mlp_config, name="h_map")(h_one)
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
                w_same = MLP([self.config.emb_dim], self.mlp_config, linear_out=self.config.use_linear_out,
                             name="w_same")(w_same)
                w_diff = MLP([self.config.emb_dim], self.mlp_config, linear_out=self.config.use_linear_out,
                             name="w_diff")(w_diff)

            w_uu = w_same[..., : n_up * n_up, :].reshape(batch_dims + (n_up, n_up, -1))
            w_ud = w_diff[..., : n_up * n_dn, :].reshape(batch_dims + (n_up, n_dn, -1))
            w_du = w_diff[..., n_up * n_dn :, :].reshape(batch_dims + (n_dn, n_up, -1))
            w_dd = w_same[..., n_up * n_up :, :].reshape(batch_dims + (n_dn, n_dn, -1))
            h_mapped = MLP([w_uu.shape[-1]], self.mlp_config, name="h_map")(h_one)
            h_u = h_mapped[..., None, :n_up, :]
            h_d = h_mapped[..., None, n_up:, :]
            emb_up = jnp.sum(w_uu * h_u, axis=-2) + jnp.sum(w_ud * h_d, axis=-2)
            emb_dn = jnp.sum(w_du * h_u, axis=-2) + jnp.sum(w_dd * h_d, axis=-2)
            embeddings_el_el = jnp.concatenate([emb_up, emb_dn], axis=-2)

        if self.config.use_el_ion_stream:
            h_ion_mapped = MLP([h_el_ion.shape[-1]], self.mlp_config, name="h_ion_map")(h_ion)
            embeddings_el_ions = jnp.sum(h_el_ion * h_ion_mapped[..., None, :, :], axis=-2)
            return embeddings_el_el, embeddings_el_ions
        else:
            return embeddings_el_el,


class FermiNetEmbedding(hk.Module):
    def __init__(self, config: EmbeddingConfigFermiNet, mlp_config: MLPConfig, n_up, name=None):
        super().__init__(name=name)
        self.config = config
        self.mlp_config = mlp_config
        self.n_up = n_up

    def _split_into_same_diff(self, features_el_el):
        n_el = features_el_el.shape[-3] # [batch x n_el x n_el x features]
        batch_dims = features_el_el.shape[:-3]
        n_dn = n_el - self.n_up
        h_uu = features_el_el[..., : self.n_up, : self.n_up, :].reshape(batch_dims + (self.n_up * self.n_up, -1))
        h_ud = features_el_el[..., : self.n_up, self.n_up :, :].reshape(batch_dims + (self.n_up * n_dn, -1))
        h_du = features_el_el[..., self.n_up :, : self.n_up, :].reshape(batch_dims + (n_dn * self.n_up, -1))
        h_dd = features_el_el[..., self.n_up :, self.n_up :, :].reshape(batch_dims + (n_dn * n_dn, -1))
        return [jnp.concatenate([h_uu, h_dd], axis=-2), jnp.concatenate([h_ud, h_du], axis=-2)]

    def __call__(self, features: InputFeatures):
        h_el = features.el
        h_ion = features.ion
        h_el_ion = features.el_ion

        if self.config.use_h_two_same_diff:
            h_el_el = self._split_into_same_diff(features.el_el)
        else:
            h_el_el = features.el_el

        for i in range(self.config.n_iterations - 1):
            h_el_in = SymmetricFeatures(self.config, self.mlp_config, self.n_up, name=f"symm_features_{i}")(h_el, h_ion, h_el_el, h_el_ion)
            h_el = MLP([self.config.n_hidden_one_el[i]], self.mlp_config, residual=True, name=f"h_el_{i}")(h_el_in)

            if self.config.use_h_two_same_diff:
                h_el_el[0] = MLP([self.config.n_hidden_two_el[i]], self.mlp_config, residual=True, name=f"h_same_{i}")(h_el_el[0])
                h_el_el[1] = MLP([self.config.n_hidden_two_el[i]], self.mlp_config, residual=True, name=f"h_diff_{i}")(h_el_el[1])
            else:
                h_el_el = MLP([self.config.n_hidden_two_el[i]], self.mlp_config, residual=True, name=f"h_el_el_{i}")(h_el_el)

            if self.config.use_el_ion_stream:
                h_el_ion = MLP([self.config.n_hidden_two_el[i]], self.mlp_config, residual=True, name=f"h_el_ion_{i}")(h_el_ion)

        # We have one more 1-electron layer than 2-electron layers: Now apply the last 1-electron layer
        h_el_in = SymmetricFeatures(self.config, self.mlp_config, self.n_up)(h_el, h_ion, h_el_el, h_el_ion)
        h_el = MLP([self.config.n_hidden_one_el[i]], self.mlp_config, residual=True, name="h_el")(h_el_in)
        if not self.config.use_el_ion_stream:
            h_el_ion = h_el[..., jnp.newaxis, :]

        return Embeddings(h_el, h_ion, h_el_el, h_el_ion)

class PauliNetEmbedding(hk.Module):
    def __init__(self, config: EmbeddingConfigDeepErwin1, mlp_config: MLPConfig, n_up, name=None):
        super().__init__(name=name)
        self.config = config
        self.mlp_config = mlp_config
        self.n_up = n_up

    def _split_into_same_diff(self, features_el_el):
        n_el = features_el_el.shape[-3] # [batch x n_el x n_el x features]
        batch_dims = features_el_el.shape[:-3]
        n_dn = n_el - self.n_up
        h_uu = features_el_el[..., : self.n_up, : self.n_up, :].reshape(batch_dims + (self.n_up * self.n_up, -1))
        h_ud = features_el_el[..., : self.n_up, self.n_up :, :].reshape(batch_dims + (self.n_up * n_dn, -1))
        h_du = features_el_el[..., self.n_up :, : self.n_up, :].reshape(batch_dims + (n_dn * self.n_up, -1))
        h_dd = features_el_el[..., self.n_up :, self.n_up :, :].reshape(batch_dims + (n_dn * n_dn, -1))
        return [jnp.concatenate([h_uu, h_dd], axis=-2), jnp.concatenate([h_ud, h_du], axis=-2)]

    def __call__(self, features: InputFeatures):
        h_el_el = features.el_el
        h_el = features.el
        h_ion = features.ion
        h_el_ion = features.el_ion

        h_ion_mapped = MLP([self.config.embedding_dim], self.mlp_config, name="h_ion_map")(h_ion)

        f_pairs_same, f_pairs_diff = self._split_into_same_diff(h_el_el)

        n_el = h_el_el.shape[-3] # [batch x n_el x n_el x features]
        n_dn = n_el - self.n_up
        batch_dims = h_el_el.shape[:-3]
        for n in range(self.config.n_iterations):
            h_same = MLP(self.config.n_hidden_h + [self.config.embedding_dim], self.mlp_config, residual=False, name=f"h_same_{n}")(h_el)
            h_diff = MLP(self.config.n_hidden_h + [self.config.embedding_dim], self.mlp_config, residual=False, name=f"h_diff_{n}")(h_el)

            h_u_u = h_same[..., jnp.newaxis, :self.n_up, :]
            h_d_d = h_same[..., jnp.newaxis, self.n_up:, :]
            h_u_d = h_diff[..., jnp.newaxis, self.n_up:, :]
            h_d_u = h_diff[..., jnp.newaxis, :self.n_up, :]

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

            w_u_u = w_same[..., :self.n_up * self.n_up, :].reshape(batch_dims + (self.n_up, self.n_up, -1))
            w_u_d = w_diff[..., :self.n_up * n_dn, :].reshape(batch_dims + (self.n_up, n_dn, -1))
            w_d_u = w_diff[..., self.n_up * n_dn:, :].reshape(batch_dims + (n_dn, self.n_up, -1))
            w_d_d = w_same[..., self.n_up * self.n_up:, :].reshape(batch_dims + (n_dn, n_dn, -1))

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


class EnvelopeOrbitals(hk.Module):
    def __init__(self, config: EnvelopeOrbitalsConfig, physical_config: PhysicalConfig, mlp_config: MLPConfig, n_dets, full_det, name=None):
        super().__init__(name=name)
        self.config = config
        self.physical_config = physical_config
        self.mlp_config = mlp_config
        self.n_el = physical_config.n_electrons
        self.n_up = physical_config.n_up
        self.n_dets = n_dets
        self.output_size_up = physical_config.n_electrons if full_det else self.n_up
        self.output_size_dn = physical_config.n_electrons if full_det else (self.n_el - self.n_up)
        if self.config.initialization == "analytical":
            weights, alphas = get_envelope_exponents_from_atomic_orbitals(
                self.physical_config, pad_full_det=full_det
            )
            self._alpha_up_init = lambda s, t: jnp.tile(alphas[0], [1, n_dets])
            self._alpha_dn_init = lambda s, t: jnp.tile(alphas[1], [1, n_dets])
            self._weights_up_init = lambda s, t: jnp.tile(weights[0], [1, n_dets])
            self._weights_dn_init = lambda s, t: jnp.tile(weights[1], [1, n_dets])
        else:
            self._alpha_up_init = jnp.ones
            self._alpha_dn_init = jnp.ones
            self._weights_up_init = jnp.ones
            self._weights_dn_init = jnp.ones

    def __call__(self, dist_el_ion, emb_el):
        # Backflow factor
        bf_up = MLP(
            self.config.n_hidden + [self.n_dets * self.output_size_up],
            self.mlp_config,
            output_bias=self.config.use_bias,
            linear_out=True,
            name="bf_up",
        )(emb_el[..., : self.n_up, :])
        bf_dn = MLP(
            self.config.n_hidden + [self.n_dets * self.output_size_dn],
            self.mlp_config,
            output_bias=self.config.use_bias,
            linear_out=True,
            name="bf_dn",
        )(emb_el[..., self.n_up :, :])
        # output-shape: [batch x n_up x (n_dets * n_up_orb)]
        bf_up = jnp.swapaxes(bf_up.reshape(bf_up.shape[:-1] + (self.n_dets, self.output_size_up)), -3, -2)
        bf_dn = jnp.swapaxes(bf_dn.reshape(bf_dn.shape[:-1] + (self.n_dets, self.output_size_dn)), -3, -2)
        # output-shape: [batch x n_dets x n_up x n_up_orb]

        # Envelopes
        if self.config.envelope_type == "isotropic_exp":
            mo_matrix_up, mo_matrix_dn = self._envelope_isotropic(dist_el_ion)
        else:
            raise ValueError(f"Unknown envelope type: {self.config.envelope_type}")
        return mo_matrix_up * bf_up, mo_matrix_dn * bf_dn

    def _envelope_isotropic(self, el_ion_dist):
        n_ions = self.physical_config.n_ions
        shape_up = [n_ions, self.n_dets * self.output_size_up]
        shape_dn = [n_ions, self.n_dets * self.output_size_dn]
        alpha_up = hk.get_parameter("alpha_up", shape_up, init=self._alpha_up_init)
        alpha_dn = hk.get_parameter("alpha_dn", shape_dn, init=self._alpha_dn_init)
        weights_up = hk.get_parameter("weights_up", shape_up, init=self._weights_up_init)
        weights_dn = hk.get_parameter("weights_dn", shape_dn, init=self._weights_dn_init)

        d_up = el_ion_dist[..., :self.n_up, :, jnp.newaxis]  # [batch x el_up x ion x 1 (det*orb)]
        d_dn = el_ion_dist[..., self.n_up:, :, jnp.newaxis]
        exp_up = jax.nn.softplus(alpha_up) * d_up  # [batch x el_up x ion x (det*orb)]
        exp_dn = jax.nn.softplus(alpha_dn) * d_dn
        exp_up = register_scale_and_shift(exp_up, d_up, scale=alpha_up, shift=None)
        exp_dn = register_scale_and_shift(exp_dn, d_dn, scale=alpha_dn, shift=None)

        # Sum over ions; new shape: [batch x  el_up x (det*orb)]
        orb_up = jnp.sum(weights_up * jnp.exp(-exp_up), axis=-2)
        orb_dn = jnp.sum(weights_dn * jnp.exp(-exp_dn), axis=-2)
        orb_up = jnp.reshape(orb_up, orb_up.shape[:-1] + (self.n_dets, self.output_size_up))
        orb_dn = jnp.reshape(orb_dn, orb_dn.shape[:-1] + (self.n_dets, self.output_size_dn))
        orb_up = jnp.moveaxis(orb_up, -2, -3) # [batch x det x el_up x orb]
        orb_dn = jnp.moveaxis(orb_dn, -2, -3)
        return orb_up, orb_dn

class BaselineOrbitals(hk.Module):
    def __init__(self, config: BaselineOrbitalsConfig, physical_config: PhysicalConfig, mlp_config: MLPConfig, n_dets,
                 full_det, name=None):
        super().__init__(name=name)
        self.config = config
        self.mlp_config = mlp_config
        self.physical_config = physical_config
        self.n_el = physical_config.n_electrons
        self.n_up = physical_config.n_up
        self.n_dets = n_dets
        self.output_size_up = physical_config.n_electrons if full_det else self.n_up
        self.output_size_dn = physical_config.n_electrons if full_det else (self.n_el - self.n_up)
        self.full_det = full_det

    def __call__(self, diff_dist: DiffAndDistances, embeddings: Embeddings, fixed_params):
        if self.config.use_bf_shift:
            diff_dist = self.apply_backflow_shift(diff_dist, embeddings)

        # Evaluate atomic and molecular orbitals for every determinant
        mo_matrix_up, mo_matrix_dn = get_baseline_slater_matrices(diff_dist.diff_el_ion, diff_dist.dist_el_ion, fixed_params,
                                                                  self.full_det)
        if self.config.use_bf_factor:
            bf_up = MLP(
                self.config.n_hidden_bf_factor + [self.n_dets * self.output_size_up],
                self.mlp_config,
                output_bias=self.config.use_bf_factor_bias,
                linear_out=True,
                name="bf_up",
            )(embeddings.el[..., : self.n_up, :])
            bf_dn = MLP(
                self.config.n_hidden_bf_factor + [self.n_dets * self.output_size_dn],
                self.mlp_config,
                output_bias=self.config.use_bf_factor_bias,
                linear_out=True,
                name="bf_dn",
            )(embeddings.el[..., self.n_up:, :])
            # output-shape: [batch x n_up x (n_dets * n_up_orb)]
            bf_up = jnp.swapaxes(bf_up.reshape(bf_up.shape[:-1] + (self.n_dets, self.output_size_up)), -3, -2)
            bf_dn = jnp.swapaxes(bf_dn.reshape(bf_dn.shape[:-1] + (self.n_dets, self.output_size_dn)), -3, -2)
            # output-shape: [batch x n_dets x n_up x n_up_orb]

            mo_matrix_up *= bf_up
            mo_matrix_dn *= bf_dn
        return mo_matrix_up, mo_matrix_dn


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


class OrbitalNet(hk.Module):
    def __init__(self, config: OrbitalsConfig, physical_config: PhysicalConfig, mlp_config: MLPConfig, name=None):
        super().__init__(name=name)
        self.config = config
        self.physical_config = physical_config
        self.mlp_config = mlp_config

    def __call__(self, diff_dist: DiffAndDistances, embeddings: Embeddings, fixed_params):
        if self.config.envelope_orbitals:
            mo_up, mo_dn = EnvelopeOrbitals(
                self.config.envelope_orbitals,
                self.physical_config,
                self.mlp_config,
                self.config.n_determinants,
                self.config.use_full_det
            )(diff_dist.dist_el_ion, embeddings.el)
        else:
            mo_up, mo_dn = 0.0, 0.0

        # WIP
        if self.config.baseline_orbitals:
            mo_up, mo_dn = BaselineOrbitals(
                self.config.baseline_orbitals,
                self.physical_config,
                self.mlp_config,
                self.config.n_determinants,
                self.config.use_full_det
            )(diff_dist, embeddings, fixed_params)

        return mo_up, mo_dn


class JastrowFactor(hk.Module):
    def __init__(self, config: JastrowConfig, mlp_config: MLPConfig, n_up, name=None):
        super().__init__(name=name)
        self.config = config
        self.mlp_config = mlp_config
        self.n_up = n_up

    def __call__(self, embeddings: Embeddings):
        if self.config.differentiate_spins:
            jastrow_up = MLP(self.config.n_hidden + [1], self.mlp_config, linear_out=True, output_bias=False, name="up")(
                embeddings.el[..., : self.n_up, :])
            jastrow_dn = MLP(self.config.n_hidden + [1], self.mlp_config, linear_out=True, output_bias=False, name="dn")(
                embeddings.el[..., self.n_up:, :])
            jastrow = jnp.sum(jastrow_up, axis=(-2, -1)) + jnp.sum(jastrow_dn, axis=(-2, -1))
        else:
            jastrow = MLP(self.config.n_hidden + [1], linear_out=True, output_bias=False, name="mlp")(embeddings.el)
            jastrow = jnp.sum(jastrow, axis=(-2, -1))
        return jastrow


def evaluate_sum_of_determinants(mo_matrix_up, mo_matrix_dn, use_full_det):
    LOG_EPSILON = 1e-8

    if use_full_det:
        mo_matrix = jnp.concatenate([mo_matrix_up, mo_matrix_dn], axis=-2)
        sign_total, log_total = jnp.linalg.slogdet(mo_matrix)
    else:
        sign_up, log_up = jnp.linalg.slogdet(mo_matrix_up)
        sign_dn, log_dn = jnp.linalg.slogdet(mo_matrix_dn)
        log_total = log_up + log_dn
        sign_total = sign_up * sign_dn
    log_shift = jnp.max(log_total, axis=-1, keepdims=True)
    psi = jnp.exp(log_total - log_shift) * sign_total
    psi = jnp.sum(psi, axis=-1)  # sum over determinants
    log_psi_sqr = 2 * (jnp.log(jnp.abs(psi) + LOG_EPSILON) + jnp.squeeze(log_shift, -1))
    return log_psi_sqr


def get_baseline_slater_matrices(diff_el_ion, dist_el_ion, fixed_params, full_det):
    atomic_orbitals, ao_cusp_params, mo_cusp_params, mo_coeff, ind_orb, ci_weights = fixed_params
    n_dets, n_up = ind_orb[0].shape
    n_dn = ind_orb[1].shape[1]

    mo_matrix_up = evaluate_molecular_orbitals(
        diff_el_ion[..., :n_up, :, :], dist_el_ion[..., :n_up, :], atomic_orbitals, mo_coeff[0], ao_cusp_params, mo_cusp_params[0]
    )
    mo_matrix_dn = evaluate_molecular_orbitals(
        diff_el_ion[..., n_up:, :, :], dist_el_ion[..., n_up:, :], atomic_orbitals, mo_coeff[1], ao_cusp_params, mo_cusp_params[1]
    )
    # 1) Select orbitals for each determinant => [(batch) x n_el x n_det x n_orb]
    # 2) Move determinant axis forward => [(batch) x n_det x n_el x n_orb]
    mo_matrix_up = jnp.moveaxis(mo_matrix_up[..., ind_orb[0]], -2, -3)
    mo_matrix_dn = jnp.moveaxis(mo_matrix_dn[..., ind_orb[1]], -2, -3)

    if full_det:
        batch_shape = mo_matrix_up.shape[:-2]
        mo_matrix_up = jnp.concatenate([mo_matrix_up, jnp.zeros(batch_shape + (n_up, n_dn))], axis=-1)
        mo_matrix_dn = jnp.concatenate([jnp.zeros(batch_shape + (n_dn, n_up)), mo_matrix_dn], axis=-1)


    # CI weights need to go somewhere; could also multiply onto mo_dn, should yield same results
    ci_weights = ci_weights[:, None, None]
    ci_weights_up = jnp.abs(ci_weights)**(1/n_up)

    # adjust sign of first col to match det sign
    ci_weights_up *= jnp.concatenate([jnp.sign(ci_weights), jnp.ones([n_dets, 1, mo_matrix_up.shape[-1]-1])], axis=-1)
    mo_matrix_up *= ci_weights_up
    return mo_matrix_up, mo_matrix_dn

class Wavefunction(hk.Module):
    def __init__(self, config: ModelConfig, phys_config: PhysicalConfig, name="wf"):
        super().__init__(name=name)
        self.config = config
        self.phys_config = phys_config

    def __call__(self, r, R, Z, fixed_params=None):
        fixed_params = fixed_params or {}
        diff_dist, features = self._calculate_features(r, R, Z, fixed_params.get('input'))
        embeddings = self._calculate_embedding(features)
        mo_up, mo_dn = self._calculate_orbitals(diff_dist, embeddings, fixed_params.get('orbitals'))
        log_psi_sqr = evaluate_sum_of_determinants(mo_up, mo_dn, self.config.orbitals.use_full_det)

        # Jastrow factor to the total wavefunction
        if self.config.jastrow:
            log_psi_sqr += self._calculate_jastrow(embeddings)

        # Electron-electron-cusps
        if self.config.use_el_el_cusp_correction:
            log_psi_sqr += self._el_el_cusp(diff_dist.dist_el_el)
        return log_psi_sqr

    def get_slater_matrices(self, r, R, Z, fixed_params=None):
        fixed_params = fixed_params or {}
        diff_dist, features = self._calculate_features(r, R, Z, fixed_params.get('input'))
        embeddings = self._calculate_embedding(features)
        mo_up, mo_dn = self._calculate_orbitals(diff_dist, embeddings, fixed_params.get('orbitals'))
        return mo_up, mo_dn


    @haiku.experimental.name_like("__call__")
    def _calculate_features(self, r, R, Z, fixed_params=None):
        Z = jnp.array(Z, dtype=float)
        diff_dist, features = InputPreprocessor(self.config.features,
                                                self.phys_config,
                                                self.config.mlp,
                                                name="input")(r, R, Z, fixed_params)
        return diff_dist, features

    @haiku.experimental.name_like("__call__")
    def _calculate_embedding(self, features):
        if self.config.embedding.name in ["ferminet", "dpe4"]:
            return FermiNetEmbedding(self.config.embedding,
                                     self.config.mlp,
                                     self.phys_config.n_up,
                                     name="embedding")(features)
        elif self.config.embedding.name in ["dpe1"]:
            return PauliNetEmbedding(self.config.embedding,
                                     self.config.mlp,
                                     self.phys_config.n_up,
                                     name="embedding_pauli")(features)
        else:
            raise ValueError(f"Unknown embedding: {self.config.embedding.name}")

    @haiku.experimental.name_like("__call__")
    def _calculate_orbitals(self, diff_dist, embeddings, fixed_params=None):
        orbital_func = OrbitalNet(self.config.orbitals, self.phys_config, self.config.mlp, name="orbitals")
        return orbital_func(diff_dist, embeddings, fixed_params)

    @haiku.experimental.name_like("__call__")
    def _calculate_jastrow(self, embeddings):
        return JastrowFactor(self.config.jastrow, self.config.mlp, self.phys_config.n_up)(embeddings)

    def _el_el_cusp(self, el_el_dist):
        factor = np.ones([self.phys_config.n_electrons, self.phys_config.n_electrons]) * 0.5
        factor[: self.phys_config.n_up, : self.phys_config.n_up] = 0.25
        factor[self.phys_config.n_up :, self.phys_config.n_up :] = 0.25

        r_cusp_el_el = 1.0
        A = r_cusp_el_el ** 2
        B = r_cusp_el_el
        # No factor 0.5 here, e.g. when comparing to NatChem 2020, [doi.org/10.1038/s41557-020-0544-y], because:
        # A) We double-count electron-pairs because we take the full distance matrix (and not only the upper triangle)
        # B) We model log(psi^2)=2*log(|psi|) vs log(|psi|) int NatChem 2020, i.e. the cusp correction needs a factor 2
        return -jnp.sum(factor * A / (el_el_dist + B), axis=[-2, -1])


    def init_for_multitransform(self):
        return self.__call__, (self.__call__, self.get_slater_matrices, self._calculate_features, self._calculate_embedding, self._calculate_orbitals, self._calculate_jastrow)


def build_log_psi_squared(config: ModelConfig, phys_config: PhysicalConfig, fixed_params, rng_seed):
    # Initialize fixed model parameters
    fixed_params = fixed_params or init_model_fixed_params(config, phys_config)

    # Build model
    model = hk.multi_transform(lambda: Wavefunction(config, phys_config).init_for_multitransform())

    # Initialized trainable parameters using a dummy batch
    n_el, _, R, Z = phys_config.get_basic_params()
    rng1, rng2 = jax.random.split(jax.random.PRNGKey(rng_seed), 2)
    r = jax.random.normal(rng1, [1, n_el, 3])
    params = model.init(rng2, r, R, Z, fixed_params)

    # Remove rng-argument (replace by None) and move parameters to back of function
    log_psi_sqr = lambda params, *batch: model.apply[0](params, None, *batch)
    orbitals = lambda params, *batch: model.apply[1](params, None, *batch)

    return log_psi_sqr, orbitals, params, fixed_params


def init_model_fixed_params(config: ModelConfig, physical_config: PhysicalConfig):
    """
    Computes CASSCF baseline solution for DeepErwin model and initializes fixed parameters.

    Args:
        casscf_config (CASSCFConfig): CASSCF hyperparmeters
        physical_config (PhysicalConfig): Description of the molecule

    Returns:
        dict: Initial fixed parameters
    """
    fixed_params = dict(input={}, orbitals=None, baseline_energies=dict(E_ref=physical_config.E_ref))

    LOGGER.debug("Calculating baseline solution...")

    if config.orbitals.baseline_orbitals:
        cas_config = config.orbitals.baseline_orbitals.baseline
        fixed_params["orbitals"], (E_hf, E_casscf) = get_baseline_solution(physical_config, cas_config, config.orbitals.n_determinants)
        fixed_params["baseline_energies"].update(dict(E_hf=E_hf, E_casscf=E_casscf))
        LOGGER.debug(f"Finished baseline calculation: E_casscf={E_casscf:.6f}")

    if config.features.use_local_coordinates:
        fixed_params["input"]["local_rotations"] = build_local_rotation_matrices(physical_config)

    return jax.tree_util.tree_map(jnp.array, fixed_params)


