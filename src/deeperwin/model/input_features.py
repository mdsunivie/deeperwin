"""
File containing input processing layer, including Pairwise Feature construction layer
"""
import functools
from typing import Literal, Optional, Tuple, Dict
import jax.numpy as jnp
import numpy as np
import haiku as hk
from deeperwin.configuration import (
    InputFeatureConfig,
    MLPConfig,
)
from deeperwin.model.definitions import *
from deeperwin.model.mlp import MLP, get_rbf_features, get_gauss_env_features
from deeperwin.utils.utils import get_distance_matrix, get_el_ion_distance_matrix
from deeperwin.model.e3nn_utils import to_irreps_array
import e3nn_jax as e3nn
import jax

class PairwiseFeatures(hk.Module):
    def __init__(self, config: InputFeatureConfig, pair_type: Literal["el_el", "el_ion"], name=None):
        super().__init__(name=name)
        self.config = config
        self.pair_type = pair_type
        self.use_differences = ((self.pair_type == 'el_el') and self.config.use_el_el_differences) or \
                               ((self.pair_type == 'el_ion') and self.config.use_el_ion_differences)

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
            features_rbf = get_rbf_features(dist, self.config.n_rbf_features)
            features.append(features_rbf)
        if self.config.n_bessel_features > 0:
            bessel_features = e3nn.bessel(dist, self.config.n_bessel_features, self.config.r_cut_bessel)
            features.append(bessel_features)
        if self.config.use_distance_features:
            if self.config.log_scale_distances:
                features.append(jnp.log(1+dist)[..., None])
            else:
                features.append(dist[..., None])
        if self.use_differences:
            diff_features = differences
            if self.config.log_scale_distances:
                diff_features *= (jnp.log(1 + dist) / dist)[..., None]
            features.append(diff_features)

        features = jnp.concatenate(features, axis=-1)
        return features


def init_particle_features(feature_inter_particle, dist, nb_features, nb_layers, gating_operation, max_scale, mlp_config, name=None):
    mlp_filter = MLP([nb_features] * nb_layers,
                     mlp_config,
                     name=name)
    msg = mlp_filter(feature_inter_particle)
    if gating_operation == "rbf":
        gate = get_rbf_features(dist, n_features=nb_features, r_max=5)
        gate = hk.Linear(nb_features, False, name="linear_mapping_el_el_rbf")(gate)
        msg = msg * gate
    elif gating_operation == "gauss":
        gauss_env_features = functools.partial(get_gauss_env_features, nb_features=nb_features, max_scale=max_scale)
        gate = gauss_env_features(dist)
        msg = msg * gate
    else:
        msg = msg
    return msg

class InputPreprocessor(hk.Module):
    def __init__(
        self,
        config: InputFeatureConfig, 
        mlp_config: MLPConfig, 
        wavefunction_definition: WavefunctionDefinition,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)
        self.config = config
        self.mlp_config = mlp_config
        self.features_el_el = PairwiseFeatures(self.config, "el_el")
        self.features_el_ion = PairwiseFeatures(self.config, "el_ion")
        self.Z_max = wavefunction_definition.Z_max
        self.Z_min = wavefunction_definition.Z_min

    def __call__(
        self, 
        n_up: int, 
        n_dn: int, 
        r: jnp.ndarray, 
        R: jnp.ndarray, 
        Z: jnp.ndarray, 
        fixed_params: Dict = None
    ) -> Tuple[DiffAndDistances, InputFeatures]:
        Z = jnp.array(Z, int)
        if Z.ndim == 1: # no batch-dim for Z => tile across batch
            Z = jnp.tile(Z, r.shape[:-2] + (1,))
        if R.ndim == 2:  # no batch-dim for R => tile across batch
            R = jnp.tile(R, r.shape[:-2] + (1, 1))

        # Compute cartesian distances and difference vectors
        if self.config.coordinates == "global_rot":
            r = jnp.einsum("ni,...i->...n", fixed_params['global_rotation'], r)
            R = jnp.einsum("ni,...i->...n", fixed_params['global_rotation'], R)
        diff_el_el, dist_el_el = get_distance_matrix(r)
        diff_el_ion, dist_el_ion = get_el_ion_distance_matrix(r, R)
        diff_ion_ion, dist_ion_ion = get_distance_matrix(R)

        if self.config.coordinates == "local_rot":
            diff_el_ion = jnp.einsum("Jni,...Ji->...Jn", fixed_params["local_rotations"], diff_el_ion)


        # Ion-ion features
        if self.config.n_ion_ion_rbf_features > 0:
            diff_ion_ion, dist_ion_ion = get_distance_matrix(R, full=True) # use pairwise features
            rbfs = get_rbf_features(dist_ion_ion, n_features=self.config.n_ion_ion_rbf_features, r_max=self.config.rmax)
            features_ion_ion = jnp.concatenate([diff_ion_ion, dist_ion_ion[..., None], rbfs], axis=-1)
        else:
            features_ion_ion = None

        
        # Ion features
        if self.config.ion_embed_type is None:
            features_ion = None
        elif self.config.ion_embed_type == 'lookup':
                self.ion_embedding = lambda Z: hk.Embed(self.Z_max - self.Z_min + 1,
                                                        self.config.n_ion_features,
                                                        lookup_style=hk.EmbedLookupStyle.ONE_HOT, name="h_ion")(
                    Z - self.Z_min)
                features_ion = self.ion_embedding(Z)
        elif self.config.ion_embed_type == 'one-hot':
            self.ion_embedding = lambda Z: jax.nn.one_hot(Z - self.Z_min, self.Z_max - self.Z_min + 1)
            features_ion = self.ion_embedding(Z)
        elif self.config.ion_embed_type == 'mlp':
            self.ion_embedding = MLP([self.config.n_ion_features], self.mlp_config, name="h_ion")
            features_ion = self.ion_embedding(Z[..., None].astype(float))
        else:
            raise ValueError(f"Unknown ion_embed_type {self.config.ion_embed_type}")

        # Electron features
        features_el_el = self.features_el_el(diff_el_el, dist_el_el)
        features_el_ion = self.features_el_ion(diff_el_ion, dist_el_ion)
        features_el = []
        el_ion_edges, el_el_edges = None, None
        if self.config.init_with_el_el_feat:
            el_el_features_fct = functools.partial(init_particle_features,
                                                           nb_features=self.config.n_el_el_features,
                                                           nb_layers=self.config.n_el_el_layers,
                                                           gating_operation=self.config.el_el_gating_operation,
                                                           max_scale=self.config.max_scale_gauss,
                                                           mlp_config=self.mlp_config,
                                                           name="el_el_features")
            el_el_edges = el_el_features_fct(features_el_el, dist_el_el)
            features_el.append(jnp.sum(el_el_edges, axis=-2))

        if self.config.init_with_el_ion_feat:
            el_ion_features_fct = functools.partial(init_particle_features,
                                                           nb_features=self.config.n_el_ion_features,
                                                           nb_layers=self.config.n_el_ion_layers,
                                                           gating_operation=self.config.el_ion_gating_operation,
                                                           max_scale=self.config.max_scale_gauss,
                                                           mlp_config=self.mlp_config,
                                                           name="el_ion_features")

            el_ion_edges = el_ion_features_fct(features_el_ion, dist_el_ion)
            features_el.append(jnp.sum(el_ion_edges, axis=-2)) # sum over ions

        if self.config.use_el_spin:
            spin_features = np.ones([*r.shape[:-1], 1])
            spin_features[..., n_up:, :] *= -1
            features_el.append(spin_features)
        if self.config.concatenate_el_ion_features:
            # concatenate all el-ion features into a long h_one feature
            features_el.append(jnp.reshape(features_el_ion, features_el_ion.shape[:-2] + (-1,)))
        if len(features_el) == 0:
            if self.config.init_as_zeros:
                features_el.append(jnp.zeros(r.shape[:-1] + (1,)))
            else:
                features_el.append(jnp.ones(r.shape[:-1] + (1,)))
        features_el = jnp.concatenate(features_el, axis=-1)


        if self.config.exp_decay_el_ion_edge:
            if el_ion_edges is not None:
                features_el_ion = el_ion_edges
            else:
                el_ion_features_fct = functools.partial(init_particle_features,
                                                        nb_features=self.config.n_el_ion_features,
                                                        nb_layers=self.config.n_el_ion_layers,
                                                        gating_operation=self.config.el_ion_gating_operation,
                                                        max_scale=self.config.max_scale_gauss,
                                                        mlp_config=self.mlp_config,
                                                        name="el_ion_features")
                features_el_ion = el_ion_features_fct(features_el_ion, dist_el_ion)

        if self.config.exp_decay_el_el_edge:
            if el_el_edges is not None:
                features_el_el = el_el_edges
            else:
                el_el_features_fct = functools.partial(init_particle_features,
                                                       nb_features=self.config.n_el_el_features,
                                                       nb_layers=self.config.n_el_el_layers,
                                                       gating_operation=self.config.el_el_gating_operation,
                                                       max_scale=self.config.max_scale_gauss,
                                                       mlp_config=self.mlp_config,
                                                       name="el_el_features")
                features_el_el = el_el_features_fct(features_el_el, dist_el_el)


        diff_dist = DiffAndDistances(diff_el_el, dist_el_el, diff_el_ion, dist_el_ion, diff_ion_ion, dist_ion_ion)
        features = InputFeatures(features_el, features_ion, features_el_el, features_el_ion, features_ion_ion)
        return diff_dist, features