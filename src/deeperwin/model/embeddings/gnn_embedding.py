import haiku as hk
import jax.numpy as jnp
from deeperwin.model.gnn import DenseGNN, MessagePassingLayer, ScaleFeatures
from deeperwin.configuration import MLPConfig, EmbeddingConfigGNN
from deeperwin.model.definitions import InputFeatures, Embeddings, DiffAndDistances
from deeperwin.model.mlp import MLP


class GNNEmbedding(hk.Module):
    def __init__(self, config: EmbeddingConfigGNN, mlp_config: MLPConfig, name=None):
        super().__init__(name=name)
        self.config = config
        self.mlp_config = mlp_config

        self.mlp_el_ion = MLP([config.el_ion_width] * config.el_ion_depth, mlp_config, linear_out=False)
        self.mlp_same = MLP([config.el_el_width] * config.el_el_depth, mlp_config, linear_out=False)
        self.mlp_diff = MLP([config.el_el_width] * config.el_el_depth, mlp_config, linear_out=False)
        self.el_ion_mpnn = MessagePassingLayer(config.gnn.message_passing, config.gnn.use_edge_bias, "el_ion_mp")
        self.gnn = DenseGNN(config.gnn, mlp_config, "el_el_gnn")

        self.ion_ion = MLP([config.ion_ion_width] * config.ion_ion_depth, mlp_config, linear_out=False)
        if self.config.ion_gnn.name == "ion_gnn":
            self.ion_gnn = DenseGNN(config.ion_gnn, mlp_config, "ion_ion_gnn")
        elif self.config.ion_gnn.name == "phisnet_ion_emb":
            self.phisnet_downmapping = MLP([config.ion_gnn.ion_width] * config.ion_gnn.ion_depth, mlp_config, linear_out=False,
                          name="phisnet_downmapping")
            self.phisnet_mpnn = MessagePassingLayer(config.ion_gnn.message_passing, use_edge_bias=config.ion_gnn.use_edge_bias, name="phisnet_mpnn")

    def _split_same_diff(self, features, n_up, n_dn, batch_dims):
        up_up = features[..., :n_up, :n_up, :].reshape(batch_dims + (n_up * n_up, -1))
        up_dn = features[..., :n_up, n_up:, :].reshape(batch_dims + (n_up * n_dn, -1))
        dn_up = features[..., n_up:, :n_up, :].reshape(batch_dims + (n_dn * n_up, -1))
        dn_dn = features[..., n_up:, n_up:, :].reshape(batch_dims + (n_dn * n_dn, -1))
        same = jnp.concatenate([up_up, dn_dn], axis=-2)
        diff = jnp.concatenate([up_dn, dn_up], axis=-2)

        return same, diff

    def _embed_edges(self, features: InputFeatures, diff_dist: DiffAndDistances, n_up):
        batch_dims = features.el_el.shape[:-3]
        n_dn = features.el.shape[-2] - n_up

        same, diff = self._split_same_diff(features.el_el, n_up, n_dn, batch_dims)

        el_ion = self.mlp_el_ion(features.el_ion)
        same = self.mlp_same(same)
        diff = self.mlp_diff(diff)
        ion_ion = self.ion_ion(features.ion_ion)

        if self.config.exp_scaling is not None:
            s_dist, d_dist = self._split_same_diff(diff_dist.dist_el_el[..., None], n_up, n_dn, batch_dims)
            same *= ScaleFeatures(self.config.exp_scaling, self.config.el_el_width, name="el_same")(s_dist)
            diff *= ScaleFeatures(self.config.exp_scaling, self.config.el_el_width, name="el_diff")(d_dist)
            ion_ion *= ScaleFeatures(self.config.exp_scaling, self.config.ion_ion_width, name="ion_ion")(diff_dist.dist_ion_ion[..., None])
            el_ion *= ScaleFeatures(self.config.exp_scaling, self.config.el_ion_width, name="el_ion")(diff_dist.dist_el_ion[..., None])

        up_up = same[..., :(n_up * n_up), :].reshape(batch_dims + (n_up, n_up, -1))
        up_dn = diff[..., :(n_up * n_dn), :].reshape(batch_dims + (n_up, n_dn, -1))
        dn_up = diff[..., (n_up * n_dn):, :].reshape(batch_dims + (n_dn, n_up, -1))
        dn_dn = same[..., (n_up * n_up):, :].reshape(batch_dims + (n_dn, n_dn, -1))
        up = jnp.concatenate([up_up, up_dn], axis=-2)
        dn = jnp.concatenate([dn_up, dn_dn], axis=-2)
        el_el = jnp.concatenate([up, dn], axis=-3)
        return el_ion, el_el, ion_ion

    def _get_distance_mask(self, dist):
        if self.config.cutoff_type == "constant":
            return None
        x = dist / self.config.cutoff_distance
        if self.config.cutoff_type == "inverse":
            return 1 / (1 + x)
        elif self.config.cutoff_type == "exponential":
            return jnp.exp(-x)
        elif self.config.cutoff_type == "cosine":
            x = jnp.clip(jnp.pi * x, 0, jnp.pi)
            return 0.5 * (jnp.cos(x) + 1)
        else:
            raise ValueError(f"Unsupported mask: {self.config.cutoff_type}")


    def __call__(self, diff_dist: DiffAndDistances, features: InputFeatures, n_up: int):
        edge_el_ion, edge_el_el, edge_ion_ion = self._embed_edges(features, diff_dist, n_up)
        mask_el_ion = self._get_distance_mask(diff_dist.dist_el_ion)
        mask_el_el = self._get_distance_mask(diff_dist.dist_el_el)

        if self.config.ion_gnn.name == "phisnet_ion_emb":
            batch_dim = features.el_el.shape[:-3]
            features_ion = jnp.tile(features.ion, batch_dim + (1, 1))
            emb_ion = self.phisnet_downmapping(features_ion)
            emb_ion, _ = self.phisnet_mpnn(emb_ion, edges=edge_ion_ion)

        elif self.config.ion_gnn.name == "ion_gnn":
            emb_ion, _ = self.ion_gnn(features.ion, edges=edge_ion_ion)
        else:
            emb_ion = features.ion

        emb_el, edge_el_ion = self.el_ion_mpnn(features.el, emb_ion, edge_el_ion, mask_el_ion)
        emb_el, edge_el_el = self.gnn(emb_el, edges=edge_el_el, mask=mask_el_el)
        return Embeddings(emb_el, features.ion, edge_el_el, edge_el_ion)

