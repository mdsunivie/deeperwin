import jax.numpy as jnp
import haiku as hk
from deeperwin.configuration import (
    TransformerConfig,
    MLPConfig,
    EmbeddingConfigTransformer,
    EmbeddingConfigAxialTranformer,
)
from deeperwin.model.attention import Attention
from deeperwin.model.mlp import MLP
from deeperwin.model.definitions import Embeddings, InputFeatures


class Transformer(hk.Module):
    def __init__(self, config: TransformerConfig, mlp_config: MLPConfig, name=None):
        super().__init__(name=name)
        self.config = config
        self.attention_value_dim = config.attention_value_dim or config.attention_dim
        self.one_el_feature_dim = config.attention_output_dim or self.attention_value_dim * config.n_heads
        self.residual = config.residual
        self.mlp_config = mlp_config

    def __call__(self, features: jnp.array, features_sender=None, edge_features=None, edge_features_sender=None):
        features = hk.Linear(self.one_el_feature_dim, with_bias=False, name="upmapping")(features)

        for n in range(self.config.n_iterations):
            self_att_feat = Attention(
                self.config.attention_dim,
                self.config.n_heads,
                self.config.residual,
                self.attention_value_dim,
                self.one_el_feature_dim,
                self.config.use_layer_norm,
                use_residual_before_lin=self.config.use_residual_before_lin,
            )(features, edge_features=edge_features)

            # if self.config.initialize_with_sender_att:
            #     if n > 0:
            #         features_sender = None

            if features_sender is not None:
                if self.config.combine_attention_blocks:
                    self_att_feat += Attention(
                        self.config.attention_dim,
                        self.config.n_heads,
                        False,
                        self.attention_value_dim,
                        self.one_el_feature_dim,
                        self.config.use_layer_norm,
                        use_residual_before_lin=self.config.use_residual_before_lin,
                    )(features, features_sender, edge_features=edge_features_sender)
                else:
                    self_att_feat = Attention(
                        self.config.attention_dim,
                        self.config.n_heads,
                        self.config.residual,
                        self.attention_value_dim,
                        self.one_el_feature_dim,
                        self.config.use_layer_norm,
                        use_residual_before_lin=self.config.use_residual_before_lin,
                    )(self_att_feat, features_sender, edge_features=edge_features_sender)

            if self.config.use_layer_norm:
                mlp_input = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(self_att_feat)
                mlp_output = MLP([self.one_el_feature_dim] * (self.config.mlp_depth + 1), self.mlp_config)(mlp_input)
            else:
                mlp_output = MLP([self.one_el_feature_dim] * (self.config.mlp_depth + 1), self.mlp_config)(
                    self_att_feat
                )

            if self.config.final_residual and (mlp_output.shape == self_att_feat.shape):
                features = mlp_output + self_att_feat
            else:
                features = mlp_output

        return features


class TransformerEmbedding(hk.Module):
    def __init__(self, config: EmbeddingConfigTransformer, mlp_config: MLPConfig, name=None):
        super().__init__(name=name)
        self.el_transformer = Transformer(config.el_transformer, mlp_config, name="transformer_el")
        self.ion_transformer = None
        self.config = config
        if config.ion_transformer:
            self.ion_transformer = Transformer(config.ion_transformer, mlp_config, name="transformer_ion")

    def __call__(self, features: InputFeatures, n_up: int):
        del n_up
        features_ion = None
        edge_ion_ion, edge_el_ion, edge_el_el = None, None, None
        if self.config.edge_feature:
            if self.config.edge_feature.ion_ion:
                edge_ion_ion = features.ion_ion
            if self.config.edge_feature.el_ion:
                edge_el_ion = features.el_ion
            if self.config.edge_feature.el_el:
                edge_el_el = features.el_el

        if self.ion_transformer:
            features_ion = self.ion_transformer(features.ion, edge_features=edge_ion_ion)

        features_el = self.el_transformer(
            features.el, features_ion, edge_features=edge_el_el, edge_features_sender=edge_el_ion
        )
        return Embeddings(features_el, features_ion, None, None)


class AxialTransformerEmbedding(hk.Module):
    def __init__(self, config: EmbeddingConfigAxialTranformer, mlp_config: MLPConfig, name=None):
        super().__init__(name=name)
        self.config_ion_attention = config.ion_attention
        self.config_el_attention = config.el_attention

        self.config = config
        self.mlp_config = mlp_config

    def __call__(self, features: InputFeatures, n_up: int):
        del n_up
        el_ion_features = features.el_ion
        features = hk.Linear(self.config.embedding_dim, with_bias=False, name="upmapping")(
            el_ion_features
        )  # Shape: bs x n_el x n_ions x features
        if not self.config.axial_attention_ion_per_layer:
            # First compute attention for each el. independently to each ion
            axial_attention_ion = Attention(
                self.config_ion_attention.attention_dim,
                self.config_ion_attention.n_heads,
                residual=True,
                attention_value_dim=None,  # use attention_dim
                output_dim=self.config.embedding_dim,
                layer_norm=self.config_ion_attention.use_layer_norm,
                name="axial_attention_ion",
            )
            features = axial_attention_ion(features)

        for n in range(self.config.n_iterations):
            if self.config.axial_attention_ion_per_layer:
                # First compute attention for each el. independently to each ion
                axial_attention_ion = Attention(
                    self.config_ion_attention.attention_dim,
                    self.config_ion_attention.n_heads,
                    residual=True,  # residual
                    attention_value_dim=None,  # attention_value_dim
                    output_dim=self.config.embedding_dim,
                    layer_norm=self.config_ion_attention.use_layer_norm,
                    name="axial_attention_ion",
                )
                features = axial_attention_ion(features)

            # Second compute attention to all other el.
            axial_attention_el = Attention(
                self.config_el_attention.attention_dim,
                self.config_el_attention.n_heads,
                residual=True,  # residual
                attention_value_dim=None,  # attention_value_dim
                output_dim=self.config.embedding_dim,
                layer_norm=self.config_el_attention.use_layer_norm,
                name="axial_attention_el",
            )

            # Swap el with ions to compute axial attention wrt. to all other el.
            features = jnp.swapaxes(features, -2, -3)
            features = axial_attention_el(features)

            # Swap back to original shape
            features = jnp.swapaxes(features, -2, -3)

            features = MLP([self.config.output_dim] * (self.config.mlp_depth + 1), self.mlp_config, residual=True)(
                features
            )

        if self.config.agg_ion_contribution:
            features = jnp.sum(features, axis=-2)
            features_el = hk.Linear(self.config.output_dim, with_bias=True, name="aggregation")(features)
        else:
            features_el = features
        return Embeddings(features_el, None, None, features)
