import jax
import haiku as hk
import jax.numpy as jnp
from deeperwin.model.mlp import MLP, get_activation
from deeperwin.model.attention import Attention
from deeperwin.configuration import DenseGNNConfig, MLPConfig, MessagePassingConfig
from typing import Optional

def _residual(x, x_old):
    if x.shape == x_old.shape:
        return x + x_old
    else:
        return x

class MessagePassingLayer(hk.Module):
    def __init__(self, config: MessagePassingConfig, use_edge_bias=True, name=None):
        super().__init__(name=name)
        self.config = config
        self.node_dim = config.node_dim
        self.edge_dim = config.edge_dim
        self.edge_bias = use_edge_bias
        self.aggfunc = dict(sum=jnp.sum, mean=jnp.mean)[config.aggregation]
        self.activation = get_activation(self.config.activation)

    def __call__(self, nodes_rec, nodes_snd=None, edges=None, mask=None):
        if nodes_snd is None:
            nodes_snd = nodes_rec
        edge_update = 0.0
        if self.config.use_node_features_for_gating:
            edge_update += hk.Linear(self.edge_dim, with_bias=self.edge_bias, name="linear_gate_receiver")(nodes_rec)[..., :, None, :]
            edge_update += hk.Linear(self.edge_dim, with_bias=self.edge_bias, name="linear_gate_sender")(nodes_snd)[..., None, :, :]
        if self.config.use_edge_features_for_gating:
            edge_update += hk.Linear(self.edge_dim, with_bias=self.edge_bias, name="linear_gate_edge")(edges)
        edge_update = self.activation(edge_update)
        if self.config.update_edge_features:
            edges = _residual(edge_update, edges)

        if self.config.weighting == "linear":
            gating = edge_update
        elif self.config.weighting == "softmax":
            gating = jax.nn.softmax(edge_update, axis=-2)
        else:
            raise ValueError(f"Unknown weighting option for gating: {self.config.weighting}")
        if mask is not None:
            gating *= mask[..., None]


        messages = hk.Linear(self.edge_dim, name="linear_message_downmap")(nodes_snd) # [batch x sender x features]
        messages = gating * messages[..., None, :, :]
        nodes_update = self.aggfunc(messages, axis=-2)
        nodes_update = hk.Linear(self.node_dim, name="linear_message_upmap")(nodes_update)
        nodes_rec = hk.Linear(self.node_dim, name="linear_nodes")(nodes_rec)
        nodes_rec = self.activation(nodes_rec + nodes_update)
        return nodes_rec, edges


class DenseGNN(hk.Module):
    def __init__(self, config: DenseGNNConfig, mlp_config: MLPConfig, name=None):
        super().__init__(name=name)
        self.config = config
        self.mlp_config = mlp_config

        self.edge_embedding = None
        self.attention_layers = []
        self.mpnn_layers = []
        self.mlp_layers = []
        if config.edge_embedding_depth > 0:
            self.edge_embedding = MLP([config.edge_embedding_width]*config.edge_embedding_depth,
                                      mlp_config,
                                      linear_out=False,
                                      name="edge_embedding")

        for n in range(config.n_iterations):
            if config.attention:
                self.attention_layers.append(Attention(self.config.attention.attention_dim,
                                                       self.config.attention.n_heads,
                                                       self.config.attention.use_residual,
                                                       layer_norm=self.config.attention.use_layer_norm,
                                                       output_linear=False,
                                                       name=f"attention_{n}"))
            if config.message_passing:
                self.mpnn_layers.append(MessagePassingLayer(self.config.message_passing,
                                                            use_edge_bias=self.config.use_edge_bias,
                                                            name=f"mpnn_{n}"))

        if self.config.final_mpnn:
            self.mpnn_layers.append(MessagePassingLayer(self.config.message_passing,
                                                        use_edge_bias=self.config.use_edge_bias,
                                                        name=f"mpnn_{config.n_iterations + 1}"))

    def __call__(self,
                 nodes_rec: jax.Array,
                 nodes_snd: Optional[jax.Array] = None,
                 edges: Optional[jax.Array] = None,
                 mask=None):
        if (edges is not None) and self.edge_embedding:
            edges = self.edge_embedding(edges)
        if nodes_snd is None:
            nodes_snd = nodes_rec

        edge_features_for_attention = edges if self.config.use_edges_in_attention else None
        for n in range(self.config.n_iterations):
            # Self-attention across nodes
            if self.attention_layers:
                nodes_rec = self.attention_layers[n](nodes_rec, nodes_snd, edge_features_for_attention, mask)

            # Message passing
            if self.mpnn_layers:
                nodes_rec, edges = self.mpnn_layers[n](nodes_rec, nodes_snd, edges, mask)

            # MLP on nodes
            if self.config.mlp_depth:
                nodes_rec = MLP([nodes_rec.shape[-1]] * self.config.mlp_depth,
                                self.mlp_config,
                                residual=True,
                                name=f"mlp_{n}")(nodes_rec)

        if self.config.final_mpnn:
                nodes_rec, edges = self.mpnn_layers[-1](nodes_rec, nodes_snd, edges, mask)
        return nodes_rec, edges

