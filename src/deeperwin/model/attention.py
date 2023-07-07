import haiku as hk
import jax
from jax import numpy as jnp
from typing import Optional


class Attention(hk.Module):
    def __init__(
        self,
        attention_dim: int,
        n_heads: int,
        residual: bool,
        attention_value_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        layer_norm: bool = False,
        output_linear: bool = True,
        use_residual_before_lin: bool = False,
        name: Optional[str] = None,
    ):
        self._attention_value_dim = attention_value_dim or attention_dim
        self._attention_dim = attention_dim
        self._output_linear = output_linear
        self._n_heads = n_heads
        self._residual = residual
        self._use_residual_before_lin = use_residual_before_lin
        self._layer_norm = layer_norm
        self.output_dim = output_dim or n_heads * self._attention_value_dim
        super().__init__(name=name)

    def _attention_linear_map(self, x, output_dim, name=None):
        x = hk.Linear(self._n_heads * output_dim, name=name, with_bias=False)(x)
        return x.reshape([*x.shape[:-1], self._n_heads, output_dim])

    def __call__(self, receiver_input, sender=None, edge_features=None, mask=None):
        """

        Args:
            receiver: [batch x n_tokens_rec x features]
            sender: [batch x n_tokens_send x features]; Optional: If None, uses receiver as sender (i.e. fully connected self-attention)
            edge_features: [batch x n_tokens_rec x n_tokens_send x edge_features] Edge features to be added to the attention matrix; Optional

        Returns:

        """
        if self._layer_norm:
            receiver = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(receiver_input)
        else:
            receiver = receiver_input
        if sender is None:
                sender = receiver
        else:
            if self._layer_norm:
                sender = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(sender)

        q = self._attention_linear_map(receiver, self._attention_dim, name="q")
        k = self._attention_linear_map(sender, self._attention_dim, name="k")
        v = self._attention_linear_map(sender, self._attention_value_dim, name="v")
        weights = jnp.einsum("...ihf, ...jhf->...ijh", q, k)
        if edge_features is not None:
            weights += hk.Linear(self._n_heads, with_bias=False, name="edge_weights")(edge_features)
        weights = jax.nn.softmax(weights, axis=-2)
        if mask is not None:
            weights *= mask[..., None]

        output = jnp.einsum("...ijh,...jhv->...ihv", weights, v)
        normalization = 1 / jnp.sqrt(self._attention_dim)
        output = output.reshape([*output.shape[:-2], -1]) * normalization

        if self._use_residual_before_lin:
            if self._residual and (receiver_input.shape == output.shape):
                output += receiver_input

        if self._output_linear:
            output = hk.Linear(self.output_dim, with_bias=False, name="output")(output)

        if not self._use_residual_before_lin:
            if self._residual and (receiver_input.shape == output.shape):
                output += receiver_input

        return output
