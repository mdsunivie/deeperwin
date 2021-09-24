### Partially taken from https://github.com/deepmind/ferminet/tree/jax

from typing import Optional, Mapping, Union
import jax.numpy as jnp

from deeperwin.kfac_ferminet_alpha import curvature_blocks as blocks
from deeperwin.kfac_ferminet_alpha import layers_and_loss_tags as tags
from deeperwin.kfac_ferminet_alpha import utils

repeated_dense_tag = tags.LayerTag("repeated_dense_tag", 1, 1)


def register_repeated_dense(y, x, w, b):
  if b is None:
    return repeated_dense_tag.bind(y, x, w)
  return repeated_dense_tag.bind(y, x, w, b)



class RepeatedDenseBlock(blocks.DenseTwoKroneckerFactored):
  """Dense block that is repeated."""

  def compute_extra_scale(self) -> Optional[jnp.ndarray]:
    (x_shape,) = self.inputs_shapes
    return utils.product(x_shape) // (x_shape[0] * x_shape[-1])

  def update_curvature_matrix_estimate(
      self,
      info: Mapping[str, blocks._Arrays],  # pylint: disable=protected-access
      batch_size: int,
      ema_old: Union[float, jnp.ndarray],
      ema_new: Union[float, jnp.ndarray],
      pmap_axis_name: str
  ) -> None:
    info = dict(**info)
    (x,), (dy,) = info["inputs"], info["outputs_tangent"]
    assert x.shape[0] == batch_size
    info["inputs"] = (x.reshape([-1, x.shape[-1]]),)
    info["outputs_tangent"] = (dy.reshape([-1, dy.shape[-1]]),)
    super().update_curvature_matrix_estimate(info, x.size // x.shape[-1],
                                             ema_old, ema_new, pmap_axis_name)


blocks.set_default_tag_to_block("repeated_dense_tag", RepeatedDenseBlock)