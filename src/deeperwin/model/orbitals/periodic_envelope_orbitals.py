from typing import Optional, Tuple, Type

import haiku as hk
import jax
import numpy as np
from jax import numpy as jnp

from deeperwin.configuration import PeriodicEnvelopeOrbitalsConfig, MLPConfig
from deeperwin.model import WavefunctionDefinition, MLP
from kfac_jax import register_scale_and_shift, register_dense
from deeperwin.curvature_tags_and_blocks import register_repeated_dense

class CustomInitializer(hk.initializers.Initializer):
    """
    Initialize the parameter for the periodic envelope all to zero except for the first k-point.
    """
    def __init__(self, kpoint_pos):
        self.kpoint_pos = kpoint_pos

    def __call__(self, shape, dtype) -> jax.Array:
        except_one = jnp.zeros(shape)

        # currently excepting the shape: [M x output]
        except_one = except_one.at[self.kpoint_pos, :].add(1.0)
        return jnp.zeros(shape) + except_one
    
class BlochWaveEnvelope(hk.Module):
    def __init__(self, determinant_schema, name: str = None):
        super().__init__(name)
        self.determinant_schema = determinant_schema

    def _get_bloch(self, r, k_points_occ):
        phase = 1.j * (r @ k_points_occ)
        phase = phase[..., None, :, :] # add dummy determinant axis => [batch x det x el x orb]
        return jnp.exp(phase)

    def __call__(self, diff_el_ion, k_points, k_twist, n_up, n_dn):
        # Select only the first n_el k_points (ie. the occupied ones)
        k_points_up = k_points[0][:, :n_up] - k_twist[:, None]
        k_points_dn = k_points[1][:, :n_dn] - k_twist[:, None]

        diff_el_mean_ion = jnp.mean(diff_el_ion, axis=-2) # mean over all ions
        if self.determinant_schema != "block_diag":
            # concat along orbital axis
            k_points_up = jnp.concatenate([k_points_up, k_points_dn], axis=-1)
            k_points_dn = k_points_up

        mo_up = self._get_bloch(diff_el_mean_ion[..., :n_up, :], k_points_up)
        mo_dn = self._get_bloch(diff_el_mean_ion[..., n_up:, :], k_points_dn)
        return mo_up, mo_dn

class PeriodicEnvelopeOrbitals(hk.Module):
    """
    Class representing a set of enveloped (spin) orbitals
    """
    def __init__(
        self,
        config: PeriodicEnvelopeOrbitalsConfig,
        mlp_config: MLPConfig,
        n_dets: int,
        determinant_schema: str
    ) -> None:
        super().__init__()
        del mlp_config # not used yet

        self.config = config
        self.n_dets = n_dets
        self.determinant_schema = determinant_schema
        assert determinant_schema == "full_det", "PeriodicEnvelopeOrbitals currently only supports full_det determinant_schema"

        self._nu_up_init = CustomInitializer(kpoint_pos=0)
        self._nu_dn_init = CustomInitializer(kpoint_pos=0)


    def __call__(self, diff_el_ion, kpoints, n_up: int, n_dn: int):
        n_el = n_up + n_dn
        diff_el_mean_ion = jnp.mean(diff_el_ion, axis=-2) # mean over all ions

        # diff_el_mean_ion: [bs x n_el x 3]; kpoints: [M x 3]
        # phase_coord: [bs x n_el x M]
        phase_coord = 1.j*(diff_el_mean_ion @ kpoints.T) # split into sin and cos
        basis = jnp.exp(phase_coord)
        basis_up = basis[..., :n_up, :]
        basis_dn = basis[..., n_up:, :]

        shape_up = [len(kpoints), n_el * self.n_dets]
        shape_dn = [len(kpoints), n_el * self.n_dets]
        nu_up = hk.get_parameter("nu_up", shape_up, init=self._nu_up_init)
        nu_dn = hk.get_parameter("nu_dn", shape_dn, init=self._nu_dn_init)

        # reduce over M the number of kpoints
        # mo_up: [bs x n_up x n_el*n_dets]

        # mo_up = hk.Linear(output_size=n_el * self.n_dets, with_bias=False, w_init=self._nu_up_init)(basis_up)
        # mo_dn = hk.Linear(output_size=n_el * self.n_dets, with_bias=False, w_init=self._nu_dn_init)(basis_dn)
        mo_up = jnp.dot(basis_up, nu_up)
        mo_dn = jnp.dot(basis_dn, nu_dn)

        #mo_up = register_repeated_dense(mo_up, basis_up, nu_up, None)
        #mo_dn = register_repeated_dense(mo_dn, basis_dn, nu_dn, None)
        #mo_up = register_scale_and_shift(mo_up, basis_up, scale=nu_up, shift=None)
        #mo_dn = register_scale_and_shift(mo_dn, basis_dn, scale=nu_dn, shift=None)

        # mo_up: [bs x n_up x n_dets x n_el (orbitals for full-det)]
        mo_up = jnp.reshape(mo_up, mo_up.shape[:-1] + (self.n_dets, n_el))
        mo_dn = jnp.reshape(mo_dn, mo_dn.shape[:-1] + (self.n_dets, n_el))

        # mo_up: [bs x n_dets x n_up x n_el]
        mo_up = jnp.moveaxis(mo_up, -2, -3)
        mo_dn = jnp.moveaxis(mo_dn, -2, -3)
        return mo_up, mo_dn
