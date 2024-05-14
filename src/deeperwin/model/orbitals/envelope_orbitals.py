from typing import Tuple

import haiku as hk
import jax
from jax import numpy as jnp

from deeperwin.configuration import EnvelopeOrbitalsConfig, MLPConfig
from deeperwin.model import MLP
from kfac_jax import register_scale_and_shift


class EnvelopeOrbitals(hk.Module):
    """
    Class representing a set of enveloped (spin) orbitals
    """
    def __init__(
        self,
        config: EnvelopeOrbitalsConfig,
        mlp_config: MLPConfig,
        n_dets: int,
        full_det: bool,
        determinant_schema: str,
        complex_wf: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.mlp_config = mlp_config
        self.n_dets = n_dets
        self.full_det = full_det
        self.complex_wf = complex_wf
        self.determinant_schema = determinant_schema

        self._alpha_up_init = jnp.ones
        self._alpha_dn_init = jnp.ones
        self._weights_up_init = jnp.ones
        self._weights_dn_init = jnp.ones


    def __call__(self, dist_el_ion: jax.Array, emb_el: jax.Array, n_ions: int, n_up: int, n_dn: int):
        (e_up, e_dn) = _determine_elec_idxs(n_up, n_dn, self.determinant_schema)

        n_up_orbitals, n_dn_orbitals = _determine_n_output_orbitals(n_up, n_dn, self.determinant_schema)

        n_up_tot = self.n_dets * n_up_orbitals
        # Backflow factor
        bf_up = MLP(
            self.config.n_hidden + [n_up_tot * (2 if self.complex_wf else 1)],
            self.mlp_config,
            output_bias=self.config.use_bias,
            linear_out=True,
            name="bf_up",
        )(emb_el[..., :e_up, :]) # output-shape: [batch x n_up x (n_dets * n_up_orb)] or [batch x n_el x (n_dets * n_up_orb)]
        if self.complex_wf:
            bf_up = bf_up[..., :, :n_up_tot] + 1.0j * bf_up[..., :, n_up_tot:]

        # output-shape: [batch x n_dets x n_up x n_up_orb] or [batch x n_dets x n_el x n_up_orb]
        bf_up = jnp.swapaxes(bf_up.reshape(bf_up.shape[:-1] + (self.n_dets, n_up_orbitals)), -3, -2)

        n_dn_tot = self.n_dets * n_dn_orbitals
        bf_dn = MLP(
            self.config.n_hidden + [n_dn_tot * (2 if self.complex_wf else 1)],
            self.mlp_config,
            output_bias=self.config.use_bias,
            linear_out=True,
            name="bf_dn",
        )(emb_el[..., -e_dn:, :]) # output-shape: [batch x n_down x (n_dets * n_down_orb)] or [batch x n_el x (n_dets * n_up_orb)]
        if self.complex_wf:
            bf_dn = bf_dn[..., :, :n_dn_tot] + 1.0j * bf_dn[..., :, n_dn_tot:]

        # output-shape: [batch x n_dets x n_down x n_down_orb] or [batch x n_dets x n_el x n_up_orb]
        bf_dn = jnp.swapaxes(bf_dn.reshape(bf_dn.shape[:-1] + (self.n_dets, n_dn_orbitals)), -3, -2)

        # Envelopes
        if self.config.envelope_type == "isotropic_exp":
            mo_matrix_up, mo_matrix_dn = self._envelope_isotropic(dist_el_ion, n_ions, e_up, e_dn)
        else:
            raise ValueError(f"Unknown envelope type: {self.config.envelope_type}")

        mo_matrix_up *= bf_up
        mo_matrix_dn *= bf_dn

        # reshape the matrices when using restricted closed shell
        if self.determinant_schema == "restricted_closed_shell":
            n_elec = mo_matrix_dn.shape[-2] // 2
            _mo_matrix_up = jnp.concatenate([mo_matrix_up[..., :n_elec, :], mo_matrix_dn[..., :n_elec, :]], axis=-1)
            _mo_matrix_dn = jnp.concatenate([mo_matrix_dn[..., n_elec:, :], mo_matrix_up[..., n_elec:, :]], axis=-1)
            mo_matrix_up = _mo_matrix_up
            mo_matrix_dn = _mo_matrix_dn

        return mo_matrix_up, mo_matrix_dn

    def _envelope_isotropic(self, el_ion_dist: jax.Array, n_ions: int, n_up: int, n_dn: int):
        n_up_orbitals, n_dn_orbitals = _determine_n_output_orbitals(n_up, n_dn, self.determinant_schema)

        shape_up = [n_ions, self.n_dets * n_up_orbitals]
        shape_dn = [n_ions, self.n_dets * n_dn_orbitals]
        alpha_up = hk.get_parameter("alpha_up", shape_up, init=self._alpha_up_init, dtype=el_ion_dist.dtype)
        alpha_dn = hk.get_parameter("alpha_dn", shape_dn, init=self._alpha_dn_init, dtype=el_ion_dist.dtype)
        weights_up = hk.get_parameter("weights_up", shape_up, init=self._weights_up_init, dtype=el_ion_dist.dtype)
        weights_dn = hk.get_parameter("weights_dn", shape_dn, init=self._weights_dn_init, dtype=el_ion_dist.dtype)

        d_up = el_ion_dist[..., :n_up, :, jnp.newaxis]  # [batch x el_up x ion x 1 (det*orb)]
        d_dn = el_ion_dist[..., -n_dn:, :, jnp.newaxis]

        # computing exponentials
        exp_up = jax.nn.softplus(alpha_up) * d_up  # [batch x el_up x ion x (det*orb)]
        exp_dn = jax.nn.softplus(alpha_dn) * d_dn
        exp_up = register_scale_and_shift(exp_up, d_up, scale=alpha_up, shift=None)
        exp_dn = register_scale_and_shift(exp_dn, d_dn, scale=alpha_dn, shift=None)

        orb_up = jnp.sum(weights_up * (jnp.exp(-exp_up)), axis=-2)
        orb_dn = jnp.sum(weights_dn * (jnp.exp(-exp_dn)), axis=-2)

        # cut down on electrons agains
        orb_up = orb_up[..., :n_up, :]
        orb_dn = orb_dn[..., :n_dn, :]

        # rearrange
        orb_up = jnp.reshape(orb_up, orb_up.shape[:-1] + (self.n_dets, n_up_orbitals))
        orb_dn = jnp.reshape(orb_dn, orb_dn.shape[:-1] + (self.n_dets, n_dn_orbitals))
        orb_up = jnp.moveaxis(orb_up, -2, -3) # [batch x det x el_up x orb]
        orb_dn = jnp.moveaxis(orb_dn, -2, -3)
        return orb_up, orb_dn


def _determine_n_output_orbitals(
    n_up: int,
    n_dn: int,
    determinant_schema: str
) -> Tuple[int]:
    """
    Function to determine the dimensionality of the outputted up & down MO matrix blocks given
    the number of up & dn electrons.
    """
    if determinant_schema == "full_det":
        return (n_up + n_dn, n_up + n_dn)
    elif determinant_schema == "block_diag":
        return (n_up, n_dn)
    elif determinant_schema == "restricted_closed_shell":
        return (n_up, n_dn)


def _determine_elec_idxs(
    n_up: int,
    n_dn: int,
    determinant_schema: str
) -> Tuple[int]:
    """
    Function to determine indices for indexing electron embeddings when inferring the
    MO matrix blocks
    """
    if determinant_schema == "full_det":
        return (n_up, n_dn)
    elif determinant_schema == "block_diag":
        return (n_up, n_dn)
    elif determinant_schema == "restricted_closed_shell":
        return (n_up + n_dn, n_up + n_dn)
