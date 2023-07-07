from typing import Tuple

import haiku as hk
from jax import numpy as jnp

from deeperwin.configuration import BaselineOrbitalsConfig, MLPConfig
from deeperwin.model import DiffAndDistances, Embeddings, MLP

from deeperwin.orbitals import OrbitalParams, evaluate_molecular_orbitals


def _determine_baseline_output_shapes(
    n_up: int,
    n_dn: int,
    determinant_schema: str
) -> Tuple[Tuple[int], Tuple[int]]:
    """
    Function to determine the dimensionality of the outputted up & down MO matrix blocks for
    baseline orbitals.

    Returns one tuple containing the amount of up & down orbitals (for each electron) &
    one tuple containing the upper & lower indices over which electrons to evaluate the orbital functions, e.g.:
    orbs = fn(emb_el[..., :upper, :]) or orbs = fn(emb_el[..., lower:, :])
    """
    if determinant_schema == "full_det":
        return (n_up + n_dn, n_up + n_dn), (n_up, n_up)
    elif determinant_schema == "block_diag":
        return (n_up, n_dn), (n_up, n_up)
    elif determinant_schema == "restricted_closed_shell":
        return (n_up + n_dn, n_up + n_dn), (n_up, n_up)


def get_baseline_slater_matrices(diff_el_ion, dist_el_ion, orbital_params: OrbitalParams, determinant_schema):
    """
    Utility function to extract Slater matrices from a baseline CASSCF calculation
    """
    n_dets, n_up = orbital_params.idx_orbitals[0].shape
    n_dn = orbital_params.idx_orbitals[1].shape[1]

    mo_matrix_up = evaluate_molecular_orbitals(
        diff_el_ion[..., :n_up, :, :],
        dist_el_ion[..., :n_up, :],
        orbital_params.atomic_orbitals,
        orbital_params.mo_coeff[0],
        orbital_params.mo_cusp_params[0]
    )
    mo_matrix_dn = evaluate_molecular_orbitals(
        diff_el_ion[..., n_up:, :, :],
        dist_el_ion[..., n_up:, :],
        orbital_params.atomic_orbitals,
        orbital_params.mo_coeff[1],
        orbital_params.mo_cusp_params[1]
    )
    # 1) Select orbitals for each determinant => [(batch) x n_el x n_det x n_orb]
    # 2) Move determinant axis forward => [(batch) x n_det x n_el x n_orb]
    mo_matrix_up = jnp.moveaxis(mo_matrix_up[..., orbital_params.idx_orbitals[0]], -2, -3)
    mo_matrix_dn = jnp.moveaxis(mo_matrix_dn[..., orbital_params.idx_orbitals[1]], -2, -3)

    if determinant_schema == 'full_det' or determinant_schema == "restricted_closed_shell":
        batch_shape = mo_matrix_up.shape[:-2]
        mo_matrix_up = jnp.concatenate([mo_matrix_up, jnp.zeros(batch_shape + (n_up, n_dn))], axis=-1)
        mo_matrix_dn = jnp.concatenate([jnp.zeros(batch_shape + (n_dn, n_up)), mo_matrix_dn], axis=-1)

    # CI weights need to go somewhere; could also multiply onto mo_dn, should yield same results
    ci_weights = orbital_params.ci_weights[:, None, None]
    ci_weights_up = jnp.abs(ci_weights)**(1/n_up)

    # adjust sign of first col to match det sign
    ci_weights_up *= jnp.concatenate([jnp.sign(ci_weights), jnp.ones([n_dets, 1, mo_matrix_up.shape[-1]-1])], axis=-1)
    mo_matrix_up *= ci_weights_up
    return mo_matrix_up, mo_matrix_dn


class BaselineOrbitals(hk.Module):
    """
    Base class representing a set of baseline (spin) orbitals, obtained from for example CASSCF,
    modified using a backflow factor (either through shifting input coordinates or adjusting orbitals)
    """
    def __init__(
        self,
        config: BaselineOrbitalsConfig,
        mlp_config: MLPConfig,
        n_dets: int,
        full_det: bool,
        output_shapes: Tuple[Tuple[int], Tuple[int]]
    ) -> None:
        super().__init__()
        self.config = config
        self.mlp_config = mlp_config
        self.n_dets = n_dets
        self.full_det =  full_det
        (self.n_up_orbitals, self.n_dn_orbitals), (self.e_up, self.e_dn) = output_shapes

    def __call__(
        self,
        diff_dist: DiffAndDistances,
        embeddings: Embeddings,
        orbital_params: OrbitalParams
    ):
        if self.config.use_bf_shift:
            diff_dist = self.apply_backflow_shift(diff_dist, embeddings)

        # Evaluate atomic and molecular orbitals for every determinant
        mo_matrix_up, mo_matrix_dn = get_baseline_slater_matrices(diff_dist.diff_el_ion,
                                                                  diff_dist.dist_el_ion,
                                                                  orbital_params,
                                                                  self.full_det)
        if self.config.use_bf_factor:
            bf_up = MLP(
                self.config.n_hidden_bf_factor + [self.n_dets * self.n_up_orbitals],
                self.mlp_config,
                output_bias=self.config.use_bf_factor_bias,
                linear_out=True,
                name="bf_up",
            )(embeddings.el[..., :self.e_up, :])
            bf_dn = MLP(
                self.config.n_hidden_bf_factor + [self.n_dets * self.n_dn_orbitals],
                self.mlp_config,
                output_bias=self.config.use_bf_factor_bias,
                linear_out=True,
                name="bf_dn",
            )(embeddings.el[..., self.e_dn:, :])
            # output-shape: [batch x n_up x (n_dets * n_up_orb)]
            bf_up = jnp.swapaxes(bf_up.reshape(bf_up.shape[:-1] + (self.n_dets, self.n_up_orbitals)), -3, -2)
            bf_dn = jnp.swapaxes(bf_dn.reshape(bf_dn.shape[:-1] + (self.n_dets, self.n_dn_orbitals)), -3, -2)
            # output-shape: [batch x n_dets x n_up x n_up_orb]

            mo_matrix_up *= bf_up
            mo_matrix_dn *= bf_dn
        return mo_matrix_up, mo_matrix_dn

    def apply_backflow_shift(
        self,
        diff_dist: DiffAndDistances,
        embeddings: Embeddings,
    ):
        shift_towards_electrons = self._calc_shift(embeddings.el, embeddings.el_el,
                                                   diff_dist.diff_el_el, diff_dist.dist_el_el, name="el")
        shift_towards_ions = self._calc_shift(embeddings.el, embeddings.el_ion, diff_dist.diff_el_ion,
                                              diff_dist.dist_el_ion, name="ion")

        # TODO: MLP for decay length-scale can output 0 and cause NaN
        decay_lengthscale = hk.get_parameter("bf_shift_decay_scale", [1], init=jnp.ones)
        decay_lengthscale = decay_lengthscale / MLP([decay_lengthscale.shape[-1]], self.mlp_config, name=f"bf_shift_decay_scaling")(embeddings.ion)
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
