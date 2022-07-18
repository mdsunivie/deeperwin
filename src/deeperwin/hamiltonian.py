"""
Computation of local energies and forces.
"""
import jax
import numpy as np
from jax import numpy as jnp
from deeperwin.configuration import ForceEvaluationConfig
from deeperwin.utils import get_el_ion_distance_matrix, get_full_distance_matrix
import functools

def get_el_el_potential_energy(r_el):
    n_el = r_el.shape[-2]
    eye = jnp.eye(n_el)
    dist_matrix = get_full_distance_matrix(r_el)
    E_pot = jnp.triu(1.0 / (dist_matrix + eye), k=1)  # add eye to diagonal to prevent div/0
    return jnp.sum(E_pot, axis=[-2, -1])


def get_ion_ion_potential_energy(R, Z):
    n_ions = R.shape[-2]
    eye = jnp.eye(n_ions)
    dist_matrix = get_full_distance_matrix(R)
    charge_matrix = jnp.expand_dims(Z, -1) * jnp.expand_dims(Z, -2)
    E_pot = jnp.triu(charge_matrix / (dist_matrix + eye), k=1)  # add eye to diagonal to prevent div/0
    return jnp.sum(E_pot, axis=[-2, -1])


def get_potential_energy(r, R, Z):
    _, dist_el_ion = get_el_ion_distance_matrix(r, R)
    E_pot_el_ions = -jnp.sum(Z / dist_el_ion, axis=[-2, -1])
    E_pot_el_el = get_el_el_potential_energy(r)
    E_pot_ion_ion = get_ion_ion_potential_energy(R, Z)
    return E_pot_el_el + E_pot_el_ions + E_pot_ion_ion


def get_kinetic_energy(log_psi_squared, trainable_params, r, R, Z, fixed_params):
    """This code here is strongly inspired by the implementation of FermiNet (Copyright 2020 DeepMind Technologies Limited.)"""
    n_coords = r.shape[-2] * r.shape[-1]
    eye = jnp.eye(n_coords)
    grad_psi_func = lambda r: jax.grad(log_psi_squared, argnums=1)(trainable_params,
                                                                   r.reshape([-1, 3]),
                                                                   R, Z, fixed_params
                                                                   ).flatten()

    grad_value, jvp_func = jax.linearize(grad_psi_func, r.flatten())
    def _loop_body(i, accumulator):
        return accumulator + jvp_func(eye[i])[i]
    laplacian = 0.25 * jnp.sum(grad_value**2) + 0.5 * jax.lax.fori_loop(0, n_coords, _loop_body, 0.0)
    return -0.5 * laplacian

@functools.partial(jax.vmap, in_axes=(None, None, 0, None, None, None))
def get_local_energy(log_psi_squared, trainable_params, r, R, Z, fixed_params):
    E_kin = get_kinetic_energy(log_psi_squared, trainable_params, r, R, Z, fixed_params)
    E_pot = get_potential_energy(r, R, Z)
    return E_kin + E_pot


def _calculate_forces_directly(diff_el_ion, d_el_ion, Z, R_cut):
    d_el_ion = d_el_ion / jnp.tanh(d_el_ion / R_cut)
    return Z[:, np.newaxis] * diff_el_ion / d_el_ion[..., np.newaxis] ** 3


def _calculate_ion_ion_forces(R, Z):
    EPSILON = 1e-8
    diff = jnp.expand_dims(R, -2) - jnp.expand_dims(R, -3)
    dist = jnp.linalg.norm(diff, axis=-1, keepdims=True)
    Z_matrix = Z[:, np.newaxis] * Z[np.newaxis, :] - jnp.diag(Z ** 2)
    forces = diff * Z_matrix[..., jnp.newaxis] / (
                dist + EPSILON) ** 3  # tiny EPS ensures that the diagonal (self-interaction) has 0 contribution
    return jnp.sum(forces, axis=1)


def calculate_forces(log_sqr_func, params, r, R, Z, fixed_params, log_psi_sqr, config: ForceEvaluationConfig, poly_coeffs=None):
    """
    Calculates the forces following closely the work of https://doi.org/10.1103/PhysRevLett.94.036404 by using antithetic sampling and per default fitting a polynomial
    to the force density close to the nuclei.
    """
    # [batch x electron x ion x xyz]

    diff_el_ion, dist_el_ion = get_el_ion_distance_matrix(r, R)
    forces = _calculate_forces_directly(diff_el_ion, dist_el_ion, Z, config.R_cut)

    if config.use_antithetic_sampling:
        ind_closest_ion = jnp.argmin(dist_el_ion, axis=-1)
        d_closest_ion = jnp.min(dist_el_ion, axis=-1, keepdims=True)
        diff_to_closest_ion = jax.vmap(jax.vmap(lambda x, ind: x[ind]))(diff_el_ion,
                                                                        ind_closest_ion)  # vmap over number of electrons and batch-size
        is_core_electron = d_closest_ion < config.R_core

        r_mirrored = jnp.where(is_core_electron, r - 2 * diff_to_closest_ion, r)
        diff_el_ion_mirrored, r_el_ion_mirrored = get_el_ion_distance_matrix(r_mirrored, R)
        mirrored_weight = jnp.exp(log_sqr_func(params, r_mirrored, R, Z, fixed_params) - log_psi_sqr)

        forces_mirrored = _calculate_forces_directly(diff_el_ion_mirrored, r_el_ion_mirrored, Z, config.R_cut)
        forces = (forces + forces_mirrored * mirrored_weight[:, None, None, None]) * 0.5

    force = jnp.mean(jnp.sum(forces, axis=1), axis=0)  # sum over electrons, average over batch
    return force + _calculate_ion_ion_forces(R, Z)
