"""
Computation of local energies and forces.
"""

import logging

import jax
import numpy as np
from jax import numpy as jnp

from deeperwin.configuration import ClippingConfig, ForceEvaluationConfig
from deeperwin.mcmc import MCMCState
from deeperwin.utils import get_el_ion_distance_matrix, get_full_distance_matrix


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


def get_kinetic_energy(log_psi_squared, r, R, Z, trainable_params, fixed_params):
    n_coords = r.shape[-2] * r.shape[-1]
    eye = jnp.eye(n_coords)
    grad_psi_func = lambda r: jax.grad(log_psi_squared, argnums=0)(
        r.reshape([-1, 3]), R, Z, trainable_params, fixed_params
    ).flatten()

    def _loop_body(i, laplacian):
        g_i, G_ii = jax.jvp(grad_psi_func, (r.flatten(),), (eye[i],))
        return laplacian + G_ii[i] + 0.5 * g_i[i] ** 2

    return -0.25 * jax.lax.fori_loop(0, n_coords, _loop_body, 0.0)


def get_kinetic_energy_fwd_diff(log_psi_squared, r, R, Z, trainable_params, fixed_params):
    n_coords = r.shape[-2] * r.shape[-1]
    eye = jnp.eye(n_coords)
    func = lambda r: log_psi_squared(r.reshape([-1, 3]), R, Z, trainable_params, fixed_params)

    def _loop_body(i, laplace):
        g, G = jax.jvp(lambda x_: jax.jvp(func, (x_,), (eye[i],))[1], (r.flatten(),), (eye[i],))
        return laplace + G + 0.5 * g ** 2

    return -0.25 * jax.lax.fori_loop(0, n_coords, _loop_body, 0.0)


def get_potential_energy(r, R, Z):
    _, dist_el_ion = get_el_ion_distance_matrix(r, R)
    E_pot_el_ions = -jnp.sum(Z / dist_el_ion, axis=[-2, -1])
    E_pot_el_el = get_el_el_potential_energy(r)
    E_pot_ion_ion = get_ion_ion_potential_energy(R, Z)
    return E_pot_el_el + E_pot_el_ions + E_pot_ion_ion


def get_local_energy(log_psi_squared, r, R, Z, trainable_params, fixed_params, use_fwd_fwd_hessian=False):
    if use_fwd_fwd_hessian:
        _ekin_batched = jax.vmap(get_kinetic_energy_fwd_diff, in_axes=(None, 0, None, None, None, None))
    else:
        _ekin_batched = jax.vmap(get_kinetic_energy, in_axes=(None, 0, None, None, None, None))

    Ekin = _ekin_batched(log_psi_squared, r, R, Z, trainable_params, fixed_params)
    Epot = get_potential_energy(r, R, Z)
    return Ekin + Epot


def clip_energies(E, center, width, clipping_config: ClippingConfig):
    if clipping_config.name == "hard":
        return jnp.clip(E, center - width, center + width)
    elif clipping_config.name == "tanh":
        return center + jnp.tanh((E - center) / width) * width
    else:
        raise ValueError(f"Unsupported config-value for optimization.clipping.name: {clipping_config.name}")


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


def _calculate_forces_polynomial_fit(diff_el_ion, d_el_ion, Z, R_core, poly_coeffs):
    r"""
    Calculate hellman-feynman forces on ions :math:`Z_m \left(\frac{1}{N} \sum_{k} \sum_i \frac{r^k_i-R_m}{|r^k_i - R_m|^3}\right)`.
    Does not include ion-ion forces.
    """
    # forces = [batch x n_el x n_ions x xyz]
    d_el_ion = d_el_ion[..., np.newaxis]
    forces_outer = diff_el_ion / d_el_ion ** 3
    poly_degree = poly_coeffs.shape[0]
    j = np.arange(1, poly_degree + 1)
    j = np.reshape(j, [-1, 1, 1, 1, 1])
    force_moments = (diff_el_ion / d_el_ion) * (d_el_ion / R_core) ** j
    forces_core = jnp.sum(poly_coeffs * force_moments, axis=0)  # sum over moments
    forces = jnp.where(d_el_ion < R_core, forces_core, forces_outer)
    forces = forces * Z[:, np.newaxis]
    return forces


def calculate_forces(r, R, Z, log_psi_sqr, log_sqr_func, func_params, config: ForceEvaluationConfig, poly_coeffs=None):
    """
    Calculates the forces following closely the work of https://doi.org/10.1103/PhysRevLett.94.036404 by using antithetic sampling and per default fitting a polynomial
    to the force density close to the nuclei.
    """
    # [batch x electron x ion x xyz]

    diff_el_ion, r_el_ion = get_el_ion_distance_matrix(r, R)
    if config.use_polynomial:
        force_function = jax.partial(_calculate_forces_polynomial_fit, R_core=config.R_core, poly_coeffs=poly_coeffs)
    else:
        force_function = jax.partial(_calculate_forces_directly, R_cut=config.R_cut)
    forces = force_function(diff_el_ion, r_el_ion, Z)

    if config.use_antithetic_sampling:
        ind_closest_ion = jnp.argmin(r_el_ion, axis=-1)
        d_closest_ion = jnp.min(r_el_ion, axis=-1, keepdims=True)
        diff_to_closest_ion = jax.vmap(jax.vmap(lambda x, ind: x[ind]))(diff_el_ion,
                                                                        ind_closest_ion)  # vmap over number of electrons and batch-size
        is_core_electron = d_closest_ion < config.R_core

        r_mirrored = jnp.where(is_core_electron, r - 2 * diff_to_closest_ion, r)
        diff_el_ion_mirrored, r_el_ion_mirrored = get_el_ion_distance_matrix(r_mirrored, R)
        mirrored_weight = jnp.exp(log_sqr_func(r_mirrored, R, Z, *func_params) - log_psi_sqr)

        forces_mirrored = force_function(diff_el_ion_mirrored, r_el_ion_mirrored, Z)
        forces = (forces + forces_mirrored * mirrored_weight[:, None, None, None]) * 0.5

    force = jnp.mean(jnp.sum(forces, axis=1), axis=0)  # sum over electrons, average over batch
    return force + _calculate_ion_ion_forces(R, Z)


def log_mcmc_debug_info(epoch_nr, mcmc_old: MCMCState, mcmc_new: MCMCState):
    dist = jnp.linalg.norm(mcmc_new.r - mcmc_old.r, axis=-1)
    logging.debug(
        f"Epoch {epoch_nr:>4d} MCMC: scale = {mcmc_new.stepsize:.4f}, stepsize mean={jnp.mean(dist):.3f}, std={jnp.std(dist):.3f}"
    )


if __name__ == "__main__":
    pass
