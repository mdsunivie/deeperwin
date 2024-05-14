"""
Computation of local energies and forces.
"""
import functools
import jax
import jax.scipy
import numpy as np
from jax import numpy as jnp
from deeperwin.configuration import ForceEvaluationConfig
from deeperwin.utils.periodic import project_into_first_unit_cell
from deeperwin.utils.utils import get_el_ion_distance_matrix, get_full_distance_matrix, get_distance_matrix
from deeperwin.utils.periodic import LatticeParams

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


def real_space_ewald(diff, lat_vectors, gamma):
    """
    Real-space Ewald potential between charges seperated by separation.
    This code here is strongly inspired by the implementation of FermiNet (Copyright 2020 DeepMind Technologies Limited.)
    References: https://github.com/deepmind/ferminet/blob/main/ferminet/pbc/hamiltonian.py
    """
    displacements = jnp.linalg.norm(
        diff - lat_vectors, axis=-1)  # |r - R|
    return jnp.sum(
        jax.scipy.special.erfc(gamma**0.5 * displacements) / displacements)

def recp_space_ewald(diff, rec_vectors, rec_vec_weights):
    """
    Returns reciprocal-space Ewald potential between charges.
    This code here is strongly inspired by the implementation of FermiNet (Copyright 2020 DeepMind Technologies Limited.)
    References: https://github.com/deepmind/ferminet/blob/main/ferminet/pbc/hamiltonian.py
    """
    return jnp.sum(jnp.exp(1.0j * jnp.dot(rec_vectors, diff)) * rec_vec_weights)

def ewald_sum(diff, volume, rec_vectors, rec_vec_weights, gamma, lat_vectors):
    """
    Evaluates combined real and reciprocal space Ewald potential.
    This code here is strongly inspired by the implementation of FermiNet (Copyright 2020 DeepMind Technologies Limited.)
    References: https://github.com/deepmind/ferminet/blob/main/ferminet/pbc/hamiltonian.py
    """
    return (real_space_ewald(diff, lat_vectors, gamma)
            + recp_space_ewald(diff, rec_vectors, rec_vec_weights)
            - jnp.pi / (volume * gamma))


def get_ewald_el_ion_potential_energy(diff_el_ion, Z, volume, rec_vectors, rec_vec_weights, gamma, lat_vectors,
                                      madelung_const):
    """Evaluates periodic atom-electron potential."""

    def _ewald_sum(diff):
        return ewald_sum(diff, volume, rec_vectors, rec_vec_weights, gamma, lat_vectors)

    batch_ewald_sum = jax.vmap(_ewald_sum, in_axes=(0,))
    n_el = diff_el_ion.shape[0]
    diff_el_ion = jnp.reshape(diff_el_ion, [-1, 3])  # flatten el x ion axis
    # calculate potential for each ae pair
    ewald = batch_ewald_sum(diff_el_ion) - madelung_const
    return jnp.sum(-jnp.tile(Z, n_el) * ewald) # TODO Check if tile is necessary


def get_ewald_el_el_potential_energy(diff_el_el, volume, rec_vectors, rec_vectors_weights, gamma, lat_vectors, madelung_const,
                                      include_heg_background):
    """Evaluates periodic electron-electron potential."""
    n_el = diff_el_el.shape[0]
    assert diff_el_el.shape[-3] == diff_el_el.shape[-2], "Difference matrix must be square"
    diff_el_el = jnp.reshape(diff_el_el, [-1, 3])

    def _ewald_sum(diff):
        return ewald_sum(diff, volume, rec_vectors, rec_vectors_weights, gamma, lat_vectors)

    batch_ewald_sum = jax.vmap(_ewald_sum, in_axes=(0,))

    if include_heg_background:
        ewald = batch_ewald_sum(diff_el_el)
    else:
        ewald = batch_ewald_sum(diff_el_el) - madelung_const
    ewald = jnp.reshape(ewald, [n_el, n_el])
    ewald = ewald.at[jnp.diag_indices(n_el)].set(0.0)
    if include_heg_background:
        return 0.5 * jnp.sum(ewald) + 0.5 * n_el * madelung_const
    else:
        return 0.5 * jnp.sum(ewald)

def get_ewald_ion_ion_potential_energy(R, Z, volume, rec_vectors, rec_vectors_weights, gamma, lat_vectors, madelung_const):
    # Atom-atom potential
    n_ions = R.shape[0]
    if n_ions > 1:
        diff_ion_ion, _ = get_distance_matrix(R, full=True)
        def _ewald_sum(diff):
            return ewald_sum(diff, volume, rec_vectors, rec_vectors_weights, gamma, lat_vectors)

        batch_ewald_sum = jax.vmap(_ewald_sum, in_axes=(0,))

        diff_ion_ion = jnp.reshape(diff_ion_ion, [-1, 3])
        charge_matrix = (jnp.expand_dims(Z, -1) * jnp.expand_dims(Z, -2)).flatten()
        ewald = batch_ewald_sum(diff_ion_ion) - madelung_const
        ewald = jnp.reshape(ewald, [n_ions, n_ions])
        ewald = ewald.at[jnp.diag_indices(n_ions)].set(0.0)
        ewald = ewald.flatten()
        ion_ion_potential = 0.5 * jnp.sum(charge_matrix * ewald)
    else:
        ion_ion_potential = 0.0

    return ion_ion_potential

def get_ewald_potential_energy(r, R, Z, lattice_params: LatticeParams, include_heg_background):
    """
    This code here is strongly inspired by the implementation of FermiNet (Copyright 2020 DeepMind Technologies Limited.)
    References: https://github.com/deepmind/ferminet/blob/main/ferminet/pbc/hamiltonian.py
    """

    diff_el_ion, _ = get_el_ion_distance_matrix(r, R)
    diff_el_el, _ = get_distance_matrix(r)

    diff_el_ion_first_cell = project_into_first_unit_cell(diff_el_ion, lattice_params.lattice, lattice_params.rec / (2 * np.pi))
    diff_el_el_first_cell = project_into_first_unit_cell(diff_el_el, lattice_params.lattice, lattice_params.rec / (2 * np.pi))

    return jnp.real(
        get_ewald_el_ion_potential_energy(diff_el_ion_first_cell,
                                          Z,
                                          lattice_params.volume,
                                          lattice_params.rec_vectors,
                                          lattice_params.rec_vectors_weights,
                                          lattice_params.gamma,
                                          lattice_params.lat_vectors,
                                          lattice_params.madelung_const)
        + get_ewald_el_el_potential_energy(diff_el_el_first_cell,
                                           lattice_params.volume,
                                           lattice_params.rec_vectors,
                                           lattice_params.rec_vectors_weights,
                                           lattice_params.gamma,
                                           lattice_params.lat_vectors,
                                           lattice_params.madelung_const,
                                           include_heg_background)
        + get_ewald_ion_ion_potential_energy(R,
                                             Z,
                                             lattice_params.volume,
                                             lattice_params.rec_vectors,
                                             lattice_params.rec_vectors_weights,
                                             lattice_params.gamma,
                                             lattice_params.lat_vectors,
                                             lattice_params.madelung_const))



def split_func_by_args(f, argnum):
    def f_split(*args, **kwargs):
        return f(*args, **kwargs)[argnum]
    return f_split

def get_kinetic_energy(log_psi_squared_and_phase, trainable_params, spin_state, r, R, Z, fixed_params, complex_wf):
    """This code here is strongly inspired by the implementation of FermiNet (Copyright 2020 DeepMind Technologies Limited.)"""
    phase_log_psi_squared = split_func_by_args(log_psi_squared_and_phase, argnum=0)
    log_psi_squared = split_func_by_args(log_psi_squared_and_phase, argnum=1)

    n_coords = r.shape[-2] * r.shape[-1]
    eye = jnp.eye(n_coords)
    def grad_psi_func(r):
        return jax.grad(log_psi_squared, argnums=3)(trainable_params,
                                                 *spin_state,
                                                 r.reshape([-1, 3]),
                                                 R, Z, fixed_params
                                                 ).flatten()
    grad_value, jvp_func = jax.linearize(grad_psi_func, r.flatten())

    laplacian = 0
    if complex_wf:
        def grad_phase_func(r):
            return jax.grad(phase_log_psi_squared, argnums=3)(trainable_params,
                                                            *spin_state,
                                                            r.reshape([-1, 3]),
                                                            R, Z, fixed_params
                                                            ).flatten()
        grad_phase_value, jvp_phase_func = jax.linearize(grad_phase_func, r.flatten())
        laplacian += 0.5 * jnp.sum(grad_phase_value ** 2) # TODO Check with prefactor since we have log psi squared
        laplacian -= 1.j * jnp.sum(0.5 * grad_value * grad_phase_value) # TODO double check

        def _loop_body(i, accumulator):
            return accumulator + jvp_func(eye[i])[i] + 2*1.j * jvp_phase_func(eye[i])[i]
    else:
        def _loop_body(i, accumulator):
            return accumulator + jvp_func(eye[i])[i]

    laplacian += -0.5 * (0.25 * jnp.sum(grad_value**2) + 0.5 * jax.lax.fori_loop(0, n_coords, _loop_body, 0.0))
    return laplacian

def build_local_energy(log_psi_squared, is_complex=False, is_periodic=False, include_heg_background=False, max_batch_size=None):
    def get_local_energy(trainable_params, spin_state, r, R, Z, fixed_params):
        E_kin = get_kinetic_energy(log_psi_squared, trainable_params, spin_state, r, R, Z, fixed_params, is_complex) # adapt directly the function
        if is_periodic:
            E_pot = get_ewald_potential_energy(r, R, Z, fixed_params["periodic"], include_heg_background)
        else:
            E_pot = get_potential_energy(r, R, Z) # flag to choose different potential energy
        return E_kin + E_pot
    
    # TODO: eventually just replace this with batched_vmap(get_local_energy)
    def get_local_energy_batched(trainable_params, spin_state, r, R, Z, fixed_params):
        batch_size = r.shape[0]
        f_batched = jax.vmap(lambda r_: get_local_energy(trainable_params, spin_state, r_, R, Z, fixed_params))
        if (max_batch_size is None) or (batch_size <= max_batch_size):
            return f_batched(r)
        else:
            n_batches, remainder = divmod(batch_size, max_batch_size)
            assert remainder == 0, f"Batch size {batch_size} must be divisible by max_batch_size {max_batch_size}."
            r_batches = jnp.reshape(r, (n_batches, max_batch_size) + r.shape[-2:])
            E_batched = jax.lax.scan(lambda c, x: (c, f_batched(x)),
                                     None,
                                     r_batches)[1]
            return jnp.reshape(E_batched, (batch_size,))
        
    return get_local_energy_batched

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


def calculate_localization_z(r, super_rec):
    """Calculates the unnamed many-body observable z from the modern theory of
    polarization, related to the localization of the wavefunction."""
    # r shape: batch, nelec, ndim
    return jnp.exp(1j * jnp.sum(r @ super_rec, axis=-2))
