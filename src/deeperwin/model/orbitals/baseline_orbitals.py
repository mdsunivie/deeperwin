
from jax import numpy as jnp
from deeperwin.orbitals import OrbitalParamsType, OrbitalParamsCI, OrbitalParamsHF, OrbitalParamsPeriodicMeanField, evaluate_molecular_orbitals

def _pad_to_full_det(mo_matrix_up, mo_matrix_dn):
    n_up = mo_matrix_up.shape[-1]
    n_dn = mo_matrix_dn.shape[-1]
    batch_shape = mo_matrix_up.shape[:-2]
    mo_matrix_up = jnp.concatenate([mo_matrix_up, jnp.zeros(batch_shape + (n_up, n_dn))], axis=-1)
    mo_matrix_dn = jnp.concatenate([jnp.zeros(batch_shape + (n_dn, n_up)), mo_matrix_dn], axis=-1)
    return mo_matrix_up, mo_matrix_dn

def _eval_molecular_orbitals_up_and_down(diff_el_ion, dist_el_ion, n_up, orbital_params: OrbitalParamsType):
    n_elec = diff_el_ion.shape[-3]
    n_dn = n_elec - n_up
    mo_coeff = orbital_params.mo_coeff[0][:, :n_up], orbital_params.mo_coeff[1][:, :n_dn]

    mo_matrix_up = evaluate_molecular_orbitals(
        diff_el_ion[..., :n_up, :, :],
        dist_el_ion[..., :n_up, :],
        orbital_params.atomic_orbitals,
        mo_coeff[0],
        lattice_vecs=orbital_params.get("lattice_vecs"),
        shift_vecs=orbital_params.get("shift_vecs"),
        k_twist=orbital_params.get("k_twist")
    )
    mo_matrix_dn = evaluate_molecular_orbitals(
        diff_el_ion[..., n_up:, :, :],
        dist_el_ion[..., n_up:, :],
        orbital_params.atomic_orbitals,
        mo_coeff[1],
        lattice_vecs=orbital_params.get("lattice_vecs"),
        shift_vecs=orbital_params.get("shift_vecs"),
        k_twist=orbital_params.get("k_twist")
    )
    return mo_matrix_up, mo_matrix_dn


def get_slater_matrices_HF(diff_el_ion,
                            dist_el_ion,
                            n_up: int,
                            orbital_params: OrbitalParamsType,
                            determinant_schema):
    mo_matrix_up, mo_matrix_dn = _eval_molecular_orbitals_up_and_down(diff_el_ion, dist_el_ion, n_up, orbital_params)
    mo_matrix_up = mo_matrix_up[..., None, :, :]  # Add dummy det axis
    mo_matrix_dn = mo_matrix_dn[..., None, :, :]
    if determinant_schema == 'full_det' or determinant_schema == "restricted_closed_shell":
        mo_matrix_up, mo_matrix_dn = _pad_to_full_det(mo_matrix_up, mo_matrix_dn)
    return mo_matrix_up, mo_matrix_dn

def get_slater_matrices_CI(diff_el_ion,
                            dist_el_ion,
                            n_up: int,
                            orbital_params: OrbitalParamsType,
                            determinant_schema):
    mo_matrix_up, mo_matrix_dn = _eval_molecular_orbitals_up_and_down(diff_el_ion, dist_el_ion, n_up, orbital_params)
    n_dets, n_up = orbital_params.idx_orbitals[0].shape
    # 1) Select orbitals for each determinant => [(batch) x n_el x n_det x n_orb]
    # 2) Move determinant axis forward => [(batch) x n_det x n_el x n_orb]
    mo_matrix_up = jnp.moveaxis(mo_matrix_up[..., orbital_params.idx_orbitals[0]], -2, -3)
    mo_matrix_dn = jnp.moveaxis(mo_matrix_dn[..., orbital_params.idx_orbitals[1]], -2, -3)

    if determinant_schema == 'full_det' or determinant_schema == "restricted_closed_shell":
        mo_matrix_up, mo_matrix_dn = _pad_to_full_det(mo_matrix_up, mo_matrix_dn)

    # CI weights need to go somewhere; could also multiply onto mo_dn, should yield same results
    ci_weights = orbital_params.ci_weights[:, None, None]
    ci_weights_up = jnp.abs(ci_weights)**(1/n_up)

    # adjust sign of first col to match det sign
    ci_weights_up *= jnp.concatenate([jnp.sign(ci_weights), jnp.ones([n_dets, 1, mo_matrix_up.shape[-1]-1])], axis=-1)
    mo_matrix_up *= ci_weights_up
    return mo_matrix_up, mo_matrix_dn
    

def get_baseline_slater_matrices(diff_el_ion, 
                                 dist_el_ion,
                                 n_up: int,
                                 orbital_params: OrbitalParamsType, 
                                 determinant_schema):
    """
    Utility function to extract Slater matrices from a baseline CASSCF calculation
    """
    if isinstance(orbital_params, OrbitalParamsHF):
        return get_slater_matrices_HF(diff_el_ion, dist_el_ion, n_up, orbital_params, determinant_schema)
    elif isinstance(orbital_params, OrbitalParamsPeriodicMeanField):
        return get_slater_matrices_HF(diff_el_ion, dist_el_ion, n_up, orbital_params, determinant_schema)
    elif isinstance(orbital_params, OrbitalParamsCI):
        return get_slater_matrices_CI(diff_el_ion, dist_el_ion, n_up, orbital_params, determinant_schema)
    else:
        raise ValueError(f"Evaluation of slater matrix not implemented for orbital_params type {type(orbital_params)}")


