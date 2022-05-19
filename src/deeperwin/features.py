import numpy as np
from deeperwin.configuration import PhysicalConfig
from deeperwin.orbitals import get_hartree_fock_solution, get_p_orbital_indices_per_atom


def get_degenerate_subspaces(U, eigvals, tol=1e-6):
    eigvals = eigvals / (np.max(eigvals) + tol**2)
    subspaces = [[U[:,0]]]
    for i in range(1,3):
        if np.abs(eigvals[i] - eigvals[i-1]) < tol:
            subspaces[-1].append(U[:,i])
        else:
            subspaces.append([U[:,i]])
    return [np.stack(s, axis=-1) for s in subspaces]

def align_2D_subspace_along_ref_axes(U_sub, U_ref, tol=1e-6):
    subspace_coeffs = U_ref.T @ U_sub

    for i, coeffs in enumerate(subspace_coeffs):
        if np.linalg.norm(coeffs) > tol:
            ax1 = U_sub @ coeffs
            coeffs_norm = np.array([-coeffs[1], coeffs[0]])
            ax2 = U_sub @ coeffs_norm
            return np.stack([ax1, ax2], axis=-1)
    raise AssertionError("Provided axes did not have significant overlap with any of the reference axes.")

def fix_degenerate_directions(U, eigvals, U_ref=None, tol=1e-6):
    deg_subspaces = get_degenerate_subspaces(U, eigvals, tol)
    if (len(deg_subspaces) == 3) or (U_ref is None): # eigenvalues are not degenerate or there is no reference to use => keep as is
        return np.concatenate([x for x in deg_subspaces], axis=-1)

    assert len(deg_subspaces) > 1, "There should not be a fully degenerate subspace when a reference axis has already been established"

    aligned_axes = []
    for U_sub in deg_subspaces:
        if U_sub.shape[1] == 1:
            aligned_axes.append(U_sub)
        else:
            U_sub = align_2D_subspace_along_ref_axes(U_sub, U_ref, tol)
            aligned_axes.append(U_sub)
    return np.concatenate(aligned_axes, axis=-1)

def build_local_rotation_matrices(phys_config: PhysicalConfig, tol=1e-6):
    """
    Builds a rotation matrix for each atom, defining a local coordinate system.

    Args:
        phys_config:
        tol: Numerical tolerance to detect degeneracies

    Returns:
        rot_matrices: [n_ions x 3 x 3] np.array
    """

    # minimal basis, but add p-type functions for hydrogen
    basis_set = {Z: 'STO-3G' if Z != 1 else "6-31G**" for Z in phys_config.Z}
    atomic_orbitals, hf = get_hartree_fock_solution(phys_config, basis_set)
    mo_coeffs = hf.mo_coeff
    n_occ = hf.mo_occ
    if len(mo_coeffs) != 2:
        mo_coeffs = [mo_coeffs, mo_coeffs]
        n_occ = [n_occ/2, n_occ/2]

    density_matrix = np.zeros([len(atomic_orbitals), len(atomic_orbitals)])
    for spin in range(2):
        mo_occ = mo_coeffs[spin][:, n_occ[spin] > 0]
        density_matrix += mo_occ @ mo_occ.T

    R_ions = np.array(phys_config.R)
    center_of_mass = np.sum(R_ions * np.array(phys_config.Z)[:,None], axis=0) / np.sum(phys_config.Z)
    p_orb_indices = get_p_orbital_indices_per_atom(atomic_orbitals)

    U_ref = None
    rot_matrices = []
    for ind_atom, (ao_ind, R) in enumerate(zip(p_orb_indices, R_ions)):
        submatrix = density_matrix[ao_ind, :][:, ao_ind]
        eigvals, U = np.linalg.eigh(submatrix)

        # Fix rotation of eigenvectors in degenerate subspaces
        U = fix_degenerate_directions(U, eigvals, U_ref, tol)
        if U_ref is None:
            U_ref = U

        # Fix sign of eigenvectors
        ref_axes = np.concatenate([(center_of_mass - R)[:,None], U_ref], axis=-1)
        for i in range(3):
            for ax_ref in ref_axes.T:
                dot_prod = np.dot(U[:,i], ax_ref)
                if np.abs(dot_prod) > tol:
                    U[:,i] *= np.sign(dot_prod)
                    break
        rot_matrices.append(U.T)
    return np.stack(rot_matrices, axis=0)
