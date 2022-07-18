import numpy as np
from deeperwin.configuration import PhysicalConfig
from deeperwin.orbitals import get_hartree_fock_solution, get_p_orbital_indices_per_atom

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
        U = _fix_degenerate_directions(U, eigvals, U_ref, tol)
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

def _get_degenerate_subspaces(U, eigvals, tol=1e-6):
    eigvals = eigvals / (np.max(eigvals) + tol**2)
    subspaces = [[U[:,0]]]
    for i in range(1,3):
        if np.abs(eigvals[i] - eigvals[i-1]) < tol:
            subspaces[-1].append(U[:,i])
        else:
            subspaces.append([U[:,i]])
    return [np.stack(s, axis=-1) for s in subspaces]

def _align_2D_subspace_along_ref_axes(U_sub, U_ref, tol=1e-6):
    subspace_coeffs = U_ref.T @ U_sub

    for i, coeffs in enumerate(subspace_coeffs):
        if np.linalg.norm(coeffs) > tol:
            ax1 = U_sub @ coeffs
            coeffs_norm = np.array([-coeffs[1], coeffs[0]])
            ax2 = U_sub @ coeffs_norm
            return np.stack([ax1, ax2], axis=-1)
    raise AssertionError("Provided axes did not have significant overlap with any of the reference axes.")

def _fix_degenerate_directions(U, eigvals, U_ref=None, tol=1e-6):
    deg_subspaces = _get_degenerate_subspaces(U, eigvals, tol)
    if (len(deg_subspaces) == 3) or (U_ref is None): # eigenvalues are not degenerate or there is no reference to use => keep as is
        return np.concatenate([x for x in deg_subspaces], axis=-1)

    assert len(deg_subspaces) > 1, "There should not be a fully degenerate subspace when a reference axis has already been established"

    aligned_axes = []
    for U_sub in deg_subspaces:
        if U_sub.shape[1] == 1:
            aligned_axes.append(U_sub)
        else:
            U_sub = _align_2D_subspace_along_ref_axes(U_sub, U_ref, tol)
            aligned_axes.append(U_sub)
    return np.concatenate(aligned_axes, axis=-1)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    distortions = np.array([-0.4, -0.01, 0, 0.01, 0.4])
    # distortions = np.zeros(5)
    a_values = 1.0 + distortions
    b_values = 1.0 - distortions

    U_matrices = []
    R_values = []
    for a,b in zip(a_values,b_values):
        R = [[-a, -b, 0],
             [-a, b, 0],
             [a, -b, 0],
             [a, b, 0]]
        phys = PhysicalConfig(R=R, Z=[1,1,1,1], n_electrons=4, n_up=2)
        U = build_local_rotation_matrices(phys)
        R_values.append(np.array(R))
        U_matrices.append(U)

#%%
    arrow_scale = 0.5
    plt.close("all")
    fig, axes = plt.subplots(1,5, dpi=100, figsize=(18,4))
    for i, (R,U) in enumerate(zip(R_values, U_matrices)):
        axes[i].scatter(R[:,0], R[:,1], color='gray')
        for R_, U_ in zip(R, U):
            axes[i].arrow(R_[0], R_[1], U_[1,0] * arrow_scale, U_[1,1] * arrow_scale, color='C0', width=0.03)
            axes[i].arrow(R_[0], R_[1], U_[2,0] * arrow_scale, U_[2,1] * arrow_scale, color='C1', width=0.03)
            axes[i].set_title(f"a/b = {a_values[i]/b_values[i]:.2f}")
        # axes[i].axis("equal")
        axes[i].set_ylim([-1.5, 1.5])
        axes[i].set_xlim([-1.5, 1.5])
    fig.tight_layout()
    plt.savefig("/home/mscherbela/ucloud/results/local_coords_H4_sweep.png", bbox_inches='tight')



