# %%
import numpy as np
import jax.numpy as jnp
import jax
from pyscf.pbc.gto.eval_gto import eval_gto
import optax
import scipy.linalg
from deeperwin.utils.periodic import project_into_first_unit_cell, cartesian_to_fractional, fractional_to_cartesian
import logging

# def _get_uniform_points_in_cell(lattice, n_points):
#     r_frac = np.random.uniform([0, 0, 0], [1.0, 1.0, 1.0], size=[n_points, 3])
#     r = r_frac @ lattice
#     return r


def get_integral_points(lat, R, points_per_atom, gaussian_width=4.0, grid_size=2):
    """Draws points from a gaussian distribution around each atom and discards all points that fall outside the first unit cell.

    Since the points are (on purpose) not drawn uniformly (but more densly around the atoms), this function also returns their
    unnormalized probability density p, to allow reweighting of subsequent integrals.
    """
    # Get grid of atoms repeated in neighboring cells
    ndims = lat.shape[0]
    ordinals = np.arange(-grid_size, grid_size + 1)
    ordinals = np.stack(np.meshgrid(*[ordinals] * ndims), axis=-1).reshape([-1, ndims])
    centers = ((ordinals @ lat)[:, None, :] + R[None, :, :]).reshape([-1, ndims])

    # Generate gaussian points around each atom and keep only those that are in the original cell
    r = np.random.normal(size=[points_per_atom, *centers.shape], scale=gaussian_width) + centers
    r = r.reshape([-1, ndims])
    r_frac = cartesian_to_fractional(r, inv_lattice=np.linalg.inv(lat))
    is_in_cell = np.all((r_frac >= 0) & (r_frac < 1), axis=-1)
    r = fractional_to_cartesian(r_frac[is_in_cell], lattice=lat)

    # Compute their likelihood, being a sum of gaussians and reweight them
    dist_to_centers = np.linalg.norm(r[:, None, :] - centers[None, :, :], axis=-1)
    p = np.sum(np.exp(-0.5 * (dist_to_centers / gaussian_width) ** 2), axis=-1)  # sum over centers
    return r, p


# TODO: Get analytic integrals for the polarization matrix, instead of using MC integration
def _build_polarization_matrix(cell, mo_coeff, k_twist=None, n_integration=100_000, batch_size=5_000):
    lattice_sc = np.array(cell.a)
    rec = 2 * np.pi * np.linalg.inv(lattice_sc)  # rec vectors in the columns

    # For a general lattice is not sufficient to compute <mo | exp(-i <rec_l, r>) | mo>,
    # since this can lead to distorted orbitals. Choosing theses 6 reciprocal lattice vectors ensures proper
    # localization for all lattices, according to
    # Silvestrelli, PRB 1999, Maximally localized Wannier functions for simulations with supercells of general symmetry
    # 10.1103/PhysRevB.59.9703
    if np.any(lattice_sc - np.diag(np.diag(lattice_sc)) != 0):
        additional_rec = np.sum(rec, axis=1, keepdims=True) - rec
        rec = np.concatenate([rec, additional_rec], axis=1)  # => [3 x 6]

    # rec_sc = 2 * np.pi * np.linalg.inv(lattice_sc)

    n_batches = int(np.ceil(n_integration / batch_size))
    batch_size = n_integration // n_batches

    chi = 0.0
    logging.getLogger("dpe").debug(f"Building polarization matrix with {n_integration} points in {n_batches} batches")
    for _ in range(n_batches):
        r_integration, p = get_integral_points(lattice_sc, cell.atom_coords(), batch_size)
        aos = eval_gto(cell, "GTOval_cart", r_integration, kpt=k_twist)
        mos = aos @ mo_coeff

        # Reweigh the molecular orbitals inversely proportional to their sample likelikhood
        # This way \sum mo_weighted_i mo_weighted_j = \int mo_i(r) mo_j(r) p(r) / p(r) = S_ij
        mos_weighted = mos / np.sqrt(p[:, None])
        mos_weighted = mos_weighted / np.linalg.norm(mos_weighted, axis=0, keepdims=True)

        phase = np.exp(-1.0j * (r_integration @ rec))
        chi += np.einsum("bn,bl,bm->nml", mos_weighted.conj(), phase, mos_weighted)
    print("Finished building polarization matrix")
    return chi / n_batches


# TODO: Get analytic integrals for the overlap matrix, instead of using MC integration
def _build_overlap_matrix(cell, k_twist=None, n_integration=10_000):
    lattice_sc = np.array(cell.a)
    r_integration, p = get_integral_points(lattice_sc, cell.atom_coords(), n_integration)

    aos = eval_gto(cell, "GTOval_cart", r_integration, kpt=k_twist)
    aos_weighted = aos / np.sqrt(p[:, None])
    aos_weighted = aos_weighted / np.linalg.norm(aos_weighted, axis=0, keepdims=True)
    S = np.einsum("bi,bj->ij", aos_weighted.conj(), aos_weighted)
    return S


def _get_U(A):
    A_symm = (A + A.T.conj()) * 0.5
    U = jax.scipy.linalg.expm(1j * A_symm, upper_triangular=False)
    return U


def _loss_func(A, chi):
    U = _get_U(A)
    Omega = jnp.einsum("nm,mlx,ln->nx", U.T.conj(), chi, U)
    loss = 1 - jnp.mean(Omega.real**2 + Omega.imag**2)
    return loss


def set_orbital_phase(mo_coeff):
    """Set the phase of each orbital such that the real part of each orbital is maximal"""
    phases = np.angle(np.sum(mo_coeff, axis=0))
    mo_coeff *= np.exp(-1j * phases)[None, :]
    return mo_coeff


def _matrix_inv_sqrt(H, eps=1e-6):
    s, U = np.linalg.eigh(H)
    s_inv = np.where(s > eps, 1.0 / np.sqrt(s), 0)
    s_2 = np.einsum("ni,i,im->nm", U, s_inv, U.T.conj())
    return s_2


def _orthonormalize(mo_coeffs, S, eps=1e-6):
    overlap = mo_coeffs.T.conj() @ S @ mo_coeffs
    s_2 = _matrix_inv_sqrt(overlap)
    mo_coeffs_orth = mo_coeffs @ s_2
    return mo_coeffs_orth


def localize_orbitals_pbc(
    mo_coeff,
    pyscf_cell,
    k_twist=None,
    mo_guess=None,
    n_steps=10_000,
    n_integration=100_000,
    lr=1e-2,
    rng_key=None,
    mo_are_orthonormal=True,
    align_phase=True,
    get_loss_curve=False,
    force_complex=True,
):
    """
    Localize the orbitals of a periodic system using Foster-Boys localization (=spatial variance minimization).

    For mathematical details, see 10.1103/RevModPhys.84.1419, II.F.2 (Gamma-point formulation for supercells).
    This implementation assumes an orthorombic lattice (i.e. a diagonal lattice matrix), and a Gamma-point calculation,
    i.e. only one k-point with k=0.

    Args:
        mo_coeff (np.ndarray): The original orbitals, shape (n_basis_functions, n_occupied_orbitals)
        pyscf_cell (pyscf.pbc.gto.Cell): The periodic system
        mo_guess (np.ndarray): An initial guess for the localized orbitals (optional). Does not have to be orthogonal, but orbitals must be linearly independent., shape (n_basis_functions, n_occupied_orbitals)
        n_steps (int): Number of optimization steps
        lr (float): Learning rate for optimization. A cosine-scheduler is additionally applied, ramping the LR towards 0 at the end of the optimization.
        rng_key (jax.random.PRNGKey): Random number generator key for initialization of the transformation matrix U
        mo_are_orthonormal (bool): Whether the original orbitals are already orthonormal. If False, the overlap matrix is computed and the orbitals are orthonormalized.
        align_phase (bool): Whether to align the phase of the localized orbitals such that the real part of each orbital is maximal.
        get_loss_curve (bool): Whether to return the full loss curve along the optimization (True) or only the final loss (False). Loss values should be between 0 and 1. In small supercells relatively high values (e.g. 0.7) are expected.
        force_complex (bool): Whether to force the mo_coeff to be complex. If True, the mo_coeff will be cast to complex128. If False, the mo_coeff will keep its original dtype.
    """
    logging.getLogger("dpe").debug(
        f"Running orbital localization for periodic system; occ. mo_coeffs shape: {mo_coeff.shape}"
    )
    lattice = np.array(pyscf_cell.a)
    rec = 2 * np.pi * np.linalg.inv(lattice)
    # is_diagonal = np.all(lattice == np.diag(np.diag(lattice)))
    # assert is_diagonal, "Orbital localization is currently only implemented for diagonal, orthorombic lattices"
    n_orbitals = mo_coeff.shape[1]
    if force_complex:
        mo_coeff = mo_coeff * (1 + 0j)

    # Ensure that the original orbitals are othonormal to allow easy projection
    if not mo_are_orthonormal:
        S = _build_overlap_matrix(pyscf_cell, k_twist)
        mo_coeff = _orthonormalize(mo_coeff, S)

    # Compute the matrix chi = <phi_i | exp(-i 2pi/L r) | phi_j>, where phi_i are the original orbitals
    chi = _build_polarization_matrix(pyscf_cell, mo_coeff, k_twist, n_integration)

    # Initialize the transformation either randomly or from a guess
    if mo_guess is None:
        rng = rng_key if rng_key is not None else jax.random.PRNGKey(np.random.randint(0, 2**32))
        A = jax.random.normal(rng, [n_orbitals, n_orbitals], dtype=mo_coeff.dtype) * 0.1
    else:
        assert mo_guess.shape == mo_coeff.shape
        assert np.linalg.matrix_rank(mo_guess) == n_orbitals, "The guess orbitals are not linearly independent"
        S = _build_overlap_matrix(pyscf_cell, k_twist)

        # Project guess orbitals onto the original orbitals
        X = mo_coeff.T.conj() @ S @ mo_guess
        # Orthogonalize projection of the guess orbitals
        U_init = X @ _matrix_inv_sqrt(X.T.conj() @ X)
        A = -1j * scipy.linalg.logm(U_init)
        A = jnp.array(A, dtype=mo_coeff.dtype)

    # Built the optimizer and initialize the parameters
    def lr_schedule(t):
        x = t / n_steps
        return lr * jnp.cos(0.5 * np.pi * x) ** 2

    optimizer = optax.adam(lr_schedule)
    opt_state = optimizer.init(A)

    # Define gradient step and perform optimization
    @jax.jit
    def grad_step(A, chi, opt_state):
        loss, grad = jax.value_and_grad(_loss_func, argnums=0)(A, chi)
        updates, opt_state = optimizer.update(grad, opt_state)
        A = optax.apply_updates(A, updates.conj())
        return loss, A, opt_state

    loss = np.zeros(n_steps)
    for i in range(n_steps):
        loss[i], A, opt_state = grad_step(A, chi, opt_state)

    if not get_loss_curve:
        loss = loss[-1]

    # Compute the unitary matrix U that maximally localizes the orbitals and apply the transformation
    U = _get_U(A)
    mo_coeff = mo_coeff @ U

    # Eq. 9 in Silvestrelli PRB 1999, 10.1103/PhysRevB.59.9703
    chi_loc = jnp.einsum("nm,mlx,ln->nx", U.T.conj(), chi[..., :3], U)  # only use the 3 actual rec vectors
    orbital_pos = -np.log(chi_loc).imag
    rec_norm = np.linalg.norm(rec, axis=0)
    M = rec.T / rec_norm[:, None]  # M_lk = <b_l, e_k> / |b_l|
    mapping_matrix = np.linalg.inv(M) / rec_norm[None, :]  # (M^-1)_lk / b_k
    orbital_pos = mapping_matrix @ orbital_pos.T  # [3x3] x [3 x n_orbitals]=> [3 x n_orbital]

    offset = 0.5
    orbital_pos = project_into_first_unit_cell(orbital_pos.T + offset, lattice=lattice).T - offset

    # Get all-electron complex polarization z
    z_loc = np.linalg.det(np.moveaxis(chi[:, :, :3], 2, 0))

    logging.getLogger("dpe").debug(f"Localized orbital positions: {orbital_pos}")

    if align_phase:
        # Apply another transformation, which sets the phase of each orbital,
        # such that the real part of each orbital is maximal
        mo_coeff = set_orbital_phase(mo_coeff)

    metrics = dict(loc_loss=loss)
    for dim, val in enumerate(z_loc):
        metrics[f"baseline_loc_z_{dim}_real"] = val.real
        metrics[f"baseline_loc_z_{dim}_imag"] = val.imag
        metrics[f"baseline_loc_abs_{dim}"] = np.abs(val)
    return mo_coeff, metrics, orbital_pos
