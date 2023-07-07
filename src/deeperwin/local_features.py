import numpy as np
from deeperwin.configuration import PhysicalConfig
import jax.numpy as jnp

def build_local_rotation_matrices(R, Z, length_scale=1.0, tolerance=1e-6):
    """Returns array of shape [n_ions x coord_axis (3) x xyz (3)]"""
    R = np.array(R)
    Z = np.array(Z, int)
    local_coords = []
    for ind_atom, R_ in enumerate(R):
        diff = R - R_
        dist = np.linalg.norm(diff, axis=-1)
        weighted_diff = diff * (Z * np.exp(-dist / length_scale))[..., None]
        weighted_diff_centered = weighted_diff - np.mean(weighted_diff, axis=0, keepdims=True)
        cov_matrix = weighted_diff_centered.T @ weighted_diff_centered
        _, eigen_vectors = np.linalg.eigh(cov_matrix)
        eigen_vectors = eigen_vectors.T

        ref_vectors = [
            np.sum(weighted_diff, axis=0),
            np.sum(diff, axis=0)
            ]
        fallback_vectors = [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1])
        ]
        for j, v in enumerate(eigen_vectors):
            eigen_vectors[j, :] = align_with_reference_vectors(v, ref_vectors + fallback_vectors, tolerance)
            ref_vectors.append(np.cross(ref_vectors[0], v))
        local_coords.append(eigen_vectors)
    return np.stack(local_coords)

def align_with_reference_vectors(v, ref_vectors, tol=1e-6):
    """
    Adjust the sign of an input vector v, such that it has positive overlap with a given reference vector.

    Args:
        v: np.array, input vector to be aligned
        ref_vectors: List[np.array], reference vectors to be used to determine the sign of v, sorted by precendence (most important first)
        tol: float, tolerance to be used when determining whether two vectors are orthogonal

    This functions loops over the given reference vectors until a reference vector is found that has non-zero overlap with v.
    The sign of v is then adjusted s.t. v has positive overlap with the reference vector.
    If all reference vectors are orthogonal to v, a ValueError is raised.

    Returns:
        v_adjusted: np.array
    """
    for v_ref in ref_vectors:
        overlap = v @ v_ref
        if np.abs(overlap) >= tol:
            return v * np.sign(overlap)
    raise ValueError("Could not determine sign of coordinate axis")

def build_global_rotation_matrix(R, Z):
    R = jnp.array(R)
    Z = jnp.array(Z, float)
    unweighted_center_of_mass = jnp.mean(R, axis=0)
    diff = R - unweighted_center_of_mass
    dist = jnp.linalg.norm(diff, axis=-1, keepdims=True)
    weights = dist**2 * Z[..., None]
    weighted_center_of_mass = jnp.sum(diff * weights, axis=0) / jnp.sum(weights, axis=0)
    cov_matrix = diff.T @ diff
    _, eigen_vectors = np.linalg.eigh(cov_matrix)
    eigen_vectors = eigen_vectors.T
    ref_vector = weighted_center_of_mass - unweighted_center_of_mass

    overlap = np.where(eigen_vectors @ ref_vector < 0, -1, 1)
    eigen_vectors = eigen_vectors * overlap[:, None]
    return eigen_vectors

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    phys = PhysicalConfig(name="Ethene")
    _, _, R, Z = phys.get_basic_params()
    R = R[:, [2,1,0]]
    phi = 345 * np.pi / 180
    rot_mat = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
    R[-2:, :] *= 1.2
    R[:, :2] = R[:, :2] @ rot_mat

    U = build_global_rotation_matrix(R, Z)

    plt.close("all")
    fig, ax = plt.subplots(1,1)
    ax.scatter(R[:,0], R[:,1], color='gray')
    arrow_scale = 0.2
    for i, v in enumerate(U):
        ax.arrow(0, 0, v[0] * arrow_scale, v[1] * arrow_scale, color=f'C{i}', width=0.03)
    ax.axis("equal")
