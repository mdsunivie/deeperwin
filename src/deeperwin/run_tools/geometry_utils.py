import numpy as np
import scipy.spatial

from deeperwin.utils.utils import get_distance_matrix

BOHR_IN_ANGSTROM = 0.529177249


def distort_molecule(R, n, d_min_factor, d_max_factor, noise_scale):
    R_out = []

    n_atoms = len(R)
    if n_atoms == 2:
        center = (R[0] + R[1]) / 2
        diff_vec = (R[1] - R[0]) / 2
        R_out = np.zeros([n, 2, 3])
        scale = np.linspace(d_min_factor, d_max_factor, n)
        R_out[:, 0, :] = center - diff_vec * scale[:, None]
        R_out[:, 1, :] = center + diff_vec * scale[:, None]
        return R_out

    distances_orig = get_distance_matrix(R, full=False)[1]
    n_accept = 0
    while n_accept < n:
        noise = np.random.normal(size=[n_atoms, 3]) * noise_scale
        noise -= np.mean(noise, axis=-1, keepdims=True)
        R_new = np.array(R) + noise
        distance_new = get_distance_matrix(R_new, full=False)[1]
        dist_ratio = distance_new / distances_orig
        if np.all((d_min_factor <= dist_ratio) & (dist_ratio <= d_max_factor)):
            R_out.append(R_new)
            n_accept += 1
    return np.array(R_out)


def generate_geometry_variants(
    R_orig, n_variants, rotate=True, distort=True, include_orig=True, random_state: int = None, noise_scale=0.5
):
    if random_state is not None:
        np.random.seed(random_state)
    if distort:
        R = distort_molecule(R_orig, n_variants, d_min_factor=0.9, d_max_factor=1.3, noise_scale=noise_scale)
    else:
        R = np.tile(R_orig, [n_variants, 1, 1])

    if rotate:
        U = scipy.spatial.transform.Rotation.random(n_variants).as_matrix()
        R = np.einsum("...ni,...ij->...nj", R, U)

    if include_orig:
        R[0, :, :] = R_orig
    return R
