import h5py

from deeperwin.geometries import parse_coords, ANGSTROM_IN_BOHR
from deeperwin.utils.utils import get_distance_matrix
import os
import numpy as np

np.random.seed(0)


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
    return R_out


heat_dir = "/home/mscherbela/develop/deeperwin_jaxtest/datasets/geometries/HEAT"
heat_data = []
for fname in os.listdir(heat_dir):
    full_name = os.path.join(heat_dir, fname)
    if os.path.isfile(full_name) and fname.startswith("coord"):
        content = open(full_name).read()
        molecule_name = fname.split(".")[-1]
        heat_data.append((*parse_coords(content), molecule_name))

print(f"Molecules in HEAT          :         {len(heat_data):4d}")

# Remove Fluorine, because it is not present in QM7X
heat_data = [h for h in heat_data if (9 not in h[1])]
print(f"Molecules in HEAT (excl. F):         {len(heat_data):4d}")


n_distortions_per_geom = 100
d_min_factor = 0.9
d_max_factor = 1.5
noise_scale = 1.0

output_data = []
for R, Z, name in heat_data:
    print(f"Distorting {name}...")
    n_atoms = len(Z)
    R_distorted = distort_molecule(R, n_distortions_per_geom, d_min_factor, d_max_factor, noise_scale)
    rmsd = np.sqrt(np.sum((R_distorted - R) ** 2, axis=-1).mean(axis=-1)) / ANGSTROM_IN_BOHR
    output_data.append((R, Z, f"HEAT-{name}-opt"))
    for ind_dist, R_ in enumerate(R_distorted):
        output_data.append((R_, Z, f"HEAT-{name}-d{ind_dist+1}"))

with h5py.File("/home/mscherbela/runs/datasets/HEAT_distorted/HEAT_distorted.h5py", "w") as f_out:
    for R, Z, name in output_data:
        f_out.create_dataset(f"{name}/atXYZ", data=R / ANGSTROM_IN_BOHR)
        f_out.create_dataset(f"{name}/atNUM", data=np.array(Z, int))
print(f"Added {len(output_data)} geometries to {f_out}")
