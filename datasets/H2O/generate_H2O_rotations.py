import numpy as np
from deeperwin.run_tools.geometry_utils import generate_geometry_variants
from deeperwin.run_tools.geometry_database import Geometry, GeometryDataset, save_geometries, save_datasets
from deeperwin.configuration import PhysicalConfig
import ase.visualize

np.random.seed(0)

SAVE_TO_DB = True
N_GEOMETRIES_TRAIN = [1, 10, 20, 50]
N_GEOMETRIES_TEST = 10
VISUALIZE = False


def get_intermolecule_distance(R1, R2):
    dist = np.linalg.norm(R1[:, None, :] - R2[None, :, :], axis=-1)
    return np.min(dist)


# %% Single, rotated H2O
phys = PhysicalConfig(name="H2O")
_, _, R_orig, Z = phys.get_basic_params()

for use_distortion in [False, True]:
    base_name = "H2O_rot_distorted" if use_distortion else "H2O_rot"
    R = generate_geometry_variants(
        R_orig,
        max(N_GEOMETRIES_TRAIN) + N_GEOMETRIES_TEST,
        rotate=True,
        distort=use_distortion,
        include_orig=True,
        random_state=int(use_distortion),
    )
    R = R - R[:, :1, :]
    datasets = []
    geometries = [Geometry(R=R_, Z=Z, name="H2O", comment=f"{base_name}_{i:03d}") for i, R_ in enumerate(R)]
    for n in N_GEOMETRIES_TRAIN:
        datasets.append(GeometryDataset(geometries[:n], name=f"{base_name}_{n}geoms"))
    datasets.append(GeometryDataset(geometries[-N_GEOMETRIES_TEST:], name=f"{base_name}_test_{N_GEOMETRIES_TEST}geoms"))

    if VISUALIZE:
        all_geometries = {g.hash: g for g in geometries}
        ase.visualize.view(datasets[-1].as_ase(all_geometries))
    if SAVE_TO_DB:
        save_geometries(geometries)
        save_datasets(datasets)

# %% Multiple, rotated H2O
MIN_INTERMOL_DISTANCE = 1.8  # Bohr

for dist in [3.0, 5.0, 10.0]:
    base_name = f"2xH2O_rot_{dist:.2f}ao"
    R = generate_geometry_variants(
        R_orig, (500 + N_GEOMETRIES_TEST) * 2, rotate=True, distort=False, include_orig=True, random_state=3
    )
    R = R - R[:, :1, :]
    R = R.reshape([-1, 6, 3])
    R[:, 3:, 0] += dist

    inter_mol_dist = np.array([get_intermolecule_distance(R_[:3, :], R_[3:]) for R_ in R])
    R = R[inter_mol_dist >= MIN_INTERMOL_DISTANCE]
    assert len(R) >= max(N_GEOMETRIES_TRAIN) + N_GEOMETRIES_TEST
    R = R[: max(N_GEOMETRIES_TRAIN) + N_GEOMETRIES_TEST]

    datasets = []
    geometries = [Geometry(R=R_, Z=list(Z) * 2, name="2xH2O", comment=f"{base_name}_{i:03d}") for i, R_ in enumerate(R)]
    for n in N_GEOMETRIES_TRAIN:
        datasets.append(GeometryDataset(geometries[:n], name=f"{base_name}_{n}geoms"))
    datasets.append(GeometryDataset(geometries[-N_GEOMETRIES_TEST:], name=f"{base_name}_test_{N_GEOMETRIES_TEST}geoms"))

    if VISUALIZE:
        all_geometries = {g.hash: g for g in geometries}
        ase.visualize.view(datasets[-1].as_ase(all_geometries))
    if SAVE_TO_DB:
        save_geometries(geometries)
        save_datasets(datasets, overwite_existing=True)
