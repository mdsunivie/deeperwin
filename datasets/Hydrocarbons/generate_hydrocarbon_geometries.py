import numpy as np
from deeperwin.run_tools.geometry_utils import BOHR_IN_ANGSTROM, generate_geometry_variants
from deeperwin.run_tools.geometry_database import Geometry, GeometryDataset, save_geometries, save_datasets
import ase.visualize

np.random.seed(0)

SAVE_TO_DB = True
N_GEOMETRIES_TRAIN = [10, 50]
N_GEOMETRIES_TEST = 10
VISUALIZE = False

# %% CH4
Z = [6, 1, 1, 1, 1]
R_orig = np.array(
    [
        [0, 0, 0],
        [1.18599212, 1.18599212, 1.18599212],
        [1.18599212, -1.18599212, -1.18599212],
        [-1.18599212, 1.18599212, -1.18599212],
        [-1.18599212, -1.18599212, 1.18599212],
    ]
)
R = generate_geometry_variants(
    R_orig, max(N_GEOMETRIES_TRAIN) + N_GEOMETRIES_TEST, rotate=True, distort=True, include_orig=True, random_state=0
)
R = R - R[:, :1, :]
geometries = [Geometry(R=R_, Z=Z, comment=f"CH4_rot_distorted_{i:03d}", name="CH4") for i, R_ in enumerate(R)]
if SAVE_TO_DB:
    save_geometries(geometries)
for n in [10, 50]:
    dataset = GeometryDataset(geometries[:n], f"CH4_rot_distorted_{n}geoms")
    if SAVE_TO_DB:
        save_datasets(dataset)
dataset_test = GeometryDataset(geometries[-N_GEOMETRIES_TEST:], f"CH4_rot_distorted_test_{N_GEOMETRIES_TEST}geoms")
if SAVE_TO_DB:
    save_datasets(dataset_test)

if VISUALIZE:
    all_geometries = {g.hash: g for g in geometries}
    ase_geoms = dataset.as_ase(all_geometries)
    ase.visualize.view(ase_geoms)

# %% CH3p
CH_BOND_LENGTH = 1.088 / BOHR_IN_ANGSTROM  # sp2 CH bond-length
Z = [6, 1, 1, 1]
s60, c60 = np.sin(60 * np.pi / 180), np.cos(60 * np.pi / 180)

R_orig = np.array(
    [
        [0, 0, 0],
        [-CH_BOND_LENGTH, 0, 0],
        [CH_BOND_LENGTH * c60, CH_BOND_LENGTH * s60, 0],
        [CH_BOND_LENGTH * c60, -CH_BOND_LENGTH * s60, 0],
    ]
)
R = generate_geometry_variants(
    R_orig, max(N_GEOMETRIES_TRAIN) + N_GEOMETRIES_TEST, rotate=True, distort=True, include_orig=True, random_state=1
)
R = R - R[:, :1, :]
geometries = [
    Geometry(R=R_, Z=Z, comment=f"CH3p_rot_distorted_{i:03d}", name="CH3p", charge=1) for i, R_ in enumerate(R)
]
if SAVE_TO_DB:
    save_geometries(geometries)
for n in [10, 50]:
    dataset = GeometryDataset(geometries[:n], f"CH3p_rot_distorted_{n}geoms")
    if SAVE_TO_DB:
        save_datasets(dataset)
dataset_test = GeometryDataset(geometries[-N_GEOMETRIES_TEST:], f"CH3p_rot_distorted_test_{N_GEOMETRIES_TEST}geoms")
if SAVE_TO_DB:
    save_datasets(dataset_test)

if VISUALIZE:
    all_geometries = {g.hash: g for g in geometries}
    ase_geoms = dataset.as_ase(all_geometries)
    ase.visualize.view(ase_geoms)


# %% Ethene
def get_ethene(CH_bond_length, CC_bond_length, CCH_angle, twist):
    s, c = np.sin((180 - CCH_ANGLE) * np.pi / 180), np.cos((180 - CCH_ANGLE) * np.pi / 180)
    stwist, ctwist = np.sin(twist * np.pi / 180), np.cos(twist * np.pi / 180)

    R_orig = np.array(
        [
            [CC_bond_length / 2, 0, 0],
            [-CC_bond_length / 2, 0, 0],
            [CC_bond_length / 2 + CH_bond_length * c, CH_bond_length * s * ctwist, CH_bond_length * s * stwist],
            [CC_bond_length / 2 + CH_bond_length * c, -CH_bond_length * s * ctwist, -CH_bond_length * s * stwist],
            [-CC_bond_length / 2 - CH_bond_length * c, CH_bond_length * s, 0],
            [-CC_bond_length / 2 - CH_bond_length * c, -CH_bond_length * s, 0],
        ]
    )
    return R_orig


CH_BOND_LENGTH = 1.087 / BOHR_IN_ANGSTROM  # sp2 CH bond-length
CC_BOND_LENGTH = 1.30 / BOHR_IN_ANGSTROM
CCH_ANGLE = 121.3
N_GEOMETRIES = 100
R_orig = get_ethene(CH_BOND_LENGTH, CC_BOND_LENGTH, CCH_ANGLE, twist=90)
Z = [6, 6, 1, 1, 1, 1]

# DISTANCES = [1.3, 1.38]
# TWISTS = np.arange(0, 91, 10)
DISTANCES = [1.38]
TWISTS = np.arange(0, 91, 30)

geometries = []
for dist in DISTANCES:
    for twist in TWISTS:
        geometries.append(
            Geometry(
                R=get_ethene(CH_BOND_LENGTH, dist / BOHR_IN_ANGSTROM, CCH_ANGLE, twist),
                Z=Z,
                name="C2H4",
                comment=f"C2H4_CC{dist:.2f}A_{twist:.0f}deg",
            )
        )
dataset = GeometryDataset(geometries, name=f"C2H4_{len(DISTANCES)}stretch_{len(TWISTS)}twist_{len(geometries)}geoms")

if SAVE_TO_DB:
    save_geometries(geometries)
    save_datasets(dataset)

if VISUALIZE:
    all_geometries = {g.hash: g for g in geometries}
    ase_geoms = dataset.as_ase(all_geometries)
    ase.visualize.view(ase_geoms)
