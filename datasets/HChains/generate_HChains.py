import numpy as np
from deeperwin.run_tools.geometry_database import save_geometries, save_datasets, Geometry, GeometryDataset

ATOM_CHARGE = 1
BOND_DISTANCES = np.arange(1.6, 2.1, 0.2)
PAIR_DISTANCES = np.arange(2.0, 2.7, 0.2)
EQUIDISTANT_DISTANCES = np.arange(1.6, 2.7, 0.2)
N_ATOMS_EQUIDISTANT = np.arange(2, 21, 2)
N_ATOMS_DIMERIZATION = np.arange(4, 21, 2)

# %% Dimerization of hydrogen; alternating short and long bond lengths
geometries = []
datasets = []
for n in N_ATOMS_DIMERIZATION:
    geometries_n = []
    for bond in BOND_DISTANCES:
        for pair in PAIR_DISTANCES:
            R = np.zeros([n, 3])
            Z = [ATOM_CHARGE] * n
            for i in range(0, n, 2):
                R[i, 0] = (bond + pair) * i / 2
                R[i + 1, 0] = (bond + pair) * i / 2 + bond
            geometries_n.append(Geometry(R=R, Z=Z, comment=f"HChain{n}_{bond:.2f}_{pair:.2f}", name=f"HChain{n}"))
    datasets.append(GeometryDataset(geometries_n, f"HChain{n}_dimerization_{len(geometries_n)}geoms"))
    geometries += geometries_n
save_geometries(geometries)
save_datasets(datasets)

# %% Equidistant hydrogen chains
geometries = []
datasets = []
for n in N_ATOMS_EQUIDISTANT:
    geometries_n = []
    for dist in EQUIDISTANT_DISTANCES:
        R = np.zeros([n, 3])
        Z = [ATOM_CHARGE] * n
        R[:, 0] = np.arange(n) * dist
        geometries_n.append(Geometry(R=R, Z=Z, comment=f"HChain{n}_{dist:.2f}", name=f"HChain{n}"))
    datasets.append(
        GeometryDataset(
            geometries_n,
            f"HChain{n}_equidist_{min(EQUIDISTANT_DISTANCES):.2f}-{max(EQUIDISTANT_DISTANCES):.2f}_{len(geometries_n)}geoms",
        )
    )
    geometries += geometries_n
save_geometries(geometries)
save_datasets(datasets)

# %% Equidistant hydrogen chains with fixed spacing
geometries = []
SPACING = 1.8
N_ATOMS = np.arange(2, 29, 2)
for n in N_ATOMS:
    R = np.zeros([n, 3])
    Z = [ATOM_CHARGE] * n
    R[:, 0] = np.arange(n) * SPACING
    geometries.append(Geometry(R=R, Z=Z, comment=f"HChain{n}_{SPACING:.2f}", name=f"HChain{n}"))
save_geometries(geometries)
dataset = GeometryDataset(geometries, name=f"HChain_equidist_{min(N_ATOMS)}-{max(N_ATOMS)}_{SPACING:.2f}")
save_datasets(dataset)
