# %%
import numpy as np
from deeperwin.run_tools.geometry_database import (
    save_geometries,
    save_datasets,
    Geometry,
    GeometryDataset,
    PeriodicLattice,
)


def get_reduced_kpoints_1D(n_k_grid):
    if n_k_grid % 2 == 0:
        n_reduced = n_k_grid // 2 + 1
        k_grid = np.arange(n_reduced) / n_k_grid
        weights = np.array([1] + [2] * (n_reduced - 2) + [1])
    else:
        n_reduced = (n_k_grid + 1) // 2
        k_grid = np.arange(n_reduced) / n_k_grid
        weights = np.array([1] + [2] * (n_reduced - 1))
    return k_grid, list(weights)


# %%
NONPERIODIC_SPACING = 100.0
ATOM_CHARGE = 1
# N_ATOMS = np.arange(4, 23, 2)
# N_K_GRID_NONREDUCED = [1, 4, 5, 8, 9]
# N_ATOMS = np.arange(10, 23, 4)
# N_ATOMS = [12, 16, 20, 24, 28]
# N_ATOMS = np.arange(12, 21, 4)
N_ATOMS = [4, 6, 8, 10, 12]
N_K_GRID_NONREDUCED = [4]
BOND_DISTANCES = [1.7, 1.8, 1.9, 2.0]
# BOND_DISTANCES = np.arange(1.2, 3.7, 0.6)
# BOND_DISTANCES = np.array([1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 3.0, 3.6])
# BOND_DISTANCES = [1.8]

for n_k_grid in N_K_GRID_NONREDUCED:
    geometries_all = []
    for n_atoms in N_ATOMS:
        geometries_for_dataset = []
        for bond_dist in BOND_DISTANCES:
            twists, weights = get_reduced_kpoints_1D(n_k_grid)
            for twist in twists:
                R = np.array([[0, 0, 0], [bond_dist, 0, 0]])
                Z = [ATOM_CHARGE] * 2
                lat_vecs = np.diag([bond_dist * 2, NONPERIODIC_SPACING, NONPERIODIC_SPACING])
                lat = PeriodicLattice(lat_vecs, [n_atoms // 2, 1, 1], k_twist=[twist, 0, 0])
                g = Geometry(
                    R=R,
                    Z=Z,
                    periodic=lat,
                    comment=f"HChainPBC{n_atoms}_{bond_dist:.2f}_k={twist:.3f}",
                    name=f"HChainPBC{n_atoms}",
                )
                geometries_for_dataset.append(g)
        # ds_name = f"HChainPBC_{min(N_ATOMS)}-{max(N_ATOMS)}_{min(BOND_DISTANCES):.2f}-{max(BOND_DISTANCES):.2f}_{n_k_grid}kgrid"
        if len(BOND_DISTANCES) == 1:
            ds_name = f"HChainPBC{n_atoms}_{BOND_DISTANCES[0]:.2f}_{n_k_grid}kgrid"
        else:
            ds_name = f"HChainPBC{n_atoms}_{min(BOND_DISTANCES):.2f}-{max(BOND_DISTANCES):.2f}_{len(BOND_DISTANCES)}geoms_{n_k_grid}kgrid"
        print(f"{ds_name:<40} {len(geometries_for_dataset)}")
        # ds = GeometryDataset(geometries_for_dataset, name=ds_name, weights=weights * len(BOND_DISTANCES))
        # datasets.append(ds)
        geometries_all += geometries_for_dataset
    ds_name = f"HChainPBC{min(N_ATOMS)}-{max(N_ATOMS)}_{min(BOND_DISTANCES):.2f}-{max(BOND_DISTANCES):.2f}_{len(geometries_all)}geoms_{n_k_grid}kgrid"
    print(f"{ds_name:<40} {len(geometries_all)}")
    ds = GeometryDataset(geometries_all, name=ds_name, weights=weights * len(BOND_DISTANCES) * len(N_ATOMS))
    save_geometries(geometries_all)
    save_datasets(ds)

# save_geometries(geometries_all)
# save_datasets(datasets)

# %%
# ds = GeometryDataset(datasets=[f"HChainPBC{i}_1.20-3.60_8geoms_1kgrid" for i in range(10, 27, 4)],
#                      name="HChainPBC10-26_1.20-3.60_40geoms_1kgrid")
# save_datasets(ds)
