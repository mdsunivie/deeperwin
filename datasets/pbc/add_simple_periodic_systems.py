# %%
from deeperwin.run_tools.geometry_database import (
    Geometry,
    GeometryDataset,
    PeriodicLattice,
    save_datasets,
    save_geometries,
)
import numpy as np


def get_conv_bcc(a, Z):
    assert len(Z) == 2
    lattice = np.eye(3) * a
    R = [[0, 0, 0], [a / 2, a / 2, a / 2]]
    return R, Z, lattice


def format_k_string(k):
    return ",".join([f"{ki:.2f}" for ki in k])


# LiH_datasets
lattice_constants = np.arange(4.0, 5.7, 0.4)
k_twists = [[0, 0, 0], [1 / 3, 0, 0], [1 / 3, 1 / 3, 0], [1 / 3, 1 / 3, 1 / 3]]
supercells = [1, 2]

for sc in supercells:
    sc_string = f"{sc}x{sc}x{sc}"
    all_geoms = []
    for a in lattice_constants:
        R, Z, lattice_prim = get_conv_bcc(a, [3, 1])
        for k_twist in k_twists:
            periodic = PeriodicLattice(lattice_prim, supercell=[sc, sc, sc], k_twist=k_twist)
            geom = Geometry(
                R,
                Z,
                periodic=periodic,
                name=f"LiH_bcc_{sc_string}",
                comment=f"LiH_bcc_{sc_string}_a={a:.1f}_k={format_k_string(k_twist)}",
            )
            all_geoms.append(geom)
    gamma_geoms = [g for g in all_geoms if sum(g.periodic.k_twist) == 0]
    equ_geoms = [g for g in all_geoms if np.isclose(g.periodic.lattice_prim[0, 0], 5.2)]

    # dataset_both = GeometryDataset(all_geoms, name=f"LiH_bcc_{sc_string}_5geom_4twist_equil")
    # dataset_geoms = GeometryDataset(gamma_geoms, name=f"LiH_bcc_{sc_string}_5geom_gamma_equil")
    dataset_twist = GeometryDataset(equ_geoms, name=f"LiH_bcc_{sc_string}_a5.2_4twist")

    save_geometries(all_geoms)
    save_datasets([dataset_twist])
