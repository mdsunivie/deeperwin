# %%
import pyscf.__config__

pyscf.__config__.pbc_symm_space_group_symprec = 1e-3

import numpy as np

from deeperwin.run_tools.geometry_database import (
    Geometry,
    GeometryDataset,
    PeriodicLattice,
    save_datasets,
    save_geometries,
)
from deeperwin.utils.periodic import map_to_first_brillouin_zone
import logging
from deeperwin.configuration import PhysicalConfig
from pyscf.pbc import gto
from pyscf.pbc.lib.kpts import KPoints
import matplotlib.pyplot as plt


def graphene_geometry():
    lattice = [[4.02795, -2.3255, 0.0], [4.02795, 2.3255, 0.0], [0.0, 0.0, 99.99995]]
    R = [[2.6853, 0.0, 0.0], [5.3706, 0.0, 0.0]]
    Z = [6, 6]

    return R, Z, lattice


def get_rotmat(phi_deg):
    U = np.eye(3)
    s, c = np.sin(np.deg2rad(phi_deg)), np.cos(np.deg2rad(phi_deg))
    U[0, 0] = c
    U[0, 1] = -s
    U[1, 0] = s
    U[1, 1] = c
    return U


#
# def get_ds_twists(cell):
#     k_1 = (1/3) * cell.reciprocal_vectors()[0, :] + (1/3) * cell.reciprocal_vectors()[1, :]
#     k_2 = (2/3) * cell.reciprocal_vectors()[0, :] + (1/3) * cell.reciprocal_vectors()[1, :]
#
#     print(f"k_1: {k_1}")
#     print(f"k_2: {k_2}")


def get_k_twists(twist_grid_size, lattice=None, R=None, Z=None):
    if twist_grid_size == 1:
        return np.array([[0.0, 0.0, 0.0]])

    # assert len(Z) == 8
    cell = gto.Cell(atom=[(Z_, *R_) for Z_, R_ in zip(Z, R)], a=lattice, unit="bohr", space_group_symmetry=True).build()
    kpts = KPoints(cell, cell.make_kpts([twist_grid_size, twist_grid_size, 1]))
    kpts.verbose = logging.INFO
    kpts.build(space_group_symmetry=True)
    k_twists_mapped = map_to_first_brillouin_zone(kpts.kpts_ibz, cell.reciprocal_vectors().T)
    k_twists_mapped = np.array(k_twists_mapped)

    # plt.figure()
    # plt.scatter(k_twists_mapped[:, 0], k_twists_mapped[:, 1], s=100)
    # plt.axis("equal")

    # Hand-crafted rules to map k-points to symm-equiv points into a contiguos subsection of the IBZ
    # This only works for Graphene, but eg not for Boron-Nitride which doesn't have the same symmetry
    # 1) 60 deg rotation
    phi = np.arctan2(k_twists_mapped[:, 1], k_twists_mapped[:, 0]) * 180 / np.pi
    rotate_down = np.abs(phi % 360 - 180) < 3
    k_twists_mapped[rotate_down] = k_twists_mapped[rotate_down] @ get_rotmat(-60)

    # 2) Mirror x-axis
    mirror_x = k_twists_mapped[:, 0] > 0
    k_twists_mapped[mirror_x, 0] *= -1

    # plt.scatter(k_twists_mapped[:, 0], k_twists_mapped[:, 1], s=30)

    k_twists = k_twists_mapped @ cell.a.T / (2 * np.pi)  # missing transpose? should be like: cell.a.T / (2 * np.pi)
    # assert np.all(np.isclose(k_twists @ cell.reciprocal_vectors().T, k_twists_mapped, atol=1e-5))

    return k_twists, kpts.weights_ibz


def format_k_string(k):
    return ",".join([f"{ki:.2f}" for ki in k])


plt.close("all")
supercells = [[1, 1, 1], [2, 2, 1], [3, 3, 1]]
for tw_grid in [12]:
    for supercell in supercells:
        sc_string = "{0}x{1}x{2}".format(*supercell)
        all_geoms = []
        R, Z, lattice_prim = graphene_geometry()
        config = PhysicalConfig(
            R=R, Z=Z, periodic=dict(lattice_prim=list(map(list, lattice_prim)), supercell=supercell)
        ).get_expanded_if_supercell()
        k_twists, weights = get_k_twists(tw_grid, lattice=config.periodic.lattice, R=config.R, Z=config.Z)
        with np.printoptions(precision=3, suppress=True):
            print(k_twists)
        print(f"Weights: {weights}")

        for tw in k_twists:
            periodic = PeriodicLattice(lattice_prim, supercell=supercell, k_twist=tw)
            geom = Geometry(
                R,
                Z,
                periodic=periodic,
                name=f"deepsolid_benchmarks_C_graphene_{sc_string}_k={format_k_string(tw)}",
                comment=f"deepsolid_benchmarks_C_graphene_{sc_string}_k={format_k_string(tw)}",
            )
            all_geoms.append(geom)
        gamma_geoms = [g for g in all_geoms if sum(g.periodic.k_twist) == 0]
        assert len(gamma_geoms) == 1
        print(f"gam geoms   {len(gamma_geoms)}")
        print(f"all geoms   {len(all_geoms)}")
        print(f"Weights: {weights}")
        dataset_geoms = GeometryDataset(
            all_geoms, weights=weights, name=f"deepsolid_benchmarks_C_graphene_{sc_string}_{len(k_twists)}twists"
        )

        datasets = [dataset_geoms]

        plt.figure()
        plt.scatter(k_twists[:, 0], k_twists[:, 1], s=weights * 1000)
        # rec_vectors = np.linalg.inv(lattice_prim).T * 2 * np.pi
        # plt.arrow(0, 0, rec_vectors[0, 0], rec_vectors[0, 1], color='k', width=0.1, length_includes_head=True)
        # plt.arrow(0, 0, rec_vectors[1, 0], rec_vectors[1, 1], color='k', width=0.1, length_includes_head=True)
        plt.axis("equal")
        plt.xlabel("k0")
        plt.ylabel("k1")
        save_geometries(all_geoms)
        save_datasets(datasets, overwite_existing=True)
