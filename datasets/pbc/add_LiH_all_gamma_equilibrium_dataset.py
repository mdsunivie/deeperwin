# %%
import itertools
from deeperwin.run_tools.geometry_database import Geometry, GeometryDataset, PeriodicLattice, save_datasets
import numpy as np


def fcc_lattice(a):
    return a / 2 * (1 - np.eye(3))


def get_rocksalt(a, Z):
    assert len(Z) == 2
    R = [[0, 0, 0], [a / 2, a / 2, a / 2]]
    return R, Z, fcc_lattice(a)


def format_k_string(k):
    return ",".join([f"{ki:.2f}" for ki in k])


def make_dataset(supercells, reduced=True, overwrite=False):
    # print(f"========== {supercell=} ==========")
    all_geoms = []

    a = equ_lat_const
    R, Z, lattice_prim = get_rocksalt(a, [3, 1])
    k_twist = [0, 0, 0]
    for supercell in supercells:
        sc_string = "{0}x{1}x{2}".format(*supercell)
        periodic = PeriodicLattice(lattice_prim, supercell=supercell, k_twist=k_twist)
        name = f"LiH_fcc_{sc_string}_a={a:.1f}_k={format_k_string(k_twist)}"
        geom = Geometry(R, Z, periodic=periodic, name=name, comment=name + "_LiH_rocksalt.py")
        all_geoms.append(geom)
    print(f"all geoms   {len(all_geoms)}")

    dataset = GeometryDataset(all_geoms, name="LiH_fcc_all_sc_equ_gamma")
    save_datasets([dataset], overwite_existing=overwrite)


if __name__ == "__main__":
    ANGSTROM2BOHR = 1.88973
    equ_lat_const = 4.0 * ANGSTROM2BOHR

    # k_twists = [[0,0,0], [1/3, 0, 0], [1/3, 1/3, 0], [1/3, 1/3, 1/3]]
    # All three length tuples of 1's and 2's
    supercells = list(itertools.product([1, 2], repeat=3))
    twist_grid_sizes = [2, 3]
    # %%

    overwrite = False
    make_dataset(supercells, overwrite=overwrite)
