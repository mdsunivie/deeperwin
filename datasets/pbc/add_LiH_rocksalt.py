# %%
import itertools
from deeperwin.run_tools.geometry_database import (
    Geometry,
    GeometryDataset,
    PeriodicLattice,
    save_datasets,
    save_geometries,
)
import numpy as np


# %%
def fcc_lattice(a):
    return a / 2 * (1 - np.eye(3))


def get_rocksalt(a, Z):
    assert len(Z) == 2
    R = [[0, 0, 0], [a / 2, a / 2, a / 2]]
    return R, Z, fcc_lattice(a)


# %%
from pyscf.pbc import gto
from pyscf.pbc.lib.kpts import KPoints


def get_k_twists(twist_grid_size, supercell=[1, 1, 1], reduced=True):
    if twist_grid_size == 1:
        return np.array([[0.0, 0.0, 0.0]])
    lattice = fcc_lattice(a=1) @ np.diag(supercell)
    cell = gto.Cell(atom="""H 0. 0. 0.""", a=lattice, unit="bohr").build()
    if reduced:
        kpts = KPoints(cell, cell.make_kpts(3 * [twist_grid_size]))
        kpts.build(space_group_symmetry=True)
        # convert to units of reciprocal lattice vectors
        k_twists = kpts.kpts_ibz @ cell.a / (2 * np.pi)
    else:
        kpts = cell.make_kpts(3 * [twist_grid_size])
        k_twists = kpts @ cell.a / (2 * np.pi)
    return k_twists


# %%
def format_k_string(k):
    return ",".join([f"{ki:.2f}" for ki in k])


# LiH_datasets, equilibrium is close to 7.6
# lattice_constants = np.arange(7.2, 8.1, 0.4)


lattice_constants_angstrom = np.array([3.8, 4.0, 4.2])
ANGSTROM2BOHR = 1.88973
lattice_constants = lattice_constants_angstrom * ANGSTROM2BOHR

# %%
# k_twists = [[0,0,0], [1/3, 0, 0], [1/3, 1/3, 0], [1/3, 1/3, 1/3]]
# All three length tuples of 1's and 2's
supercells = list(itertools.product([1, 2], repeat=3))
twist_grid_sizes = [2, 3]


# %%
def make_datasets(lattice_constants, twist_grid_size, supercell, reduced=True, overwrite=False):
    print(f"========== {supercell=} twist_grid={tg}x{tg}x{tg} {reduced=}==========")
    all_geoms = []
    k_twists = get_k_twists(twist_grid_size, supercell=supercell, reduced=reduced)
    with np.printoptions(precision=3, suppress=True):
        print(k_twists)
    sx, sy, sz = supercell
    tw_string = "{0}x{0}x{0}".format(twist_grid_size)
    sc_string = "{0}x{1}x{2}".format(*supercell)
    for a in lattice_constants:
        R, Z, lattice_prim = get_rocksalt(a, [3, 1])
        for k_twist in k_twists:
            periodic = PeriodicLattice(lattice_prim, supercell=supercell, k_twist=k_twist)
            name = f"LiH_fcc_{sc_string}_a={a:.1f}_k={format_k_string(k_twist)}"
            geom = Geometry(R, Z, periodic=periodic, name=name, comment=name + "_LiH_rocksalt.py")
            all_geoms.append(geom)

    gamma_geoms = [g for g in all_geoms if sum(g.periodic.k_twist) == 0]
    equ_lat_const = 4.0 * ANGSTROM2BOHR
    equ_geoms = [g for g in all_geoms if np.isclose(g.periodic.lattice_prim[1, 0], equ_lat_const / 2)]
    print(f"tot geoms   {len(all_geoms)}")
    print(f"equ geoms   {len(equ_geoms)}")
    print(f"gam geoms   {len(gamma_geoms)}")
    num_twists = len(k_twists)

    dataset_geoms = GeometryDataset(gamma_geoms, name=f"LiH_fcc_{sc_string}_3geom_gamma")
    dataset_both = GeometryDataset(all_geoms, name=f"LiH_fcc_{sc_string}_3geom_{num_twists}twist={tw_string}")
    dataset_twist = GeometryDataset(
        equ_geoms, name=f"LiH_fcc_{sc_string}_a{equ_lat_const:.1f}_{num_twists}twist={tw_string}"
    )

    datasets = [dataset_geoms, dataset_both, dataset_twist]
    save_geometries(all_geoms, overwite_existing=overwrite)
    print([d.name for d in datasets])
    save_datasets(datasets, overwite_existing=overwrite)


# %%
# fcc
# Dataset with different size of lattices (1x1x1, 1x1x2, etc up to 2x2x2)
# Different twist grids (2x2x2, 3x3x3), and gamma point
overwrite = False
for supercell in supercells:
    for tg in twist_grid_sizes:
        make_datasets(lattice_constants, tg, supercell, overwrite=overwrite)

# Add non-reduced 1x1x1
make_datasets(lattice_constants, 2, [1, 1, 1], reduced=False, overwrite=overwrite)
