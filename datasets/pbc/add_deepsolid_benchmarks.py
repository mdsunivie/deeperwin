from deeperwin.run_tools.geometry_database import (
    Geometry,
    GeometryDataset,
    PeriodicLattice,
    save_datasets,
    save_geometries,
)
import numpy as np

from deeperwin.run_tools.geometry_utils import BOHR_IN_ANGSTROM

LATTICE_VECTORS = {
    "LiH,rocksalt": [(0.0, 2.0305, 2.0305), (2.0305, 0.0, 2.0305), (2.0305, 2.0305, 0.0)],
    "Li,bcc": [(3.436, 0.0, 0.0), (0.0, 3.436, 0.0), (0.0, 0.0, 3.436)],
    "Be,hexagonal": [(2.2598, 0.0, 0.0), (-1.1299, 1.9571, 0.0), (0.0, 0.0, 3.5699)],
    "C,diamond": [(0.0, 1.7869, 1.7869), (1.7869, 0.0, 1.7869), (1.7869, 1.7869, 0.0)],
    "C,graphene": [(2.1315, -1.2306, 0.0), (2.1315, 1.2306, 0.0), (0, 0, 52.9177)],
}
ATOMS = {
    "LiH,rocksalt": (("Li", (0.0, 0.0, 0.0)), ("H", (2.0305, 2.0305, 2.0305))),
    "Li,bcc": (("Li", (0.0, 0.0, 0.0)), ("Li", (1.718, 1.718, 1.718))),
    "Be,hexagonal": (("Be", (1.1299, 0.6524, 2.6774)), ("Be", (0.0, 1.3047, 0.8925))),
    "C,diamond": (("C", (0.0, 0.0, 0.0)), ("C", (0.8934, 0.8934, 0.8934))),
    "C,graphene": (("C", (1.421, 0.0, 0.0)), ("C", (2.842, 0.0, 0.0))),
}
# %%
REF_ENERGIES = """LiH,rocksalt -8.5165
Li,bcc       -15.34486
Be,hexagonal -30.2416
C,diamond   -75.4009
C,graphene  -76.0350"""
REF_ENERGIES = {key: float(net) for key, net in map(str.split, REF_ENERGIES.split("\n"))}
REF_ENERGIES
# %%
key = "C,graphene"
for z, coord in ATOMS[key]:
    print(z, coord)
# %%
from pyscf.data import elements

elements.NUC
all_geoms = []
for key in ATOMS.keys():
    symbols, structure = key.split(",")
    atoms = ATOMS[key]
    lattice_prim = LATTICE_VECTORS[key]
    Z, R = zip(*[(elements.NUC[symb], pos) for symb, pos in ATOMS[key]])
    lattice_prim = np.array(lattice_prim) / BOHR_IN_ANGSTROM
    R = np.array(R) / BOHR_IN_ANGSTROM

    periodic = PeriodicLattice(lattice_prim, supercell=[1, 1, 1], k_twist=[0, 0, 0])
    geom = Geometry(
        R,
        Z,
        periodic=periodic,
        name=f"deepsolid_benchmarks_{symbols}_{structure}_1x1x1",
        comment=f"deepsolid_benchmarks_{symbols}_{structure}_1x1x1",
        E_ref=REF_ENERGIES[key],
        E_ref_source="deepsolid_supplementary",
    )
    all_geoms.append(geom)

dataset = GeometryDataset(all_geoms, name="deepsolid_benchmarks_1x1x1")

save_geometries(all_geoms, overwite_existing=True)
save_datasets([dataset], overwite_existing=True)


# %%
# %%
def fcc_lattice(a, Z):
    R = [[0, 0, 0], [a / 2, a / 2, a / 2]]
    return R, Z, a / 2 * (1 - np.eye(3))


def format_k_string(k):
    return ",".join([f"{ki:.2f}" for ki in k])


lattice_constants = np.array([3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8]) * (1 / BOHR_IN_ANGSTROM)
print(lattice_constants)

supercells = [[2, 2, 2]]

for supercell in supercells:
    sc_string = "{0}x{1}x{2}".format(*supercell)
    all_geoms = []
    for a in lattice_constants:
        R, Z, lattice_prim = fcc_lattice(a, [3, 1])
        periodic = PeriodicLattice(lattice_prim, supercell=supercell, k_twist=None)
        geom = Geometry(
            R,
            Z,
            periodic=periodic,
            name=f"deepsolid_benchmarks_LiH_fcc_{sc_string}_a={a:.1f}_k={format_k_string([0, 0, 0])}",
            comment=f"deepsolid_benchmarks_LiH_fcc_{sc_string}_a={a:.1f}_k={format_k_string([0, 0, 0])}",
        )
        all_geoms.append(geom)
    gamma_geoms = [g for g in all_geoms if sum(g.periodic.k_twist) == 0]
    assert len(gamma_geoms) == len(lattice_constants)
    print(f"gam geoms   {len(gamma_geoms)}")

    dataset_geoms = GeometryDataset(gamma_geoms, name=f"deepsolid_benchmarks_LiH_fcc_{sc_string}_8geom_gamma")

    datasets = [dataset_geoms]
    save_geometries(all_geoms)
    save_datasets(datasets)
