# %%
fname = "128.raw"
with open(fname) as f:
    lines = f.readlines()
lines = [line.strip() for line in lines]

f_out = fname.replace(".raw", ".POSCAR")
with open(f_out, "w") as f:
    f.write(lines[0] + "\n")
    f.write("  " + lines[1] + "\n    ")
    for i in range(9):
        f.write("  " + lines[i + 2])
        if i in [2, 5]:
            f.write("\n    ")
    f.write("\n")
    f.write("  " + lines[11] + "\n")
    f.write("  " + lines[12] + "\n")
    f.write(lines[13] + "\n  ")
    for i in range(14, len(lines)):
        f.write("  " + lines[i])
        if (i - 1) % 3 == 0:
            f.write("\n  ")


# %%
from ase.io.vasp import read_vasp
from deeperwin.run_tools.geometry_database import (
    Geometry,
    PeriodicLattice,
    save_geometries,
    BOHR_IN_ANGSTROM,
)
import copy

fnames = ["16.POSCAR", "32.POSCAR", "64.POSCAR", "128.POSCAR"]
atoms = [read_vasp(f) for f in fnames]
# ase.visualize.view(atoms)

all_geoms = []
for i, a in enumerate(atoms):
    R = a.get_positions() / BOHR_IN_ANGSTROM
    Z = a.get_atomic_numbers()
    n_slab_atoms = len(Z) - 3
    g = Geometry(
        R,
        Z,
        charge=0,
        spin=None,
        periodic=PeriodicLattice(a.get_cell() / BOHR_IN_ANGSTROM),
        comment=f"H2O_on_LiH_{n_slab_atoms}at_Gamma",
        name=f"H2O_on_LiH_{n_slab_atoms}at_Gamma",
    )
    g_slab = copy.deepcopy(g)
    g_slab.R = g_slab.R[:n_slab_atoms]
    g_slab.Z = g_slab.Z[:n_slab_atoms]
    g_slab.comment = g_slab.comment.replace("H2O_on_", "")
    g_slab.name = g_slab.name.replace("H2O_on_", "")
    all_geoms += [g, g_slab]
save_geometries(all_geoms)
