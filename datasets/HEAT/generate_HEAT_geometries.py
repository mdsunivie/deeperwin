from deeperwin.geometries import parse_coords
from deeperwin.run_tools.geometry_database import Geometry, GeometryDataset
import os
import numpy as np

np.random.seed(0)

heat_dir = "/home/mscherbela/develop/deeperwin_jaxtest/datasets/geometries/HEAT"
geometries = []
for fname in os.listdir(heat_dir):
    full_name = os.path.join(heat_dir, fname)
    if os.path.isfile(full_name) and fname.startswith("coord"):
        content = open(full_name).read()
        R, Z = parse_coords(content)
        molecule_name = fname.split(".")[-1].upper()
        geom = Geometry(R, Z, comment=f"{molecule_name}_HEAT", name=molecule_name)
        geometries.append(geom)

print(f"All geometries: {len(geometries)}")

geometries = [g for g in geometries if 9 not in g.Z]
print(f"No Fluorine   : {len(geometries)}")

test_molecules = ["CCH", "HCN"]

# geometries = [g for g in geometries if g.n_el % 2 == 0]
# print(f"Even nr of el : {len(geometries)}")

for g in geometries:
    print(g.comment)

dataset = GeometryDataset(geometries, "HEAT_HCNO_{len")
