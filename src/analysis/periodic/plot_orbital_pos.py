#%%
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
import re
from deeperwin.run_tools.geometry_database import load_geometries, load_datasets

dataset = "deepsolid_benchmarks_C_graphene_2x2x1_19twists"
geoms = load_datasets()[dataset].get_geometries()
all_twists_frac = np.array([g.periodic.k_twist for g in geoms])
#%%

def to_array(buffer):
    buffer = re.sub(' +', ' ', buffer)
    buffer = buffer.replace(" ", ",")
    buffer = buffer.replace("[,", "[")
    buffer = buffer.replace("\n", "")
    return np.array(literal_eval(buffer))

fname = "/home/mscherbela/tmp/graphene.out"

data = []
buffer = ""
include_line = False
for line in open(fname):
    if "Localized orbital positions: [[" in line:
        line = line[line.index("["):]
        include_line = True
    if include_line:
        buffer += line
    if include_line and ("]]" in line):
        include_line = False
        if len(data) == 6:
            print(buffer)
        data.append(to_array(buffer))
        buffer = ""

data = data[:-1]
data = np.array(data)
data = np.swapaxes(data, 1, 2)

plt.close("all")
fig, (ax_R, ax_k) = plt.subplots(1, 2, figsize=(12, 8), width_ratios=[1, 1])

R = np.array([[2.6853, 0.0, 0.0], [5.3706, 0.0, 0.0]])
lat =  np.array([[4.02795, -2.3255, 0.0], [4.02795, 2.3255, 0.0], [0.0, 0.0,
            99.99995]])
indices = np.meshgrid(np.arange(-1, 3,), np.arange(-1, 3), [0])
indices = np.stack([i.flatten() for i in indices], axis=-1)
R_grid = (indices @ lat)[:, None, :] + R[None, :, :]
R_grid_flat = R_grid.reshape(-1, 3)

ax_R.scatter(R_grid_flat[:, 0], R_grid_flat[:, 1], c="gray", s=400, label="nuc")

ind_bad = np.array([0, 3, 11, 16, 17, 18])
ind_good = [i for i in range(19) if i not in ind_bad]
for i, orb in enumerate(ind_good):
    ax_R.scatter(data[orb, :, 0], data[orb, :, 1],marker="o", color="green", alpha=0.5, s=100, label="good orbitals" if i == 0 else None)
for i, orb in enumerate(ind_bad):
    ax_R.scatter(data[orb, :, 0], data[orb, :, 1], label=f"orb {orb}", marker="x", color=f"C{i}")
ax_R.legend()
ax_R.axis("equal")

rec_vecs = np.linalg.inv(2 * lat).T * 2 * np.pi
k_twist = all_twists_frac @ rec_vecs
ax_k.scatter(k_twist[ind_good, 0], k_twist[ind_good, 1], color="green", label="Good", alpha=0.5)
for i, orb in enumerate(ind_bad):
    ax_k.scatter(k_twist[orb, 0], k_twist[orb, 1], color=f"C{i}", label=f"orb {orb}")



