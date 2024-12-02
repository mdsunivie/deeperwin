# %%
from dataloader import iter_data_buckets
import h5py
import numpy as np
import matplotlib.pyplot as plt

# data_keys = ['ccsd(t)_cbs.energy', 'wb97x_dz.energy','wb97x_dz.forces'] # The coupled cluster ANI-1ccx data set (https://doi.org/10.1038/s41467-019-10827-4)
data_keys = ["ccsd(t)_cbs.energy"]  # The coupled cluster ANI-1ccx data set (https://doi.org/10.1038/s41467-019-10827-4)
data_fname = "/home/mscherbela/data/ani_1x/ani1x-release.h5"

count = 0
n_heavy_atoms = np.zeros(50, int)
n_atoms = np.zeros(100, int)
for d in iter_data_buckets(data_fname, keys=data_keys):
    n_geoms = len(d["coordinates"])
    n_heavy = np.sum(d["atomic_numbers"] > 1)
    n_heavy_atoms[n_heavy] += n_geoms
    n_atoms[len(d["atomic_numbers"])] += n_geoms
    count += n_geoms
print(count)

# %%
plt.close("all")
plt.plot(n_heavy_atoms, label="Heavy atoms")
plt.plot(n_atoms, label="All atoms")
plt.xlabel("Number of atoms")
plt.ylabel("Number of geometries")
plt.yscale("symlog")


# %%
with h5py.File(data_fname, "r") as f:
    print(len(f))
    print(f.keys())
    g = f["C3H11N3"]
    print(g.keys())
# %%
