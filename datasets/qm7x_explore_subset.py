import h5py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import collections
from deeperwin.utils.utils import PERIODIC_TABLE

sns.set_theme()
from ase import Atoms
from ase.visualize import view


def get_sum_formula(Z):
    hist = collections.Counter()
    for Z_ in Z:
        hist[Z_] += 1
    s = ""
    for Z_, count in hist.items():
        s += f"{PERIODIC_TABLE[Z_-1]}{count}"
    return s


data = []
f = h5py.File("/Users/leongerard/Desktop/data_universal_wf/qm7x/1000.hdf5", "r")
print(len(f.values()))
for name, qm7_data in f.items():
    print(len(qm7_data.keys()))
    for i, geom_name in enumerate(qm7_data.keys()):
        if i % 1000 == 0:
            print(i)
        geom = qm7_data[geom_name]
        Z = geom["atNUM"][...]
        data.append(
            dict(
                name=name,
                formula=get_sum_formula(Z),
                E_PBE0_MBD=geom["ePBE0+MBD"][...][0],
                HOMO=geom["eH"][...][0],
                LUMO=geom["eL"][...][0],
                Z_max=max(Z),
                Z_min=min(Z),
                n_el=sum(Z),
                rmsd=geom["sRMSD"][...][0],
                is_opt=name.endswith("opt"),
            )
        )
df = pd.DataFrame(data)
df = df.set_index("name")

# %%
plt.close("all")
sns.histplot(df, binwidth=2)
df_filt = df.query("is_opt")
print(len(df_filt))

geoms = []
for name in df_filt.index:
    geoms.append(
        Atoms(
            positions=f[name]["atXYZ"][...],
            numbers=f[name]["atNUM"][...],
        )
    )
view(geoms)
