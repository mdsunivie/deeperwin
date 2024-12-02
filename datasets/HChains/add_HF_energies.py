import pandas as pd
from deeperwin.run_tools.geometry_database import load_geometries, load_energies, append_energies, load_datasets

all_geometries = load_geometries()
dataset = load_datasets()["HChain_equidist_2-28_1.80_14geoms"]
geometries = dataset.get_geometries(all_geometries)
all_energies = load_energies()

# HF_energies = [-1.1032259464263916,-2.168766975402832,-3.235598087310791,-4.303471565246582,-5.37182092666626,-6.440435409545898,-7.509200572967529,-8.578055381774902,-9.646966934204102,-10.715911865234375,-11.784880638122559,-12.853863716125488,-13.922856330871582,-14.991854667663574]
HF_energies = [
    -1.108005523681640,
    -2.174998760223388,
    -3.244351148605346,
    -4.314764976501465,
    -5.385707855224609,
    -6.456922054290771,
    -7.528292655944824,
    -8.599754333496094,
    -9.671272277832031,
    -10.74282550811767,
    -11.81440067291259,
    -12.88599109649658,
    -13.95759010314941,
    -15.02919578552246,
]
data = []
for g in geometries:
    data.append(
        dict(
            E=HF_energies[g.n_el // 2 - 1],
            geom=g.hash,
            geom_comment=g.comment,
            source="HF",
            method="indep",
            experiment="HF_HChains_6-31G**",
        )
    )
df = pd.DataFrame(data)
append_energies(df)
