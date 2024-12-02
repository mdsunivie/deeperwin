import pandas as pd
from deeperwin.run_tools.geometry_database import load_geometries, load_energies
import re

all_geometries = load_geometries()
all_energies = load_energies()

geom_by_size = dict()
for h, g in all_geometries.items():
    match = re.match(r"HChain(\d+)_1\.80$", g.comment)
    if match:
        geom_by_size[int(match.group(1))] = g


df = pd.read_csv("/home/mscherbela/runs/datasets/Gao2023/extensivity.csv")
df = df.drop(columns=["moon_err", "ferminet_err"])
df = df.melt(id_vars=["n_atoms"], value_name="E")
df["geom"] = df.n_atoms.apply(lambda n: geom_by_size[n].hash)
df["geom_comment"] = df.n_atoms.apply(lambda n: geom_by_size[n].comment)
df["source"] = "Gao_etal_2023"
df["experiment"] = df.variable.map(
    dict(
        moon_energy="Gao_etal_2023_HChain_extensivity_Moon", ferminet_energy="Gao_etal_2023_HChain_extensivity_FermiNet"
    )
)
df["method"] = "reuse"
df["epoch"] = 0
df["energy_type"] = "eval"
df["batch_size"] = 4096
df = df.drop(columns=["n_atoms", "variable"])
# append_energies(df)
