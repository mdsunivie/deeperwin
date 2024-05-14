#%%
import pandas as pd
import numpy as np

THZ_TO_MILLIHARTREE = 0.0001519828500716 * 1000

df = pd.read_csv("graphene_phonon_dos.csv", sep=";")
df = df.sort_values(["thz"])
df["mHa"] = df["thz"] * THZ_TO_MILLIHARTREE

energy = np.linspace(0, df.mHa.max(), 1000)
dos = np.interp(energy, df.mHa, df.dos)

n_atoms = 2
normalization = np.sum(dos) / (3 * n_atoms)
E0 = 0.5 * np.sum(dos * energy) / normalization
print(E0)
