# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

df = pd.read_csv("data/energies_goldstandard.csv")
E_ref = df["Experiment"]
df["Hartree-Fock"] = 1000 * (df["HF"] - E_ref)
df["FermiNet (DeepMind 2020)"] = 1000 * (df["FermiNet"] - E_ref)
df["Ours (UniVie 2022)"] = 1000 * (df["Ours"] - E_ref)
df["VMC"] = 1000 * (df["VMC_Nemec"] - E_ref)
df["Diffusion Monte Carlo (Nemec 2010)"] = 1000 * (df["DMC_Nemec"] - E_ref)
df["CCSD(T)"] = 1000 * (df["CCSD(T)"] - E_ref)
# Flatten the DataFrame with melt
df = df.melt(
    id_vars=["Molecule"],
    value_vars=[
        # "Hartree-Fock",
        # "VMC",
        "Diffusion Monte Carlo (Nemec 2010)",
        # "CCSD(T)",
        "FermiNet (DeepMind 2020)",
        "Ours (UniVie 2022)",
    ],
    var_name="Method",
    value_name="Error",
)


plt.close("all")
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.barplot(data=df, x="Molecule", y="Error", ax=ax, hue="Method", palette=["gray", "C0", "C1"])
ax.set_ylabel(r"$E - E_\mathrm{experiment}$  / mHa")
ax.legend().set_title(None)
ax.legend().set_loc("upper left")
ax.set_ylim([0, 25])
fig.tight_layout()
fig.savefig("/home/mscherbela/ucloud/results/defense/absolute_energies_v2.png", bbox_inches="tight", dpi=400)
