# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_theme(style="whitegrid")

df = pd.read_csv("data/Lih_solids_data.csv", sep=",")
df["tabc_weight"] = df["tabc_weight"].fillna(1.0)
has_sfc_correction = df["sfc_correction"].notnull()
df.loc[has_sfc_correction, "E_corr"] = df["E_mean"] + df["sfc_correction"] + df["zpve"]
df.loc[~has_sfc_correction, "E_corr"] = df["E_mean"] + df["hf_correction"] + df["zpve"]
df["E_corr_weighted"] = df["E_corr"] * df["tabc_weight"]
df = df.groupby(["method", "geom"])["E_corr_weighted"].sum().reset_index()
df = df.rename(columns={"E_corr_weighted": "E"})

E_atoms = -7.47798 - 0.4997908
E_cohesive_exp = (-175.6, -174.9)
error_exp = [np.min(E_cohesive_exp) - np.mean(E_cohesive_exp), np.max(E_cohesive_exp) - np.mean(E_cohesive_exp)]
df["E_cohesive"] = (df["E"] - E_atoms) * 1000
df["error_cohesive"] = df["E_cohesive"] - np.mean(E_cohesive_exp)
df_min = df[np.abs(df.geom - 7.6) < 0.1]


plt.close("all")
fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
for method in df.method.unique():
    df_method = df[df.method == method]
    ax.plot(df_method["geom"], df_method["E_cohesive"], label=method, marker="o")

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
methods = dict(
    deepsolid_lih222=("$\\mathbf{DeepSolid}$\n(small cell)", "C0"),
    deepsolid_lih333=("$\\mathbf{DeepSolid}$\n(large cell)", "navy"),
    moon_tao_lih222=("$\\mathbf{Our\\;work}$\n(small cell)", "C1"),
    moon_tao_lih333=("$\\mathbf{Our\\;work}$\n(large cell)", "chocolate"),
)
for ind_method, (method, (label, color)) in enumerate(methods.items()):
    df_method = df_min[df_min.method == method]
    ax.bar([ind_method], np.abs(df_method["error_cohesive"]), label=label, color=color)

ax.axhline(0, color="black", ls="-")
ax.axhline(max(error_exp), color="gray", ls="-", zorder=0)
ax.fill_between(
    [-0.5, 3.5],
    [0] * 2,
    [max(error_exp)] * 2,
    color="gray",
    alpha=0.5,
    zorder=0,
)

ax.grid(False, axis="x")
ax.set_xlim([-0.5, 3.5])
ax.set_yticks([0, 2, 4, 6, 8])
ax.set_xticks(range(len(methods)))
ax.set_xticklabels([label for label, _ in methods.values()])
ax.set_ylabel("error vs. experiment / mHa")
fig.tight_layout()
fig.savefig("/home/mscherbela/ucloud/results/defense/lih_solids.png", bbox_inches="tight", dpi=400)
