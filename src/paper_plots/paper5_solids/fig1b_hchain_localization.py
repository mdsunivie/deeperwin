# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deeperwin.run_tools.geometry_database import load_geometries
from deeperwin.utils.plotting import get_discrete_colors_from_cmap
from typing import Iterable
import os
from scipy.optimize import curve_fit

all_geoms = load_geometries()


def get_data_from_geometry(row):
    geom = all_geoms[row.geom]
    row["n_atoms"] = geom.n_atoms * geom.periodic.supercell[0]
    row["R"] = geom.R[1][0] - geom.R[0][0]
    row["k"] = geom.periodic.k_twist[0]
    return row


def z_model(a, ac, c1, c2):
    x = a - ac
    z_high = 1 - np.exp(-c1 * x - c2 * x**2)
    return np.where(a > ac, z_high, 0)


# Data from DPE runs
csv_fnames = [
    ("plot_data/fig1b_HChain_pretrain.csv", "ours_pretrain"),
    ("plot_data/fig1b_Reuse_HChain40.csv", "ours_reuse"),
]
dpe_data: Iterable[pd.DataFrame] = []
for fname, method in csv_fnames:
    df = pd.read_csv(fname)
    if "n_pretrain" not in df:
        df["n_pretrain"] = None
    columns = dict(
        loc_abs_0="z",
        loc_abs_0_sigma_corr="z_sigma",
        geom="geom",
        weight="weight",
        epoch="epochs",
        n_pretrain="n_pretrain",
    )
    df = df[list(columns.keys())].rename(columns=columns)
    df["method"] = method
    if "reuse" in method:
        df["method"] += df.n_pretrain.apply(lambda x: f"_{x/1000:.0f}kpre")
    df["method"] += df.epochs.apply(lambda x: f"_{x/1000:.0f}k")
    dpe_data.append(df)
dpe_data = pd.concat(dpe_data)
dpe_data = dpe_data.apply(get_data_from_geometry, axis=1)
# dpe_data = dpe_data[dpe_data.k > 0]
dpe_data["weight_sqr"] = dpe_data.weight**2
dpe_data["z_weighted"] = dpe_data.z * dpe_data.weight
dpe_data["z_sigma_sqr_weighted"] = dpe_data.z_sigma**2 * dpe_data.weight_sqr

# Pivot the data to get twist-average results
groupings = ["method", "epochs", "n_atoms", "R"]
pivot = (
    dpe_data.groupby(groupings)
    .agg(
        z_weighted=("z_weighted", "sum"),
        weight=("weight", "sum"),
        weight_sqr=("weight_sqr", "sum"),
        z_sigma_sqr_weighted=("z_sigma_sqr_weighted", "sum"),
    )
    .reset_index()
)
pivot["z"] = pivot.z_weighted / pivot.weight
pivot["z_sigma"] = np.sqrt(pivot.z_sigma_sqr_weighted / pivot.weight_sqr)

# Data from other works
df_ref = pd.read_csv("plot_data/Motta_et_al_metal_insulator_transition.csv", sep=";")
df_ref["method"] = df_ref.source + ", " + df_ref.method
df = pd.concat([pivot, df_ref])
df = df.sort_values(["method", "n_atoms", "R"])

colors_red = get_discrete_colors_from_cmap(4, "Reds", 0.3, 1.0)
colors_blue = get_discrete_colors_from_cmap(3, "Blues", 0.6, 1.0)
colors_orange = get_discrete_colors_from_cmap(3, "Oranges", 0.4, 1.0)


curves_to_plot = [
    ("Motta (2020), AFQMC", 40, "-", "s", "black", "AFQMC, $N_\\mathrm{atoms}$=40"),
    ("Motta (2020), DMC", 40, "-", "o", "slategray", "DMC, $N_\\mathrm{atoms}$=40"),
    ("ours_pretrain_200k", 12, ":", "none", colors_blue[0], None),
    ("ours_pretrain_200k", 16, ":", "none", colors_blue[1], "Ours pre-train, $N_\\mathrm{atoms}$=12-20"),
    ("ours_pretrain_200k", 20, ":", "none", colors_blue[2], None),
    ("ours_reuse_200kpre_5k", 40, "-", "none", "red", "Ours fine-tune, $N_\\mathrm{atoms}$=40"),
]

text_labels = [
    (3.1, 0.56, "$N_\\mathrm{atoms}$=12"),
    (3.1, 0.66, "$N_\\mathrm{atoms}$=16"),
    (3.1, 0.75, "$N_\\mathrm{atoms}$=20"),
    (3.1, 0.84, "$N_\\mathrm{atoms}$=40"),
]

plt.close("all")
fig, ax = plt.subplots(1, 1, figsize=(5, 4))
for method, n_atoms, ls, marker, color, label in curves_to_plot:
    df_plot = df[(df.method == method) & (df.n_atoms == n_atoms)]
    if len(df_plot) == 0:
        continue
    if any(~np.isnan(df_plot.z_sigma)) > 0:
        yerr = 2 * df_plot.z_sigma
    else:
        yerr = None

    if n_atoms == 40:
        popt, pcov = curve_fit(z_model, df_plot.R, df_plot.z, p0=[1.2, 1.0, 0.1], bounds=(0, np.inf))
        print(f"{method:<20}, N={n_atoms}, ac={popt[0]:.2f} +- {np.sqrt(pcov[0][0]):.2f} a0")
        a_fit = np.linspace(1.0, 3.6, 500)
        ax.plot(a_fit, z_model(a_fit, *popt), color=color, ls=ls)
    ax.errorbar([0], [0], yerr=None if yerr is None else [0], label=label, color=color, ls=ls, marker=marker, ms=4, capsize=5, capthick=1) # dummy entry for legend
    ls = ls if n_atoms != 40 else "none"
    ax.errorbar(
        df_plot.R, df_plot.z, yerr=yerr, color=color, ls=ls, marker=marker, ms=4, capsize=3, capthick=1
    )


for (x, y, text), color in zip(text_labels, colors_blue + ["red"]):
    text_box = ax.text(x, y, text, fontsize=8, ha="left", va="center", color=color)
    text_box.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor="none"))
ax.legend(loc="upper left")
for z in [0, 1]:
    ax.axhline(z, color="gray", lw=1)
ax.set_xlabel("$R$ / $a_0$")
ax.set_xlim([1.0, 3.65])
ax.set_ylabel("polarization $|z|$")
ax.text(0, 1.0, "b", transform=ax.transAxes, ha="left", va="bottom", fontsize=16, fontweight="bold")

fig_k, axes_k = plt.subplots(1, 2, figsize=(8, 4))
for n_atoms, run, ax_k in zip([20, 40], ["ours_pretrain_200k", "ours_reuse_200kpre_5k"], axes_k):
    df_k = dpe_data[(dpe_data.n_atoms == n_atoms) & (dpe_data.method == run)]
    for R in np.sort(df_k.R.unique())[::-1]:
        df_R = df_k[df_k.R == R]
        ax_k.plot(df_R.k, df_R.z, label=f"R={R}" if n_atoms == 20 else None, marker="o")
    ax_k.set_ylim([-0.1, 1.05])
    ax_k.set_xlabel("twist $k$ [$\\pi / R$]")
    ax_k.set_ylabel("polarization $|z|$")
    for z in [0, 1]:
        ax_k.axhline(z, color="gray", lw=1)
    ax_k.set_title(f"$N_\\mathrm{{atoms}}$={n_atoms}")
fig_k.legend(loc="upper center", ncol=4)
fig_k.tight_layout()
fig_k.subplots_adjust(top=0.75)


save_dir = "plot_output"
fig_k.savefig(f"{save_dir}/SI_HChains_k_resolved_z.pdf", bbox_inches="tight")
fig_k.savefig(f"{save_dir}/SI_HChains_k_resolved_z.png", bbox_inches="tight", dpi=300)
fig.savefig(f"{save_dir}/fig1b_HChains_MIT.pdf", bbox_inches="tight")
fig.savefig(f"{save_dir}/fig1b_HChains_MIT.png", bbox_inches="tight", dpi=300)
