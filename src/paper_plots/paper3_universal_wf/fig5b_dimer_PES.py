# %%
from deeperwin.run_tools.geometry_database import load_energies, get_all_geometries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm

sns.set_theme(style="whitegrid")

df = load_energies()
experiments = [
    ("2023-05-26_psiformer_indep", "PsiFormer (100k steps per geom.)", "C3", "-", "v"),
    ("Booth_2011_JCP_FCIQMC_cc-pVTZ", "Booth et al., FCIQMC", "C0", "-", "s"),
    ("2023-05-26_C2_dimer_Booth2011_gao_reuseshared_from_tinymol_500k", None, None, None, None),
    # ("2023-05-26_C2_dimer_gao_reuseshared_from_tinymol_500k", None, None, None, None),
    # ("CCSD(T)_ccpCVQZ", "CCSD(T)-4Z", "limegreen", "-", "x"),
    ("CCSD(T)_CBS_234", "CCSD(T)-CBS", "C2", "-", "x"),
    # ("UHF_CBS_2345", "UHF-CBS", "limegreen", "-", "x"),
    # ("RHF_ccpCV5Z", "RHF-5Z", "limegreen", "-", "x"),
]

geom_datasets = [
    "C2_dimer_Booth2011_14geoms",
    # "C2_dimer_10geoms"
]
psiformer_epochs = 52000
all_geom_hashes = [g.hash for g in get_all_geometries(geom_datasets)]
experiment_names = [e[0] for e in experiments]
df = df[df.experiment.isin(experiment_names) & df.geom.isin(all_geom_hashes)]
df = df[(df.experiment != "2023-05-26_psiformer_indep") | (df.epoch == psiformer_epochs)]
df["R"] = df.geom_comment.apply(lambda x: x.split("_")[-1]).astype(float)
df = df.sort_values(["experiment", "R"])

n_geoms = 14

df_ref = df[df.experiment == "2023-05-26_psiformer_indep"][["R", "E"]].rename(columns={"E": "E_ref"})
df_ref["E_ref_rel"] = (df_ref.E_ref - df_ref.E_ref.min()) * 1000
df = pd.merge(df, df_ref, on="R", how="left")

dpe_epochs = [
    0,
    14000,
    #   56000
]
# dpe_epochs = [0, 4000, 16000, 64000]


plt.close("all")
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
ax_absolute, ax_relative, ax_error = axes

for (
    ind_epoch,
    epoch,
) in enumerate(dpe_epochs):
    df_filt = df[(df.epoch == epoch) & (df.source.str.contains("dpe"))]
    color = matplotlib.cm.get_cmap("Oranges")(0.4 + 0.6 * ind_epoch / (len(dpe_epochs) + 1))
    E_relative = (df_filt.E - df_filt.E.min()) * 1000
    error_relative = E_relative - df_filt.E_ref_rel

    if epoch == 0:
        label = "Our work (zero-shot)"
        ls = "--"
    else:
        n = (epoch / n_geoms) / 1000
        label = f"Our work ({n:.0f}k steps per geom.)"
        ls = "-"
    ax_absolute.plot(df_filt.R, df_filt.E, label=label, color=color, marker="o", ls=ls)
    ax_relative.plot(df_filt.R, E_relative, label=label, color=color, marker="o", ls=ls)
    ax_error.plot(df_filt.R, error_relative, label=label, color=color, marker="o", ls=ls)

for experiment, label, color, ls, marker in experiments:
    if "gao_reuse" in experiment:
        continue
    df_filt = df[df.experiment == experiment]
    E_relative = (df_filt.E - df_filt.E.min()) * 1000
    error_relative = E_relative - df_filt.E_ref_rel

    ax_absolute.plot(df_filt.R, df_filt.E, label=label, color=color, ls=ls, marker=marker)
    ax_relative.plot(df_filt.R, E_relative, label=label, color=color, ls=ls, marker=marker)
    ax_error.plot(df_filt.R, error_relative, label=label, color=color, ls=ls, marker=marker)

for ax, ylabel, ylim, title, sublabel in zip(
    axes,
    [
        "$E$ / Ha",
        r"$(E - E_\mathrm{min})$ / mHa",
        r"$\left[(E - E_\mathrm{min}) - (E^\mathrm{\psi F} - E^{\psi F}_\mathrm{min})\right]$ / mHa",
    ],
    [[-75.95, -75.2], [-20, 500], [-15, 35]],
    ["Absolute energy", "Relative energy", "Deviation from PsiFormer"],
    "abc",
):
    if sublabel == "a":
        ax.legend()
    ax.set_xlabel("Bond length / bohr")
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.text(0.0, 1.02, f"{sublabel}", transform=ax.transAxes, fontweight="bold", va="bottom", ha="left", fontsize=14)
fig.tight_layout()


fname = "/home/mscherbela/ucloud/results/03_paper_unversal_wavefuncion/figures/C2_PES.png"
fig.savefig(fname, dpi=300, bbox_inches="tight")
fig.savefig(fname.replace(".png", ".pdf"), bbox_inches="tight")


# %%
