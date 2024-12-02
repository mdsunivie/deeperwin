# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from deeperwin.run_tools.geometry_database import load_energies, load_geometries
# from plot_utils import format_with_SI_postfix

sns.set_theme()

all_geoms = load_geometries()
df_all = load_energies()
df_orca = df_all[df_all.source.str.contains("orca")]
df_ref = df_orca[df_orca.experiment == "CCSD(T)_CBS_23"][["geom", "E"]]
df_ref.rename(columns={"E": "E_ref"}, inplace=True)

experiments = [
    # "2023-04-26_reuse_midimol_18x20",
    # "2023-04-26_reuse_midimol_18x20_dist0.05",
    "2023-04-27_reuse_midimol_699torsions_128k",
    "2023-04-27_reuse_midimol_699torsions_158k",
    "2023-04-27_reuse_midimol_699torsions_64k",
    "2023-05-01_699torsion_nc_by_std_064k",
    # "2023-03-01_gao_shared_TinyMol",
    "2023-03-06_tinymol_v10_ablation_n_pretrain",
]


def get_model(experiment_name):
    if "2023-05-01_699torsion_nc_by_std" in experiment_name:
        return "699torsions_nc_by_std"
    elif "dist0.05" in experiment_name:
        return "18x20_dist0.05"
    elif "18x20" in experiment_name:
        return "18x20"
    elif "699torsions" in experiment_name:
        return "699torsions"
    elif "gao_shared_TinyMol" in experiment_name:
        return "Scherbela et al. (2023)"
    elif "tinymol_v10" in experiment_name:
        return "Scherbela et al. (2023)"


df_all = df_all[df_all.experiment.isin(experiments)]
df_all["model"] = df_all.experiment.apply(get_model)
df_all["include"] = True
df_all.loc[
    df_all.model.str.contains("Scherbela")
    & ((df_all.n_pretrain_variational != 256_000) | (df_all.reuse_from != "256kshared_tinymol_v10")),
    "include",
] = False
df_all = df_all.query("include == True")

# TODO use CBS energies
# df_ref = pd.DataFrame([dict(geom="b2c66515e32b29b4d09cfa60705cd14c", E_ref=-116.429215980865)])
# df_ref["E_ref"] -= 0.180
df_all = pd.merge(df_all, df_ref, on="geom", how="left")
df_all["error"] = (df_all.E - df_all.E_ref) * 1000
df_all["sigma_error"] = df_all.E_sigma * 1000
df_orca = pd.merge(df_orca, df_ref, on="geom", how="left")
df_orca["error"] = (df_orca.E - df_orca.E_ref) * 1000
df = df_all.groupby(["model", "geom", "n_pretrain_variational", "epoch"]).mean()


# %%
colors = ["slategray"] + [f"C{i}" for i in range(10)]
alphas = [0.4, 0.5, 1.0, 1.0]
xticks = [0, 1000, 2000, 4000, 8000, 16000]


plt.close("all")
geom_hashes_4ood = [
    "b2c66515e32b29b4d09cfa60705cd14c",
    "3962fa508c318c990017beb42983662e",
    "151258110b9bc0d82cc4522c7978695a",
    "525c9db5e9b4210e071dfc9bfd66a9e4",
]

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, geom_hash in zip(axes.flatten(), geom_hashes_4ood):
    for ind_model, model in enumerate(df_all.model.unique()):
        for ind_pretrain, n_pretrain in enumerate(sorted(df_all.n_pretrain_variational.unique())):
            df_filt = df_all.query("n_pretrain_variational == @n_pretrain and model == @model and geom == @geom_hash")
            if len(df_filt) == 0:
                continue
            ax.errorbar(
                df_filt.epoch,
                df_filt.error,
                yerr=df_filt.sigma_error,
                marker="o",
                capsize=5,
                color=colors[ind_model],
                alpha=alphas[ind_pretrain],
                label=f"{model} {n_pretrain/1000:.0f}k",
            )
            ax.set_yscale("symlog", linthresh=10)
            ax.set_xscale("symlog", linthresh=1000)

    for ind_ref, ref_method in enumerate(["CCSD(T)_ccpCVDZ", "CCSD(T)_ccpCVTZ", "CCSD(T)_ccpCVQZ"]):
        df_filt = df_orca[df_orca.experiment == ref_method]
        if len(df_filt) > 0:
            color = ["grey", "dimgray", "black"][ind_ref]
            error = df_filt.error.iloc[0]
            ax.axhline(error, color=color, ls="--")
            ax.text(16000, error * 1.02, f"CCSD(T)-{ind_ref+2}Z", ha="right", va="bottom", color=color, fontsize=10)

    ax.legend(loc="upper center")
    ax.set_xticks(xticks)
    # ax.xaxis.set_major_formatter(lambda x, pos: format_with_SI_postfix(x))
    ax.set_xlabel("Finetuning steps")
    ax.set_ylabel("$E - E_\\mathrm{CCSD(T)-CBS}$ / mHa")
    ax.set_title(all_geoms[geom_hash].name)
fig.tight_layout()
fig.savefig("/home/mscherbela/ucloud/results/04_paper_better_universal_wf/figures/fig1_finetuning.png", dpi=200)


# %%
