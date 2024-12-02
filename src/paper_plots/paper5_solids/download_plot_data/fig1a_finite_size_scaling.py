# %%
import pandas as pd
import wandb
import re

api = wandb.Api()
runs = api.runs("schroedinger_univie/tao_HChains")


def filter_func(run):
    if run.name.startswith("FSS_NoTwistBug_4-22_1.80_4kgrid_100k_"):
        return True
    if run.name.startswith("FSS_NoTwistBug_4-22_1.80_1kgrid_50k_"):
        return True
    if re.match(r"FSS_4kgrid_50k_ReuseShared_HChainPBC\d+_1.80_4kgrid_\d+", run.name):
        return True
    if re.match(r"FSS_1kgrid_50k_ReuseIndep_.*__HChainPBC\d+_1.80_k=0.000", run.name):
        return True
    return False


runs = [r for r in runs if filter_func(r)]
all_data = []
for run in runs:
    weight = run.config.get("physical.weight_for_shared")
    weight = weight or 1.0
    twist = run.config["physical.periodic.k_twist"][0]
    if ("gamma" in run.name.lower()) or ("1kgrid" in run.name.lower()):
        k_grid = 1
    elif "4kgrid" in run.name.lower():
        k_grid = 4
    else:
        k_grid = run.name.split("_")[-2].replace("kgrid", "")
    all_data.append(
        dict(
            name=run.name,
            k_grid=int(k_grid),
            E=run.summary_metrics["E_mean"],
            E_sigma=run.summary_metrics["E_mean_sigma_corr"],
            n_atoms=run.config["physical.periodic.supercell"][0] * 2,
            twist=twist,
            weight=weight,
            E_weighted=run.summary_metrics["E_mean"] * weight,
            epochs=run.summary_metrics.get("opt_n_epoch", run.summary_metrics["opt_epoch"]),
            is_reuse=run.config.get("reuse.path") is not None,
        )
    )
df = pd.DataFrame(all_data)

df_ref = pd.read_csv("/home/mscherbela/runs/references/Motta_et_al_finite_size_scaling_webplotdigitizer.csv", sep=";")
# %%


df_ref = df_ref.rename(columns=dict(N_atoms="n_atoms"))
df_ref["k_grid"] = 1
df_ref["twist"] = 0
df_merged = df.drop(columns=["name", "E_weighted"])
df_merged["method"] = "DeepErwin"
df_merged = pd.concat([df_ref, df_merged], ignore_index=True)
df_merged.to_csv("/home/mscherbela/ucloud/results/05_paper_solids/figure_data/fig_1a_Hchain_energy.csv", index=False)
