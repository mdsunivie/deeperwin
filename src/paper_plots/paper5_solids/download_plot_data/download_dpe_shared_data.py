# %%
import wandb
import pandas as pd

project = "schroedinger_univie/tao_periodic_shared"
api = wandb.Api()
all_runs = [r for r in api.runs(project)]


# %% Graphene 2x2
original_runs = [
    r
    for r in all_runs
    if r.name.startswith(
        "MoonV6_shared_200k_rr_weight_v5_lr01_highdamp_folx_deepsolid_benchmarks_C_graphene_2x2x1_19twists_20_000"
    )
]
restart_runs = [r for r in all_runs if r.name.startswith("MoonV6_shared_200k_rr_weight_v5_lr01_highdamp_folx_restart")]

all_data = []
for r in original_runs + restart_runs:
    print(r.name)
    geom = r.config["physical.comment"].split("_")[-2].split("=")[-1]
    twists = "_".join([f"{k:.3f}" for k in r.config["physical.periodic.k_twist"]])
    df = [h for h in r.scan_history(["opt_epoch", "opt_n_epoch", "opt_E_mean", "opt_E_var"])]
    df = pd.DataFrame(df)
    df["geom_hash"] = r.config["physical.comment"].split("_")[0]
    df["twist"] = twists
    if not "restart" in r.name:
        df = df[df.opt_n_epoch < 59_000]
    all_data.append(df)
df = pd.concat(all_data, ignore_index=True)
df.to_csv("../plot_data/graphene_2x2_shared.csv", index=False)

# %% LiH 2x2x2
runs = [
    r
    for r in all_runs
    if r.name.startswith("MoonV6_shared_200k_rr_weight_v2_deepsolid_benchmarks_LiH_fcc_2x2x2_8geoms_10twi")
]
all_data = []
for r in runs:
    print(r.name)
    geom = r.config["physical.comment"].split("_")[-2].split("=")[-1]
    twists = "_".join([f"{k:.3f}" for k in r.config["physical.periodic.k_twist"]])
    df = [h for h in r.scan_history(["opt_epoch", "opt_n_epoch", "opt_E_mean", "opt_E_var"])]
    df = pd.DataFrame(df)
    df["geom"] = geom
    df["geom_hash"] = r.config["physical.comment"].split("_")[0]
    df["twist"] = twists
    all_data.append(df)
df = pd.concat(all_data, ignore_index=True)
df.to_csv("../plot_data/LiH_2x2x2.csv", index=False)
