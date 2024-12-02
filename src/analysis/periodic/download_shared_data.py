#%%
import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

api = wandb.Api()
project = "schroedinger_univie/Hbcc_shared"
runs = api.runs(project)

all_opt_keys = ['opt_epoch', 'opt_n_epoch', 'opt_E_mean_smooth', 'opt_E_mean', 'opt_E_var']
all_history = []
for i,run in enumerate(runs):
    name = run.name
    print(f"Downloading {name:<30} ({i+1}/{len(runs)})")

    group_name = "_".join(name.split("_")[:-1])
    geom = int(name.split("_")[-1])
    is_shared = run.config.get("optimization.shared_optimization.use", False)
    pretraining_steps = run.config.get("pre_training.n_epochs", 0)
    scheduling_method = run.config.get("optimization.shared_optimization.scheduling_method", None)
    use_jastrow = run.config.get("model.jastrow.use", False)
    metadata = dict(name=name,
                    group_name=group_name,
                    geom=geom,
                    is_shared=is_shared, 
                    pretraining_steps=pretraining_steps, 
                    scheduling_method=scheduling_method, 
                    use_jastrow=use_jastrow)

    opt_keys = all_opt_keys if is_shared else [k for k in all_opt_keys if k != "opt_n_epoch"]
    history = [dict(**metadata, **h) for h in run.scan_history(keys=opt_keys)]
    all_history.extend(history)
#%%
df = pd.DataFrame(all_history)
df["group_name"] = df["name"].apply(lambda s: "_".join(s.split("_")[:-1]))
df["geom"] = df["name"].apply(lambda s: int(s.split("_")[-1]))


ref_group = "ferminet_indep6geoms_gp"
df_ref = df[(df["group_name"] == ref_group) & (df["opt_epoch"] > 49_000)]
df_ref = df_ref.groupby("geom").agg(E_mean_ref=("opt_E_mean_smooth", "mean"), 
                                    E_var_ref=("opt_E_var", "mean")).reset_index()
df = df.merge(df_ref, on="geom")
df["error_E_mean"] = (df["opt_E_mean_smooth"] - df["E_mean_ref"]) * 1000
df["error_E_var"] = (df["opt_E_var"] - df["E_var_ref"])
df = df.sort_values(["is_shared", "scheduling_method", "group_name", "geom", "opt_epoch"])

fname = "/home/mscherbela/tmp/Hbcc_shared_6geoms.csv"
df.to_csv(fname, index=False)
df.to_parquet(fname.replace(".csv", ".parquet"))
