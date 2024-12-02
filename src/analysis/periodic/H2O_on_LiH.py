# %%
import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

api = wandb.Api()
all_runs = api.runs("schroedinger_univie/H2O_LiH")

def get_history(run):
    meta_data = dict(
        system="slab" if run.name.endswith("_1") else "total",
        n_atoms=16 if "16at" in run.name else 32,
    )
    history = [h for h in run.scan_history(keys=["opt_n_epoch", "opt_E_mean"], page_size=10_000)]
    if not history:
        return None
    history = pd.DataFrame(history)
    n = np.arange(min(history.opt_n_epoch), max(history.opt_n_epoch) + 1)
    E = np.interp(n, history.opt_n_epoch, history.opt_E_mean)
    df = pd.DataFrame(meta_data | dict(opt_n_epoch=n, opt_E_mean=E))
    return df

df_all = []
for run in all_runs:
    print(run.name)
    df_all.append(get_history(run))
df_all = pd.concat([d for d in df_all if d is not None], ignore_index=True)
#%%
df = df_all.groupby(["n_atoms", "system", "opt_n_epoch"]).first().reset_index()
for n_atoms in df.n_atoms.unique():
    for system in df.system.unique():
        mask = (df.n_atoms == n_atoms) & (df.system == system)
        df.loc[mask, "opt_E_mean_smooth"] = df[mask].opt_E_mean.rolling(window=5000).mean()

df = pd.pivot_table(df, values="opt_E_mean_smooth", index=["n_atoms", "opt_n_epoch"], columns="system").reset_index()
E_H2O = -76.4382  # Gerard et al.
E_ref_meV={"DMC, 36at": 167, "DMC, 64at": 209}
E_ref_mHa = {k: -v / 27.2114 for k, v in E_ref_meV.items()}
E_ref_colors = ["black", "dimgray", "lightgray"]
df["E_ads"] = (df["total"] - df["slab"] - E_H2O) * 1000

plt.close("all")
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for n_atoms in [16, 32]:
    mask = df.n_atoms == n_atoms
    ax.plot(df[mask].opt_n_epoch, df[mask].E_ads, label=f"DPE, {n_atoms} atoms")
ax.set_xlabel("Optimization steps")
ax.set_ylabel("Adsorption energy (mHa)")
ax.axhline(0, color="black", linestyle="-")
for i, (k, v) in enumerate(E_ref_mHa.items()):
    ax.axhline(v, label=k, color=E_ref_colors[i], ls="--")
ax.set_ylim([-20, 50])
ax.grid(alpha=0.5)
ax.legend()
ax.set_title("H2O on LiH\nShared opt, Gamma")
fig.savefig("/home/mscherbela/ucloud/results/H2O_LiH_preliminary.png", bbox_inches="tight")

