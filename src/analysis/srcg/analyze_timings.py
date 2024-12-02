#%%
import wandb
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

api = wandb.Api()
runs = api.runs("schroedinger_univie/srcg")
runs = [r for r in runs if re.match(".*_timings.*", r.name)]
# %%
data = []
for i,r in enumerate(runs):
    print(i)
    history = [d for d in r.scan_history(keys=["opt_epoch", "opt_t_epoch"])]
    times = [h["opt_t_epoch"] for h in history if h["opt_epoch"] >=4 ]
    data.append(dict(optimizer=r.config["optimization.optimizer.name"],
                     maxiter=r.config.get("optimization.optimizer.maxiter"),
                     n_el=r.config["physical.n_electrons"],
                     t=np.mean(times)
                     ))

df = pd.DataFrame(data)
df_kfac = df[df["optimizer"] == "kfac"][["n_el", "t"]].rename(columns={"t": "t_kfac"})
df = pd.merge(df, df_kfac, on="n_el")

# %%
df["t_rel"] = df["t"] / df["t_kfac"]
plt.close("all")
fig, ax = plt.subplots(1,1, figsize=(9,6))
sns.lineplot(data=df[df.optimizer=='srcg'], 
             x="n_el", 
             y="t_rel", 
             hue="maxiter",
             palette="tab10",
             marker='o',
             ax=ax)
ax.axhline(1, color="black", linestyle="--")
ax.set_ylabel("Timer per step (t_SRCG / t_KFAC)")
ax.set_xlabel("Number of electrons")
ax.set_title("SR+CG has 30-80% higher per-step runtime than KFAC for medium-sized molecules\n2xA100 on HGX, $C_xH_y$, 50 inter-steps, ~1 mio params")
fig.tight_layout()
fname = "/home/mscherbela/ucloud/results/SRCG_per_iteration_cost.png"
fig.savefig(fname, dpi=300, bbox_inches="tight")
fig.savefig(fname.replace(".png", ".pdf"), bbox_inches="tight")

# %%
