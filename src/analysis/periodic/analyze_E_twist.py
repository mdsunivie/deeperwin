# %%
import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
sns.set_style("whitegrid")

api = wandb.Api()
runs = api.runs("schroedinger_univie/debug_mcmc_projection")
runs = [r for r in runs if r.name.startswith("twist_sweep_v3_")]

data = []
for i,r in enumerate(runs):
    print(f"Loading run {i+1}/{len(runs)}")
    if "E_mean" not in r.summary.keys():
        continue
    k_twist = np.array(r.config["physical.periodic.k_twist"])
    data.append(
        dict(E_mean=r.summary["E_mean"],
             E_mean_sigma=r.summary["E_mean_sigma"],
             use_bloch_env = r.config["model.orbitals.use_bloch_envelopes"],
             k_twist_x=k_twist[0],
             k_twist_y=k_twist[1],
             k_twist_z=k_twist[2],
             method="DPE")
    )
df = pd.DataFrame(data)
df = df.sort_values(by=["k_twist_x", "k_twist_y", "k_twist_z"])

#%%
plt.close("all")
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
sns.lineplot(data=df, 
             x="k_twist_x", 
             y="E_mean", 
             hue="use_bloch_env",
             ax=ax)
ax.grid(alpha=0.5)

# plt.errorbar(df.k_twist_x, 
#              df.E_mean, 
#              yerr=df.E_mean_sigma,
#              capsize=4,
#              )
# plt.grid(alpha=0.5)
# plt.xlabel("k_twist_x")
# plt.ylabel("E_mean")

# %%
