# %%
import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

api = wandb.Api()
runs = api.runs("schroedinger_univie/debug_mcmc_projection")
runs = [r for r in runs if r.name.startswith("lattice_const_Be_")]

data = []
for r in runs:
    if "E_mean" not in r.summary.keys():
        continue
    data.append(
        dict(E_mean=r.summary["E_mean"], lattice_const=float(r.config["physical.periodic.lattice_const"]), method="DPE")
    )
df = pd.DataFrame(data)
df.sort_values(by="lattice_const", inplace=True)

energies_fn = """0.5  83.4416
1.0  -1.1579
1.5 -12.2532
2.0 -14.7826
2.5 -15.4279
3.0 -15.5354
3.5 -15.4816
4.0 -15.3782
4.5 -15.2719"""
energies_fn = [[float(x) for x in l.split()] for l in energies_fn.split("\n")]
energies_fn = pd.DataFrame(energies_fn, columns=["lattice_const", "E_mean"])
energies_fn["method"] = "FN"
df = pd.concat([df, energies_fn], ignore_index=True)

plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
sns.lineplot(data=df, x="lattice_const", y="E_mean", hue="method", ax=axes[0])

pivot = pd.pivot_table(df, index="lattice_const", columns="method", values="E_mean", aggfunc="mean").reset_index()
pivot["deviation"] = (pivot["DPE"] - pivot["FN"]) * 1000
axes[1].plot(pivot.lattice_const, pivot.deviation)
axes[1].set_ylabel("DPE - FN / mHa")
axes[1].axhline(0, color="black", linestyle="--")


# %%
