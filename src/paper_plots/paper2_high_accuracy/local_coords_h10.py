import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

df = pd.read_csv("/home/mscherbela/tmp/bent_h10.csv", sep=";")
df["geom"] = df["name"].apply(lambda x: int(re.search(R"_(\d{4})", x)[1]))
df["is_local"] = df["name"].apply(lambda x: re.search(R"_((True|False))_(\d{4})", x)[2])
df.is_local = df.is_local == "True"

pivot = (
    df.groupby(["is_local", "geom"])
    .agg(E_50k=("E_eval_50k", "mean"), E_100k=("E_eval_100k", "mean"), E_20k=("E_smooth_20k", "mean"))
    .reset_index()
)
df_ref = df[~df.is_local].groupby(["geom"]).agg(E_ref=("E_eval_100k", "mean")).reset_index()
pivot = pd.merge(pivot, df_ref, "left", "geom")

E0 = pivot.E_100k.min()
plt.close("all")
fig, axes = plt.subplots(1, 2, dpi=100, figsize=(12, 8))
colors = {20: "C0", 50: "C1", 100: "C2"}
for n_epochs in [20, 50, 100]:
    for is_local in [True, False]:
        df_filt = pivot[pivot.is_local == is_local]
        ls = "-" if is_local else "--"

        axes[0].plot(
            df_filt.geom,
            df_filt[f"E_{n_epochs}k"],
            ls=ls,
            color=colors[n_epochs],
            label=f"{n_epochs}k: {'local feat.' if is_local else 'global feat.'}",
        )
        axes[1].plot(
            df_filt.geom,
            (df_filt[f"E_{n_epochs}k"] - df_filt.E_ref) * 1e3,
            ls=ls,
            color=colors[n_epochs],
            label=f"{n_epochs}k: {'local feat.' if is_local else 'global feat.'}",
        )
for ax in axes:
    ax.grid(alpha=0.5)
    ax.legend()
    ax.set_xlabel("Geometry nr")
    ax.set_xticks(np.arange(df.geom.max() + 1))

axes[0].set_ylabel("Energy / Ha")
axes[1].set_ylabel("Energy Differences / mHa")
