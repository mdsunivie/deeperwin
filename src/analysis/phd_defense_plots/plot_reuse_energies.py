# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.interpolate import CubicSpline

sns.set_theme(style="whitegrid")

df = pd.read_csv("data/paper1_reuse_plot_data.csv")
df = df.melt(
    id_vars=["subfigure", "molecule", "method", "steps"],
    var_name="geometry",
    value_name="error",
)
df["geometry"] = df["geometry"].str.extract(r"(\d+)").astype(int)
df = df.groupby(["subfigure", "molecule", "method", "steps"])["error"].mean().reset_index()

df = df[df.molecule == "Ethene"]

plt.close("all")
fig, ax = plt.subplots(1, 1, figsize=(7, 4))
ax.axhline(0, color="black", ls="-")

for (
    method,
    label,
    color,
    marker,
) in zip(
    ["indep", "reuse_same", "reuse_smaller"],
    ["Independent optimization", "\nReuse from similar\n\n", "\nReuse from smaller\n\n"],
    ["C0", "C4", "C2"],
    ["o", "s", "v"],
):
    df_plot = df[df.method == method]
    ax.plot(df_plot.steps, df_plot.error, color=color, label=label, marker=marker, ls="-", ms=7)
ax.set_xscale("log")
xticks = np.geomspace(2**6, 2**14, 5)
ax.set_xticks(xticks)


def format_tick(x):
    if x < 1000:
        return f"{x:.0f}"
    return f"{x//1000:.0f}k"


ax.set_xticklabels([format_tick(x) for x in xticks])
ax.set_xlabel("Optimization steps per geometry")
ax.set_ylabel("E - E$_{ref}$ / mHa")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("/home/mscherbela/ucloud/results/defense/paper1_reuse_plot.png", dpi=400, bbox_inches="tight")
