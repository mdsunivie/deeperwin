# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.interpolate import CubicSpline


def plot_line(ax, dist, E, color, label, marker):
    cs = CubicSpline(dist, E)
    x_plot = np.linspace(np.min(dist), np.max(dist), 1000)
    ax.plot(dist, E, color=color, marker=marker, ls="none")
    ax.plot(x_plot, cs(x_plot), color=color, ls="-", label=label, marker=marker, ms=0.1)


sns.set_theme(style="whitegrid")

energy_source = "E_eval_50k"

df_full = pd.read_csv("data/fig5_N2_sweep.csv", sep=";")
df = df_full[["category"]]
df["E"] = df_full[energy_source]
df["dist"] = df.category.apply(lambda x: float(x.split("_")[-1]))
df["method"] = df_full.category.apply(lambda x: x.split("_")[0])
df = df.groupby(["dist", "method"])["E"].mean().reset_index()

df_experiment = pd.read_csv("data/N2_energies_experiment.csv")
spline_exp = CubicSpline(df_experiment["dist"], df_experiment["E"])

df_ccsdt = pd.read_csv("data/N2_errors_ccsdt.csv")
df_ccsdt["E"] = spline_exp(df_ccsdt["dist"]) + df_ccsdt["energy_error"] / 1000
df_ccsdt = df_ccsdt[["dist", "E", "method"]]

df = pd.concat([df, df_experiment, df_ccsdt])
df = df[df.dist < 5]
df["E_ref"] = spline_exp(df["dist"])
df["error"] = (df["E"] - df["E_ref"]) * 1000

plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(10, 5))


for method, label, color, marker in zip(
    ["experiment", "UCCSD(T)/CBS", "fermi", "dpe"],
    ["Experiment", "UCCSD(T)/CBS", "FermiNet", "Ours"],
    ["k", "dimgray", "C0", "C1"],
    ["none", "v", "o", "s"],
):
    df_plot = df[df["method"] == method]
    plot_line(
        axes[0],
        df_plot["dist"],
        df_plot["E"],
        color=color,
        label=None,
        marker=marker,
    )

    axes[1].plot(df_plot["dist"], df_plot["error"], color=color, label=label, marker=marker, ls="-")

fig.legend(loc="upper center", ncol=4)
axes[0].set_xlabel("Bond length / bohr")
axes[0].set_ylabel("Energy / Ha")
axes[0].set_title("Absolute energy")
axes[1].set_xlabel("Bond length / bohr")
axes[1].set_ylabel("Error vs. experiment / mHa")
axes[1].set_title("Relative energy vs. experiment")
fig.tight_layout()
fig.subplots_adjust(top=0.85)
fig.savefig("/home/mscherbela/ucloud/results/defense/N2.png", bbox_inches="tight", dpi=600)
