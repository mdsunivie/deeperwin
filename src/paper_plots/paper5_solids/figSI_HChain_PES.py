# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from deeperwin.run_tools.geometry_database import load_datasets
from scipy.optimize import curve_fit

# %%
# Extract data from log file; skip and directly load generated CSV instead
# eval_data = []
# with open("/home/mscherbela/tmp/GPU.out") as f:
#     for line in f:
#         if "; E_mean=" in line:
#             data = dict(
#                 E=float(line.split("; E_mean=")[1].split(";")[0]),
#                 E_sigma=float(line.split("; E_mean_sigma=")[1].split(";")[0]),
#                 steps=int(line.split("opt_n_epoch=")[1].split(";")[0]),
#                 geom_id=int(line.split("geom_id=")[1].split(";")[0]),
#             )
#             eval_data.append(data)
# df = pd.DataFrame(eval_data)
# dataset = load_datasets()["HChainPBC4-12_1.70-2.00_60geoms_4kgrid"]
# geoms = dataset.get_geometries()

# df["n_atom"] = df.geom_id.apply(lambda x: geoms[x].n_atoms * geoms[x].periodic.supercell[0])
# df["R"] = df.geom_id.apply(lambda x: geoms[x].R[1][0])
# df["k"] = df.geom_id.apply(lambda x: geoms[x].periodic.k_twist[0])
# df["weight"] = df.geom_id.apply(lambda x: dataset.weights[x] / 4)
# df.to_csv("plot_data/HChain_PES.csv", index=False)


df = pd.read_csv("plot_data/HChain_PES.csv")
df["E_weighted"] = df.E * df.weight
df["E_sigma_weighted"] = df.E_sigma * df.weight
df_tabc = df.groupby(["steps", "n_atom", "R"])[["E_weighted", "E_sigma_weighted"]].sum().reset_index()
df_tabc.rename(columns={"E_weighted": "E", "E_sigma_weighted": "E_sigma"}, inplace=True)
df_tabc["E"] /= df_tabc.n_atom
df_tabc["E_sigma"] /= df_tabc.n_atom


def morse(R, E0, R0, a):
    E_inf = -0.5
    return E0 + (E_inf - E0) * (1 - np.exp(-a * (R - R0))) ** 2


plt.close("all")
df_tabc = df_tabc[df_tabc.steps == 50_000]
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

n_at_values = df_tabc.n_atom.unique()[::-1]
R_min_values = []
sigma_R_values = []

ax_E, ax_R = axes
for idx_atom, n_atom in enumerate(n_at_values):
    color = f"C{idx_atom}"
    df_n = df_tabc[df_tabc.n_atom == n_atom]
    ax_E.errorbar(
        df_n.R,
        df_n.E,
        yerr=df_n.E_sigma,
        capsize=5,
        fmt="s-",
        label=f"$N_\\text{{atoms}}={n_atom}$",
        color=color,
        ls="none",
    )
    fit_coeff, fit_cov = curve_fit(morse, df_n.R, df_n.E, p0=[-0.55, 1.8, 1], sigma=df_n.E_sigma)
    dR = np.sqrt(fit_cov[1, 1])
    R_fit = np.linspace(df_n.R.min(), df_n.R.max(), 100)
    E_fit = morse(R_fit, *fit_coeff)
    E_min, R_min = fit_coeff[:2]
    R_min_values.append(R_min)
    sigma_R_values.append(dR)
    ax_E.plot(R_fit, E_fit, color=color)
    ax_E.errorbar([R_min], [E_min], xerr=dR, capsize=5, marker="x", ms=6, color=color)

ax_R.errorbar(n_at_values, R_min_values, yerr=sigma_R_values, marker="o", capsize=5, ls="none", color="k")
ax_E.legend(loc="upper center", ncol=2)
ax_E.set_xlabel("Atom distance $R$ / bohr")
ax_E.set_ylabel("Energy per atom / Ha")
ax_R.set_xlabel("$N_\\text{atoms}$")
ax_R.set_ylabel("Equilibrium distance $R_0$ / bohr")
ax_E.set_ylim([None, -0.563])
for ax, label in zip(axes, "ab"):
    ax.grid(alpha=0.5, color="grey", ls="--")
    ax.text(0, 1.0, f"{label}", transform=ax.transAxes, va="bottom", ha="left", fontweight="bold", fontsize=12)
fig.tight_layout()
save_dir = "plot_output"
fig.savefig(f"{save_dir}/HChain_PES.pdf", bbox_inches="tight")
fig.savefig(f"{save_dir}/HChain_PES.png", dpi=300, bbox_inches="tight")
