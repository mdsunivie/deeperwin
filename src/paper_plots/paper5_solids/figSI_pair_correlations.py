#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Extract data
fnames = ["plot_data/H20_correlations_1.2a0.npz", "plot_data/H20_correlations_3.6a0.npz"]
R_values = [1.2, 3.6]
n_atoms = 20
n_pairs_diff = (n_atoms // 2) * (n_atoms // 2) * 2
n_pairs_same = (n_atoms // 2) * (n_atoms // 2 - 1) * 2 

ind_geoms = np.array([0, 1, 5, 7])
rho_same = []
rho_diff = []
for fname in fnames:
    raw_data = np.load(fname)
    xmin, xmax = raw_data["r_min"][0], raw_data["r_max"][0]
    n_grid = raw_data["n_grid"][0]
    rho_same.append(raw_data["rho"][0, 0, :, 0, 0] + raw_data["rho"][1, 1, :, 0, 0])
    rho_diff.append(raw_data["rho"][1, 0, :, 0, 0] + raw_data["rho"][0, 1, :, 0, 0])

x_rel = np.linspace(xmin, xmax, n_grid) / R_values[-1]
x_rel = (x_rel + n_atoms//2) % n_atoms - n_atoms//2
n_shift = np.argmin(x_rel)
x_rel = np.roll(x_rel, -n_shift)

all_data = []
for i in range(len(rho_same)):
    rho_same[i] = np.roll(rho_same[i], -n_shift)
    rho_diff[i] = np.roll(rho_diff[i], -n_shift)

    L_tot = (xmax - xmin) / R_values[-1]
    rho_same[i] = n_pairs_same * rho_same[i] / np.sum(rho_same[i]) * L_tot
    rho_diff[i] = n_pairs_diff * rho_diff[i] / np.sum(rho_diff[i]) * L_tot
    
    all_data.append(pd.DataFrame(np.array([x_rel, rho_same[i], rho_diff[i]]).T, columns=["x_rel", "rho_same", "rho_diff"]))
    all_data[-1]["R"] = R_values[i]

df = pd.concat(all_data)
df.to_csv("plot_data/pair_correlation_H20.csv", index=False)


#%%
plt.close("all")
df = pd.read_csv("plot_data/pair_correlation_H20.csv")
R_values = df["R"].unique()

fig, axes = plt.subplots(1,2, figsize=(9, 4))
for idx_R, R in enumerate(R_values):
    df_R = df[df["R"] == R]
    axes[0].plot(df_R.x_rel, df_R.rho_same, label=f"R = {R:.1f}$a_0$")
    axes[1].plot(df_R.x_rel, df_R.rho_diff, label=f"R = {R:.1f}$a_0$")

for ax, ax_label in zip(axes, "ab"):
    for i in range(-10, 10+1):
        if i % 2 == 0:
            ax.axvline(i, color="gray", lw=0.5, ls="-", zorder=-1)
        else:
            ax.axvline(i, color="gray", lw=0.5, ls="--", zorder=-1)
    ax.set_ylim([0, 30])
    ax.set_xlabel("(x - x') / R")
    ax.set_ylabel("p(x, x')")
    ax.set_xticks(np.arange(-10, 11, 2))
    ax.legend(loc="lower left")
    ax.text(0, 1.0, ax_label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom", ha="left")
axes[0].set_title("Same spin")
axes[1].set_title("Opposite spin")
fig.tight_layout()
fig.savefig("plot_output/pair_correlation_H20.pdf", bbox_inches="tight")
fig.savefig("plot_output/pair_correlation_H20.png", bbox_inches="tight", dpi=300)

