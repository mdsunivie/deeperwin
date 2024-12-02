# %%
import numpy as np
import matplotlib.pyplot as plt
from deeperwin.run_tools.geometry_database import load_geometries, load_energies
from deeperwin.utils.utils import get_distance_matrix
from scipy import stats
import seaborn as sns

sns.set_theme(style="whitegrid")

label_rmax = "max $R_{IJ}$ / bohr"


def _lin_reg(df_subset, x_pred, log=False):
    x, y = (
        np.array(df_subset[label_rmax]).reshape(-1, 1),
        np.array(df_subset["E - $E_\\mathrm{PBE0+MBD}$ / mHa"]).reshape(-1, 1),
    )

    if log:
        y = np.log(y)
    res = stats.linregress(x.squeeze(), y.squeeze())

    x_pred = np.array(x_pred).reshape(-1)
    if log:
        return np.exp(res.intercept + res.slope * x_pred)
    return res.intercept + res.slope * x_pred


all_geometries = load_geometries()
energies = load_energies()

df_dft = energies[energies.experiment == "ePBE0+MBD_QM7X"]
df_dft = df_dft.rename(columns={"E": "$E_\\mathrm{PBE0+MBD}$ / Ha"}, inplace=False)
df_fm = energies[energies.experiment == "2023-05-11_reuse_midimol_699torsions_256k_largescale_qm7"]
df_fm = df_fm.rename(columns={"E": "E / Ha"}, inplace=False)

df = df_dft.merge(df_fm, left_on="geom", right_on="geom")
df["Nb. heavy atoms"] = df.geom.apply(lambda x: len(all_geometries[x].Z[all_geometries[x].Z != 1]))
df["|E - $E_\\mathrm{PBE0+MBD}$| / mHa"] = np.abs(df["E / Ha"] - df["$E_\\mathrm{PBE0+MBD}$ / Ha"]) * 1000
df["E - $E_\\mathrm{PBE0+MBD}$ / mHa"] = (df["E / Ha"] - df["$E_\\mathrm{PBE0+MBD}$ / Ha"]) * 1000

df = df.sort_values(by=["E - $E_\\mathrm{PBE0+MBD}$ / mHa"])


def _calc_rmax(hash):
    R = all_geometries[hash].R
    diff, dist = get_distance_matrix(R)
    R_max = np.max(np.array(dist))
    return R_max


df[label_rmax] = df.geom.apply(_calc_rmax)

### Plotting ###
max_heavy_atoms = 2
metadata = [("", "2", None), ("", "3", None), ("", "4", None), ("", "5", None), ("", "6", None), ("", "7", None)]

fig, ax = plt.subplots(1, 3, figsize=(10, 3))

# Scatter: E vs E_Dft
ax[0].plot([-375, -65], [-375, -65], color="black")

marker_option = ["P", "^", "s", "p", "h", "o"]
sns.scatterplot(
    data=df,
    x="$E_\\mathrm{PBE0+MBD}$ / Ha",
    y="E / Ha",
    style="Nb. heavy atoms",
    hue="Nb. heavy atoms",
    markers=marker_option,
    ax=ax[0],
    s=100,
    zorder=10,
    legend=True,
)


#
# df['color'] = df['Nb. heavy atoms'].apply(lambda x: markers[x])
# ax[0].scatter(list(df['$E_\\mathrm{PBE0+MBD}$ / Ha']), list(df['E / Ha']), marker=list(df['marker']))

# Histogram: Energy diff


sns.histplot(
    data=df,
    x="E - $E_\\mathrm{PBE0+MBD}$ / mHa",
    hue="Nb. heavy atoms",
    ax=ax[1],
    kde=False,
    multiple="stack",
    legend=False,
)
ax[1].set_xlim([-130, 1250])

handles, labels = ax[0].get_legend_handles_labels()
ax[0].legend_.remove()
ax[1].legend(handles, labels, loc="upper right", ncols=2, title="Nr. of heavy atoms", markerscale=1.5)


# Scatter: E diff vs Rmax
sns.scatterplot(
    data=df[df["Nb. heavy atoms"] >= max_heavy_atoms],
    x=label_rmax,
    y="E - $E_\\mathrm{PBE0+MBD}$ / mHa",
    hue="Nb. heavy atoms",
    ax=ax[2],
    style="Nb. heavy atoms",
    markers=marker_option,
    s=100,
    zorder=10,
    legend=False,
)

ax[2].set_yscale("symlog", linthresh=200)
ax[2].set_ylim([-130, 4000])
ax[2].set_yticks([-100, 0, 100, 200, 1000])
ax[2].set_xticks(range(0, 20, 4))

ax[2].minorticks_on()
ax[2].grid(which="minor", axis="y", linestyle=":", linewidth=0.5, color="lightgrey")

ax[2].vlines(11, ymin=-130, ymax=4000, colors="black")
ax[2].text(0.125, 0.9, s="max $R^{train}_{IJ}$", fontsize=11, transform=ax[2].transAxes)  # , ha="left", va="top")

x_pred = [6.0, 14.0]
y_pred = _lin_reg(df[df["Nb. heavy atoms"] == 5], x_pred)
ax[2].plot(x_pred, y_pred, color="red", alpha=0.5)

x_pred = [8.5, 15.0]
y_pred = _lin_reg(df[df["Nb. heavy atoms"] == 6], x_pred)
ax[2].plot(x_pred, y_pred, color="purple", alpha=0.5)

x_pred = [9.0, 17.5]
df_subset = df[df["Nb. heavy atoms"] == 7]
y_pred = _lin_reg(df_subset, x_pred, log=True)
ax[2].plot(x_pred, y_pred, color="brown", alpha=0.5)


for a, letter in zip(ax, "abc"):
    a.text(x=0, y=1.01, s=letter, transform=a.transAxes, fontsize=14, fontweight="bold", ha="left", va="bottom")


fig.tight_layout()
fname = "/Users/leongerard/ucloud/Shared/results/04_paper_better_universal_wf/figures/fig6_largescale_vs_dft.png"
fig.savefig(fname, dpi=300, bbox_inches="tight")
fig.savefig(fname.replace(".png", ".pdf"), bbox_inches="tight")
# %%
