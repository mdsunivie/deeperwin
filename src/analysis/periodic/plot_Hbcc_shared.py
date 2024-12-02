# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def smooth(x, y, blocksize=50):
    n = (len(x) // blocksize) * blocksize
    x = np.array(x)[:n]
    y = np.array(y)[:n]
    x_mean = x.reshape(-1, blocksize).mean(axis=-1)
    y_mean = y.reshape(-1, blocksize).mean(axis=-1)
    return x_mean, y_mean


def plot_smooth(ax, x, y, blocksize=50, **kwargs):
    x, y = smooth(x, y, blocksize)
    ax.plot(x, y, **kwargs)


fname = "/home/mscherbela/tmp/Hbcc_shared_6geoms.csv"
# df = pd.read_csv(fname)
df = pd.read_parquet(fname.replace(".csv", ".parquet"))

group_names = df.group_name.unique()
n_geoms = 6
lattice_constants = [2.4, 3.0, 3.6, 4.2, 4.8, 5.4]

ref_group = "ferminet_indep6geoms_gp"
ref_lines = [df[(df.group_name == ref_group) & (df.geom == i)] for i in range(n_geoms)]

plt.close("all")
for metric, title, ylim, ylabel in zip(
    ["error_E_mean", "error_E_var"],
    ["Energy error", "Variance error"],
    [(-5, 80), (-0.01, 0.08)],
    ["E - E_FN_50k / mHa", "Var - Var_FN_50k"],
):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for group, ax in zip(group_names, axes.flatten()):
        df_group = df[df.group_name == group]  #
        for geom in range(n_geoms):
            plot_smooth(
                ax,
                ref_lines[geom].opt_epoch / 1000,
                ref_lines[geom][metric],
                blocksize=200,
                label="FermiNet" if geom == 0 else None,
                color="lightgray",
            )
        for geom in range(n_geoms):
            df_filt = df_group[df_group.geom == geom]
            plot_smooth(ax, 
                        df_filt.opt_epoch / 1000,
                        df_filt[metric], 
                        blocksize=200,
                        label=f"a={lattice_constants[geom]:.1f}")
        ax.set_title(group)
        ax.set_ylim(ylim)
        ax.set_xlabel("step per geom / k")
        ax.set_ylabel("E - E_FN_50k / mHa")
        ax.grid(alpha=0.5, color="grey")
        ax.legend()

    fig.suptitle(f"{title} vs. FermiNet")
    fig.tight_layout()
    fig_fname = "/home/mscherbela/ucloud/results/Hbcc_shared_6geoms_" + metric + ".png"
    fig.savefig(fig_fname, bbox_inches="tight", dpi=400)
    fig.savefig(fig_fname.replace(".png", ".pdf"), bbox_inches="tight")
