#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from deeperwin.run_tools.geometry_database import load_geometries, load_datasets, Geometry
from deeperwin.utils.plotting import get_discrete_colors_from_cmap
all_geoms = load_geometries()

# fname = "/home/mscherbela/tmp/logfiles/Reuse40.csv"
# ds = load_datasets()["HChainPBC40_1.20-3.60_88geoms_20kgrid"]

# fname = "/home/mscherbela/tmp/logfiles/GPU_240pretrain_4dets_SpinFlip.csv"
# fname = "/home/mscherbela/tmp/logfiles/GPU_PBC12-20_4dets_v2.csv"
# fname = "/home/mscherbela/tmp/logfiles/GPU_PBC12-20_8dets.csv"
# fname = "/home/mscherbela/tmp/logfiles/GPU_PBC12-20_DZ_Mixed_v4.csv"
fname = "/home/mscherbela/tmp/logfiles/GPU_PBC40Reuse.csv"




# ds = load_datasets()["HChainPBC12-20_1.20-3.60_120geoms_8kgrid"]
ds = load_datasets()["HChainPBC40_1.20-3.60_88geoms_20kgrid"]

df = pd.read_csv(fname)
geom_data = []
for i, g in enumerate(ds.get_geometries(all_geoms)):
    geom_data.append(dict(geom_id=i, 
                          k=g.periodic.k_twist[0],
                          n_atoms=g.n_atoms * g.periodic.supercell[0],
                          R=g.R[1][0] - g.R[0][0],
                          ))
df = pd.merge(df, pd.DataFrame(geom_data), on="geom_id")
df["E_prim"] = df.opt_E_mean_smooth / df.n_atoms
group_keys = ["geom_id", "k", "n_atoms", "R"]

df_last = df.groupby(group_keys).last().reset_index()
df_intermed = df[df.epoch==2_000].groupby(group_keys).last().reset_index()
df_last_energy = df_last[group_keys + ["opt_E_mean_smooth"]].rename(columns={"opt_E_mean_smooth": "opt_E_mean_smooth_final"})
df_intermed = pd.merge(df_intermed, df_last_energy, on=group_keys)
df_intermed["energy_error"] = (df_intermed.opt_E_mean_smooth - df_intermed.opt_E_mean_smooth_final) * 1000


plt.close("all")
fig, axes = plt.subplots(2,2, figsize=(12, 7))
plot_kwargs = dict(palette="plasma", hue="k", x="R")
sns.scatterplot(df_last, y="opt_E_var", ax=axes[0,0], **plot_kwargs)
sns.scatterplot(df_last, y="opt_norm_constraint_factor", ax=axes[0,1], **plot_kwargs)
# sns.scatterplot(df_last, y="epoch_per_geom", ax=axes[1, 0], **plot_kwargs)
sns.scatterplot(df_intermed, y="energy_error", ax=axes[1, 1], **plot_kwargs)
axes[0, 1].axhline(1, color='k', alpha=0.2)
# axes[1, 0].axhline(df_last.epoch_per_geom.mean(), color='k', alpha=0.2)
axes[1, 1].axhline(0, color='k', alpha=0.2)
fig.suptitle(f"Epochs total: {df_last.epoch.mean():.0f}")

n_atoms = df_last.n_atoms.max()
cmap = get_discrete_colors_from_cmap(8, "plasma", 0.0, 1.0)
fig, axes = plt.subplots(1,2)
axes = np.atleast_2d(axes)
df_last_gamma = df_last[df_last.k == 0].rename(columns={"opt_E_mean_smooth": "opt_E_mean_smooth_gamma"})
df_last_gamma = df_last_gamma[["n_atoms", "R"] + ["opt_E_mean_smooth_gamma"]]
df_plot = pd.merge(df_last, df_last_gamma, on=["n_atoms", "R"])
df_plot["energy_minus_gamma"] = df_plot.opt_E_mean_smooth - df_plot.opt_E_mean_smooth_gamma
df_plot = df_plot[df_plot.n_atoms == n_atoms]
sns.lineplot(data=df_plot, x="k", y="energy_minus_gamma", hue="R", ax=axes[0, 0], palette=cmap)
sns.lineplot(data=df_plot, x="R", y="opt_E_mean_smooth", hue="k", ax=axes[0, 1], palette="plasma")
fig.suptitle(f"N = {n_atoms} atoms")
# sns.lineplot(data=df_plot, x="R", y="E_prim", hue="k", ax=axes[0, 1], palette="plasma")

df_print = df_last.query("k == 0 and n_atoms == 20 and R == 1.2")
print(df_print[["k", "n_atoms", "R", "opt_E_mean_smooth"]])

#%%
group_keys = ["geom_id", "k", "n_atoms", "R"]
df_delta = df_last.groupby(group_keys).mean() - df_last_200k.groupby(group_keys).mean()
df_delta = df_delta.sort_values("opt_E_mean_smooth", ascending=False)






