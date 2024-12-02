#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dirname = "/home/mscherbela/tmp/logfiles/"
fnames_ref = [
    # "GPU_200pretrain.csv",
    # "GPU_0pretrain_4dets.csv",
    # "GPU_240pretrain_4dets_SpinFlip.csv",
    # "GPU_PBC12-20_4dets_v2.csv",
    # "GPU_PBC12-20_8dets.csv",
    # "GPU_PBC12-20_DZ.csv",
    # "GPU_PBC12-20_DZ_Mixed_v2.csv",
    # "GPU_PBC12-20_DZ_Mixed_v3.csv",
]

fnames = [
    "GPU_PBC40Reuse.csv"
    # "GPU_PBC12-20_DZ_Mixed_v4.csv",
    # "GPU_PBC12-20_DZ_Mixed_v4_100To200.csv",
]
fnames = fnames_ref + fnames

run_dfs = [pd.read_csv(dirname + f) for f in fnames]

geom_ids = [0, 40, 80]
# ref_energies = {
#     0: [-5.893, -5.952],
#     40: [-7.8, -7.928, -8.0088],
#     80: [-9.939, -10.031],
# }

geom_ids = [g + 11 for g in [0, 1, 2, 3, 4]]
# geom_ids = [40, 45, 50, 55]
ref_energies = {}

# plt.close("all")
fig, axes = plt.subplots(2, len(geom_ids), figsize=(14, 6), sharex=True)
E_min = np.zeros(len(geom_ids))
for i, (fname, df) in enumerate(zip(fnames, run_dfs)):
    if i < len(fnames_ref):
        run_label = None
        color = "gray"
    else:
        run_label = fname.replace("GPU_", "").replace(".csv", "")
        color=None
    for ind_geom, (g, (ax_E, ax_var)) in enumerate(zip(geom_ids, axes.T)):
        df_g = df[df.geom_id == g]
        ax_E.plot(df_g.epoch / 1000, df_g.opt_E_mean, label=run_label, color=color)
        ax_E.set_title(f"Geom {g}")
        ax_E.set_ylabel("E / Ha")
        ax_E.legend(loc='upper right')
        E_min[ind_geom] = min(E_min[ind_geom], np.quantile(df_g.opt_E_mean, 0.2))

        ax_var.semilogy(df_g.epoch / 1000, df_g.opt_E_var, label=run_label, color=color)
        ax_var.set_xlabel("Total epochs / k")
        ax_var.set_ylabel("Var / Ha")

for ax, E in zip(axes[0], E_min):
    ax.set_ylim([E - 0.005, E + 0.2])


    
# for ind_geom, E_refs in ref_energies.items():
#     ax = axes[0, geom_ids.index(ind_geom)]
#     for E in E_refs:
#         ax.axhline(E, color='k', alpha=1.0 if E == min(E_refs) else 0.2)
#     ax.set_ylim([min(E_refs) - 0.05, min(E_refs) + 0.3])
fig.tight_layout()
    