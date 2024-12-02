#%%
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")

# fnames = [
# "/home/mscherbela/tmp/parsed_logs/v10.csv",
# #"/home/mscherbela/tmp/parsed_logs/v16.csv",
# "/home/mscherbela/tmp/parsed_logs/PhisNetSTO6G.csv",
# "/home/mscherbela/tmp/parsed_logs/sizeAbl_16det.csv",
# "/home/mscherbela/tmp/parsed_logs/sizeAbl_16det_depth3.csv",
# "/home/mscherbela/tmp/parsed_logs/sizeAbl_16det_width512.csv",
# ]
fnames = [
    # "/home/mscherbela/runs/parsed_logs/BestGNN_18x20.csv",
    # "/home/mscherbela/runs/parsed_logs/1000geoms.csv",
    # "/home/mscherbela/runs/parsed_logs/ablation2_gnn.csv",
    # "/home/mscherbela/runs/parsed_logs/ablation3_phisnet.csv",
    # "/home/mscherbela/runs/parsed_logs/ablation4_distortion.csv",
    # "/home/mscherbela/runs/parsed_logs/ablation5_compounds.csv",
    # "/home/mscherbela/runs/parsed_logs/midi_699torsion.csv",
    "/home/mscherbela/runs/parsed_logs/midi699torsion_nc_by_std.csv",
    "/home/mscherbela/runs/parsed_logs/midi699torsion_exp_phisnet2.csv",
    "/home/mscherbela/runs/parsed_logs/midi699torsion_exp.csv",
]
run_names = [fname.split("/")[-1].replace("_parsed.csv", "") for fname in fnames]
color_dict = {r:f'C{i}' for i,r in enumerate(run_names)}
n_inter_steps_pretraining = 20
n_geoms_per_compound = 20
n_epochs_bar_chart = 35000

compound_names = ['CH4', 'NH3', 'OH2', 'C2H2', 'N2', 'CNH', 'C2H4', 'N2H2', 'CNH3', 'O2', 'NOH', 'COH2', 'C2H6', 'N2H4', 'CNH5', 'O2H2', 'NOH3', 'COH4']
# errorbar_func = None
errorbar_func = lambda x: (x.min(), x.max())


dfs = []
for fname, run_name in zip(fnames, run_names):
    print(f"Loading {fname}...")
    df = pd.read_csv(fname)
    df["run"] = run_name
    dfs.append(df)
df = pd.concat(dfs, axis=0, ignore_index=True)
df['geom_id'] = df.geom_id.astype(int)
df['compound_id'] = df.geom_id // n_geoms_per_compound
df = df.groupby(["run", "phase", "epoch", "compound_id"]).mean().reset_index()
#df['compound_name'] = df.compound_id.apply(lambda i: compound_names[i])

df_pre = df[df.phase == "pre"].reset_index()
df_opt = df.loc[df.phase == "opt", :].reset_index()
df_final_pre = df_pre.loc[df_pre.epoch == df_pre.epoch.max()].reset_index()
df_final_opt = df_opt.loc[df_opt.epoch == df_opt.epoch.max()].reset_index()
#
plt.close("all")
fig, axes = plt.subplots(2,4, figsize=(14,9))

# Pretraining loss curve
sns.lineplot(df_pre, x="epoch", y="loss", hue="run", hue_order=color_dict, ax=axes[0][0], errorbar=lambda x: (x.min(), x.max()))
axes[0][0].set_yscale("log")

# # Pretraining final loss
# sns.barplot(df_final_pre, y="compound_id", x="loss", hue="run", hue_order=color_dict, ax=axes[0][1], orient="horizontal")
# axes[0][1].set_xscale("log")
# axes[0][1].set_xticks(10.0 ** np.arange(-4, -1))

# sns.barplot(df[df.epoch == n_epochs_bar_chart], y="compound_name", x="opt_E_var", hue="run", hue_order=color_dict, ax=axes[0][2], orient="horizontal")
# axes[0][2].set_xscale("log")
# axes[0][2].set_xticks(10.0 ** np.arange(-1, 1))
# axes[0][2].set_xlabel(f"opt_E_var @{n_epochs_bar_chart//1000}k")


# Variational opt
metrics = ['opt_E_mean_smooth', 'opt_E_var', 'opt_norm_constraint_factor', 'opt_grad_norm', 'opt_precon_grad_norm']
opt_history_axes = axes.flatten()[3:]
for i, metric in enumerate(metrics):
    print(f"Plotting {metric}")
    ax = opt_history_axes[i]
    sns.lineplot(df_opt.query("epoch > 200"), 
                 x="epoch", 
                 y=metric, 
                 hue="run",
                 hue_order=color_dict, 
                 ax=ax, 
                 errorbar=errorbar_func,
                 legend=False,
                 )
    if metric in ['opt_E_var', 'opt_grad_norm']:
        ax.set_yscale("log")
    if metric.startswith("opt_E_mean"):
        ax.set_ylim([df_final_opt[metric].mean() - 10e-3, df_final_opt[metric].mean() + 500e-3])
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
    ax.set_xscale("log")
    ax.set_title(metric)
    for x in [64, 128, 174, 256]:
        ax.axvline(x*1000, color='dimgray', zorder=0)
# fig.suptitle(fname.split("/")[-1])
fig.tight_layout()

#%%
plt.close("all")
fig, ax = plt.subplots(1,1, figsize=(4,3))
df_plot = df_opt.query("epoch > 100")

def smooth(x, tau=1000):
    q = 1-1/tau
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = q*y[i-1] + (1-q)*x[i]
    return y

E_smooth = smooth(df_plot.opt_E_mean.values, tau=2000)
E_var_smooth = smooth(df_plot.opt_E_var.values, tau=2000)
# ax.plot(df_plot.epoch, df_plot.opt_E_mean_smooth, label="energy")

ax.set_ylabel("E  -E$_\\mathrm{final}$ / Ha", labelpad=0)
ax.set_xlabel("optimization step / k")
# ax2 = ax.twinx()
# axes = [ax, ax2]
ln1=ax.plot(df_plot.epoch/1000, E_smooth - E_smooth[-1], label="energy")
ln2=ax.plot(df_plot.epoch/1000, np.sqrt(E_var_smooth), label="variance", color="C1")
ax.set_yscale("symlog", linthresh=1)
ax.set_ylim([-0.2, 100])
ax.legend(ln1+ln2, ["mean energy", "energy std. dev."], loc="upper right", title="Running avg.")
fig.tight_layout()
fname = "/home/mscherbela/ucloud/results/pretraining_learning_curve.png"
fig.savefig(fname, bbox_inches="tight", dpi=600)

# ax.plot(df_plot.epoch, E_var_smooth, label="variance")
# %%
