import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set_theme()

n_inter_steps_pretraining = 20
n_geoms_per_compound = 20

compound_names = ['CH4', 'NH3', 'OH2', 'C2H2', 'N2', 'CNH', 'C2H4', 'N2H2', 'CNH3', 'O2', 'NOH', 'COH2', 'C2H6', 'N2H4', 'CNH5', 'O2H2', 'NOH3', 'COH4']

# fname = "/home/mscherbela/tmp/tinymol_5x20_v3_parsed.csv"
# fname = "/home/mscherbela/tmp/tinymol_360_v2_parsed.csv"
fname = "/home/mscherbela/tmp/tinymol_18x20geoms_v10_parsed.csv"
# fname = "/home/mscherbela/tmp/tinymol_18x10geoms_v9_parsed.csv"

print("Loading CSV...")
df = pd.read_csv(fname)
df['geom_id'] = df.geom_id.astype(int)
df['compound_id'] = df.geom_id // n_geoms_per_compound
df['pre_epoch_per_geom'] = (df.mcmc_step_nr - n_inter_steps_pretraining) // n_inter_steps_pretraining

df_final_pretraining_loss = df.loc[df.pre_epoch_per_geom == df.pre_epoch_per_geom.max()]

print("Filling NaNs...")
metrics_to_extract = ['opt_E_mean_smooth', 'opt_E_var', 'opt_norm_constraint_factor', 'opt_grad_norm', 'opt_precon_grad_norm']
df_opt = df.iloc[-100_000:]
df_opt = df_opt[~df_opt.opt_n_epoch.isnull()].pivot(index='opt_n_epoch', columns='geom_id', values=metrics_to_extract)
df_opt = df_opt.fillna(method='ffill').fillna(method='bfill').reset_index()
df_opt = df_opt.melt(id_vars=["opt_n_epoch"])
df_opt.columns = ["opt_n_epoch", "metric", "geom_id", "value"]
df_opt_history = df_opt.groupby(["metric", "opt_n_epoch"])["value"].agg(["mean", "min", "max"]).reset_index()

# print("Creating subsampled per-geometry dataframe...")
# df_opt_compound = df_opt.query("opt_n_epoch % 100 == 0")
# df_opt_compound = df_opt_compound.groupby(["metric", "opt_n_epoch", "geom_id"])["value"].agg(["mean", "min", "max"]).reset_index()
# df_opt_compound['compound_id'] = df_opt_compound.geom_id // n_geoms_per_compound
# df_opt_compound['compound_name'] = df_opt_compound.compound_id.apply(lambda i: f"{i:02d}_{compound_names[i]}")

print("Computing final DF...")
n_opt_epochs_final = df_opt.opt_n_epoch.max()
df_opt_final = df_opt.query("opt_n_epoch == @n_opt_epochs_final")
df_opt_final['compound_id'] = df_opt_final.geom_id // n_geoms_per_compound
df_opt_final['compound_name'] = df_opt_final.compound_id.apply(lambda i: f"{i:02d}_{compound_names[i]}")
df_opt_final = df_opt_final.groupby(["geom_id", "compound_name", "metric", "opt_n_epoch"])["value"].agg(["mean", "min", "max"]).reset_index()
df_E_final = df_opt_final.query("metric == 'opt_E_mean_smooth'").groupby(["geom_id"])["mean"].mean().reset_index()
df_E_final.columns = ["geom_id", "opt_E_smooth_final"]
print("Preprocessing complete")

#%%
#
plt.close("all")
fig, axes = plt.subplots(2,4, figsize=(14,9))

# Pretraining loss curve
sns.lineplot(df, x="n_epoch", y="loss", ax=axes[0][0])
axes[0][0].set_yscale("log")

# Pretraining final loss
sns.barplot(df_final_pretraining_loss, y="compound_id", x="loss", ax=axes[0][1], orient="horizontal")
axes[0][1].set_xscale("log")
axes[0][1].set_xticks(10.0 ** np.arange(-4, -1))

sns.barplot(df_opt_final.query("metric == 'opt_E_var'"), y="compound_name", x="mean", ax=axes[0][2], orient="horizontal")
axes[0][2].set_xscale("log")
axes[0][2].set_xticks(10.0 ** np.arange(-1, 1))
axes[0][2].set_xlabel("Final opt_E_var")


# Variational opt
for i, metric in enumerate(metrics_to_extract):
    ax = axes.flatten()[i+3]
    df_filt = df_opt_history.query("metric == @metric")
    ax.plot(df_filt.opt_n_epoch, df_filt["mean"], color="C0")
    ax.fill_between(df_filt.opt_n_epoch, df_filt["min"], df_filt["max"], color="C0", alpha=0.2)
    if metric in ['opt_E_var', 'opt_grad_norm']:
        ax.set_yscale("log")
    if metric.startswith("opt_E_mean"):
        ax.set_ylim([df_filt["mean"].iloc[-1] - 10e-3, df_filt["mean"].iloc[-1] + 50e-3])
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
    ax.set_title(metric)
#
# df_E_var = df_opt.query("metric == 'opt_E_var'")
# axes[1][1].plot(df_E_var.opt_n_epoch, df_E_var["mean"], color="C0")
# axes[1][1].fill_between(df_E_var.opt_n_epoch, df_E_var["min"], df_E_var["max"], color="C0", alpha=0.2)
#

fig.suptitle(fname.split("/")[-1])
fig.tight_layout()


#%%
# plt.close("all")
#
# fig_per_compound, per_compound_axes = plt.subplots(2,3, figsize=(17,10), sharey="row")
# n_compounds_per_plot = 6
# for i in np.arange(0, df_opt_compound.compound_id.max()+1, n_compounds_per_plot):
#     ax_var = per_compound_axes[0][i//n_compounds_per_plot]
#     ax_error = per_compound_axes[1][i//n_compounds_per_plot]
#
#     df_filt = df_opt_compound.query("(compound_id >= @i) and (compound_id < (@i + @n_compounds_per_plot))")
#     sns.lineplot(df_filt.query("metric == 'opt_E_var'"), x="opt_n_epoch", y="mean", hue="compound_name", ax=ax_var)
#     ax_var.set_yscale("log")
#
#     df_filt_E = df_filt.query("metric == 'opt_E_mean_smooth'")
#     df_filt_E = pd.merge(df_filt_E, df_E_final, "left", "geom_id")
#     df_filt_E["error_vs_final"] = (df_filt_E["mean"] - df_filt_E["opt_E_smooth_final"]) * 1000
#     sns.lineplot(df_filt_E, x="opt_n_epoch", y="error_vs_final", hue="compound_name", ax=ax_error)
#     ax_error.set_ylim([-10, 50])
#
# fig_per_compound.tight_layout()

