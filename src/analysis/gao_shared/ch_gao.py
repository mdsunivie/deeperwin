import pandas as pd
from deeperwin.run_tools.geometry_database import load_energies, load_geometries, load_datasets
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set_theme()

save_figs = False

def format_with_SI_postfix(x, decimals=2):
    postfixes = [(1e9, "b"), (1e6, "M"), (1e3, "k")]
    postfix = ""
    for v,p in postfixes:
        if x >= v:
            x/= v
            postfix = p
            break
    x = np.round(x, decimals)
    x = str(x)
    if "." in x:
        x = str(x).rstrip("0").rstrip(".")
    return x + postfix

def plot(ax, df, x, y, experiment, ls_dict, label_dict, color_dict, marker_dict, shade_range=False, **kwargs):
    experiments = sorted(df[experiment].unique())
    for e in experiments:
        pivot = df[df[experiment] == e].groupby(x)[y].agg(['mean', 'min', 'max']).reset_index()
        ax.plot(pivot[x], pivot['mean'],
                color=color_dict.get(e, 'k'),
                label=label_dict.get(e, ""),
                ls=ls_dict.get(e, '-'),
                marker=marker_dict.get(e, 'o'),
                **kwargs
                )
        if shade_range:
            ax.fill_between(pivot[x], pivot['min'], pivot['max'], alpha=0.4, color=color_dict[e])


dataset = load_datasets()['C2H4_1stretch_4twist_4geoms']
all_geometries = load_geometries()
experiments = [
    ("2023-02-23_gao_shared_20xC2H4_v2", "C0", "Our work: Shared"),
    ("2023-02-21_gao_20xC2H4_reuseshared_v2_from64k_lroffset32k", "C3", "Our work: Reuse from 50xCH$_3^+$+50xCH$_4$ (64k)"),
    ("2023-02-24_gao_reuseshared_20xC2H4_from20xCH4_v2_64k", "C2", "Our work: Reuse from 20xCH$_4$ (24k)"),
    ("Scherbela_etal_2022_shared", "C0", "DeepErwin 2022: Shared"),
    ("Scherbela_etal_2022_reuse", "C2", "DeepErwin 2022: Reuse from 20xCH$_4$"),
    ("Gao_etal_2023_HF_pretraining", "C0", "Gao et al.: Shared"),
    ("Gao_etal_2023_pretrained_on_smaller", "C2", "Gao et al.: Reuse from 20xCH$_4$"),
    ("MRCI_Ethene_20geoms", None, None)
]
palette = {e[0]: e[1] for e in experiments}
method_labels = {e[0]: e[2] for e in experiments}
marker_styles = dict(Gao_etal_2023_pretrained_on_smaller='^', Gao_etal_2023_HF_pretraining='^', Scherbela_etal_2022_shared='s', Scherbela_etal_2022_reuse='s')
line_styles = dict(Gao_etal_2023_pretrained_on_smaller='--', Gao_etal_2023_HF_pretraining='--', Scherbela_etal_2022_shared=':', Scherbela_etal_2022_reuse=':')
experiments = [e[0] for e in experiments]

df_full = load_energies()
df_full = df_full.query("molecule == 'C2H4'")
df_full = df_full[df_full.experiment.isin(experiments)].reset_index()
df_ref = df_full.query("source == 'MRCI'")
df_full = df_full.query("source != 'MRCI'")
df_full = pd.merge(df_full, df_ref[['geom', 'E']], 'left', 'geom', suffixes=("", "_ref"))
df_full['error'] = (df_full['E'] - df_full['E_ref']) * 1000
df_full['batch_size'] = df_full.batch_size.fillna(2048)
df_full['samples'] = df_full.epoch * df_full.batch_size

df_spread = df_full.groupby(["method", "experiment", "epoch", "samples", "source"])['error'].agg(lambda x: np.max(x) - np.min(x)).reset_index()

#%%
plt.close("all")
fig, axes = plt.subplots(2,2, figsize=(14,8))

for ind_ax in range(2):
    ax_error = axes[0][ind_ax]
    ax_spread = axes[1][ind_ax]

    plot(ax_error,
         df_full.query("epoch > 0"),
         "epoch" if ind_ax == 0 else "samples",
         "error",
         "experiment",
         line_styles,
         method_labels,
         palette,
         marker_styles
         )
    if ind_ax == 1:
        ax_error.legend(loc='upper left', handlelength=3)

    plot(ax_spread,
         df_spread.query("epoch > 0"),
         "epoch" if ind_ax == 0 else "samples",
         "error",
         "experiment",
         line_styles,
         method_labels,
         palette,
         marker_styles
         )

    ax_error.set_ylim([-7, 50])
    ax_error.set_ylabel("error vs MRCI / mHa")
    ax_error.axhline(1.6, color='k', ls=':')

    ax_spread.set_ylim([0, 100])
    ax_spread.set_ylabel("spread of predicton errors / mHa")

    for ax in [ax_error, ax_spread]:
        x_ticks = np.array([250, 500, 1000, 2000, 4000, 8000, 16000, 32000])
        if ind_ax == 1:
            ax.set_xlabel("MC samples")
            x_ticks *= 2000
        else:
            ax.set_xlabel("opt. steps")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(lambda x, pos: format_with_SI_postfix(x))
        ax.set_xticks(x_ticks)
        ax.set_xlim([min(x_ticks)*0.95, max(x_ticks)*1.15])


fig.suptitle("Transfer to C$_2$H$_4$")
fig.tight_layout()
if save_figs:
    fig.savefig(f"/home/mscherbela/ucloud/results/2023-02-21-reuse_CH3_CH4_to_Ethene.png", dpi=400, bbox_inches="tight")
    fig.savefig(f"/home/mscherbela/ucloud/results/2023-02-21-reuse_CH3_CH4_to_Ethene.pdf", bbox_inches="tight")


#%%
def plot_error_histogram(errors, ax, **kwargs):
    ax.hist(errors, bins=np.arange(-10, 55, 2), **kwargs)
    ax.axvline(np.mean(errors), color=kwargs.get('color'))
    ax.set_xlabel("deviation vs MRCI / mHa")
    ax.set_ylabel("Nr. of geometries")
    ax.legend()

hist_fig, hist_axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
df_64mio = df_full.query("(samples > 60_000_000 and samples < 70_000_000) or (experiment == '2023-02-23_gao_20xC2H4_reuseshared_from_20xCH4_24k' and samples > 15_000_000)")
df_dpe_shared = df_64mio.query("experiment == '2023-02-23_gao_shared_20xC2H4_v2'")
df_gao_shared = df_64mio.query("experiment == 'Gao_etal_2023_HF_pretraining'")
df_dpe_reuse = df_64mio.query("experiment == '2023-02-21_gao_20xC2H4_reuseshared_v2_from64k_lroffset32k'")
df_gao_reuse = df_64mio.query("experiment == 'Gao_etal_2023_pretrained_on_smaller'")
df_scherbela_reuse = df_full.query("samples > 600e6 and experiment == 'Scherbela_etal_2022_reuse'")
plot_error_histogram(df_dpe_shared.error, hist_axes[0][0], label="Our work, Shared", color='C0')
plot_error_histogram(df_gao_shared.error, hist_axes[1][0], label="Gao et al, Shared", color='C0', alpha=0.8)
plot_error_histogram(df_dpe_reuse.error, hist_axes[0][1], label="Our work, Reuse", color='C2')
plot_error_histogram(df_gao_reuse.error, hist_axes[1][1], label="Gao et al, Reuse", color='C2', alpha=0.8)
# hist_axes[0][0].set_title("Our work")
# hist_axes[0][1].set_title("Gao et al. 2023")
hist_fig.suptitle("Distribution of errors for C$_2$H$_4$ after 64 mio opt samples")
hist_fig.tight_layout()
if save_figs:
    fig.savefig(f"/home/mscherbela/ucloud/results/2023-02-21-reuse_CH3_CH4_to_Ethene_hist.png", dpi=400, bbox_inches="tight")
    fig.savefig(f"/home/mscherbela/ucloud/results/2023-02-21-reuse_CH3_CH4_to_Ethene_hist.pdf", bbox_inches="tight")


hashes_in_order = ['00572fed1563d8dffead78f6206d159f', '68a9222ecdedfa1c28c47bff5f561f55', 'b74f6a57c30a3e480b3da1f843210ed7', '84b0dea9a5b8580c5ad7556234c44aa4', 'd90bf56c430e7b00af9f7159cc088373', 'a08e544a9d4c501fdd0fd072cc74887e', '76bdef6a1423fb3014cb908ef6a0908f', '488f4bc14b1fdb0e935909eab7dfa107', '18097ae85beec490d7622cf897c84ee9', 'fa8c21dfffb579a9f5d81cbddbaf2aba', 'ea2433e4ae16ce0d47efe7de81c3d4fe', '4e675543969704f3d8db34129e98bc9c', 'c0b27949941497134d7610d509959862', '47e257a2330f298dfd055c77f550bd65', 'c72eeb5e47b3563cdebd8204fd3edf0d', '7fd93195486569c04ee65f4f184f0080', 'e6601d2a42568498980443310fa4bb7f', 'a03e416bda2d0f3c7ff8efb1e9a0e3be', '9c81488f923fe7d3fe64cabbaf119583', 'bdb183f76ba8b72586a333cccd2e3bf3']
twists = np.arange(0, 91, 10)

fig, ax = plt.subplots(1,1, figsize=(7,5))
for df, label, marker, color,ls in zip([df_ref, df_dpe_reuse, df_gao_reuse, df_scherbela_reuse, df_dpe_shared, df_gao_shared],
                     ["MRCI", "Our work, Reuse", "Gao et al 2023, Reuse", "Scherbela et al 2022, Reuse", "Our work, Shared", "Gao et al 2023, Shared"],
                     ['*', 'o', '^', 's', 'o', '^'],
                     ['k', 'C0', 'C1', 'C2', "C0", "C1"],
                     ['-','-','-','-', '--', '--']):
    energies = df.set_index('geom').loc[hashes_in_order[:10], 'E'].values
    ax.plot(twists, energies, label=label, marker=marker, color=color, ls=ls)
ax.legend()
ax.set_xticks(twists)
ax.set_xlabel("Twist angle / deg")
ax.set_ylabel("E / Ha")
ax.set_title("Potential energy surface for C$_2$H$_4$")
fig.tight_layout()
if save_figs:
    fig.savefig(f"/home/mscherbela/ucloud/results/2023-02-23-C2H4_PES_twist.png", dpi=400, bbox_inches="tight")
    fig.savefig(f"/home/mscherbela/ucloud/results/2023-02-23-C2H4_PES_twist.pdf", bbox_inches="tight")

