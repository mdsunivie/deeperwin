#%%
from deeperwin.run_tools.geometry_database import load_energies
from plot_utils import plot_df, format_with_SI_postfix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from PIL import Image
sns.set_theme(style="whitegrid")

save_figs = True
experiments = [
    ("MRCI_Ethene_20geoms", "k", "*", "MRCI"),
    ("MRCI_HChain10_23geoms", None, None, None),
    ("Gao_et_al_2022_Ferminet_Cyclobutadiene", None, None, None),
    ("Scherbela_etal_2022_reuse", "C1",  "s", "DeepErwin (2022)"),
    ("Gao_etal_2023_pretrained_on_smaller", "C2", "^", "GLOBE (2023)"),
    ("2023-02-24_gao_reuseshared_20xC2H4_from20xCH4_v2_64k", "C0", "o", "Our work"),
    ("2023-03-01_gao_23xHChain10_reuseshared_from_23xHChain6_v2_64k", "C0", "o", "Our work"),
    ("2023-03-08_gao_reuseshared_2xCyclobutadiene_from20xC2H4_64k", "C0", "o", "Our work"),
]
label_dict = {e[0]:e[3] for e in experiments}
marker_dict = {e[0]:e[2] for e in experiments}
color_dict = {e[0]:e[1] for e in experiments}
lw_dict = {e[0]: 2.5 for e in experiments if "reuseshared" in e[0]}
plot_kwargs = dict(label_dict=label_dict, marker_dict=marker_dict, color_dict=color_dict, lw_dict=lw_dict)

experiments = [e[0] for e in experiments]


df_full = load_energies()
df_full = df_full[df_full.experiment.isin(experiments)].reset_index()
is_ref = df_full.source.isin(["MRCI", "Gao_et_al_2022_Ferminet"])
df_ref = df_full[is_ref]
df_full = df_full[~is_ref]
df_full = pd.merge(df_full, df_ref[['geom', 'E']], 'left', 'geom', suffixes=("", "_ref"))
df_full['error'] = (df_full['E'] - df_full['E_ref']) * 1000
df_full['batch_size'] = df_full.batch_size.fillna(2048)
df_full['samples'] = df_full.epoch * df_full.batch_size

df_spread = df_full.groupby(["method", "molecule", "experiment", "epoch", "samples"])['error'].agg(lambda x: np.max(x) - np.min(x)).reset_index()
df_relative = df_full.groupby(["method", "molecule", "experiment", "epoch", "samples"])[['E', 'E_ref']].agg(lambda x: np.max(x) - np.min(x)).reset_index()
df_relative["error"] = (df_relative["E"] - df_relative["E_ref"]) * 1000

hashes_in_order = ['00572fed1563d8dffead78f6206d159f', '68a9222ecdedfa1c28c47bff5f561f55', 'b74f6a57c30a3e480b3da1f843210ed7', '84b0dea9a5b8580c5ad7556234c44aa4', 'd90bf56c430e7b00af9f7159cc088373', 'a08e544a9d4c501fdd0fd072cc74887e', '76bdef6a1423fb3014cb908ef6a0908f', '488f4bc14b1fdb0e935909eab7dfa107', '18097ae85beec490d7622cf897c84ee9', 'fa8c21dfffb579a9f5d81cbddbaf2aba']
twists = np.arange(0, 91, 10)
df_reuse_final = df_full[(df_full.molecule == "C2H4") & (df_full.method.str.contains("reuse"))]
max_samples_per_method = df_reuse_final.groupby("experiment").agg(max_samples=("samples", "max")).reset_index()
df_reuse_final = df_reuse_final.merge(max_samples_per_method, "left", on="experiment")
df_reuse_final = df_reuse_final.query("samples == max_samples")
df_reuse_final = pd.concat([df_reuse_final, df_ref], ignore_index=True)
df_reuse_final = df_reuse_final[df_reuse_final.geom.isin(hashes_in_order)]
df_reuse_final["twist"] = df_reuse_final.geom.apply(lambda h: twists[hashes_in_order.index(h)])


plt.close("all")
fig = plt.figure(figsize=(10,8))
gs = fig.add_gridspec(3, 3)
axes = dict()

axes["error_HChain10"] = fig.add_subplot(gs[0, 0])
axes["error_C2H4"] = fig.add_subplot(gs[0, 1])
axes["error_Cyclobutadiene"] = fig.add_subplot(gs[0, 2])
axes["spread_HChain10"] = fig.add_subplot(gs[1, 0])
axes["spread_C2H4"] = fig.add_subplot(gs[1, 1])
axes["spread_Cyclobutadiene"] = fig.add_subplot(gs[1, 2])
axes["PES_C2H4"] = fig.add_subplot(gs[2, :])

for molecule in ["HChain10", "C2H4", "Cyclobutadiene"]:
    df_reuse_error = df_full[(df_full.molecule == molecule) & (df_full.method.str.contains("reuse"))]
    # df_reuse_spread = df_spread[(df_spread.molecule == molecule) & (df_spread.method.str.contains("reuse"))]
    df_relative_spread = df_relative[(df_relative.molecule == molecule) & (df_relative.method.str.contains("reuse"))]
    ax_error = axes["error_" + molecule]
    ax_spread = axes["spread_" + molecule]
    plot_df(ax_error,
            df_reuse_error,
            x="samples",
            y="error",
            experiment="experiment",
            experiment_order=experiments,
            boxplots=True,
            **plot_kwargs)

    plot_df(ax_spread,
            # df_reuse_spread,
            df_relative_spread,
            x="samples",
            y="error",
            experiment="experiment",
            boxplots=False,
            **plot_kwargs)

    for ax in [ax_error, ax_spread]:
        ax.set_xscale("log")
        ax.set_xlabel("samples for fine-tuning")
        x_ticks = [1e6, 3e6, 1e7, 3e7, 10e7]
        ax.set_xticks(x_ticks)
        ax.set_xlim([min(x_ticks), max(x_ticks)])
        ax.xaxis.set_major_formatter(lambda x, pos: format_with_SI_postfix(x))


    if molecule == "HChain10":
        ax_error.set_ylim([-2, 20])
        ax_spread.set_ylim([-5, 5])
        ax_error.set_ylabel("$E - E_\\mathrm{ref}$ / mHa")
        ax_spread.set_ylabel("$\\Delta E - \\Delta E_\\mathrm{ref}$ / mHa")
        # ax_spread.set_ylabel("relative energy deviation\n$\max (E-E_\\mathrm{ref}) - \min (E-E_\\mathrm{ref}$ / mHa")
        ax_error.legend(loc="upper right")
    elif molecule == "C2H4":
        ax_error.set_ylim([-10, 100])
        ax_spread.set_ylim([-10, 100])
    elif molecule == "Cyclobutadiene":
        pass
        ax_error.set_ylim([-10, 350])
        ax_spread.set_ylim([-5, 100])
        # ax_error.legend(loc="upper right")
    ax_error.set_title(dict(C2H4="CH$_4$ $\\rightarrow$ C$_2$H$_4$",
                            HChain10="H$_6$ $\\rightarrow$ H$_{10}$",
                            Cyclobutadiene="C$_2$H$_4$ $\\rightarrow$ C$_4$H$_4$")[molecule], fontsize=14)
    ax_spread.axhline(color='k')
    ax_error.axhline(color='k')

ax_PES = axes["PES_C2H4"]
plot_df(ax_PES,
        df_reuse_final,
        x="twist",
        y="E",
        experiment="experiment",
        experiment_order=experiments,
        **plot_kwargs)
ax_PES.legend(loc="upper left")
ax_PES.set_title("PES for C$_2$H$_4$", fontsize=14)
ax_PES.set_ylabel("E / Ha")
ax_PES.set_xlabel("twist angle of C=C bond / °")
ax_PES.set_xticks([0, 30, 60, 90])
ax_PES.set_xlim([-1, 91])
fig.tight_layout(w_pad=0.01, h_pad=0)
bbox = ax_PES._position
ax_PES.set_position([bbox.min[0], bbox.min[0]-0.045, bbox.width, bbox.height])

fig.text(0.02, 0.97, "a.", fontsize=16, fontweight="bold")
fig.text(0.02, 0.65, "b.", fontsize=16, fontweight="bold")
fig.text(0.02, 0.31, "c.", fontsize=16, fontweight="bold")

img_ax = ax_PES.inset_axes(bounds=[0.7, 0.1, 0.4, 0.45])


img_ethene = Image.open("/results/03_paper_unversal_wavefuncion/figures/Ethene_twist.png")
img_ax.imshow(img_ethene)
img_ax.axis("off")

if save_figs:
    fig.savefig( f"/results/03_paper_unversal_wavefuncion/figures/reuse_from_smaller.png", dpi=400, bbox_inches="tight")
    fig.savefig(f"/results/03_paper_unversal_wavefuncion/figures/reuse_from_smaller.pdf", bbox_inches="tight")

# %%
