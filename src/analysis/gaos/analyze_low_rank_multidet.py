#%%
from wandb import Api
from deeperwin.run_tools.load_wandb_data import load_wandb_data
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from deeperwin.utils.utils import add_value_texts_to_barchart
sns.set_theme(style="whitegrid")

data = load_wandb_data("gao",
                       fname="/home/mscherbela/tmp/run_tmps/multidet.csv",
                       run_name_filter_func=lambda n: n.startswith("2023-03-31_tao"),
                       load_opt_energies=True,
                       load_fast=True)

def get_det_mode(run_name):
    if "ElDownmapV2" in run_name:
        return "el_downmap"
    elif "OrbDownmap" in run_name:
        return "orb_downmap"
    elif "tao_full" in run_name:
        return "full"
    return "other"

data['n_determinants'] = data['model.orbitals.n_determinants']
data['det_mode'] = data['name'].apply(get_det_mode)
data['det_mode'] = data.det_mode + data.n_determinants.apply("_{:.0f}".format)
data = data[~data.det_mode.isin(["other"])]

df_ref = pd.read_csv("/home/mscherbela/runs/references/high_accuracy_references.csv")
df_ref = df_ref[["molecule", "E_DPE_NEURIPS_50k"]]
df_ref = df_ref.rename(columns={"E_DPE_NEURIPS_50k": "E_ref"})
data = pd.merge(data, df_ref, how='left', on='molecule')
data['error_20k'] = (data['E_mean_20k'] - data.E_ref) * 1000
data['error_50k'] = (data['E_mean_50k'] - data.E_ref) * 1000

# %%
data_filt = data[~data.molecule.isin(["Benzene", "N2_bond_breaking"])]

plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

hues = dict(full_4='lightgray', full_16='dimgray', el_downmap_16='C0', orb_downmap_16='C1')

sns.barplot(x="molecule", y="error_20k", hue="det_mode", palette=hues, hue_order=hues.keys(), data=data_filt, ax=axes[0])
sns.barplot(x="molecule", y="error_50k", hue="det_mode", palette=hues, hue_order=hues.keys(), data=data_filt, ax=axes[1])

axes[0].set_title("20k epochs")
axes[1].set_title("50k epochs")
add_value_texts_to_barchart(axes, space=0.005, fontsize=10, rotation=90, color='k')

for ax in axes:
    ax.set_ylabel("Error rel. to 50k DPE NEURIPS (32dets) [mHa]")
    ax.set_ylim([0, 5])
fig.tight_layout()
fig.savefig("/home/mscherbela/ucloud/results/gao_multidet.png", dpi=300, bbox_inches="tight")

# %%
