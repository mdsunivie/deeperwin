import numpy as np
import matplotlib.pyplot as plt
from paper_plots.paper2_high_accuracy.waterfall import plot_waterfall
import wandb
import pandas as pd
from matplotlib.cm import get_cmap
import json

#%%
project = "ablation_pauli"
def load_runs_wandb(filter_ids):

    data = {}

    api = wandb.Api()
    runs = api.runs(f"schroedinger_univie/{project}")
    for i, run in enumerate(runs):
        if i % 10 == 0:
            print(f"Rows {i}")
        if all(f not in run.id for f in filter_ids):
            continue
        if run.state == 'running':
            continue
        print(run.name)
        name = "_".join(run.name.split("_")[:-1])
        if name not in data.keys():
            data[name] = dict(intermed_error=[], epochs=[], e_mean=[])
        wandb_data = [row for row in
                          run.scan_history(keys=["opt_epoch", "error_intermed_eval", "sigma_intermed_eval"],
                                           page_size=10_000)]
        intermed_error = [row['error_intermed_eval'] for row in wandb_data] + [run.summary['error_eval']]
        epochs = [row['opt_epoch'] for row in wandb_data] + [50_000]

        data[name]['intermed_error'].append(intermed_error)
        data[name]['epochs'].append(epochs)

        e_mean = [row['opt_E_mean_unclipped'] for row in
                          run.scan_history(keys=["opt_E_mean_unclipped"],
                                           page_size=10_000)]
        data[name]['e_mean'].append(e_mean)
    return data

# 1&2: env + pauli, 3&4: cas + pauli, 5: deeperwin
filter_NH3 = ['3h4guxx7', '3r4mxzfx', '1y224nu6', 'zcljdrjm', '2c5j584v', '1vdho2py', 'd2tnhc4r'][:5]
data_pauli = load_runs_wandb(filter_NH3)
data_fname = "/home/mscherbela/ucloud/results/02_paper_figures_high_acc/plot_data/ablation_paulinet.json"
with open(data_fname, 'w') as f:
    json.dump(data_pauli, f)

#%%
data_fname = "/home/mscherbela/ucloud/results/02_paper_figures_high_acc/plot_data/ablation_paulinet.json"
with open(data_fname) as f:
    data_pauli = json.load(f)

def smooth(x, ratio=0.2):
    ratio = 1 - ratio
    x = np.array(x)
    x_smooth = [np.mean(x[int(ratio*i):max(i,1)]) for i in range(len(x))]
    return x_smooth

nb_samples = int(50000*0.1)
for key in data_pauli.keys():
    e_mean = data_pauli[key]['e_mean'][0][:50_000]
    mva = []
    for i in range(0, len(e_mean) - nb_samples):
        mva.append(np.mean(e_mean[i:i+nb_samples]))
    data_pauli[key]['mva'] = smooth(e_mean)

def correct_nb_epochs(x):
    if "fixed" in x:
        nb_epochs = x.split("_")[-3]
        return int(nb_epochs)
    nb_epochs = x.split("_")[-2]
    return int(nb_epochs)


excel_file_path = "/home/mscherbela/ucloud/results/02_paper_figures_high_acc/plot_data/pretrainingV3.xlsx"
df_pretraining = pd.read_excel(excel_file_path)

#%%
published_errors = {'O': np.NAN, 'NH3': 1.4}
molecule = 'NH3'
ablation_error = dict(
    ThisWork=[0.71, 0.73],
    NoFullDet=[0.84, 0.93],
    EmbHP=[2.04, 2.05],
    NoEnv=[10.43, 10.51],
    RemIonDiff=[14.51, 15.38],
    #CusplessFeat=[13.72, 14.32],
    PauliNet=[13.13, 13.19]
)
labels_delta = ["Block-diag.\ndeterminant", "PauliNet\nemb. + HP", "CASSCF\nenvelope", 'No $\mathbf{r}_i - \mathbf{R}_I$\nfeature',
                'Cusp less\nfeatures']

color_changes = get_cmap("inferno")(0.35) #'lightgrey'
colors_delta = [color_changes, color_changes, color_changes, color_changes]
#color_paulishift = 'lightslategrey'
color_pauli = get_cmap("inferno")(0.05) #'dimgrey'
color_env = get_cmap("inferno")(0.65) #'teal'
color_dpe = get_cmap("viridis")(0.85) #'lightskyblue'
color_summary = [color_dpe, color_env,color_pauli]
labelpad=2
plt.close("all")

fig, ax = plt.subplots(1, 3, dpi=100, figsize=(11, 4.5), gridspec_kw={'width_ratios': [1.0, 1.0, 1.1]})
errors = [np.mean(np.array(error)) for _, error in ablation_error.items()]
uncertainties = [np.std(np.array(error)) for _, error in ablation_error.items()]
plot_waterfall(errors,
               label_start="This work",
               labels_delta=labels_delta,
               label_end="CASSCF\nPauliNet",
               summary_points=[(2, 'Exp. envelope\nPauliNet')],
               color_delta=colors_delta,
               color_summary=color_summary,
               label_rotation=0,
               y_err=uncertainties,
               ylim=[min(np.nanmin(errors)*1.1-0.5, 0), min(max(np.nanmax(errors)*1.1,0.2), 62)],
               ax=ax[0],
               value_position='center',
               horizontal=True,
               textcolors=['k', 'k', 'k', 'k', 'w', 'w', 'w', 'w'],
               label_in_bar_thr=0.12,
               labeloffset=0.8
               )

ax[0].axvline(published_errors[molecule], color='k', ls='--',
              label='Best published\nvariational: [Pfau 2020],\n200k epochs')
ax[0].text(0.08, 1.04, "a.)", ha='right', va='center', transform=ax[0].transAxes, weight='bold')
ax[0].set_xlabel("Energy rel. to reference / mHa")

ax[0].grid(axis='x', alpha=0.5)
ax[0].axvline(0, color='k', zorder=10)
ax[0].set_title("CASSCF vs. exp. envelope")
ax[0].legend(loc='upper right', framealpha=0.9, fontsize=9)


labels = ["Exp. envelope PauliNet", "CASSCF PauliNet", "This work"]
for i, key in enumerate(['ablation_pauli_01_NH3', 'ablation_env_05_NH3', 'copy_of_2_regression_nips_fulldet_NH3_tf32']):
    error = (56.5644 + np.array(data_pauli[key]['mva']))*1000

    if 'ablation_env_05' in key:
        label = labels[0]
        color = color_env
    elif 'ablation_pauli_01_NH3' in key:
        label = labels[1]
        color = color_pauli
    elif 'copy_of_2_regression_nips_fulldet_NH3_tf32' in key:
        label = labels[2]
        color = color_dpe

    ax[1].plot(np.arange(100, len(error))/1000, error[100:], label=label, color=color)
ax[1].set_ylim([0, 25])
ax[1].axhline(published_errors[molecule], color='k', ls='--', label='Best published variational:\n[Pfau 2020], 200k epochs')
ax[1].legend(loc='upper right', framealpha=0.9, fontsize=9)
ax[1].set_title(f"CASSCF vs. exp. envelope")
ax[1].set_xlabel("Epochs / 1k")
ax[1].grid(alpha=0.5)
ax[1].set_ylabel("Energy rel. to reference / mHa", labelpad=labelpad)
ax[1].text(0.08, 1.04, "b.)", ha='right', va='center', transform=ax[1].transAxes, weight='bold')


metrics = ['error_final', 'error_intermed20000']
#color = ['forestgreen', 'yellowgreen', 'brown', 'darkgoldenrod']
color = [get_cmap("inferno")(0.2), get_cmap("inferno")(0.35), get_cmap("viridis")(0.8), get_cmap("viridis")(0.85)]
data_pretraining = df_pretraining[df_pretraining['molecule'] == 'NH3']

for j, meth in enumerate(['prod_cas', 'prod_hf']):
    data = data_pretraining[data_pretraining['method'] == meth]
    data = data.assign(nb_epochs=data['name'].apply(correct_nb_epochs))
    data = data.sort_values(by=['nb_epochs'])
    data = data.groupby(['nb_epochs']).agg(['mean', 'std']).reset_index()

    if meth == 'prod_cas':
        label = "CASSCF"
    else:
        label = "Hartree Fock"
    ax[2].errorbar(data['nb_epochs'].unique()/1000, data['error_final']['mean'], yerr=data['error_final']['std'],
                      label=f"{label} - 50k opt. epochs", color=color[2*j], capsize=3)
    ax[2].errorbar(data['nb_epochs'].unique()/1000, data['error_intermed20000']['mean'],
                      yerr=data['error_intermed20000']['std'], label=f"{label} -  20k opt. epochs", color=color[2*j + 1], capsize=3, linestyle='--')

ax[2].set_xscale("log")
ax[2].set_xticks(np.array([0.2, 1, 5, 20, 100]))
ax[2].set_xticklabels([0.2, 1, 5, 20, 100])
ax[2].set_xlabel("Pretraining epochs / 1k")
ax[2].set_ylim([0.0, 3.0])
ax[2].grid(alpha=0.5)
ax[2].set_title(f"Pretraining")
ax[2].legend(loc='upper right', framealpha=0.9, fontsize=9)
ax[2].set_ylabel("Energy rel. to reference / mHa", labelpad=labelpad)
ax[2].text(0.07, 1.04, "c.)", ha='right', va='center', transform=ax[2].transAxes, weight='bold')

fig.tight_layout()
fig.subplots_adjust(wspace=0.225)
fig_fname = f"/home/mscherbela/ucloud/results/02_paper_figures_high_acc/Prior_physics.png"
fig.savefig(fig_fname, bbox_inches='tight', dpi=400)
fig.savefig(fig_fname.replace('.png', '.pdf'), bbox_inches='tight')

