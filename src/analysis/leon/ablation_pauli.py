import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from paper_plots.paper2_high_accuracy.waterfall import plot_waterfall
import wandb
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
            data[name] = dict(intermed_error=[], epochs=[])
        wandb_data = [row for row in
                          run.scan_history(keys=["opt_epoch", "error_intermed_eval", "sigma_intermed_eval"],
                                           page_size=10_000)]
        intermed_error = [row['error_intermed_eval'] for row in wandb_data] + [run.summary['error_eval']]
        epochs = [row['opt_epoch'] for row in wandb_data] + [50_000]

        data[name]['intermed_error'].append(intermed_error)
        data[name]['epochs'].append(epochs)

    return data

#filter_O = ['1bzogum8', '1qp0inwz', '3lqxt6iu', '20uw46wf', '3knm2htv', '2u6cmjzq']
filter_NH3 = ['11hfe4r1', '1vdho2py', 'd2tnhc4r', '1y224nu6', 'zcljdrjm']
data = load_runs_wandb(filter_NH3)


#%%
# metric = 'error_smooth_40k'


published_errors = {'O': np.NAN, 'NH3': 1.4}
labels_delta = ["Features", "Envelope"]
# ablation_error = dict(
#     PauliNet=[5.51, 5.22],
#     Features=[4.2, 4.6],
#     Envelope=[0.47, 0.26]
# )
labels_delta = ["No el-el \n cusp", "Dist. \n feat.", "Ion \n diff. feat.", 'Envelope', 'HP + \n Embed.']
ablation_error = dict(
    PauliNet=[5.11, 5.95],
    No_explicit_el_el_cusp=[8.6, 6.31],
    Distance=[5.4, 6.28],
    IonDiff=[4.22, 4.58],
    Envelope=[0.41, 0.26],
    DPE11=[-0.7, -0.6]
)

molecule = 'NH3'
ablation_error = dict(
    PauliNet=[13.13, 13.19],
    No_explicit_el_el_cusp=[13.72, 14.32],
    Distance=[13.11, 13.11],
    IonDiff=[8.72, 9.04],
    Envelope=[2.32, 1.67],
    DPE11=[0.71, 0.73]
)



ablation_error_no_cusp_shift = [l[-1] for l in data['ablation_paulishift_08_NH3']['intermed_error']]

color_our_improvements = 'C0'
color_other_improvements = 'gray'
colors_delta = [color_our_improvements, color_our_improvements, color_our_improvements,
                color_our_improvements, color_our_improvements, color_our_improvements]
plt.close("all")

fig, ax = plt.subplots(1,2, dpi=100, figsize=(12,4))
#ax = [axs]
errors = [np.mean(np.array(error)) for _, error in ablation_error.items()]
uncertainties = [np.std(np.array(error)) for _, error in ablation_error.items()]
plot_waterfall(errors,
               label_start="CASSCF \n PauliNet",
               labels_delta=labels_delta,
               label_end="DeepErwin",
               summary_points=[(4, 'Env. \n PauliNet')],
               color_delta=colors_delta,
               color_summary='brown',
               y_err=uncertainties,
               ylim=[np.nanmin(errors)*1.1-0.5, min(max(np.nanmax(errors)*1.1,0.2), 20)],
               ax=ax[0])
#ax.bar([np.nan], [np.nan], color=color_other_improvements, label="Improvements through earlier works")
ax[0].bar([np.nan], [np.nan], color=color_our_improvements, label="")
ax[0].bar([np.nan], [np.nan], color='brown', label="Sub-total")
ax[0].axhline(published_errors[molecule], color='k', ls='--', label='FermiNet [Pfau 2020], 200k')
#ax[0].axhline(np.mean(ablation_error_shift), color='k', ls='--', label="CASSCF PauliNet + Features + BF Shift", alpha=0.6)
ax[0].axhline(np.mean(ablation_error_no_cusp_shift), color='grey', ls='--', label="CASSCF PauliNet + BF Shift", alpha=0.6)

ax[0].set_ylabel("energy rel. to CCSD(T)/CBS / mHa")
ax[0].axhline(0, color='k')
ax[0].grid(axis='y', alpha=0.5)
ax[0].set_title(f"Ablation study {molecule}")
ax[0].legend(loc='upper right')


colors = ['brown', 'grey', 'darksalmon']
labels = ["Envelope PauliNet", "CASSCF PauliNet", "CASSCF PauliNet + BF Shift"]
for i, key in enumerate(data):
    epochs = data[key]['epochs'][0]
    print(epochs)
    error = np.array(data[key]['intermed_error']).mean(axis=0)
    std = np.array(data[key]['intermed_error']).std(axis=0)


    if 'ablation_paulishift_08' in key:
        label = labels[2]
        color = colors[2]
    elif 'ablation_env_05' in key:
        label = labels[0]
        color = colors[0]
    else:
        label = labels[1]
        color = colors[1]

    ax[1].errorbar(epochs, error, yerr=std, label=label, color=color)
ax[1].set_title(f"CASSCF vs. Envelope")
ax[1].set_xlabel("Epochs")
ax[1].legend()
ax[1].grid(alpha=0.5)
fig.tight_layout()

