import pandas as pd
import numpy as np
from deeperwin.wandb_utils import get_all_eval_energies
import re
import pickle

wandb_entity = "schroedinger_univie"
optimizer = 'kfac'
cache_fname = f"/home/mscherbela/tmp/data_H4p_barrier.pkl"

def get_geom_from_exp_name(name):
    return int(re.search(r"[^\d](\d{4})[^\d]", name+" ").group(1))

def get_twist_and_spacing(r):
    #comments have the form: rCC_1.43_twist_10
    tokens = r['physical.comment'].split('_')
    r['rCC'], r['twist'] = float(tokens[1]), float(tokens[3])
    return r

def extract_energy_data(df, geom_key):
    data = {}
    energy_columns = [col for col in list(df) if col.startswith("E_mean_eval_")]
    df_PES = df.groupby(geom_key)[energy_columns].min().reset_index()

    for col in energy_columns:
        n_epochs = int(col.split('_')[-1])
        indices = ~df_PES[col].isnull()
        data[n_epochs] = {}
        data[n_epochs]['geom'] = list(df_PES.loc[indices, geom_key])
        data[n_epochs]['E'] = list(df_PES.loc[indices, col])
    return data

# full_plot_data = dict(H4p={}, Ethene={})
#
# molecule = 'H4p'
# for curve in ['Indep', 'Shared95']:
#     project_name = f"Barrier_{molecule}_{curve}_{optimizer}{'_eval' if 'Shared' in curve else ''}"
#     df = get_all_eval_energies(f"{wandb_entity}/{project_name}", print_progress=True)
#     df['geom'] = df.experiment_name.apply(get_geom_from_exp_name)
#     full_plot_data[molecule][curve] = extract_energy_data(df, geom_key='geom')
#     for data in full_plot_data[molecule][curve].values():
#         data['reaction_coord'] = np.array(data['geom']) / (19 - 1)
#
# with open(cache_fname, 'wb') as f:
#     pickle.dump(full_plot_data, f)

#%%
# molecule='Ethene'
# for curve in ['Indep', 'Shared95']:
#     project_name = f"PES_{molecule}_{curve}_{optimizer}{'_eval' if 'Shared' in curve else ''}"
#     df = get_all_eval_energies(f"{wandb_entity}/{project_name}", print_progress=True)
#     df = df.apply(get_twist_and_spacing, axis=1)
#     full_plot_data[molecule][curve] = extract_energy_data(df, geom_key='twist')
#     for data in full_plot_data[molecule][curve].values():
#         data['reaction_coord'] = np.array(data['geom'])

# with open(cache_fname, 'wb') as f:
#     pickle.dump(full_plot_data, f)

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm
import pickle

optimizer = 'kfac'
epochs_to_plot = [7000]
ref_methods = ['MRCI-D-F12', 'HF', 'CCSD(T)-F12', 'Alijah2008']
colors = {'Indep': 'C0', 'Shared95':'r', 'MRCI-D-F12': 'k', 'HF': 'gray', 'CCSD(T)-F12': 'slategray', 'Alijah2008': 'C2'}
linestyles = {'CCSD(T)-F12': '--', 'Alijah2008': 'None'}
molecule_name = dict(H4p='$H_4^+$', Ethene='Ethene')
curve_labels = {'Indep':'DeepErwin w/o weight-sharing', 'Shared95':'DeepErwin w/ weight-sharing', 'HF':'Hartree-Fock', 'Alijah2008': 'CAS AV7Z (Alijah 2008)'}
markers = {'CCSD(T)-F12':'o', 'Alijah2008': 'd'}
marker_size = {'Alijah2008': 7}
n_epochs_to_plot=dict(H4p=dict(Indep=7000, Shared95=7000), Ethene=dict(Indep=8192, Shared95=8192))

cache_fname = f"/home/mscherbela/tmp/data_H4p_barrier.pkl"
with open(cache_fname, 'rb') as f:
    full_plot_data = pickle.load(f)

plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ind_mol, molecule in enumerate(['H4p', 'Ethene']):
    # Prepare and plot reference data
    if molecule == 'H4p':
        df_ref = pd.read_csv("/home/mscherbela/runs/references/H4p_Barrier/H4plus_barrier.csv")
        df_ref['reaction_coord'] = df_ref['geom'] / (len(df_ref) - 1)
        indices_min = np.array([0, -1], int)
        index_barrier = 9
        ylims = [0, 8]
        images = [plt.imread(f"/home/mscherbela/ucloud/results/paper_figures/molecule_structures/povray/H4p_{i}crop.png") for i in
                  [0, 2,0]]
        inset_ax_size = 0.30
        inset_ax_y = 0.75
    elif molecule == 'Ethene':
        df_ref = pd.read_csv("/home/mscherbela/runs/references/Ethene/Ethene_energies.csv")
        df_ref = df_ref[df_ref.distance.isin([1.28, 1.33, 1.43])]
        df_ref['reaction_coord'] = df_ref['twist']
        indices_min = np.array([0], int)
        index_barrier = -1
        ylims = [0, 220]
        images = [plt.imread(f"/home/mscherbela/ucloud/results/paper_figures/molecule_structures/povray/Ethene.{i:02d}.png") for i in
                  [0, 5, 9]]
        inset_ax_size = 0.3
        inset_ax_y = 0.7
        del ref_methods[ref_methods.index('Alijah2008')]
    df_ref = df_ref.groupby('reaction_coord')[ref_methods].min().reset_index()

    ax = axes[ind_mol]
    for ref_method in ref_methods:
        if ref_method not in list(df_ref):
            continue
        offset = np.array(df_ref[ref_method])[indices_min].mean()
        ax.plot(df_ref.reaction_coord, 1e3 * (df_ref[ref_method] - offset),
                label=curve_labels.get(ref_method, ref_method),
                color=colors[ref_method],
                ls=linestyles.get(ref_method, '-'),
                marker=markers.get(ref_method, None),
                markersize=marker_size.get(ref_method, None))

    for curve in full_plot_data[molecule]:
        n_epochs = n_epochs_to_plot[molecule][curve]
        data = full_plot_data[molecule][curve][n_epochs]
        offset = np.array(data['E'])[indices_min].mean()
        E_aligned = np.array(data['E']) - offset
        ax.plot(data['reaction_coord'], 1e3 * E_aligned,
                label=f'{curve_labels.get(curve, curve)}',
                color=colors[curve],
                ls=linestyles.get(curve, '-'))
    ax.set_title(molecule_name[molecule], fontsize=14)
    ax.set_ylim(ylims)

    n_images = len(images)
    for i, img in enumerate(images):
        center_pos = np.linspace(0.15,0.85,n_images)[i]
        axis_x_pos = center_pos - inset_ax_size/2
        img_ax = ax.inset_axes([axis_x_pos, inset_ax_y, inset_ax_size, inset_ax_size])
        img_ax.imshow(img)
        img_ax.axis('off')

for ax in axes:
    ax.grid(alpha=0.5, color='gray', ls='--')
    ax.set_xlabel("Reaction coordinate")
    ax.set_ylabel("Energy / mHa")
axes[0].legend(loc='lower center')
axes[1].legend(loc='center left')
axes[1].set_xticks([0, 30, 60, 90])
fig.tight_layout()

fname = f"/home/mscherbela/ucloud/results/paper_figures/jax/Barrier_{optimizer}.pdf"
fig.savefig(fname)
fig.savefig(fname.replace(".pdf", ".png"), dpi=400, bbox_inches='tight')

for i, ax in enumerate(axes):
    renderer = fig.canvas.get_renderer()
    bbox = ax.get_tightbbox(renderer)
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted()).padded(0.1)
    plt.savefig(fname.replace(".pdf", f"_{i}.pdf"), bbox_inches=bbox)






