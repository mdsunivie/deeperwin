import wandb
import pandas as pd
import pickle
import gzip
import numpy as np
from collections import namedtuple

EnsembleEnergies = namedtuple("EnsembleEnergies", "n_epochs, n_samples, error_mean, error_std, error_25p, error_75p, error_05p, error_95p")

def remove_data_at_epoch(data: EnsembleEnergies, n_epoch):
    if n_epoch in data.n_epochs:
        ind = data.n_epochs.index(n_epoch)
        for value in data:
            del value[ind]

def get_all_eval_energies(wandb_id, print_progress=False, n_max=-1):
    api = wandb.Api()
    runs = api.runs(wandb_id)
    data = []
    for i,run in enumerate(runs):
        if len(data) == n_max:
            break
        # if (run.state != "finished") or ('E_mean' not in run.summary):
        #     if print_progress:
        #         print("Failed or not evaluated: skipping")
        #     continue
        if print_progress:
            print(f"Loading {wandb_id}: {i+1}/{len(runs)}")
        data_dict = {}
        data_dict.update(run.config)
        data_dict.update(run.summary)
        data_dict['name'] = run.name

        E_ref = float(data_dict['physical.E_ref']) if 'physical.E_ref' in data_dict else None

        history = run.scan_history(keys=["opt_epoch", "E_intermed_eval_mean"], page_size=10000)
        for row in history:
            n_epoch = row['opt_epoch']
            data_dict[f'E_mean_eval_{n_epoch:05d}'] = row['E_intermed_eval_mean']
            if E_ref is not None:
                data_dict[f'error_eval_{n_epoch:05d}'] = (float(row['E_intermed_eval_mean']) - E_ref) * 1e3

        if 'E_mean' in run.summary:
            n_epoch = int(run.config['optimization.n_epochs']) + int(run.config['optimization.n_epochs_prev'])
            data_dict[f'E_mean_eval_{n_epoch:05d}'] = run.summary['E_mean']
        data.append(data_dict)
    return pd.DataFrame(data)

def get_ensable_averge_energy_error(df, exclude_unrealistic_deviations=False):
    epochs = sorted([int(c.split('_')[-1]) for c in list(df) if c.startswith('E_mean_eval_')])
    data = EnsembleEnergies([], [], [], [], [], [], [], [])

    for n_epoch in epochs:
        c = f'E_mean_eval_{n_epoch:05d}'
        error_eval = (df[c].astype(float) - df['physical.E_ref'].astype(float)).values*1e3
        error_eval = np.array(error_eval, dtype=float)
        error_eval = error_eval[np.isfinite(error_eval)]
        if exclude_unrealistic_deviations:
            error_eval = error_eval[np.abs(error_eval) < 100]
        data.n_epochs.append(n_epoch)
        data.n_samples.append(len(error_eval))
        if len(error_eval) == 0:
            data.error_mean.append(np.nan)
            data.error_std.append(np.nan)
            data.error_05p.append(np.nan)
            data.error_25p.append(np.nan)
            data.error_95p.append(np.nan)
            data.error_75p.append(np.nan)
        else:
            data.error_mean.append(np.mean(error_eval))
            data.error_std.append(np.std(error_eval))
            data.error_05p.append(np.percentile(error_eval, 5))
            data.error_25p.append(np.percentile(error_eval, 25))
            data.error_95p.append(np.percentile(error_eval, 95))
            data.error_75p.append(np.percentile(error_eval, 75))
    return data


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    #%%
    df_full = get_all_eval_energies("schroedinger_univie/PES_H10_Shared75_adam_eval", print_progress=True, n_max=265)
    df_full['optimizer'] = df_full['optimization.optimizer.name']

    #%%
    names = df_full.name

    wf = {}
    for i in range(49):
        wf[i] = []
    for name in names:
        wf_nb = int(name.split("/")[0].split("_")[-1])

        wf[wf_nb].append(int(name.split("/")[1].replace("chkpt", "")))

    #%%
    optimizers = df_full.optimizer.unique()

    plt.close("all")
    for opt in optimizers:
        df_filt = df_full[(df_full.optimizer == opt) & (df_full.name.str.contains('indep'))]
        n_epochs, n_samples, error_eval = get_ensable_averge_energy_error(df_filt)[:3]
        print(n_samples)
        plt.semilogx(n_epochs, error_eval, label=f'Independent: {opt}', marker='s')

    for opt in optimizers:
        df_filt = df_full[(df_full.optimizer == opt) & (df_full.name.str.contains('reuse'))]
        if len(df_filt) == 0:
            continue
        n_epochs, n_samples, error_eval = get_ensable_averge_energy_error(df_filt)[:3]
        print(n_samples)
        plt.semilogx(n_epochs, error_eval, label=f'Reuse after 16k pretrain: {opt}', marker='s')

    # dir = "/Users/leongerard/Desktop/JAX_DeepErwin/data/Fig2_weight_sharing.pkl.gz"
    # with gzip.open(dir, "rb") as f:
    #     df_tf = pickle.load(f)
    # df_tf = df_tf[df_tf.name == 'HChain6']
    # df_tf = df_tf[df_tf.sharing == '']
    # df_tf = df_tf.groupby('n_epochs').error_eval.mean()
    # plt.semilogx(df_tf.index, df_tf.values, label=f'TensorFlow/arxiv independent (adam)', marker='s')

    plt.axhline(1.6, label="Chemical accuracy", color='gray')
    plt.legend()
    plt.xlabel("Training epochs")
    plt.ylabel("Error / mHa")
    plt.grid(alpha=0.7, ls=':')
    x_ticks = [2**n for n in range(9, 15)]
    plt.gca().set_xticks(x_ticks)
    plt.gca().set_xticklabels(map(str, x_ticks))
    plt.minorticks_off()
    plt.title("H6: 49 geometries")
    #plt.savefig("/home/mscherbela/ucloud/results/H6_JAX_independent.png")
    #%%






