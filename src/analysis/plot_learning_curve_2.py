import parse
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import load_from_file
import re
from matplotlib.ticker import FormatStrFormatter
default_hp = {'momentum': '0.0', 'norm_constraint': '0.001', 'damping': '0.001', 'learning_rate': '0.0001', 'lr_decay_time_kfac': '6000'}


REFERENCE_ENERGIES = {'He': -2.90372, 'Li': -7.47806, 'Be': -14.66736, 'B': -24.65391, 'C': -37.845, 'N': -54.5892, 'O': -75.0673, 'F': -99.7339, 'Ne': -128.9376, 'H2': -1.17448, 'LiH': -8.07055, 'N2': -109.5423, 'Li2': -14.9954, 'CO': -113.3255, 'Be2': -29.338, 'B2': -49.4141, 'C2': -75.9265, 'O2': -150.3274, 'F2': -199.5304, 'H4Rect': -2.0155, 'H3plus': -1.3438355180000001, 'H4plus': -1.8527330000000002, 'HChain6': -3.3761900000000002, 'HChain10': -5.6655}
REFERENCE_ENERGIES['H10'] = REFERENCE_ENERGIES['HChain10']

def smoothen_timeseries(x, tau=20):
    """
    Smoothen a 1D timeseries by applying an exponentially decaying. Samples are assumed to be spaced equidistant.
    Args:
        x (np.array): Input data
        tau: decay constant (longer = smoother)
    Returns:
        (np.array): Smoothed timeseries of same length as input
    """
    epsilon = 1e-4
    kernel_length = int(max(-tau * np.log(epsilon), 2))
    kernel = np.exp(-np.arange(kernel_length)/tau)
    kernel = kernel / np.sum(kernel)
    x_temp = np.pad(x, kernel_length, 'edge')
    y = np.convolve(x_temp, kernel, 'valid')
    y = y[1:-kernel_length]
    return y

def plot_smoothend_data(t, y, tau=20, color=None, axis = None, label=None):
    y_smooth = smoothen_timeseries(y, tau)
    if axis is None:
        axis = plt.gca()
    axis.plot(t, y, alpha=0.2, color=color)
    axis.plot(t, y_smooth, color=color, label=label)

def load_learning_curves(fname):
    with open(fname) as f:
        content = f.read()
    n_opt = []
    E_opt_mean = []
    E_opt_std = []
    #if "Benchmark" in fname:


    for r in parse.findall("opt Epoch  {n_epoch}: opt_E_mean={E_epoch:9.5f}; opt_E_std={E_epoch_std:9.5f}; opt_mcmc_stepsize={step_size}; opt_mcmc_max_age={max_age:9.5f}; opt_t_epoch={time}", content):
        n_opt.append(r['n_epoch'])
        E_opt_mean.append(r['E_epoch'])
        E_opt_std.append(r['E_epoch_std'])
    return np.array(n_opt, int), np.array(E_opt_mean), np.array(E_opt_std)
    # for r in parse.findall("Epoch {n_epoch:>d}/{n_opt_steps:>4}: E={E_epoch:9.5f}+-{E_epoch_std:9.5f}", content):
    #     n_opt.append(r['n_epoch'])
    #     E_opt_mean.append(r['E_epoch'])
    #     E_opt_std.append(r['E_epoch_std'])
    # return np.array(n_opt, int), np.array(E_opt_mean), np.array(E_opt_std)

def build_folder_dict(root_dir):
    directory, dirs, fnames = next(os.walk(root_dir))

    runs = {}
    for d in dirs:
        # if not re.match(".*"+filter+".*", d):
        runs[d] = []

    return runs


def load_all_runs(root_dir, load_full_data=True, filter=""):

    runs = build_folder_dict(root_dir)
    for directory, dirs, fnames in os.walk(root_dir):
        if ('GPU.out' in fnames):
            name = "/".join(directory.split('/')[-2])
            full_data = load_from_file(os.path.join(directory, 'results.bz2'))
            runs[name].append(full_data)

    return runs


def plot_learning_curves_for_directory_only_eval(root_dir, molecule=None, filter="", sorting_key=None, smoothing=10):
    names, data, full_data = load_all_runs(root_dir, filter=filter, load_full_data=True)
    if sorting_key is not None:
        ind_sorted = np.argsort([sorting_key(n) for n in names])
    else:
        ind_sorted = np.argsort(names)
    names = [names[i] for i in ind_sorted]
    data = [data[i] for i in ind_sorted]
    full_data = [full_data[i] for i in ind_sorted]

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 7))
    for i, (name, d, full_d) in enumerate(zip(names, data, full_data)):
        if full_d is not None:
            E_eval = full_d['eval'].E_mean
            if molecule is not None:
                energy_error = 1e3 * (np.nanmean(E_eval) - REFERENCE_ENERGIES[molecule])
                std_energy_error = 1e3 * np.nanstd(E_eval) / np.sqrt(len(E_eval))
                name = name + f" ({energy_error:+.1f} +- {std_energy_error:.1f} mHa)"
            plot_smoothend_data(np.arange(len(E_eval)), E_eval, label=name, color=f'C{i}', axis=axes[i], tau=smoothing)
            axes[i].set_ylabel("E mean eval")
            axes[i].set_xlabel("epoch eval")
            axes[i].set_ylim([REFERENCE_ENERGIES[molecule]-0.3, REFERENCE_ENERGIES[molecule]+0.3])

        # plot_smoothend_data(d[0], d[1], label=name, color=f'C{i}', axis=axes[0], tau=smoothing)
        # plot_smoothend_data(d[0], d[2], label=name, color=f'C{i}', axis=axes[1], tau=smoothing)

    # axes[0].set_ylabel("E mean")
    # axes[1].set_ylabel("E std")
    # axes[1].set_xlabel("epoch train")
    # axes[2].set_ylabel("E mean eval")
    # axes[2].set_xlabel("epoch eval")
    # if molecule is not None:
    #     for i in [0,2]:
    #         axes[i].axhline(REFERENCE_ENERGIES[molecule], color='k', linestyle='--', label=f'Reference {molecule}')
    #     axes[2].set_ylim(REFERENCE_ENERGIES[molecule]+np.array([-0.5, 0.5]))
    # axes[1].axhline(0, color='k', linestyle='--')

    # axes[0].set_ylim([REFERENCE_ENERGIES[molecule] -4.0 , REFERENCE_ENERGIES[molecule] +4.0])
    # axes[1].set_ylim([0, 4.0])
    for n in range(3):
        axes[n].legend()
        axes[n].grid(alpha=0.3, color='gray')

def plot_learning_curves_for_directory(root_dir, molecule=None, filter="", sorting_key=None, smoothing=10):
    runs = load_all_runs(root_dir, filter=filter, load_full_data=True)


    fig, axes = plt.subplots(len(runs.keys()), 1, sharex=True, figsize=(12, 7))
    color = {'adam': 'C0', 'kfac': 'C1', 'slbfgs': 'C2'}
    for i, name in enumerate(runs.keys()):
        mol = runs[name]
        kfac_counter = 0
        for j in range(len(mol)):
            epochs = mol[j]['metrics']['opt_epochs']
            E_std = mol[j]['metrics']['opt_E_std']
            error = np.round(mol[j]['metrics']['error_plus_2_stdev'], 2)
            label = f"{mol[j]['config']['optimization.optimizer.name']}: {error:.2f} mHa"


            if kfac_counter == 1 and mol[j]['config']['optimization.optimizer.name'] == 'kfac':
                plot_smoothend_data(epochs, E_std, label=label,
                                    color='C4', axis=axes[i],
                                    tau=smoothing)
                kfac_counter == 0
            else:
                plot_smoothend_data(epochs, E_std, label=label, color=color[mol[j]['config']['optimization.optimizer.name']], axis=axes[i],
                                    tau=smoothing)

            if mol[j]['config']['optimization.optimizer.name'] == 'kfac':
                kfac_counter +=1
            print(label, len(epochs), np.round(mol[j]['metrics']['error_plus_2_stdev'], 2))


        axes[i].set_ylabel("Std. Dev.")
        axes[i].set_ylim([0, 1])
        axes[i].grid(alpha=0.3, color='gray')
        axes[i].set_title(name)
        axes[i].legend()
    axes[2].set_xlabel("Steps")

def plot_evaluation(directory, molecule=None, axis=None, color='C0'):
    data = load_from_file(os.path.join(directory, 'results.bz2'))
    if ('eval' not in data) or (len(data['eval'].E_mean) < 50):
        return
    if axis is None:
        axis = plt.gca()
    E = data['eval'].E_mean
    plot_smoothend_data(np.arange(len(E)), E, color='C0', axis=axis)
    plt.axhline(np.mean(E), color=color, linestyle='--', alpha=0.7)
    axis.set_ylabel("E eval mean")
    axis.set_xlabel("Evaluation epoch")
    if molecule is not None:
        E_ref = REFERENCE_ENERGIES[molecule],
        axis.axhline(E_ref, color='k', linestyle='--', label=f'Reference {molecule}')
        print(f"Evaluation error: {float((np.mean(E)-E_ref)*1e3):+3.1f} mHa")
    axis.grid(alpha=0.3, color='gray')

def load_eval_results(fname, name):
    d = load_from_file(os.path.join(fname, 'results.bz2'))
    E_mean = np.nanmean(d['E_eval_mean'])
    E_mean_sigma = np.nanstd(d['E_eval_mean']) / np.sqrt(len(d['E_eval_mean']))


    tick = d['config'][name]

    return np.array(E_mean), np.array(E_mean_sigma), tick



if __name__ == '__main__':
    plt.close("all")
    directory = '/Users/leongerard/Desktop/JAX_DeepErwin/data/compare_stddev/'

    #plot_eval_energies(directory)
    for molecule in ['C']:
        plot_learning_curves_for_directory(directory, filter='', molecule=molecule)
        plt.suptitle('Comparison of Std. Dev.')

