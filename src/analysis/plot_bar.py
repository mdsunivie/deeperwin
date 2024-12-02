import parse
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import load_from_file
from matplotlib.ticker import FormatStrFormatter
default_hp = {'momentum': '0.0', 'norm_constraint': '0.001', 'damping': '0.001', 'learning_rate': '0.0001', 'decay': '6000'}


REFERENCE_ENERGIES = {'He': -2.90372, 'Li': -7.47806, 'Be': -14.66736, 'B': -24.65391, 'C': -37.845, 'N': -54.5892, 'O': -75.0673, 'F': -99.7339, 'Ne': -128.9376, 'H2': -1.17448, 'LiH': -8.07055, 'N2': -109.5423, 'Li2': -14.9954, 'CO': -113.3255, 'Be2': -29.338, 'B2': -49.4141, 'C2': -75.9265, 'O2': -150.3274, 'F2': -199.5304, 'H4Rect': -2.0155, 'H3plus': -1.3438355180000001, 'H4plus': -1.8527330000000002, 'HChain6': -3.3761900000000002, 'HChain10': -5.6655}
REFERENCE_ENERGIES['H10'] = REFERENCE_ENERGIES['HChain10']

# def mapping_fname_config_option(name, d):
#     if name == "decay":
#         return d['config']['optimization.optimizer.decay_time']
#     elif name == "learning_rate":
#         return d['config']['optimization.learning_rate']
#     elif name == "momentum":
#         return d['config']['optimization.optimizer.momentum']
#     elif name == "damping":
#         return d['config']['optimization.optimizer.damping']
#     else:
#         return d['config']['optimization.optimizer.damping']
#
#
# def load_eval_results(fname, name = None):
#
#     d = load_from_file(os.path.join(fname, 'results.bz2'))
#
#     if "Bench" in fname.split("/")[-1]:
#         tick = "Adam Benchmark" #d['config']['optimization']['optimizer']['name']
#         error = (np.nanmean(d['metrics']['E_mean'])- REFERENCE_ENERGIES[ d['config']['physical.name']])*1000
#         E_mean_sigma = d['metrics']['E_mean_sigma']*1000
#     else:
#         #if fname.split("/")[-1] == "KFAC":
#         tick = mapping_fname_config_option(fname.split("/")[-1], d) #d['config']['optimization.optimizer.name']
#         error = d['metrics']['error_eval']
#         E_mean_sigma =d['metrics']['E_mean_sigma']*1000
#         # else:
#         #     tick = fname.split("/")[-1]#d['config'].optimization.optimizer.name #mapping_fname_config_option(fname.split("/")[-2], d)#
#         #     error = (np.nanmean(d['eval'].E_mean) - REFERENCE_ENERGIES[name])*1000#d['metrics']['error_eval']
#         #     E_mean_sigma = (np.nanstd(d['eval'].E_mean)/np.sqrt(len(d['eval'].E_mean)))*1000#d['metrics']['sigma_error_eval']
#
#     return np.array(error), np.array(E_mean_sigma), tick
#
#
#

# def plot_bar(name, energies, axis=None, fig=None):
#     # sort energies?
#     energies = sorted(energies, key=lambda x: x[2])
#     y_e = np.array([float(e) for (e, std, tick) in energies if e.astype(bool)])
#     where_are_NaNs = np.isnan(y_e)
#     y_e[where_are_NaNs] = 0.0
#
#     y_std = np.array([float(std) for (e, std, tick) in energies if e.astype(bool)])
#     where_are_NaNs = np.isnan(y_std)
#     y_std[where_are_NaNs] = 0.0
#
#     x_tick = [tick for (e, std, tick) in energies if e.astype(bool)]
#
#     x = np.arange(len(y_e))
#
#     axis.bar(x,
#              y_e,
#              yerr=y_std,
#              alpha=0.5,
#              color="C1"
#              )
#
#     axis.set_ylabel("Error / mHa")
#     axis.set_title(name)
#     axis.set_xticks(x)
#     axis.set_xticklabels(x_tick)
#     axis.grid(alpha=0.3)
#     if name == "O" or name == "F" or name == "Ne":
#         axis.set_ylim(bottom=0.0, top=20)
#     else:
#         axis.set_ylim(bottom=0.0, top=10)
#     axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#
#     # Color all FermiNet HP
#     if name in default_hp.keys():
#         for i, text in enumerate(axis.get_xticklabels()):
#             if text.get_text() == default_hp[name]:
#                 axis.get_xticklabels()[i].set_color('red')
#
#
# def plot_eval_energies(root_dir, filter=[]):
#     energies_eval = load_eval(root_dir, filter)
#
#     fig, ax = plt.subplots(3, 2, figsize=(12, 5))
#     fig.suptitle("Optimizer Comparison")
#     for i, name in enumerate(sorted(energies_eval.keys())):
#         plot_bar(name, energies_eval[name], axis=ax[i // 2][i % 2], fig=fig)
#     fig.tight_layout()
#
#
# def plot_bar(name, energies, axis=None, fig=None):
# # sort energies?
#     energies = sorted(energies, key=lambda x: x[2])
#     y_e = np.array([float(e) for (e, std, tick) in energies if e.astype(bool)])
#     where_are_NaNs = np.isnan(y_e)
#     y_e[where_are_NaNs] = 0.0
#
#     y_std = np.array([float(std) for (e, std, tick) in energies if e.astype(bool)])
#     where_are_NaNs = np.isnan(y_std)
#     y_std[where_are_NaNs] = 0.0
#
#     x_tick = [tick for (e, std, tick) in energies if e.astype(bool)]
#
#     x = np.arange(len(y_e))
#
#     axis.bar(x,
#              y_e,
#              yerr=y_std,
#              alpha=0.5,
#              color="C1"
#              )
#
#     axis.set_ylabel("Error / mHa")
#     axis.set_title(name)
#     axis.set_xticks(x)
#     axis.set_xticklabels(x_tick)
#     axis.grid(alpha=0.3)
#     if name == "O" or name == "F" or name == "Ne":
#         axis.set_ylim(bottom=0.0, top=20)
#     else:
#         axis.set_ylim(bottom=0.0, top=10)
#     axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#
#     # Color all FermiNet HP
#     if name in default_hp.keys():
#         for i, text in enumerate(axis.get_xticklabels()):
#             if text.get_text() == default_hp[name]:
#                 axis.get_xticklabels()[i].set_color('red')
#
# def plot_eval_energies(root_dir, filter=[]):
#     energies_eval = load_eval(root_dir, filter)
#
#     fig, ax = plt.subplots(3, 2, figsize=(12, 5))
#     fig.suptitle("Optimizer Comparison")
#     for i, name in enumerate(sorted(energies_eval.keys())):
#         plot_bar(name, energies_eval[name], axis=ax[i//2][i%2], fig=fig)
#     fig.tight_layout()


import parse
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import load_from_file
from matplotlib.ticker import FormatStrFormatter

default_hp = {'momentum': '0.0', 'norm_constraint': '0.001', 'damping': '0.001', 'learning_rate': '0.0001',
              'decay': '6000'}

REFERENCE_ENERGIES = {'He': -2.90372, 'Li': -7.47806, 'Be': -14.66736, 'B': -24.65391, 'C': -37.845, 'N': -54.5892,
                      'O': -75.0673, 'F': -99.7339, 'Ne': -128.9376, 'H2': -1.17448, 'LiH': -8.07055, 'N2': -109.5423,
                      'Li2': -14.9954, 'CO': -113.3255, 'Be2': -29.338, 'B2': -49.4141, 'C2': -75.9265, 'O2': -150.3274,
                      'F2': -199.5304, 'H4Rect': -2.0155, 'H3plus': -1.3438355180000001, 'H4plus': -1.8527330000000002,
                      'HChain6': -3.3761900000000002, 'HChain10': -5.6655}
REFERENCE_ENERGIES['H10'] = REFERENCE_ENERGIES['HChain10']




def optimizer_comparison(fname, name=None):
    d = load_from_file(os.path.join(fname, 'results.bz2'))
    tick = fname.split("/")[-1]
    error = d['metrics']['error_eval']
    E_mean_sigma = d['metrics']['E_mean_sigma'] * 1000

    return np.array(error), np.array(E_mean_sigma), tick




def build_eval_dict(root_dir):
    directory, dirs, fnames = next(os.walk(root_dir))

    energies_eval = {}
    for d in dirs:
        # if not re.match(".*"+filter+".*", d):
        energies_eval[d] = []

    return energies_eval


def load_eval(root_dir):
    energies_eval = build_eval_dict(root_dir)

    for directory, dirs, fnames in os.walk(root_dir):
        if ('GPU.out' in fnames):
            name = directory.split("/")[-2]
            d = optimizer_comparison(directory, name)

            energies_eval[name].append(d)
            # except:
            #     pass
    return energies_eval


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

root_dir = '/Users/leongerard/Desktop/JAX_DeepErwin/data/compare_optimizer'
energies_eval = load_eval(root_dir)

#%%
import pandas as pd

def plot_eval_energies_groubed_bar_chart(energies_eval, plot_learning_curve=False):


    kfac = []
    kfac_std = []
    adam = []
    adam_std = []
    adam_17 = []
    adam_std_17 = []
    bfgs = []
    bfgs_std = []
    labels = []

    molecules = ['C', 'N', 'O', 'F', 'Ne']
    for k in molecules:
        for el in energies_eval[k]:
            if el[-1] == "KFAC":
                kfac.append(el[0])
                kfac_std.append(el[1])
            elif el[-1] == "Adam 17k":
                adam_17.append(el[0])
                adam_std_17.append(el[1])
            elif el[-1] == "Adam":
                adam.append(el[0])
                adam_std.append(el[1])
            elif el[-1] == "BFGS":
                bfgs.append(el[0])
                bfgs_std.append(el[1])
        labels.append(k)

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, axes = plt.subplots(1, 1, figsize=(12, 8))
    ax = axes
    ax.grid(alpha=0.2)

    rects1 = ax.bar(x-0.1, kfac, width, yerr=kfac_std, label='KFAC', color="navy", alpha = 1.0)
    rects1 = ax.bar(x - 0.3 , bfgs, width, yerr=bfgs_std, label='BFGS', color="blueviolet", alpha = 1.0)
    rects2 = ax.bar(x +0.1, adam_17, width, yerr=adam_std_17, label='Adam 17k Epoch', color="skyblue", alpha = 1.0)
    rects3 = ax.bar(x + 0.3 , adam, width, yerr=adam_std, label='Adam', color="slategray", alpha = 1.0)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Error / mHa')
    ax.set_title('Optimizer comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)
    if plot_learning_curve:
        full_data = []
        directory = ['/Users/leongerard/Desktop/JAX_DeepErwin/data/kfac_vs_adam_2/N/Adam',
                     '/Users/leongerard/Desktop/JAX_DeepErwin/data/kfac_vs_adam_2/N/KFAC']
        full_data.append(load_from_file(os.path.join(directory[0], 'results.bz2')))
        full_data.append(load_from_file(os.path.join(directory[1], 'results.bz2')))
        plot_smoothend_data(np.arange(len(full_data[0]['metrics']['opt_epochs'])), full_data[1]['metrics']['opt_E_std'], label='KFAC', color=f'C{1}', axis=axes[1], tau=10)
        plot_smoothend_data(np.arange(len(full_data[1]['metrics']['opt_epochs'])), full_data[0]['metrics']['opt_E_std'], label='Adam', color=f'C{0}', axis=axes[1], tau=10)
        axes[1].legend()
        axes[1].set_title('Std. Dev. KFAC vs. Adam for N')
        axes[1].set_ylim([0, 1])
    fig.tight_layout()

    plt.show()


plt.close("all")
plot_eval_energies_groubed_bar_chart(energies_eval)


#    plot_eval_energies(directory, "optimizer_comparison")

