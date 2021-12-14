import numpy as np
import matplotlib.pyplot as plt
import os
from deeperwin.dispatch import load_from_file
import re
import pandas as pd

_REFERENCE_ENERGIES = {'He': -2.90372, 'Li': -7.478067, 'Be': -14.66733, 'B': -24.65371, 'C': -37.84471, 'N': -54.58882,
                       'O': -75.06655, 'F': -99.7329, 'Ne': -128.9366, 'H2': -1.17448, 'LiH': -8.070548,
                       'N2': -109.5423, 'Li2': -14.9954, 'CO': -113.3255, 'Be2': -29.338, 'B2': -49.4141,
                       'C2': -75.9265, 'O2': -150.3274, 'F2': -199.5304, 'H4Rect': -2.0155,
                       'H3plus': -1.3438355180000001, 'H4plus': -1.8527330000000002, 'HChain6': -3.3761900000000002,
                       'HChain10': -5.6655}


d = "/Users/leongerard/Desktop/JAX_DeepErwin/data/old/C2/res"
df = pd.read_csv(d + "/train_stats.csv")
molecule = 'C'
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


#%%

fig, axes = plt.subplots(1, 1, sharex=True, figsize=(12, 7))
axes.set_title("Fermi vs DeepErwin")

epoch = np.arange(0, 10000, 0.25)
epoch_eval = np.arange(10000, 20000, 0.25)
plot_smoothend_data(epoch, np.sqrt(df['variance'][:40000]), axis=axes, label="FermiNet Optimization")
axes.set_ylim([0, 5])
plot_smoothend_data(epoch_eval, np.sqrt(df['variance'][40000:]), axis=axes, label="FermiNet Evaluation")


# deeperwin_path = "/Users/leongerard/Desktop/JAX_DeepErwin/data/compare_optimizer/O/kfac"
# data = load_from_file(os.path.join(deeperwin_path, 'results.bz2'))
# plot_smoothend_data(np.arange(0, len(data['metrics']['opt_epochs'])*4, 4), data['metrics']['opt_E_std'],
#                     label='DeepErwin', color=f'C{4}', axis=axes, tau=10)
axes.legend()
print(np.mean(df['energy'][40000:]) - _REFERENCE_ENERGIES[molecule])