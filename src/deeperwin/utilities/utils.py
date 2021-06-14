import os
import numpy as np
import tensorflow as tf
import subprocess
import getpass

from deeperwin.references.references import get_reference_energies
from deeperwin.utilities.erwinConfiguration import DefaultConfig

TARGET_ENERGIES = get_reference_energies()
MOLECULES_IN_DEFAULT_ORDER = ['H', 'He', 'H2', 'Li', 'LiH', 'Be', 'B', 'C', 'N', 'O', 'Be2', 'F', 'Ne']


def getUserWorkingDirectories():
    """
    Utility functions that user-specific paths to enable cross-user scripts.
    """
    RUN_DIR, RESULTS_DIR, TB_DIR = "", "", ""
    try:
        username = getpass.getuser()
        if username == 'Michael Scherbela':
            RUN_DIR = r'C:\PhD\runs'
            RESULTS_DIR = r'C:\PhD\ucloud\results'
            TB_DIR = r'C:\PhD\tb'
        elif username == 'mscherbela':
            RUN_DIR = '/users/mscherbela/runs'
            RESULTS_DIR = '/users/mscherbela/ucloud/results'
            TB_DIR = '/users/mscherbela/tb'
        elif username == 'rafael':
            RUN_DIR = '/home/rafael/Work/Projects/2018_Deep_Learning/runs/erwin'
            RESULTS_DIR = '/home/rafael/Work/Projects/2018_Deep_Learning/results/erwin'
            TB_DIR = '/home/rafael/Work/Projects/2018_Deep_Learning/runs/erwin/tb'
        elif username == 'leongerard':
            RUN_DIR = ""
    except KeyError:
        pass
    return RUN_DIR, RESULTS_DIR, TB_DIR
RUN_DIR, RESULTS_DIR, TB_DIR = getUserWorkingDirectories()


def sortTogether(index_list, *lists, descending=False):
    """Sort a and multiple variables b together according to the order of a"""
    for b in lists:
        assert len(b) == len(index_list), "List lengths for index list and target list do not match"
    indices = [i[0] for i in sorted(enumerate(index_list), key=lambda x:x[1])]
    if descending:
        indices = indices[::-1]
    r = [[index_list[i] for i in indices]]
    for b in lists:
        r.append([b[i] for i in indices])
    return tuple(r)

def getAutoCorrelation(x):
    """
    Calculates the autocorrelation of a 1D timeseries with itself, i.e. :math:`c(\tau) = \sum_t x(t) x(t+\tau)`
    Args:
        x (np.array): Input data

    Returns:
        (np.array): Autocorrelation of x for all :math:`\tau` >= 0
    """
    x = np.array(x)
    x = (x - np.mean(x)) / np.std(x)
    N = len(x)
    corr_raw = np.correlate(x, x, mode='same')[N//2:]
    n_terms = np.arange(len(x), len(x)-len(corr_raw), -1)
    return corr_raw / n_terms

def smoothTimeseries(x, tau=20):
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

def getRuningAverageStdDev(x, n=20):
    """
    Calculate the std-dev. of input data x along a sliding window. Basically out(i) == std(x[i:i+n]).

    Args:
        x (np.array): Input data
        n (int): window size. At the edges the effective window size linearly decreases down to 1

    Returns:
        (np.array): std-dev along the timeseries. Same length as input

    """
    output = []
    for i in range(len(x)):
        output.append(
            np.std(x[max(i-n,0):i+1])
        )
    output[0] = output[1] # otherwise std-dev of first sample is always 0
    return np.array(output)


def getCodeVersion():
    """
    Determine the current git commit by calling 'git log -1'.

    Returns:
         (str): Git commit hash and latest commit message
    """
    try:
        path = os.path.dirname(__file__)
        msg = subprocess.check_output(['git', 'log', '-1'], cwd=path, encoding='utf-8')
        return msg.replace('\n', '; ')
    except Exception as e:
        print(e)
        return "Unknown code version"

def apply_module_shortcut_dict(module_names):
    """
    Converts module names (embedding, backflow_factor_orbitals, backflow_factor_general, symmetric) to shortcuts which maps each module to their specific neural network weights.
    This is e.g. needed in the shared optimization to differentiate between shared and none shared weights.

    Args:
        module_names (list): List of strings where each string represents a module

    Returns:
        (list of str): Shortcuts of modules.
    """
    ret = []

    shortcut_dict = {"embedding": ["emb"],
                     "backflow_factor_orbitals": ["backflow_factor_net_spin","ci_weigths", "backflow_factor_weight"],
                     "backflow_factor_general": ["backflow_factor_general_net", "backflow_factor_weight"],
                     "symmetric": ["sym"]}

    for name in module_names:
        if name in shortcut_dict:
            for shortcut in shortcut_dict[name]:
                ret.append(shortcut)
        else:
            ret.append(name)
    return list(set(ret)) #make it unique

# docstr-coverage:excused `self explanatory`
def is_in_module(var_name, module_name):
    return module_name in var_name


def get_hist_range(ion_positions, margin=2.0):
    """
    Calculates the corners of a 2D rectangle that contains x,y coords of all ions + a specified margin.

    Args:
        ion_positions (np.array): Shape [Nx3]
        margin (float): Margin from ions to box edge

    Returns:
        (list): x_min, x_max, y_min, y_max

    """
    ion_positions = ion_positions.numpy()#np.array(ion_positions)
    x_min = np.min(ion_positions, axis=0)[:2] - margin
    x_max = np.max(ion_positions, axis=0)[:2] + margin
    return [[x_min[0], x_max[0]], [x_min[1], x_max[1]]]


def get_initializer_from_config(config: DefaultConfig):
    """
    Based on the configuration creates an initializer for the neural network weights. Per default
    the tf.keras.initializers.GlorotUniform is chosen.

    Args:
        config (erwinConfiguration.DefaultConfig): Global config-object.

    Returns:
        (tf.keras.initializers): Neural network weight initializer. The default is tf.keras.initializers.GlorotUniform
    """
    initializer = tf.keras.initializers.get(config.model.initializer_name)
    name = config.model.initializer_name.lower().replace("_", "")
    if name == "randomnormal":
        initializer.stddev = config.model.initializer_scale
    elif name == "randomuniform":
        initializer.minval = -config.model.initializer_scale
        initializer.maxval = config.model.initializer_scale
    elif name == "constant":
        initializer.value = config.model.initializer_scale
    elif name in ['glorotuniform', 'glorotnormal']:
        pass
    else:
        raise ValueError(f"Unknown initializer: {config.model.initializer_name}; shortened: {name}")
    return initializer

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x = np.linspace(-5,20,200)
    y = np.zeros(len(x))
    y[:50] = 1
    y_sm = smoothTimeseries(y, 5/(x[1]-x[0]))

    plt.close("all")
    plt.plot(x, y)
    plt.plot(x, y_sm)