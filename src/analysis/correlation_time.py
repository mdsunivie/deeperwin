
import numpy as np
import matplotlib.pyplot as plt
from utils import load_from_file
import os

#%%
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


#%%
# directory = '/Users/leongerard/Desktop/JAX_DeepErwin/data/register_scale/register_scale_O_6000_False'
#
# full_data = load_from_file(os.path.join(directory, 'results.bz2'))
# eval_E_mean = full_data['metrics']['eval_E_mean']
# eval_E_mean = np.array(eval_E_mean)
#
#
# block_size = [1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 32, 64, 128, 256]
#
# std = []
# for i in block_size:
#     nb_blocks = int(len(eval_E_mean) / i)
#     eval = np.reshape(eval_E_mean[:int(i*np.floor(nb_blocks))], (i, nb_blocks))
#
#     mean_each_block = np.mean(eval, axis=0)
#
#     std_dev = np.var(mean_each_block) * (1/nb_blocks)
#
#     print(std_dev, np.var(mean_each_block), nb_blocks)
#     std.append(std_dev)
#
#
# std = np.array(std)*1e3
#
# fig, axes = plt.subplots(1, 1, sharex=True, figsize=(12, 7))
# axes.plot(block_size, std)
# axes.set_xscale('log')

# auto_corr = getAutoCorrelation(eval_E_mean)
# axes.plot(np.arange(len(auto_corr)), auto_corr)
#
# axes.set_xlim([0, 300])
# axes.set_ylim([-1, 1])

#%%
import jax.numpy as jnp
# directory = ['/Users/leongerard/Desktop/JAX_DeepErwin/data/compare_stddev/O/reg_nc1_2_O',
#              '/Users/leongerard/Desktop/JAX_DeepErwin/data/compare_stddev/O/reg_nc1_O',
#              '/Users/leongerard/Desktop/JAX_DeepErwin/data/compare_stddev/O/deltag_bugfix_O_10000',
#              '/Users/leongerard/Desktop/JAX_DeepErwin/data/compare_stddev/O/adam_O']
# optimizer = ['kfac bad', 'kfac good', 'bfgs', 'adam']

directory = ['/Users/leongerard/Desktop/JAX_DeepErwin/data/compare_scale/register_scale_O_6000_False',
             '/Users/leongerard/Desktop/JAX_DeepErwin/data/compare_scale/deltag_bugfix_O_10000',
'/Users/leongerard/Desktop/JAX_DeepErwin/data/compare_scale/register_scale_O_3000_False',
'/Users/leongerard/Desktop/JAX_DeepErwin/data/compare_scale/dt_damp_O_3000',
'/Users/leongerard/Desktop/JAX_DeepErwin/data/compare_scale/dt_damp_O_6000',
            ]
root_dir = '/Users/leongerard/Desktop/JAX_DeepErwin/data/long_runs'

data = []
names = []
for directory, dirs, fnames in os.walk(root_dir):
    if ('GPU.out' in fnames):

        full_data = load_from_file(os.path.join(directory, 'results.bz2'))
        data.append(full_data)

        temp = directory.split("_")
        names.append(f"Register Scale {temp[-1]}; Molecule {temp[-2]}")


modules = ['bf_fac', 'bf_shift', 'embed', 'jastrow']

for i, d in enumerate(data):
    full_data = d
    parameters = full_data['weights']['trainable']

    name = names[i]

    print("---------------")
    print(f"Opt: {name}")

    for m in modules:
        if m == 'bf_shift':
            print(f"{m} scale {parameters[m]['scale_el']}")
            print(f"{m} norm first layer {jnp.linalg.norm(parameters[m]['w_el'][0][0])}")
        elif m != 'embed':
            print(f"{m} scale {parameters[m]['scale']}")
            print(f"{m} norm first layer {jnp.linalg.norm(parameters[m]['dn'][0][0])}")