
import numpy as np
import matplotlib.pyplot as plt
import os

from deeperwin.checkpoints import load_from_file

base_path = "/Users/leongerard/Desktop/data_schroedinger/mcmc_proposal/correlation_time/"
jobs_Ne = [("local one el", "mcmc_Ne_local_one_el"), ("local rmax 1", "mcmc_Ne_local_r_max_local_1"),
           ("local rmax 3", "mcmc_Ne_local_r_max_local_3"),
           ("normal", "mcmc_Ne_normal"), ("normal one el", "mcmc_Ne_normal_one_el")]

e_mean_Ne = {}
for j in jobs_Ne:
    path = os.path.join(base_path, j[1])
    fname = "results.bz2"
    data = load_from_file(os.path.join(path, fname))
    e_mean_Ne[j[1]] = data['metrics']['eval_E_mean']

    del data
print("Finished loading data!")


jobs_P = [("local one el", "mcmc_P_local_one_el"), ("local rmax 1", "mcmc_P_local_r_max_local_1"),
           ("local rmax 2", "mcmc_P_local_r_max_local_2"), ("local rmax 3", "mcmc_P_local_r_max_local_3"),
           ("normal", "mcmc_P_normal"), ("normal one el", "mcmc_P_normal_one_el")]

e_mean_P = {}
for j in jobs_P:
    path = os.path.join(base_path, j[1])
    fname = "results.bz2"
    data = load_from_file(os.path.join(path, fname))
    e_mean_P[j[1]] = data['metrics']['eval_E_mean']

    del data

#%%

### Autocorrelation data

def var_per_block(eval_mean, length_block):
    eval_mean = eval_mean[: (len(eval_mean)//length_block * length_block)]
    eval_mean_block = np.reshape(eval_mean, newshape= (len(eval_mean)//length_block, length_block))
    return np.var(eval_mean_block.mean(axis=-1))

# print(f"Nb blocks: {[(len(eval_mean)//l_b) for l_b in length_blocks]}")
# autocorr = [var_per_block(eval_mean[:], l_b)/(len(eval_mean)//l_b) for l_b in length_blocks]

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

fig, ax = plt.subplots(2, 2, figsize=(12,12))
length_blocks = [2**i for i in range(10)]


jobs_Ne = [("local one el", "mcmc_Ne_local_one_el"), ("local rmax 1", "mcmc_Ne_local_r_max_local_1"),
           ("local rmax 3", "mcmc_Ne_local_r_max_local_3"),
           ("normal", "mcmc_Ne_normal"), ("normal one el", "mcmc_Ne_normal_one_el")]


for j, (label, key) in enumerate(jobs_Ne):

    e_mean = e_mean_Ne[key]
    ax[0][0].semilogy(np.abs(getAutoCorrelation(e_mean))[:100], color=f"C{j}", label=label)
    ax[0][0].grid(alpha=0.5)
    ax[0][0].set_title("Autocorrelation")
    ax[0][0].set_xlabel("Steps")
    ax[0][0].set_ylabel("Autocorrelation")

    autocorr = [var_per_block(e_mean[:], l_b)/(len(e_mean)//l_b) for l_b in length_blocks]
    ax[1][0].semilogx(length_blocks, autocorr, color=f"C{j}")
    ax[1][0].grid(alpha=0.5)
    ax[1][0].set_xticks(length_blocks)
    ax[1][0].set_xticklabels(length_blocks)
    ax[1][0].set_xlabel("Block length")
    ax[1][0].set_ylabel("Variance")
    ax[1][0].set_title("Blocking of Energy")


ax[0][0].legend()
fig.suptitle("Autocorrelation")


for j, (label, key) in enumerate(jobs_P):

    e_mean = e_mean_P[key]
    ax[0][1].semilogy(np.abs(getAutoCorrelation(e_mean))[:100], color=f"C{j}", label=label)
    ax[0][1].grid(alpha=0.5)
    ax[0][1].set_title("Autocorrelation")
    ax[0][1].set_xlabel("Steps")
    ax[0][1].set_ylabel("Autocorrelation")

    autocorr = [var_per_block(e_mean[:], l_b)/(len(e_mean)//l_b) for l_b in length_blocks]
    ax[1][1].semilogx(length_blocks, autocorr, color=f"C{j}")
    ax[1][1].grid(alpha=0.5)
    ax[1][1].set_xticks(length_blocks)
    ax[1][1].set_xticklabels(length_blocks)
    ax[1][1].set_xlabel("Block length")
    ax[1][1].set_ylabel("Variance")
    ax[1][1].set_title("Blocking of Energy")


ax[0][1].legend()


fig.suptitle("Autocorrelation")