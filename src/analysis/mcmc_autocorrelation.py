import numpy as np
import matplotlib.pyplot as plt
from deeperwin.checkpoints import load_run


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

run_dir = "/home/mscherbela/runs/ablation/P_512/ablation_04_fermi_iso_fulldet_hp_emb_P_512_rep1"
data, config = load_run(run_dir, ignore_extra_settings=True)
E = np.array(data['metrics']['eval_E_mean'], float)
E = E - np.mean(E)
del data
#%%
plt.close("all")

# ##### Raw timeseries ##########
# plt.figure(dpi=100, figsize=(14,8))
# plt.plot(E*1e3)
# plt.grid(alpha=0.3)
# plt.ylabel("E - E_mean / mHa")
#
# ##### Sensitivity to outliers ##########
# E_abs_sorted = E[np.argsort(np.abs(E))[::-1]]
# n_drop_values = np.arange(1000)
# E_clipped = np.zeros_like(n_drop_values, float)
# for i,n in enumerate(n_drop_values):
#     E_clipped[i] = np.mean(E_abs_sorted[n:]) * 1e3
#
# plt.figure(dpi=100, figsize=(14,8))
# plt.plot(n_drop_values, E_clipped)
# plt.grid(alpha=0.3)
# plt.xlabel("nr of excluded outliers")
# plt.ylabel("Change in E_mean / mHa")

##### Autocorrelation function ##########

def get_autocorrelation_time(autocorr):
    n_samples = min(np.where(autocorr < 0)[0])
    if n_samples < 2:
        return 0
    autocorr = autocorr[:n_samples]
    slope, _ = np.polyfit(np.arange(n_samples), np.log(autocorr), 1)
    return -1/slope

plt.figure(dpi=100, figsize=(14,8))
n_repetitions = 20
for n in range(n_repetitions):
    ind_start = int(len(E) * n / n_repetitions)
    ind_end = int(len(E) * (n+1) / n_repetitions)
    autocorr = getAutoCorrelation(E[ind_start:ind_end])
    plt.semilogy(autocorr[:10], color='gray', alpha=0.3)
    tau = get_autocorrelation_time(autocorr)
    print(tau)

plt.grid(alpha=0.3)
plt.ylabel("E - E_mean / mHa")

