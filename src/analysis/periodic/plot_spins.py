#%%
from deeperwin.checkpoints import load_run
import numpy as np
import matplotlib.pyplot as plt
import pathlib

def get_geom_index(R, N, n_k=5):
    R_values = [1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 3.0, 3.6]
    N_values = [12, 16, 20]
    return N_values.index(N) * n_k * len(R_values) + R_values.index(R) * n_k


def get_spin_order(fname):
    data = load_run(fname, parse_config=False, parse_csv=False)
    x = data.mcmc_state.r[:, :, 0]
    n_el = x.shape[-1]
    ordered_pos = np.argsort(x, axis=1)
    spin = (ordered_pos < n_el // 2) * 2 - 1
    return spin

def get_correlation(spin):
    return np.mean(spin * np.roll(spin, 1, axis=1))


directories = ["Reuse40"]
R_values = [1.2, 1.4, 1.6]
N = 12
n_k = 11
epochs = [0, 2, 5]

spins = []
for ep in epochs:
    for d in directories:
        for R in R_values:
            ind_geom = get_geom_index(R, N, n_k)
            fname = pathlib.Path("/home/mscherbela/tmp/spin_checkpoints") / d / f"{ind_geom:04d}" / f"chkpt{ep*1000:06d}.zip"
            spins.append(get_spin_order(fname))
spin_correlations = [(get_correlation(s[:1024]), get_correlation(s[1024:])) for s in spins]
ind_plot = np.concatenate([np.arange(20), np.arange(1024, 1024+20)])

plt.close("all")
fig, axes = plt.subplots(len(epochs), len(directories) * len(R_values), figsize=(10, 5))
axes = np.atleast_2d(axes)

ind = 0
for ep in epochs:
    for d in directories:
        for R in R_values:
            ax = axes.flatten()[ind]
            ax.imshow(spins[ind][ind_plot])
            corr = spin_correlations[ind]
            ax.set_title(f"{d},R={R:.1f},N={N}\nCorr: {corr[0]:+.2f}, {corr[1]:+.2f} = {np.mean(corr):+.2f}")
            ind += 1

# fig.suptitle(f"R = {R}, N = {N}", fontsize=16)
# for ax, ep in zip(axes[:, 0], epochs):
#     ax.set_ylabel(f"Epoch {ep}k", fontsize=16)
fig.tight_layout()

