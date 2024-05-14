#%%
import numpy as np
import matplotlib.pyplot as plt


def get_r(r_min, r_max, n_grid):
    dr = (r_max - r_min) / n_grid
    r = (0.5 + np.arange(n_grid)) * dr + r_min
    return r


# run = "SpinCorrelations_geom0115_H20_R3.6_k0.000"
# R = 3.6
# run = "SpinCorrelations_geom0100_H20_R2.0_k0.000"
# R = 2.0
run = "SpinCorrelations_geom0080_H20_R1.2_k0.000"
R = 1.2

n_atoms = 20
L = n_atoms * R

input_dir = "/home/mscherbela/runs/paper5_solids/HChains/spin_correlations"
fname_1el = f"{input_dir}/{run}/rho_1el_200000.npz"
fname_2el = f"{input_dir}/{run}/rho_2el_rel_200000.npz"

data_1el = np.load(fname_1el)
data_2el = np.load(fname_2el)
r_min, r_max, n_grid, rho_1el = data_1el["r_min"], data_1el["r_max"], data_1el["n_grid"], data_1el["rho"]
_, _, _, rho_2el = data_2el["r_min"], data_2el["r_max"], data_2el["n_grid"], data_2el["rho"]
r1 = get_r(r_min[0], r_max[0], n_grid[0])
rho_1el = rho_1el.squeeze(axis=(-1, -2))
rho_2el = rho_2el.squeeze(axis=(-1, -2))
rho_2el = rho_2el.astype(float)
rho_2el = np.stack([rho_2el[[0, 1], [0, 1]].sum(axis=0), rho_2el[[0, 1], [1, 0]].sum(axis=0)])

sigma_1el = np.sqrt(rho_1el)
sigma_2el = np.sqrt(rho_2el)
sigma_plot = 5

# Normalize
dr = L / len(r1)
normalization_1el = (n_atoms / 2) / np.sum(rho_1el, axis=-1, keepdims=True) / dr
normalization_same = (n_atoms / 2 - 1) / np.sum(rho_2el[0], axis=-1) / dr
normalization_diff = (n_atoms / 2) / np.sum(rho_2el[1], axis=-1) / dr
normalization_2el = np.array([normalization_same, normalization_diff])[:, None]
rho_1el = rho_1el * normalization_1el
rho_2el = rho_2el * normalization_2el
sigma_1el = sigma_1el * normalization_1el
sigma_2el = sigma_2el * normalization_2el


r_center = (r_min[0] + r_max[0]) / 2
r2 = r1.copy()
map_back = r2 > (n_atoms * R / 2)
r2[map_back] -= n_atoms * R
rho_2el = np.roll(rho_2el, np.sum(map_back), axis=-1)
r2 = np.roll(r2, np.sum(map_back))

# plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(10, 6))
ax_1el, ax_2el = axes
ax_1el.plot(r1 / R, rho_1el[0], label="Spin up")
ax_1el.plot(r1 / R, rho_1el[1], label="Spin dn")
ax_1el.fill_between(r1 / R, rho_1el[0] - sigma_plot*sigma_1el[0], rho_1el[0] + sigma_plot*sigma_1el[0], color="C0", alpha=0.2)
ax_1el.fill_between(r1 / R, rho_1el[1] - sigma_plot*sigma_1el[1], rho_1el[1] + sigma_plot*sigma_1el[1], color="C1", alpha=0.2)

ax_2el.plot(r2 / R, rho_2el[0], color="C2", label="Parallel spins")
ax_2el.plot(r2 / R, rho_2el[1], color="C5", label="Antiparallel spins")
ax_2el.fill_between(r2 / R, rho_2el[0] - sigma_plot*sigma_2el[0], rho_2el[0] + sigma_plot*sigma_2el[0], color="C2", alpha=0.2)
ax_2el.fill_between(r2 / R, rho_2el[1] - sigma_plot*sigma_2el[1], rho_2el[1] + sigma_plot*sigma_2el[1], color="C5", alpha=0.2)


for i in range(n_atoms+1):
    ax_1el.axvline(i, color='gray', alpha=0.1)
    ax_2el.axvline(i-n_atoms//2, color='gray', alpha=0.1)

for ax in axes:
    ax.legend()
    ax.set_ylim([0, None])

ax_1el.set_xlabel("r / R")
ax_2el.set_xlabel("$\\Delta$ r / R")
ax_1el.set_title("Spin density")
ax_2el.set_title("Pair density")
fig.suptitle(run)
fig.tight_layout()

plt.figure()
plt.plot(np.abs(np.fft.fft(rho_2el, axis=-1).T), marker='o')
# fig.savefig(f"/home/mscherbela/ucloud/results/spin_correlations/{run}.png")


