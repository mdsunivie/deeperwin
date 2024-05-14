# %%
import numpy as np
from deeperwin.configuration import Configuration
import glob
import matplotlib.pyplot as plt
from deeperwin.orbitals import _get_electron_configuration, _get_effective_charge, _initialize_walkers_around_atom
from deeperwin.utils.utils import generate_exp_distributed
import jax
import seaborn as sns
from wandb import Api
sns.set_theme(style="whitegrid")


def load_radial_density(directory):
    config = Configuration.load(directory + "/config.yml")
    n_inter_steps = config.optimization.intermediate_eval.mcmc.n_inter_steps
    n_el, n_up, R, Z = config.physical.get_basic_params()
    mcmc_files = glob.glob(directory + "/mcmc_state_*.npz")
    mcmc_files = sorted(mcmc_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    n_epochs = len(mcmc_files)
    n_ions = len(Z)

    n_bins = 30
    # radial_bins = np.concatenate([[0.0], np.geomspace(1e-3, 5.0, n_bins)])
    radial_bins = np.linspace(0, 1.5, n_bins + 1)
    radial_histogram = np.zeros([n_epochs, n_ions, len(radial_bins) - 1])

    for ind_epoch, fname in enumerate(mcmc_files):
        if ind_epoch % 20 == 0:
            print(ind_epoch)
        mcmc_r = np.load(fname)["r"]
        dist = np.linalg.norm(mcmc_r[:, :, None, :] - R, axis=-1)
        dist = dist.reshape([-1, n_ions])
        for ind_ion in range(n_ions):
            radial_histogram[ind_epoch, ind_ion] = np.histogram(dist[:, ind_ion], bins=radial_bins)[0]
    return radial_histogram, radial_bins, n_el, R, Z, n_inter_steps


radial_hist_exponential, _, _, _, _, _ = load_radial_density(
    "/home/mscherbela/runs/initialization/el_init_Glycine_exponential"
)
radial_hist_gaussian, radial_bins, n_el, R, Z, n_inter_steps = load_radial_density(
    "/home/mscherbela/runs/initialization/el_init_Glycine_gaussian"
)
r = (radial_bins[:-1] + radial_bins[1:]) * 0.5
volume = radial_bins[1:] ** 3 - radial_bins[:-1] ** 3


wandb_runs = ["schroedinger_univie/debug_burnin/runs/7vz5p0i5", "schroedinger_univie/debug_burnin/runs/nun5gg3v"]
api = Api()
energies = []
for run in wandb_runs:
    run = api.run(run)
    history = run.scan_history(keys=["eval_E_mean"])
    energies.append(np.array([x["eval_E_mean"] for x in history]))

# %%
def get_sampled_density_gaussian(n_samples):
    r = np.random.normal(size=[n_samples, 3])
    r = np.linalg.norm(r, axis=-1)
    return np.histogram(r, bins=radial_bins)[0] / volume


def get_sampled_density_exponential(n_samples, Z):
    r = _initialize_walkers_around_atom(jax.random.PRNGKey(0), np.zeros(3), Z, n_samples, Z, Z // 2)
    r = np.concatenate(r, axis=1)
    r = np.linalg.norm(r, axis=-1)
    r = r.reshape([-1])
    return np.histogram(r, bins=radial_bins)[0] / volume


Z_values = [6, 7, 8]
t_values = [0, 5, 10, 20, 30]
n_averaging = 5
z = 8
ind_Z = Z == z

plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

normalization = np.sum(radial_hist_exponential[-20:, ind_Z].mean(axis=-1).mean(axis=-1))
density_final = radial_hist_exponential[-20:, ind_Z].mean(axis=0).mean(axis=0) / volume

ax_density = axes[0]
density_exp_init = get_sampled_density_exponential(200_000, z)
density_exp_init *= normalization / (density_exp_init * volume).sum()
density_gaussian_init = get_sampled_density_gaussian(10_000_000)
density_gaussian_init *= normalization / (density_gaussian_init * volume).sum()

ax_density.plot(r, density_gaussian_init, label=f"Gaussian init.", color="C0")
ax_density.plot(r, density_exp_init, label=f"Exponential init.", color="C1")
ax_density.plot(r, density_final, label="Ground truth $\\psi^2$", color="k")
ax_density.legend()
ax_density.set_yscale("log")
ax_density.set_xlabel("distance from nucleus / bohr")
ax_density.set_ylabel("Electron density, a.u.")
ax_density.set_ylim([300, 1e6])


energy_ax = axes[1]
E_0 = np.mean(energies[1][-50:])
for E, label in zip(energies, ["Gaussian init.", "Exponential init."]):
    energy_ax.plot(np.arange(len(E)) * n_inter_steps / 1000, E, label=label)
energy_ax.axhline(E_0, color='k', label="Ground truth")
energy_ax.legend()
energy_ax.set_xlabel("MCMC steps / k")
energy_ax.set_ylabel("Energy / Ha")
energy_ax.set_xlim([0, 11])
energy_ax.set_ylim([-285, -280])

for ax, letter in zip(axes, "ab"):
    ax.text(x=0, y= 1.01, s=letter, transform=ax.transAxes, fontsize=14, fontweight="bold", ha="left", va="bottom")
    
fig.tight_layout()
fname = f"/home/mscherbela/ucloud/results/04_paper_better_universal_wf/figures/fig2_electron_initialization_Glycine_Z{z}.png"
fig.savefig(fname, dpi=300, bbox_inches="tight")
fig.savefig(fname.replace(".png", ".pdf"), bbox_inches="tight")

# %%
