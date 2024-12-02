import os
from deeperwin.run_tools.available_gpus import assign_free_GPU_ids
os.environ['CUDA_VISIBLE_DEVICES'] = assign_free_GPU_ids()
os.environ['OMP_NUM_THREADS'] = '10'
os.environ['DISPLAY'] = "localhost:10.0"
import matplotlib
matplotlib.use("TkAgg")

from deeperwin.configuration import Configuration
from deeperwin.model import build_log_psi_squared
from deeperwin.optimization import pretrain_orbitals
from deeperwin.mcmc import MetropolisHastingsMonteCarlo, MCMCState
from deeperwin.loggers import build_dpe_root_logger, BasicLogger
from deeperwin.utils.utils import get_el_ion_distance_matrix
from deeperwin.model import get_baseline_slater_matrices
import matplotlib.pyplot as plt
import numpy as np

def orbital_ref(r, R, fixed_params):
    # mo_up, mo_dn, ci_weights = orbital_func(r, R, Z, params, fixed_params)
    diff_el_ion, dist_el_ion = get_el_ion_distance_matrix(r, R)
    mo_up, mo_dn, ci_weights = get_baseline_slater_matrices(diff_el_ion, dist_el_ion, fixed_params)
    mo_up, mo_dn = [m * ci_weights[:,None,None] for m in [mo_up, mo_dn]]
    return mo_up, mo_dn

config = Configuration.load("/home/scherbelam20/pycharm/sample_configs/config_fermi.yml")
build_dpe_root_logger(config.logging.basic)
loggers = BasicLogger(config.logging.basic)

log_psi_squared, orbital_func, trainable_params, fixed_params = build_log_psi_squared(config.model, config.physical)
mcmc = MetropolisHastingsMonteCarlo(config.mcmc)
mcmc_state = MCMCState.initialize_around_nuclei(config.mcmc.n_walkers_opt, config.physical)

trainable_params_list = [trainable_params]
opt_state = None
for _ in range(2):
    trainable_params, opt_state, mcmc_state = pretrain_orbitals(orbital_func, mcmc, mcmc_state, trainable_params,
                                                                fixed_params, config.pre_training, loggers, opt_state)
    trainable_params_list.append(trainable_params)

def generate_r_plot(x_plot, ind_el):
    n_plot = len(x_plot)
    r_plot = np.random.normal(size=[config.physical.n_electrons, 3]) * 1.0
    r_plot = np.tile(r_plot, [n_plot, 1, 1])
    r_plot[:, ind_el, :] = 0
    r_plot[:, ind_el, 0] = x_plot
    return r_plot

n_up = config.physical.n_up
x_plot = np.linspace(-2, 2, config.mcmc.n_walkers_opt)
r_plot_up = generate_r_plot(x_plot, 0)
r_plot_dn = generate_r_plot(x_plot, n_up)
R = np.array(config.physical.R)
ind_det = 0

plt.close("all")
for i, trainable_params in enumerate(trainable_params_list):
    mo_up, _, _ = orbital_func(r_plot_up, R, config.physical.Z, trainable_params, fixed_params)
    _, mo_dn, _ = orbital_func(r_plot_dn, R, config.physical.Z, trainable_params, fixed_params)
    mo_up_ref, _ = orbital_ref(r_plot_up, R, fixed_params)
    _, mo_dn_ref = orbital_ref(r_plot_dn, R, fixed_params)

    orbitals = np.concatenate([m[:, ind_det, 0, :] for m in [mo_up, mo_dn]], axis=-1)
    orbitals_ref = np.concatenate([m[:, ind_det, 0, :] for m in [mo_up_ref, mo_dn_ref]], axis=-1)
    n_orbitals = orbitals.shape[-1]

    plt.figure()
    plt.suptitle(f"{i * config.pre_training.n_epochs} epochs")
    for n in range(n_orbitals):
        plt.subplot(2,2,n+1)
        plt.plot(x_plot, orbitals[:,n], color=f'C{n}', label=str(n))
        plt.plot(x_plot, orbitals_ref[:,n], color=f'C{n}', ls='--')
        plt.axhline(0, color='k', alpha=0.5)
    plt.pause(0.01)

plt.figure()
dist_from_ion = np.linalg.norm(mcmc_state.r, axis=-1).flatten()
plt.subplot(1,2,1)
for n in range(mcmc_state.r.shape[-2]):
    plt.scatter(mcmc_state.r[:,n,0], mcmc_state.r[:,n,1], alpha=0.3)
plt.subplot(1,2,2)
plt.hist(dist_from_ion, bins=20)
plt.pause(0.01)



