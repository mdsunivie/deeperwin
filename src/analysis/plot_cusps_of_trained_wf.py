import jax.config as jax_config
import jax.numpy as jnp
import numpy as np
jax_config.update("jax_disable_jit", True)
from utils import load_from_file
from model import build_log_psi_squared, build_log_psi_squared_baseline_model
from hamiltonian import get_local_energy
from configuration import Configuration, build_nested_dict
import matplotlib.pyplot as plt
import copy


data = load_from_file("/home/mscherbela/runs/cusps/no_distance_feat/cusp03_smoothdist_bfgs_B/results.bz2")
config = data['config']
for k in ['n_params', 'code_version', 'tags']:
    config.pop(k)
config = Configuration.parse_obj(build_nested_dict(config))

print("Building log_psi_sqr, incl. baseline calc...")
_, log_psi_squared, _, _ = build_log_psi_squared(config.model, config.physical)
# _, log_psi_squared_baseline, _, _ = build_log_psi_squared_baseline_model(config.model.baseline, config.physical)

params_trainable, params_fixed = data['weights']['trainable'], data['weights']['fixed']
params_trainable_minimal = copy.deepcopy(params_trainable)
params_trainable_minimal['bf_fac']['scale'] = jnp.array([-100.0])
params_trainable_minimal['bf_shift']['scale_el'] = jnp.array([-100.0])
params_trainable_minimal['jastrow']['scale'] = jnp.array([-100.0])
empty_trainable_params = {}

#%%
N_samples = 1000
r = np.zeros([N_samples, config.physical.n_electrons, 3]) + data['weights']['mcmc'].r[0]
r_plot = np.linspace(-1, 1, N_samples)
r[:,0, 1:] = 0
r[:,0,0] = r_plot
r = jnp.array(r)

R, Z = jnp.array(config.physical.R), jnp.array(config.physical.Z)
print("Calculating log_psi_sqr...")
# log_sqr_baseline = log_psi_squared_baseline(r, R, Z, empty_trainable_params, params_fixed)
log_sqr_dpe_zero = log_psi_squared(r, R, Z, params_trainable_minimal, params_fixed)
log_sqr = log_psi_squared(r, R, Z, params_trainable, params_fixed)

print("Calculating Eloc...")
# E_loc_baseline = get_local_energy(log_psi_squared_baseline, r, R, Z, empty_trainable_params, params_fixed)
E_loc_dpe_zero = get_local_energy(log_psi_squared, r, R, Z, params_trainable_minimal, params_fixed)
E_loc = get_local_energy(log_psi_squared, r, R, Z, params_trainable, params_fixed)

#%%
plt.close("all")
plt.figure(figsize=(14,7))
plt.subplot(1,2,1)
# plt.plot(r_plot, log_sqr_baseline, label='Baseline')
plt.plot(r_plot, log_sqr_dpe_zero, label='Baseline: No NNs', ls='--')
log_sqr_shift = np.nanmean(log_sqr - log_sqr_dpe_zero)
plt.plot(r_plot, log_sqr - log_sqr_shift, label='DeepErwin (aligned with Baseline)')
plt.grid()
plt.xlabel("x0")
plt.title(r"log $\psi^2$")
plt.legend()

plt.subplot(1,2,2)
# plt.plot(r_plot, E_loc_baseline, label='Baseline')
plt.plot(r_plot, E_loc_dpe_zero, label='Baseline: No NNs')
plt.axhline(data['metrics']['E_casscf'], label="E casscf", ls='--', color='C0', alpha=0.3)
plt.plot(r_plot, E_loc, label='DeepErwin')
plt.axhline(data['metrics']['E_mean'], label="Emean DeepErwin", ls='--', color='C1', alpha=0.3)
plt.grid()
plt.xlabel("x0")
plt.title(r"$E_{loc}$")
plt.legend()
# plt.suptitle("Wavefunction for LiH, close to nucleus")
# plt.savefig("/home/mscherbela/ucloud/results/Cusps_LiH.png")





