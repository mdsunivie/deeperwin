import jax.nn
import numpy as np

from deeperwin.checkpoints import load_run
from haiku.data_structures import tree_size
from deeperwin.model.wavefunction import build_log_psi_squared
from deeperwin.model.definitions import NParticles

# data = load_run("/home/mscherbela/runs/debug/hp_1det_bdiag_a40_el_ion_N2_bond_breaking_False/chkpt000000.zip")
# data = load_run("/home/mscherbela/runs/debug/hp_1det_bdiag_a40_el_ion_N2_bond_breaking_False/chkpt019000.zip")
data = load_run("/home/mscherbela/runs/debug/hp_1det_bdiag_a40_f_width_Ethene_128/chkpt019000.zip")
# data = load_run("/home/mscherbela/runs/debug/hp_1det_bdiag_ref_Ethene_rep2/chkpt049000.zip")
#
#%%
for k, v in data.params.items():
    print(f"{k:<90}: {tree_size(v)//1000}k")

max_n_ions = len(data.config.physical.Z)
n_up = int(data.config.physical.n_up)
n_dn = int(data.config.physical.n_dn)
max_n_particles = NParticles(n_ion=max_n_ions, n_up=n_up, n_dn=n_dn)
log_psi_sqr, _, _, _ = build_log_psi_squared(data.config.model, data.config.physical, max_n_particles, False, data.fixed_params, 0)
psi = log_psi_sqr(data.params, n_up, n_dn, *data.mcmc_state.build_batch(data.fixed_params))

#%%
alpha_up = data.params['wf/orbitals/envelope_orbitals']['alpha_up']
alpha_dn = data.params['wf/orbitals/envelope_orbitals']['alpha_dn']

np.set_printoptions(precision=2, suppress=True, linewidth=200)
print(jax.nn.softplus(alpha_up))
print(jax.nn.softplus(alpha_dn))
