#%%
from deeperwin.checkpoints import load_run
import matplotlib.pyplot as plt
import numpy as np

def align_curves(y, y_ref):
    best_residual = np.sum((y - y_ref)**2)
    best_y_shifted = y
    for shift in range(1, len(y)):
        y_shifted = np.roll(y, shift)
        residual = np.sum((y_shifted - y_ref)**2)
        if residual < best_residual:
            best_residual = residual
            best_y_shifted = y_shifted
    return best_y_shifted

input_dir = "/home/mscherbela/tmp/8dets/"
fname = input_dir + "0082" + "/chkpt050000.zip"
run = load_run(fname, parse_csv=False)
mo_coeff = run.fixed_params["baseline_orbitals"].mo_coeff[0]
n_el = 20
n_atoms = n_el
n_occ = n_el // 2
n_basis_per_atom = 2

mo_coeff = mo_coeff.reshape([n_atoms, n_basis_per_atom, -1])[:, :, :n_occ]
norm_per_atom = np.linalg.norm(mo_coeff, axis=1)
norm_per_atom_aligned = np.stack([align_curves(n, norm_per_atom[:, 0]) for n in norm_per_atom.T], axis=1)
ind_max = np.argmax(np.mean(norm_per_atom_aligned, axis=1), axis=0)
norm_per_atom_aligned = np.roll(norm_per_atom_aligned, -ind_max + n_atoms // 2, axis=0)

# ind_max = np.argmax(norm_per_atom, axis=0)
# norm_per_atom_aligned = np.stack([np.roll(n, -i+10) for i, n in zip(ind_max, norm_per_atom.T)])


#%%
# plt.close("all")
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.plot(norm_per_atom_aligned)
