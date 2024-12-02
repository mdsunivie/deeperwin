import numpy as np
from deeperwin.checkpoints import load_from_file
import matplotlib.pyplot as plt

def get_jacobi_mat(n_particles):
    jacobi_mat = np.zeros([n_particles, 2])
    jacobi_mat[0::2, 0] = np.arange(n_particles // 2)
    jacobi_mat[1::2, 0] = np.arange(n_particles // 2) + 1
    jacobi_mat[0::2, 1] = np.arange(n_particles // 2)
    jacobi_mat[1::2, 1] = np.arange(n_particles // 2)
    return jacobi_mat

def runing_mean(x):
    return np.cumsum(x) / np.arange(1,len(x)+1)

# fname = '/home/mscherbela/runs/PES/H6/H6_indep_adam_force_eval/0027/results.bz2'
fname = '/home/mscherbela/runs/PES/H6/H6_indep_kfac_force_eval_40k/0009/results.bz2'
Fx_ref, Fa_ref = 0.18999, 0.22009

data = load_from_file(fname)
Ft = np.array(data['metrics']['forces'])
Ft = (Ft - np.mean(np.mean(Ft, axis=0),axis=0))[...,0] @ get_jacobi_mat(6)
Fa = Ft[:,0]
Fx = Ft[:,1]

plt.close("all")
plt.subplot(1,2,1)
plt.plot(Fa, label="Fa", alpha=0.5)

plt.subplot(1,2,2)
plt.plot(runing_mean(Fa), label="runing mean (Fa)")
plt.axhline(Fa_ref, color='navy', label="reference")
plt.legend()
# plt.ylim(Fa_ref * np.array([0.8 , 1.2]))










