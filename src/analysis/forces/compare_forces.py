import matplotlib.pyplot as plt
import numpy as np
from utils import load_from_file, morse_potential, fit_morse_potential

# fnames = ["/home/mscherbela/runs/forces/LiH_highres/LiH_eval_highres_2.8_False_False/results.bz2",
# "/home/mscherbela/runs/forces/LiH_highres/LiH_eval_highres_2.8_True_False/results.bz2"]
# labels = ['Naive + Cutoff', 'Antithetic + Cutoff']

# fnames = ["/home/mscherbela/runs/forces/LiH_highres/LiH_eval_highres_3.2_False_True/results.bz2",
# "/home/mscherbela/runs/forces/LiH_highres/LiH_eval_highres_3.2_True_True/results.bz2"]
# labels = ['Naive + poly', 'Antithetic + poly']

# fnames = ["/home/mscherbela/runs/forces/LiH_noshift/LiH_noshift_2.6/results.bz2",
# "/home/mscherbela/runs/forces/LiH_noshift/LiH_noshift_3.6/results.bz2"]
# labels = ['Noshift 2.6', 'Noshift 3.6']

fnames = ["/home/mscherbela/runs/forces/newshift/newshift_ref_noshift_False_C/results.bz2",
"/home/mscherbela/runs/forces/atoms/forces_atoms_C/results.bz2"]
labels = ['Noshift C; Run 1', 'Noshift C; Run 2']


FxLi = []
for f in fnames:
    data = load_from_file(f)
    force_history = np.array(data['metrics']['forces'])
    # force_history = np.clip(force_history, -10, 10)
    FxLi.append(force_history[:, 0, 0]*1000)

# plt.close("all")
plt.figure(figsize=(14,7))
N_samples = len(FxLi[0])
for i in range(2):
    plt.plot(FxLi[i], label=labels[i], alpha=0.5)
    plt.plot(np.cumsum(FxLi[i])/np.arange(1,N_samples+1), color=f'C{i}')
# plt.ylim([-100, 100])
plt.grid()
plt.legend()



