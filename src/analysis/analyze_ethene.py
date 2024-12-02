from deeperwin.checkpoints import load_from_file
import pathlib
import os
import numpy as np
import pickle
import yaml

#%%
directory = '/Users/leongerard/Desktop/JAX_DeepErwin/data/dgx_runs/ethene_bfgs/'
fnames = [f for f in pathlib.Path(directory).rglob(os.path.join('results.bz2'))]
fnames = sorted(fnames)

folder = ['chkpt000256', 'chkpt000512', 'chkpt001024', 'chkpt002048']

results = {'chkpt000256': [],
           'chkpt000512': [],
           'chkpt001024': [],
           'chkpt002048': [],
           'chkpt004096': []
}
#%%

geom = []
e_mean = []
e_mean_casscf = []

for i, f in enumerate(fnames):
    f_str_split = str(f).split("/")
    f_str_folder = f_str_split[-2]

    if f_str_folder in folder:
        df = load_from_file(f)
        E_eval = df['metrics']['E_mean']
        f_str_config = "/".join(f_str_split[:-1])

        with open(f_str_config + "/full_config.yml", 'r') as stream:
            config = yaml.safe_load(stream)
        pos = config['physical']['R']
        key = str(round(pos[1][0], 1)) + "_" + str(round(pos[2][0] - pos[1][0], 1))

        error = (E_eval - config['physical']['E_ref']) * 1000
        results[f_str_folder].append(error)
    else:
        df = load_from_file(f)
        E_eval = df['metrics']['E_mean']
        f_str_config = "/".join(f_str_split[:-1])

        with open(f_str_config + "/full_config.yml", 'r') as stream:
            config = yaml.safe_load(stream)
        pos = config['physical']['R']
        key = str(round(pos[1][0], 1)) + "_" + str(round(pos[2][0] - pos[1][0], 1))
        geom.append(key)
        error = (E_eval - config['physical']['E_ref'])*1000
        results['chkpt004096'].append(error)
        e_mean.append((config['physical']['comment'], E_eval))
        e_mean_casscf.append(df['metrics']['E_casscf'])




print(np.mean(results['chkpt004096']), np.std(results['chkpt004096']))
print(np.mean(results['chkpt002048']), np.std(results['chkpt002048']))
print(np.mean(results['chkpt001024']), np.std(results['chkpt001024']))
print(np.mean(results['chkpt000512']), np.std(results['chkpt000512']))
#%%
import matplotlib.pyplot as plt

plt.close("all")

epochs = [256, 512, 1024, 2048, 4096]
ethene_kfac = [np.mean(results[key]) for key in results.keys()]
plt.semilogx(epochs, ethene_kfac, label=f'JAX KFAC Shared: Ethene', marker='s')

epochs = [512, 1024, 2048, 4096, 8192]
ethene_kfac_indep = [21.147244, 15.789501, 9.516232, 5.584550, 2.499732]
plt.semilogx(epochs, ethene_kfac_indep, label=f'JAX KFAC Independent: Ethene', marker='s', color='C2')

epochs_indep = [512, 1024, 2048, 4096, 8192]
ethene_adam_indep = [49.189077089843636, 35.776601503906136, 23.53371209960926, 16.989980410156136, 11.39076776367176]
plt.semilogx(epochs_indep, ethene_adam_indep, label=f'JAX Adam Independent: Ethene', marker='^', color='C1')

# ethene_kfac_indep = [17.29061601562165, 12.362027148434152, 7.0214509765591515, 4.0650605956997765, 1.1124849121060265]
# plt.semilogx(epochs_indep, ethene_kfac_indep, label=f'JAX KFAC Independent: Ethene', marker='^', color='C2')
#
# ethene_bfgs_indep = [31.339445136715938, 20.70597650878625, 14.402189277340938, 8.27769281738, 3.9537334667940627]
# plt.semilogx(epochs_indep, ethene_bfgs_indep, label=f'JAX BFGS Independent: Ethene', marker='^', color='C3')

plt.axhline(1.6, label="Chemical accuracy", color='gray')
plt.legend()
plt.xlabel("Training epochs")
plt.ylabel("Error / mHa")
plt.grid(alpha=0.7, ls=':')
x_ticks = [2**n for n in range(9, 14)]
plt.gca().set_xticks(x_ticks)
plt.gca().set_xticklabels(map(str, x_ticks))
plt.minorticks_off()
plt.title("Ethene: 10 geometries")
#%%
import matplotlib.pyplot as plt

plt.close()
e_mean = sorted(e_mean)
twist = np.arange(0, 10*len(e_mean), 10)
mean = [el[1] for el in e_mean]

plt.plot(twist, mean, label=f'JAX KFAC Shared: Ethene', marker='^', color='C1')

mrci = [-78.57744597, -78.57548243, -78.5695788, -78.55986352, -78.54643791, -78.52954886, -78.50967954, -78.4879336, -78.46748786, -78.45763211]
plt.plot(twist, mrci, label=f'MRCI-D-F12', marker='o', color='C2')

e_cas = sorted(e_mean_casscf)
e_cas_shifted = np.array(e_cas) - (e_cas[0] - mrci[0])
plt.plot(twist, e_cas_shifted, label=f'CASSCF', marker='^', color='C4')

# mean_indep = [-78.577049, -78.573273, -78.567886, -78.557846, -78
# .544174, -78.525009, -78.506180, -78.484344, -78.463600, -78.453957]
# plt.plot(twist, mean_indep, label=f'JAX KFAC Indep: Ethene', marker='^', color='C0')
# plt.title("Ethene PES")

#
# ccsd_shifted = [-78.57744597, -78.57548402, -78.56961579, -78.55989991, -78.54645174, -78.52947537, -78.50933895, -78.48677163, -78.46353767, -78.37198719]
# plt.plot(twist, ccsd_shifted, label=f'CCSD(T)-F12, shifted to match MRCI-D-F12 at twist=0', marker='^', color='C3')
#

plt.grid(alpha=0.5)
# x_ticks = [n for n in range(0, 100, 10)]
# plt.gca().set_xticks(x_ticks)
# plt.gca().set_xticklabels(map(str, x_ticks))
plt.xlabel("Twist")
plt.ylabel("Energy/ Ha")
plt.legend()