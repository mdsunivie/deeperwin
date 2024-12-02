

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("/Users/leongerard/Desktop/data_schroedinger/excel_analysis/fermi_dperwin.csv")

#%%

dpe_v11 = df['reg_faster_dpe4_O_4_256_32_32_32_rep2 - error_intermed_eval'][1:]
fermi = df['fermi_bench_10_2_O - error_intermed_eval'][1:]
#little_beast = df['deeperwin_bench_11_2_Ethene_fermi - error_intermed_eval'][:-1]

epochs = [10000, 20000, 40000, 60000, 80000][1:]
epoch_names = ["10k", "20k", "40k", "60k"][1:]
fig, ax = plt.subplots(1, 1, figsize=(6,4))

name = "O"
ax.set_title(f"Molecule: {name}")
ax.plot(epochs[:len(fermi)], fermi, label="FermiNet", color="grey")
ax.plot(epochs[:len(dpe_v11)], dpe_v11, label="New DeepErwin", color="darkblue")

#ax.plot(epochs[:len(little_beast)], little_beast, label="DeepErwin", color="dodgerblue")

ax.set_xticks(epochs[:-1])
ax.set_xticklabels(epoch_names)
ax.set_xlabel("Epochs")

ax.grid(alpha=0.5)
ax.hlines(0, 20000, 60000, linestyles="dashed", color="grey", alpha=0.5, label="Reference")
ax.legend()
ax.set_ylabel("Error (mHa)")

#%%
# df = pd.read_csv("/Users/leongerard/Desktop/data.csv")
# fermi = df['fermi_bench_O - opt_E_std']
# deeperwin = df['Big_Emb_1 - opt_E_std']
# x = np.arange(0, len(deeperwin), 20)
# fermi = [x for i, x in enumerate(fermi) if i%20==0 ]
# deeperwin = [x  for i, x in enumerate(deeperwin) if i%20==0]
#
#
# plt.semilogy(x, fermi, label="FermiNet", color="darkblue")
# plt.semilogy(x, deeperwin, label="DeepErwin", color="dodgerblue")
# plt.grid(alpha=0.5)
# plt.legend()
# plt.title("Std. Dev. for Oxygen")
# Data

# Ethene
error_ethene = dict(
    fermi = [32, 20, 11, 5],
    fermi_mix = [12, 5, 1.8]
)

# Oxygen
error_oxygen = dict(
    fermi = (75.067 - np.array([75.061, 75.058, 75.066, 75.0663]))*1000,
    fermi_mix = [2.5, -0.2, 0.4]
)

# H10
error_h10 = dict(
    fermi = [2.5, 2.4, 2.3, 2.3],
    fermi_mix = [2.46, 2.3, 2.2, 2.2]
)

data = [("Ethene", error_ethene), ("H10", error_h10), ("Oxygen",error_oxygen)]


fig, ax = plt.subplots(1, len(data), figsize=(4,4))
epochs = [10000, 20000, 40000, 60000, 80000]
epoch_names = ["10k", "20k", "40k", "60k"]
for i, (name, mol) in enumerate(data):

    if len(data) == 1:
        ax.set_title(f"Molecule: {name}")
        ax.plot(epochs[:len(mol['fermi'])], mol['fermi'], label="FermiNet", color="darkblue")
        ax.plot(epochs[:len(mol['fermi_mix'])], mol['fermi_mix'], label="DeepErwin", color="dodgerblue")

        ax.set_xticks([10000, 20000, 40000, 60000])
        ax.set_xticklabels(epoch_names)
        ax.set_xlabel("Epochs")

        ax.grid(alpha=0.5)
        ax.hlines(0, 10000, 60000, linestyles="dashed", color="grey", alpha=0.5)
    else:
        ax[i].set_title(f"Molecule: {name}")
        ax[i].plot(epochs[:len(mol['fermi'])], mol['fermi'], label="FermiNet", color = "darkblue")
        ax[i].plot(epochs[:len(mol['fermi_mix'])], mol['fermi_mix'], label="DeepErwin", color = "dodgerblue")

        ax[i].set_xticks([10000, 20000, 40000, 60000])
        ax[i].set_xticklabels(epoch_names)
        ax[i].set_xlabel("Epochs")

        ax[i].grid(alpha=0.5)
        ax[i].hlines(0, 10000, 60000, linestyles="dashed", color="grey", alpha=0.5)
ax[0].legend()
ax[0].set_ylabel("Error (mHa)")