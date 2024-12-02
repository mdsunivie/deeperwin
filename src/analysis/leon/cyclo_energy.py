

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%

path = "/Users/leongerard/Desktop/data_schroedinger/gao_reference/"
gao_cyclo = "pesnet_cyclobutadiene.csv"
fermi_cyclo = "cyclobutadiene_ferminet.csv"

df_fermi = pd.read_csv(path + fermi_cyclo)
energy_fermi = df_fermi[df_fermi['system'] == 'cycbut-ground']['energy']
energy_fermi_transition = df_fermi[df_fermi['system'] == 'cycbut-trans']['energy']

#%%
def smooth(x, ratio=0.2):
    ratio = 1 - ratio
    x = np.array(x)
    x_smooth = [np.mean(x[int(ratio*i):max(i,1)]) for i in range(len(x))]
    return x_smooth


fermi_smooth = smooth(energy_fermi)
#%%

fermi_smooth[-1]
#%%
mva_fermi = []
mva_weight = int(len(energy_fermi)*0.2)
print("Fermi", mva_weight)
for i in range(0, len(energy_fermi) - mva_weight):
    mva_fermi.append(np.mean(energy_fermi[i:i+mva_weight]))

mva_fermi_transition = []
mva_weight = int(len(energy_fermi_transition)*0.2)
for i in range(0, len(energy_fermi_transition) - mva_weight):
    mva_fermi_transition.append(np.mean(energy_fermi_transition[i:i+mva_weight]))



df_gao = pd.read_csv(path + gao_cyclo)
energy_gao = df_gao['Ground']
energy_gao_transition = df_gao['Transition']

mva_gao = []
mva_weight = int(len(energy_gao)*0.2)
print("PESNet", mva_weight)
for i in range(0, len(energy_gao) - mva_weight):
    mva_gao.append(np.mean(energy_gao[i:i+mva_weight]))

mva_gao_transition = []
mva_weight = int(len(energy_gao_transition)*0.2)
for i in range(0, len(energy_gao_transition) - mva_weight):
    mva_gao_transition.append(np.mean(energy_gao_transition[i:i+mva_weight]))

#%%

plt.loglog(154.68 + np.array(mva_fermi), label="Fermi Ground", color="orange")
plt.loglog(154.68 + np.array(mva_fermi_transition), label="Fermi Transition", color="yellow")


plt.loglog(154.68 + np.array(mva_gao), label="PESNet Ground", color="green")
plt.loglog(154.68 + np.array(mva_gao_transition), label="PESNet Transition", color="lightgreen")
plt.legend()
plt.grid(alpha=0.5)



#%%



print(mva_gao[-1] - mva_gao_transition[-1])
print(mva_fermi[-1] - mva_fermi_transition[-1])


print(np.mean(energy_fermi[-20_000:]) - np.mean(energy_fermi_transition[-20_000:]))
#%%
path = "/Users/leongerard/Desktop/data_schroedinger/ferminet/"
fermi = "train_stats.csv"

df_fermi = pd.read_csv(path + fermi)
energy_fermi = np.array(df_fermi['energy'])[-20_000:]
#plt.loglog(energy_fermi)

print(np.mean(energy_fermi))