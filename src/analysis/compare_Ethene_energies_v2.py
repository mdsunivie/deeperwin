import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_all_runs

def extract_geometry(r):
    r['d_CC'], r['d_CH'], r['CCH_angle'], r['twist'] = map(float, r['dirname'].split('_'))
    return r

ref_fname = r'/home/mscherbela/runs/Ethene/references/Ethene_energies.csv'
df_ref = pd.read_csv(ref_fname, delimiter=';')
df_ref = df_ref.apply(extract_geometry, axis=1)
df_ref.sort_values(['d_CC', 'twist'], inplace=True)
df_ref = df_ref[df_ref.d_CC == 1.3300]


df = load_all_runs("/home/mscherbela/runs/Ethene/twist1")
df['twist'] = df.dirname.apply(lambda s: float(s.split('_')[1]))
df.sort_values(['optimization.optimizer.name', 'twist'], inplace=True)

plt.close("all")
fig, ax = plt.subplots(1,1,figsize=(5,4))
E_ferminet = -78.58578
for color, method, label in zip(['orange', 'coral', 'brown'], ['HF', 'CCSD(T)-F12', 'MRCI-D-F12'], ['Hartree-Fock', 'Coupled Clusters: CCSD(T) + D-F12', 'Multi-ref-CI + D-F12']):
    ax.plot(df_ref.twist, df_ref[method], label=label, marker='o', markersize=4, color=color)
    # shift = df_ref[method].min() - E_ferminet
    # axes[1].plot(df_ref.twist, df_ref[method] - shift, label=method + ' shifted to FermiNet', marker='o', markersize=4, color=color)

df_filt = df[df['optimization.optimizer.name'] == 'adam']
ax.plot(df_filt.twist, df_filt.E_mean, label=f"DeepErwin", marker='o', markersize=4, color='blue')



# for color, optimizer in zip(['C0', 'navy'], ['adam', 'slbfgs']):
#     df_filt = df[df['optimization.optimizer.name'] == optimizer]
#     ax.plot(df_filt.twist, df_filt.E_mean, label=f"DeepErwin: {optimizer}", marker='o', markersize=4, color=color)

ax.grid()
ax.set_xlabel("Twist / degrees", fontsize=14)
ax.set_ylabel("Energy / Hartree", fontsize=14)
ax.legend()
ax.set_xticks([0, 30, 60, 90])
plt.tight_layout()

plt.savefig("/home/mscherbela/ucloud/results/Ethene_methods_wo_FermiNet.png", dpi=800)
ax.plot([0], [-78.58578], label='FermiNet', marker='s', markersize=4, color='green')
ax.legend()
plt.savefig("/home/mscherbela/ucloud/results/Ethene_methods_w_FermiNet.png", dpi=800)




