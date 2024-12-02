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
method = 'CCSD(T)-F12'
df_ref = df_ref[df_ref.d_CC == 1.3300]


df = load_all_runs("/home/mscherbela/runs/Ethene/twist1")
df['twist'] = df.dirname.apply(lambda s: float(s.split('_')[1]))
df.sort_values(['optimization.optimizer.name', 'twist'], inplace=True)

plt.close("all")
fig, axes = plt.subplots(1,2,figsize=(14,7))
E_ferminet = -78.58578
for color, method in zip(['orange', 'brown', 'coral'], ['HF', 'MRCI-D-F12', 'CCSD(T)-F12']):
    axes[0].plot(df_ref.twist, df_ref[method], label=method + ' raw', marker='o', markersize=4, color=color)
    shift = df_ref[method].min() - E_ferminet
    axes[1].plot(df_ref.twist, df_ref[method] - shift, label=method + ' shifted to FermiNet', marker='o', markersize=4, color=color)

for color, optimizer in zip(['C0', 'navy'], ['adam', 'slbfgs']):
    df_filt = df[df['optimization.optimizer.name'] == optimizer]
    axes[0].errorbar(df_filt.twist, df_filt.E_mean, yerr=df_filt.E_mean_sigma, label=f"DeepErwin: {optimizer}", marker='o', markersize=4, color=color)
    shift = df_filt.E_mean.min() - E_ferminet
    axes[1].errorbar(df_filt.twist, df_filt.E_mean - shift, label=f"DeepErwin: {optimizer}", marker='o', markersize=4, color=color)
# plt.plot([0], [-78.523783], label='Lowest energy from CCCB Benchmark Database', marker='d', markersize=7, color='red')
for ax in axes:
    ax.plot([0], [-78.58578], label='FermiNet', marker='s', markersize=7, color='green')
    ax.grid()
    ax.set_xlabel("Twist / degree")
    ax.set_ylabel("E / Ha")
    ax.legend()
axes[0].set_title("Raw Energies")
axes[1].set_title("Energies shifted to FermiNet at twist==0")


# plt.plot([0], [-78.574295043945], label='DeepErwin BFGS (FermiNet geometry)', marker='s', markersize=7, color='lightblue')

plt.suptitle("Ethene: Comparison of methods", fontsize=18)
plt.savefig("/home/mscherbela/ucloud/results/Ethene_ref_comparison.png")




