import pandas as pd
import wandb
import numpy as np
import matplotlib.pyplot as plt

api = wandb.Api()

run_ids = ["/schroedinger_univie/norm_constraint/runs/1jjxm6mw",
           "schroedinger_univie/norm_constraint/runs/2mdmj7ua"]

df_all = pd.DataFrame()
for id in run_ids:
    run = api.run(id)
    data = []
    for i,row in enumerate(run.scan_history(keys=["opt_epoch", "opt_E_std_clipped", "opt_norm_constraint_factor"], page_size=10_000)):
        data.append(row)
        if (i % 1000) == 0:
            print(i)
    df = pd.DataFrame(data)
    df['molecule'] = run.config['physical.name']
    df_all = pd.concat([df_all, df], axis=0, ignore_index=True)


#%%
molecules = df_all.molecule.unique()

def ema(x, factor=0.8):
    x = np.array(x)
    y = np.zeros_like(x)
    y[0] = x[0]
    for i, x_ in enumerate(x[1:]):
        y[i+1] = y[i] * factor + (1-factor) * x_
    return y

df_all['norm'] = 1.0 / df_all["opt_norm_constraint_factor"]


plt.close("all")
fig, axes = plt.subplots(1,2, figsize=(9,6), dpi=100)
ax_stdE, ax_norm = axes

for molecule in molecules:
    df_filt = df_all[df_all.molecule == molecule]
    ax_norm.semilogy(df_filt.opt_epoch / 1000, ema(df_filt.norm), label=molecule)
    ax_stdE.semilogy(df_filt.opt_epoch / 1000, ema(df_filt.opt_E_std_clipped), label=molecule)

ax_norm.set_ylim([2e-2, 20])
ax_stdE.set_ylim([4e-2, 10])

epoch_transition = dict(F=11.0, C=4.6)

for ax in axes:
    for i, molecule in enumerate(molecules):
        ax.axvline(epoch_transition[molecule], ls='--', color=f"C{i}")
    ax.legend()
    ax.set_xlabel("epoch / k")
    ax.grid(alpha=0.5)

ax_norm.axhline(1, color='k')
ax_stdE.set_title("std(E_loc)")
ax_norm.set_title("gradient norm / norm_constraint")

fig.tight_layout()
fig.savefig("/home/mscherbela/ucloud/results/norm_constraint.png", dpi=400, bbox_inches='tight')







