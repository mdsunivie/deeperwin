import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def smooth(x, ratio=0.2):
    ratio = 1 - ratio
    x = np.array(x)
    x_smooth = [np.mean(x[int(ratio*i):max(i,1)]) for i in range(len(x))]
    return x_smooth

data_dir = "/home/mscherbela/tmp/full_runs"
data = []
for fname in os.listdir(data_dir):
    df = pd.read_csv(os.path.join(data_dir, fname), sep=';')
    df['molecule'] = fname.split('_')[0]
    df['init'] = fname.split('_')[1]
    data.append(df)
df_all = pd.concat(data, axis=0, ignore_index=True)
df_eval = df_all[df_all.metric_type == 'eval']
df_all = df_all[df_all.metric_type == 'opt']
for id in df_all.id.unique():
    df_all.loc[df_all.id == id, 'E_mean_smooth'] = smooth(df_all.loc[df_all.id==id, "opt_E_mean_unclipped"])
    df_all.loc[df_all.id == id, 'E_std_smooth'] = smooth(df_all.loc[df_all.id == id, "opt_E_std_unclipped"])

#%%
molecule = 'Ar'
df = df_all[df_all.molecule == molecule]
E_ref = dict(
    Fe=-1263.655, # Update!
    K=-599.925,
    Ar=-527.541 # Update!
             )

plt.close("all")

labels=dict(constant="$\\omega=1$ initialization", hf="$\\omega=Z/n_k$ initialization")
fig, axes = plt.subplots(1,2,dpi=100, figsize=(6,5))
for ind_init, init in enumerate(['constant', 'hf']):
    color=f"C{ind_init}"
    df_filt = df[df.init == init]
    axes[0].semilogy(df_filt.opt_epoch/1e3, (df_filt.E_mean_smooth - E_ref[molecule])*1e3, label=labels[init], color=color)
    axes[1].semilogy(df_filt.opt_epoch/1e3, df_filt.E_std_smooth**2, label=labels[init], color=color)
    df_eval_filt = df_eval[(df_eval.init == init) & (df_eval.molecule == molecule)]
    axes[0].errorbar(df_eval_filt.opt_epoch/1e3,
                     (df_eval_filt.E_eval - E_ref[molecule])*1e3,
                     yerr=df_eval_filt.sigma_eval*1e3,
                     capsize=2,
                     color=['navy', 'red'][ind_init],
                     marker='d',
                     ms=3,
                     elinewidth=2,
                     ls='None')
for ax in axes:
    ax.grid(alpha=0.5)
    ax.set_xlabel("epochs / k")
    ax.legend(loc='upper right')

axes[0].set_ylim([None, 500])
axes[0].set_title("energy error")
axes[0].set_ylabel("$E - E_{100k}$ / mHa")
axes[0].set_yticks([7,8,9,10,20,30,40,50,60,70,80,90,100, 200, 300, 400])
axes[1].set_yticks([4,5,6,7,8,9,10,20,40,80])

axes[1].set_ylim([None, 90])
axes[1].set_title("energy variance")
axes[1].set_ylabel("var(E) / $\\mathrm{Ha}^2$")
fig.suptitle(f"Envelope initialization for {molecule} atom")
fig.subplots_adjust(top=0.93)
fig.tight_layout()
fig_fname = f"/home/mscherbela/ucloud/results/02_paper_figures_high_acc/envelope_initialization_{molecule}.png"
fig.savefig(fig_fname, bbox_inches='tight')
fig.savefig(fig_fname.replace('.png', '.pdf'), bbox_inches='tight')



