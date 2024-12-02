from deeperwin.run_tools.wandb_utils import get_all_eval_energies
import matplotlib.pyplot as plt
import numpy as np

df = get_all_eval_energies("mscherbela/TalkFigures", print_progress=True)
df = df[df.name.str.contains("opt_comp")]
#%%
df = df.sort_values(by='error_eval', ascending=False)
nice_label=dict(adam="First Order: Adam", kfac="Second Order: KFAC", slbfgs="Second Order: L-BFGS")

plt.figure(figsize=(5,6))
for i,r in df.iterrows():
    epochs, E = [], []
    for j,c in enumerate(list(r.index)):
        if c.startswith("E_mean_eval_"):
            epochs.append(int(c.split('_')[-1]))
            E.append(r[j])
    plt.plot(np.array(epochs)/1e3, E, label=nice_label[r['optimization.optimizer.name']])
plt.axhline(r['physical.E_ref'], color='k', ls='--', label="Ground truth")
plt.legend(fontsize=12)
plt.grid()
plt.xlabel("Optimization step / 1k steps", fontsize=12)
plt.ylabel(r"Energy $E_{\theta}$", fontsize=12)
plt.tight_layout()
plt.savefig("/home/mscherbela/ucloud/results/OptimizerComparison.png", dpi=400)



