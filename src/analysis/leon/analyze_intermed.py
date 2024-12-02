
import os

import numpy as np
import wandb
import matplotlib.pyplot as plt

api = wandb.Api()

runs = api.runs("schroedinger_univie/anaalyze_intermed")
break_id = 10

epochs = tuple([1000*i for i in range(1, 30)])



comparison = {}
for i, run in enumerate(runs):
    print(f"Run {i}")
    if run.state == "crashed":
        continue
    if i >= break_id:
        break
    name = "_".join(run.name.split("_")[2:])
    opt_error_smooth = [row['opt_error_smooth'] for row in run.scan_history(keys=["opt_error_smooth", "opt_epoch"], page_size=2000)
                        if row['opt_epoch'] in epochs]

    intermed_error = [row['error_intermed_eval'] for row in run.scan_history(keys=["opt_epoch", "error_intermed_eval", "sigma_intermed_eval"],
                                                      page_size=2000)]

    print(name, len(opt_error_smooth), len(intermed_error), i)
    comparison[name] = (opt_error_smooth, intermed_error)


#%%
fig, ax = plt.subplots(len(runs)-1, 1, figsize=(16, 8))
fig_hist, ax_hist = plt.subplots(len(runs)-1, 1, figsize=(16, 8))

skip = 0
for i, name in enumerate(comparison):
    opt_error_smooth, intermed_error = comparison[name]
    opt_error_smooth, intermed_error = opt_error_smooth[skip:], intermed_error[skip:]
    print(name, len(opt_error_smooth), len(intermed_error), i)

    ax[i].grid(alpha=0.5)

    ax[i].set_title(f"MCMC: {name}")
    ax[i].plot(opt_error_smooth, label=f"opt_smooth", color=f"darkblue", marker="v")
    ax[i].plot(intermed_error, label=f"intermed", color=f"red", marker="o")
    ax_hist[i].hist(np.array(opt_error_smooth) - np.array(intermed_error))
ax[0].legend()
fig.tight_layout()

# fig, ax = plt.subplots(1, 1)
# errors = []
# for i, name in enumerate(comparison):
#     opt_error_smooth, intermed_error = comparison[name]
#     error = np.abs(np.array(opt_error_smooth) - np.array(intermed_error))
#     errors.append(error[10:])
#
# errors = np.array(errors)
# ax.grid(alpha=0.5)
#
# ax.set_title(f"Diff MCMC")
# #ax.plot(errors.mean(axis=0), , label=f"error", color=f"darkblue", marker="v")
# ax.boxplot(errors)
# ax.legend()
# fig.tight_layout()


