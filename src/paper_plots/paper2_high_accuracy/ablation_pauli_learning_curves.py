import numpy as np
import matplotlib.pyplot as plt
import wandb
from matplotlib.cm import get_cmap
import json

project = "ablation_pauli"


def load_runs_wandb(filter_ids):
    data = {}

    api = wandb.Api()
    runs = api.runs(f"schroedinger_univie/{project}")
    for i, run in enumerate(runs):
        if i % 10 == 0:
            print(f"Rows {i}")
        if all(f not in run.id for f in filter_ids):
            continue
        if run.state == "running":
            continue
        print(run.name)
        name = "_".join(run.name.split("_")[:-1])
        if name not in data.keys():
            data[name] = dict(intermed_error=[], epochs=[], e_mean=[])
        wandb_data = [
            row
            for row in run.scan_history(
                keys=["opt_epoch", "error_intermed_eval", "sigma_intermed_eval"], page_size=10_000
            )
        ]
        intermed_error = [row["error_intermed_eval"] for row in wandb_data] + [run.summary["error_eval"]]
        epochs = [row["opt_epoch"] for row in wandb_data] + [50_000]

        data[name]["intermed_error"].append(intermed_error)
        data[name]["epochs"].append(epochs)

        e_mean = [
            row["opt_E_mean_unclipped"] for row in run.scan_history(keys=["opt_E_mean_unclipped"], page_size=10_000)
        ]
        data[name]["e_mean"].append(e_mean)
    return data


# 1&2: env + pauli, 3&4: cas + pauli, 5: deeperwin
filter_NH3 = ["3h4guxx7", "3r4mxzfx", "1y224nu6", "zcljdrjm", "2c5j584v", "1vdho2py", "d2tnhc4r"][:5]
data_pauli = load_runs_wandb(filter_NH3)
data_fname = "/home/mscherbela/ucloud/results/02_paper_figures_high_acc/plot_data/ablation_paulinet.json"
with open(data_fname, "w") as f:
    json.dump(data_pauli, f)

# %%
data_fname = "/home/mscherbela/ucloud/results/02_paper_figures_high_acc/plot_data/ablation_paulinet.json"
with open(data_fname) as f:
    data_pauli = json.load(f)


def smooth(x, ratio=0.2):
    ratio = 1 - ratio
    x = np.array(x)
    x_smooth = [np.mean(x[int(ratio * i) : max(i, 1)]) for i in range(len(x))]
    return x_smooth


nb_samples = int(50000 * 0.1)
for key in data_pauli.keys():
    e_mean = data_pauli[key]["e_mean"][0][:50_000]
    mva = []
    for i in range(0, len(e_mean) - nb_samples):
        mva.append(np.mean(e_mean[i : i + nb_samples]))
    data_pauli[key]["mva"] = smooth(e_mean)


def correct_nb_epochs(x):
    if "fixed" in x:
        nb_epochs = x.split("_")[-3]
        return int(nb_epochs)
    nb_epochs = x.split("_")[-2]
    return int(nb_epochs)


# %%
plt.close("all")
figsize = (4, 4)
fig, ax = plt.subplots(1, 1, figsize=figsize)
color_env = "dimgray"  # get_cmap("inferno")(0.05) #'dimgrey'
color_pauli = get_cmap("inferno")(0.6)  #'teal'
color_dpe = get_cmap("viridis")(0.8)  #'lightskyblue'

color_summary = [color_dpe, color_env, color_pauli]
kind = "tall"  # talll
labels = dict(
    ablation_pauli_01_NH3="Built-in approximate solution",
    copy_of_2_regression_nips_fulldet_NH3_tf32="Standard architecture",
)
colors = dict(
    ablation_pauli_01_NH3=get_cmap("inferno")(0.6), copy_of_2_regression_nips_fulldet_NH3_tf32=get_cmap("viridis")(0.8)
)
for i, key in enumerate(["ablation_pauli_01_NH3", "copy_of_2_regression_nips_fulldet_NH3_tf32"]):
    error = (56.5644 + np.array(data_pauli[key]["mva"])) * 1000
    ax.plot(range(100, len(error)), error[100:], label=labels[key], color=colors[key])

ax.set_xticks([0, 10000, 20000, 30000, 40000, 50000])
ax.set_ylim([0, 18])
ax.set_xticklabels([0, 10, 20, 30, 40, 50])
# ax.axhline(published_errors[molecule], color='k', ls='--', label='Best published\nvariational: [Pfau 2020],\n200k epochs')
ax.legend(loc="center right", framealpha=0.9, fontsize=9)
# ax[set_title(f"CASSCF vs. exp. envelope")
ax.set_xlabel("Epochs / 1k")
# ax.grid(alpha=0.5)
ax.set_ylabel("Energy error / mHa")
# ax[text(0.05, 1.04, "b.)", ha='right', va='center', transform=ax[transAxes, weight='bold')

fig.tight_layout()
fig_fname = f"/home/mscherbela/Nextcloud/PhD/talks_and_conferences/2022_NEURIPS/figures/prior_physics_{figsize[0]}x{figsize[1]}.png"
fig.savefig(fig_fname, bbox_inches="tight", dpi=400)
