import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from deeperwin.run_tools.wandb_utils import get_all_eval_energies

df_full = get_all_eval_energies("schroedinger_univie/Benchmark_vs_PauliNet", print_progress=True)
df_full["molecule"] = df_full["physical.name"].str.replace("HChain10", "H10")

molecules = ["LiH", "Li2", "Be", "B", "C", "H10"]
energy_errors = dict(adam=[], kfac=[], paulinet=[0.185, 1.069, 0.122, 3.354, 8.254, 4.3853073120097])  # Li2: 30 dets
for opt in ["adam", "kfac"]:
    for m in molecules:
        df = df_full[(df_full["optimization.optimizer.name"] == opt) & (df_full["molecule"] == m)]
        if len(df) == 0:
            energy_errors[opt].append(np.nan)
        elif len(df) == 1:
            energy_errors[opt].append(df.iloc[0].error_eval_10000)
        else:
            raise ValueError("Multiple values for same molecule!")
# %%
labels = dict(
    adam="DeepErwin w/o weight-sharing\n+ Adam (~110k parameters)",
    kfac="DeepErwin w/o weight-sharing\n+ KFAC (~110k parameters)",
    paulinet="PauliNet + Adam\n(~150k parameters)",
)
colors = dict(adam="C1", kfac="darkred", paulinet=(0.2, 0.2, 0.2))

plt.close("all")
plt.figure(figsize=(3.6, 4), dpi=100)
plt.grid(axis="y", zorder=-1, which="major")
plt.grid(axis="y", zorder=-1, which="minor", alpha=0.5, ls="--")


x_ticks = np.array(range(len(molecules)))
bar_width = 0.25
for ind, method in enumerate(["kfac", "adam", "paulinet"]):
    plt.bar(
        x_ticks + (ind - 1) * bar_width,
        energy_errors[method],
        width=bar_width,
        label=labels[method],
        color=colors[method],
        zorder=3,
    )
plt.xticks(x_ticks, molecules)
plt.ylabel("energy error / mHa")
plt.legend(loc="upper left", framealpha=0.9)
plt.gca().set_yticks(np.arange(0, 10, 0.5), minor=True)
plt.ylim([0, 10])
plt.gca().yaxis.labelpad = 0

plt.savefig("/home/mscherbela/ucloud/results/paper_figures/jax/deeperwin_vs_paulinet.png", dpi=400, bbox_inches="tight")
plt.savefig("/home/mscherbela/ucloud/results/paper_figures/jax/deeperwin_vs_paulinet.pdf", bbox_inches="tight")
df_dump = pd.DataFrame(
    dict(
        molecule=molecules,
        deeperwin_kfac_mHa=energy_errors["kfac"],
        deeperwin_adam_mHa=energy_errors["adam"],
        paulinet_mHa=energy_errors["paulinet"],
    )
)
df_dump.to_csv(
    "/home/mscherbela/ucloud/results/paper_figures/jax/figure_data/Fig1_accuracy_no_weight_sharing.csv", index=False
)
