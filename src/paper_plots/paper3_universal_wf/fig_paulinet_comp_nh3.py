# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def smooth(x, ratio=0.2):
    ratio = 1 - ratio
    x = np.array(x)
    x_smooth = [np.mean(x[int(ratio * i) : max(i, 1)]) for i in range(len(x))]
    return np.array(x_smooth)


sns.set_theme(style="whitegrid")

path = "/Users/leongerard/Desktop/opt_emean"
run_names = [
    "ablation_dpe_06_NH3_rep1",
    "ablation_paulishift_08_NH3_rep1",
    "ablation_pauli_01_NH3_rep1",
    "bench_tao_dpe_NH3",
]
data = [pd.read_csv(path + f"_{name}.csv") for name in run_names]
E_ref = -56.56381

fig, ax = plt.subplots(1, 1, figsize=(5, 4))

metadata = {
    "ablation_dpe_06_NH3_rep1": ("Gerard et al.", "C2"),
    "ablation_paulishift_08_NH3_rep1": ("PauliNet + BF Shift", "red"),
    "ablation_pauli_01_NH3_rep1": ("PauliNet", "pink"),
    "bench_tao_dpe_NH3": ("Our work", "orange"),
}
for i, (name, d) in enumerate(zip(run_names, data)):
    epochs = range(len(d[name]))

    energies = smooth(np.array(d[name].tolist()))
    errors = (energies - E_ref) * 1000

    ax.plot(epochs, errors, color=metadata[name][1], label=metadata[name][0])

ax.set_ylim([-1, 15])
ax.set_xlim([0, 50000])
# ax.set_yscale("log")
ax.set_ylabel("E - $E_0$ / mHa")
ax.set_xlabel("Steps")
ax.hlines(1.6, 0, 50000, color="grey", linestyles="--", label="Chemical acc.")
ax.legend()
ax.set_title(r"$NH_3$")
fig.tight_layout()

# fig.savefig(f"/Users/leongerard/ucloud/Shared/results/03_paper_unversal_wavefuncion/figures/fig7_paulinet_comp_nh3.pdf",
#             bbox_inches="tight")
