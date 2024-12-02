import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os 


path = ""

# DeepSolid data for LiH
all_data = []
a_values = np.arange(34, 48 + 1, 2)
for a in a_values:
    fname = os.path.join(path, f"plot_data/deepsolid_simulation_data/lih_a0_{a}.csv")
    df = pd.read_csv(fname)
    is_eval = df.index >= (len(df) - 50_000)
    is_intermed_eval = (df.index > 50_000) & (df.step < 50_000) & (~is_eval)
    is_opt = ~(is_eval | is_intermed_eval)
    df.loc[is_eval, "type"] = "eval"
    df.loc[is_intermed_eval, "type"] = "intermed_eval"
    df.loc[is_opt, "type"] = "opt"
    df["a0"] = float(a / 10)
    all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)
df_eval = df_all[df_all.type == "eval"].pivot_table(index="step", columns="a0", values="energy")
df_eval = df_eval.mean(axis=0)

# Load reuse from indep
twists = [
    "0_0_0",
    "-0.4_-0.4_0",
    "-0.4_-0.4_-0.4",
    "-0.2_-0.6_-0.6",
    "-0.2_-0.4_-0.6",
    "-0.2_-0.2_0",
    "-0.2_-0.2_0.2",
    "-0.2_-0.2_-0.6",
    "-0.2_-0.2_-0.4",
    "-0.2_-0.2_-0.2",
]

twist_weights = [0.008, 0.048, 0.064, 0.096, 0.192, 0.048, 0.192, 0.192, 0.096, 0.064]
twist_weights = defaultdict(lambda: 0, dict(zip(twists, twist_weights)))

df_all = pd.read_csv(os.path.join(path, "plot_data/reuse_from_indep.csv"))

df_all.opt_E_mean =df_all.opt_E_mean* df_all.twist.map(twist_weights) / 8 / 8
df_all = df_all.groupby("opt_epoch")[["opt_E_mean"]].sum().reset_index() 

smoothing_window_LiH = 1000
df_dpe = pd.read_csv(os.path.join(path, "plot_data/LiH_2x2x2_shared.csv"))
twists = [
    "0.000_0.000_0.000",
    "-0.400_-0.400_0.000",
    "-0.400_-0.400_-0.400",
    "-0.200_-0.600_-0.600",
    "-0.200_-0.400_-0.600",
    "-0.200_-0.200_0.000",
    "-0.200_-0.200_0.200",
    "-0.200_-0.200_-0.600",
    "-0.200_-0.200_-0.400",
    "-0.200_-0.200_-0.200",
]
twist_weights = [0.008, 0.048, 0.064, 0.096, 0.192, 0.048, 0.192, 0.192, 0.096, 0.064]
twist_weights = defaultdict(lambda: 0, dict(zip(twists, twist_weights)))


df_dpe["E_weighted_10tw"] = df_dpe.opt_E_mean * df_dpe.twist.map(twist_weights)
df_dpe = df_dpe.pivot_table(index="opt_n_epoch", columns=["geom", "twist"], values=["E_weighted_10tw"])
df_dpe = df_dpe.fillna(method="ffill")
df_dpe_smooth_LiH = pd.DataFrame(
    data=dict(
        step=df_dpe.index,
        E_10tw=df_dpe["E_weighted_10tw"].sum(axis=1).rolling(window=smoothing_window_LiH).mean() / 8 / 8,
    )
)


df_dpe_eval = pd.read_csv(os.path.join(path, "plot_data/fig_3_lih_data.csv"))
df_dpe_eval = df_dpe_eval[df_dpe_eval.method == "moon_tao_lih222"]
df_dpe_eval_gamma = df_dpe_eval[df_dpe_eval.tabc_weight == 0.008]
E_eval_dpe_10tw = (df_dpe_eval.E_mean * df_dpe_eval.tabc_weight).sum()

fig, ax_LiH = plt.subplots(1, 1, figsize=(8, 3.5))

# Plot LiH
remove_smoothing_artifact = 80
ax_LiH.plot(
    100 + 10*8 * df_all.index[remove_smoothing_artifact:] / 1000,
    df_all.opt_E_mean[remove_smoothing_artifact:],
    label="Reuse from Independent",
    color="C0",
)

epochs_used_to_eval = 100_000
ax_LiH.plot(
    df_dpe_smooth_LiH.step[:epochs_used_to_eval] / 1000, df_dpe_smooth_LiH.E_10tw[:epochs_used_to_eval], label="Shared", color="C3"
)

ax_LiH.set_xlabel("Total optimization steps / k")
ax_LiH.set_ylabel("Averaged energy per unit cell / Ha")
ax_LiH.set_ylim([-8.185, -8.155])
ax_LiH.legend()
ax_LiH.grid(alpha=0.2, color="gray")

ax_LiH.set_title("LiH", fontsize=10)

fig.tight_layout()
save_dir = "plot_output"
fig.savefig(os.path.join(path, f"{save_dir}/figSI_reuse_from_indep.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(path, f"{save_dir}/figSI_reuse_from_indep.png"), bbox_inches="tight", dpi=300)

plt.show()