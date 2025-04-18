# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import subprocess
smoothing_window_LiH = 1000
smoothing_window_graphene = 4000

path = ""
if not os.path.isfile("plot_data/deepsolid_simulation_data/lih_a0_34.csv"):
    print("Downloading deepsolid data from github...")
    subprocess.call("git clone https://github.com/GiantElephant123/solid_simulation_data.git plot_data/deepsolid_simulation_data".split())


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
df_opt = df_all[df_all.type == "opt"].pivot_table(index="step", columns="a0", values="energy").dropna()
df_opt = df_opt.rolling(window=smoothing_window_LiH).mean()
df_eval = df_all[df_all.type == "eval"].pivot_table(index="step", columns="a0", values="energy").dropna()
df_eval = df_eval.mean(axis=0)
a_values = list(df_opt)
df_deepsolid_mean_LiH = df_opt.mean(axis=1)


# DeepSolid data for graphene
fnames = ["graphene_k1.csv", "graphene_k2.csv", "graphene_k3.csv"]
fnames = [f"plot_data/deepsolid_simulation_data/{fname}" for fname in fnames]
weights = np.array([1, 6, 2]) / 9

all_data = []
for fname, weight in zip(fnames, weights):
    df = pd.read_csv(os.path.join(path, fname))
    df["E_weighted"] = df.energy * weight
    df["type"] = "opt"
    df.loc[df.index >= 300_000, "type"] = "eval"
    all_data.append(df)
df_all = pd.concat(all_data, ignore_index=True)
df_opt_graphene = df_all[df_all["type"] == "opt"].groupby("step")[["E_weighted"]].sum().reset_index()
df_eval_graphene = df_all[df_all["type"] == "eval"].groupby("step")[["E_weighted"]].sum().reset_index()
df_opt_graphene["E_smooth"] = df_opt_graphene.E_weighted.rolling(window=smoothing_window_graphene).mean()


# %%
# DeepErwin data for LiH
df_dpe_eval = pd.read_csv(os.path.join(path, "plot_data/fig_3_lih_data.csv"))
df_dpe_eval = df_dpe_eval[df_dpe_eval.method == "moon_tao_lih222"]
df_dpe_eval_gamma = df_dpe_eval[df_dpe_eval.tabc_weight == 0.008]
E_eval_dpe_gamma = df_dpe_eval_gamma.E_mean.sum()
E_eval_dpe_10tw = (df_dpe_eval.E_mean * df_dpe_eval.tabc_weight).sum()

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
df_dpe["E_gamma"] = df_dpe.loc[df_dpe.twist == twists[0], "opt_E_mean"]
df_dpe = df_dpe.pivot_table(index="opt_n_epoch", columns=["geom", "twist"], values=["E_weighted_10tw", "E_gamma"])
df_dpe = df_dpe.fillna(method="ffill")
df_dpe_smooth_LiH = pd.DataFrame(
    data=dict(
        step=df_dpe.index,
        E_gamma=df_dpe["E_gamma"].sum(axis=1).rolling(window=smoothing_window_LiH).mean() / 8 / 8,
        E_10tw=df_dpe["E_weighted_10tw"].sum(axis=1).rolling(window=smoothing_window_LiH).mean() / 8 / 8,
    )
)
epochs_used_to_eval = 100_000
df_dpe_smooth_LiH = df_dpe_smooth_LiH[:epochs_used_to_eval]


# DeepErwin data
df_dpe = pd.read_csv(os.path.join(path, "plot_data/graphene_2x2_shared.csv"))
df_dpe_eval = pd.read_csv(os.path.join(path, "plot_data/graphene_data.csv"))
weights_3twists = defaultdict(
    lambda: 0,
    {
        "3af54b12dae1e261a35aa9dbf6455d5a": 1 / 9,
        "c34cc71657325f9acbe715b3718fbc23": 6 / 9,
        "3f88c90cf1ee073aa3f6657b95cfcfc9": 2 / 9,
    },
)
geom_hashes_19twists = [
    "3af54b12dae1e261a35aa9dbf6455d5a",
    "9ea4b0891a2249078b8ef182fe17edb7",
    "7c5eed689c58205c44cf6aea61f21d8d",
    "3f88c90cf1ee073aa3f6657b95cfcfc9",
    "c34cc71657325f9acbe715b3718fbc23",
    "bb220817f77bcd6e8a39bcdf3f88c041",
    "ba57a95f704ed8dc477ea832d61fc220",
    "d58c8c6e8da1bb0d4aa1323893fe4db5",
    "19e7d6ebdeb6b0dd235f3ae015041919",
    "a9d4642411bdabc443aa9f0fc2cf9603",
    "ed8047da7085e5501c98d1e425ab2860",
    "b4af03e0160b2ac4615277a5d2f22e00",
    "eeae5570d6fce24692fbd074c910dfb5",
    "a514a275be992dbc170aae5385086e0c",
    "6c079e0f5d6aff664979fe4b91ee6c5b",
    "fd23d3b9534440a3c0bd3dc00db40b37",
    "a3179cd17a72a16bf88ec3e13edcb694",
    "cec05f7fe45bb4d8bb22a8277d212b16",
    "faa5caaa69701fdec32c8a3b16e2a90d",
]
weights_19twists = [
    0.006944444444444444,
    0.020833333333333332,
    0.041666666666666664,
    0.013888888888888888,
    0.041666666666666664,
    0.08333333333333333,
    0.041666666666666664,
    0.041666666666666664,
    0.041666666666666664,
    0.08333333333333333,
    0.08333333333333333,
    0.041666666666666664,
    0.041666666666666664,
    0.08333333333333333,
    0.08333333333333333,
    0.08333333333333333,
    0.08333333333333333,
    0.041666666666666664,
    0.041666666666666664,
]
weights_19twists = defaultdict(lambda: 0, dict(zip(geom_hashes_19twists, weights_19twists)))
df_dpe_eval["E_weighted_3twists"] = df_dpe_eval.E_mean * df_dpe_eval.hash.map(weights_3twists)
df_dpe_eval["E_weighted_19twists"] = df_dpe_eval.E_mean * df_dpe_eval.hash.map(weights_19twists)
E_dpe_eval_3tw = df_dpe_eval["E_weighted_3twists"].sum()
E_dpe_eval_19tw = df_dpe_eval["E_weighted_19twists"].sum()


df_dpe["E_weighted_3tw"] = df_dpe.opt_E_mean * df_dpe.geom_hash.map(weights_3twists)
df_dpe["E_weighted_19tw"] = df_dpe.opt_E_mean * df_dpe.geom_hash.map(weights_19twists)
df_dpe = df_dpe.pivot_table(index="opt_n_epoch", columns="geom_hash", values=["E_weighted_3tw", "E_weighted_19tw"])
df_dpe = df_dpe.fillna(method="ffill")

df_dpe_smooth_graphene = pd.DataFrame(
    data=dict(
        step=df_dpe.index,
        E_3tw=df_dpe["E_weighted_3tw"].sum(axis=1).rolling(window=smoothing_window_graphene).mean() / 4,
        E_19tw=df_dpe["E_weighted_19tw"].sum(axis=1).rolling(window=smoothing_window_graphene).mean() / 4,
    )
)


# %%


plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
ax_graphene, ax_LiH = axes

# Plot graphene
ax_graphene.plot(
    3 * df_opt_graphene.step / 1000, df_opt_graphene.E_smooth, label="DeepSolid (3$\\times$3 twist-grid)", color="C0"
)
ax_graphene.plot([3 * df_opt_graphene.step.max() / 1000], [df_eval_graphene.E_weighted.mean()], "x", color="C0")
ax_graphene.plot(
    df_dpe_smooth_graphene.step / 1000,
    df_dpe_smooth_graphene.E_3tw,
    label="Ours (3$\\times$3 twist subset)",
    color="C1",
)
ax_graphene.plot([df_dpe_smooth_graphene.step.max() / 1000], [E_dpe_eval_3tw], "x", color="C1")
ax_graphene.plot(
    df_dpe_smooth_graphene.step / 1000,
    df_dpe_smooth_graphene.E_19tw,
    label="Ours (12$\\times$12 twist-grid)",
    color="C3",
)
ax_graphene.plot([df_dpe_smooth_graphene.step.max() / 1000], [E_dpe_eval_19tw], "x", color="C3")
ax_graphene.set_xlabel("Total optimization steps / k")
ax_graphene.set_ylabel("Twist-averaged energy per unit cell / Ha")
ax_graphene.set_ylim([-76.255, -76.200])
ax_graphene.legend()
ax_graphene.grid(alpha=0.2, color="gray")
fig.tight_layout()


# Plot LiH
ax_LiH.plot(
    8 * df_deepsolid_mean_LiH.index / 1000,
    df_deepsolid_mean_LiH.values,
    label="DeepSolid ($\\Gamma$ point)",
    color="C0",
)
ax_LiH.plot([8 * df_deepsolid_mean_LiH.index.max() / 1000], [df_eval.mean()], "x", color="C0")
ax_LiH.plot(df_dpe_smooth_LiH.step / 1000, df_dpe_smooth_LiH.E_gamma, label="Ours ($\\Gamma$ subset)", color="C1")
ax_LiH.plot([df_dpe_smooth_LiH.step.max() / 1000], [E_eval_dpe_gamma / 8], "x", color="C1")

ax_LiH.plot(
    df_dpe_smooth_LiH.step / 1000, df_dpe_smooth_LiH.E_10tw, label="Ours (5$\\times$5$\\times$5 twist-grid)", color="C3"
)
ax_LiH.plot([df_dpe_smooth_LiH.step.max() / 1000], [E_eval_dpe_10tw / 8], "x", color="C3")


ax_LiH.set_xlabel("Total optimization steps / k")
ax_LiH.set_ylabel("Averaged energy per unit cell / Ha")
ax_LiH.set_ylim([-8.175, -8.11])
ax_LiH.legend()
ax_LiH.grid(alpha=0.2, color="gray")

for ax, letter, titles in zip(axes, "ab", ["Graphene", "LiH"]):
    ax.text(0.00, 1.0, letter, transform=ax.transAxes, va="bottom", ha="left", fontweight="bold", fontsize=12)
    ax.set_title(titles, fontsize=10)

fig.tight_layout()
save_dir = "plot_output"
fig.savefig(f"{save_dir}/figSI_optimization_curves.pdf", bbox_inches="tight")
fig.savefig(f"{save_dir}/figSI_optimization_curves.png", bbox_inches="tight", dpi=300)
