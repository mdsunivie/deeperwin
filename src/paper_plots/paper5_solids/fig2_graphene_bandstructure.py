# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("plot_data/fig_2_graphene_twist_path.csv")
df["idx_k"] = np.arange(len(df))
df_pretrain = df[df.in_original_training_dataset]

# Grid of 19 k-points for pretraining
k_vec1, k_vec2 = df_pretrain[["kx", "ky"]].iloc[[-2, 1]].values
grid = [6, 4, 3, 1, 0]

grid_points = []
for i, n in enumerate(grid):
    k = i * k_vec1 + np.arange(n + 1)[:, None] * k_vec2
    grid_points.append(k)
grid_points = np.concatenate(grid_points, axis=0)

plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
axes[0].scatter(grid_points[:, 0], grid_points[:, 1], c="gray", s=120, label="Pretraining grid")
axes[0].plot(df.kx, df.ky, marker="o", color="red", label="Fine-tuned dataset")
axes[0].set_aspect("equal")
axes[1].plot(df.idx_k, df.energy, marker="o", color="red", ms=4, label="Fine-tuned energy")
axes[1].plot(
    df_pretrain.idx_k,
    df_pretrain.energy,
    marker="o",
    ls="none",
    color="gray",
    ms=8,
    zorder=0,
    label="in pretraining grid",
)


ind_special = [0, 18, 26, 42]
label_special = [r"$\Gamma$", "M", "K", r"$\Gamma$"]
pos_special = df.iloc[ind_special][["kx", "ky"]].values
offsets = np.array([[0.005, -0.01], [0.01, 0], [0, -0.01]])

axes[1].set_xticks(ind_special)
axes[1].set_xticklabels(label_special)
for i in ind_special:
    axes[1].axvline(i, color="gray", linestyle="-", alpha=0.5, zorder=-1)

for label, pos, offset in zip(label_special[:-1], pos_special[:-1], offsets):
    axes[0].text(*(pos + offset), label, fontsize=12, ha="center", va="center")

axes[0].legend(loc="upper left")
axes[1].legend(loc="upper center")
axes[0].set_xlim([None, 0.01])
axes[0].set_xlabel(r"$k_x$ / $a_0^{-1}$")
axes[0].set_ylabel(r"$k_y$ / $a_0^{-1}$")
axes[1].set_xlabel(r"Twist index")
axes[1].set_ylabel("Energy / Ha")

fig.tight_layout()
fig.savefig("plot_output/fig2_graphene_bandstructure.pdf", bbox_inches="tight")
fig.savefig("plot_output/fig2_graphene_bandstructure.png", bbox_inches="tight", dpi=300)
