# %%
import pandas as pd
import wandb

# api = wandb.Api()
# runs = api.runs("schroedinger_univie/graphene_basis")
# df = []
# for r in runs:
#     assert r.summary["opt_n_epoch"] == 100_000
#     df.append(
#         dict(
#             E=r.summary["E_mean"],
#             E_sigma=r.summary["E_mean_sigma_corr"],
#             basis=r.config["baseline.basis_set"],
#             twist_nr=int(r.name.split("_")[-1]),
#             twist_weight=r.config["physical.weight_for_shared"] / 9.0,
#             E_hf=r.summary["E_periodic_hf"],
#         )
#     )
#     print(r.name)

# df = pd.DataFrame(df)
# df.to_csv("plot_data/graphene_basis_set_dependance.csv", index=False)

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("plot_data/graphene_basis_set_dependance.csv")
df["E_weighted"] = df["E"] * df["twist_weight"]
df["E_sigma_weighted"] = df["E_sigma"] ** 2 * df["twist_weight"]
df["E_hf_weighted"] = df["E_hf"] * df["twist_weight"]
df_tabc = df.groupby("basis")[["E_weighted", "E_hf_weighted", "E_sigma_weighted"]].sum().reset_index()
df_tabc["E_sigma_weighted"] = np.sqrt(df_tabc["E_sigma_weighted"])
print(df_tabc)

plt.close("all")
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(df_tabc["basis"], df_tabc["E_hf_weighted"], label="Hartree-Fock", marker="s")
ax.errorbar(
    df_tabc["basis"], df_tabc["E_weighted"], yerr=df_tabc["E_sigma_weighted"], capsize=5, label="This work", marker="o"
)
ax.set_ylabel("Energy / Ha")
ax.set_xlabel("Basis set")
ax.grid(alpha=0.5)
ax.legend()
fig.tight_layout()
fig_fname = "plot_output/basis_set_dependence"
fig.savefig(f"{fig_fname}.pdf", bbox_inches="tight")
fig.savefig(f"{fig_fname}.png", bbox_inches="tight", dpi=400)
