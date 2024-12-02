# %%
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

api = wandb.Api()
run = api.run("tum_daml_nicholas/cumulene/runs/n8xgnqko")

df = run.history(keys=["opt/step", "opt/E_smooth"], pandas=True, samples=10_000)
df["step"] = df["opt/step"] / 1000

plt.close("all")
fig, ax = plt.subplots(1, 1, figsize=(3.5, 4))
sns.lineplot(data=df, x="step", y="opt/E_smooth", ax=ax)
ax.set_xlabel("Optimization steps / 1k")
ax.set_ylabel("Energy / Ha")
ax.set_ylim([-78.6, -78.4])
ax.set_xlim([0, 8])
fig.tight_layout()
fig.savefig("/home/mscherbela/ucloud/results/defense/opt_curve.png", bbox_inches="tight", dpi=600)
