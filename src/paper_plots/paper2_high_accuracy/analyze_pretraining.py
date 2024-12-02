import pandas as pd
import matplotlib.pyplot as plt
import wandb


# %%
def load_wandb():
    project = "pretrainingV2"

    api = wandb.Api()
    runs = api.runs(f"schroedinger_univie/{project}")

    filter_names = tuple(["prod_cas_tf32_O", "prod_hf_tf32_O", "prod_cas_NH3", "prod_hf_NH3"])  #

    data = []
    for counter, run in enumerate(runs):
        if counter % 10 == 0:
            print(f"Rows {counter}")

        if run.state == "running" or run.state == "crashed":
            continue
        if all(f not in run.name for f in filter_names):
            continue

        method = "_".join(run.name.split("_")[2:4])
        print(method)
        d = dict(
            name=run.name, molecule=run.config["physical.name"], method=method, error_final=run.summary["error_eval"]
        )
        intermed_error = [
            row
            for row in run.scan_history(
                keys=["opt_epoch", "error_intermed_eval", "sigma_intermed_eval"], page_size=10_000
            )
        ]
        for row in intermed_error:
            d[f"error_intermed{row['opt_epoch']}"] = row["error_intermed_eval"]
        data.append(d)

    print("finished")
    name = "pretrainingV2.xlsx"
    excel_file_path = "/Users/leongerard/Desktop/data_schroedinger/" + name

    df = pd.DataFrame(data)
    df.to_excel(excel_file_path)


# %%


def correct_nb_epochs(x):
    nb_epochs = x.split("_")[-2]
    return int(nb_epochs)


name = "pretrainingV2.xlsx"
excel_file_path = "/Users/leongerard/Desktop/data_schroedinger/" + name
df = pd.read_excel(excel_file_path)

# %%

methods = ["prod_cas", "prod_hf", "prod_cas", "prod_hf"]
metrics = ["error_final", "error_intermed20000"]

# fig, ax = plt.subplots(1, len(df['molecule'].unique()), figsize=(14, 6))
fig, axs = plt.subplots(1, 1, dpi=100, figsize=(5, 3.5))
ax = [axs]
color = ["firebrick", "lightcoral", "darkblue", "lightskyblue"]
for i, m in enumerate(["NH3"]):  # df['molecule'].unique()
    data_pretraining = df[df["molecule"] == m]

    for j, meth in enumerate(["prod_cas", "prod_hf"]):
        data = data_pretraining[data_pretraining["method"] == meth]
        data = data.assign(nb_epochs=data["name"].apply(correct_nb_epochs))
        data = data.sort_values(by=["nb_epochs"])
        data = data.groupby(["nb_epochs"]).agg(["mean", "std"]).reset_index()

        if meth == "prod_cas":
            label = "CASSCF"
        else:
            label = "Hartree Fock"
        ax[i].errorbar(
            data["nb_epochs"].unique(),
            data["error_final"]["mean"],
            yerr=data["error_final"]["std"],
            label=f"{label} - 50k opt. epochs",
            color=color[2 * j],
            capsize=3,
        )
        ax[i].errorbar(
            data["nb_epochs"].unique(),
            data["error_intermed20000"]["mean"],
            yerr=data["error_intermed20000"]["std"],
            label=f"{label} -  20k opt. epochs",
            color=color[2 * j + 1],
            capsize=3,
        )

    ax[i].set_xscale("log")
    ax[i].set_xticks(list(data["nb_epochs"].unique()))
    ax[i].set_xticklabels([0.2, 1, 5, 20, 100, 200])
    ax[i].set_xlabel("Pretraining epochs / 1k")

    if m == "O":
        ax[i].set_ylim([-0.5, 1.8])
    else:
        ax[i].set_ylim([0.0, 3.0])

    ax[i].grid(alpha=0.5)
    ax[i].set_title("$NH_3$")

ax[0].legend(loc="upper right", framealpha=0.9, fontsize=9)
ax[0].set_ylabel("Energy rel. to reference / mHa")
fig.tight_layout()
fig_fname = "/Users/leongerard/Desktop/data_schroedinger/paper_plot/Pretraining_NH3.png"
fig.savefig(fig_fname, bbox_inches="tight")
fig.savefig(fig_fname.replace(".png", ".pdf"), bbox_inches="tight")
