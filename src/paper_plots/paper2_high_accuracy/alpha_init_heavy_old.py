import matplotlib.pyplot as plt
import pandas as pd
import re

# E_ref = -341.257800
# E_ref = -341.2588
E_ref = dict(
    Fe=-1263.655,  # Update!
    K=-599.935,
    # Fe = 0
)
# def std_err(data):
#     return np.std(data) / np.sqrt(len(data)-1)

molecule = "K"
# molecule = 'Fe'

df = pd.read_csv("/home/mscherbela/tmp/alpha_init.csv", sep=";")
df = df[(df.molecule == molecule) & (df.category == "analytical_0.004damp_4000pre_60inter")]
df["category"] = "analytical"

df_ref = pd.read_csv("/home/mscherbela/tmp/largeatom.csv", sep=";")
df_ref = df_ref[(df_ref.molecule == molecule) & (df_ref.category == "0.004damp_4000pre_60inter")]
df_ref["category"] = "constant"
df = pd.concat([df, df_ref], axis=0, ignore_index=True)

id_columns = ["molecule", "category", "name"]
columns = id_columns + [c for c in list(df) if re.match("E_smooth_\d*k", c)]
df = df[columns]
df_long = df.melt(id_vars=id_columns, var_name="n_epochs", value_name="E")
df_long["n_epochs"] = df_long["n_epochs"].apply(lambda x: int(x.split("_")[-1][:-1]))
df_long["E_ref"] = df_long.molecule.map(E_ref)
df_long["error"] = (df_long["E"] - df_long["E_ref"]) * 1e3

df_pivot = df_long.groupby(["molecule", "category", "n_epochs"]).agg({"error": ["mean", "std", "count"]}).reset_index()
df_pivot.columns = ["molecule", "category", "n_epochs", "error_mean", "error_std", "n_runs"]

plt.close("all")
for molecule in df_pivot.molecule.unique():
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=100)
    labels = dict(
        constant="Constant $\\alpha$ initialization",
        reuse="Re-used $\\alpha$ exponents",
        analytical="Analytical initialization using HF",
    )

    for category in ["constant", "reuse", "analytical"]:
        df_filt = df_pivot[(df_pivot.molecule == molecule) & (df_pivot.category == category)]
        ax.errorbar(
            df_filt.n_epochs, df_filt.error_mean, yerr=df_filt.error_std, capsize=3, label=labels[category], lw=2
        )
    ax.set_xlabel("Training epochs / k")
    ax.set_ylabel("$E - E_0$ / mHa")
    ax.grid(alpha=0.5)
    ax.legend()
    ax.set_yscale("log")
    ax.set_ylim([0.3, None])
    fig.tight_layout()
    fname = f"/home/mscherbela/ucloud/results/02_paper_figures_high_acc/alpha_init_{molecule}.png"
    fig.savefig(fname, bbox_inches="tight", dpi=600)
    fig.savefig(fname.replace(".png", ".pdf"), bbox_inches="tight")
