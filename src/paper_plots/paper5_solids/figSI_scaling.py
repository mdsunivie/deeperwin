# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

fname = "plot_data/runtimes.csv"
df = pd.read_csv(fname)
df.loc[df.method == "FN", "method"] = "FermiNet"
df.loc[df.method == "TAOs_batched", "method"] = "Ours"
df = df[(df.n_el >= 8) & (df.n_el <= 50)]
nel_fit = np.arange(20, 50)


plt.close("all")
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
for method, color in zip(["FermiNet", "Ours"], ["C0", "C1"]):
    for system, marker in zip(["Hchain", "dimer"], ["o", "^"]):
        df_plot = df[(df.method == method) & (df.system == system)]

        df_fit = df_plot[df_plot.n_el.isin(nel_fit)]
        k, d = linregress(np.log(df_fit.n_el), np.log(df_fit.t))[:2]
        ax.plot(nel_fit, np.exp(d) * nel_fit**k, color=color, alpha=0.2)

        ax.plot(
            df_plot.n_el,
            df_plot.t,
            label=f"{method} ({system}): $t \\sim n^{{{k:.1f}}}$",
            color=color,
            ls="none",
            marker=marker,
        )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks([8, 12, 16, 20, 24, 28, 36, 44, 48])
ax.xaxis.set_major_formatter("{x:.0f}")
ax.set_yticks([0.2, 0.5, 1, 2, 5, 10])
ax.yaxis.set_major_formatter("{x:.1f}")
ax.minorticks_off()
ax.legend()
ax.set_xlabel("Number of electrons")
ax.set_ylabel("Timer per opt. step / s")
fig.tight_layout()
fname = "plot_output/scaling.pdf"
fig.savefig(fname, bbox_inches="tight")
fig.savefig(fname.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
