import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deeperwin.checkpoints import load_run
from pathlib import Path
import numpy.linalg

checkpoints = [p for p in Path("/home/mscherbela/runs/transformer/scaling").glob("*/chkpt*.zip")]

all_data = []
for i, fname in enumerate(checkpoints):
    print(f"{i+1}/{len(checkpoints)}: {fname.parent.name}")
    run = load_run(fname, parse_config=False, parse_csv=True, load_pkl=False)
    opt_t_epoch = run.history.opt_t_epoch.values[1:]

    model = run.config['model']['name']
    if model == "transformer":
        width = run.config['model']['embedding']['n_heads'] * run.config['model']['embedding']['attention_dim']
    else:
        width = run.config['model']['embedding']['n_hidden_one_el'][0]
    data = dict(n_el=run.config['physical']['n_electrons'],
                model=model,
                width=width,
                with_det=not run.config['model']['disable_determinant'],
                opt_t_epoch=np.median(opt_t_epoch)
                )
    all_data.append(data)
df_all = pd.DataFrame(all_data)
#%%
def get_scaling_law(x, y):
    x = np.stack([np.ones_like(x), np.log(x)], axis=-1)
    y = np.log(y)
    coeffs = np.linalg.lstsq(x, y, rcond=None)
    ln_prefac, exponent = coeffs[0]
    return np.exp(ln_prefac), exponent

pivot = pd.pivot(df_all, ["model", "width", "n_el"], ["with_det"], "opt_t_epoch").reset_index()
pivot = pivot.rename(columns={False: 't_embed', True: 't_total'})
pivot['t_det'] = pivot['t_total'] - pivot['t_embed']

colors = dict(dpe4="C0", transformer="C1")

plt.close("all")
fig, axes = plt.subplots(1,2, figsize=(8,5))
metric = 't_embed'
for ax, width in zip(axes, [256, 512]):
    for model in ['dpe4', 'transformer']:
        df = pivot[(pivot.model == model) & (pivot.width == width)]
        x = df.n_el
        y = df[metric]
        prefac, scaling = get_scaling_law(x, y)
        x_fit = np.linspace(min(x), max(x), 100)
        y_fit = prefac * x_fit**scaling
        color = colors[model]
        ax.plot(x, y, label=f"{model}: t ~ n**{scaling:.2f}", color=color)
        ax.plot(x_fit, y_fit, label=None, color=color, ls='--')
    ax.set_xlabel("n electrons")
    ax.set_ylabel(metric)
    ax.legend(loc='upper left')
    ax.set_title(f"Width: {width}")
    ax.grid(alpha=0.3)
fig.suptitle("Transfomers offer better scaling than MPNN", fontsize=16)
# fig.savefig("/home/mscherbela/ucloud/results/scaling_transformer_vs_dpe.png", dpi=400)

