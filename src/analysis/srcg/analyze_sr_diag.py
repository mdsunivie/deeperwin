#%%
import numpy as np
import matplotlib.pyplot as plt
import jax
import pandas as pd
import seaborn as sns

def flatten_dict(d, parent_key="", sep="/"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

fname = "/home/mscherbela/runs/srcg/grads.npy"
full_data = np.load(fname, allow_pickle=True).item()

# checkpoint = "/storage/scherbelam20/runs/srcg/srdiag/srcg_nc_dpe4_checkpoints/chkpt003000.zip"
all_data = []
for checkpoint in full_data:
    n_steps = int(checkpoint.split("/")[-1].replace("chkpt", "").replace(".zip", ""))
    data = full_data[checkpoint]
    g_squared = flatten_dict(data["grads_squared"])
    g_mean = flatten_dict(data["grads_mean"])

    variance = jax.tree_util.tree_map(lambda s, m: s - m**2, g_squared, g_mean)
    mean_variance_per_param = jax.tree_util.tree_map(lambda v: np.mean(v), variance)
    max_variance_per_param = jax.tree_util.tree_map(lambda v: np.max(v), variance)
    min_variance_per_param = jax.tree_util.tree_map(lambda v: np.min(v), variance)

    for k in mean_variance_per_param:
        var_mean = mean_variance_per_param[k]
        var_max = max_variance_per_param[k]
        var_min = min_variance_per_param[k]
        print(f"{k:100}: {var_mean*1000:6.2f}, [{var_min*1000:6.2f},  {var_max*1000:6.2f}]")
        all_data.append(dict(n_steps=n_steps, 
                             param=k,
                             var_mean=mean_variance_per_param[k], 
                             var_min=max_variance_per_param[k], 
                             var_max=min_variance_per_param[k],
                             g_squared_mean=np.mean(g_squared[k]),
                             g_mean_squared=np.mean(g_mean[k]**2),
                             ))
df = pd.DataFrame(all_data)

#%%
plt.close("all")
fig, axes = plt.subplots(2,2, figsize=(16,9))
for ax, metric in zip(axes.flatten()[:3], ["var_mean", "var_min", "var_max"]):
    sns.lineplot(data=df, 
                x="n_steps", 
                y=metric, 
                hue="param",
                palette="tab20",
                ax=ax,
                legend=False)
    ax.set_yscale("log")
    
sns.scatterplot(data=df, 
             x="g_mean_squared", 
             y="var_mean", 
             hue="n_steps",
             ax=axes[1],
             legend=True)
axes[1].set_xscale("log")
axes[1].set_yscale("log")



# %%
