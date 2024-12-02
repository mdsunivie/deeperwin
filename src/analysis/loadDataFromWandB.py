import wandb
import pandas as pd
import matplotlib.pyplot as plt

api = wandb.Api()
data = []
runs = api.runs("schroedinger_univie/model_size")
for r in runs:
    data_dict = dict(r.config)
    data_dict.update(r.summary)
    data_dict['name'] = r.name
    data.append(data_dict)
df_full = pd.DataFrame(data)

#%%
keys = {'model.embedding.net_depth': 'embed_depth',
        'model.embedding.net_width': 'embed_width',
        'model.embedding.embedding_dim': 'embedding_dim',
        'model.embedding.n_iterations': 'embed_iterations',
        'model.net_width': 'backflow_width',
        # 'model.net_depth': 'backflow_depth'
        }
df_full.sort_values([k for k in keys], inplace=True)


molecules = ['C', 'O']
limits = [2.0, 7.0]

plt.close("all")
fig, axes = plt.subplots(len(molecules), len(keys), figsize=(15, 8), sharey='row')
for ind_mol, molecule in enumerate(molecules):
    for i, (key, run_name) in enumerate(keys.items()):
        ax = axes[ind_mol][i]
        ax.grid(alpha=0.3, axis='y')
        df = df_full[(df_full.name.str.contains(run_name)) & (df_full['physical.name'] == molecule)]
        ax.bar(range(len(df)), df['error_plus_2_stdev'])
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([str(x) for x in df[key]])
        ax.set_xlabel(key)
        ax.set_ylim([0, limits[ind_mol]])
        if i == 0:
            ax.set_ylabel(molecule, fontsize=16)
    plt.suptitle("Evaluation error + 2 sigma, in mHa\nBFGS 10k opt + 4k eval")
plt.savefig('/home/mscherbela/ucloud/results/NewArch_model_size.png')

