import pandas as pd
import pathlib
from utils import load_from_file, morse_potential, fit_morse_potential
import matplotlib.pyplot as plt
import numpy as np

# paths = pathlib.Path("/home/mscherbela/runs/forces/LiH_sweep/").glob("*/results.bz2")
paths = pathlib.Path("/home/mscherbela/runs/forces/LiH_highres/").glob("*/results.bz2")

data = []
for p in paths:
    data_content = load_from_file(p)
    config = data_content['config']
    metrics = data_content['metrics']
    Fx0, Fx1 = 1e3*metrics['forces_mean'][0,0], 1e3*metrics['forces_mean'][1,0]
    data_dict = dict(name=p.parent.name,
                molecule=config['physical.name'],
                dist=config['physical.R'][1][0] - config['physical.R'][0][0],
                Fx0=Fx0,
                Fx1=Fx1,
                F_attractive = 0.5 * (Fx0 - Fx1),
                energy=data_content['metrics']['E_mean'])
    for k in config:
        if k.startswith('evaluation.forces'):
            data_dict[k.replace('evaluation.forces.', '')] = config[k]
    data.append(data_dict)
df_full = pd.DataFrame(data)
df_full.sort_values('dist', inplace=True)

include_in_fit = df_full['name'].str.contains('eval_highres')
morse_params_LiH = fit_morse_potential(df_full[include_in_fit].dist, df_full[include_in_fit].energy)
d_plot = np.linspace(df_full.dist.min(), df_full.dist.max())
E_morse = morse_potential(d_plot, *morse_params_LiH) * 1000
F_morse = np.gradient(E_morse) / (d_plot[1] - d_plot[0])
plt.close("all")
plt.figure(figsize=(14,7))
plt.subplot(1,2,1)
plt.scatter(df_full[include_in_fit].dist, df_full[include_in_fit].energy * 1000, alpha=0.5)
plt.scatter(df_full[~include_in_fit].dist, df_full[~include_in_fit].energy * 1000, alpha=0.5)
plt.plot(d_plot, E_morse, color='gray', lw=2)
plt.grid()

plt.subplot(1,2,2)
plt.plot(d_plot, F_morse)
plt.grid()


#%%

df1 = df_full[df_full.name.str.contains('eval_highres') & df_full.use_polynomial  & df_full.use_antithetic_sampling]
df2 = df_full[df_full.name.str.contains('eval_highres') & df_full.use_polynomial  & ~df_full.use_antithetic_sampling]
df3 = df_full[df_full.name.str.contains('eval_highres') & ~df_full.use_polynomial  & df_full.use_antithetic_sampling]
df4 = df_full[df_full.name.str.contains('eval_highres') & ~df_full.use_polynomial  & ~df_full.use_antithetic_sampling]
dfs = [df1, df2, df3, df4]
labels = ['Polynomial + Antithetic', 'Polynomial naive', 'Cutoff + Antithetic', 'Cutoff naive']

# dfs = [
# df_full[df_full['name'].str.contains('baseline') & ~df_full.use_polynomial & ~df_full.use_antithetic_sampling]]
# labels = ['Baseline cutoff naive']

def calculate_MAE(distances, forces):
    error = []
    for d,F in zip(distances, forces):
        ind = np.argmin(np.abs(d_plot - d))
        error.append(F - F_morse[ind])
    return np.mean(np.abs(error))

plt.close("all")
plt.figure(figsize=(14,7))
plt.subplot(1,2,1)
maes = []
for df, label in zip(dfs, labels):
    plt.plot(df.dist, df.F_attractive, label=label)
    maes.append(calculate_MAE(df.dist, df.F_attractive))
plt.plot(d_plot, F_morse, color='gray', lw=2)
plt.grid()
plt.legend()
plt.axhline(0, color='k')
plt.xlabel("distance")
plt.ylabel("Force in mHa/bohr")

plt.subplot(1,2,2)
plt.bar(range(len(labels)), maes)
plt.xticks(range(len(labels)), labels, fontsize=8)
plt.ylabel("MAE relative to Morse potential in mHa/bohr")


