import pandas as pd
import pathlib
from utils import load_from_file, morse_potential, fit_morse_potential
import matplotlib.pyplot as plt
import numpy as np

paths = pathlib.Path("/home/mscherbela/runs/forces/atoms/").glob("*/results.bz2")

data = []
for p in paths:
    data_content = load_from_file(p)
    config = data_content['config']
    metrics = data_content['metrics']
    forces_mean = 1e3*metrics['forces_mean']
    forces_std = 1e3*np.std(data_content['metrics']['forces'], axis=0)
    data_dict = dict(dirname=p.parent.name,
                molecule=config['physical.name'],
                Fx=forces_mean[0][0],
                Fy=forces_mean[0][1],
                Fz=forces_mean[0][2],
                FxStd=forces_std[0][0],
                FyStd=forces_std[0][1],
                FzStd=forces_std[0][2],
                energy=data_content['metrics']['E_mean'])
    for k in config:
        if k.startswith('evaluation.forces'):
            data_dict[k.replace('evaluation.forces.', '')] = config[k]
    data.append(data_dict)
df_full = pd.DataFrame(data)
df_full.sort_values(['R_cut', 'R_core'], inplace=True)
df_full['error'] = np.sqrt(df_full['Fx']**2 + df_full['Fy']**2 + df_full['Fz']**2)

#%%
df = df_full[df_full.dirname.str.contains('reeval_params')]
pivot = pd.pivot(df, 'R_cut', 'R_core', 'error')
print(pivot)

plt.close("all")
for R_cut in pivot.index:
    r = pivot.loc[R_cut]
    plt.semilogy(r.index, list(r), label=f'R_cut = {R_cut:.2f}')
plt.legend()
plt.grid()
plt.xlabel("R_core")
plt.ylabel("Force error / mHa/bohr")





