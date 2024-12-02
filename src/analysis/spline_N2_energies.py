import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.interpolate
import functools

interpolate = functools.partial(scipy.interpolate.interp1d, kind='linear', bounds_error=False, fill_value='extrapolate')

df_all = pd.read_csv("/home/mscherbela/runs/references/N2/n2_ferminet.csv")

# x_interp = np.arange(1.6, 6.001, 0.01)
x_interp = np.array([1.60151171, 2.068, 2.13534895, 2.66918618, 3.20302342, 3.73686066, 4.0, 4.27069789, 4.80453513, 5.33837237])


E_nitrogen = -54.5892  # "Exact" reference from original FermiNet paper
bond_energy = 0.363801  # Well depth used in LeRoy 2006, which serves as experimental N2 reference for FermiNet paper

df_exp = df_all[df_all.method == 'expt']
df_exp.sort_values(['bond_length'], inplace=True)
# energy_offset = df_exp.energy.iloc[-1] - 2 * E_nitrogen
energy_offset = bond_energy - 2 * E_nitrogen
df_all.loc[df_all.method == 'expt', 'energy'] -= energy_offset
df_exp = df_all[df_all.method == 'expt']

exp_interp_func = interpolate(df_exp.bond_length, df_exp.energy, kind='cubic')
df_all['energy_exp_interp'] = exp_interp_func(df_all.bond_length)
df_all['delta_to_exp'] = df_all.energy - df_all.energy_exp_interp

plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(14, 10), dpi=100)

interpolated_data = []
for method in df_all.method.unique():
    df = df_all[df_all.method == method]
    # plt.figure(dpi=100)
    axes[0].plot(df.bond_length, df.energy, marker='o', label=method, markersize=2)
    axes[1].plot(df.bond_length, df.delta_to_exp * 1e3, marker='o', label=method, markersize=2)
    interpolated_data.append(dict(method=[method] * len(x_interp),
                                  bond_length=x_interp,
                                  delta_to_exp=interpolate(df.bond_length, df.delta_to_exp, )(x_interp)))

for ax in axes:
    ax.legend()
    ax.grid(alpha=0.3)
axes[0].set_ylim([df_all.energy.min() - 0.05, 2 * E_nitrogen + 0.1])
axes[1].set_ylim([-5, 65])

df_interp = pd.concat([pd.DataFrame(d) for d in interpolated_data], ignore_index=True)
df_interp['E_exp'] = exp_interp_func(df_interp.bond_length)
df_interp['energy'] = df_interp.delta_to_exp + df_interp.E_exp
df_interp.to_csv("/home/mscherbela/runs/references/N2/n2_interpolated_10points.csv", index=False)
df_all.to_csv("/home/mscherbela/runs/references/N2/n2_with_exp_reference.csv", index=False)
