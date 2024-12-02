import pathlib
import parse
import os
from analysis.parse_outputs import read_all_data
import numpy as np
import matplotlib.pyplot as plt

def get_MCMC_debug_data(dir):
    with open(os.path.join(dir, 'GPU.out')) as f:
        content = f.read()
    return [_ for _ in parse.findall("Epoch {:d} MCMC: scale = {:.f}, stepsize mean={:.f}, std={:.f}\n", content)]

def get_mean_stepsize(dir):
    data = get_MCMC_debug_data(dir)
    return np.mean([x[2] for x in data[-100:]])


def plot_value_and_std(x, y, std, axis, color, **kwargs):
    axis.plot(x, y, color=color, **kwargs)
    axis.fill_between(x, y-std, y+std, color=color, alpha=0.5)


df_full = read_all_data("/users/mscherbela/runs/jaxtest/langevin/test4/")
df_full['MCMC_mean_stepsize'] = df_full.directory.apply(get_mean_stepsize)
df_full = df_full.sort_values(['mcmc_proposal', 'mcmc_r_min', 'mcmc_langevin_scale'])

plt.close("all")
fig, axes = plt.subplots(2,2, figsize=(15,7), sharex=True)
for i,molecule in enumerate(df_full.molecule.unique()):
    df = df_full[df_full.molecule == molecule]
    df_norm = df[df.mcmc_proposal == 'normal']
    df_fixed = df[(df.mcmc_proposal == 'approx-langevin') & (df.mcmc_r_min == 1.0)]
    df_local = df[(df.mcmc_proposal == 'approx-langevin') & (df.mcmc_r_min == 0.1)]
    axes[0][i].set_title(molecule)
    assert len(df_norm) == 1
    axes[0][i].axhline(df_norm.MCMC_mean_stepsize.iloc[0], label='Normal MCMC', color='k')
    axes[0][i].plot(df_fixed.mcmc_langevin_scale, df_fixed.MCMC_mean_stepsize, label='Langevin; Fixed stepsize')
    axes[0][i].plot(df_local.mcmc_langevin_scale, df_local.MCMC_mean_stepsize, label='Langevin; Local stepsize')

    axes[1][i].axhline(df_norm.error_eval.iloc[0], label='Normal MCMC', color='k')
    plot_value_and_std(df_fixed.mcmc_langevin_scale, df_fixed.error_eval, 1e3*df_fixed.sigma_E_eval, axes[1][i], color='C0', label='Langevin; Fixed stepsize')
    plot_value_and_std(df_local.mcmc_langevin_scale, df_local.error_eval, 1e3*df_local.sigma_E_eval, axes[1][i], color='C1', label='Langevin; Local stepsize')
    axes[1][i].axhline(0, color='gray')
    axes[1][i].set_xlabel("Langevin scale")
    axes[0][i].set_ylabel("MCMC step size / bohr")
    axes[1][i].set_ylabel("Eval error / mHa")
    axes[0][i].legend()
    axes[0][i].grid(alpha=0.2)
    axes[1][i].grid(alpha=0.2)



