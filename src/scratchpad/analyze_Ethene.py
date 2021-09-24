import pandas as pd
import wandb
import matplotlib.pyplot as plt
import numpy as np

api = wandb.Api()
runs = api.runs("schroedinger_univie/develop")

data = []
for r in runs:
    name = r.name
    if not name.startswith("largemol"):
        continue
    if not r.state == 'finished':
        continue
    if not "Ethene" in r.config['physical.name']:
        continue
    data.append(dict(
        geom="TransitionState" if "Barrier" in r.config['physical.name'] else "GroundState",
        opt=r.config['optimization.optimizer.name'],
        eval_E_mean=r.summary['E_mean'] * 1000,
        eval_E_sigma = r.summary['E_mean_sigma'] * 1000,
        E_cas = r.summary['E_casscf'] * 1000,
        name=r.name
    ))
df = pd.DataFrame(data)
df = df.sort_values("eval_E_mean", ascending=False)
#%%
E_ref = df.eval_E_mean.min()
plt.close("all")
for i,r in df.iterrows():
    mean, sigma = r.eval_E_mean - E_ref, r.eval_E_sigma
    color = f'C{i}'
    plt.axhline(mean, label=r['name'], color=color)
    plt.fill_between([0,1], np.array([1,1]) * (mean+sigma), np.array([1,1]) * (mean-sigma), color=color, alpha=0.4)
plt.legend()
plt.grid()

KCAL_PER_MOL = 0.6275
df2 = df.set_index('name')
barrier_adam = df2.loc['largemol_adam_EtheneBarrier'].eval_E_mean - df2.loc['largemol_adam_Ethene'].eval_E_mean
barrier_bfgs = df2.loc['largemol_bfgs_EtheneBarrier'].eval_E_mean - df2.loc['largemol_bfgs2_Ethene'].eval_E_mean
barrier_cas = df2.loc['largemol_bfgs_EtheneBarrier'].E_cas - df2.loc['largemol_bfgs2_Ethene'].E_cas
print(f"Barrier Adam: {barrier_adam:.1f} mHa = {barrier_adam*KCAL_PER_MOL:.1f} kcal/mol ")
print(f"Barrier BFGS: {barrier_bfgs:.1f} mHa = {barrier_bfgs*KCAL_PER_MOL:.1f} kcal/mol ")
print(f"Barrier CAS : {barrier_cas:.1f} mHa = {barrier_cas*KCAL_PER_MOL:.1f} kcal/mol ")






