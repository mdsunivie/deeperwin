import wandb
import pandas as pd
import pickle
import gzip

def get_all_eval_energies(wandb_id, print_progress=False):
    api = wandb.Api()
    runs = api.runs(wandb_id)
    data = []
    for i,run in enumerate(runs):
        if run.state != "finished" or 'eval_E_mean' not in run.summary:
            continue
        if print_progress:
            print(f"Loading run {i}/{len(runs)}")
        data_dict = {}
        data_dict.update(run.config)
        data_dict.update(run.summary)
        data_dict['name'] = run.name

        history = run.scan_history(keys=["opt_epoch", "opt_E_std"], page_size=10000)
        for row in history:
            n_epoch = row['opt_epoch']
            data_dict[f'E_mean_eval_{n_epoch:05d}'] = row['E_intermed_eval_mean']

        n_epoch = run.config['optimization.n_epochs']
        data_dict[f'E_mean_eval_{n_epoch:05d}'] = run.summary['E_mean']
        data.append(data_dict)
    return pd.DataFrame(data)

#if __name__ == '__main__':
import matplotlib.pyplot as plt

#%%
df_full = get_all_eval_energies("leongerard/Plot_KFAC_Adam", print_progress=True)
df_full['optimizer'] = df_full['optimization.optimizer.name']
#%%
df =df_full[df_full.optimizer == 'adam']
#%%

df2 = df[['E_mean', 'physical.R']]
E_mean = list(df2['E_mean'])
R = list(df2['physical.R'])
r = zip(E_mean, R)
r_sorted = sorted(r)
x = []

for i in range(len(r_sorted)):
    x.append(float(r_sorted[i][1][2][0]) - float(r_sorted[i][1][1][0]))
x
E_mean = [r_sorted[i][0] for i in range(len(r_sorted))]
plt.close()
plt.plot(x, E_mean)
#%%
energy_columns = [c for c in list(df) if c.startswith('E_mean_eval_')]
for c in energy_columns:
    n_epoch = int(c.split('_')[-1])
    df[f'error_eval_{n_epoch:05d}'] = (df[c] - df['physical.E_ref'].astype(float)) * 1e3
error_columns = [c for c in list(df) if c.startswith('error_eval_')]

#%%
df1 = df.groupby('optimization.n_epochs_prev').agg({k:'mean' for k in error_columns})
df1 = df1.stack().reset_index()
plt.close("all")


#%%
epochs = [512, 1024, 2048, 4096, 8192, 16384]
plt.semilogx(epochs, df1.loc[:, 0], label=f'H6 Indep.: BFGS', marker='s')
epochs = [512, 1024, 2048, 4096, 8192, 16384]
error_eval = [2.8213275601757264, 2.1375674037388834, 1.5481820791439578, 1.2200597260767543, 0.9477370518898921, 0.8059120571214768]
plt.semilogx(epochs,  error_eval, label=f'H6 Independent: Adam', marker='^', color='C1')

n=[256, 512, 1024, 2048, 4096, 8192, 16384]
error=[7.57154527, 4.84058695, 3.29734728, 1.93361734, 1.3882762 ,
       0.97738367, 0.78287304]
plt.semilogx(n, error, label=f'H6 TensorFlow/arxiv Independent: Adam', marker='s', color='C4')

epochs = [512, 1024, 2048, 4096]
error_eval = [1.2074131699686017, 1.0460458657837444, 0.9698290558938946, 0.9323644307338721]
plt.semilogx(epochs, error_eval, label=f'H6 Shared: BFGS', marker='s', color='C6')

with gzip.open("/Users/leongerard/ucloud/Shared/deeperwin_datasets/processed/parallel_training.pkl.gz", "rb") as f:
    df_tf = pickle.load(f)
df_tf = df_tf[df_tf.name == 'HChain6']
df_tf = df_tf[df_tf.sharing == 'MostShared']
df_tf = df_tf.groupby('n_epochs').error_eval.mean()
plt.semilogx(df_tf.index//49, df_tf.values, label=f'H6 TensorFlow/arxiv Shared: Adam', marker='s', color='C5')


plt.axhline(1.6, label="Chemical accuracy", color='gray')
plt.legend()
plt.xlabel("Training epochs")
plt.ylabel("Error / mHa")
plt.grid(alpha=0.7, ls=':')
x_ticks = [2**n for n in range(9, 15)]
plt.gca().set_xticks(x_ticks)
plt.gca().set_xticklabels(map(str, x_ticks))
plt.minorticks_off()
plt.title("H6: 49 geometries")

d = "/Users/leongerard/Desktop/"
plt.savefig( d+"H6_Shared_Adam.png")
#%%






