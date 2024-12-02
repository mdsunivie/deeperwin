from deeperwin.checkpoints import load_from_file
import numpy as np
import matplotlib.pyplot as plt

file = "/Users/leongerard/Desktop/data_schroedinger/mcmc_info_P/mcmc_info.bz2"
df = load_from_file(file)


#%%
df
#%%
eval = df['metrics']['eval_E_mean']

intervall = 500
nb_split = len(eval)//intervall

for i in range(nb_split):
    eval_intervall = eval[i*intervall: (i+1)*intervall]

    print((-df['metrics']['E_ref'] + np.mean(eval_intervall))*1000, np.std(eval_intervall))
error = (-df['metrics']['E_ref'] + eval_intervall)
plt.hist(error, color = f"C{i}", bins=200)

