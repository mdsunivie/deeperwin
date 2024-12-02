from deeperwin.checkpoints import load_from_file
import matplotlib.pyplot as plt
import numpy as np

path = ["/Users/leongerard/Desktop/data_schroedinger/debug/regression_nips_fulldet_Ne_rep2/results.bz2",
     "/Users/leongerard/Desktop/data_schroedinger/debug/regression_nips_fulldet_Ne_rep1/results.bz2"]


for i, p in enumerate(path):
    data = load_from_file(p)
    metrics = data['metrics']
    del data


    plt.plot(metrics['eval_E_mean'], label=f"{np.mean(metrics['eval_E_mean'])}")
    plt.grid(alpha=0.5)

plt.legend()