from utils import load_from_file, get_distance_matrix
from model import build_backflow_shift, get_rbf_features
import numpy as np
import matplotlib.pyplot as plt

E_eval_tf = np.load('/users/mscherbela/runs/RegressionTestsSimpleSchnet/regSimSBe/history/mean_energies_eval.npy')

fnames = ['/users/mscherbela/runs/jaxtest/conv/test16/conv16_Be/results.bz2']
E_eval_jax = [load_from_file(f)['eval'].E_mean for f in fnames]

plt.close("all")
plt.plot(E_eval_tf, label='TF', alpha = 0.7)
plt.plot(E_eval_jax[0], label='JAX (normal)', alpha = 0.7)
# plt.plot(E_eval_jax[1], label='JAX (cauchy)', alpha = 0.7)
plt.legend()

