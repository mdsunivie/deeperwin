

import numpy as np
import matplotlib.pyplot as plt


n_features = 16


r_rbf_max = 5.0
q = np.linspace(0, 1.0, n_features)
mu = q** 2 * r_rbf_max

sigma = r_rbf_max / (n_features - 1) * (2 * q + 1 / (n_features - 1))
rbf_de_jax = lambda dist: dist[..., np.newaxis] ** 2 * np.exp(- dist[..., np.newaxis] - ((dist[..., np.newaxis] - mu) / sigma) ** 2)
rbf_de_tf = lambda dist: np.exp(- ((dist[..., np.newaxis] - mu) / sigma) ** 2)

sigma_paulinet = (1/7)*(1 + r_rbf_max*q)
rbf_paulinet = lambda dist: dist[..., np.newaxis] ** 2 * np.exp(- dist[..., np.newaxis] - ((dist[..., np.newaxis] - mu) / sigma_paulinet) ** 2)
x = np.arange(0, 1, 0.01)

fig, axes = plt.subplots(1, 3, figsize=(12, 8))

rbfs = [("tf", rbf_de_tf), ("jax", rbf_de_jax), ("paulinet", rbf_paulinet)]
for j, f in enumerate(rbfs):
    for i in range(n_features):
        axes[j].plot(x, f[1](x)[:, i], label=f[0])
    axes[j].set_title(f[0])