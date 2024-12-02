from utils import load_from_file, get_distance_matrix
from model import build_backflow_shift, get_rbf_features
import numpy as np
import matplotlib.pyplot as plt


tf_dir = "/users/mscherbela/runs/RegressionTestsSimpleSchnet/regSimSC/"
E_eval_tf = np.load(tf_dir + "history/mean_energies_eval.npy")
E_opt_mean_tf = np.load(tf_dir + "history/mean_energies_train.npy")
E_opt_std_tf = np.load(tf_dir + "history/std_energies_train.npy")
r_tf = np.load(tf_dir + "walkers_eval.npy")

fname = "/users/mscherbela/runs/jaxtest/conv/test10/C_10000_4000_approx-langevin/results.bz2"
# fname = '/users/mscherbela/runs/jaxtest/conv/test10/C_10000_4000_normal/results.bz2'

data = load_from_file(fname)
params = data["params"]
config = data["config"]
mcmc = data["mcmc"]
r_jax = mcmc[0]
E_eval_jax = data["E_eval_mean"]

plt.close("all")
fig, axes = plt.subplots(3, 2, figsize=(12, 9))


def plot_curve(values, color, axis, xlabel, ylabel, label, ylim=None):
    axis.plot(values, alpha=0.7, color=color, label=label)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    if ylim is not None:
        axis.set_ylim(ylim)
    axis.grid(alpha=0.5)
    axis.legend()


def plot_E_eval(E, color, axis):
    axis.plot(E, alpha=0.7, color=color)
    axis.set_xlabel("Evaluation epoch")
    axis.set_ylabel("E eval mean")
    axis.grid(alpha=0.5)


def plot_electron_coords(r, color, axis):
    for i in range(r.shape[-2]):
        axis.scatter(r[:, i, 0], r[:, i, 1], color=color, alpha=0.1)
    axis.set_xlabel("x")
    axis.set_xlabel("y")
    axis.axis("equal")


def plot_el_ion_dist_hist(r, color, axis):
    radius = np.linalg.norm(r, axis=-1).flatten()
    axis.hist(radius, bins=np.linspace(0, 4, 60), alpha=0.5, color=color)
    axis.set_xlabel("el-ion distance")
    axis.set_ylabel("Histogram count")


def plot_el_el_hist(r, color, axis):
    _, dist = get_distance_matrix(r)
    dist = dist.flatten()
    axis.hist(dist, bins=np.linspace(0, 4, 60), alpha=0.5, color=color)
    axis.set_xlabel("el-el distance")
    axis.set_ylabel("Histogram count")


for r, E_eval, E_opt_mean, E_opt_std, color, label in zip(
    [r_jax, r_tf],
    [E_eval_jax, E_eval_tf],
    [data["E_opt_mean"], E_opt_mean_tf],
    [data["E_opt_std"], E_opt_std_tf],
    ["C0", "C1"],
    ["JAX", "TF"],
):
    plot_curve(E_opt_mean, color, axes[0][0], "Optimization epoch", "E mean opt", label, ylim=[-37.9, -37.7])
    plot_curve(E_opt_std, color, axes[0][1], "Optimization epoch", "E std opt", label, ylim=[-0.1, 2.0])
    plot_curve(E_eval, color, axes[1][0], "Eval epoch", "E mean eval", label)
    plot_electron_coords(r, color, axes[1][1])
    plot_el_ion_dist_hist(r, color, axes[2][0])
    plot_el_el_hist(r, color, axes[2][1])
plt.suptitle("10k epoch optimization for Carbon: JAX vs. TF")
plt.tight_layout()
plt.savefig(
    "/users/mscherbela/ucloud/results/Carbon_TF_vs_JAX.png",
    dpi=400,
    bbox_inches="tight",
)
