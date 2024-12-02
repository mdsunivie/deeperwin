

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from deeperwin.checkpoints import load_from_file, save_to_file

#%%
base_path = "/Users/leongerard/Desktop/data_schroedinger/mcmc_proposal/increasing_stepsize/"
jobs = ["mcmc_info_O_increasing_stepsize", "mcmc_info_Ne_increasing_stepsize_gpu_a40dual",
        "mcmc_info_P_increasing_stepsize_gpu_a40dual"][-1:]

electron_rs = {}
for j in jobs:
    path = os.path.join(base_path, j)
    fname = "mcmc_info.bz2"
    data = load_from_file(os.path.join(path, fname))

    electron_rs[j] = data['mcmc_r']

    del data
print("Finished loading data!")
#%%
path_increasing_stepsize = os.path.join(base_path, "numpy_increasing_stepsize")
def compute_r_deltas(r, key, one_el=False):
    r_deltas = []
    r_pos = []
    for i in range(1, len(r)):
        if one_el:
            index = (i) % r[i].shape[-2] # offset by 5 for P
            r_delta = np.linalg.norm(r[i] - r[i - 1],
                                     axis=-1)  # np.mean(np.linalg.norm(r - rs[i-1], axis=-1), axis=0)
            r_deltas.append(r_delta[:, index])
            r_norm = np.linalg.norm(r[i - 1], axis=-1)  # np.mean(np.linalg.norm(rs[i-1], axis=-1), axis=0)
            r_pos.append(r_norm[:, index])
        else:
            r_delta = np.linalg.norm(r[i] - r[i-1], axis=-1) # np.mean(np.linalg.norm(r - rs[i-1], axis=-1), axis=0)
            r_deltas.append(r_delta)
            r_norm = np.linalg.norm(r[i-1], axis=-1) # np.mean(np.linalg.norm(rs[i-1], axis=-1), axis=0)
            r_pos.append(r_norm)

    with open(os.path.join(path_increasing_stepsize, key + "_r_deltas.npy"), "wb") as f:
        np.save(f, r_deltas)
    with open(os.path.join(path_increasing_stepsize, key + "_r_pos.npy"), "wb") as f:
        np.save(f, r_pos)

for key in electron_rs:
    compute_r_deltas(electron_rs[key], key, True)

print("Finished")
del electron_rs
#%%

def get_divided_bins(r_delta, r_pos, steps=20, init_stepsize=0.04):
    stepsize = [init_stepsize * 1.25 ** k for k in range(steps)]
    r_delta_stepsize = {}
    acceptance_rate_stepsize = {}

    print(r_delta.shape)

    points = 500
    for j in range(steps):
        if j == steps-1:
            print(j, steps)
            r_d = r_delta[-499:, :]
            r_p = r_pos[-499:, :]
        else:
            r_d = r_delta[points * j:points * (j + 1), :]
            r_p = r_pos[points * j:points * (j + 1), :]
        bins = np.geomspace(0.01, 3, 20)
        r_delta_binned = []
        acceptance_rate = []

        for i, b in enumerate(bins[1:]):
            r_delta_mean = r_d[np.logical_and(bins[i] <= r_p, r_p < b)]
            #distribution.append(r_delta_mean.shape[0])
            acc = np.mean(r_delta_mean > 1e-6)
            acceptance_rate.append(acc)

            r_delta_mean = np.mean(r_delta_mean)
            r_delta_binned.append(r_delta_mean)

        r_delta_stepsize[stepsize[j]] = r_delta_binned
        acceptance_rate_stepsize[stepsize[j]] = acceptance_rate

    return acceptance_rate_stepsize, r_delta_stepsize, bins

def get_metrics(key, path, nb_steps=20, init_stepsize=0.02):

    r_delta = np.load(os.path.join(path, key + "_r_deltas.npy"))
    r_pos = np.load(os.path.join(path, key + "_r_pos.npy"))

    acceptance_rate, r_delta_binned, bins = get_divided_bins(r_delta, r_pos, nb_steps, init_stepsize)
    return acceptance_rate, r_delta_binned, bins

def opt_stepsize(r_delta_binned, init_stepsize, steps):
    stepsize = [init_stepsize * 1.25 ** k for k in range(steps)]
    max_delta_per_bin = np.argmax(np.array([r_delta_binned[key] for key in r_delta_binned]), axis=0)
    stepsize_max_delta = [stepsize[i] for i in max_delta_per_bin]
    return stepsize_max_delta


#%%
jobs = [("O", "mcmc_info_O_increasing_stepsize"), ("Ne", "mcmc_info_Ne_increasing_stepsize_gpu_a40dual"),
        ("P", "mcmc_info_P_increasing_stepsize_gpu_a40dual")]
fig, ax = plt.subplots(3, len(jobs), figsize=(12, 12))

path_increasing_stepsize = os.path.join(base_path, "numpy_increasing_stepsize")


optimal_stepsize_molecules = []
import matplotlib.colors as mcolors

overlap = list({name for name in mcolors.CSS4_COLORS
           if f'xkcd:{name}' in mcolors.XKCD_COLORS})

print(overlap)

for j, (label, job) in enumerate(jobs):
    print(f"{label}")

    acceptance_rate, r_delta_binned, bins = get_metrics(job, path_increasing_stepsize)
    bins = np.geomspace(0.01, 3, 20)[:-1]

    for h, (key, key2) in enumerate(zip(acceptance_rate, r_delta_binned)):
        ax[0][j].grid(alpha=0.5)
        ax[0][j].scatter(bins, acceptance_rate[key], label=label + f"_{key:.3f}", color=overlap[h])
        ax[0][j].set_ylabel("Mean r_delta")
        ax[0][j].set_xlabel("Bins")
        ax[0][j].set_title(f"Acceptance {label}")
        ax[0][j].legend()

        ax[1][j].grid(alpha=0.5)
        ax[1][j].scatter(bins, r_delta_binned[key2], label=label + f"_{key2:.3f}", color=overlap[h])
        ax[1][j].set_ylabel("Mean r_delta")
        ax[1][j].set_xlabel("Bins")
        ax[1][j].set_title(f"R delta {label}")

    optimal_stepsize = opt_stepsize(r_delta_binned, init_stepsize=0.02, steps=20)

    ax[-1][-1].plot(bins, optimal_stepsize, label=f"Stepsize_{label}")
    ax[-1][-1].set_title("Stepsize for max r_delta")
    ax[-1][-1].legend()
    ax[-1][-1].grid(alpha=0.5)

    optimal_stepsize_molecules.append(optimal_stepsize)

fig.tight_layout()


#%%

#%%
from sklearn.linear_model import LinearRegression
bins = np.geomspace(0.01, 3, 20)[:-1]
x = np.array([bins, bins]).reshape(-1, 1)
print(x.shape)
print(x)
y = np.array(optimal_stepsize_molecules).reshape(-1)
print(y.shape)
reg = LinearRegression().fit(x, y)

y_pred = reg.predict(bins.reshape(-1, 1))
plt.plot(y_pred)
#%%
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
print(diabetes_X.shape)
# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]
print(diabetes_X.shape)
# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
print(diabetes_X_train.shape)
# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]
print(diabetes_y_train.shape)
# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)