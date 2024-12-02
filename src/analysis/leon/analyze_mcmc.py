

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from deeperwin.checkpoints import load_from_file, save_to_file

#%%
#path = "/Users/leongerard/Desktop/data_schroedinger/mcmc_info_P"
#path = "/Users/leongerard/Desktop/data_schroedinger/mcmc_info_P_local_mcmc_v2"

# paths = ["/Users/leongerard/Desktop/data_schroedinger/mcmc_proposal/P/mcmc_info_P_comp_proposal_local",
#          "/Users/leongerard/Desktop/data_schroedinger/mcmc_proposal/P/mcmc_info_P_comp_proposal_normal_one_el",
#          "/Users/leongerard/Desktop/data_schroedinger/mcmc_proposal/P/mcmc_info_P_comp_proposal_normal",
#          "/Users/leongerard/Desktop/data_schroedinger/mcmc_proposal/P/mcmc_info_P_comp_proposal_v2_local_one_el"][-1:]

paths = ["/Users/leongerard/Desktop/data_schroedinger/mcmc_proposal/closed_shell/mcmc_info_Ne_comp_proposal_local",
         "/Users/leongerard/Desktop/data_schroedinger/mcmc_proposal/closed_shell/mcmc_info_Ne_comp_proposal_normal_one_el",
         "/Users/leongerard/Desktop/data_schroedinger/mcmc_proposal/closed_shell/mcmc_info_Ne_comp_proposal_normal",
         "/Users/leongerard/Desktop/data_schroedinger/mcmc_proposal/closed_shell/mcmc_info_Ne_comp_proposal_local_one_el"][-1:]
rs = []
log_psis = []
for p in paths:
    fname = "mcmc_info.bz2"
    data = load_from_file(os.path.join(p, fname))

    r = data['mcmc_r']#[:2000]
    print(len(r))
    log_psi = data['log_psi']#[:2000]
    rs.append(r)
    log_psis.append(log_psi)
    del data
#save_to_file("mcmc_info_small.bz2", **{'mcmc_r': r, 'log_psi': log_psi})

#%%

def compute_r_deltas(r, one_el=False):
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


    r_deltas = np.array(r_deltas[1:]).reshape((-1))# first delta order of magnitude larger (?)
    r_pos = np.array(r_pos[1:]).reshape((-1))
    r_array = np.stack([r_deltas, r_pos], axis=-1)

    return r_array



def get_binned_metrics(r_array):
    bins = np.geomspace(0.01, 3, 20)
    r_delta_binned = []
    acceptance_rate = []
    distribution = []
    for i, b in enumerate(bins[1:]):
        r_delta_mean = r_array[np.logical_and(bins[i] <= r_array[:, 1], r_array[:, 1] < b)]
        distribution.append(r_delta_mean.shape[0])
        acc = np.mean(r_delta_mean[:, 0] > 1e-6)
        acceptance_rate.append(acc)

        r_delta_mean = np.mean(r_delta_mean[:, 0])
        r_delta_binned.append(r_delta_mean)

    return acceptance_rate, r_delta_binned, bins, distribution

def get_divided_bins(r_delta, r_pos, steps=10):
    stepsize = [0.04 * 1.25 ** k for k in range(steps)]
    r_delta_stepsize = {}
    acceptance_rate_stepsize = {}

    for j in range(steps):
        r_d = r_delta[1000*j:1000*(j+1), :]
        r_p = r_pos[1000*j:1000*(j+1), :]

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

    return acceptance_rate_stepsize, r_delta_stepsize, bins, []

def get_metrics(r, one_el = False):
    r_array = compute_r_deltas(r, one_el)
    print(r_array.shape)
    acceptance_rate, r_delta_binned, bins, distribution = get_binned_metrics(r_array)

    return acceptance_rate, r_delta_binned, bins, distribution

def get_from_numpy(label, path):
    acceptance_rate = np.load(path + f"acc_rate_{label}.npy")
    r_delta_binned = np.load(path + f"r_delta_binned_{label}.npy")
    print(f"finished with {label}")
    return acceptance_rate, r_delta_binned



#%%

fig, ax = plt.subplots(3, 1, figsize=(12, 12))
labels = ["local", "normal", "local_one_el", "normal_one_el", "normal_one_el_10k"][:-1]

path = "/Users/leongerard/Desktop/data_schroedinger/mcmc_proposal/numpy_data_Ne/"

for j, label in enumerate(labels):
    print(f"{labels[j]}")

    one_el = True if "one_el" in label else False
    # #acceptance_rate, r_delta_binned, bins, distribution = get_metrics(rs[j], one_el)
    if label == "local_one_el":
        acceptance_rate, r_delta_binned, bins, distribution = get_metrics(rs[0], True)
        bins = np.geomspace(0.01, 3, 20)[:-1]
        print(type(acceptance_rate))
    else:
        bins = np.geomspace(0.01, 3, 20)[:-1]
        acceptance_rate, r_delta_binned = get_from_numpy(label, path)
    #acceptance_rate, r_delta_binned = acceptance_rate[:skip], r_delta_binned[:skip]
    if type(acceptance_rate) == dict:
        for h, (key, key2) in enumerate(zip(acceptance_rate, r_delta_binned)):
            ax[0].grid(alpha=0.5)
            ax[0].scatter(bins, acceptance_rate[key], label=label + f"_{key}", color=f"C{h}")
            ax[0].set_ylabel("Mean r_delta")
            ax[0].set_xlabel("Bins")
            #ax[0].set_xticks(bins)
            # ax[0].set_xtickslabels(bins)
            ax[0].set_title("Acceptance")

            ax[1].grid(alpha=0.5)
            ax[1].scatter(bins, r_delta_binned[key2], label=label + f"_{key2}", color=f"C{h}")
            ax[1].set_ylabel("Mean r_delta")
            ax[1].set_xlabel("Bins")
            ax[1].set_title("R delta")

        stepsize = [0.04 * 1.25 ** k for k in range(10)]
        r_delta_list = np.array([r_delta_binned[key] for key in r_delta_binned])
        max_delta_per_bin = np.argmax(r_delta_list, axis=0)
        stepsize_max_delta = [stepsize[i] for i in max_delta_per_bin]

        ax[2].plot(bins, stepsize_max_delta, label="Stepsize")
        ax[2].set_title("Stepsize for max r_delta")
        ax[2].legend()
        ax[2].grid(alpha=0.5)

    else:

        ax[0].grid(alpha=0.5)
        ax[0].scatter(bins, acceptance_rate, label=label, color=f"C{j}")
        ax[0].set_ylabel("Mean r_delta")
        ax[0].set_xlabel("Bins")
        #ax[0].set_xticks(bins)
        #ax[0].set_xtickslabels(bins)
        ax[0].set_title("Acceptance")

        ax[1].grid(alpha=0.5)
        ax[1].scatter(bins, r_delta_binned, label=label, color=f"C{j}")
        ax[1].set_ylabel("Mean r_delta")
        ax[1].set_xlabel("Bins")
        ax[1].set_title("R delta")

    # ax[2+j].bar(bins[:-1], distribution, label=labels[j], color=f"C{j}")
    # ax[2+j].set_title("Distribution")

ax[1].legend()
fig.tight_layout()



#%%


print(stepsize_max_delta)
print(max_delta_per_bin)
print(r_delta_list.shape)

plt.plot(stepsize_max_delta)
#%%

p = "/Users/leongerard/Desktop/data_schroedinger/mcmc_proposal/numpy_data_Ne/"

for j, r in enumerate(rs):
    one_el = True if labels[j] == "normal_one_el" else False
    acceptance_rate, r_delta_binned, bins, distribution = get_metrics(r, one_el)
    with open(p + f"acc_rate_{labels[j]}.npy", "wb") as f:
        np.save(f, acceptance_rate)
    with open(p + f"r_delta_binned_{labels[j]}.npy", "wb") as f:
        np.save(f, r_delta_binned)