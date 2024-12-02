# %%
from deeperwin.run_tools.geometry_database import load_geometries, load_datasets
import ase
import ase.visualize
import numpy as np


def align_to_main_axes(R):
    _, _, U = np.linalg.svd(R)
    return R @ U.T


all_datasets = load_datasets()
all_geometries = load_geometries()

geometries = []
datasets = [
    "Kinal_Piecuch_Bicyclobutane",
]
names = ["train", "test_in_distr", "test_out_of_distr"]
output_dir = "/home/mscherbela/ucloud/results/04_paper_better_universal_wf/figures/renders"

geometries = []
for d in datasets:
    geometries += all_datasets[d].get_geometries(all_geometries, all_datasets)

# for g in geometries:
#     g.R = align_to_main_axes(g.R)

ase.visualize.view([g.as_ase() for g in geometries])


# %%
