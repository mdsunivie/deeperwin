#!/usr/bin/env python

"""
Script to reuse pre-trained base model with pre-trained PhisNet model for fine-tuning experiments.
It currently allows only geometries and molecules from the database (see folder: datasets/db/geomtries.json).
To use a new molecule, you have two options:
1. Add a geometry to the database and use the corresponding hash
2. Use the "config_reuse_from_basemodel_template.yml": Define the physical section (with the new molecule; see e.g.
config_dpe.yml), the model section (should be the same the pre-trained model used) and the set the paths to the
checkpoints (pre-trained base model and pre-trained PhisNet)
"""

import os.path
from deeperwin.cli import main
from deeperwin.run_tools.geometry_database import load_geometries, load_datasets
from deeperwin.checkpoints import load_run
import ruamel.yaml

### Dataset: Choose the dataset or geometry (hash) you want to compute
dataset = "QM7_large_scale_exp_es"
# dataset = "b2c66515e32b29b4d09cfa60705cd14c" # Hash for a specific C3H4 geometry


### Checkpoints to reuse
# 1. Checkpoint for pre-trained base model
checkpoint = "midimol_2023-05-01_699torsion_nc_by_std_256k.zip"

# 2. Checkpoint for pre-trained PhisNet model to use ml generated orb. descriptors
phisnet_checkpoint = "phisnet_3LayerL2_47kGeoms_174Epochs.zip"

### Constants
reuse_config_fname = "config_reuse_from_basemodel_template.yml"
calc_name = checkpoint.split("/")[-1].replace(".zip", "")
calc_dir = "reuse_exp_" + calc_name

### Create directory for reusing from one specific checkpoint
if not os.path.isdir(calc_dir):
    os.mkdir(calc_dir)
os.chdir(calc_dir)

### Load model config from checkpoint and use it to replace the reuse model config
with open("../" + reuse_config_fname) as f:
    reuse_config = ruamel.yaml.YAML().load(f)
checkpoint_config = load_run(checkpoint, load_pkl=False, parse_config=False).config
reuse_config["model"] = checkpoint_config["model"]
reuse_config["reuse"]["path"] = checkpoint
reuse_config["reuse"]["path_phisnet"] = phisnet_checkpoint
reuse_config["experiment_name"] = calc_dir

### Get all geometries for single points
all_geometries = load_geometries()
all_datasets = load_datasets()
if dataset in all_geometries:
    geometry_hashes = [dataset]
else:
    geometry_hashes = all_datasets[dataset].get_hashes()

### Submit the actual calculations
for geom_hash in geometry_hashes:
    geom = all_geometries[geom_hash]
    n_heavy = geom.n_heavy_atoms
    reuse_config["experiment_name"] = calc_dir

    # Write config and submit job
    with open("config.yml", "w") as f:
        ruamel.yaml.YAML().dump(reuse_config, f)
    print(f"Generating job for {geom_hash}")
    cmd = "setup -i config.yml"
    cmd += f" -p physical {geom_hash}"
    main(cmd)
os.chdir("../tested")
