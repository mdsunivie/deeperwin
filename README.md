# DeepErwin

DeepErwin is python package that implements and optimizes wave function models for numerical solutions to the multi-electron Schrödinger equation.

DeepErwin is based on JAX and supports:
- Optimizing a wavefunction for a single nuclear geometry
- Optimizing wavefunctions for multiple nuclear geometries at once, while sharing neural network weights across these wavefunctions to speed-up optimization
- Using pre-trained weights of a network to speed-up optimization for entirely new wavefunctions
- Using second-order optimizers such as KFAC 

A detailed description of our method and the corresponding results can be found in our publications: 

[Solving the electronic Schrödinger equation for multiple nuclear geometries with weight-sharing deep neural networks](https://www.nature.com/articles/s43588-022-00228-x) \
Scherbela, M., Reisenhofer, R., Gerard, L. et al. Published in: Nat Comput Sci 2, 331–341 (2022). \
Code version: [arxiv_2105.08351v2](https://github.com/mdsunivie/deeperwin/releases/tag/arxiv_2105.08351v2)

[Gold-standard solutions to the Schrödinger equation using deep learning: How much physics do we need?](https://proceedings.neurips.cc/paper_files/paper/2022/hash/430894999584d0bd358611e2ecf00b15-Abstract-Conference.html) \
Gerard, L., Scherbela, M., et al. Published in: Advances in Neural Information Processing Systems (2022). \
Code version: [arxiv_2205.09438v2](https://github.com/mdsunivie/deeperwin/releases/tag/arxiv_2205.09438v2)

[Towards a Foundation Model for Neural Network Wavefunctions](https://www.nature.com/articles/s41467-023-44216-9) \
Scherbela, M., Gerard, L., and Grohs, P. \
Code version: [Transferable atomic orbitals](https://github.com/mdsunivie/deeperwin/releases/tag/transferable_atomic_orbitals)

[Variational Monte Carlo on a Budget — Fine-tuning pre-trained Neural Wavefunctions](https://papers.nips.cc/paper_files/paper/2023/hash/4b5721f7fcc1672930d860e0dfcfee84-Abstract-Conference.html) \
Scherbela, M., Gerard, L., and Grohs, P. \
Code version: [Transferable atomic orbitals](https://github.com/mdsunivie/deeperwin/releases/tag/transferable_atomic_orbitals)

[Transferable Neural Wavefunctions for Solids](https://arxiv.org/abs/2405.07599) \
Gerard, L., Scherbela, M., Sutterud, H., Foulkes, M. and Grohs, P.

Please cite the respective publication when using our codebase. 

On [figshare](https://figshare.com/articles/online_resource/Pre-trained_neural_wavefunction_checkpoints_for_the_GitHub_codebase_DeepErwin/23585358/1) we store checkpoints for:
1. A pre-trained PhisNet reimplementation to generate orbital descriptors for a neural wavefunction.
2. A pre-trained neural wavefunction on a dataset of 18 compounds with Hartree-Fock orbital descriptors.
3. A pre-trained neural wavefunction on a dataset of 98 compounds with PhisNet orbital descriptors. 

To use the checkpoints please checkout the code version [Transferable atomic orbitals](https://github.com/mdsunivie/deeperwin/releases/tag/transferable_atomic_orbitals).

# Quick overview

## Installation

DeepErwin is a python3 package and has been tested on Ubuntu and macOS.
To get the most up-to-date version of the code, we recommend to checkout our repository from github:
https://github.com/mdsunivie/deeperwin

To install deeperwin and all its dependencies after you cloned our codebase:
```bash
    pip install -e .
```
To install the kfac fork we are using:
```bash
    git checkout master
    git pull origin master
    git submodule init
    git submodule update
    cd kfac_jax
    pip install -e .
```
This will install the repository "in-place", so you can make changes to the source code without having to reinstall the package.
If you need CUDA support to run the JAX code on GPUs (recommended), additionally install the prepackaged jax[cuda] wheel:
```bash
    pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Running a simple calculation


To run a DeepErwin calculation, all configuration options must be specified in a YAML file, typically named *config.yml*.
For all options that are not specified explicitly, sensible default values will be used. The default values are defined in :~deeperwin.configuration: and a full_config.yml will also be created for each calculation listing the full configuration.

The absolute minimum that must be specified in a config-file is the physical system that one is interested in, i.e. the positions and charges of the nuclei.

```yaml
    physical:
        R: [[0,0,0], [3.0,0,0]]
        Z: [3, 1]
```


By default, DeepErwin assumes a neutral, closed shell calculation, i.e. the number of electrons equals the total charge of all nuclei, and the number of spin-up electrons is equal to the number of spin-down electrons.
For a system with an uneven number of electrons, it is assumed that there is one extra spin-up electron.
To calculate charged or spin-polarized systems, simply state the total number of electrons and the total number of spin-up electrons, e.g.

```yaml
    physical:
        R: [[0,0,0], [3.0,0,0]]
        Z: [3, 1]
        n_electrons: 4
        n_up: 2
```

Additionally, you might want to specifiy settings for the CASSCF-baseline model: The number of active electrons and active orbitals.

```yaml
    physical:
        R: [[0,0,0], [3.0,0,0]]
        Z: [3, 1]
        n_electrons: 4
        n_up: 2
        n_cas_electrons: 2
        n_cas_orbitals: 4
```

For several small molecules (e.g. H2, LiH, Ethene, first and second row elements) we have predefined their geometries and spin-settings.
Instead of setting all these parameters manually, you can just specify them using the tag :code:`physical: name`:

```yaml
    physical:
        name: LiH
```

You can also partially overwrite settings, e.g. to calculate a modified geometry of a molecule. For example to calculate a streteched LiH molecule with a bond-length of 3.5 bohr and 5000 optimization steps use this configuration:

```yaml
    physical:
        name: LiH
        R: [[0,0,0],[3.5,0,0]]
    optimization:
        n_epochs: 5000
```

To run an actual calculation, run the python package as an executable:

```bash
    deeperwin run config.yml
```

This will combine your supplied configuration with default values for all other settings and dump it as *full_config.yml*. It will then run a calculation in the current directory, writing its output to the standard output and logfile.
You can also set-up factorial sweeps of config-options, by using ```deeperwin setup``` with the -p flag.
The following call will set-up 12 subdirectories (4 molecules x 3 learning-rates) and start calculations for all of them.
If you run this on a SLURM-cluster, the jobs will not be executed directly, but instead SLURM-jobs will be submitted for parallel computation.

```bash
    deeperwin setup -p experiment_name my_sweep -p physical.name B C N O -p optimization.optimizer.learning_rate 1e-3 2e-3 5e-3 -i config.yml
```

The code runs best on a GPU, but will in principle also work on a CPU. It will generate several output files, in particular containing:

* **GPU.out** containing a detailed debug log of all steps of the calculation
* **full_config.yml** containing all configuration options used for this calculation: Your provided options, as well as all default options. Take a look at this file to see all the available config options for DeepErwin
* **checkpoint** files containing a compressed, pickled representation of all data (including history and model weights)
  
A single run for LiH and 5000 steps should take less than an hour on a single GPU.


## Major configuration options


To see a structure of all possible configuration options, take a look at the class ```deeperwin.configuration.Configuration``` which contains a full tree of all possible config options.
Alternatively you can see the full configuration tree when looking at the *full_config.yml* file that is being generated at every run.


## Optimization using weight-sharing
 
When calculating wavefunctions for multiple related wavefunctions (e.g. for different geometries of the same molecule or even of different molecules), the naive approach would be to conduct independent wavefuntion optimiziations for each run.
Another approach with a potential speed-up and a generalized wavefunction across compounds is to use the so called TAOs (transferable atomic orbitals), see also `arxiv publication`_.

Therefore you need to specify multiple geometries in the physical config, choose the shared optimization flag and use in the model settings TAOs:

```yaml
    physical:
        name: LiH
        changes:
          - R: [[0,0,0],[3.0,0,0]]
            comment: "Equilibrium bond length"
          - R: [[0,0,0],[2.8,0,0]]
            comment: "Compressed molecule"
          - R: [[0,0,0],[3.2,0,0]]
            comment: "Stretched molecule"
    optimization:
        shared_optimization:
            use: True

    model:
      orbitals:
        generalized_atomic_orbitals:
          atom_types: [1, 6, 7, 8]
          basis_set: "STO-6G"
          envelope_width: 128
          backflow_width: 256
          orb_feature_gnn:
            n_iterations: 2
          phisnet_model:
```

A complete config can be found in the folder sample_configs/pre_trained_basemodel/config_bm_hfcoeff.yml. Here,
a wavefunction
is optimized across 18 compounds with in total 360 geometries using Hartree Fock orbital descriptors as explained in our [arxiv publication](https://arxiv.org/abs/2303.09949). Pre-trained neural network weights (and the corresponding checkpoint) can be found at [https://doi.org/10.6084/m9.figshare.23585358.v1](https://doi.org/10.6084/m9.figshare.23585358.v1).
For an example to reuse pre-trained model weights see also sample_configs/finetuning/config_reuse_from_basemodel_template.yml
and the corresponding setup_finetuning_exp.py file.

## Datasets and Geometries

To handle large pre-training molecule datasets for the base model (as in [https://arxiv.org/abs/2303.09949](https://arxiv.org/abs/2303.09949)) we have a geometry database.
It stores geometries of various molecules
and groups them in datasets (cf. folder: datasets/db/datasets.json or datasets/db/geometries.json). Each geometry has a
unique hash and each dataset has a unique name. Instead of defining molecules by name one can also use:



```yaml
    physical: eeed25f9e4dc8b8b44c0b8245cf1210c
```

or for a whole group of molecules:

```yaml
    physical: TinyMol_CNO_rot_dist_train_42compounds_10geoms_no_overlap_qm7
```

This can be useful when pre-training a wavefunction across hundreds of geometries of various compounds, preventing the need to define each geometry manually in a yaml file as it was done in the section "Optimization using weight-sharing".
We have gathered additional example configs in the folder sample_configs.


# About

DeepErwin is a collaborative effort of Michael Scherbela, Leon Gerard, Rafael Reisenhofer, Philipp Marquetand, and Philipp Grohs.\
The code was written by Michael Scherbela, Leon Gerard, and Rafael Reisenhofer.\
If you have any questions, freel free to reach out via [e-mail](mailto:deeperwin.datascience@univie.ac.at).

