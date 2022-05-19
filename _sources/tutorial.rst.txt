==================
DeepErwin Tutorial
==================

Installation
============

DeepErwin is a python3 package and has been tested on Linux and macOS.

Installation using pip
----------------------

DeepErwin is available as a pypi package, allowing installation via:

.. code-block:: bash

    pip install deeperwin

To install from source and being able to modify the package, go to the repository root (containig the file setup.py) and install the package via:

.. code-block:: bash

    pip install -e .

Note that you need to have **python >= 3.8** and we recommend to install the source in a separate conda- or virtual-environment.

To enable GPU-support, use a pre-built jaxlib version with CUDA-support:

.. code-block:: bash

    pip install jax jaxlib==0.3.0+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_releases.html


Running a simple calculation
============================

To run a DeepErwin calculation, all configuration options must be specified in a YAML file, typically named *config.yml*.
For all options that are not specified explicitly, sensible default values will be used. The default values are defined in :~deeperwin.configuration: and a full_config.yml will also be created for each calculation listing the full configuration.

The absolute minimum that must be specified in a config-file is the physical system that one is interested in, i.e. the positions and charges of the nuclei.

.. code-block:: yaml

    physical:
        R: [[0,0,0], [3.0,0,0]]
        Z: [3, 1]


By default, DeepErwin assumes a neutral, closed shell calculation, i.e. the number of electrons equals the total charge of all nuclei, and the number of spin-up electrons is equal to the number of spin-down electrons.
For a system with an uneven number of electrons, it is assumed that there is one extra spin-up electron.
To calculate charged or spin-polarized systems, simply state the total number of electrons and the total number of spin-up electrons, e.g.

.. code-block:: yaml

    physical:
        R: [[0,0,0], [3.0,0,0]]
        Z: [3, 1]
        n_electrons: 4
        n_up: 2

Additionally, you might want to specifiy settings for the CASSCF-baseline model: The number of active electrons and active orbitals.

.. code-block:: yaml

    physical:
        R: [[0,0,0], [3.0,0,0]]
        Z: [3, 1]
        n_electrons: 4
        n_up: 2
        n_cas_electrons: 2
        n_cas_orbitals: 4

For several small molecules (e.g. H2, LiH, Ethene, first and second row elements) we have predefined their geometries and spin-settings.
Instead of setting all these parameters manually, you can just specify them using the tag :code:`physical: name`:

.. code-block:: yaml

    physical:
        name: LiH

You can also partially overwrite settings, e.g. to calculate a modified geometry of a molecule. For example to calculate a streteched LiH molecule with a bond-length of 3.5 bohr use this configuration:

.. code-block:: yaml

    physical:
        name: LiH
        R: [[0,0,0],[3.5,0,0]]

To run an actual calculation, run the python package as an executable:

.. code-block:: bash

    deeperwin config.yml

This will:

* Create a subdirectory for this specific run
* Create a full configuration, consisting of your input config-file and all relevant default parameters
* Run a calculation in the subdirectory

You can also set-up factorial sweeps of config-options, by using the -p flag.
The following call will set-up 12 subdirectories (4 molecules x 3 learning-rates) and start calculations for all of them.
If you run this on a SLURM-cluster, the jobs will not be executed directly, but instead SLURM-jobs will be submitted for parallel computation.

.. code-block:: bash

    deeperwin -p experiment_name my_sweep -p physical.name B C N O -p optimization.learning_rate 1e-3 2e-3 5e-3 config.yml

The code runs best on a GPU, but will in principle also work on a CPU. It will generate several output files, in particular containing:

* **GPU.out** containing a detailed debug log of all steps of the calculation
* **full_config.yml** containing all configuration options used for this calculation: Your provided options, as well as all default options. Take a look at this file to see all the available config options for DeepErwin
* **results.bz2** containing a compressed, pickled representation of all data (including history and model weights)


Major configuration options
===========================

To see a structure of all possible configuration options, take a look at the class :class:`~deeperwin.configuration.Configuration` which contains a full tree of all possible config options.
Alternatively you can see the full configuration tree when looking at the *full_config.yml* file that is being generated at every run.

Here are some of the most important configuration options:

============== ======================================================== ============================================================================================================================================
Category       Option                                                   Description
============== ======================================================== ============================================================================================================================================
optimization   optimizer.name                                           Type of optimizer, e.g. "adam", "rmsprop", "kfac", "slbfgs"
-------------- -------------------------------------------------------- --------------------------------------------------------------------------------------------------------------------------------------------
optimization   learning_rate                                            Initial learning-rate during optimization. May be modified during optimization by the LR-schedule (optimization.schedule).
-------------- -------------------------------------------------------- --------------------------------------------------------------------------------------------------------------------------------------------
optimization   optimizer.name                                           Type of optimizer, e.g. "adam", "rmsprop", "kfac", "slbfgs"
-------------- -------------------------------------------------------- --------------------------------------------------------------------------------------------------------------------------------------------
optimization   batch_size                                               Size of a single backprop batch. Use lower batch-size if GPU-memory is insufficient.
-------------- -------------------------------------------------------- --------------------------------------------------------------------------------------------------------------------------------------------
optimization   n_epochs                                                 Number of epochs to train the wavefunction model. In each epoch all n_walkers walkers are updated using MCMC and then optimized batch-by-batch.
-------------- -------------------------------------------------------- --------------------------------------------------------------------------------------------------------------------------------------------
optimization   shared_optimization.use                                  Boolean flag to enable interdependent optimization of multiple geometries use weight-sharing between them (disabled by default). This can significantly reduce the total number of epochs required when optimizing wavefunctions for multiple geometries.
-------------- -------------------------------------------------------- --------------------------------------------------------------------------------------------------------------------------------------------
model          baseline.n_determinants                                  Number of determinants to use for building the wavefunction
-------------- -------------------------------------------------------- --------------------------------------------------------------------------------------------------------------------------------------------
model          use_bf_shift, use_bf_factor, use_jastrow                 Boolean flags to enable/disable parts of the architecture
-------------- -------------------------------------------------------- --------------------------------------------------------------------------------------------------------------------------------------------
model          embedding.n_iterations                                   Number of iterations of the SchNet embedding
-------------- -------------------------------------------------------- --------------------------------------------------------------------------------------------------------------------------------------------
model          n_hidden_bf_factor, n_hidden_bf_shift, n_hidden_jastrow  List of integers, specifiying the number of hidden units per network layer
-------------- -------------------------------------------------------- --------------------------------------------------------------------------------------------------------------------------------------------
evaluation     n_epochs                                                 Number of evaluation steps after the wavefunction optimization
-------------- -------------------------------------------------------- --------------------------------------------------------------------------------------------------------------------------------------------
mcmc           n_walkers_opt, n_walkers_eval                            Number of MCMC-walkers to use for optimization and evaluation
-------------- -------------------------------------------------------- --------------------------------------------------------------------------------------------------------------------------------------------
logging        wandb.entity, wandb.project                              When set, this enables logging of the experiment to Weights&Biases. Set logging.wandb=None to disable W&B-logging (default).
-------------- -------------------------------------------------------- --------------------------------------------------------------------------------------------------------------------------------------------
restart        path                                                     Path to a directory containing a previously successfully finished wavefunction optimization to use as initializer for this experiment.
============== ======================================================== ============================================================================================================================================


Optimization using weight-sharing
=================================
ATTENTION: The weight-sharing technique is currenlty not supported on the master branch. A fully functioning codebase for weight-sharing
can be found under the "weight_sharing" branch.

When calculating wavefunctions for multiple related wavefunctions (e.g. for different geometries of the samemolecule), the naive approach would be to conduct independent wavefuntion optimiziations for each run.
To do this you can set *changes* to the physical-configuration, to launch multiple independent experiments with the same configuration, but different physical systems.

.. code-block:: yaml

    physical:
        name: LiH
        changes:
          - R: [[0,0,0],[3.0,0,0]]
            comment: "Equilibrium bond length"
          - R: [[0,0,0],[2.8,0,0]]
            comment: "Compressed molecule"
          - R: [[0,0,0],[3.2,0,0]]
            comment: "Stretched molecule"

As outlined in our `arxiv publication`_, the optimization can be sped-up significantly when not optimizing all geometries independently, but sharing weights between them.
This interdependent, weight-sharing optimization can be enabled be setting :code:`optimization.shared_optimization.use = True`.
To disable weight-sharing, simply set :code:`optimization.shared_optimization = None`(default).

.. code-block:: yaml

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


.. _arxiv publication: https://arxiv.org/pdf/2105.08351.pdf