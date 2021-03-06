==================
DeepErwin Tutorial
==================

Installation
============

DeepErwin is a python3 package and has been tested on Linux and macOS.

Installation from source
------------

To get the most up-to-date version of the code, we recommend to checkout our repository from github:
https://github.com/mdsunivie/deeperwin

To install deeperwin and all its dependencies, go to the downloaded directory and run

.. code-block:: bash

    pip install -e .

This will install the repository "in-place", so you can make changes to the source code without having to reinstall the package.
If you need CUDA support to run the JAX code on GPUs (recommended), additionally install the prepackaged jax[cuda] wheel:

.. code-block:: bash

    pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Installation using pip
----------------------

DeepErwin is also available as a pypi package, however **note that we may not always have the latest version of our code on pypi**:

.. code-block:: bash

    pip install deeperwin

To install from source and being able to modify the package, go to the repository root (containig the file setup.py) and install the package via:

.. code-block:: bash

    pip install -e .

Note that you need to have **python >= 3.8** and we recommend to install the source in a separate conda- or virtual-environment.


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

    deeperwin run config.yml

This will combine your supplied configuration with default values for all other settings and dump it as *full_config.yml*. It will then run a calculation in the current directory, writing its output to the standard output and logfile.

You can also set-up factorial sweeps of config-options, by using ```deeperwin setup``` with the -p flag.
The following call will set-up 12 subdirectories (4 molecules x 3 learning-rates) and start calculations for all of them.
If you run this on a SLURM-cluster, the jobs will not be executed directly, but instead SLURM-jobs will be submitted for parallel computation.

.. code-block:: bash

    deeperwin setup -p experiment_name my_sweep -p physical.name B C N O -p optimization.learning_rate 1e-3 2e-3 5e-3 config.yml

The code runs best on a GPU, but will in principle also work on a CPU. It will generate several output files, in particular containing:

* **GPU.out** containing a detailed debug log of all steps of the calculation
* **full_config.yml** containing all configuration options used for this calculation: Your provided options, as well as all default options. Take a look at this file to see all the available config options for DeepErwin
* **checkpoint** files containing a compressed, pickled representation of all data (including history and model weights)


Major configuration options
===========================

To see a structure of all possible configuration options, take a look at the class :class:`~deeperwin.configuration.Configuration` which contains a full tree of all possible config options.
Alternatively you can see the full configuration tree when looking at the *full_config.yml* file that is being generated at every run.


Here are some of the most important configuration options:

.. csv-table:: Major configuration options
   :file: major_config_options.csv
   :widths: 15, 25, 60
   :header-rows: 1


Optimization using weight-sharing
=================================
**ATTENTION**: The weight-sharing technique is currently not supported on the master branch. A fully functioning codebase for weight-sharing
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