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

    pip install jax jaxlib==0.1.69+cuda101 -f https://storage.googleapis.com/jax-releases/jax_releases.html


General Remarks
===============

To run a DeepErwin calculation, all configuration options must be specified in a YML file, typically named *config.yml*.
For all options that are not specified explicitly, sensible default values will be used. The default values are defined in :~deeperwin.configuration: and a full_config.yml will also be created for each calculation listing the full configuration.

The absolute minimum that must be specified in a config-file is the physical system that one is interested in, i.e. the positions and charges of the nuclei.

.. code-block:: yml

        physical:
            R: [[0,0,0], [3.0,0,0]],
            Z: [3, 1]


By default, DeepErwin assumes a neutral, closed shell calculation, i.e. the number of electrons equals the total charge of all nuclei, and the number of spin-up electrons is equal to the number of spin-down electrons.
For a system with an uneven number of electrons, it is assumed that there is one extra spin-up electron.
To calculate charged or spin-polarized systems, simply state the total number of electrons and the total number of spin-up electrons, e.g.

.. code-block:: yml

        physical:
            R: [[0,0,0], [3.0,0,0]],
            Z: [3, 1]
            n_electrons: 4
            n_up: 2

Additionally, you might want to specifiy settings for the CASSCF-baseline model: The number of active electrons and active orbitals.

.. code-block:: json

        physical:
            R: [[0,0,0], [3.0,0,0]],
            Z: [3, 1]
            n_electrons: 4
            n_up: 2
            n_cas_electrons: 2
            n_cas_orbitals: 4

For several small molecules (e.g. H2, LiH, Ethene, first and second row elements) we have predefined their geometries and spin-settings.
Instead of setting all these parameters manually, you can just specify them using the tag :code:`physical: name`:

.. code-block:: json

    physical:
        name: LiH

You can also partially overwrite settings, e.g. to calculate a modified geometry of a molecule. For example to calculate a streteched LiH molecule with a bond-length of 3.5 bohr use this configuration:

.. code-block:: json

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
* **results.bzw** containing a compressed, pickled representation of all data (including history and model weights)
