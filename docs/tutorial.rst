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

Note that you need to have **python >= 3.8** and we recommend to install the source in a separate conda- or virtual-environment.
Note that the pypi package does not include the tests or the example calculations.

Installation from Source
------------------------

To install the code from source follow these steps. Note that you need to have **python >= 3.8** to run DeepErwin:

1. `Download a snapshot of our code`_ and extract it into a directory of your choice, e.g. deeperwin_src/

2. Change into the deeperwin_src/ directory, which contains the file setup.py

3. Install the package and all dependencies using :code:`pip install -e .` This will install the package in-place in "editable" mode, so that you can also modify the sources.
Major dependencies of DeepErwin are tensorflow (for neural networks), pyscf (for a basline quantum chemistry model) and ase (for postprocessing and visualization).
We recommend to use a conda- or virtual-environment to cleanly separate your python versions.
Run the following commands to install conda (if not already installed), create a new conda environment called "deeperwin" and install all dependencies.
Whenever you want to run deeperwin-code, make sure you activate the conda environment using :code:`conda activate deeperwin`

.. code-block:: bash

    sudo apt install conda
    conda create -n deeperwin python=3.8
    conda activate deeperwin
    cd /insert/path/to/deeperwin_src/
    pip install -e .


General Remarks
===============

To run a DeepErwin calculation, all configuration options must be specified in a JSON file (typically named config.in).
For all options that are not specified explicitly, sensible default values will be used.

The absolute minimum that must be specified in a config-file is the physical system that one is interested in, i.e. the positions and charges of the nuclei.

.. code-block:: json

    {
        "physical.ion_positions": [[0,0,0], [3.0,0,0]],
        "physical.ion_charges": [3, 1]
    }

By default, DeepErwin assumes a neutral, closed shell calculation, i.e. the number of electrons equals the total charge of all nuclei, and the number of spin-up electrons is equal to the number of spin-down electrons.
For a system with an uneven number of electrons, it is assumed that there is one extra spin-up electron.
To calculate charged or spin-polarized systems, simply state the total number of electrons and the total number of spin-up electrons, e.g.

.. code-block:: json

    {
        "physical.ion_positions": [[0,0,0], [3.0,0,0]],
        "physical.ion_charges": [3, 1],
        "physical.n_electrons": 4,
        "physical.n_spin_up": 2
    }

Additionally, you might want to specifiy settings for the CASSCF-baseline model: The number of active electrons and active orbitals.
If these are not specified, DeepErwin estimates the number of valence elctrons and uses this for the number of active elctrons.
To control this explicitly, include these 2 settings in config.in:

.. code-block:: json

    {
        "physical.ion_positions": [[0,0,0], [3.0,0,0]],
        "physical.ion_charges": [3, 1],
        "physical.n_electrons": 4,
        "physical.n_spin_up": 2,
        "model.slatermodel.n_active_orbitals": 10,
        "model.slatermodel.n_cas_electrons": 2
    }

For several small molecules (e.g. H2, LiH, first and second row elements) we have predefined their geometries and spin-settings.
Instead of setting all these parameters manually, you can just specify them using the tag :code:`physical.name`:

.. code-block:: json

    {
        "physical.name": "LiH"
    }


To run an actual calculation, run the python package as an executable:

.. code-block:: bash

    deeperwin config.in


The code can run both on CPUs, as well as on GPUs. It will generate several output files, in particular containing:

* **erwin.log** containing a detailed debug log of all steps of the calculation
* **config.json** containing all configuration options used for this calculation: Your provided options, as well as all default options. Take a look at this file to see all the available config options for DeepErwin
* **tb/** directory, containing files that can be inspected using the tensorflow tensorboard
* **history/** directory, containing pickled numpy arrays that contain the full optimization and evaluation history of this calculation
* Various data-files that contain the final weights of the neural network

In the following we give three minimal working examples for the different methods described in our `paper on arxiv <https://arxiv.org/pdf/2105.08351.pdf>`_.
For all examples the number of optimization and evaluation steps are deliberatly chosen very low. The examples should run on a desktop computer, and will only take a few minutes each.
To obtain publication-quality results, typically significantly more optimization steps and correspondingly longer computational time will be required.

H2: Simple wavefunction optimization
====================================

**To run this and all the following examples, please go to the corresponding examples/ directories, which are found in the root directory of the repository.**

Besides the main features of shared optimization and restarting, DeepErwin can also simply be trained on a single geometry.
An example for the H2 molecule with bond length 1.4 a.u. is given in “examples/01_H2_simple_optimization”

By calling

.. code-block:: bash

    bash h2.sh

through the command line tool in the H2 folder, a folder is created and main.py with config_h2.in is started.
To generalize it to more complicated systems like the linear H6 chain one must change the config file to

.. code-block:: json

    {
    "physical.name": "HChain6",
    "physical.ion_positions": "[[0, 0, 0], [1.4, 0, 0], [2.8, 0, 0], [4.2, 0, 0], [5.6, 0, 0], [7.0, 0, 0]]"
    }

whereas here an equidistant spacing of the H atoms with distance 1.4 a.u. is chosen.

To evaluate H2 on multiple geometries at the same time, we recommend the shared optimization technique.

H2: Shared optimization
=======================

For leveraging the shared optimization technique, the specific geometries and the shared network parts have to be specified.
An example configuration file can be found in “examples/02_H2_weight_sharing/config_h2_pt.in”.
To tell DeepErwin to use the shared optimization, you need to add :code:`"parallel_trainer.use": true` to the config file.


To set which parts of the network should be shared, pass a list of the modules you would like to share:

:code:`"parallel_trainer.shared_weights": "['embedding', 'symmetric', 'backflow_factor_general', 'backflow_shift']"`

The list can contain any subset of these modules:

    * **embedding** (SchNet version)
    * **symmetric** (Jastrow factor)
    * **backflow_factor_general** (Generalized backflow factor)
    * **backflow_factor_orbital** (Orbital specific backflow factor)
    * **backflow_shift**

To reproduce the results in our paper for 75% of shared weights, include *embedding* and *symmetric*.
To reproduce our results for 95% of weights being shared, additionally include *backflow_factor_general*, *backflow_shift*.

The set of geometries is defined as:

.. code-block:: json

    "parallel_trainer.config_changes": "[{'physical.ion_positions': [[0, 0, 0], [x, 0, 0]]} for x in [1.2, 1.4, 1.6]]"

Note that a config-file can contain arbitrary python code to construct a list of geometries on the fly. In this case it generates 3 H2 geometries, with spacings of 1.2, 1.4 and 1.6 a.u. respectively.
Run the example by calling the bash script :code:`bash h2_pt.sh`

LiH: Re-use pre-trained weights
===============================

To further investigate a Potential Energy Surface, pre-trained weights from a shared optimization can be used to speed up the optimization for unseen geometries.
This can in some instances even work across chemical space, i.e. you can pre-train a network on one molecule, and use the obtained weights to optimize a different molecule.

To generate pre-trained weights, run a shared optimization using the example :code:`bash lih_pretrain.sh`. Check out the previous section, for details of the shared optimization.


To re-use these weights, you need to specify a path to a shared optimization calculation. DeepErwin will then load this pre-trained model and use it to initialize weights.
The config-file for re-using must specify where the weights can be found, and which modules should be re-used (vs. initialized randomly):

.. code-block:: json

    "model.reuse_weights": "ReuseWeightsConfig(reuse_dirs=['/paths/to/shared/optimization/calculation/'], weights=['embedding', 'backflow_factor_general', 'backflow_shift', 'symmetric'])"

An example can be found in “examples/03_LiH_pretrain_and_reuse” and can be launched using :code:`bash lih_reuse.sh`


.. _Download a snapshot of our code: https://static.scherbela.com/deeperwin_src.zip

