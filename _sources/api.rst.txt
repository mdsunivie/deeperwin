=================================
Full documentation for developers
=================================

The DeepErwin codebase consists of several key modules:

* :mod:`~deeperwin.main` is the main entry point to using the code. It sets up calculations and dispatches them to be calculated locally or on a SLURM cluster.
* :mod:`~deeperwin.process_molecule` and :mod:`~deeperwin.proces_molecules_inter` contain the main program for optimizing a single molecule, or multiple molecules in an interdependent way (e.g. by using weight-sharing).
* :mod:`~deeperwin.model` contains all neural network architectures. The contain methods to build the wavefunction which computes :math:`\psi` from electron coordinates :math:`r`.
* :mod:`~deeperwin.optimization` and :mod:`~deeperwin.evaluation` contain the respective methods to optimize and later evaluate wavefunctions. They contain imports for more advanced second-order optimizers such as KFAC or BFGS.
* :mod:`~deeperwin.hamiltonian` defines the basic Schrödinger equation, in particular defining how to compute local energies from a wavefunction using automatic differentiation.
* :mod:`~deeperwin.configuration` contains all default settings for running a DeepErwin calculation
* :mod:`~deeperwin.utils` contains various utility functions to support the core code.

If you have checked out the GIT repository, you will also find these directories next to the actual deeperwin-python-package:

* **doc** containing this documentation
* **sample_configs** containing exemplary calculations with configurations that are ready to run. These example purposefully use computationally cheap settings, so they can easily be tried on a local computer. For publishable calculations we recommend to use the (computationally more demanding) default settings.

.. autosummary::
   :toctree: _autosummary
   :template: custom_module.rst
   :recursive:

   deeperwin