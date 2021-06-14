=================================
Full documentation for developers
=================================

The DeepErwin codebase consists of several key modules:

* :mod:`~deeperwin.main` contains the core logic of our approch.
  It contains the :class:`~deeperwin.main.WaveFunction` class, which contains methods for optimization of a wavefunction,
  as well as evaluation of energies and forces. It can be executed directly by passing a configuration file as outlined in the examples.
* :mod:`~deeperwin.models` contains all neural network architectures. In particular :class:`~deeperwin.models.base.WFModel`
  defines an abstract wavefunction-model which computes :math:`\psi` from electron coordinates :math:`r`. The actual
  implementations of specific models are contained in the respective submodels. Of particular interest will be
  :class:`~deeperwin.models.DeepErwinModel.DeepErwinModel` which implements the neural network part of our model and
  :class:`~deeperwin.models.CASSCFModel.CASSCFModel` which implements the connection to established quantum chemistry methods.
* :mod:`~deeperwin.utilities` contains various utility functions to support the core code. In particular this contains the
  MCMC logic in :mod:`~deeperwin.utilities.mcmc`, learning rate schedulers and other callbacks in :mod:`~deeperwin.utilities.callbacks`,
  all configurations and defaults (such as neural network sizes, learning rates) in :mod:`~deeperwin.utilities.erwinConfiguration`, and post-processing tools.
* :mod:`~deeperwin.references` contains reference energies from other published works to allow comparison of accuracies

If you have checked out the GIT repository, you will also find these directories next to the actual deeperwin-python-package:

* **doc** containing this documentation
* **examples** containing exemplary calculations with configurations that are ready to run. These example purposefully use computationally cheap settings, so they can easily be tried on a local computer. For publishable calculations we recommend to use the (computationally more demanding) default settings.
* **tools** containing scripts to start and post-process DeepErwin on various SLURM-based clusters

.. autosummary::
   :toctree: _autosummary
   :template: custom_module.rst
   :recursive:

   deeperwin.main
   deeperwin.models
   deeperwin.utilities
   deeperwin.references