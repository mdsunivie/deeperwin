============
Introduction
============


DeepErwin
=========

DeepErwin is python 3.8+ package that implements and optimizes TF 2.x wave function models for numerical solutions to the multi-electron Schrödinger equation.

In particular DeepErwin supports:

- Optimizing a wavefunction for a single nuclear geometry
- Optimizing wavefunctions for multiple nuclear geometries in parallel, while sharing neural network weights across these wavefunctions to speed-up optimization
- Use pre-trained weights of a network to speed-up optimization for entirely new wavefunctions


You can `download a snapshot of the code here`_ and we will also soon publish it on GitHub.
A detailed description of our method and the corresponding results can be found in our recent `arxiv publication`_. Please cite this paper, whenever you use any parts of DeepErwin.


Getting Started
===============

The quickest way to get started with DeepErwin is to have a look at our :doc:`tutorial`.
It covers installation, usage of core functionality and major configuration options.

Afterwards take a look at the comprehensive documentation of the source code and APIs: :doc:`api`

About
=====

DeepErwin is a collaborative effort of Rafael Reisenhofer, Philipp Grohs, Philipp Marquetand, Michael Scherbela, and Leon Gerard (University of Vienna).
For questions regarding this code, freel free to `reach out via e-mail`_


.. _reach out via e-mail: mailto:rafael.reisenhofer@univie.ac.at
.. _arxiv publication: https://arxiv.org/pdf/2105.08351.pdf
.. _download a snapshot of the code here: https://static.scherbela.com/deeperwin_src.zip
