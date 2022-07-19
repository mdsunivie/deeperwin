============
Introduction
============


DeepErwin
=========

DeepErwin is python 3.8+ package that implements and optimizes wave function models for numerical solutions to the multi-electron Schrödinger equation.
DeepErwin is built on JAX, allowing to define wavefunctions using the familiar numpy-syntax and compiling to highly-performant GPU-models.

In particular DeepErwin supports:

- Optimizing a wavefunction for a single nuclear geometry
- Optimizing wavefunctions for multiple nuclear geometries in parallel, while sharing neural network weights across these wavefunctions to speed-up optimization
- Using pre-trained weights of a network to speed-up optimization for entirely new wavefunctions
- Using second-order optimizers such as KFAC or BFGS

A detailed description of our method and the corresponding results can be found in our publications:
 - `Gold-standard solutions to the Schrödinger equation using deep learning: How much physics do we need? <https://arxiv.org/abs/2205.09438>`_
 - `Solving the electronic Schrödinger equation for multiple nuclear geometries with weight-sharing deep neural networks <https://doi.org/10.1038/s43588-022-00228-x>`_

Please cite the corresponding papers, whenever you use any parts of DeepErwin.


Getting Started
===============

The quickest way to get started with DeepErwin is to have a look at our :doc:`tutorial`.
It covers installation, usage of core functionality and major configuration options.

Afterwards take a look at the comprehensive documentation of the source code and APIs: :doc:`api`

About
=====

DeepErwin is a collaborative effort of Michael Scherbela, and Leon Gerard, Rafael Reisenhofer, Philipp Grohs, and Philipp Marquetand (all University of Vienna).
For questions regarding this code, freel free to `reach out via e-mail`_


.. _reach out via e-mail: mailto:deeperwin.datascience@univie.ac.at
