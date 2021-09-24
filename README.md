# DEPRECATION NOTICE

This is a deprecated, legacy version of the deeperwin codebase originally built using TensorFlow. **We highly recommend to switching to the master-branch, containing a faster, better and more accurate version of deeperwin based on JAX.**

# DeepErwin

DeepErwin is python package that implements and optimizes TF 2.x wave function models for numerical solutions to the multi-electron Schrödinger equation.

In particular, DeepErwin supports:
- Optimizing a wavefunction for a single nuclear geometry
- Optimizing wavefunctions for multiple nuclear geometries in parallel, while sharing neural network weights across these wavefunctions to speed-up optimization
- Use pre-trained weights of a network to speed-up optimization for entirely new wavefunctions

A detailed description of our method and the corresponding results can be found in our recent [arxiv publication](https://arxiv.org/pdf/2105.08351.pdf). When you use DeepErwin in your work, please cite:

M. Scherbela, R. Reisenhofer, L. Gerard, P. Marquetand, and P. Grohs.<br>
Solving the electronic Schrödinger equation for multiple nuclear geometries with weight-sharing deep neural networks.<br>
arXiv preprint [arXiv:2105.08351](https://arxiv.org/pdf/2105.08351.pdf) (2021).


## Getting Started

The quickest way to get started with DeepErwin is to have a look at our documentation. It has a detailed description of our python codebase and will also guide you through several [examples](examples), which should help you to quickly get up-and-running using DeepErwin.

## About

DeepErwin is a collaborative effort of Rafael Reisenhofer, Philipp Grohs, Philipp Marquetand, Michael Scherbela, and Leon Gerard (University of Vienna).
For questions regarding this code, freel free to reach out via [e-mail](mailto:deeperwin.datascience@univie.ac.at).

