# DeepErwin

DeepErwin is python package that implements and optimizes wave function models for numerical solutions to the multi-electron Schrödinger equation.

DeepErwin is based on JAX and supports:
- Optimizing a wavefunction for a single nuclear geometry
- Optimizing wavefunctions for multiple nuclear geometries in parallel, while sharing neural network weights across these wavefunctions to speed-up optimization
- Using pre-trained weights of a network to speed-up optimization for entirely new wavefunctions
- Using second-order optimizers such as KFAC or L-BFGS 

A detailed description of our method and the corresponding results can be found in our recent [arxiv publication](https://arxiv.org/pdf/2105.08351.pdf). When you use DeepErwin in your work, please cite:

M. Scherbela, R. Reisenhofer, L. Gerard, P. Marquetand, and P. Grohs.<br>
Solving the electronic Schrödinger equation for multiple nuclear geometries with weight-sharing deep neural networks.<br>
arXiv preprint [arXiv:2105.08351](https://arxiv.org/pdf/2105.08351.pdf) (2021).


## Getting Started

The quickest way to get started with DeepErwin is to have a look at our [documentation](https://mdsunivie.github.io/deeperwin/). It contains a detailed description of our python codebase and a [tutorial](https://mipunivie.github.io/deeperwin/tutorial.html) which should help you to quickly get up-and-running using DeepErwin.

## About

DeepErwin is a collaborative effort of Michael Scherbela, Rafael Reisenhofer, Leon Gerard, Philipp Marquetand, and Philipp Grohs.\
The code was written by Michael Scherbela, Leon Gerard, and Rafael Reisenhofer.\
If you have any questions, freel free to reach out via [e-mail](mailto:deeperwin.datascience@univie.ac.at).
