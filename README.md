# DeepErwin

DeepErwin is python package that implements and optimizes TF 2.x wave function models for numerical solutions to the multi-electron Schrödinger equation.

In particular DeepErwin supports:
- Optimizing a wavefunction for a single nuclear geometry
- Optimizing wavefunctions for multiple nuclear geometries in parallel, while sharing neural network weights across these wavefunctions to speed-up optimization
- Use pre-trained weights of a network to speed-up optimization for entirely new wavefunctions

A detailed description of our method and the corresponding results can be found in our recent [arxiv publication](https://arxiv.org/pdf/2105.08351.pdf). Please cite this paper, whenever you use any parts of DeepErwin.

## Getting Started

The quickest way to get started with DeepErwin is to have a look at our documentation. It has a detailed description of our python codebase and will also guide you through several [examples](examples), which should help you to quickly get up-and-running using DeepErwin.

## About

DeepErwin is a collaborative effort of Rafael Reisenhofer, Philipp Grohs, Philipp Marquetand, Michael Scherbela, and Leon Gerard (University of Vienna).
For questions regarding this code, freel free to reach out via [e-mail](mailto:rafael.reisenhofer@univie.ac.at).

