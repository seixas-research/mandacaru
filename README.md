<h1 align="center" style="margin-top:20px; margin-bottom:50px;">

<a href="https://github.com/seixas-research/carcara" target="_blank" rel="noopener noreferrer">
  <picture>
    <source srcset="https://raw.githubusercontent.com/seixas-research/carcara/refs/heads/main/logo/logo_dark.png" media="(prefers-color-scheme: dark)">
    <source srcset="https://raw.githubusercontent.com/seixas-research/carcara/refs/heads/main/logo/logo_light.png" media="(prefers-color-scheme: light)">
    <img src="https://raw.githubusercontent.com/seixas-research/carcara/refs/heads/main/logo/logo_light.png" alt="Carcará logo" style="height: auto; width: auto; max-height: 100px;">
  </picture>
</a>
</h1> 

[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)    [![PyPI](https://img.shields.io/pypi/v/carcara?color=red&style=for-the-badge)](https://pypi.org/project/carcara/)

# Carcará

**Carcará** is a framework for fermionic quantum simulation based on variational quantum algorithms, engineered from the ground up for deployment on real quantum hardware.


# Overview

Carcará connects theoretical condensed matter physics with NISQ-era quantum hardware. Engineered around variational workflows, the framework streamlines the pipeline from mapping complex fermionic Hamiltonians onto qubit operators to optimizing ansatz states and executing error-mitigated circuits on real quantum backends.


## Key Features

* **Fermion-to-Qubit Mapping:** Built-in, optimized transformations including Jordan-Wigner, Bravyi-Kitaev, and parity mappings to translate fermionic creation/annihilation operators into Pauli strings.

* **Hardware-Efficient & Physics-Inspired Ansatzes:** Ready-to-use ansatz generation, including Unitary Coupled Cluster (UCCSD) and hardware-efficient templates designed to minimize circuit depth and gate errors on real QPUs.

* **Hybrid Variational Solvers:** Robust implementation of the Variational Quantum Eigensolver (VQE) and its time-dependent variants, coupled with state-of-the-art classical optimizers (e.g., SPSA, COBYLA, SLSQP).

* **Real Hardware Deployment:** Seamless integration with major quantum cloud providers (IBM Quantum Platform) with native support.

* **Advanced Error Mitigation:** Built-in noise-resilient pipelines featuring Zero-Noise Extrapolation (ZNE) and symmetry verification.

# Installation

## From pip

The easiest way to install Carcará is with pip:

```console
pip install carcara
```

## From github

To install Carcará directly from the GitHub repository, run the following commands:

```console
git clone https://github.com/seixas-research/carcara.git
cd carcara
pip install -e .
```

# Getting started

## One- and two-body integrals for H2

The `carcara.integrals` module computes real-space one- and two-body integrals
over any localized basis. The example below builds a minimal basis of one
hydrogen 1s orbital on each proton and evaluates the core Hamiltonian and the
electron-repulsion tensor. The full script lives in
[`examples/h2_integrals.py`](examples/h2_integrals.py).

```python
import numpy as np

from carcara.basis import HydrogenicOrbital
from carcara.integrals import Grid, IntegralEngine

# Geometry (atomic units): two protons at the H2 equilibrium bond length.
Z, R = 1.0, 1.4
nuclei = np.array([[0.0, 0.0, -R / 2], [0.0, 0.0, +R / 2]])

def nuclear_potential(x, y, z):
    v = np.zeros_like(x, dtype=float)
    for Rx, Ry, Rz in nuclei:
        r = np.sqrt((x - Rx) ** 2 + (y - Ry) ** 2 + (z - Rz) ** 2)
        v -= Z / np.maximum(r, 1e-12)
    return v

grid = Grid(center=[0.0, 0.0, 0.0], box_size=10.0, points=64)
basis = [HydrogenicOrbital(1, 0, 0, Z=Z, center=nuclei[0]),
         HydrogenicOrbital(1, 0, 0, Z=Z, center=nuclei[1])]

engine = IntegralEngine(basis, grid)

# One-body: kinetic T and nuclear attraction V -> core Hamiltonian.
T, V = engine.one_body(nuclear_potential)
h_core = T + V

# Two-body electron-repulsion tensor (ab|cd) in chemists' notation.
eri = engine.two_body(method="fft")

print("Core Hamiltonian h = T + V (Ha):")
print(h_core.real)
print(f"(00|00) on-site repulsion = {eri[0, 0, 0, 0].real:.4f} Ha")
```

Running it prints the `2 x 2` core Hamiltonian and the on-site repulsion
`(00|00) ~ 0.62 Ha`, in agreement with the exact hydrogen 1s value of `5/8 Ha`.

# License

This is an open source code under [MIT License](https://raw.githubusercontent.com/seixas-research/carcara/refs/heads/main/LICENSE).

# Acknowledgements

We thank financial support from [INCT Materials Informatics](https://inct-mi.pesquisa.ufabc.edu.br/) (Grant No. 406447/2022-5).
