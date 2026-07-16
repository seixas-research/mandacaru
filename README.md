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
[`examples/H2_integrals.py`](examples/H2_integrals.py).

```python
import numpy as np

from carcara.basis import HydrogenicOrbital
from carcara.integrals import Grid, IntegralEngine, Potentials

# Geometry: the user-facing API uses Angstrom for lengths and eV for energies.
# H2 equilibrium bond length ~0.74 A; two protons about the origin.
Z, R = 1.0, 0.74
proton_a = np.array([0.0, 0.0, -R / 2])
proton_b = np.array([0.0, 0.0, +R / 2])

# External electron-nuclear potential V(r) = -sum_A Z / |r - R_A|.
potentials = Potentials([(Z, proton_a), (Z, proton_b)])

grid = Grid(center=[0.0, 0.0, 0.0], box_size=5.0, h=0.10)  # Angstrom
basis = [HydrogenicOrbital(1, 0, 0, Z=Z, center=proton_a),
         HydrogenicOrbital(1, 0, 0, Z=Z, center=proton_b)]

engine = IntegralEngine(basis, grid)

# One-body: kinetic T and nuclear attraction V -> core Hamiltonian (eV).
T, V = engine.one_body(potentials.nuclear_potential)
h_core = T + V

# Two-body electron-repulsion tensor <ab|cd> in physicists' notation (eV).
eri = engine.two_body(method="fft")

print("Core Hamiltonian h = T + V (eV):")
print(h_core.real)
print(f"<00|00> on-site repulsion = {eri[0, 0, 0, 0].real:.3f} eV")
```

Running it prints the `2 x 2` core Hamiltonian and the on-site repulsion
`<00|00> ~ 17.0 eV`, in agreement with the exact hydrogen 1s value of
`5/8 Ha = 17.007 eV`.

## A heteronuclear molecule: LiH

The same machinery scales to multi-orbital, heteronuclear systems. The example
[`examples/LiH_integrals.py`](examples/LiH_integrals.py) builds a small minimal
basis for LiH -- the Li 1s, 2s and 2p_z orbitals plus the H 1s -- using the
*true* nuclear charges (`Z_Li = 3`, `Z_H = 1`) in the potential and *effective*
charges from **Slater's rules** for the hydrogenic basis orbitals via
`HydrogenicOrbital.from_slater`:

```python
labels = ["Li 1s", "Li 2s", "Li 2pz", "H 1s"]
basis = [HydrogenicOrbital.from_slater(1, 0, 0, atomic_number=3, center=li_pos),
         HydrogenicOrbital.from_slater(2, 0, 0, atomic_number=3, center=li_pos),
         HydrogenicOrbital.from_slater(2, 1, 0, atomic_number=3, center=li_pos),
         HydrogenicOrbital.from_slater(1, 0, 0, atomic_number=1, center=h_pos)]

potentials = Potentials([(3.0, li_pos), (1.0, h_pos)])  # true nuclear charges
engine = IntegralEngine(basis, grid)
T, V = engine.one_body(potentials.nuclear_potential)
eri = engine.two_body(method="fft")
```

This yields the `4 x 4` one-body matrices and the `4 x 4 x 4 x 4`
electron-repulsion tensor. The H 1s on-site integral `<33|33> ~ 17.0 eV` again
recovers the exact `5/8 Ha`.

## Fermionic Hamiltonian and fermion-to-qubit mapping

The `carcara.core` module assembles the second-quantized molecular Hamiltonian
from those integrals and maps it to a qubit (Pauli) operator. It is built as a
`Fermion` operator in physicists' notation,
`H = Σ_pq h_pq a†_p a_q + ½ Σ_pqrs ⟨pq|rs⟩ a†_p a†_q a_s a_r`, and
`map_to_qubits` translates it into Pauli strings via **Jordan-Wigner** (the
default), **Bravyi-Kitaev**, or **parity** -- the last with an optional
two-qubit reduction that exploits particle-number symmetry. The full script is
in [`examples/H2_mapping.py`](examples/H2_mapping.py).

```python
import numpy as np

from carcara.core import HydrogenicIntegrals, minimal_hydrogenic_basis
from carcara.integrals import Grid

# H2: a minimal Slater-screened 1s basis, one orbital per atom (Angstrom).
R = 0.74
nuclei = [(1.0, np.array([0.0, 0.0, -R / 2])),
          (1.0, np.array([0.0, 0.0, +R / 2]))]
basis = minimal_hydrogenic_basis(nuclei)
grid = Grid(center=[0.0, 0.0, 0.0], box_size=6.0, h=0.15)

# Second-quantized molecular Hamiltonian over spin-orbitals (a `Fermion`).
integrals = HydrogenicIntegrals(nuclei, basis, grid)
H = integrals.molecular_hamiltonian()          # 2 spatial -> 4 spin-orbitals

# Map to a qubit operator (a `PauliSum` of Pauli strings).
H_jw = H.map_to_qubits(method="jordan_wigner")           # default
H_bk = H.map_to_qubits(method="bravyi_kitaev")
H_parity = H.map_to_qubits(method="parity",
                           two_qubit_reduction=True, num_particles=(1, 1))

print(f"Jordan-Wigner: {H_jw.num_qubits} qubits, "
      f"{len(H_jw.simplify().terms)} Pauli terms")
print(f"parity + two-qubit reduction: {H_parity.num_qubits} qubits")

# Exact ground state by diagonalizing the (Hermitian) qubit Hamiltonian.
E0 = np.linalg.eigvalsh(H_jw.to_matrix()).min()
print(f"ground-state energy = {E0:.4f} Ha")
```

Under Jordan-Wigner the 4-spin-orbital H2 Hamiltonian becomes a **4-qubit**
Pauli operator (27 terms); the parity mapping's two-qubit reduction tapers it to
**2 qubits** while preserving the ground-state energy `-1.1154 Ha`. Hand the
result to Qiskit with `H_jw.to_sparse_pauli_op()`.

# License

This is an open source code under [MIT License](https://raw.githubusercontent.com/seixas-research/carcara/refs/heads/main/LICENSE).

# Acknowledgements

We thank financial support from [INCT Materials Informatics](https://inct-mi.pesquisa.ufabc.edu.br/) (Grant No. 406447/2022-5).
