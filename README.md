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

* **Native basis sets, generated from scratch:** analytic **hydrogenic** orbitals, confined **numerical atomic orbitals** (NAO), and Gaussian **STO-nG** and split-valence **6-31G(d)** bases — all built on the fly by fitting Slater-type orbitals, with *no tabulated basis-set data* — feeding a real-space one-/two-body integral engine (OpenMP C backend with a NumPy fallback).

* **Second quantization & fermion-to-qubit mappings:** molecular `Fermion` Hamiltonians in physicists' notation, translated to Pauli operators via **Jordan-Wigner**, **Bravyi-Kitaev**, and **parity** mappings (the last with an optional two-qubit reduction).

* **Hartree-Fock & the molecular-orbital basis:** restricted (**RHF**) and unrestricted (**UHF**) self-consistent-field solvers that supply the MO basis correlated methods need.

* **Variational solvers:** an exact state-vector **VQE** with the **UCCSD** ansatz, and **ADAPT-VQE** with four pluggable operator pools — **fermionic**, **qubit** (qubit-ADAPT), **QEB**, and **CEO** — warm-started re-optimization, and **circuit profiling** (CNOT count and depth in a native `{CNOT, U}` gate set) at every step.

* **Circuit analysis:** ansatz **expressibility** (KL divergence of the fidelity distribution from Haar), with native support for tracking it as ADAPT-VQE grows the circuit.

* **Classical optimizers:** a SciPy-backed interface over COBYLA, SLSQP, L-BFGS-B, Nelder-Mead, and Powell, with cost-history tracking.

* **On the roadmap:** real-hardware execution (IBM Quantum via Qiskit) and error mitigation (Zero-Noise Extrapolation, symmetry verification) — see [`plan/roadmap.md`](plan/roadmap.md).

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

## Ground states with VQE and ADAPT-VQE

`carcara.algorithms` provides both a fixed-ansatz **VQE** and an adaptive
**ADAPT-VQE** that grows a compact, problem-tailored ansatz one operator at a
time. ADAPT works in the Hartree-Fock molecular-orbital basis
(`molecular_hamiltonian(mo_basis=True, ...)`), where single-excitation gradients
vanish and the physical correlating excitations are selected first. The pool is
pluggable — `"fermionic"`, `"qubit"`, `"qeb"`, or `"ceo"` — and each grown
circuit is profiled for its CNOT count and depth. The full script is in
[`examples/run_adapt_vqe.py`](examples/run_adapt_vqe.py).

```python
import numpy as np

from carcara.algorithms import AdaptVQE
from carcara.core import HydrogenicIntegrals, minimal_hydrogenic_basis
from carcara.integrals import Grid

R = 0.74
nuclei = [(1.0, np.array([0.0, 0.0, -R / 2])),
          (1.0, np.array([0.0, 0.0, +R / 2]))]
grid = Grid(center=[0.0, 0.0, 0.0], box_size=6.0, h=0.20)

# Hamiltonian in the RHF molecular-orbital basis (2 electrons, 4 qubits).
H = HydrogenicIntegrals(nuclei, minimal_hydrogenic_basis(nuclei),
                        grid).molecular_hamiltonian(mo_basis=True, n_electrons=2)

adapt = AdaptVQE(H, pool="ceo", num_particles=(1, 1), n_spatial_orbitals=2)
result = adapt.run(max_iterations=15, gradient_tol=1e-6)

print(f"ADAPT-VQE energy = {result.optimal_energy:.8f} Ha "
      f"({result.num_operators} operators)")
print(f"CNOTs = {result.metrics.cnot_count}, depth = {result.metrics.depth}")
```

Every pool reaches the exact (FCI) ground state on H₂; on hardware-minded pools
it does so with far fewer CNOTs (the qubit pool reaches it in 6 CNOTs versus 48
for the fermionic pool).

## Measuring ansatz expressibility

`carcara.algorithms.expressivity` scores how uniformly a parameterized circuit
covers its accessible Hilbert space by comparing its random-parameter fidelity
distribution to the Haar distribution (lower KL = more expressive). Because the
fermionic ansätze conserve particle number and spin, the Haar reference uses the
**number-conserving sector** dimension, not the full `2^N`. The
`ADAPTExpressivityTracker` records the score as ADAPT-VQE grows the ansatz — see
[`examples/adapt_expressivity.py`](examples/adapt_expressivity.py).

```python
from carcara.algorithms import compute_expressibility
from carcara.circuits import UCCSD

ansatz = UCCSD(n_spatial_orbitals=2, num_particles=(1, 1))
result = compute_expressibility(ansatz, num_samples=2000, num_particles=(1, 1))
print(result)   # ExpressibilityResult(E=..., d=4, n_samples=2000)
```

## Potential-energy surfaces and basis-set comparison

The [`examples/generate_h2_pes.py`](examples/generate_h2_pes.py) and
[`examples/generate_lih_pes.py`](examples/generate_lih_pes.py) scripts scan a bond
length and export RHF dissociation curves for the hydrogenic, STO-3G, and
6-31G(d) bases to CSV; [`examples/plot_pes.py`](examples/plot_pes.py) renders the
multi-curve comparison.

# License

This is an open source code under [MIT License](https://raw.githubusercontent.com/seixas-research/carcara/refs/heads/main/LICENSE).

# Acknowledgements

We thank financial support from [INCT Materials Informatics](https://inct-mi.pesquisa.ufabc.edu.br/) (Grant No. 406447/2022-5).
