<h1 align="center" style="margin-top:20px; margin-bottom:50px;">

<a href="https://github.com/seixas-research/carcara" target="_blank" rel="noopener noreferrer">
  <picture>
    <source srcset="https://raw.githubusercontent.com/seixas-research/carcara/refs/heads/main/logo/logo_dark.png" media="(prefers-color-scheme: dark)">
    <source srcset="https://raw.githubusercontent.com/seixas-research/carcara/refs/heads/main/logo/logo_light.png" media="(prefers-color-scheme: light)">
    <img src="https://raw.githubusercontent.com/seixas-research/carcara/refs/heads/main/logo/logo_light.png" alt="Carcará logo" style="height: auto; width: auto; max-height: 100px;">
  </picture>
</a>
</h1> 

[![License: MIT](https://img.shields.io/badge/License-MIT-red?style=for-the-badge)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11+-fcbc2c.svg?style=for-the-badge)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/carcara.svg?style=for-the-badge)](https://pypi.org/project/carcara/)
[![Documentation Status](https://readthedocs.org/projects/carcara/badge/?version=latest&style=for-the-badge)](https://carcara.readthedocs.io/en/latest/?badge=latest)
![Size](https://img.shields.io/github/repo-size/leseixas/carcara?style=for-the-badge&color=orange)


# Carcará

**Carcará** is a lightweight, high-performance Python framework for fermionic quantum simulations based on variational quantum algorithms (VQAs). Developed with an end-to-end physical simulation pipeline, it targets both noise-free research validation and real NISQ-era quantum computing execution (driven via IBM Qiskit).

From molecular geometry inputs, Carcará constructs real-space grids, evaluates one- and two-body integrals, performs Hartree-Fock reference calculations, maps operators to qubit systems, and executes variational eigensolving through both standard VQE and adaptive growth algorithms (ADAPT-VQE) with multiple operator pools. Beyond ground states, it computes **excited states** (variational deflation and subspace-search / SSVQE), offers a **stochastic adaptive** solver (VASQE) with temperature annealing, and handles **periodic crystals** (Bloch band structure and Born–von Kármán total energies). All solvers share a common, extensible driver architecture.

---

## Key Features

### 1. Localized Basis Sets (Generated Native)
All basis set functions are generated from scratch mathematically rather than relying on tabulated basis databases. Supported localized single-particle basis sets include:
- **FAO (Full Atomic Orbital):** Analytic hydrogen-like orbitals built from the actual atomic number (bare nuclear charge, no Slater screening), one per occupied subshell.
- **NAO (Numerical Atomic Orbital):** Confined Sankey/SIESTA-type atomic orbitals solved numerically on radial grids within a hard-wall sphere boundary dictated by a user-specified energy shift.
- **GTO (Gaussian-Type Orbital):** Minimal STO-nG bases generated via scale-covariant least-squares fitting of primitives to Slater-type orbitals.
- **Pople Split-Valence:** Contracted GTO split-valence bases (e.g., 6-31G and 6-31G(d)), featuring native polarization d-shells.

### 2. High-Performance C-Accelerated Integral Engine
A basis-agnostic integration engine handles the heavy lifting of one-body (kinetic $T$, nuclear attraction $V$) and two-body electron-repulsion integrals (ERI, $\langle ab|cd \rangle$ in physicists' notation) in real space:
- **Geometry-Agnostic Grids:** Supports cubic, anisotropic (orthorhombic), and non-orthogonal grids (sampling skewed crystal lattices directly).
- **Fast ERI Solver:** Features an $O(N \log N)$ FFT-based Poisson solver alongside a direct real-space double-sum method.
- **C Backend Acceleration:** An OpenMP-parallelized C backend (`libcarcara_integrals`) built with ctypes zero-copy pointer passing.
- **Graceful Fallback:** Automatically falls back to a vectorized NumPy reference implementation if the C shared library is not compiled.

### 3. Second Quantization & Fermion-to-Qubit Mappings
A robust second-quantized algebra layer implements:
- **`Fermion` Operator:** Full creation/annihilation operator algebra, including helper methods to construct Hamiltonians directly from molecular integrals.
- **`PauliSum` Output:** Clean qubit Pauli operator representation wrapping Qiskit's sparse Pauli operators.
- **Fermion-to-Qubit Mappings:** Jordan-Wigner (default), Parity (with optional two-qubit reduction), and Bravyi-Kitaev mappings.

### 4. Variational Quantum Algorithms (VQAs)
- **VQE (Variational Quantum Eigensolver):** High-precision state-vector simulator employing parameterized quantum circuits (e.g., UCCSD) and classical optimizers (SPSA, COBYLA, Nelder-Mead, SLSQP, Adam, L-BFGS-B).
- **ADAPT-VQE:** Adaptive grows-then-reoptimizes ansatz builder utilizing energy gradients to grow ansätze one operator at a time. It supports four distinct operator pools:
  - `fermionic` (spin-adapted fermionic excitations, Jordan-Wigner mapped).
  - `qubit` (individual JW Pauli strings, providing the shallowest individual operators).
  - `qeb` (qubit-excitation generators with Jordan-Wigner Z-strings dropped).
  - `ceo` (coupled-exchange operators sharing entangling structures, yielding the highest accuracy per CNOT).
- **Excited States:** Both `VQE` and `ADAPTVQE` expose `energy_levels(num_states=...)`, computing the ground state and low-lying excited states by **variational quantum deflation** (VQD); every returned level is a true eigenvalue within the ansatz's reachable sector. `SubspaceVQE` / `SubspaceADAPTVQE` implement **subspace-search VQE (SSVQE)**, finding the ground state and several excited states *simultaneously* in one optimization (one shared unitary over orthogonal references, weighted-energy cost) — returning variational upper bounds (Hylleraas–Undheim).
- **VASQE (Variational Adaptive Stochastic Quantum Eigensolver):** ADAPT-VQE with **stochastic operator selection** — the next operator is sampled from a softmax of the pool gradients, `P(i,τ) = exp(|gᵢ|/τ) / Σⱼ exp(|gⱼ|/τ)`. Low temperature reduces exactly to greedy ADAPT-VQE; a high (optionally **annealed**) temperature explores the ansatz space, with **constant**, **exponential**, **linear**, and **logarithmic** cooling schedules. A `SubspaceVASQE` combines it with subspace search.
- **Periodic Crystals (`BlochADAPTVQE`):** A general 1-/2-/3-D crystal driver from an ASE primitive cell. Solves the single-particle **Bloch Hamiltonian** `H(k)c = ε(k)S(k)c` for the band structure, and computes a correlated **total energy over all k-points** via the Born–von Kármán supercell equivalence.
- **Hartree-Fock Reference Drivers:** Restricted Hartree-Fock (RHF) and Unrestricted Hartree-Fock (UHF) models to supply stable molecular-orbital bases and stationary reference states.
- **Frozen-Core Approximation:** Both `VQE` and `ADAPTVQE` accept `frozen_core` (`True`/`"auto"` for the chemical noble-gas core, or an integer count of lowest MOs) and `frozen_orbitals` (an explicit list of core spatial-MO indices). Frozen core orbitals are removed from the active space and replaced by their mean-field contribution (a constant core energy plus an effective one-body potential), shrinking the qubit count.
- **Spin-Polarized References:** The initial spin state is set the ASE way, through the atoms' initial magnetic moments (`Atoms(..., magmoms=[1, 1])` for a triplet); the calculators read it and build a reference with `n_alpha - n_beta` unpaired electrons.
- **Sparse Large Active Spaces:** For 12+ qubits `ADAPTVQE` automatically switches to a sparse operator pool (`sparse="auto"`) that screens with the exact analytic gradient and applies each excitation with a closed-form `exp(θA)`, keeping frozen-core problems such as water tractable on an exact state-vector backend.
- **Expressibility & Profiling Analysis:** Evaluates parameterized quantum circuit expressibility (KL-divergence vs. Haar distribution within symmetry-conserving subspaces) and tracks circuit complexity (CNOT counts and depth compilation).

### 5. ASE Calculator Integration
The molecular drivers (`VQE`, `ADAPTVQE`, `VASQE`, and their subspace variants) act as standard calculators for the **Atomic Simulation Environment (ASE)**:
```python
atoms.calc = VQE(basis="FAO", optimizer="COBYLA", h=0.20)
# Asking ASE for the energy executes the entire quantum simulation pipeline!
energy_ev = atoms.get_total_energy()
```

### 6. Extensible Driver Architecture
Every variational solver subclasses a single `VariationalDriver` base that owns the shared machinery — the ASE-calculator surface (basis / grid / k-points / spin / frozen core), Hamiltonian materialization (dense or sparse), the state-vector expectation `energy(psi)`, and timing/profiling. Concrete algorithms implement only their optimization loop, and cross-cutting capabilities are **composable mixins**: excited-state deflation (`energy_levels`) and subspace search plug into any driver. Adding a new method (a new operator-selection rule, ansatz, or excited-state technique) requires no changes to the setup code.

---

## Project Structure

```
carcara/
├── src/
│   └── carcara/
│       ├── algorithms/  # VariationalDriver base; VQE, ADAPT-VQE, VASQE, subspace
│       │                #   (SSVQE) + deflation excited states, Bloch crystals,
│       │                #   HF (RHF/UHF), expressibility
│       ├── backends/    # Hardware and simulation devices, error mitigation
│       ├── basis/       # Localized basis sets (FAO, NAO, GTO/STO-nG, Pople)
│       ├── circuits/    # Ansatz protocol, UCCSD & AdaptAnsatz, gates, pools, profiling
│       ├── core/        # Fermionic operators, mappings, molecular integrals
│       ├── integrals/   # Real-space grid and Poisson engine, C backend
│       │   └── csrc/    # C implementation and CMake build files
│       ├── optimizers/  # Classical optimizers for hybrid loops
│       ├── utils/       # Profiling (timing/memory), logging, start-up banner
│       ├── units.py     # Unified conversion factors (Angstrom/eV <-> Bohr/Hartree)
│       └── version.py   # Package versioning (CalVer YY.M.patch)
├── examples/            # Walkthroughs: ADAPT-VQE (H2, LiH, H2O frozen core, BeH2,
│                        #   mappings, O2 triplet), H-chain bands, energy levels
│                        #   (deflation), Subspace-VQE (SSVQE), and VASQE
├── test/                # Comprehensive pytest suite
└── docs/                # Sphinx source files and configuration
```

---

## Installation & Build

### 1. Prerequisites
- **Python** $\ge 3.11$
- **C compiler** with OpenMP support (e.g., GCC, Clang)
- **CMake** $\ge 3.15$

### 2. Installation via pip
You can install the stable release of Carcará directly from PyPI:
```bash
pip install carcara
```

### 3. Installation from Source (Developer Setup)
The package can be used directly from source via `PYTHONPATH` or installed in editable mode:
```bash
# Clone the repository
git clone https://github.com/seixas-research/carcara.git
cd carcara

# Install in editable mode
pip install -e .
```

### 4. Compile the C Integral Backend (Recommended)
Compile the C shared library to enable multi-threaded OpenMP acceleration. The compiled artifact will automatically be detected by `_backend.py`.

On **macOS** (requires Homebrew `libomp`):
```bash
cd src/carcara/integrals/csrc
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DOpenMP_ROOT=$(brew --prefix libomp)
cmake --build build
```

On **Linux**:
```bash
cd src/carcara/integrals/csrc
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

---

## Quickstart Examples

### Example 1: Evaluating Real-Space Integrals (H₂)
Build a minimal basis of Hydrogen 1s orbitals and compute core Hamiltonian matrices:
```python
import numpy as np
from carcara.basis import FullAtomicOrbital
from carcara.integrals import Grid, IntegralEngine, Potentials

# Geometry setup (H2 bond length R = 0.74 A)
R = 0.74
proton_a = np.array([0.0, 0.0, -R / 2])
proton_b = np.array([0.0, 0.0, +R / 2])

# Potential and Grid (spacing h = 0.10 A)
potentials = Potentials([(1.0, proton_a), (1.0, proton_b)])
grid = Grid(center=[0.0, 0.0, 0.0], box_size=5.0, h=0.10)

# Minimal Full Atomic Orbital basis
basis = [FullAtomicOrbital(1, 0, 0, Z=1.0, center=proton_a),
         FullAtomicOrbital(1, 0, 0, Z=1.0, center=proton_b)]

engine = IntegralEngine(basis, grid)

# Compute kinetic T, potential V, and electron-repulsion tensor
T, V = engine.one_body(potentials.nuclear_potential)
h_core = T + V
eri = engine.two_body(method="fft")

print("Core Hamiltonian (eV):\n", h_core.real)
print(f"On-site repulsion <00|00> (eV): {eri[0,0,0,0].real:.3f}")
```

### Example 2: ASE-Driven VQE Simulation
Use the Atomic Simulation Environment (ASE) to run a standard VQE simulation with UCCSD ansatz:
```python
from ase import Atoms
from carcara.algorithms import VQE

# Define H2 molecule in a unit cell
atoms = Atoms("H2",
              positions=[[4.0, 4.0, 3.63], [4.0, 4.0, 4.37]],
              cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
              pbc=True)

# Attach VQE calculator
atoms.calc = VQE(basis="FAO", mapping="jordan_wigner", optimizer="COBYLA", h=0.20)

# Run calculation (energy returned in eV)
energy_ev = atoms.get_total_energy()
result = atoms.calc.vqe_result

print(f"VQE Energy: {result.optimal_energy:.6f} Ha ({energy_ev:.6f} eV)")
```

### Example 3: Running ADAPT-VQE
Compute H₂ ground state adaptively using the hardware-optimized Coupled-Exchange Operator (`"ceo"`) pool:
```python
from ase import Atoms
from carcara.algorithms import ADAPTVQE

atoms = Atoms("H2",
              positions=[[4.0, 4.0, 3.63], [4.0, 4.0, 4.37]],
              cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
              pbc=True)

# Attach ADAPT-VQE calculator
atoms.calc = ADAPTVQE(
              pool="ceo",
              basis="FAO",
              optimizer="COBYLA",
              gradient="parameter-shift",
              h=0.20,
              max_iterations=15,
              gradient_tolerance=1e-6
)

# Run adaptive loop
atoms.get_total_energy()
result = atoms.calc.adapt_result

print(f"ADAPT-VQE Converged: {result.converged}")
print(f"Optimal Energy: {result.optimal_energy:.8f} Ha")
print(f"CNOT Count: {result.metrics.cnot_count}")
```

### Example 4: Excited States (deflation and subspace search)
Compute the ground state and the first excited state, either one after another
(deflation) or simultaneously (SSVQE):
```python
from ase import Atoms
from carcara.algorithms import VQE, SubspaceVQE

atoms = Atoms("H2", positions=[[4.0, 4.0, 3.63], [4.0, 4.0, 4.37]],
              cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]], pbc=True)

# (a) Deflation: excited states one after another.
atoms.calc = VQE(basis="FAO", h=0.20)
atoms.get_potential_energy()                     # configures the solver
levels = atoms.calc.energy_levels(num_states=2, restarts=4)
print("levels (eV):", levels.in_units("eV"))

# (b) SSVQE: ground + excited states in a single optimization.
atoms.calc = SubspaceVQE(basis="FAO", h=0.20, num_states=2)
atoms.get_potential_energy()
print("levels (eV):", atoms.calc.subspace_result.in_units("eV"))
```

### Example 5: VASQE (stochastic ADAPT with temperature annealing)
Grow the ansatz by sampling operators from a softmax of the gradients, annealing
the selection temperature from exploratory to greedy:
```python
from ase import Atoms
from carcara.algorithms import VASQE

atoms = Atoms("H2", positions=[[4.0, 4.0, 3.63], [4.0, 4.0, 4.37]],
              cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]], pbc=True)

atoms.calc = VASQE(basis="FAO", pool="fermionic", h=0.20, optimizer="L-BFGS-B",
                   temperature=2.0, final_temperature=0.01, schedule="exponential",
                   max_iterations=12, gradient_tolerance=1e-5)
atoms.get_total_energy()
result = atoms.calc.vasqe_result

print(f"Energy: {result.optimal_energy:.8f} Ha")
print(f"Operators: {result.operators}")
print(f"Selection temperatures: {result.temperatures}")
```

---

## Testing

Carcará features a comprehensive unit testing suite verifying integrals, basis definitions, operators, Hartree-Fock solvers, VQE, and ADAPT-VQE algorithms. 

To run the complete test suite:
```bash
# From the project root directory
pytest
```

---

## Documentation

Documentation is built using Sphinx:
```bash
cd docs
make html
# Output will be located in docs/build/html/index.html
```

---

## License & Development

Carcará is released under the [MIT License](https://raw.githubusercontent.com/seixas-research/carcara/refs/heads/main/LICENSE).

Developer: **Leandro Seixas Rocha** (<leandro.rocha@ilum.cnpem.br>)

Website/Code: [seixas-research/carcara](https://github.com/seixas-research/carcara)

Documentation: [carcara.readthedocs.io](https://carcara.readthedocs.io/)

# Acknowledgements

We thank financial support from [INCT Materials Informatics](https://inct-mi.pesquisa.ufabc.edu.br/) (Grant No. 406447/2022-5).