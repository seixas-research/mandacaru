<h1 align="center" style="margin-top:20px; margin-bottom:50px;">

<a href="https://github.com/seixas-research/carcara" target="_blank" rel="noopener noreferrer">
  <picture>
    <source srcset="https://raw.githubusercontent.com/seixas-research/carcara/refs/heads/main/logo/logo_dark.png" media="(prefers-color-scheme: dark)">
    <source srcset="https://raw.githubusercontent.com/seixas-research/carcara/refs/heads/main/logo/logo_light.png" media="(prefers-color-scheme: light)">
    <img src="https://raw.githubusercontent.com/seixas-research/carcara/refs/heads/main/logo/logo_light.png" alt="Carcará logo" style="height: auto; width: auto; max-height: 100px;">
  </picture>
</a>
</h1> 

[![License: MIT](https://img.shields.io/badge/License-MIT-red?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11+-fcbc2c.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/carcara.svg?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/carcara/)
[![Documentation Status](https://readthedocs.org/projects/carcara/badge/?version=latest&style=for-the-badge&logo=readthedocs&logoColor=white)](https://carcara.readthedocs.io/en/latest/?badge=latest)


# Carcará

**Carcará** is a lightweight, high-performance Python framework for fermionic quantum simulations based on variational quantum algorithms (VQAs). Developed with an end-to-end physical simulation pipeline, it targets both noise-free research validation and real NISQ-era quantum hardware — running on **IBM Qiskit**, **Amazon Braket** (including real QPUs) and **Google Cirq** through one unchanged API.

From molecular geometry inputs, Carcará constructs real-space grids, evaluates one- and two-body integrals, performs Hartree-Fock reference calculations, maps operators to qubit systems, and executes variational eigensolving through both standard VQE and adaptive growth algorithms (**ADAPT-VQE**, the default method) with multiple operator pools. Beyond ground states, it computes **excited states** (variational deflation and subspace-search / SSVQE) and handles **periodic systems** (Bloch band structure and Born–von Kármán total energies). All algorithms are reached through a single ASE calculator, `Carcara`, which selects the solver with a `method=` argument, and share a common, extensible driver architecture underneath.

---

## Key Features

### 1. Localized Basis Sets (Generated Native)
All basis set functions are generated from scratch mathematically rather than relying on tabulated basis databases. Supported localized single-particle basis sets include:
- **FAO (Full Atomic Orbital):** Analytic hydrogen-like orbitals built from the actual atomic number (bare nuclear charge, no Slater screening), one per occupied subshell.
- **NAO (Numerical Atomic Orbital):** Confined Sankey/SIESTA-type atomic orbitals solved numerically on radial grids within a hard-wall sphere boundary dictated by a user-specified energy shift. A `size` argument selects **multiple-zeta and polarized** variants (`SZ`, `DZ`, `DZP`, `TZP`, `TZ2P`, `QZP`, ...): extra zetas are built by the SIESTA split-valence construction, polarization by an $l+1$ shell solved in the same confining sphere.
- **Named Gaussian families (Pople, Dunning, Karlsruhe):** the standard basis-set names — `3-21G` … `6-311+G(2df,2p)`, `cc-pVDZ` … `cc-pV5Z`, `aug-cc-pVDZ`, `cc-pCVDZ`, `def2-SV(P)` … `def2-QZVPPD`, plus `STO-3G` … `STO-6G` — are parsed into their structure (core contraction, valence split, polarization, diffuse and tight functions) and generated natively with the **published shell structure and function count** (`cc-pVTZ` C = `[4s3p2d1f]`), exponents from Carcará's own Slater-orbital fits rather than a basis-set table. `basis="cc-pVDZ"`, `basis="def2-TZVP"`.
- **Per-element basis mappings:** `basis={"O": {"name": "NAO", "size": "DZP"}, "H": "6-31G", "*": "FAO"}` puts a different family on different elements — a polarized basis where the chemistry happens and a minimal one on a spectator ion — accepted by every driver, the dry run and `BasisSet.build`.
- **NAO-AE (All-Electron Numerical Atomic Orbital):** every occupied shell of the self-consistent LDA atom — core included — re-solved under a smooth *exponential-wall* confinement (zero up to an onset radius, divergent at `onset + width`), plus hydrogen-like **tiers**: a polarization shell at $l_\max+1$ and diffuse / contracted functions per valence channel, whose effective charges are derived from the atom's own valence radius rather than tabulated. Each $l$ channel is Gram–Schmidt orthonormalized. `basis={"name": "NAO-AE", "tier": 1}`.

### 2. High-Performance C-Accelerated Integral Engine
A basis-agnostic integration engine handles the heavy lifting of one-body (kinetic $T$, nuclear attraction $V$) and two-body electron-repulsion integrals (ERI, $\langle ab|cd \rangle$ in physicists' notation) in real space:
- **Geometry-Agnostic Grids:** Supports cubic, anisotropic (orthorhombic), and non-orthogonal grids (sampling skewed crystal lattices directly).
- **Fast ERI Solver:** Features an $O(N \log N)$ FFT-based Poisson solver alongside a direct real-space double-sum method.
- **C Backend Acceleration:** An OpenMP-parallelized C backend (`libcarcara_integrals`) built with ctypes zero-copy pointer passing, checked and compiled on demand before any integration; the FFT two-body step runs in memory-bounded blocks over the Hermitian pair densities (`CARCARA_ERI_MEMORY_MB`).
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
- **Excited States:** Every `Carcara` exposes `energy_levels(num_states=...)`, computing the ground state and low-lying excited states by **variational quantum deflation** (VQD); every returned level is a true eigenvalue within the ansatz's reachable sector. `method="subspace-vqe"` / `method="subspace-adapt-vqe"` implement **subspace-search VQE (SSVQE)**, finding the ground state and several excited states *simultaneously* in one optimization (one shared unitary over orthogonal references, weighted-energy cost) — returning variational upper bounds (Hylleraas–Undheim).
- **Periodic Systems (`BlochCalculator`):** A general 1-/2-/3-D crystal calculator from an ASE primitive cell, with the correlated solver selected by `method=` (`"adapt-vqe"` by default, or `"vqe"`). Solves the single-particle **Bloch Hamiltonian** `H(k)c = ε(k)S(k)c` for the band structure, and computes a correlated **total energy over all k-points** via the Born–von Kármán supercell equivalence.
- **Hartree-Fock Reference Drivers:** Restricted Hartree-Fock (RHF) supplies the molecular-orbital basis and stationary reference for closed shells; **open-shell systems** — any odd electron count, or a high-spin state set through `magmoms` — are built in the **natural-orbital basis of the unrestricted (UHF) solution**, one spatial basis shared by both spins, so radicals and doublets run through the same solvers, pools and mappings (`Carcara` on an `OH` radical gives a `(5, 4)` reference on 12 qubits).
- **Interaction energies on a shared grid:** `interaction_energy(complex, fragments=[[0,1,2],[3]], charges=[0,1], charge=1, method="adapt-vqe", ...)` evaluates the complex and every fragment on one grid at identical coordinates — the only way a real-space difference of total energies is meaningful — with `method="rhf"` for a circuit-free mean-field check; also `Carcara(...).interaction_energy(...)`.
- **Frozen-Core Approximation:** Every method accepts `frozen_core` (`True`/`"auto"` for the chemical noble-gas core, or an integer count of lowest MOs) and `frozen_orbitals` (an explicit list of core spatial-MO indices). Frozen core orbitals are removed from the active space and replaced by their mean-field contribution (a constant core energy plus an effective one-body potential), shrinking the qubit count.
- **Spin States & Open Shells:** The initial spin state is set the ASE way, through the atoms' initial magnetic moments (`Atoms(..., magmoms=[1, 1])` for a triplet, `[1, 1, 1]` for a quartet); an odd electron count is a doublet by default. Odd-electron systems — radicals, ions, single atoms — are built in the UHF natural-orbital basis in every basis family, plane waves included, and run through the same solvers, pools, mappings, frozen core and dry run.
- **Sparse Large Active Spaces:** For 12+ qubits the adaptive methods automatically switch to a sparse operator pool (`sparse="auto"`) that screens with the exact analytic gradient and applies each excitation with a closed-form `exp(θA)`, keeping frozen-core problems such as water tractable on an exact state-vector backend.
- **Expressibility & Profiling Analysis:** Evaluates parameterized quantum circuit expressibility (KL-divergence vs. Haar distribution within symmetry-conserving subspaces) and tracks circuit complexity (CNOT counts and depth compilation).
- **Dynamic Parametrization (`quenching`):** `True` (default) re-optimizes every variational parameter at each iteration — standard ADAPT-VQE. `False` freezes previously optimized angles and varies only the newest one, turning each growth step into a cheap one-dimensional line search.

### 5. Multi-Backend Execution (Qiskit / Braket / Cirq)
`backend_provider` selects which quantum SDK builds — and, with `execute_circuits=True`, runs — the ansatz circuits. Each generator is an anti-Hermitian `PauliSum` whose terms commute, so `exp(θA)` factorizes **exactly** into Pauli rotations (no Trotter error). One shared gate stream (`X`, `H`, `S`, `S†`, `CNOT`, `Rz`) is translated per SDK, so all three reproduce the internal NumPy state vector **to machine precision**:

```text
provider            E (Ha)   err vs FCI   ops  cnots  depth
(matrix)       -6.88824276     1.34e-07     8    208    273
qiskit         -6.88824283     6.27e-08     8    208    273
braket         -6.88824279     1.09e-07     8    208    359
cirq           -6.88824281     8.17e-08     8    208    358
```

### 6. Real Quantum Hardware via Amazon Braket
A QPU never returns a state vector — Braket rejects the `StateVector` result type whenever `shots > 0`, and every QPU requires it. Carcará therefore implements the **shot-based** protocol hardware actually supports:

- **Qubit-wise commuting (QWC) grouping** partitions `H = Σ cⱼ Pⱼ` into simultaneously measurable sets — 118 Pauli terms collapse to 29 measurement circuits for LiH — and `⟨H⟩` is assembled from the returned bit-string counts, converging as `1/√shots`.
- **Device registry:** the local simulator, the AWS managed simulators (SV1/DM1/TN1), and the IonQ / IQM / Rigetti QPUs — or any Braket ARN. Naming a QPU without `shots` is rejected up front rather than at submission.

```python
atoms.calc = Carcara(method="vqe", basis="FAO",
                               device="braket-ionq-aria", shots=8192)
atoms.get_total_energy()          # measured on a trapped-ion QPU
```

> **Scope:** the *energy evaluation* is hardware-native. ADAPT-VQE's pool-gradient screening is still classical, so fixed-ansatz `method="vqe"` is the fully hardware-native method today. Run `examples/13_braket_aws_compatibility.py` for a verified compatibility report (no AWS account needed).

### 7. Reusable Hamiltonians (Parquet / JSON Cache)
Building the qubit Hamiltonian — integrals plus the fermion-to-qubit mapping — is the most expensive stage of a run and is independent of the algorithm that follows. It can be serialized and replayed:

```python
Carcara(method="adapt-vqe", basis="FAO",
                  save_hamiltonian="lih.parquet")   # build once
Carcara(method="adapt-vqe", pool="ceo",
                  load_hamiltonian="lih.parquet")   # reload: no geometry,
                                                    # no integrals, no mapping
```

Two formats, selected with `hamiltonian_format`: **Parquet** (compressed, columnar, queryable straight from pandas; ~4× smaller) and **JSON** (plain text, no native dependency). Loading **detects the format automatically** — from the extension, else from the file's leading bytes. Because the file also records `num_particles` and `n_spatial_orbitals`, a reloaded calculator runs with no `Atoms` object at all, turning a pool/optimizer/mapping sweep into seconds.

### 8. ASE Calculator Integration
`Carcara` is a standard calculator for the **Atomic Simulation Environment (ASE)**; the `method` argument selects the solver (`"adapt-vqe"` — the default — `"vqe"`, `"subspace-vqe"`, `"subspace-adapt-vqe"`):
```python
atoms.calc = Carcara(method="vqe", basis="FAO", optimizer="COBYLA", h=0.20)
# Asking ASE for the energy executes the entire quantum simulation pipeline!
energy_ev = atoms.get_total_energy()
```

### 9. Dry Run and the Command Line
A **dry run** reports the qubit budget of a calculation — one qubit per active spin-orbital, after the frozen core or plane-wave cutoff is accounted for — without computing an integral, mapping a Hamiltonian or executing a circuit, and compares it with the capacity of the target device:
```bash
carcara water.xyz --frozen-core --dry-run                  # 12 qubits
carcara H2O --basis NAO --basis-option size=DZP --device braket-ionq-aria --dry-run
carcara --load-hamiltonian lih.parquet --dry-run --json
```
The same `carcara` command runs the full calculation without `--dry-run`. From Python, `Carcara(...).dry_run(atoms)` returns the `QubitEstimate`, and `dry_run=True` makes every driver stop before the Hamiltonian is built.

### 10. Extensible Driver Architecture
Underneath `Carcara`, every variational solver subclasses a single `VariationalDriver` base that owns the shared machinery — the ASE-calculator surface (basis / grid / k-points / spin / frozen core), Hamiltonian materialization (dense or sparse), the state-vector expectation `energy(psi)`, and timing/profiling. Concrete algorithms implement only their optimization loop, and cross-cutting capabilities are **composable mixins**: excited-state deflation (`energy_levels`) and subspace search plug into any driver. Adding a new method (a new operator-selection rule, ansatz, or excited-state technique) requires no changes to the setup code.

---

## Project Structure

```
carcara/
├── src/
│   └── carcara/
│       ├── algorithms/  # VariationalDriver base; VQE, ADAPT-VQE (default), subspace
│       │                #   (SSVQE) + deflation excited states, Bloch crystals,
│       │                #   HF (RHF/UHF), expressibility, the Carcara calculator, dry run
│       ├── experimental/ # features under development -- not stable API
│       ├── backends/    # hardware.py    device registry (ideal sim, Braket, QPUs)
│       │                # providers.py   Qiskit / Braket / Cirq circuit builders
│       │                # measurement.py QWC grouping, shot-based <H>
│       │                # mitigation.py  error mitigation (stub)
│       ├── basis/       # Localized basis sets (FAO, NAO, NAO-AE, GTO/STO-nG, Pople)
│       ├── circuits/    # Ansatz protocol, UCCSD & AdaptAnsatz, gates, pools, profiling
│       ├── core/        # Fermionic operators, mappings, molecular integrals,
│       │                #   Parquet/JSON Hamiltonian serialization
│       ├── integrals/   # Real-space grid and Poisson engine, C backend
│       │   └── csrc/    # C implementation and CMake build files
│       ├── optimizers/  # Classical optimizers for hybrid loops
│       ├── utils/       # Profiling (timing/memory), logging, start-up banner
│       ├── units.py     # Unified conversion factors (Angstrom/eV <-> Bohr/Hartree)
│       └── version.py   # Package versioning (CalVer YY.M.patch)
├── examples/            # 17 runnable walkthroughs (see below)
│   └── data/            #   all generated logs, CSV and plots land here
├── test/                # Comprehensive pytest suite (609 tests)
└── docs/                # Sphinx source files and configuration
```

### Examples

| | |
|---|---|
| `01`–`06` | ADAPT-VQE: H₂, LiH, H₂O (frozen core), BeH₂, mapping comparison, O₂ triplet |
| `07`, `11` | Periodic systems: H-chain bands, `BlochCalculator` with both methods |
| `08`, `09` | Excited states: deflation (`energy_levels`) and Subspace-VQE (SSVQE) |
| `12` | ADAPT-VQE on LiH across **all three backend providers**, from one cached Hamiltonian |
| `13` | **Amazon Braket compatibility report** — gate set, shots constraint, QWC grouping, QPU cost |
| `15` | **Expressibility growth** during ADAPT-VQE + PQC-vs-Haar fidelity distributions |
| `16` | LiH energy vs. bond distance across **pools × mappings** (two-column subplots) |
| `17` | Hamiltonian cache round-trip in **Parquet and JSON** |

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

### 4. The C Integral Backend
Real-space integrals run on an OpenMP C library. Carcará prefers it and **compiles it automatically** the first time an integral engine is created in a process (CMake, or the system C compiler when CMake is absent; Homebrew `libomp` on macOS), falling back to the NumPy reference kernels only when that compile fails (with a `RuntimeWarning`). `carcara.integrals.check_backend()` reports what is loaded, `ensure_backend()` runs the same check the engine runs, and `CARCARA_BACKEND=auto|c|numpy` sets the policy. To compile by hand instead:

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
from carcara.algorithms import Carcara

# Define H2 molecule in a unit cell
atoms = Atoms("H2",
              positions=[[4.0, 4.0, 3.63], [4.0, 4.0, 4.37]],
              cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
              pbc=True)

# Attach the calculator with the VQE method
atoms.calc = Carcara(method="vqe", basis="FAO", mapping="jordan_wigner",
                               optimizer="COBYLA", h=0.20)

# Run calculation (energy returned in eV)
energy_ev = atoms.get_total_energy()
result = atoms.calc.result

print(f"VQE Energy: {result.optimal_energy:.6f} Ha ({energy_ev:.6f} eV)")
```

### Example 3: Running ADAPT-VQE
Compute H₂ ground state adaptively using the hardware-optimized Coupled-Exchange Operator (`"ceo"`) pool:
```python
from ase import Atoms
from carcara.algorithms import Carcara

atoms = Atoms("H2",
              positions=[[4.0, 4.0, 3.63], [4.0, 4.0, 4.37]],
              cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
              pbc=True)

# Attach the calculator with the ADAPT-VQE method
atoms.calc = Carcara(
              method="adapt-vqe",
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
result = atoms.calc.result

print(f"ADAPT-VQE Converged: {result.converged}")
print(f"Optimal Energy: {result.optimal_energy:.8f} Ha")
print(f"CNOT Count: {result.metrics.cnot_count}")
```

### Example 4: Excited States (deflation and subspace search)
Compute the ground state and the first excited state, either one after another
(deflation) or simultaneously (SSVQE):
```python
from ase import Atoms
from carcara.algorithms import Carcara

atoms = Atoms("H2", positions=[[4.0, 4.0, 3.63], [4.0, 4.0, 4.37]],
              cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]], pbc=True)

# (a) Deflation: excited states one after another.
atoms.calc = Carcara(method="vqe", basis="FAO", h=0.20)
atoms.get_potential_energy()                     # configures the solver
levels = atoms.calc.energy_levels(num_states=2, restarts=4)
print("levels (eV):", levels.in_units("eV"))

# (b) SSVQE: ground + excited states in a single optimization.
atoms.calc = Carcara(method="subspace-vqe", basis="FAO", h=0.20,
                               num_states=2)
atoms.get_potential_energy()
print("levels (eV):", atoms.calc.result.in_units("eV"))
```

### Example 6: Cache the Hamiltonian, then sweep
Build the expensive part once and replay it — the reload needs no geometry,
no integrals and no fermion-to-qubit mapping:
```python
from ase import Atoms
from carcara.algorithms import Carcara

atoms = Atoms("LiH", positions=[[7.5, 7.5, 6.7], [7.5, 7.5, 8.3]],
              cell=[[15, 0, 0], [0, 15, 0], [0, 0, 15]], pbc=True)

# Build once (use hamiltonian_format="json" for a plain-text cache).
atoms.calc = Carcara(method="adapt-vqe", basis="FAO", h=0.25,
                               save_hamiltonian="lih.parquet")
atoms.get_total_energy()

# Compare every pool against the *same* operator, in seconds.
for pool in ("fermionic", "qubit", "qeb", "ceo"):
    result = Carcara(method="adapt-vqe", pool=pool,
                               load_hamiltonian="lih.parquet",
                               verbose=False).run()
    print(f"{pool:<10} {result.optimal_energy:.8f} Ha  "
          f"{result.num_operators} ops  {result.metrics.cnot_count} CNOTs")
```

### Example 7: Choose a backend — or a real QPU
The calculator API does not change; only the device does:
```python
from carcara.algorithms import Carcara

# Build and execute the circuits with Cirq (or "braket", or "qiskit").
Carcara(method="adapt-vqe", basis="FAO",
                  backend_provider="cirq", execute_circuits=True)

# Braket's local simulator, shot-based -- the same protocol a QPU uses.
Carcara(method="vqe", basis="FAO", device="braket-local", shots=8192)

# The AWS managed simulator, or a real trapped-ion QPU.
Carcara(method="vqe", basis="FAO", device="braket-sv1", shots=8192)
Carcara(method="vqe", basis="FAO", device="braket-ionq-aria",
                  shots=8192)   # needs AWS credentials
```

---

## Testing

Carcará features a comprehensive unit testing suite (900+ tests) verifying integrals, basis definitions, operators, Hartree-Fock solvers, VQE/ADAPT-VQE, the Hamiltonian cache, backend-provider equivalence, and the Braket shot-based measurement path.

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