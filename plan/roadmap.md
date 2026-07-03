# Carcará — Development Roadmap

> **Carcará** is a framework for fermionic quantum simulation based on
> variational quantum algorithms (VQAs), engineered from the ground up for
> deployment on real, NISQ-era quantum hardware.

This roadmap turns the vision in the [README](../README.md) into a concrete,
phased engineering plan. It maps each capability onto the existing package
skeleton and defines milestones, deliverables, and acceptance criteria.

---

## 1. Vision & Scope

Carcará provides an end-to-end pipeline for simulating fermionic systems
(molecules and condensed-matter lattice models) on quantum computers:

1. Fermionic system (molecule / model)
2. Second-quantized Hamiltonian
3. Qubit (Pauli) Hamiltonian
4. Parameterized ansatz |ψ(θ)⟩
5. VQA optimization
6. QPU / Simulator
7. Error mitigation
8. Result + observables


**In scope:** Hamiltonian construction, fermion-to-qubit mappings, ansatz
generation, variational solvers (VQE and variants), classical optimizers,
hardware execution, and error mitigation.

**Out of scope (for now):** fault-tolerant algorithms (QPE on logical qubits),
full quantum error correction, and tensor-network classical backends.

---

## 2. Current State (baseline)

Package version: **26.6.4**. Build: `hatchling`. Python ≥ 3.10.
Core dependencies: `numpy`, `scipy`, `qiskit`, `qiskit-nature`,
`qiskit-ibm-runtime`, `ase`.

| Module | Path | Status |
|---|---|---|
| Package init | `src/carcara/__init__.py` | exports `__version__` |
| Wavefunction (hydrogen-like, ASE I/O) | `src/carcara/wavefunction.py` | implemented (~247 LOC) |
| Hamiltonian | `src/carcara/core/hamiltonian.py` | **empty stub** |
| Mappings | `src/carcara/core/mappings.py` | **empty stub** |
| Operator pools | `src/carcara/circuits/pools.py` | **planned** (new) |
| ADAPT-VQE | `src/carcara/algorithms/adapt.py` | **planned** (new) |
| Ansatz | `src/carcara/circuits/ansatz.py` | **empty stub** |
| Gates | `src/carcara/circuits/gates.py` | **empty stub** |
| VQE | `src/carcara/algorithms/vqe.py` | **empty stub** |
| Optimizers | `src/carcara/optimizers/optim.py` | header only |
| Hardware backend | `src/carcara/backends/hardware.py` | **empty stub** |
| Error mitigation | `src/carcara/backends/mitigation.py` | **empty stub** |
| Tests | `test/test_wavefunction.py` | wavefunction only |
| Docs | `docs/` (Sphinx + ReadTheDocs) | scaffolded, VQE tutorial stub |

**Implication:** the directory architecture is in place; the core scientific
logic must now be filled in. The roadmap below follows this existing layout so
no restructuring is required.

---

## 3. Guiding Principles

1. **Hardware-first.** Every feature must run on a noisy backend, not only on a
   statevector simulator. Circuit depth and gate count are first-class metrics.
2. **Qiskit-native, but decoupled.** Lean on `qiskit` / `qiskit-nature` /
   `qiskit-ibm-runtime`, while keeping a thin Carcará API so backends can be
   swapped later.
3. **Validate against exact results.** Each numerical component is checked
   against exact diagonalization (small systems) or published reference values.
4. **Reproducible science.** Seeded runs, serializable configs (`pyyaml`), and
   logged provenance for every experiment.
5. **Test-driven.** Unit tests land with each module; no empty stub is marked
   "done" without coverage.

---

## 4. Phased Plan

### Phase 0 — Foundations & Project Hygiene
*Goal: a healthy repository ready for sustained development.*

- Define the public API surface in `__init__.py` (lazy imports of `core`,
  `circuits`, `algorithms`, `optimizers`, `backends`).
- Set up tooling: `ruff`/`black` formatting, `mypy` type checks, `pytest` +
  `pytest-cov`, and a GitHub Actions CI matrix (Python 3.10–3.12).
- Add `optional-dependencies` groups in `pyproject.toml` (`dev`, `docs`, `hw`).
- Establish the logging utility (`utils/logging.py`) as the project-wide logger.
- Populate `CHANGELOG.md` (currently empty) with a Keep-a-Changelog format.

**Deliverables:** green CI, `pip install -e .[dev]` works, contributor guide.
**Acceptance:** CI passes on a trivial PR; coverage report published.

---

### Phase 1 — Core: Hamiltonians & Mappings
*Goal: turn a fermionic system into a qubit operator.*
*Files: `core/hamiltonian.py`, `core/mappings.py`.*

- **`FermionicHamiltonian`**: represent second-quantized operators
  (one- and two-body integrals). Two construction paths:
  - *Molecular*: from geometry via `qiskit-nature` drivers (PySCF) and/or the
    existing `Wavefunction`/ASE pipeline.
  - *Model*: lattice Hamiltonians (Hubbard, Heisenberg, t–J, SSH) built
    programmatically.
- **Mappings** (`mappings.py`): wrap and expose
  - Jordan–Wigner
  - Bravyi–Kitaev
  - Parity (with two-qubit reduction)
  - Optional: symmetry tapering (`Z2Symmetries`) to shrink qubit count.
- Compute the qubit Hamiltonian as a Pauli `SparsePauliOp` plus metadata
  (qubit count, number of terms, locality).

**Deliverables:** `H_qubit = jordan_wigner(H_fermionic)` style API.
**Acceptance:** H₂ (STO-3G) qubit Hamiltonian reproduces the known
ground-state energy under exact diagonalization (−1.137 Ha at equilibrium).

---

### Phase 2 — Circuits: Gates & Ansätze
*Goal: build the parameterized state |ψ(θ)⟩.*
*Files: `circuits/gates.py`, `circuits/ansatz.py`.*

- **`gates.py`**: reusable gate primitives — fermionic excitation gates,
  Givens rotations, particle-number-preserving blocks.
- **`ansatz.py`**:
  - **UCCSD** (and k-UpCCGSD) physics-inspired ansatz via `qiskit-nature`.
  - **Hardware-efficient** templates (parameterized `RY`/`RZ` + entangling
    layers) with configurable depth and connectivity.
  - Reference-state preparation (Hartree–Fock initial state).
- Expose ansatz metadata: parameter count, depth, two-qubit gate count.

**Deliverables:** `ansatz = UCCSD(mapping, n_electrons, n_orbitals)` returning a
parameterized `QuantumCircuit`.
**Acceptance:** UCCSD on H₂ has the expected parameter count; circuit
transpiles to a target basis without error.

---

### Phase 3 — Optimizers
*Goal: classical parameter optimization for hybrid loops.*
*File: `optimizers/optim.py`.*

- Unified optimizer interface (`minimize(cost_fn, x0, ...) -> Result`).
- Gradient-free: **COBYLA**, **SLSQP**, **Nelder–Mead** (via `scipy`).
- Stochastic: **SPSA** (essential for noisy hardware) with calibration.
- Optional gradient-based: parameter-shift gradients.
- Callbacks for convergence tracking and energy-vs-iteration logging.

**Deliverables:** pluggable optimizers consumable by the VQE solver.
**Acceptance:** all optimizers minimize a known quadratic to tolerance; SPSA
converges under simulated shot noise.

---

### Phase 4 — Algorithms: VQE & Variants
*Goal: the hybrid variational solver tying everything together.*
*File: `algorithms/vqe.py`.*

- **`VQE`** solver: takes `(qubit_hamiltonian, ansatz, optimizer, backend)` and
  returns ground-state energy, optimal parameters, and convergence history.
- Expectation-value estimation via `qiskit` primitives (`Estimator`), with shot
  budgeting and operator grouping for measurement reduction.
- **Variants (stretch within this phase):**
  - VQD (excited states),
  - SSVQE,
  - time-dependent / McLachlan variational dynamics ("its variants" per README).
- Result objects are serializable (YAML/JSON) for provenance.

**Deliverables:** end-to-end `VQE(...).run()` on a statevector simulator.
**Acceptance:** H₂ and LiH ground-state energies within chemical accuracy
(≤ 1.6 mHa) of FCI on a noiseless simulator.

> The *fixed-ansatz* VQE of this phase is the baseline. The **adaptive**
> ansatz construction (ADAPT-VQE) is the centerpiece of Phase 5 and reuses this
> solver's expectation/optimization machinery.

---

### Phase 5 — ADAPT-VQE & Operator Pools ⭐
*Goal: adaptively grow a compact, problem-tailored ansatz instead of a fixed
template — the framework's flagship algorithm.*
*Files: `algorithms/adapt.py`, `circuits/pools.py`, `circuits/gates.py`.*

ADAPT-VQE (Grimsley *et al.*, 2019) builds the ansatz one operator at a time.
Each iteration:

```
 1. For every operator A_i in the pool, estimate the energy gradient
       g_i = ∂E/∂θ_i |_{θ=0} = ⟨ψ| [H, A_i] |ψ⟩
 2. Select the operator with the largest |g_i|.
 3. Append exp(θ_k A_k) to the ansatz (θ_k initialized to 0).
 4. Re-optimize ALL parameters {θ_1..θ_k} with the Phase 3/4 VQE loop.
 5. Stop when max_i |g_i| < ε  (or max operators / energy plateau reached).
```

This yields shallower, hardware-friendlier circuits than fixed UCCSD while
recovering most of the correlation energy.

#### 5.1 — Pool abstraction (`circuits/pools.py`)
A common `OperatorPool` interface so pools are interchangeable and
user-extensible:

```python
class OperatorPool(Protocol):
    def operators(self) -> list[PoolOperator]: ...   # generators A_i
    def commutator_with(self, H) -> list[SparsePauliOp]: ...  # for gradients
    def circuit(self, op, theta) -> QuantumCircuit: ...       # exp(θ A_i)
    metadata: PoolMetadata   # size, locality, CNOT cost per element
```

Pools to implement (in priority order):

- **Fermionic pool** — original ADAPT: spin-adapted single + double fermionic
  excitation generators, mapped through Phase 1 (JW/BK/parity). Most accurate,
  deepest circuits (Jordan–Wigner Z-strings).
- **Qubit pool** — qubit-ADAPT (Tang *et al.*, 2021): individual Pauli-string
  generators. Largest pool, shallowest per-operator circuits, hardware-minded.
- **QEB pool — Qubit-Excitation-Based** (Yordanov *et al.*, 2021):
  particle-number-preserving *qubit excitation* generators that drop the JW
  parity strings. Single excitation generator `½(X_pY_q − Y_pX_q)`; double
  excitations are 8-term operators on the 4 involved qubits. Implemented with
  CNOT-ladder circuits whose two-qubit cost is **independent of qubit distance**
  — a major depth win over fermionic excitations.
- **CEO pool — Coupled-Exchange Operators** (Ramôa *et al.*, 2024): groups of
  QEBs that act on the **same set of qubit indices** and therefore can share a
  single entangling (CNOT) structure. Two variants:
  - **OVP-CEO** (one variational parameter): the QEBs in a group combine with
    *fixed* relative weights set by the gradient direction → one parameter,
    one shared circuit.
  - **MVP-CEO** (multiple variational parameters): each QEB in the group keeps
    its own parameter but reuses the shared CNOT ladder → more expressivity at
    essentially the same two-qubit depth.
  CEO pools currently give the best accuracy-per-CNOT among these pools and are
  the recommended default for hardware runs.

A `PoolFactory`/registry lets users select `"fermionic" | "qubit" | "qeb" |
"ceo-ovp" | "ceo-mvp"` by name or register custom pools via entry points.

#### 5.2 — ADAPT driver (`algorithms/adapt.py`)
- **`AdaptVQE`** orchestrating the grow→optimize loop on top of the Phase 4 VQE.
- **Gradient estimation strategies:** exact commutator expectation on
  simulators; measurement-based (grouped Pauli) estimation on hardware;
  optional parameter-shift fallback.
- **Convergence controls:** gradient threshold ε, max operators, energy-change
  plateau, and wall-clock/shot budget.
- **Ansatz growth bookkeeping:** ordered operator list, parameter vector,
  per-iteration energy/gradient history (serializable).
- **Warm starts:** initialize newly added parameter at 0 and reuse previous
  optimum for the rest (ADAPT's key efficiency property).

#### 5.3 — Advanced ADAPT variants (stretch within this phase)
- **Tetris-ADAPT-VQE** (Anastasiou *et al.*, 2024): add multiple
  *disjoint-support* operators per iteration to pack circuit "moments" and cut
  depth/iteration count.
- **Selected/screened pools:** pre-filter the pool by symmetry sector and by
  cheap gradient pre-screening to reduce per-iteration measurement cost.
- **Excited states:** MORE-ADAPT / state-specific penalties reusing the Phase 4
  VQD/SSVQE machinery.
- **Noise-aware selection:** weight operator selection by estimated circuit
  noise (CNOT cost) so cheaper operators win ties — pairs naturally with CEO.

**Deliverables:** `AdaptVQE(H, pool="ceo-mvp", optimizer=..., backend=...).run()`
with a pluggable pool and full convergence history.
**Acceptance (simulator):**
- Fermionic-pool ADAPT reaches FCI within chemical accuracy on H₂, LiH, and a
  4-site Hubbard model.
- QEB- and CEO-pool ADAPT reach the same accuracy with **strictly fewer CNOTs**
  than fermionic-pool ADAPT at equal accuracy (the headline benchmark plot).
- Gradient screening reproduces the exact operator selection on small systems.

---

### Phase 6 — Backends: Real Hardware Execution
*Goal: run on IBM Quantum.*
*File: `backends/hardware.py`.*

- Abstract `Backend` interface; concrete `IBMBackend` via
  `qiskit-ibm-runtime` (Sampler/Estimator primitives, sessions).
- Transpilation pipeline targeting device coupling maps and basis gates;
  layout/routing tuned for low two-qubit depth.
- Local simulators (statevector, noisy `AerSimulator` with device noise models)
  behind the same interface for offline testing.
- Job management: submission, retrieval, retry, and result caching.

**Deliverables:** the Phase 4 VQE and Phase 5 ADAPT-VQE run unchanged against a
real QPU.
**Acceptance:** a small VQE/ADAPT run (e.g. H₂, 2 qubits after tapering)
completes on an IBM device; results logged with backend/calibration metadata.

---

### Phase 7 — Error Mitigation
*Goal: noise-resilient results.*
*File: `backends/mitigation.py`.*

- **Zero-Noise Extrapolation (ZNE)**: gate folding + Richardson/exponential
  extrapolation.
- **Symmetry verification / post-selection** (particle number, spin).
- **Measurement-error mitigation** (readout calibration).
- Pluggable into the estimation step so any solver can opt in.

**Deliverables:** `Estimator(mitigation=ZNE(...))` style composition.
**Acceptance:** mitigated H₂ energy on a noisy simulator is measurably closer
to FCI than the unmitigated run.

---

### Phase 8 — Documentation, Examples & Release
*Goal: usable, citable, published.*

- Complete Sphinx docs: API reference (autodoc) + the VQE tutorial stub in
  `docs/source/tutorial/vqe.md`.
- Worked examples/notebooks: H₂ dissociation curve, Hubbard dimer, a hardware
  run walkthrough, and an **ADAPT-VQE pool comparison** (fermionic vs qubit vs
  QEB vs CEO: energy error vs CNOT count).
- Quickstart in the README's empty "Getting started" section.
- Versioning + PyPI release automation; archive a release on Zenodo for a DOI.
- (Stretch) a short methods/software paper (JOSS-style).

**Deliverables:** docs live on ReadTheDocs; tagged release on PyPI.
**Acceptance:** a new user reproduces the H₂ tutorial end-to-end from `pip install`.

---

## 5. Dependency Graph (phase ordering)

```
Phase 0 ─► Phase 1 ─► Phase 2 ─► Phase 4 ─► Phase 5 ─► Phase 6 ─► Phase 7 ─► Phase 8
                 └────► Phase 3 ─────┘        (ADAPT)   (HW)      (mitig.)   (release)
                 └────► Phase 5 pools ◄───────┘
```

Phases 1–3 can progress partly in parallel; Phase 4 integrates them into a
fixed-ansatz VQE. Phase 5 (ADAPT-VQE) builds on Phases 1, 2, and 4 and is the
flagship; the operator **pools** depend on the Phase 1 mappings and Phase 2
gate primitives, so pool work can begin alongside Phase 2. Phases 6–7 require a
working solver (Phase 4 or 5). Phase 8 runs continuously but is gated for
release at the end.

---

## 6. Cross-Cutting Concerns

- **Testing:** unit tests per module; integration tests for the full pipeline;
  regression tests pinning reference energies. Target ≥ 85% coverage.
- **Validation:** maintain a `benchmarks/` table of (system, method, pool,
  energy, CNOT count) versus exact/published values.
- **Performance:** track circuit depth, two-qubit gate count, shot count,
  pool size, ADAPT iterations, and wall-clock per VQE/ADAPT iteration.
- **Reproducibility:** global seeding; serialize run configs and results;
  serialize the **selected ADAPT ansatz** (ordered operators + parameters) so a
  converged circuit can be replayed without re-running the adaptive loop.
- **Provenance & logging:** every hardware run records backend name,
  calibration snapshot, and transpilation settings.
- **Extensibility:** plugin/registry pattern (entry points) for mappings,
  ansätze, **operator pools**, optimizers, backends, and mitigation methods, so
  users add new ones without editing core.
- **Config-driven experiments + CLI:** describe a full run (system, mapping,
  pool, optimizer, backend, mitigation, budgets) in a YAML file and launch via a
  `carcara run config.yaml` command; results and ansatz emitted as artifacts.
- **Resource estimation:** a dry-run mode that reports qubit count, pool size,
  per-operator and total CNOT cost, and estimated shots **before** submitting to
  hardware.

---

## 7. Milestones Summary

| Milestone | Phases | Definition of done |
|---|---|---|
| **M0 — Repo ready** | 0 | CI green, tooling in place |
| **M1 — Hamiltonian → qubits** | 1 | H₂ qubit Hamiltonian validated |
| **M2 — Ansatz + optimizers** | 2, 3 | UCCSD/HEA built; optimizers tested |
| **M3 — VQE on simulator** | 4 | H₂/LiH within chemical accuracy |
| **M4 — ADAPT-VQE + pools** | 5 | Fermionic/qubit/QEB/CEO pools; CEO ≤ CNOTs of fermionic at equal accuracy |
| **M5 — Hardware run** | 6 | VQE/ADAPT completes on IBM QPU |
| **M6 — Mitigated results** | 7 | ZNE improves noisy accuracy |
| **M7 — Public release** | 8 | Docs + PyPI + tutorial reproducible |

---

## 8. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| NISQ noise dominates results | High | Tapering, shallow ansätze, CEO pool, Phase 7 mitigation |
| ADAPT gradient measurement cost (pool × shots) | High | Commutator grouping, pool pre-screening, symmetry filtering, Tetris batching |
| Qiskit API churn | Medium | Pin versions; isolate Qiskit behind Carcará interfaces |
| Barren plateaus in optimization | Medium | Physics-inspired init; ADAPT is comparatively robust; SPSA, warm starts |
| Hardware queue/cost limits | Medium | Noisy simulator parity; small validated systems first |
| Scope creep (variants/models) | Medium | Treat VQE/ADAPT variants and exotic models as stretch goals |

---

## 9. Key References

*Verify exact citations/DOIs before publication; concepts and lead authors below.*

- **ADAPT-VQE** — Grimsley, Economou, Barnes, Mayhall, *Nat. Commun.* **10**,
  3007 (2019). Adaptive operator-by-operator ansatz growth.
- **qubit-ADAPT-VQE** — Tang *et al.*, *PRX Quantum* **2**, 020310 (2021).
  Pauli-string (qubit) operator pool.
- **QEB-ADAPT-VQE** — Yordanov, Arvidsson-Shukur, Barnes, *Phys. Rev. A* **102**,
  062612 (2021). Qubit-excitation operators with distance-independent CNOT cost.
- **CEO (Coupled-Exchange Operators)** — Ramôa, Anastasiou, Santos, Mayhall,
  Barnes, Economou (2024). OVP/MVP grouping of QEBs sharing entangling structure
  to reduce circuit depth.
- **Tetris-ADAPT-VQE** — Anastasiou *et al.* (2024). Disjoint-support batching of
  operators per iteration.

---

*Living document — update milestones and statuses as phases complete. Keep
`CHANGELOG.md` in sync with each merged phase.*
