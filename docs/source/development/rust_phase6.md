# Rust migration: Phase 6 advanced quantum and correlation milestone

This milestone moves shared-subspace UCCSD VQE/ADAPT, ordered Pauli time
propagation, finite-register QPE, and real closed-shell MP2 natural-orbital
analysis into the Python-independent Rust core. Python selects options, converts
input/output arrays, exports circuits and integrates with ASE. Native numerical
loops release the interpreter and call no Python numerical routines.

The broad Phase 6 capability groups remain partially migrated. The remaining
work is listed explicitly below; selecting Rust never silently runs a Python
solver for an unsupported method. The default engine remains Python.

## Shared-subspace VQE and ADAPT

```python
from mandacaru import Mandacaru

calc = Mandacaru(
    method="subspace-vqe", engine="rust",
    load_hamiltonian="h2_hamiltonian.json", num_states=2,
    ansatz={"name": "uccsd", "trotter": False},
    optimizer={"method": "L-BFGS", "tol": 1e-9},
    engine_options={"threads": 1}, trace=False,
)
result = calc.run()
energies = result.energies                 # eV by default
levels = result.levels
```

The same geometry, native mean-field model, cache, mapping and sector paths
used by native VQE work here. For example, pass
`**reference.as_quantum_problem()` from a native RHF result instead of
`load_hamiltonian`. ASE energy requests report the lowest level.
`method="subspace-adapt-vqe"` uses the native fermionic pool and a shared
product ansatz. `max_iterations` limits growth; optimizer `maxiter` and `maxfev`
limit each inner optimization. Both BFGS and L-BFGS are available.

Rust enumerates determinants in the fixed alpha/beta sector, ordered by
excitation count from HF then lexicographically by occupied modes. It evolves
one shared unitary and minimizes the weighted sum of expectations. Weights
default to `(k, k-1, ..., 1)` and must be finite, positive and strictly
decreasing. Exact summed UCC uses differentiated exponential actions;
the product form propagates exact forward derivatives. ADAPT screens the
weighted sum of pool gradients across every reference and preserves an inner
optimizer failure as nonconvergence. Evaluation counts are weighted objective
calls; screening and final state reconstruction are additional work.

Reported levels are **bare expectations**, sorted together with their states
and reference indices, matching the existing Python semantics. They are not
Ritz eigenvalues obtained by diagonalizing the final subspace. States use the
problem's full or sector-compressed basis. States are orthonormal because the
references share a unitary; finite ansatz expressivity and optimizer convergence
still determine their energy accuracy. Native UCCSD does not promise a fixed
total spin. Spin-adapted HVA references are refused.

## Time propagation and QPE

```python
import numpy as np
from mandacaru.core.mapping import PauliSum
from mandacaru.workflows.time_evolution import SuzukiTrotter
from mandacaru.workflows.qpe import QuantumPhaseEstimation

h = PauliSum({"X": 0.4, "Z": 0.3, "I": -0.7}, num_qubits=1)
psi = np.array([1., 0.], dtype=complex)
propagator = SuzukiTrotter(h, order=4, engine="rust")
evolved = propagator.evolve(psi, 0.7, steps=3)
recovered = propagator.evolve(evolved, 0.7, steps=3, inverse=True)
qpe = QuantumPhaseEstimation(
    h, n_evaluation_qubits=6, engine="rust", verbose=False,
    engine_options={"threads": 1},
).run(psi)
```

`time_evolve` also accepts `engine` and `engine_options`. Energies inside the
engine are Hartree and times are atomic units, with qubit zero the most
significant bit. The propagator snapshots insertion-ordered terms, isolates
the identity phase and uses exact Pauli rotations. First order and positive
even Suzuki orders are supported. Inversion reverses both rotation order and
signs, including at first order. Orders whose uncompressed schedule exceeds
one million entries are refused before construction. This is full state-vector
simulation, not a polynomial-memory many-body approximation.

QPE preserves the existing energy-window margin, FFT sign, probability bins,
units, peaks and optional collapsed states. Up to twelve system qubits it uses
a native Hermitian eigensolve; above that it applies a matrix-free scaled Taylor
exponential for successive powers. Both paths use RustFFT for the evaluation
register transform. Neither uses a Trotter approximation. The native context's
memory budget is an additional hard limit beyond the public memory estimate;
larger dense problems may need an explicitly increased context budget.
Individual overlaps within degenerate eigenspaces depend on the eigenbasis.
Ground-state overlap now sums the whole degenerate ground eigenspace for both
engines (absolute degeneracy tolerance `1e-10 Ha`).

Rust QPE takes a prepared full-register state. Checkpoint decoding remains a
Python input adapter; tuple-based circuit preparation is explicitly refused
for native simulation in this milestone. Phase 7 adds native checkpoint and
tuple preparation, superseding this restriction. Circuit export still uses the Python Qiskit integration.
Echo/OTOC orchestration and sampled spectroscopy have not migrated with this
propagation primitive.

## MP2 and frozen natural orbitals

```python
reference = Mandacaru(method="rhf", engine="rust", trace=False).run(
    geometry=atoms)
mp2 = reference.mp2_natural_orbitals()
correlation_ha = mp2.correlation_energy
occupations = mp2.occupations
rotation = mp2.rotation
```

This path analyzes the owned native spatial model without exporting its
integrals for a Python calculation. It requires a converged RHF reference.
It returns the existing `MP2Result`; the correlation energy is always in
Hartree, independently of the reference result's output units. With a reduced
model it analyzes that model, including its frozen-core effective integrals.

Array callers can use
`mp2_natural_orbitals(h, eri, n_occ, engine="rust")` with dense spatial
physicists' integrals, or
`mp2_natural_orbitals_from_block(oovv, energies, n_occ, canonical, engine="rust")`
with an already semicanonical occupied/virtual block. Packed Python tensor
objects are explicitly refused by the dense adapter. Native molecular inputs
reuse their model directly.

Rust builds the closed-shell Fock matrix and diagonalizes occupied and virtual
blocks separately when their off-diagonal entries exceed the existing
`1e-4 Ha` SCF tolerance. Four rectangular contractions form the semicanonical
`oovv` tensor without materializing a rotated full `M^4` tensor. Correlation
energy and spin-summed unrelaxed density use the closed-shell exchange
prefactors. Only the virtual density is diagonalized and ranked in descending
occupation; occupied orbitals retain semicanonical order. The full density
conserves particle count, and the returned rotation preserves occupied/virtual
blocks. The near-zero denominator threshold is `1e-6 Ha`; the imaginary-part
tolerance on real integrals is `1e-8`.

This adds analysis, not automatic MP2 active-space selection or a standalone
`method="mp2"` calculator. Native active-space reduction still uses the earlier
energy/index selection contract.

## Validation and remaining scope

`test/migration/test_native_advanced.py` compares ordered propagation and its
discrete adjoint with the Python formula, dense exponentials and exported
circuits; checks dense and matrix-free QPE distributions/collapsed states;
compares MP2 density/energy with the Python implementation and independent
spin-orbital prefactors; and reconstructs shared-unitary states independently
under all four mappings. It also covers context ownership and budget failures,
invalid inputs, nonconvergence and numerical-callback exclusion.
`crates/core/tests/advanced.rs` exercises these components without Python.
Installed-wheel smoke checks include all four new paths with runtime compilers
disabled. `test/migration/phase6/validation.json` records exact outcomes and
artifact/source hashes. Frozen Phase 0 references are unchanged.

Pending in the broad Phase 6 scope:

- Variational deflation, spin-adapted/general subspace ansatz families,
  MCAS-VQE/VALQA, learned GP/ranker/GNN proposals, JAX training and expressivity.
- Echo/nested-OTOC orchestration and sampled response/spectral analysis beyond
  propagation and QPE.
- Open-shell and DLPNO-MP2, native packed/direct MP2 inputs, automatic MP2
  active-space selection and integral rotation/reduction for those selectors.
- Charge partitions, magnetic properties, real-space dipoles/volumetric fields,
  interaction energies and general spectroscopy workflows.

Previously documented Phase 4/5 limitations remain. This milestone does not
assert that every Phase 6 capability or the full scientific suite is migrated.
