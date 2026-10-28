# Rust migration: Phase 2 cached quantum workflow

The first complete native workflow loads a cached Hamiltonian and solves it
with VQE or basic ADAPT-VQE. Rust owns the cache parser, operators, basis,
excitation pool, states, derivatives, optimizer, screening and growth loop.
Python validates interface options, converts output units, and constructs the
existing `VQEResult` / `ADAPTVQEResult` objects.

```python
from mandacaru import Mandacaru

calc = Mandacaru(
    method="adapt-vqe", engine="rust",
    load_hamiltonian="test/migration/reference/h2_hamiltonian.json",
    pool="fermionic", sector=True, sparse="matrix-free",
    optimizer={"method": "L-BFGS", "maxiter": 300, "tol": 1e-9},
    convergence={"gradient": 1e-7, "energy": 1e-8},
    engine_options={"threads": 1, "memory_limit_bytes": 536870912},
    trace=False,
)
result = calc.run()
print(result.optimal_energy)  # approximately -25.8718954554466 eV
```

`engine="python"` remains the default. The native engine is selected explicitly;
it never falls back to Python scientific implementations for an unsupported
option. Integral-backend selection (`MANDACARU_BACKEND`) remains independent of
solver-engine selection. Native cached solves need no integral backend.

For fixed UCCSD use `method="vqe"`; select the ordered product with
`ansatz={"name": "uccsd", "trotter": True}`. The default retains the exact
exponential of the summed cluster operator. `include_singles=False` is also
supported. VQE starts from zero amplitudes unless `run(initial_parameters=...)`
is supplied. Basic ADAPT starts with an empty ansatz.

## Implemented boundary

The Python-independent `mandacaru-core::quantum` modules provide:

| Component | Native behavior |
| --- | --- |
| Pauli algebra | Complex sums, products, adjoints, commutators and matrix-free action; deterministic mask representation. |
| Fermionic mapping | Ordered ladder products and sums, Jordan–Wigner, parity, Bravyi–Kitaev and fixed two-qubit parity reduction. |
| Basis | Full registers or fixed `(n_alpha, n_beta)` sectors, enumerated from combinations without scanning `2**N` determinants. |
| Hamiltonian application | Dense (`sparse=False`), compressed columns (`sparse=True`) or matrix-free (`"matrix-free"` / `"auto"`). |
| UCCSD / fermionic pool | Spin-conserving occupied-to-virtual singles and doubles in the same order and normalization as Python. |
| Evolution | Scaled Taylor exponential actions with convergence/normalization checks; exact forward derivatives of both summed and ordered-product exponentials. |
| Optimization | BFGS and L-BFGS with strong-Wolfe line search, explicit iteration/evaluation budgets, analytic gradients and truthful termination status. |
| ADAPT | Full pool screening, deterministic largest-gradient selection, warm-start joint reoptimization, gradient/energy convergence and per-growth records. |
| Properties | Complex spin-orbital one- and two-particle RDMs, including reduced parity, in physicists' tensor notation. |

The register limit is 62 modes, with allocation feasibility determined by the
context memory budget. A 40-mode `(1,1)` sector has 400 states; its full register
is refused before allocation under the test budget. Sector invariance checks
combine Pauli terms sharing a flip mask before deciding whether an operator
leaks, so legitimate cancellations survive. A nonconserving Hamiltonian is
rejected when a sector is requested.

The schema-v1 JSON loader validates the format/version, finite coefficients,
Hermiticity, mapping, dimensions and particle counts. Coefficients are Hartree;
an explicit metadata energy unit other than `Ha` is refused. Nuclear and other
constant energies remain in the identity term. Input JSON is limited to 64 MiB
and parsed under a conservative memory reservation. Large caches should await
the native binary/Parquet input milestone rather than bypass this limit.

Numerical buffers, matrices, operator intermediates and optimizer workspaces
use the Phase 1 context budget. Output histories reserve their maximum size;
ADAPT therefore reserves `max_growth * max_evaluations * 8` history bytes.
Reservations are released on failure and when native results/problems are
dropped. NumPy outputs own independent copies. Python objects and those output
copies are outside the native allocation budget. These quantum loops currently
run on one thread; the context's requested thread limit is a ceiling, not a
promise of parallel speedup.

An inner optimization failure stops ADAPT and records the failure. The final
pool gradient is recomputed at the returned state even on budget exhaustion.
Evaluation counts describe optimizer objective calls, including line-search
trials; reference, final-state and pool-screening evaluations are separate.
The public adapter returns energies/histories in eV, or Ha with
`atomic_units=True`. Screening gradients retain the existing Hartree convention.
`txt=` writes a compact native termination report.

## Standalone use and validation

No Python interpreter is required for either solver:

```bash
cargo run --offline --manifest-path Cargo.toml \
  --package mandacaru-core --example solve_cached -- \
  test/migration/reference/h2_hamiltonian.json

cargo run --offline --manifest-path Cargo.toml \
  --package mandacaru-core --example solve_cached -- \
  test/migration/reference/h2_hamiltonian.json adapt

cargo test --offline --manifest-path Cargo.toml \
  --package mandacaru-core --package mandacaru-kernels

.venv/bin/python -m pytest test/migration/test_native_quantum.py
```

Both standalone runs must converge to `-0.9507746213870719 Ha` from the
`-0.9371240086056045 Ha` reference. ADAPT selects `D(0,2->1,3)`.

Tests compare all four mappings against Python complex fermion algebra,
anticommutation relations and matrix-free action. Random noncommuting UCCSD
amplitudes are checked against independent SciPy dense exponentials and central
finite differences, in full and sector spaces. RDM tests check Hermiticity,
antisymmetry, particle number and contraction identities. H₂ energies and state
projectors match the frozen Phase 0 references across Hamiltonian representations
and both optimizers. Native tests also solve Rosenbrock, exercise a limited-memory
optimization that cannot allocate a dense Hessian, and check failure/budget cleanup.

`check_wheel.py` now exercises both native solvers outside the checkout with
compiler invocation disabled. The Phase 1 wheel/sdist build procedure still
applies; the new code and locked dependencies travel in the existing extension.

Recorded results are in `test/migration/phase2/validation.json`: 210 migration,
API and documentation tests pass (including 68 quantum binding tests), and all
14 Rust tests pass. The broader selected regression run has 647 passes and two
failures; both failures reproduce on the frozen Phase 0 source: the CEO/QEB
gate-count ratio and the profiling assertion that expects `openmp_threads` on
this Accelerate-based environment. The macOS arm64 CPython 3.14 wheel smoke and
fresh offline source-archive build pass. This is not a full-suite green claim.

## Remaining capability coverage

This milestone establishes the complete cached-H₂ workflow identified after
Phase 1. The Phase 0 capability index also groups broader quantum features under
Phase 2; that index is not a claim that every feature has migrated. The following
remain Python implementations and are refused by the opt-in native adapter:

- General Z₂ symmetry discovery/tapering beyond fixed parity reduction, total-N
  spin-mixing sectors and spin-orbit pools.
- HVA, qubit/QEB/CEO pools, adjoint gradients, screened subsets, TETRIS and pruning.
- SLSQP, CG, COBYLA, Nelder–Mead, SPSA, constraints and shot-based optimization.
- Hardware/provider execution, checkpoints/resume, callbacks, excited-state
  deflation, Parquet loading and dry-run estimates.

ASE geometry-to-Hamiltonian construction, forces, and real-space properties
belong to subsequent migration work. Selecting `engine="rust"` does not imply
those paths have migrated. Existing Python workflows remain available throughout.
