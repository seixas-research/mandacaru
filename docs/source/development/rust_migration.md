# Rust migration: Phase 0

The target is a Rust-owned scientific engine with Python retained for the public
interface, command-line tools, visualization, and ASE integration. Phase 0
establishes evidence about the current implementation; it changes no production
algorithms or package dependencies.

## Reference artifacts

The migration tooling lives in `test/migration/`:

| File | Purpose |
| --- | --- |
| `inventory.py` | Static inventory of source hashes, module imports, declared signatures/defaults, result fields, constants, exports, and existing test functions. |
| `capabilities.json` | Capability groups, migration phases, implementation paths, existing test evidence, and reference-workload associations. |
| `cases.json` | Explicit geometries, options, seeds, tolerances, dataset requirements, and validation checks. |
| `baseline.py` | Isolated workload execution, H₂ cache generation, and scientific-output comparison. |
| `suite.py` | Bounded execution of the existing test suite, preserving failures, errors, and skips. |
| `reference/` | Captured source inventory, H₂ Hamiltonian, environment pins, numerical/resource references, and test-suite findings. |

The JSON inventory is **static evidence**. Imports include guarded and optional
imports, and signatures are source expressions rather than resolved runtime
values. Test-function counts are not parameterized test counts or coverage
measurements. The capability index does not certify every combination of its
listed features. A workload association identifies a representative case, not
complete coverage of a group.

Read `test/migration/reference/findings.md` for the measured status and limits of
the initial capture. Failed or unmeasured areas remain visible; they are not
silently removed from the migration scope.

## Reproducing the references

Use Python 3.14 and the repository root as the working directory. The reference
pins describe the captured environment, including transitive dependencies:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r test/migration/reference/requirements.txt
uv pip install --python .venv/bin/python --no-deps -e .

.venv/bin/python test/migration/inventory.py \
  --output .mandacaru/migration/inventory.json

.venv/bin/python test/migration/baseline.py run \
  --output .mandacaru/migration/numpy.json --backend numpy --tier all

.venv/bin/python test/migration/baseline.py compare \
  test/migration/reference/numpy.json .mandacaru/migration/numpy.json

.venv/bin/python test/migration/baseline.py run \
  --output .mandacaru/migration/rust.json --backend rust --tier all

.venv/bin/python test/migration/baseline.py compare \
  test/migration/reference/numpy.json .mandacaru/migration/rust.json

.venv/bin/python test/migration/suite.py \
  --output .mandacaru/migration/suite.json

.venv/bin/python -m pytest test/migration test/test_repo_hygiene.py \
  test/test_single_entry_point.py
```

The Rust comparison measures the **existing integral backend**, with Python
still implementing the surrounding program. It is not a claim that the future
Rust engine exists. Native workloads require an already-built library: the
runner calls `ensure_backend(build=False)` so compilation is outside timed
work. The current developer build command is `mandacaru --build-backend`;
select its backend with `MANDACARU_BACKEND=rust` when a Rust build is needed.

The `run` defaults are three fresh interpreter samples, one requested thread, a 180-second
timeout per sample, and the `quick` tier. `--tier all` also selects external PAW
datasets; `--case ID` selects individual cases. Missing datasets are reported
as skips and yield a nonzero run status. The initial PAW workload inputs are
`pbe-sr/H.parquet` and `lda-sr/Si.parquet` under `MANDACARU_PAW_PATH`. Reports
retain their SHA-256 hashes, so changing a dataset invalidates comparison.

Generated reports and per-sample stdout live under `.mandacaru/`, which is
already ignored. Each sample gets a separate working directory, preventing
automatic output and cache files from leaking into the repository root.
Checked reports retain `runtime_artifact_root`; their per-sample log paths
resolve relative to that directory. Raw logs remain local ignored artifacts.
Timeouts terminate the sample's process group. The runner currently targets
macOS and Linux; its memory accounting and process control use POSIX interfaces.

Regenerate a checked reference only after reviewing why it changes. In
particular, regenerate the H₂ fixture explicitly with the NumPy backend and
fixed threads:

```bash
MANDACARU_BACKEND=numpy MANDACARU_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python test/migration/baseline.py export-h2 \
  --output .mandacaru/migration/h2_hamiltonian.json
```

This writes a candidate fixture. Review it before replacing
`test/migration/reference/h2_hamiltonian.json`. The fixture is a schema-v1
Hartree-valued Jordan–Wigner Hamiltonian exported through the public mean-field
result API, with geometry, basis, grid settings, particle counts, and source
provenance in its metadata. Later cached-Hamiltonian runs read the checked
fixture directly and do not regenerate geometry-dependent integrals.

## What the measurements mean

All workload definitions are coarse **discretized program references**, not
basis/grid-converged physical predictions. The molecular corpus includes H₂,
an open-shell H atom, water in STO-3G, and PAW/PBE H₂. The silicon case includes
a nontrivial k-point mesh. Two cached-H₂ cases exercise full-register dense VQE
and sector/matrix-free ADAPT. Complex operator algebra exercises all four
mappings, including reduced parity.

Checks supplement snapshots with independent invariants: dense versus
matrix-free action, Hermiticity, sector conservation, state normalization,
particle number, small-sector exact diagonalization, ASE/result unit agreement,
k-point weight normalization, a fixed-grid force finite difference, and an
A→B→A geometry round trip. The force case disables translation projection so
the force being checked is the derivative of the discretized energy.

The comparison requires the same workload definitions, fixture, dataset hashes,
case set, numerical shapes/fields, units and representation contract. Every
sample must pass every invariant, and repeated reference samples must agree.
It rejects failed, skipped, empty, nonfinite, or partial evidence. It compares
energies, integral arrays, spectra and state projectors using the declared
absolute and relative tolerances. Projectors avoid an arbitrary global phase
on quantum states. Future orbital/subspace comparisons must also account for
degenerate eigenvector gauges.

Iteration counts and optimizer evaluation counts are diagnostics rather than
equality requirements. Resource fields are also separate from scientific
acceptance:

- `compute_wall_s` measures calculation and snapshot extraction after common
  package imports and backend setup. It includes invariant checks and is a
  reproducible workload measurement, not a kernel-only benchmark.
- `process_wall_s` includes process startup, imports and result serialization.
- `peak_rss_bytes` is the fresh child process's peak RSS, including imports;
  it excludes descendant processes. macOS/Linux reporting units are normalized.
- CPU times and median/minimum/maximum resource measurements are retained.

Reports record Python and dependency versions, platform/CPU/memory information,
BLAS implementations and thread counts, controlled thread environment, source
hash, revision, tooling hashes and loaded native-library hash. The policy
`rust` confirms the integral library selected by the current loader; it does
not assert that every operation in the program is native. The scientific
source digest excludes generated builds and binaries, whose native-library
hash is recorded separately.

Measure performance while other calculations are idle, on the same hardware
and thread configuration. Scientific comparison intentionally applies no
timing threshold. Establish workload-specific performance budgets from repeated
measurements before using performance as a migration acceptance gate. Large
memory-pressure, mixed-precision, disk-buffer, metallic/SOC, stress, advanced
quantum and hardware-provider workloads still need dedicated measured cases;
the capability index and existing suite retain their scope.

The initial thread audit found that Rust shell and two-center kernels call
`std::thread::available_parallelism()` independently of the configured thread
limit. The reports' `threads` and backend-reported thread count describe the
request, not measured concurrency for every kernel. Thus the initial NumPy/Rust
timings are not guaranteed comparisons at equal thread counts. Unifying thread
control is a Phase 1 requirement; scientific-output comparison remains valid.

## Architecture contracts established for Phase 1

1. Preserve the public `Mandacaru` entry point, validated method/option names,
   ASE signs/units, and supported result views. Normalize geometry, units and
   options once at the binding boundary.
2. Let Rust own arrays, solver state, workspaces, caches and iteration loops.
   Define explicit lifetimes for returned NumPy views and mapped buffers;
   Python callbacks should not sit inside numerical hot loops.
3. Keep Hartree/Bohr, physicists' tensor notation, complex conjugation,
   spin-blocked ordering, qubit endianness, exact/Trotter semantics, and
   energy/free-energy distinctions explicit in Rust types and tests.
4. Reuse the existing native kernels behind checked slice/shape interfaces.
   Audit unsafe pointer access, integer sizes, aliasing, thread partitioning
   and error propagation before expanding the binding surface.
5. Preserve direct/packed/Cholesky/matrix-free and workspace-reuse strategies.
   Use a calculation-scoped memory/thread budget rather than independent
   nested thread pools or mutable process-global buffer ownership.
6. Prove linear algebra, FFT, sparse exponential and optimizer replacements
   against the references before expanding their scope. No numerical package
   replacement is selected solely because it is written in Rust.
7. Resolve force differentiation and learned-model training currently supplied
   by JAX, and execution/transpilation currently supplied by Python quantum
   SDKs. Temporary adapters may support intermediate milestones; they do not
   satisfy the final Python-only-interface/ASE target.

The next deliverable is the Rust workspace and binding foundation. The first
functional workflow after that is the checked cached-H₂ Hamiltonian solved
entirely in Rust, followed by an ASE-geometry-to-Hamiltonian workflow. Existing
test failures are investigation items, not behavior to reproduce deliberately.

## Phase 0 acceptance

- A reproducible source/API/dependency inventory and capability/test index exist.
- A small, explicit corpus has numerical snapshots and isolated resource data.
- A portable cached-Hamiltonian input exists for the first Rust solver milestone.
- Missing inputs, failures, test-scope limits and unmeasured areas are recorded.
- Baseline/comparison tooling is tested for rejecting misleading evidence.
- Production implementation and existing tests remain unchanged.

A green existing suite is not asserted by these criteria. Any incomplete suite
capture remains explicitly incomplete; Phase 1 must investigate relevant
failures and avoid introducing additional regressions.
