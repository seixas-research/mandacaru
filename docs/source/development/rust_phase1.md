# Rust migration: Phase 1 foundation

Phase 1 adds the Rust workspace, checked engine types, Python bindings and native
wheel packaging. The existing `Mandacaru` calculator and numerical algorithms
retain their public interface. This is the foundation for moving solver loops;
the cached-Hamiltonian solver itself belongs to Phase 2.

## Layout and build

The current workspace uses the repository-root `Cargo.toml` and separate crates
in `crates/mandacaru-core`, `crates/mandacaru-kernels` and
`crates/mandacaru-python`. The original nested integral workspace was removed;
see [the current architecture](architecture.md). This page records the earlier
milestone and does not define the current public interface.

## Engine contracts

`Geometry` owns atomic numbers, positions, lattice and periodic axes; positions
and cells normalize to Bohr once at construction. The Python-only ASE adapter
`geometry_from_atoms` snapshots ASE data without retaining the ASE object.
Dummy atomic numbers are refused; ghost basis specifications remain a separate
solver concern. Partially periodic geometries permit zero unused cell vectors.

`Grid` owns a checked index shape, symmetric positive-definite inverse metric
and positive volume element. All shape products, ABI conversions and allocation
byte counts are checked before entering a kernel.

`Context` owns its thread limit, shared byte budget, memory high-water accounting
and worker counters. `Workspace` storage keeps its reservation alive even when
the creating Python context is released. Calls on the same context serialize
kernel execution; independent contexts can run concurrently with distinct
thread limits. No context mutates environment variables to control workers.

The checked interfaces cover one-body integrals, double/single real Laplacians,
complex projector contractions, spherical Bessel and Fourier-Bessel transforms,
radial tridiagonal eigenpairs, Numerov recurrences and real shell evaluation.
Inward Numerov is the homogeneous recurrence and explicitly refuses a nonzero source.
They operate in atomic units. Type hints live in `src/mandacaru/_native.pyi`.
The finite-difference stencils use the existing zero-outside-domain boundary;
periodic FFT operators remain in their existing implementation for now.

```python
import numpy as np
from mandacaru import _native

context = _native.Context(threads=2, memory_limit_bytes=64 * 2**20)
grid = _native.Grid.orthogonal([8, 8, 8], [0.5, 0.5, 0.5])
field = np.ones((1, 8, 8, 8), dtype=np.float64)
laplacian = context.laplacian(grid, field)
```

Inputs require matching native NumPy dtypes, aligned C-contiguous storage and
valid shapes. Bindings reserve input storage and copy it into Rust before
detaching Python. Results own independent NumPy allocations, so changing or
retaining one cannot invalidate a Rust workspace. No borrowed or writable Rust
views are exposed. Read the native workspace's `SAFETY.md` for the raw ABI
preconditions and the boundary between checked and legacy calls.

The byte budget covers Rust input snapshots/workspaces/output storage and
estimated large kernel scratch arrays. It is not a hard process-memory limit;
returned NumPy allocations, SDK memory, allocator overhead, worker stacks and
legacy unchecked calls remain outside it. The module explicitly uses the GIL;
full free-threaded array/view support is not claimed.

## Thread-policy correction

Every legacy parallel kernel now uses the shared policy, including the shell
and two-center paths that ignored the limit in Phase 0. Precedence is the active
context, `MANDACARU_NUM_THREADS`, `OMP_NUM_THREADS`, then available hardware.
Counts are positive and capped at 1,024; explicit invalid context limits raise
errors. `worker_peak` and `workers_launched` observe actual checked-call worker
activity. `THREAD_POLICY_VERSION=1` is available through both bindings and C ABI.

Phase 0 reports remain historical evidence of the earlier thread inconsistency.
Their timing table is not retroactively a comparison at equal thread counts.
Phase 1 numerical verification compares current outputs with those frozen
scientific references; it does not claim a new performance acceptance threshold.

## Packaging and verification

Build wheels and source archives with the normal PEP 517 build frontend. To
check an installed wheel, provision a separate environment with the pinned
runtime dependencies and that wheel, then run:

```bash
.venv/bin/python test/migration/check_wheel.py path/to/mandacaru.whl \
  --python path/to/isolated-env/bin/python \
  --output .mandacaru/migration/phase1/wheel.json

.venv/bin/python test/migration/baseline.py run \
  --output .mandacaru/migration/phase1/rust.json \
  --backend rust --tier all --repeat 1

.venv/bin/python test/migration/baseline.py compare \
  test/migration/reference/numpy.json .mandacaru/migration/phase1/rust.json
```

The wheel check verifies native tags, binary inclusion, README transformation
and entry points, then starts the isolated interpreter from a temporary working
directory. It verifies installed-package paths, checked native execution,
legacy ABI loading, ASE H₂ RHF and cached-H₂ VQE while blocking runtime compiler
discovery/invocation. The migrated Rust solver is not used for the latter yet.

Validation covers malformed shapes/metrics, unsupported layouts/dtypes,
nonfinite input, byte budgets, buffer lifetimes, independent contexts, shared
context limits, SciPy/NumPy numerical references and Phase 0 scientific outputs.
The local platform is macOS arm64/CPython 3.14; Linux, Windows and free-threaded
builds still require their own packaging matrix. Existing broader-suite failures
from Phase 0 remain visible and are not recast as successful validation.

Saved evidence is in `test/migration/phase1/validation.json` and `wheel.json`.
The final binding/API/documentation run passed 142 tests; eight Rust tests and
all 11 numerical-reference workloads passed. A broader kernel run had 568 passes,
five skips and four failures. All four were reproduced against a separate copy
whose scientific source files matched the exact Phase 0 manifest: two BLAS
introspection assumptions, the existing force-result cleanup behavior and a
packed-integral round-off threshold. Those tests and algorithms were not weakened
to obtain a green result. The source archive also rebuilt in a fresh source tree
and fresh Cargo target directory with network access disabled.

## Next component boundary

Phase 2 can implement operators, sectors, matrix-free application and local
optimization in `mandacaru-core`, then expose a complete cached-H₂ workflow
through the binding crate. The new core contains no Python or ASE dependency.
Replacing JAX differentiation/training, full linear algebra/FFT stacks and
quantum-SDK execution/transpilation remains gated work for their owning phases;
no replacement library is selected by this foundation alone.
