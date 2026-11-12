# Native boundary contracts

`mandacaru-core` is the checked Rust entry point. `mandacaru-python` copies
validated NumPy inputs into Rust before detaching Python, then returns independent
NumPy allocations. `mandacaru-kernels` has no C ABI (its exported symbols were
removed on 2026-11-09): its raw-pointer functions are `pub unsafe fn`s called
from `mandacaru-core` only, and every call requires the contracts below.

## Common requirements

- Every pointer must be aligned, non-null, and live for its declared extent.
  Read-only inputs may overlap other read-only inputs. Mutable outputs must not
  overlap inputs or other outputs. No caller may mutate input storage while a
  kernel reads it. All scoped workers join before a call returns.
- Dimensions must be positive, products must fit `usize`, `isize` byte extents
  and the declared integer types, and buffers must cover the complete
  products. The signatures use `i32` for small dimensions and `i64` for grid
  counts/indices. The `#[repr(C)]` complex structures contain consecutive real
  and imaginary components.
- Arrays use C order. Geometry, grid spacing, inverse metrics, potentials,
  quadrature and integral coefficients are in atomic units. The inverse metric
  is a symmetric positive-definite 3×3 matrix. Physical scalars and input fields
  must be finite, with positive grid volumes and spacings.

## Kernel-specific extents

| Family | Required input/output contract |
| --- | --- |
| One-body | `psi[M,G]`, `potential[G]`, inverse metric `[9]`, and separate kinetic/local outputs `[M,M]`; `G=nx*ny*nz`. Each worker owns disjoint output columns and a private Laplacian scratch field. |
| Laplacian | Input and separate output `[N,nx,ny,nz]`, inverse metric `[9]`; workers own disjoint field/x planes. |
| Projectors | `psi[M,G]`, `chi[P,G]`, output `[M,P]`; workers own disjoint orbital rows. |
| Real/Bloch shells | Coordinates `[3,G]`, center `[3]`, strictly increasing knots `[B]` with `B>=2`, interval-major cubic coefficients `[B-1,4]`, valid angular indices `-l..l`, derivative `-1..2`, destination row indices inside `M`. Bloch inputs additionally need translations `[I,3]`, phases `[K,I]` and output `[K,M,G]`. Real outputs use `[M,G]`. Destination rows must be unique to prevent overlapping writes. `bloch::bloch_shell_weighted` (Rust-internal, no `extern` symbol) is `bloch_shell` plus one by-value `weight` in `-1..=2`: `-1` is the plain sum, `0..=2` is allowed only with `derivative` in `0..=2` and multiplies every image's derivative by that component of its own argument `r - center - R` (the strain tangent); it reads no extra memory, so the pointer contract above is its whole contract. `bloch::bloch_shell` keeps its signature and calls it with `-1`. |
| Bloch projection | Shell inputs above, columns `[P,G]` and output `[I,nm,P]`; all dimensions and derived block counts must be positive. The point blocks form fixed groups of `PROJECT_GROUP_BLOCKS` (independent of the thread count); each group sums into a private copy, at most one group per worker at a time, and the copies are added in group order. |
| Two-center | Radii `[nr]`, wave numbers `[nk]`, Bessel output `[nr,nk]`; Fourier-Bessel weights `[nr]` and output `[nk]`. Parallel output ranges are disjoint. |
| Numerov | Coefficients, source and solution all `[N]`, `N>=3`, valid boundary below `N-1`, two initial boundary values, positive squared step and nonsingular recurrence denominators. Solution is the sole mutable buffer. |
| Eigenpair | Finite diagonal `[N]`, off diagonal `[N-1]`, `N>=1`, `0<=index<N`, scalar eigenvalue output and independent vector `[N]`. Scratch arrays belong to the call. |
| One-center radial | Matching radial/quadrature/density extents, valid nonnegative multipoles, increasing positive radii where powers/division are evaluated, and independent outputs (`onecenter::mandacaru_radial_coulomb`). |

The checked Phase 1 interfaces cover one-body, both real Laplacian precisions,
complex projectors, Bessel/Fourier-Bessel, radial eigenpairs, Numerov, real
shells, and the Bloch shells and Bloch projections of the periodic setup
(`kernels::bloch_shell`, `kernels::bloch_shell_weighted`, `kernels::bloch_project`: the two complex types share
layout, asserted at compile time). They validate integer conversions, shape products, finiteness, spline
sizes and angular/derivative bounds before allocating independent outputs and
entering unsafe kernels. The remaining raw functions (the one-center radial
Coulomb) are validated by their `mandacaru-core` caller.

## Ownership, memory and threads

Calculation contexts share atomic byte reservations across their clones.
Rust-owned workspaces keep reservations alive independently of the creating
context. Calls on one context serialize kernel execution, while different
contexts may execute concurrently. The thread-local policy is restored on
unwind and propagated to every kernel worker; worker counters observe launches
and peak activity. The same policy parser serves every legacy parallel kernel.

The byte budget covers checked input snapshots, persistent Rust workspaces,
kernel output storage, and explicitly estimated large scratch arrays. It is
not a process-RSS limit: Python/NumPy-owned results, allocator overhead, worker
stacks, SDKs and unwrapped legacy paths are outside it. Returned NumPy arrays
do not borrow Rust context storage. No memory-mapped Rust views are exposed in
Phase 1; the existing Python mapped-buffer implementation remains separate.

The binding declares `gil_used=true`. Copying input data precedes interpreter
detachment; detached work touches only Rust-owned arrays. Full free-threaded
NumPy aliasing and writable-view support require a separate audit.

## Test-only counting allocators

Five test modules install a counting `#[global_allocator]`
(`mandacaru-core/src/parquet/codec/zstd.rs` under `#[cfg(test)]`,
`mandacaru-core/tests/parquet_errors.rs`, `tests/parquet_memory.rs`,
`tests/paw_onecenter_memory.rs` and `tests/periodic_load_memory.rs`). They
exist to prove that the memory a decoder really allocates stays below the bytes
reserved from the `Context`. They are never part of a library build. The
contract: every `alloc`, `dealloc` and `realloc` is forwarded unchanged to
`System`, so the `GlobalAlloc` obligations (layout fidelity, alignment, no
unwinding) are `System`'s; the bookkeeping is atomic or thread-local counters
that never allocate, and thread-locals are read with `try_with` so an allocation
made during thread teardown is ignored instead of re-entering the allocator. A
test that needs the process-wide counter serializes itself behind a mutex. An
allocator that refuses growth (the emulated hard limit in `parquet_memory.rs`)
returns a null pointer, as the trait allows.

`paw_onecenter_memory.rs` measures the heap of `exact_atom` against its
reservation with thread-local counters only, so it holds one test (a second
test in the binary would share the harness's baseline); it is a separate file
because a `#[global_allocator]` owns its whole test binary.
`periodic_load_memory.rs` does the same for `Crystal::load` (the heap of a saved crystal
against the reservation of the load), under the same one-test rule.

`paw_record_memory.rs` is a sixth process-wide counting allocator, under the
same contract as `parquet_memory.rs` (it forwards to `System` unchanged, keeps
atomic counters that never allocate, serializes its measurements behind a mutex
and refuses growth past an emulated hard limit by returning null). It measures
`paw::record::PawDataset::read`: the live bytes of a read of `pbe-fr/Pb`, of a
hostile payload, of a footer declaring 2^40 rows and of byte-mutated dataset
files never exceed `Context::memory_peak()` plus a fixed 1 MiB. Like the
others it is a separate test binary because a `#[global_allocator]` owns its
whole test binary, and it never ships in a library build.

`archive_tile_memory.rs` is a seventh, with thread-local counters only (the same
contract as `periodic_load_memory.rs`, one test per binary). It measures the heap
of a scratch-array read: a tile of a Bloch orbital stack read from disk
(`ComplexTiles::tile`) allocates nothing (live growth and total bytes both zero,
for the float64 and the float32 archive), and the projection and the density
accumulation over a stack on disk take no more heap than the same products over
the stack in memory, plus the one tile of scratch they reserve from the
`Context`. The scratch arrays themselves (`mandacaru-core/src/math/archive.rs`)
use no `unsafe`: positional reads go through `FileExt::read_exact_at` into a
fixed stack block, and the descriptor count is bounded by `MAX_OPEN_HANDLES`.

`periodic_description_memory.rs` is an eighth, thread-local counters only (the contract of
`periodic_load_memory.rs`, one test per binary). It measures the heap of
`Description::parse` and `PointGroup::parse` on a reduced-mesh snapshot that carries a `setup`
block against the bytes they reserve from the `Context`, and that a `Context` too small for them
is refused with zero bytes of heap taken (the reservation precedes the first allocation).

## Returning freed heap pages

`mandacaru-kernels/src/heap.rs` declares one foreign function per supported
platform and calls it with no pointer from this program: on macOS
`malloc_zone_pressure_relief(NULL, 0)` (a null zone is every zone of the
process, a zero goal is as much as possible) and on glibc Linux
`malloc_trim(0)` (no slack kept at the top of the heap). Both only hand pages
the allocator already holds free back to the operating system: they read and
write allocator metadata under the allocator's own lock, never program data, so
live allocations, their addresses and their contents are untouched, and a later
allocation simply takes new pages. They are called from any thread and may be
called concurrently with allocation and deallocation (the allocator
synchronizes them). Any other platform compiles a function that does nothing
and reports `false`. The unit test in the same file keeps a large live vector
across the call and checks it afterwards.
