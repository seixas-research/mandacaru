# Native boundary contracts

`mandacaru-core` is the checked Rust entry point. `mandacaru-python` copies
validated NumPy inputs into Rust before detaching Python, then returns independent
NumPy allocations. `mandacaru-kernels` retains the historical raw C ABI;
calling that ABI directly still requires the contracts below. Adding a Python
extension does not turn arbitrary raw-pointer calls into checked operations.

## Common requirements

- Every pointer must be aligned, non-null, and live for its declared extent.
  Read-only inputs may overlap other read-only inputs. Mutable outputs must not
  overlap inputs or other outputs. No caller may mutate input storage while a
  kernel reads it. All scoped workers join before a call returns.
- Dimensions must be positive, products must fit `usize`, `isize` byte extents
  and the declared ABI integer types, and buffers must cover the complete
  products. Rust's legacy ABI uses `i32` for small dimensions and `i64` for
  grid counts/indices. Its `#[repr(C)]` complex structures contain consecutive
  real and imaginary components. The historical C implementation uses C `long`
  in its corresponding slots; the Python binder selects scalar widths per ABI.
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
| Real/Bloch shells | Coordinates `[3,G]`, center `[3]`, strictly increasing knots `[B]` with `B>=2`, interval-major cubic coefficients `[B-1,4]`, valid angular indices `-l..l`, derivative `-1..2`, destination row indices inside `M`. Bloch inputs additionally need translations `[I,3]`, phases `[K,I]` and output `[K,M,G]`. Real outputs use `[M,G]`. Destination rows must be unique to prevent overlapping writes. |
| Bloch projection | Shell inputs above, columns `[P,G]` and output `[ceil(G/BLOCK),I,nm,P]`; all dimensions and derived block counts must be positive. Workers own different point blocks. |
| Two-center | Radii `[nr]`, wave numbers `[nk]`, Bessel output `[nr,nk]`; Fourier-Bessel weights `[nr]` and output `[nk]`; two-center quadrature weights `[nk]`, distances `[nR]` and output `[nR]`. Parallel output ranges are disjoint. |
| Numerov | Coefficients, source and solution all `[N]`, `N>=3`, valid boundary below `N-1`, two initial boundary values, positive squared step and nonsingular recurrence denominators. Solution is the sole mutable buffer. |
| Eigenpair | Finite diagonal `[N]`, off diagonal `[N-1]`, `N>=1`, `0<=index<N`, scalar eigenvalue output and independent vector `[N]`. Scratch arrays belong to the call. |
| One-center radial | Matching radial/quadrature/density extents, valid nonnegative multipoles, increasing positive radii where powers/division are evaluated, and independent outputs. Slater waves have `[nw,N]` storage and tensor output `[nw,nw,nw,nw]`. These raw operations remain behind the existing Python validation paths. |
| Localized | Per-atom half-open boxes inside the grid, nonnegative orbital counts and nonoverlapping orbital ranges inside `M`, and one live sampled-orbital pointer per atom with `[m_atom,box_volume]` storage. Density/potential matrices use `[M,M]`. Grid, density, derivative and output buffers must have the extents specified by the raw signature. Workers partition x planes or output orbital pairs. |
| Bader | Consistent grid/cell transforms, `[nstart,3]` starts, three grid-gradient fields, `[natom,3]` positions, `[natom+1]` monotone core offsets into radial arrays, `[natom]` supports, output owners `[nstart]` and final indices `[nstart,3]`. Workers own disjoint start ranges. |

The checked Phase 1 interfaces cover one-body, both real Laplacian precisions,
complex projectors, Bessel/Fourier-Bessel, radial eigenpairs, Numerov and real
shells. They validate integer conversions, shape products, finiteness, spline
sizes and angular/derivative bounds before allocating independent outputs and
entering unsafe kernels. Other raw interfaces remain caller-validated legacy
paths and must gain checked interfaces when their owning components migrate.

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
