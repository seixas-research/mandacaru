# Native calculations

`engine="rust"` selects the native computational engine. Python snapshots an
ASE geometry, normalizes options and units, and presents results. Rust owns
scientific buffers, SCF iterations, state vectors and the numerical work of
supported methods.

## Execution and ownership

```python
from mandacaru import Mandacaru

calculator = Mandacaru(
    method="dft", engine="rust", xc="r2scan", basis="HAO",
    engine_options={"threads": 2, "memory_limit_bytes": 512 * 1024**2},
    grid_spacing=0.35, trace=False,
)
```

The thread pool belongs to a context. The memory limit covers native-owned
buffers and reserved workspaces; it does not cap the whole Python process or
external SDKs. Exceeding it raises `MemoryError`. Inputs are copied before the
interpreter is released; modifying an input later cannot change a calculation.
Output arrays own their storage.

## Coverage

| Calculation | Native support | Current limits |
| --- | --- | --- |
| Molecular mean field | RHF, UHF, GHF | Stationary gradients require real collinear orbitals and spectral kinetic energy |
| Molecular DFT | `lda`, `pbe`, `r2scan`, `hse06`, `pbe+rvv10`, `r2scan+rvv10` | All-electron geometry path; PAW setup has additional migration work |
| Periodic DFT | LDA/PBE SCF, occupations, eigensolves, bands and Gaussian DOS | Python PAW assembly; advanced XC and full strain response remain unported |
| Variational methods | UCCSD VQE, fermionic ADAPT, shared-subspace searches | Other ansätze, learned proposals and architecture searches need native implementations |
| Correlation | Real closed-shell MP2/FNO | Open-shell and local variants remain unported |
| Persistence | Native JSON Hamiltonians and completed-run checkpoints | Columnar files still use a Python codec adapter |
| Quantum execution | State operations, mapping, grouping and circuit compilation | External SDK/service calls remain Python integrations |

Unsupported combinations raise an error. The [audit](../development/native_cutover.md)
records remaining numerical Python work. These limits describe the current
engine, rather than a promise of parity for every option in older examples.

## Reproducibility

Keep geometry, basis, grid, occupations, mapping and precision with the result.
Compare energies and derivatives against scientific references before deleting
an implementation. The migration fixtures retain source hashes and units;
they are separate from newly generated benchmark results.

A JSON Hamiltonian can restart quantum work without rebuilding the geometry.
A completed VQE or ADAPT checkpoint can supply a warm start for a compatible
problem. It does not restore an optimizer's Hessian or an interrupted iteration.

## Developer entry points

The independent crate is `mandacaru-core`; `mandacaru-python` is the PyO3
boundary exposed as `mandacaru._native`. `Context` owns thread and allocation
policy. `Geometry` and `SpatialGrid` record units and layout. Molecular and
periodic objects retain their native context.

Run `cargo test` for the independent crate and `python -m pytest test/migration`
for Python-boundary and scientific checks. Rebuild the extension before running
Python tests after any Rust change.
