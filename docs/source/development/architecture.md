# Architecture

Mandacaru has two layers with one boundary between them. **Rust owns the numerics**:
arrays, integrals, mean-field iterations and variational optimization. **Python is the
interface**: the ASE calculator, array conversion, visualization and the adapters for
quantum-computing services. Numerical work moves to Rust before its Python is deleted;
there is no fallback to Python and no execution selector, and an unsupported request
raises instead of selecting a different calculation. The state of that move is on
[the cutover page](native_cutover.md).

```text
ASE Atoms / Geometry + options + CalculationResources
                       │
          Mandacaru   (Python: validates options, converts units)
                       │
        mandacaru._native   (PyO3 boundary: owned-array copies)
                       │
     Rust solvers: molecular · periodic · quantum · PAW
                       │
          shared math, scientific models, kernels
```

## The crates and their dependency rule

The repository root `Cargo.toml` is the workspace of three crates:

| Crate | Owns |
| --- | --- |
| `mandacaru-kernels` | Low-level numerical kernels (one-body and two-center integrals, radial and localized functions). `SAFETY.md` holds the contract of its raw ABI, and every `unsafe` block needs an entry there. |
| `mandacaru-core` | Everything scientific: grids, FFTs, linear algebra, models, basis generation, molecular and periodic solvers, PAW, quantum states and optimizers. |
| `mandacaru-python` | The PyO3 boundary: array transfer, context dispatch, and the molecular, periodic, quantum and PAW bindings. It contains no numerics. |

The dependency direction is `python -> core -> kernels`. Neither `mandacaru-core` nor
`mandacaru-kernels` depends on Python or NumPy, and each dependency version is
declared once, in the workspace `Cargo.toml`. Inside the core, `math` depends on no
physical model: grids, FFTs, radial quadrature, splines and the separable
eigenproblem are usable without knowing what an atom is. The default workspace
members are the core and the kernels, so `cargo test` tests them; the Python crate is
a cdylib built by the installation. `python scripts/check_architecture.py` enforces
the rules, including grouped Rust imports, retired selectors in the supported
Python boundary and Python dependencies in either scientific crate.

## Where the numerics live

| Core module | Contents |
| --- | --- |
| `math/` | Spatial grids, FFTs, radial quadrature, splines, linear algebra with a verified Jacobi fallback, symmetry, reciprocal geometry, separable generalized eigenproblems. |
| `models/` | Atoms, spherical Hartree and exchange-correlation models, the functional registry, nonlocal correlation, dispersion primitives, spin-orbit operators. |
| `basis/` | Native basis generation: radial functions, NAO and NAO-AE. |
| `parquet/` | The Parquet container (Thrift footer, PLAIN and dictionary pages, Snappy and ZSTD codecs, writer); depends on no physical model, and its one third-party crate, `ruzstd`, is used only in `codec/zstd.rs`. Python reads and writes Hamiltonian caches and pseudopotential datasets through it ([Parquet files](#native-parquet)). |
| `molecular/` | Molecular integrals, RHF/UHF/GHF and Kohn-Sham, forces and geometry response, orbital selection and MP2, PAW-LCAO gradients. |
| `periodic/` | Crystal SCF and spinors, Ewald sums, spectra, Wannier functions. |
| `paw/` | PAW dataset storage and evaluation, generation, projector channels, augmentation, molecular PAW. |
| `quantum/` | Pauli and fermion algebra, mappings, sectors, pools, states, evolution, the optimizers, ADAPT, subspace and deflation solvers, architecture search, execution hooks. |

The functional registry and the physical constants live in Rust and are exported to
Python, so there is one list of exchange-correlation functionals.

## The Python layer

`src/mandacaru/` is the interface layer. Numerics that still sit in it are awaiting
their move to Rust and are tracked on the cutover page; no new numerical routine is
added in Python.

| Module | Role |
| --- | --- |
| `calculation.py`, `structure.py`, `resources.py`, `results.py`, `specifications.py` | The `Mandacaru` calculator, geometry input, per-calculation resources, native result views and the method and basis specifications. |
| `_native.pyi` | The typed contract with the extension. |
| `integrations/` | ASE conversion, PAW adapters, circuit execution, run reports. |
| `analysis/` | Views of native results. |
| `workflows/` | Quantum phase estimation, time evolution, quantum echoes. |
| `backends/` | Device registry, circuit providers, measurement planning. |
| `basis/`, `models/`, `pseudopotentials/` | Basis families and metadata, input recipes, PAW-LCAO and UPAW-LCAO generation and dataset layout. |
| `circuits/`, `core/`, `integrals/`, `optimizers/` | Ansatz and pool definitions, fermion and Pauli algebra, integral helpers, the optimizer wrapper. |
| `utils/` | The run-log protocol, profiling, bibliography, cube and XSF writers, `Viewer3D`. |
| `cli.py`, `units.py`, `version.py`, `backend_policy.py` | The `mandacaru` command, unit constants, version, the one parser of `MANDACARU_BACKEND`. |

## The PyO3 boundary

The rules, which `_native.pyi` records and the bindings must satisfy in the same
change:

- Inputs are aligned, C-contiguous NumPy arrays of the declared dtype. The binding
  validates them and copies them into budgeted Rust buffers while it still holds the
  interpreter, then releases it. It is not an end-to-end zero-copy API.
- Returned arrays own independent storage. No borrowed or writable Rust view is
  exposed, so changing or retaining a result cannot invalidate a native workspace.
- A crystal input is small JSON metadata plus a dictionary of typed arrays, connected
  by string keys. JSON files are persistence, not the runtime transport.
- Native objects keep their `Context` and shared immutable state through reference
  counting. Each calculator has its own context, lock and optional scratch directory.
- A `Context` carries a thread limit and a byte budget. The budget covers native
  buffers and estimated scratch; it is not a limit on the process, and the NumPy
  copies returned to Python are outside it.
- Units inside the boundary are Hartree and Bohr.
- When a signature in the bindings changes, `_native.pyi` changes with it.

## One calculation

`Mandacaru` validates its options in the constructor and builds the native problem
on first use. A run takes the geometry, builds the integrals on the grid, runs the
mean-field iteration, and for a quantum method hands the orbitals to the quantum
solver; each step is one call across the boundary that returns an owned object.
Properties are lazy: reading an energy or a force leaves the transformed four-index
Hamiltonian unbuilt, and only an explicit request for it (`result.hamiltonian`, or a
quantum method) allocates it. HF requests Coulomb factors lazily as pivoted Cholesky
columns, never the full pair matrix; semilocal Kohn-Sham allocates no repulsion
factors at all.

Orbital selection (`molecular/active.rs`) builds the active-space Hamiltonian
directly from the Coulomb factors, without a full source-MO tensor, and the selected
model owns its orbital rotation and occupations. Molecular PAW adds the projector,
augmentation and one-center terms in Rust; the Python adapter only decodes the dataset
and supplies the radial tables.

The public configuration is fixed at construction. Closing a calculator releases its
state and scratch, while results you kept continue to own their buffers; closing one
calculator never affects another.

## Adding a method

A method is not finished until all of the steps below are done; the last two are
checked by tests.

1. **Rust.** Implement it in the owning core module with tests in
   `crates/mandacaru-core/tests/` (one file per topic), including a derivative or
   reference check. It must reproduce an independent number, or the one it replaces,
   from the frozen references in `test/migration/reference`.
2. **Binding.** Expose it in the matching `mandacaru-python` module and update
   `_native.pyi` in the same change. Rebuild the extension (`pip install -e '.[dev]'`);
   a change under `crates/` is invisible to Python until you do.
3. **Specification.** Add a `Capability(quantum, forces, periodic)` entry to `METHODS`
   in `specifications.py`, and the dispatch in `calculation.py`. Reach the method
   only through `Mandacaru(method=...)`.
4. **Refusals.** Every combination of options the method cannot honor must raise in
   the constructor or at the run, with a message that names the option.
5. **Bibliography.** Register the reference in `utils/bibliography.py`; the registry
   sweep in `test/utils/test_bibliography.py` fails without one.
6. **Documentation.** Add the row to [the methods table](../reference/methods.md)
   (`test/test_docs_links.py` compares the table with `METHODS`), write the guide,
   put anything partial in [the not-yet-available table](../reference/not_yet_available.md)
   and update the cutover page.
7. **Example.** Add a numbered script to the matching `examples/` category and run it
   with `scripts/check_examples.py`.

## Automated checks

| Check | What it covers |
| --- | --- |
| `cargo test` | Numerical limits, scientific references and derivatives in the core and kernels. |
| `python -m pytest` | The Python suite, including `test/architecture` (boundary behavior, lazy Hamiltonians, calculator isolation, array ownership) and `test/migration` (frozen references and native parity). |
| `python scripts/check_architecture.py` | Crate dependencies, retired selectors, packaging inputs. |
| `python scripts/check_wheel.py` | An installed wheel works outside the checkout with compilers blocked. |
| `python scripts/check_examples.py` | Runs disposable copies of the examples and records failures, convergence and peak native allocation. Heavy examples have no default time limit; outbound connections are blocked so hardware examples cannot submit jobs. |
| `make -C docs html SPHINXOPTS="-W"` and `test/test_docs_links.py` | The manual: warnings are errors, and every link and code sample is checked against the code. |

Passing the native tests does not certify the examples or the full scientific suite,
and none of these checks claims parity for the remaining legacy modules.
