# Calculation architecture

`Mandacaru` is an ASE `Calculator`. Users attach it directly with
`atoms.calc = Mandacaru(...)`. The Python calculator owns one native context,
its geometry snapshot, scientific result views and resource lifetimes. Rust
owns numerical arrays, integral kernels, mean-field iterations and variational
optimization. Standalone `Geometry` input is also available through `run`.

```text
ASE Atoms / Geometry + specifications + CalculationResources
                       │
                 Mandacaru.run
                       │
               PyO3 owned array boundary
                       │
     Rust molecular / periodic / quantum solvers
                       │
           shared math + scientific models
```

The Rust math layer provides spatial grids, Fourier operators, reciprocal
geometry, radial quadrature, differentiation and eigensolvers. It cannot depend
on physical models, basis construction or solvers. Spherical Hartree and XC
models are shared by reference atoms and PAW; the atomic model does not depend
on PAW.
Separable generalized eigenproblems live in `math/separable.rs`. PAW radial
channels use them through `paw/spectrum.rs`: the tridiagonal kinetic operator
and projector corrections stay structured, including the overlap metric.
Confinement and radial spectrum checks call this Rust implementation.
The functional registry and physical unit constants live in Rust and are
exported to Python. Python basis recipe metadata is separate from sampled
basis generation.

## Repository layout

```text
Cargo.toml                         Rust workspace and shared dependencies
Cargo.lock                         Locked native dependencies
crates/
  mandacaru-core/
    src/basis/                     Native basis generation and radial tables
    src/math/                      Grids, FFTs, radial quadrature, linear algebra
    src/models/                    Atoms, spherical Hartree/XC, nonlocal and dispersion models
    src/molecular/                 Molecular integrals, HF/DFT, forces, transformations
    src/periodic/                  Crystal SCF, electrostatics and spectra
    src/paw/                       PAW radial potentials and projectors
    src/quantum/                   States, optimizers and variational solvers
  mandacaru-kernels/                Low-level numerical kernels
  mandacaru-python/                 PyO3 boundary and owned NumPy transfers
build_hooks/native.py              Native extension construction during packaging
src/mandacaru/
  calculation.py                   Rust-only calculation orchestration
  structure.py                     Geometry input values
  resources.py                     Per-calculation resource lifecycle
  results.py                       Native state/result views
  models/                          Python input recipes and metadata
  integrations/                    ASE, dataset codecs and external SDK adapters
  workflows/                       Workflows over native quantum operations
  analysis/                        Native correlation result views
scripts/                           Local architecture and wheel validation
```

The workspace is rooted at the repository's `Cargo.toml`. No Rust source is
nested inside the Python package, and no compatibility directories or import
aliases preserve the former layout. The core depends on the kernel crate;
the Python binding depends on the core. Neither scientific crate depends on
Python or NumPy. Workspace-level dependency versions are declared once.

The binding entry point registers types. Array conversion, geometry and spatial-grid values,
context dispatch, molecular states, periodic states, quantum problems, shared scientific models and PAW
adapters live in separate binding modules. PAW dataset storage and evaluation
live in `paw/dataset.rs`, construction in `paw/generate.rs`, and projector
operators in `paw/channel.rs`. Shared radial interpolation belongs to `math`.

Python workflows import from `mandacaru.workflows.qpe`,
`mandacaru.workflows.time_evolution`, `mandacaru.workflows.quantum_echoes` and
`mandacaru.workflows.nested_otoc`. Their tests live in `test/workflows`. ASE
conversion and calculator adapters live in `mandacaru.integrations.ase`.

## Lazy electronic interactions

Molecular setup builds basis samples and one-body matrices. Semilocal SCF uses
density fields without allocating electron-repulsion factors. HF requests
Coulomb factors lazily. Pivoted Cholesky computes a diagonal and selected
Poisson columns, avoiding the full pair matrix. The same factor contractions
serve HF and screened hybrid exchange.

Molecular PAW adds atom-centered projector/local-potential quadrature,
augmented overlap, compensation multipoles and exact one-center terms in Rust.
Its smooth Coulomb operator uses Cholesky factors; the signed augmentation
factors remain separate. The combined correction is not assumed positive
semidefinite. The resulting representation supports both Fock contractions and
explicit orbital transformations without an intermediate molecular four-index
tensor. Only an explicit transformed Hamiltonian request allocates that tensor.

The molecular PAW adapter decodes datasets and supplies basis radial tables.
Rust constructs their nonuniform cubic interpolants and owns all operator
arrays. A single closed 1s core is recovered exactly from its stored density,
preserving the dataset's reference convention; other frozen-core partitions
remain unsupported. PAW field views sample the smooth valence density and
report augmentation charge separately from the charge present in a cube file.

A mean-field state contains orbitals, densities and SCF records. Its transformed
four-index model is a separately cached explicit request. Quantum calculations
request that model; ordinary energy, forces and snapshots do not. Native locks
protect factor and model initialization. Failed construction does not cache a
partial model.

Orbital selection lives in `molecular/active.rs`. Energy selection and
closed-shell MP2 frozen natural orbitals build the selected Hamiltonian
directly from Coulomb factors and signed PAW factors. MP2 requests only an
occupied–occupied–virtual–virtual block for its density. Neither path constructs
the full source-MO four-index tensor. Frozen-core contributions use native
Coulomb and exchange contractions. The selected model owns its orbital
rotation and occupations; density views consume that same native model,
including after the calculator closes.

## Array ownership

NumPy input buffers must be aligned and contiguous with exact dtypes. The
binding copies into budgeted Rust buffers while attached to Python, then
releases the interpreter. A crystal input contains small JSON metadata plus a
map of typed arrays; string keys connect those records. The buffers move into
the native crystal where possible. Small matrix representations can require
an additional layout conversion; this is not an end-to-end zero-copy API.

JSON files are explicit persistence, rather than the runtime bulk transport.
Every returned snapshot has independent NumPy storage. Native objects retain
their context and shared immutable state through reference-counted ownership.

## Resource lifecycle

Each calculator has its own context, lock and optional temporary directory.
Repeated runs replace its current reference; externally retained result views
remain valid. Closing releases the calculator's state and deletes its scratch,
while retained results continue to own their buffers. Closing one calculator
must not affect another calculator's resources.

The public configuration is fixed at construction. Cached problems preserve
their mapping; an explicit conflicting mapping raises. Unsupported requests
must raise instead of selecting a different model.

## Automated checks

`scripts/check_architecture.py` checks dependencies and retired selectors in
the supported boundary, duplicate orchestration, packaging inputs and dependency
direction between
math, models, basis construction and solvers. It checks grouped Rust imports as
well as direct paths and rejects Python dependencies in either scientific crate.
`test/architecture` checks behavior: unused integration imports blocked in a subprocess,
lazy Hamiltonians for all six functionals, native quantum references and
checkpoints, calculator isolation, array ownership and periodic validation.

Run the local validation scripts against a wheel installed with all required
dependencies in a fresh environment. ASE is part of that installation. The
import-blocking check separately verifies that numerical calls do not eagerly
load unused SDKs or SciPy. GitHub workflows are not used.
Rust tests check numerical limits, scientific references and derivatives. Sphinx builds with warnings as
errors. These checks do not imply parity for all remaining standalone legacy
modules; their migration is tracked in the [coverage audit](native_cutover.md).

Public recovery also requires running `python scripts/check_examples.py`.
It preserves every example, runs disposable copies, and records each failure
or timeout. Outbound connections are blocked so hardware examples cannot
submit remote jobs. Native tests and architecture checks alone do not certify
that the ASE examples or the full scientific suite work.
