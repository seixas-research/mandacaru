# Rust migration: Phase 3 molecular workflow

The native engine now accepts ASE geometries for molecular RHF, UHF, GHF, VQE
and basic ADAPT-VQE. Rust owns grid construction, analytic basis sampling,
one- and two-electron integration, orthogonalization, SCF, MO transformations,
frozen-core reduction and the Phase 2 quantum solve. Python snapshots ASE inputs,
validates option names, converts output units and exposes result objects.

```python
from ase import Atoms
from mandacaru import Mandacaru

atoms = Atoms("H2", positions=[[2.5, 2.5, 2.13], [2.5, 2.5, 2.87]], cell=[5]*3)
atoms.calc = Mandacaru(
    method="rhf", engine="rust", basis="HAO", grid_spacing=0.35,
    engine_options={"threads": 1, "memory_limit_bytes": 536870912},
    trace=False,
)
energy = atoms.get_potential_energy()  # approximately -25.500443358556 eV

# Keep the native MO integrals, without rebuilding a geometry or creating a
# Python fermion/Pauli operator in the calculation path.
reference = atoms.calc.result
calc = Mandacaru(method="adapt-vqe", trace=False, **reference.as_quantum_problem())
result = calc.run()                   # approximately -25.871895455447 eV
```

For an uninterrupted geometry-to-quantum calculation set `method="vqe"` or
`"adapt-vqe"` on the ASE calculator itself. `run(geometry=atoms)` also works.
`engine="python"` remains the default; native calculations never fall back to
Python scientific implementations. This Phase 3 milestone left native forces
unavailable. Phase 4 now adds stationary collinear SCF forces; see
[rust_phase4.md](rust_phase4.md) for their supported scope.

## Native components

| Component | Implemented behavior |
| --- | --- |
| Grids | Molecular bounding boxes centered on the coherent molecular geometry; explicit anisotropic/triclinic grids; periodic grids as a numerical primitive. |
| Basis | HAO occupied and optional virtual shells, STO-nG/GTO, native Pople/Dunning/Karlsruhe Gaussian recipes, per-element specifications, real spherical harmonics. |
| One body | Spectral kinetic energy, checked finite-difference kinetic energy, softened point-nucleus potential, bare overlap and symmetric Löwdin orthogonalization. |
| Coulomb | Isolated split-kernel FFT convolution, periodic zero-mode convention and perpendicular-axis slab truncation; complex densities and Nyquist alias averaging where required. |
| Two body | Real pair-packed ERIs and pivoted Cholesky factorization; physicists' tensor indices and explicit dense snapshots. |
| SCF | RHF, UHF, complex spinor GHF, coupled-spin Pulay DIIS, adaptive level shift, multiple starts and explicit convergence status. |
| Export | RHF MO integrals, UHF natural orbitals and actual determinant state, GHF spinor integrals/mode ordering, native fermion-to-qubit mapping and schema-v1 JSON caches. |
| Active spaces | Explicit or noble-gas frozen cores; energy-ordered active counts or explicit ascending index sets; core energy and effective one-body terms. |

HAO retains the current definition using the bare nuclear charge. Gaussian
recipes use Slater screening and effective principal numbers. Named Gaussian
families retain Mandacaru's **generated native recipes**, not published exponent
tables; their numerical interpretation has not changed.

The 80 scale-invariant Gaussian fits for occupied `(n,l)` shells through krypton
and contraction lengths 1–10 live in
`crates/mandacaru-core/data/slater_fits.json`. They were
generated once by the existing deterministic unit-Slater fitter. The data records
its source hash and NumPy/SciPy versions. Rust selects and partitions
contractions, applies Slater scaling and polarization/diffuse rules, normalizes
orbitals and samples them. No Python fitting or sampling runs during a native
calculation. Missing fit states fail explicitly. HAO does not use this table.

Regeneration is an explicit developer operation, not an installation/runtime step:

```bash
.venv/bin/python test/migration/export_gaussian_fits.py \
  --output .mandacaru/migration/slater_fits_candidate.json
```

Review a candidate before replacing the checked data. Scientific source hashes
now include JSON files so changing native numerical data changes provenance.

## Options and units

All numerical core work uses Hartree and Bohr. ASE receives eV; results use eV
unless `atomic_units=True`. Unit conversion follows the Python multiplication
convention, including its rounding at nuclear centers. The molecular grid keeps
the existing inclusive endpoint/count arithmetic, whereas a periodic grid omits
the repeated endpoint. Explicit Python `Grid` objects are snapshotted from their
shape, Bohr origin and step vectors; Python does not reconstruct their arrays
inside a solver loop. ASE minimum-image unwrapping remains an ASE integration
operation before the owned geometry snapshot.

Examples of additional native options:

```python
calc = Mandacaru(
    method="uhf", engine="rust", basis="STO-3G", grid_spacing=0.25,
    charge=0,
    scf_options={"max_iterations": 300, "tolerance": 1e-9,
                 "diis": True, "guesses": 3},
    two_body={"method": "cholesky", "tolerance": 1e-10},
    trace=False,
)
```

Initial ASE magnetic moments and the electron count determine the spin
populations. An odd count defaults to a doublet. RHF requires equal populations;
UHF permits unequal populations and includes a broken-symmetry start for a
closed-shell population. GHF includes the converged collinear UHF start.

`two_body=None`, `"dense"` or `"packed"` uses pair-packed real integrals;
`"cholesky"` additionally factors that pair matrix. The current Cholesky builder
first constructs the packed matrix: it does **not** claim the grid-direct memory
scaling of Python's factor builder. Fock construction and MO transformations
contract the factors directly; MO export materializes the kept MO tensor.
The native public interface currently selects active spaces by energy or explicit
indices, for example `active_space={"frozen": "auto", "orbitals": 4}`. Occupied
orbitals must be frozen or retained. A frozen orbital must be doubly occupied.

The native result retains the existing mean-field fields and SCF result types.
Its Python `fermion_hamiltonian` is materialized lazily when read; the native
`as_quantum_problem()` path reuses Rust storage directly and supplies
`engine="rust"`. Reused models keep their owning context; configure thread and
memory limits when creating the mean-field reference. The GHF model can be exported as a qubit Hamiltonian, but its
native quantum solve still needs the later spin-orbit pool migration.

`save_hamiltonian="model.json"` writes a native schema-v1 Hartree cache after
building the geometry. That cache can be loaded by either engine. Native cache
export currently uses JSON paths; Parquet stays in the Python workflow.
Geometry options are refused for a cache or a precomputed native model, where
they would otherwise be inert. `txt=` retains the compact native termination
report introduced in Phase 2; the full Python structured reporting protocol has
not migrated.

## Numerical and ownership checks

The native engine uses locked pure Rust FFT and linear algebra dependencies
(`rustfft`, `nalgebra`, `num-complex`, `libm`), with no Python dependency in the
core crate. FFT plans, transform buffers, operator matrices, integral buffers,
SCF/DIIS workspaces and result storage are accounted against the Phase 1 context.
Plans and third-party matrix temporaries use conservative reservations. NumPy
output copies and Python objects remain outside that native byte budget; it is
not an RSS limit. Numerical calls release the interpreter and use owned inputs.
The current FFT and SCF loops are sequential; the context thread setting remains
a ceiling rather than a promise of parallel execution.

Invalid cell metrics, dependent overlap matrices, invalid occupations,
unsupported options and allocation/iteration failures remain explicit. A
nonconverged SCF returns a false success flag for a mean-field calculation and
blocks a geometry-to-quantum calculation. Resolution ratios use analytic radial
kinetic energies to preserve coarse-grid warnings. The Phase 0 reference grids
are regression inputs, not basis/grid-converged physical predictions.

`test/migration/test_native_molecular.py` checks integral arrays and overlap
metrics against Python; atom-centered basis samples through high angular
momenta; complex FFT solves on orthogonal/skewed grids; all four qubit mappings;
native reference reuse; frozen-core and Cholesky invariants; stretched-bond UHF;
state normalization; unit agreement; cache reload and geometry A→B→A freshness.
It also forbids entry into Python basis/integral/SCF/mapping implementations
during complete native ASE calculations. The original H₂, H, GHF H₂ and STO-3G
water energies agree with the frozen Phase 0 corpus.

The standalone example exercises geometry → RHF → VQE without an interpreter:

```bash
cargo run --offline --release \
  --manifest-path Cargo.toml \
  --package mandacaru-core --example solve_molecule

cargo test --offline --manifest-path Cargo.toml \
  --package mandacaru-core --package mandacaru-kernels

.venv/bin/python -m pytest test/migration/test_native_molecular.py
```

The wheel smoke now includes native ASE RHF, VQE and ADAPT-VQE with runtime
compiler invocation disabled. See `test/migration/phase3/validation.json` for
recorded checks and regression findings.

## Remaining coverage

The complete native molecular workflow is available. The broader Phase 0
capability groups still contain work beyond this milestone:

- Integral-direct SCF, grid-direct Cholesky construction and localized/analytic
  two-center assembly are not wired into the new molecular engine.
- Periodic/slab Poisson and grids are native numerical primitives; full crystal
  SCF, Ewald terms, Bloch sampling and k-point workflows remain in Python.
- MP2/natural-occupation active-space ranking, symmetry expansion, explicit SCF
  stability Hessians and custom spinor initial guesses remain in Python.
- Pseudopotentials, numerical atomic orbitals, DFT and forces belong to later
  migration work. General GHF-spinor quantum pools and other Phase 2 gaps remain
  as documented for that milestone.

Unsupported native routes fail explicitly. Existing Python routes remain
available through the default engine.
