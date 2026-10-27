# Rust migration: Phase 4 atoms, molecular DFT and stationary forces

This milestone adds native NAO/NAO-AE numerical atomic orbitals, nonrelativistic spherical
LDA atoms, molecular restricted/collinear unrestricted LDA and PBE Kohn–Sham SCF,
and stationary RHF/UHF/Kohn–Sham nuclear gradients. The broader Phase 4 capability
group is **partially migrated**: PAW/UPAW and the other capabilities listed below
remain pending. Native requests never fall back to their Python implementations.

```python
from ase import Atoms
from mandacaru import Mandacaru

atoms = Atoms("H2", positions=[[2.5, 2.5, 2.13], [2.5, 2.5, 2.87]], cell=[5]*3)
atoms.calc = Mandacaru(
    method="dft", engine="rust", xc="pbe", basis="HAO",
    grid_spacing=0.35, trace=False, project_translation=False,
    engine_options={"threads": 1, "memory_limit_bytes": 536870912},
)
energy = atoms.get_potential_energy()  # eV
forces = atoms.get_forces()           # eV/Angstrom; -dE/dR
reference = atoms.calc.result
quantum = Mandacaru(method="vqe", trace=False, **reference.as_quantum_problem()).run()
```

The exported many-body Hamiltonian contains the ordinary electron interaction
in the Kohn–Sham orbitals. Its reference determinant energy is
`reference.reference_energy`; the Kohn–Sham functional energy is
`reference.optimal_energy`. These are different quantities. Existing
`KohnShamResult`/`KohnShamUResult` views retain Hartree-valued SCF energies,
eigenvalues and energy decompositions. Spin populations follow the electron
count and collinear ASE magnetic moments; `spin=True` or `unrestricted=True`
requests separate channels even with equal populations.

## Numerical ownership and differentiation

Rust evaluates Slater exchange with the continuous Perdew–Zunger correlation
fit for LDA, and PBE exchange with spin-interpolated Perdew–Wang correlation.
A small forward derivative type differentiates the energy expression with
respect to both densities and all three gradient contractions. No JAX callback
runs in the native engine. The PBE gradient and divergence use the existing
molecular eighth-order central stencil with wrapped boundaries, including its
Nyquist behavior and density floor. Their discrete adjoint relation makes the
matrix potential the derivative of the same grid energy. PBE currently requires
an orthorhombic molecular grid; LDA also accepts the Phase 3 explicit skew grid.

The Kohn–Sham loop owns densities, Poisson workspaces, DIIS history, convergence
status and orbital transformations. Occupations are integer aufbau occupations;
ensemble filling of degenerate open shells has not migrated. The Hartree builder
uses a reusable isolated Poisson solver. The common molecular container still
builds pair-packed integrals for the many-body export, even for a DFT energy.
Consequently this milestone does not claim an integral-direct memory bound.

Nuclear gradients differentiate the stationary AO Lagrangian. Hellmann–Feynman
terms include the derivative of the energy's clipped nuclear potential and
bare nucleus–nucleus repulsion. Pulay terms include basis motion in kinetic,
local, Hartree, exchange/XC terms and the energy-weighted overlap correction.
Cartesian solid harmonics provide analytic HAO/Gaussian derivatives without
polar-coordinate singularities; numerical orbitals use the derivative of their
cubic interpolant. The force loop streams occupied pair potentials instead of
building a four-index derivative tensor. It performs no displaced SCF solves.

Forces require a converged real collinear SCF, the full orbital space and
spectral kinetic energy. HF forces require packed integrals; the derivative of
the truncated Cholesky representation is pending. A DFT force differentiates
its Poisson Hartree energy irrespective of the representation used for the
exported many-body Hamiltonian. The calculator freezes the integration grid
when forces are first requested, retains the force breakdown, and reuses an
energy's SCF if geometry, options and grid agree. `project_translation=False`
exposes the derivative of the discretized energy for validation. The point
potential's clipping boundary and orbital cusps retain their existing
nondifferentiability conventions.

Context memory reservations cover native live storage and temporary work;
NumPy outputs are independent Python-owned copies. Old results retain their
own native storage after subsequent geometries. Native DFT defaults to in-memory
storage; explicit disk buffers remain unsupported. Python DFT keeps its previous
storage default.

## Radial atoms and numerical orbitals

`basis="NAO"` now constructs its confined screened-Coulomb radial functions in
Rust, including the existing default DZP size. Supported options are
`energy_shift`, `r_c` (Bohr), `n_grid`, `size` (SZ through QZ with polarization),
and `zeta_split` (first/last zeta or explicit tail norms). For example:

```python
calc = Mandacaru(
    method="dft", engine="rust", xc="lda", trace=False,
    basis={"name": "NAO", "size": "DZP", "r_c": 7.0, "n_grid": 2000},
)
```

Native shell selection preserves the valence/semicore rules, Slater effective
charges, hard wall, radial node index, sign/normalization, not-a-knot cubic
interpolation, split-tail polynomial, tail-norm conventions and polarization
ordering. The original NAO definition includes valence functions, not a complete
all-electron core basis for every element; the electron-capacity check still
applies. `NAO-AE` now constructs a complete occupied all-electron minimal basis from a
confined self-consistent LDA atom. Tiers 1–2 add atom-sized polarization,
diffuse and contracted functions, with optional custom `(n,l,z)` triples,
tail-norm shortening, extent ordering and three-pass Gram–Schmidt within each
angular channel. Its smooth exponential wall and nonuniform final spline knot
match the Python species construction. Supported options are `tier`, `onset`
and `width` (Angstrom), `scale`, `extra`, `linear_dependence_tol`, `tail_norm`,
`points` and `r_max` (Bohr). For example use
`basis={"name": "NAO-AE", "tier": 1}`. A nonconverged confined reference atom
fails explicitly; automatic relaxed f-block references remain pending.

The spherical atomic solver can be selected explicitly:

```python
from mandacaru.basis.atomic_solver import solve_atom
atom = solve_atom(8, engine="rust", points=4000, r_max=25.0,
                  engine_options={"threads": 1})
```

Rust owns the nonrelativistic uniform-grid LDA eigensolves, radial Hartree
integration, XC, density-metric Pulay mixing and convergence check in both
volume and shell densities. An explicit configuration and an array-valued
confinement are supported. Automatic relaxed f-block configuration selection,
PBE radial potentials, relativistic treatments, logarithmic grids and callable
confinement are refused. `Context.confined_radial`, `Context.solve_atom`,
`Context.exchange_correlation`, `MolecularIntegrals.kohn_sham` and
`MeanField.gradient` expose the checked native primitives. The Rust core also
runs independently of Python; `cargo run --example solve_dft` demonstrates it.

## Evidence and remaining coverage

`test/migration/test_native_dft.py` checks the frozen Phase 0 H2 LDA energy and
force, Python/native XC and molecular SCF comparisons, spin limits, an
independent XC directional derivative, fixed-grid force finite differences for
HAO/Gaussian/NAO restricted and unrestricted states, radial samples and atomic
fixed points, geometry reuse/round trips, budget cleanup and nonconvergence.
Python numerical entry points are disabled in a native end-to-end test. Rust
integration tests independently exercise the energy/gradient, spin symmetry,
radial equation, atomic charge and analytic orbital derivatives. The checked
validation report is `test/migration/phase4/validation.json`; local JUnit and
packaging logs are under `.mandacaru/migration/phase4/`.

Pending within the broad Phase 4 scope:

- Additional diffuse NAO shell recipes beyond NAO-AE tiers/custom functions,
  radial PBE, relaxed configurations and scalar/Dirac atoms.
- Native PAW/UPAW datasets, generation, augmentation and one-center terms,
  dataset-library selection and serialization. The external PAW/PBE Phase 0
  case remains a Python reference and is not claimed as native evidence.
- Spinor Kohn–Sham, fractional degenerate-shell occupations, r2SCAN, HSE06,
  rVV10, D4, Hubbard and electric-field/response terms.
- Spinor/finite-difference-kinetic/Cholesky HF forces, active-projector and
  correlated quantum orbital response. Quantum force requests still fail
  explicitly. Periodic SCF and response belong to Phase 5.

The coarse reference grids measure program parity, not converged physical
predictions. The frozen Phase 0 artifacts are unchanged, and the full scientific
suite is not asserted green by this milestone.
