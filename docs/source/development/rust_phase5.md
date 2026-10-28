# Rust migration: Phase 5 periodic SCF and analysis

This milestone moves the periodic restricted/collinear-spin LDA/PBE Kohn–Sham
iterations, metallic occupations, mixing, eigensolves and spectral contractions
into the Rust core. It also adds native reciprocal-mesh reduction, Ewald
energy/potential/forces/stress, metric-normalized trial-subspace projections and
Wilson-loop primitives. The broader Phase 5 capability group remains partially
migrated; the pending work is listed below.

## Geometry workflow and the explicit setup adapter

The PAW species and density-independent Bloch integrals still come from the
Python PAW setup implementation left pending in Phase 4. This is an explicit
adapter, recorded as `result.timings["setup_engine"] == "python-paw-adapter"`.
Rust owns the complete SCF after that snapshot. It calls no Python numerical
routines during an iteration. A cached crystal also removes the setup adapter
from the solve.

```python
from ase.build import bulk
from mandacaru import Mandacaru

atoms = bulk("Si", "diamond", a=5.43)
atoms.calc = Mandacaru(
    method="dft", engine="rust", xc="lda",
    basis={"name": "PAW-LCAO", "size": "SZ", "filter_cutoff": 200},
    grid_spacing=0.4, kpts={"size": (2, 2, 2), "gamma": True},
    smearing={"method": "fermi-dirac", "width": 0.01},
    engine_options={"threads": 1}, trace=False,
    save_crystal="silicon_crystal.json",
)
energy = atoms.get_potential_energy()   # about -211.601939758958 eV per cell
free_energy = atoms.calc.results["free_energy"]
bands = atoms.calc.band_structure(path="GX", npoints=30)
energies, dos = atoms.calc.dos(width=0.1)
energies, shells = atoms.calc.pdos(energies=energies)
```

`PAW-LCAO` and `UPAW-LCAO` use the same explicit adapter. The geometry needs
periodic boundary conditions, and nonperiodic directions require one k-point.
The electrostatics are the existing three-dimensional jellium convention,
including for a vacuum-separated cell; this does not implement a new slab
boundary condition. Spin-orbit datasets/noncollinear moments are refused.
Initial collinear ASE moments select two spin channels and keep the full mesh
without spatial averaging, so a magnetic seed is not projected onto the
nonmagnetic space group. `unrestricted=True` also requests two channels. No molecular active space, qubit mapping or
two-body representation applies to this Kohn–Sham problem. `scf_options` is for
molecules; configure this loop with `periodic_scf_options`, for example:

```python
options = {"max_iterations": 300, "tolerance": 1e-7,
           "density_tolerance": 1e-5, "mixing": 0.25, "history": 7}
```

Off-mesh bands reuse the Python adapter for the new density-independent PAW
matrices and diagonalize the frozen potential in Rust. Gaussian DOS/PDOS use
the stored SCF mesh. The standard ASE getters expose eigenvalues/Fermi level in
eV, fractional irreducible k-points, weights and spin count. Molecular Rust
workflows from earlier milestones retain their own route.

## Portable crystal input

```python
calc = Mandacaru(
    method="dft", engine="rust", load_crystal="silicon_crystal.json",
    smearing={"method": "fermi-dirac", "width": 0.01}, trace=False,
)
result = calc.run()
assert result.timings["setup_engine"] == "cached-crystal"
```

The schema-v1 JSON representation is `periodic-paw-snapshot`, in Hartree and
Bohr. It contains the grid, k-points/weights, complex Bloch samples, overlap,
fixed Hamiltonians and projector overlaps, compensation multipole blocks/form
factors, compact Coulomb couplings, ionic/core terms, initial densities and
symmetry maps. Complex arrays are flattened row-major `[real, imaginary]` pairs.
The Fourier phase and XC derivative symbols preserve the original grid origin
and lattice alias convention. Dataset hashes and geometry are recorded in
provenance by the adapter. The file is self-contained for its stored mesh and
functional; it cannot regenerate geometry-dependent PAW data for a new cell.

Rust checks schema/units, dimensions, finite values, positive normalized
weights, Hermitian overlap/fixed operators, positive overlap spectrum,
spherical-harmonic conjugation of the multipoles, Hermitian compact Coulomb
couplings and symmetry permutations before solving. Context reservations cover
parsing, live arrays, FFT/matrix work and mixing history. Returned NumPy arrays
are independent copies. The core's limits cover native work, not the temporary
Python PAW setup or Python-owned output arrays.

The checked portable Si setup is
`test/migration/phase5/si_crystal.json`. It preserves the Phase 0 geometry,
dataset hash and discretization. It is an explicitly generated candidate input,
not a replacement for any frozen Phase 0 artifact. Regenerate a candidate for
review with:

```bash
.venv/bin/python test/migration/export_crystal.py \
  --output .mandacaru/migration/si_crystal_candidate.json
```

The standalone Rust engine needs neither Python nor a dataset library:

```bash
cargo run --offline --release \
  --manifest-path Cargo.toml \
  --package mandacaru-core --example solve_crystal -- \
  test/migration/phase5/si_crystal.json
```

## Native numerical behavior

The generalized complex Hermitian problem is reduced with a cached symmetric
inverse square root of each positive overlap. Densities and compensation
moments are summed over k-point weights and symmetrized from the input maps.
Multipoles stay in projector space: the SCF does not allocate one dense AO
moment matrix per channel. FFT workspaces and input-independent matrices are
reused. Density/moment DIIS uses the existing volume metric and Kerker factor;
only the total charge is Kerker-damped. A spin calculation mixes linearly for
its first twelve iterations, then uses DIIS. Convergence checks free-energy
change, the integrated charge/magnetization residual, and the moment residual;
nonconvergence is retained in the result.

Fermi–Dirac, Gaussian and first-order Methfessel–Paxton smearing have native
occupations, Fermi-level bisection and entropy. The stored internal energy is
`scf.energy`, the variational free energy is `scf.free_energy`, and the ASE
energy is the zero-width estimate: `(E + F)/2` for Fermi–Dirac/Gaussian and `F`
for Methfessel–Paxton. Crystal forces/stress must differentiate `F` rather than
this extrapolated energy. The frozen Phase 0 energy and bands agree with the
native Si solve to about 1e-12 eV. Its 0.01 eV occupation function saturates
through an interval inside the gap: a reported chemical potential in that
interval is numerically nonunique. Validation keeps the measured Fermi-level
difference visible and requires both values to lie in the same gap with the
same electron count. Metallic cases compare the Fermi level directly.

`Context.kmesh` constructs Monkhorst–Pack/Gamma-centered meshes, their
reciprocal-rotation/time-reversal orbits, normalized weights and full-to-reduced
mapping. Supplied integer rotations must be unimodular and preserve the mesh.
Finding the crystal's space group and grid-compatible operations is still part
of the setup adapter.

`Context.ewald` supplies a separate native lattice primitive in atomic units:
energy including self/background, arbitrary-point softened potential, analytic
ionic forces and analytic stress under joint cell/coordinate strain. It wraps
sites modulo the cell and supports skew cells. Its stress uses ASE's positive
energy derivative convention. This is the **ionic contribution**, not the
full electronic PAW force/stress. Periodic calculator force/stress requests
continue to fail explicitly until the PAW setup derivatives migrate.

## Analysis primitives

The native result supports generalized-overlap Loewdin weights and Gaussian
DOS/PDOS contractions. `PeriodicResult.projectability(trials)` accepts an
`(n_basis, n_targets)` complex trial matrix. At each k-point it orthonormalizes
the targets in `S(k)`, returns each band's weight in that subspace and the
Hamiltonian projected into it. Linear dependence is refused. This is a trial
subspace projection; it does not implement Wannier disentanglement or spread
minimization.

`Context.wilson_loop(links)` accepts occupied-space overlap matrices around a
closed loop. Polar unitarization and determinant phases give a gauge-invariant
phase and the smallest link singular value. Singular links are refused. Tests
include both phases of the SSH chain and arbitrary local gauges. Constructing
the physical PAW links, polarization strings and their field derivatives
remains pending.

## Validation and remaining scope

Evidence lives in `test/migration/test_native_periodic.py`, Rust integration
tests `crates/core/tests/periodic.rs`, and
`test/migration/phase5/validation.json`. It covers the frozen Si energy/bands,
nonuniform-k weights, periodic PBE and collinear spin on skew cells, free-energy
thermodynamic derivatives of all smearing schemes, analytic Ewald displacement
and strain derivatives, DOS sum rules, subspace spectra, Wilson-loop gauge
invariance, ownership/budget cleanup, cache rejection and nonconvergence. A
cached solve disables Python SCF, mixing, FFT, linear-algebra and XC entry
points. Wheel smoke tests run the cached crystal outside the checkout with
runtime compilers disabled.

Pending within the broad Phase 5 scope:

- Native PAW species/Bloch-integral/symmetry assembly and its geometry/strain
  derivatives; full electronic PAW forces and free-energy stress.
- Periodic many-body drivers, spinor/SOC and advanced functionals/corrections.
- Dense-mesh/tetrahedron DOS and symmetry-resolved projected analysis beyond
  the stored-mesh Gaussian implementation.
- Physical Berry/polarization/dielectric/Born/piezoelectric response,
  Wannier/disentanglement/localization, cRPA and general downfolding workflows.

The complete scientific suite is not asserted green. The validation report
retains broader periodic regression outcomes and any prior-revision findings.
