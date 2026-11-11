# Not yet available

(not-yet-available)=
## What the manual or the older API had that does not run now

Mandacaru raises when it cannot do what you asked; it never substitutes another
calculation. This table lists every feature an earlier version of the program or
of this manual offered that the current build does not implement, what you get
when you ask for it, and where it stands. Each "What happens" entry was checked by
making the request on this build (version 26.10.43) unless the row says otherwise.
No other page of the manual describes these features as usable.

"Plan" names a maintainers' plan file (`dev_plan/`, kept in the maintainers'
working tree rather than the published repository). "No plan" means no one has
scheduled the work. The migration status of the native engine as a whole is on
[the cutover page](../development/native_cutover.md).

| Area | Feature | What happens now | Status |
| --- | --- | --- | --- |
| Periodic quantum | k-resolved quantum methods beyond the supercell | `vqe`, `adapt-vqe`, `vasqa` and `valsqa` on a periodic structure solve the Born-von Karman supercell of the Gamma-centered `kpts` mesh ([the crystal guide](../guide/bloch.md)) with `crystal_orbitals="real-space"`; the default `crystal_orbitals="k-space"` (a register per k-point with momentum-conserving excitations) raises `NotImplementedError`. `get_spectral_function()` resolves `A(k, omega)` at the mesh points only (another k raises `ValueError`; `path=` and `irreducible=` select the mesh points on a band path and the star representatives, see [the crystal guide](../guide/bloch.md)). There is no correlated band between mesh points and no PAW-LCAO supercell (`NotImplementedError`); forces are central finite differences of the energy (no analytic gradient) and `get_stress()` raises `NotImplementedError` (a strain changes the plane waves inside the band limit of the supercell orbitals). | Plan: `dev_plan/BLOCH_VQE_PLAN.md`, stages 4 and 6 and the per-k register (K0-K7). |
| Mean field | All-electron analytic forces and active spaces on a fractional-occupation Kohn-Sham state | Molecular Kohn-Sham takes Fermi-Dirac occupations with one Fermi level per spin ([Occupations](#dft-degenerate-shell)); a degenerate or nearly degenerate frontier shell gets fractional ones. On such a state, PAW-LCAO analytic forces work (screened hybrid included); the all-electron (`HAO`, Gaussian) analytic force and an `active_space` raise `NotImplementedError` naming the fractional occupations. | No plan. |
| DFT+U | DFT+U with a screened hybrid, with a spin-orbit dataset or in single precision | A molecule takes `hubbard_u` for `method="dft"` on a PAW-LCAO basis ([the DFT+U guide](../guide/dft_u.md)); its analytic force carries the +U term. `hse06`: `NotImplementedError`. Spin-orbit (fully relativistic) dataset: `NotImplementedError` (a molecule on such a dataset runs spinor Kohn-Sham without DFT+U; spinor molecular DFT+U is not implemented). `precision="float32"`: `ValueError`. Read from the code, not requested on this build. | Plan: `dev_plan/DFTU_PLAN.md`. The Dudarev and Liechtenstein forms, linear-response U, the symmetry-reduced k mesh, a magnetic crystal's moment-preserving group, and crystal forces and stress are implemented. Noncollinear spinor Dudarev model implemented in `models/hubbard_spinor.rs` (SO-1..SO-8). Molecular +U forces are implemented (Dudarev and Liechtenstein, restricted and spin-polarized); spinor +U (molecular and in the periodic SCF) is open; screened hybrids and float32 remain unsupported. |
| Periodic DFT | Validated and analytic HSE06 for a crystal | **Runs, and is unvalidated.** A hybrid crystal carries the compensation charges and the one-center screened terms of its spheres, but no crystal HSE06 energy or gap is checked against an all-electron or literature value. A water molecule in a periodic box stays about 1 mHa from the molecular HSE06, unexplained; PAW HSE06 minus PBE for H₂O is 7.5 mHa below the all-electron valence-only value (grid resolution and two omitted one-center pieces are part of it). Forces and stress are analytic in Rust; the analytic gradient carries the one-center screened and compensation terms. A hybrid setup written without its `sites` block raises. | Open. See [Screened hybrid](#dft-screened-hybrid). |
| Periodic analysis | Berry phase and polarization | `get_polarization` does not exist: `AttributeError`. `get_dipole_moment()` raises `PropertyNotImplementedError`. Only a Wilson-loop primitive exists in Rust. | No plan. |
| Periodic analysis | Tetrahedron integration weights for the total energy and the Fermi level | `dos(integration="tetrahedron")` and `pdos(integration="tetrahedron")` compute the linear-tetrahedron density of states on a full Gamma-centered mesh; the crystal SCF still occupies its bands with `smearing`, and there is no Bloechl curvature correction. A shifted mesh raises `ValueError`. | No plan. |
| Molecular PAW | Scalar-relativistic dataset generation (`generate_paw`, `generate_upaw`, whose default relativity is scalar) | `NotImplementedError` ("native spherical atoms currently support nonrelativistic uniform-grid LDA"). `relativity="none"` generates. Datasets already in the library are read as before. | No plan. |
| Molecular PAW | Frozen-core exchange for elements with a core beyond one closed 1s shell (Cl and Cu among them) | HCl and CuH raise `ValueError` ("requires an empty core or a single closed 1s shell"). | No plan. |
| Mean field | Noncollinear initial moments of a crystal; fractional occupations in spinor Kohn-Sham | A molecule takes one moment vector per atom ([noncollinear moments](../guide/open_shell.md)); a crystal given moment vectors raises `NotImplementedError` (collinear moments start a spin-orbit crystal's noncollinear magnetization). Spinor Kohn-Sham fills integer occupations, so a nearly degenerate frontier is not shared as collinear Kohn-Sham shares it. | No plan. |
| Properties | Hirshfeld, Voronoi and Bader charges; per-atom magnetic moments | `get_charges()` and `get_magnetic_moments()` raise `PropertyNotImplementedError`. The total moment `magmom` of a `dft` run is available. | No plan. |
| Properties | Finite electric field (`electric_field=`) | Not a keyword: `TypeError`. | No plan. |
| Quantum | Circuit execution of the `subspace-*`, `vasqa` and `valsqa` methods | `NotImplementedError` ("subspace SDK execution requires a multi-reference executor", "native architecture search currently uses local state vectors and fresh chains"). Checkpoint and restart of the `subspace-*` methods is available ([checkpoints](../guide/checkpoints_qpe.md)). | Listed in the cutover table (multi-reference SDK execution). |
| Quantum | Analytic variational forces on some references | Variational forces (`vqe`, `adapt-vqe`, `vasqa`, `valsqa`) are analytic on PAW-LCAO, closed-shell or unrestricted Hartree-Fock references, with canonical, MP2 and frozen-orbital active spaces. A `dlpno-mp2` active space, a Kohn-Sham, spinor or fractional-occupation reference, a Z2-tapered register and the double-factorized two-body representation raise `NotImplementedError` (the tapered register `ValueError`). | The remaining references are later MG-ZA steps (`dev_plan/RUST_MIGRATION_PLAN.md`). |
| Quantum | Measured RDMs and device forces of a tapered or subspace run; device forces past 256 measured elements | With `taper=True`, `measured_rdms()` and the device force raise `NotImplementedError`. The device force propagates its statistical error exactly up to 256 unique measured RDM elements and raises `NotImplementedError` above. Its parameter Hessian costs `1 + 2 R^2` energy evaluations for `R` Pauli rotations, refused by `measurement_budget` before any job. | No plan. |
| Quantum | Other low-rank two-body representations (tensor hypercontraction); forces and the factorized measurement on the double-factorized one | `two_body` takes `"representation": "dense"` (default) or `"double-factorized"` with its `truncation` in eV ([the double-factorized representation](#double-factorized-representation)); any other name raises `ValueError`. On a double-factorized model the analytic forces, `mp2_natural_orbitals()` and `measurement_scheme="double-factorized"` raise `NotImplementedError`. | No plan. |

## Removed on purpose

Variational forces on an all-electron basis (`basis="HAO"` and the Gaussian bases)
raise `NotImplementedError` by owner decision: the variational force is analytic
and PAW-LCAO only, and the finite-difference route it replaced survives only as a
test oracle.

The `engine` and `engine_options` arguments no longer exist (`TypeError`). Rust is
the only implementation, so there is nothing to select, and there is no fallback
route to an older one.

A Gaussian-process proposal model for `valsqa` is not planned, and `proposal="gp"`
raises `ValueError` (search proposals are `gradient` or `uniform`). The learned
proposals are the native pairwise Bradley-Terry ranker and graph network:
`analysis.proposals.fit(kinds=("ranker", "graph"))` trains them, and `valsqa` draws
from the ready one (`proposal_model=`).

Examples that depend on a row above are kept in `examples/` and fail at that
request. `scripts/check_examples.py` reports the failure instead of filtering the
example out.
