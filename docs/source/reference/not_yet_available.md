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
| Periodic quantum | `bloch-vqe` and `bloch-adapt-vqe` (quantum methods on a crystal; spectral functions) | `method="bloch-vqe"` raises `ValueError` ("unsupported native method"). Any quantum method on a periodic geometry raises `NotImplementedError` ("periodic geometry calculations currently provide native DFT"). | Plan: `dev_plan/BLOCH_VQE_PLAN.md`, six stages, not started. |
| Dispersion | D4 (`vdw="d4"`) | `vdw="d4"` raises `NotImplementedError` ("the D4 dispersion correction is not yet implemented"). Native dispersion options are `vdw="rvv10"` (rVV10 nonlocal correlation) and `vdw="ts"` (Tkatchenko-Scheffler pairwise dispersion). | Plan: `dev_plan/D4_PLAN.md` (D4). TS is implemented natively in Rust. |
| Response | Constrained RPA and Wannier downfolding (`screening_cutoff`, Hubbard U and Hund J computed from the screening; a `U` given as input is [DFT+U](../guide/dft_u.md)) | `calc.downfold` does not exist: `AttributeError`. Wannier localization itself (`calc.wannier()`) is native. | Plan: `dev_plan/CRPA_PLAN.md`. The polarizability accumulator and its Dyson solve exist in Rust; the augmented transition-density tables, the binding and the Python call do not. |
| Mean field | Ensemble occupation of a degenerate frontier shell | **Nothing is raised.** Restricted O2 (PAW-LCAO SZ, LDA, `grid_spacing=0.3`, bond 1.21 angstrom, no magnetic moments) converges in a few iterations to -819.798 eV, 40.56 eV above the triplet at -860.362 eV on the same grid. | Open defect, found 2026-10-29. Request the spin state explicitly (initial magnetic moments, or `unrestricted=True`) for any molecule with a degenerate frontier shell. Cause and scope are in the maintainers' session log for that date. |
| Mean field | Active-space orbital selection for an open-shell reference (`active_space={"method": "mp2", ...}`) and correlating-pair constraints | Open-shell MP2 raises `ValueError` ("requires a closed-shell Hartree-Fock reference"). Correlating-pair constraints have no option. | No plan; listed in the cutover table. |
| Periodic DFT | `r2scan` product-rule check | **Runs, and is unchecked.** The molecular r2SCAN energy and force were repaired on 2026-10-29 (gradient taken from the orbitals, as the kinetic density is). The periodic code has the same stencil pattern and was not examined. | Unverified; treat periodic `r2scan` and `r2scan+rvv10` energies as provisional. |
| Run log | `[SCF SETUP]` and `[SCF ITERATIONS]` of a crystal calculation | A crystal log holds `[SYSTEM]`, `[BASIS]`, `[ELECTRONS]` and `[SCF SUMMARY]` only; `[ELECTRONS]` prints its particle counts as NumPy scalars. | Defect. The blocks missing from other logs are in [the run log reference](#run-log-not-written). |
| Periodic DFT | DFT+U with a symmetry-reduced k mesh (`symmetry=True` on a crystal whose space group permutes atoms) | The Rust loader raises: "DFT+U needs the shell occupations symmetrized with the density, and the symmetry table of this crystal has no such map yet: use a full k mesh without symmetry (symmetry=False)". A calculation with nonzero initial magnetic moments already keeps the full mesh. Read from the code, not requested on this build. | Plan: `dev_plan/DFTU_PLAN.md` (HU-4). Uses native space-group operations to symmetrize shell occupations over atom maps and rotations. |
| Periodic DFT | DFT+U with separate `U` and `J` (Liechtenstein form) | There is one form, `dudarev-fll`. `J` is accepted and enters only as `U_eff = U - J`; no option selects another form. Read from the code. | Plan: `dev_plan/DFTU_PLAN.md` (HL-0..HL-6). Gaunt coupling tensor and Slater integrals ($F^0, F^2, F^4, F^6$) implemented in `models/hubbard_slater.rs`; periodic SCF wiring in progress. |
| Periodic DFT | `U` from linear response (Cococcioni and de Gironcoli) | Not an option: `hubbard_u` takes `U` and `J` as input. Constrained RPA (the row above) computes an interaction on a Wannier basis, not this `U`. Read from the code. | Plan: `dev_plan/DFTU_PLAN.md` (LR-1..LR-13). Susceptibility kernel inversion and odd-part fitting implemented in `models/hubbard_response.rs`; solver hooks and Python API in progress. |
| Periodic DFT | DFT+U on a molecule, with a screened hybrid, with a spin-orbit dataset or in single precision | Molecule: `ValueError`. `hse06`: `NotImplementedError`. Spin-orbit (fully relativistic) dataset: `NotImplementedError` (a molecule on such a dataset runs spinor Kohn-Sham without DFT+U; spinor molecular DFT+U is not implemented). `precision="float32"`: `ValueError`. Read from the code, not requested on this build. | Plan: `dev_plan/DFTU_PLAN.md`. Noncollinear spinor Dudarev model implemented in `models/hubbard_spinor.rs` (SO-1..SO-8). Molecular Kohn-Sham core gate lift in MP-1..MP-5. Screened hybrids and float32 remain unsupported. |
| Periodic DFT | Validated and analytic HSE06 for a crystal | **Runs, and is unvalidated.** A hybrid crystal carries the compensation charges and the one-center screened terms of its spheres, but no crystal HSE06 energy or gap is checked against an all-electron or literature value. A water molecule in a periodic box stays about 1 mHa from the molecular HSE06, unexplained; PAW HSE06 minus PBE for H₂O is 7.5 mHa below the all-electron valence-only value (grid resolution and two omitted one-center pieces are part of it). Forces and stress are analytic in Rust; the analytic gradient carries the one-center screened and compensation terms. A magnetic spinor hybrid, and a hybrid setup written without its `sites` block, raise. `resources` threads do not yet bound the Bloch kernels of the Python setup. | Open. See [Screened hybrid](#dft-screened-hybrid). |
| Periodic analysis | Berry phase and polarization | `get_polarization` does not exist: `AttributeError`. `get_dipole_moment()` raises `PropertyNotImplementedError`. Only a Wilson-loop primitive exists in Rust. | No plan. |
| Periodic analysis | Tetrahedron density of states | `dos(method="tetrahedron")` is not a keyword: `TypeError`. `dos` and `pdos` are Gaussian-broadened. | Planned as stage 5 of `dev_plan/BLOCH_VQE_PLAN.md`. |
| Molecular PAW | Scalar-relativistic dataset generation (`generate_paw`, `generate_upaw`, whose default relativity is scalar) | `NotImplementedError` ("native spherical atoms currently support nonrelativistic uniform-grid LDA"). `relativity="none"` generates. Datasets already in the library are read as before. | No plan. |
| Molecular PAW | Frozen-core exchange for elements with a core beyond one closed 1s shell (Cl and Cu among them) | HCl and CuH raise `ValueError` ("requires an empty core or a single closed 1s shell"). | No plan. |
| Mean field | Noncollinear initial moments (one vector per atom) | `ValueError` ("magnetic_moments must be one finite value per atom"), at the first run. | No plan. |
| Properties | Hirshfeld, Voronoi and Bader charges; per-atom magnetic moments | `get_charges()` and `get_magnetic_moments()` raise `PropertyNotImplementedError`. The total moment `magmom` of a `dft` run is available. | No plan. |
| Properties | Finite electric field (`electric_field=`) | Not a keyword: `TypeError`. | No plan. |
| Quantum | Z2 symmetry tapering (`taper=`) | Not a keyword: `TypeError`. `mapping="parity_reduced"` removes the two qubits fixed by the spin populations, and no other. | No plan. |
| Quantum | Quenching (`quenching=`) | Not a keyword: `TypeError`. ADAPT-VQE always re-optimizes every parameter after a growth step. | Not implemented in the native solver; no plan. |
| Quantum | TETRIS growth and pruning (`tetris=`, `prune=`) | Not keywords: `TypeError`. | No plan. |
| Quantum | Mid-run checkpoints (`checkpoint_every=`) | Not a keyword: `TypeError`. `checkpoint=` writes the final state and `resume=` restarts from it with a fresh optimizer. | Listed in the cutover table. |
| Quantum | Checkpoint and restart of the `subspace-*` methods; circuit execution of the `subspace-*`, `mcas-vqe` and `valqa` methods | `NotImplementedError` ("subspace checkpoint restart is not implemented", "subspace SDK execution requires a multi-reference executor", "native architecture search currently uses local state vectors and fresh chains"). | Listed in the cutover table (multi-reference SDK execution). |
| Quantum | Measured reduced density matrices and forces on a circuit device | With `shots > 0` or `execute_circuits=True`, `get_forces()` raises `PropertyNotImplementedError`. | No plan. |
| Quantum | Double-factorized (and other low-rank) two-body representations and their measurement | `two_body="double-factorized"` raises `ValueError` ("currently accepts Cholesky factors and their tolerance"). | No plan. |
| Quantum | Gaussian-process and pairwise-ranker proposal models for `mcas-vqe` and `valqa` | `proposal="gp"` raises `ValueError` (proposals are `gradient` or `uniform`). The graph `proposal_model` of `valqa` is native. | Listed in the cutover table. |

## Removed on purpose

The `engine` and `engine_options` arguments no longer exist (`TypeError`). Rust is
the only implementation, so there is nothing to select, and there is no fallback
route to an older one.

Examples that depend on a row above are kept in `examples/` and fail at that
request. `scripts/check_examples.py` reports the failure instead of filtering the
example out.
