# DFT development plan

Living plan for `Mandacaru(method="dft")`. Dated decisions, measured numbers and
post-mortems go to `HISTORY.md` (sections dated 2026-09-30 and 2026-10-01); this
file says **what exists, what is left, in what order, and how each step is
accepted**.

Status as of 2026-10-01: nothing below is committed. The full default suite
passes except the pre-existing ONCV library-drift test
(`test_oncv.py::TestLibrary::test_shipped_h_matches_a_fresh_generation`).

---

## 1. What exists

### 1.1 Molecular Kohn-Sham (`algorithms/dft.py`)

- `Mandacaru(method="dft", xc="lda"|"pbe"|"r2scan", dispersion=None|"d4")`;
  CLI `--method dft --xc ... --dispersion d4`. `device` is ignored.
- Closed-shell RKS in the Loewdin basis of the shared builder; Hartree from the
  same ERI tensor as the many-body Hamiltonian; XC on the grid
  (`integrals/exchange_correlation.py`): LDA/PBE kernels shared with the radial
  atom (`basis/xc.py::xc_partials`), r2SCAN in `basis/r2scan.py` (jax autodiff,
  validated against libxc to 2e-7).
- Every basis family; PAW-LCAO needs no one-center XC term (frozen
  augmentation, see HISTORY); smooth core inside the functional; per-atom
  `core_correction_offset` keeps the valence-only energy zero.
- D4 from the external `dftd4` package (`[dispersion]` extra); PBE and r2SCAN.
- Result exports the many-body Hamiltonian in the KS orbitals
  (`as_quantum_problem()` -> ADAPT-VQE).
- **Analytic forces** (`kohn_sham_gradient`): the integral derivatives of
  `pseudo_nuclear_gradient` through its `energy=`/`field=` hooks, XC
  linearized about the converged density, moving partial core, D4 gradient.
  LDA/PBE agree with finite differences to 1e-6 eV/Angstrom.

### 1.2 Periodic Kohn-Sham (`pseudopotentials/periodic_paw.py`, `algorithms/periodic_dft.py`, `integrals/reciprocal.py`)

- Any `atoms.pbc` -> crystal path: Bloch sums of the PAW-LCAO basis on the
  cell grid, `kpts` Monkhorst-Pack mesh reduced by time reversal, smearing
  (`fermi-dirac`, `gaussian`, `methfessel-paxton`), Pulay mixing with a Kerker
  preconditioner, energy as a functional of the output density matrices.
- Electrostatics: one neutral charge (smooth density + compensation multipoles
  + Gaussian ions) in reciprocal space; compact charges summed densely with an
  analytic on-site tail -- the Fourier filter that removes the finite-grid
  dependence.
- Validated: molecule-in-a-box equals the extrapolated molecular limit (H2,
  H2O); k-mesh equals its supercell to 1e-8 Ha/cell (Si 2x2x2); Al metal with
  three smearings.

### 1.3 Periodic validation (Vera, 2026-10-01) -- passed

PAW-LCAO LDA, DZP, h = 0.25 Angstrom, Gamma-centered meshes; independent
reference pyscf (GTH-pade, gth-dzvp) for trends:

- Si equation of state: a0 = 5.495 Angstrom, B = 90 GPa at 4x4x4 (~5.487 /
  89 with the 6x6x6 shift); pyscf 5.431 / 92. 1.1 % above pyscf, 1.6 % above
  plane-wave LDA (see K9).
- Si gaps (a = 5.4): mesh indirect 0.963 (4x4x4) / 0.846 eV (6x6x6) against
  pyscf 0.656 / 0.682; Gamma direct 2.69 / 2.70 against 2.60 / 2.62. The
  Gamma15/Gamma2' crossing at 5.6 Angstrom (gap 2.04 eV) is reproduced by
  pyscf.
- Al: smearing choices agree to 1.9 meV at 6x6x6; F <= E for FD/Gaussian;
  entropy formulas verified by finite difference to 1e-11; electron count
  exact; k-mesh not converged at 8x8x8 (Gamma-centered fcc), same pattern in
  pyscf (4 -> 6: -0.384 vs -0.387 eV).
- Grid insensitivity at fixed filter: 1.8e-6 Ha spread over h = 0.30 / 0.25 /
  0.20; rigid translation of the atoms by 0.1 Angstrom: 7e-7 Ha.
- Vera's memory: `.claude/agent-memory/vera-validator/periodic-bloch-dft-si-al-2026-10-01.md`;
  scripts in `/tmp/vera_pbc/` (not durable).

---

## 2. Known issues to settle before new features

| # | Issue | Where | Action |
|---|---|---|---|
| K1 | r2SCAN energy surface rough at ~1 uHa on a finite grid; molecular r2SCAN forces cannot be checked by FD beyond ~0.02 eV/Angstrom | `basis/r2scan.py`, `algorithms/dft.py` | Measure on the **periodic** path, whose electrostatics is filtered. If the roughness persists it is the alpha = (tau - tau_W)/tau_unif ill-conditioning in the tails: try a density threshold for the meta-GGA terms (as libxc does) and re-measure the cubic-fit residual (HISTORY 2026-10-01). |
| K2 | r2SCAN never run on a crystal | `algorithms/periodic_dft.py` (`_tau`, `bloch_gradients`) | Add a test: Si r2SCAN converges, tau >= 0, stationarity of F in the density matrices at one k. |
| K3 | Molecular DFT still pays for the builder's RHF + MO ERI transform | `algorithms/dft.py::_build_hamiltonian` / `_hamiltonian_from_atoms.py` | Build integrals without `molecular_hamiltonian(mo_basis=True)` for `method="dft"`; export the KS-basis Hamiltonian only when `as_quantum_problem()` is asked for (lazy). Accept: same energies, wall time drops. |
| K4 | Molecular DFT uses the ERI for J (O(M^4)); periodic uses a Poisson solve | `algorithms/dft.py::KohnSham._hartree` | Optional: switch molecules to the Poisson route with compensation charges on the grid (reuse the periodic electrostatics with an isolated kernel). Changes the energy assembly: validate against the ERI route to 1e-6 Ha before switching. |
| K5 | CLI has no `--kpts` / `--smearing` | `cli.py` | Add both; refuse for non-periodic geometries with a clear message. |
| K6 | Crystal examples missing; Doug's docs predate the crystal path | `examples/new/`, `docs/source/guide/dft.md` | Doug: a Si band-gap example and an Al smearing example (outputs under `examples/new/outputs/`). |
| K7 | Pre-existing "GPAW" in docstrings/docs (FYI from Rita) | `basis/filtering.py`, `pseudopotentials/confinement.py`, guides | User decision: the clean-room rule was for new work; check `dry_run.py` and `basis_report.py` at least never print it. |
| K8 | `DFT_PLAN.md` itself | repo root | Allowed in `test_repo_hygiene.py::ROOT_ALLOWED`; decide whether it is gitignored like `TODO.md`. |
| K9 | Si lattice constant 0.06 Angstrom (1.1 %) above pyscf/gth-dzvp | basis vs dataset | Scan basis size (DZP -> TZP) and the filter cutoff at fixed h; if it persists, compare the Si dataset's atomic properties. Not a code bug as far as known. |
| K10 | Absolute eigenvalues and the Fermi level shift with h (lowest Si level -8.76 / -7.98 / -7.10 eV at h = 0.30 / 0.25 / 0.20) while energies and gaps do not | `algorithms/periodic_dft.py::_potentials` | The G = 0 of the electrostatic potential is dropped, so the eigenvalue zero is arbitrary and grid-dependent. Fix the reference (e.g. the average of the total local potential, including the short-range parts the grid does not carry) or document that eigenvalues are relative; until then never compare Fermi levels across h. |
| K11 | `band_gap` is a mesh gap (upper bound: the Si CBM is off-mesh) | `PeriodicKohnShamResult.band_gap` | Say so in the docstring and the guide; Phase B's band paths give the true gap. |

---

## 3. Remaining features, in order

Each phase ends with the agent pipeline: tests (Ted) -> physics (Vera) ->
review (Rita) -> docs (Doug), and a dated HISTORY entry.

### Phase A -- Symmetry-reduced k-points (IBZ)

The largest speed-up available: Al 6x6x6 solves 112 k-points today (~8 min);
the IBZ has ~16-28.

1. Space group with spglib (`core/symmetry.py::crystal_symmetry`,
   `irreducible_kpoints` already exist) -> IBZ points and weights.
2. **Density symmetrization**, both pieces:
   - smooth density on the grid: rho_sym(G) = (1/N_S) sum_S rho(S^T G)
     e^{-i G.tau_S}, in reciprocal space (the FFT set must map to itself:
     require a grid commensurate with the operations; refuse otherwise);
   - compensation multipoles q^A_LM: rotate with Wigner D-matrices of the
     complex Y_LM convention (`basis/_angular.py`) and permute atoms by the
     operation's atom map.
3. Projections/atomic density matrices need no separate treatment if q is
   symmetrized (the energy depends on them only through q and D^ion).
4. Accept: IBZ energy equals the full time-reversed mesh to 1e-8 Ha/cell for
   Si, Al and a low-symmetry cell (hexagonal/monoclinic); speed-up measured.

### Phase B -- Band structure, DOS, PDOS, fatbands

1. **Non-self-consistent bands**: freeze the converged potential (V_grid, w_a)
   and diagonalize H(k) on a path (`algorithms/kpath.py::resolve_path`,
   `band_path`). Needs `PeriodicPAW` able to build `KPointMatrices` for
   arbitrary k after the SCF (factor `_kpoints` into a per-k builder).
2. **DOS**: Gaussian/tetrahedron-free broadening over a dense non-SCF mesh;
   integrated DOS must give the electron count.
3. **PDOS / fatbands**: projection on atomic orbitals with Loewdin or Mulliken
   weights per k (S(k)^1/2 C), summed per atom and per l; checks: weights sum
   to one per state; PDOS sums to DOS.
4. API: `atoms.calc.band_structure(path=...)`, `atoms.calc.dos(...)`,
   `atoms.calc.pdos(...)` returning arrays (eV); plotting left to examples.
5. Accept: Si LDA indirect gap and X/L energies consistent with Vera's mesh
   numbers; band energies at mesh k equal the SCF eigenvalues to 1e-10 Ha.

### Phase C -- Periodic forces and stress

1. Forces: the molecular gradient's structure carries over -- Pulay terms from
   moving Bloch sums (displaced sampling with phases), Hellmann-Feynman from
   moving projector spheres, short-range spheres, compensation charges in the
   dense G sums (analytic G derivative: -i G times the structure factor), the
   Gaussian ions, the erfc pair term; plus the moving filtered core.
   Smearing: forces are derivatives of F, not E0.
2. Accept: FD of F on a fixed grid to 1e-4 eV/Angstrom (Si with one atom
   displaced; Al with a vacancy-free distortion); sum of forces ~ 0.
3. Stress: strain derivative of every term (kinetic, Hartree/ion G sums,
   XC (incl. the GGA gradient term), sphere quadratures); accept against FD of
   F with respect to strain. Then variable-cell relaxations via ASE filters.
4. `periodic_forces.py` holds the Bloch-method precedent; reuse its stress FD
   helper for the acceptance test.

### Phase D -- Spin polarization (molecules and crystals)

1. UKS: spin-resolved densities; LDA/PBE need the spin-polarized kernels
   (zeta interpolation) -- write them once in jax as r2SCAN is, energy
   density f(rho_up, rho_dn, sigma_uu, sigma_ud, sigma_dd, tau_up, tau_dn),
   autodiff partials; check against libxc (pyscf) pointwise.
2. Molecules: `n_alpha != n_beta` no longer refused; export of the many-body
   Hamiltonian in the UKS natural orbitals (as UHF does).
3. Crystals: two Fermi levels or one with fixed moment; `magmoms` from ASE.
4. Accept: O2 triplet below singlet; Fe bcc moment ~2.2 mu_B (LDA, with the
   caveat of the LDA dataset).

### Phase E -- Hybrid functional HSE06

1. Screened exchange kernel 4 pi/G^2 (1 - exp(-G^2/4 omega^2)), omega =
   0.11 a0^-1, on top of PBE.
2. Molecules: build the screened ERI (`two_body_with_kernel` already takes a
   solver) and add 1/4 of the short-range exact exchange; r2SCAN-like
   generalized KS.
3. Crystals: exact exchange couples k-points (sum over k-q); cost O(Nk^2 M^4)
   -- start with Gamma-only supercells, then a q-mesh.
4. Accept: molecular HSE06 vs pyscf for H2O/CH4 gaps; Si gap increase over PBE
   ~0.6-0.7 eV.

### Phase F -- VV10 nonlocal correlation

1. Roman-Perez-Soler interpolation (kernel tabulated over q1, q2, convolution
   by FFT on the grid), periodic and molecular (isolated kernel).
2. Pair with r2SCAN (rVV10, b = 15.7) as the recommended dispersion-inclusive
   meta-GGA; D4 stays the default for PBE.
3. Accept: Ar2 / benzene dimer binding vs reference curves; periodic layered
   case (graphite interlayer distance).

### Phase G -- Remaining post-processing

1. Fermi surface: energies on a dense non-SCF mesh, written as `.bxsf`
   (XCrySDen) for visualization; tests check the file round-trips.
2. Dielectric constant:
   - molecules: polarizability from finite fields (uniform field as an extra
     linear potential; `core/dipole.py`), accept against FD of the dipole;
   - solids: Berry-phase polarization under finite field, or DFPT -- choose
     Berry phase first (reuses the k-mesh machinery, no response code).
3. Accept: H2O polarizability vs pyscf; Si epsilon_infinity (LDA ~12-13).

### Phase H -- Dataset and dispersion follow-ups

1. Dedicated PAW PBE datasets (user decision 2026-09-30: LDA for now) and a
   check of whether they improve PBE/r2SCAN trends (Vera saw the PBE-LDA bond
   shift at ~70 % of the all-electron value).
2. Periodic D4 (dftd4 supports lattices: pass the cell and pbc).

---

## 4. Order and dependencies

```
K-items (Vera's report first)
  -> A (IBZ)           speed; everything periodic below gets cheaper
  -> B (bands/DOS/PDOS) needs the per-k builder; benefits from A
  -> C (forces/stress)  independent of B; needed for relaxations
  -> D (spin)           touches xc kernels, molecular and periodic SCF
  -> E (HSE06)          after D only if spin-polarized hybrids are wanted
  -> F (VV10)           independent; after K1 (meta-GGA tails)
  -> G (Fermi surface, dielectric)  needs B's non-SCF machinery
  -> H                  whenever the datasets are generated
```

## 5. Ground rules that apply to every phase

- Everything goes through `Mandacaru(method="dft", ...)`; engine classes may
  be tested directly, drivers never.
- The word "GPAW" appears nowhere in new code, comments, docs or tests.
- Test files mirror `src/`; new tests stay inside the budgets (one test
  < 180 s, session < 1200 s, RSS < 8 GB) or are marked `slow`.
- Every new functional, smearing, mixing or method gets a bibliography entry
  and is wired into `citation_keys`.
- Each phase's numbers go to `HISTORY.md` with the date; this plan is updated
  (move the phase to "What exists") rather than appended to.
