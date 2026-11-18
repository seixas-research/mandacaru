(dev-paw-formulation)=
# PAW-LCAO formulation

Mandacaru implements a frozen-augmentation Blöchl projector augmented-wave (PAW)
formulation across molecular (`paw/molecular/`) and periodic (`periodic/`)
calculations, with separate one-center treatments for Kohn–Sham density
functional theory and two-body wave-function methods.

## Linearized Kohn–Sham vs. exact-$\Delta W$ variational / Hartree–Fock

1. **Kohn–Sham DFT (`method="dft"`).** Inside each augmentation sphere, the
   all-electron minus smooth-plus-compensation one-center Hartree and
   exchange–correlation terms are **linearized** around the reference atom's
   projector density matrix $D^0$ (`paw/onecenter_terms.rs`,
   `paw/molecular/onecenter.rs`). This gives a constant species offset
   (`dft_constant` / `one_center_energy`), the linear projector coupling
   `ks_difference`, and full nonlinear multipole compensation
   $\frac{1}{2} Q^T U Q + Q^T W^s$ (molecules) or `hats` + `u`/`uion`/`linear`
   (crystals), using Blöchl compensation rather than a self-consistent nonlinear
   Kresse–Joubert one-center KS solver.
2. **Hartree–Fock and variational quantum methods.** For `rhf`, `uhf`, `ghf`,
   and the variational solvers (`vqe`, `adapt-vqe`, `vasqa`, `valsqa`), the
   two-body one-center electron–electron Coulomb tensor $\Delta W_{abcd}$
   (`one_center_tensor`, `PairCorrection` signed rank-1 factors) and the
   exact core–valence exchange operator (`FrozenCore` from format-3
   `core_states`) enter explicitly in the Hamiltonian, while the linearized
   one-center XC and core–XC offsets of the sphere remain frozen at $D^0$.
3. **Screened hybrids (`hse06`) and beyond-multipole remainders.** Screened
   molecular hybrids add the range-separated `erfc(omega r)/r` one-center
   Coulomb/exchange tensor and spin-polarized short-range exchange (`Screened`,
   `ScreenedAtom`), requiring empty cores or closed subshells verified by
   `FrozenCore::from_states`. In periodic screening, optics, BSE, and Wannier
   downfolding (`screening/transitions.rs`, `wannier/downfold/pairs.rs`,
   `optics/{bse,dielectric}.rs`), `periodic/setup/remainder.rs` supplies the
   beyond-multipole reciprocal transform $E_{ab}(\mathbf{g})$ of
   $\phi_a^*\phi_b - \tilde{\phi}_a^*\tilde{\phi}_b - \hat{Q}_{ab}$ together
   with `onsite_remainder` ($\Delta W$). Periodic KS SCF intentionally keeps
   the linearized multipole scheme without wiring `Remainder` into SCF: the
   zero-multipole remainder produces identically zero potential outside
   $\max(r_g, r_c)$, and `crates/mandacaru-core/tests/paw_audit.rs` verifies
   that its inter-atomic cross term is $< 10^{-5}\ \text{Ha}$ at non-overlapping
   separations (`Si`: `7.2e-7 Ha`) and shifts by only `3.1e-5 Ha/atom` under
   $-10\%$ compression of FCC `Cu`.

## Smooth-core XC convention (`nlcc`)

When nonlinear core correction is active (`nlcc.applied = true`,
`source = "smooth_core"`), the real-space grid evaluates all semilocal and
meta-GGA XC terms (`energy`, potentials, and `core_tau` when required) on
$\tilde{n}(\mathbf{r}) + \tilde{n}_c(\mathbf{r})$ (`smooth_core_density`). The
one-center reference tables evaluate the smooth side on $\tilde{n}^1_0 + \tilde{n}_c$
and subtract XC of the isolated frozen core from the all-electron side on
$n^1_0 + n_c$, keeping the frozen core's self-XC out of total energies.

## Compensation terms and molecular grid cross-term (`ws`)

Every dataset stores a single compensation radius $r_g = \min_l r_c$ with
polynomial shape $g_L(r) \propto r^L (1 - r^2 / r_g^2)^3$. On molecules:

- Overlap, projectors, `ks_difference`, local short-range spheres, ionic
  couplings, and compensation self-interactions $U_{ab}$ are evaluated on
  atom-centered spherical quadratures and remain translation invariant
  ($|\Delta S| < 10^{-16}$, mesh-free net force sum $< 4\times 10^{-14}\ \text{Ha/Bohr}$).
- The cross-term $W^s_a = \sum_k \phi_i(\mathbf{r}_k)\phi_j(\mathbf{r}_k) V_a(\mathbf{r}_k - \mathbf{R}_A)\, dV$
  is summed on the spatial grid (`paw/molecular/mod.rs`) because $\phi_i\phi_j$
  spans two-center basis products against long-range multipole potentials
  $V_a$. Under rigid shifts on fixed grids (`paw_audit.rs`), `ws` contributes
  ~1–2% (`H2`) to ~15–26% (`O2`, `H2O`, `CO`) of the grid egg-box ripple,
  shrinks monotonically with grid spacing, and matches the analytic
  `compensation_center` + `compensation_basis` gradient sum to
  $10^{-10}\text{–}10^{-7}\ \text{eV/\AA}$.

## Format 3 standard, validation gates, and generator invariants

- **Format 3 `Variational` standard:** `PawDataset` requires format version 3
  with stored `core_states` (`core_wave_n{n}_l{l}`), generated under the
  variational radial XC scheme (`models::spherical`, `onecenter_terms.rs`).
- **Validation (`paw/validate/`) and `GhostStateError`:** `mandacaru-build` and
  `pseudopotentials.generation` search unitarily (`norm_deficit = 0 (unitary)`)
  for ghost-free parameters; in `mode="refuse"` or `"enforce"`, a failing
  search raises `GhostStateError` rather than emitting a defective dataset.
- **`UPAW-LCAO` (`norm_deficit = 0`):** The unitary variant fixes $q = 0$,
  yielding unitary augmentation ($T^\dagger T = I$) with exact norm
  conservation (`norm_deficit = 0 (unitary)`).

## Molecular core and spin-orbit limits

- **Hartree–Fock and screened hybrids (`hse06`):** `FrozenCore::from_states`
  supports empty cores and any closed-subshell core whose stored `core_states`
  reproduce the core electron count and `core_density` (to `1e-7`). Partially
  filled core subshells or missing/inconsistent `core_states` are refused by
  name (`core_refusal` / `ValueError`).
- **Spin-orbit (`lda-fr`, `pbe-fr`) molecules:** `method="dft"` runs spinor
  Kohn–Sham with the PAW spin-orbit projector term (including analytic forces),
  and `ghf`, `vqe`, `adapt-vqe`, `vasqa`, and `valsqa` run via the spinor `ghf`
  reference. Spin-restricted/unrestricted `rhf`/`uhf` and molecular DFT+U raise
  on spin-orbit datasets.
