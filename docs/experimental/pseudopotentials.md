# Pseudopotentials

> **Experimental.** Norm-conserving pseudopotentials are still under
> development and are not part of the stable API. This page is deliberately
> kept outside the Sphinx manual (`docs/source/`) and is not built with it;
> the code stays importable (`pseudopotentials=True` on any driver,
> `carcara --pseudopotentials`), but expect the interface and the numerics to
> change. The stable way to cut the qubit count is the frozen-core
> approximation (`frozen_core=True`).


```python
atoms.calc = Carcara(method="adapt-vqe", basis="FAO",
                               pseudopotentials=True, h=0.15)
```

That switch turns an all-electron calculation into a valence-only one: the core
electrons are removed, the basis becomes smooth pseudo-atomic orbitals, and the
singular $-Z/r$ external potential is replaced by a bounded local channel plus
Kleinman–Bylander projectors.

## Families

*(2026-09-14)* Pseudopotentials come in **families**, and the driver only ever
talks to a family through its registry entry. Three are shipped: the
norm-conserving **Troullier–Martins** family, `"tm"` — what
`pseudopotentials=True` has always meant and still means — Hamann's
**ONCVPSP** family, `"oncvpsp"` (alias `"oncv"`, [below](#oncvpsp-optimized-norm-conserving-vanderbilt-potentials)),
and Blöchl's **PAW** family, `"paw"` ([below](#paw-projector-augmented-waves)),
the first one that is *not* norm-conserving and therefore the first to use the
overlap correction. All of these select the default:

```python
Carcara(method="adapt-vqe", pseudopotentials=True)
Carcara(method="adapt-vqe", pseudopotentials="tm")           # or "ncpp", "ncpp-tm"
Carcara(method="adapt-vqe", pseudopotentials={"family": "ncpp-tm",
                                              "directory": "/my/library",
                                              "size": "DZP"})
```

Names are case-insensitive; an unknown one raises a `ValueError` that lists
the registered families (`carcara.experimental.pseudopotentials.family_names()`).
The family name is also **stored in every pseudopotential file** (`family`
field, format version 2). Files written before the field existed — the whole
bundled library — load as `"tm"`, so nothing was regenerated.

The registry lives in `carcara.experimental.pseudopotentials.families`:

```python
from carcara.experimental.pseudopotentials import (
    PSEUDO_FAMILIES, FamilySpec, register_family, resolve_family)

spec = resolve_family("ncpp")          # -> PSEUDO_FAMILIES["tm"]
spec.name, spec.aliases                # "tm", ("ncpp", "ncpp-tm")
spec.norm_conserving                   # True
spec.generate("O")                     # generate_pseudopotential("O")
spec.get("O")                          # the (cached) library loader
spec.build(atoms, grid, h, charge, spin, options, kinetic)
```

`build` returns exactly the 5-tuple the all-electron path returns —
`(hamiltonian, num_particles, n_spatial_orbitals, integration_profile,
context)` — and `_pseudopotential_hamiltonian` in the driver is now a thin
dispatcher on it. A new family (ONCVPSP with several projectors per channel;
PAW with an overlap correction) is added by registering a `FamilySpec`:

```python
register_family(FamilySpec(name="oncvpsp", description="...",
                           generate=..., get=..., build=...,
                           norm_conserving=True, aliases=("oncv",)))
```

and needs no change to the driver, the dry run or the calculator.

## ONCVPSP: optimized norm-conserving Vanderbilt potentials

*(2026-09-14, step 2 of the family plan.)* The second shipped family is
`"oncvpsp"` (alias `"oncv"`), D. R. Hamann's construction, *Phys. Rev. B*
**88**, 085117 (2013), written from scratch in
`carcara.experimental.pseudopotentials.oncv` on top of the same LDA radial
atom as the TM family:

```python
atoms.calc = Carcara(method="adapt-vqe", pseudopotentials="oncv", h=0.25)
atoms.calc = Carcara(method="vqe", pseudopotentials={"family": "oncvpsp",
                                                     "size": "DZP"})
```

What makes it different from TM is **two projectors per angular-momentum
channel** built from two reference energies, and a local potential that is
*not* one of the channels — so the s channel of H and Li carries projectors
too, and H₂/LiH genuinely run through the $2\times2$ Vanderbilt blocks of the
general separable form.

### Construction

1. **All-electron reference.** The self-consistent LDA atom supplies the
   screened potential. For every valence $l$ two partial waves are integrated
   with the Numerov method on the atom's uniform grid (1500 points per unit
   of $Z$, at least 6000, to 30 Bohr; the regular solution is seeded at the
   origin with a six-term power series — the naive two-term start admixes the
   irregular solution and shifts the O 2s eigenvalue by 0.02 Ha):
   $\varphi_1$ is the valence bound state, refined by outward/inward
   shooting (`bound_state`) so it satisfies the radial equation to fourth
   order; $\varphi_2$ is a second bound state of that $l$ if the atom has one,
   otherwise the scattering state at $\varepsilon_2 = \varepsilon_1 + \Delta$
   with $\Delta = 1$ Ha (`energy_offset`), integrated outward and normalized
   to one inside $r_c$.
2. **Pseudo partial waves.** Inside $r_c$, $\tilde\varphi_i = \sum_{n=1}^{8}
   c_{in}\,j_l(q_n r)$ with the $q_n$ the interleaved zeros of $j_l$ and
   $j_l'$ at $r_c$ (`bessel_wavevectors`). The RRKJ choice — every basis
   function carrying the AE logarithmic derivative — is *not* usable here:
   with $q\,j_l'(qr_c) = L\,j_l(qr_c)$ the Bessel equation turns every
   derivative row into $j_l(q_n r_c)\times$ a polynomial in $q_n^2$, and the
   third-derivative condition becomes a combination of the value and
   second-derivative conditions. Constraints: value and first three
   derivatives at $r_c$ (Hamann's `ncon = 4`, the targets taken from the AE
   radial equation, `matching_targets`) and the **generalized norm
   conservation** $\langle\tilde\varphi_i|\tilde\varphi_j\rangle_{r<r_c}
   = \langle\varphi_i|\varphi_j\rangle_{r<r_c}$ for all $i, j$. In the
   remaining freedom the **residual kinetic energy**
   $E^r_i(q_c) = \tfrac12\int_{q_c}^\infty q^4|\tilde\varphi_i(q)|^2\,dq$
   ($q_c = 5$ Bohr⁻¹, `q_cut`; the transform of the whole wave, AE tail
   included) is minimized *exactly*: the linear constraints are eliminated
   through their null space and the one quadratic constraint (the wave's own
   norm) is a Lagrange multiplier found as a one-dimensional root on the
   Moré–Sorensen branch (`constrained_minimum`). No SLSQP; the norm matrix is
   satisfied to 1e-14.
3. **Local potential.** An even polynomial continuation of the screened AE
   potential inside $r_{cl} = 0.9\,\min_l r_c$ (value and four derivatives
   matched, `polynomial_local_potential`; Hamann's `dvloc0` shift of the
   origin value is available as `local_shift`, default 0), unscreened with
   the Hartree and LDA xc potentials of the pseudo valence density using the
   TM machinery.
4. **Projectors and coupling.** Because $T_l\,j_l(qr) = \tfrac12 q^2 j_l(qr)$
   the projectors are analytic, $\chi_i = \sum_n c_{in}(\varepsilon_i -
   \tfrac12 q_n^2 - V^{scr}_{loc})\,j_l(q_n r)$, zero beyond $r_c$.
   $B_{ij} = \langle\tilde\varphi_i|\chi_j\rangle$ must be symmetric by
   generalized norm conservation — the generator asserts an asymmetry below
   1e-6 Ha (achieved: 7e-11 H, 3e-10 Li, 1e-9 O) — and $D = B^{-1}$
   symmetrized is the $2\times2$ block passed as `nonlocal_coupling` for
   every `(atom, l, m)`. The raw Vanderbilt form is what is stored and used
   (its off-diagonal coupling is real and asserted nonzero);
   `diagonalized_projectors` gives Hamann's orthogonalized pair with a
   diagonal coupling — the same operator to 1e-9.

Defaults (`DEFAULT_CUTOFFS`, Bohr): H 1.30; Li 2.60; C 1.50; N 1.45; O 1.45;
F 1.40 (both channels). Lithium is deliberately *not* pushed further out —
at $r_c = 3.0$ the LiH energy jumps by 0.15 Ha and at 3.3 Bohr the s channel
grows a ghost state at −0.83 Ha (also at 2.2 Bohr): the polynomial local
potential over so wide a core no longer resembles the atom. The coupling
matrices are large ($|D|$ up to ~300 Ha for H) because $B$ has one small
eigenvalue: the two partial waves at $\varepsilon_1$ and $\varepsilon_1+1$ Ha
are nearly proportional in the core. That is harmless to first order — the
nonlocal energy error of the bound wave is exactly $2\,\delta p_1$, the
$1/b$ cancels — and only the second-order term sees $\|D\|$; $\Delta = 2$ Ha
would halve $\|D\|$ but degrades the log-derivative match between the
references (O midpoint 8e-7 → 3e-4) while moving the H₂/LiH energies by only
4e-5 Ha, so $\Delta = 1$ stays.

### Validation (pinned by `test/experimental/test_oncvpsp.py`)

Atomic, on the radial grid (freshly generated potentials; `check_oncv_channel`,
`radial_spectrum` = 3-point Laplacian on a 0.01/0.02 Bohr resampled grid,
Richardson-extrapolated, `log_derivative_ps` solving the nonlocal radial
equation exactly through the homogeneous + two inhomogeneous Numerov
solutions):

| | $r_c$ | $\varepsilon_1, \varepsilon_2$ (Ha) | lowest eigenvalue − $\varepsilon_1$ | $|\Delta L|$ at $\varepsilon_1$ / $\varepsilon_2$ / midpoint | $E^r$ (Ha) |
|---|---|---|---|---|---|
| H s | 1.30 | −0.234, +0.766 | −5.5e-8 | 1.8e-6 / 5.2e-5 / 5.9e-6 | 5.6e-4, 4.8e-3 |
| Li s | 2.60 | −0.106, +0.894 | +7.4e-9 | 3.0e-8 / 1.1e-5 / 2.5e-3 (L = −1.84) | 5.0e-7, 8.0e-5 |
| O s | 1.45 | −0.871, +0.129 | −4.3e-8 | 3.0e-7 / 1.7e-4 / 8.2e-6 | 1.7e-4, 2.3e-3 |
| O p | 1.45 | −0.338, +0.662 | +9.5e-8 | 7.0e-7 / 5.2e-6 / 3.0e-4 | 3.9e-2, 1.9e-1 |

No ghost state below $\varepsilon_1$ in any channel (the next eigenvalue is a
box state above zero), pseudo = AE beyond $r_c$ to 1e-14, norm matrix to
1e-10 or better. The O 2p residual is large at $q_c = 5$ Bohr⁻¹ — a 12.5 Ha
plane-wave cutoff — as it is for any first-row 2p; it is a diagnostic, not a
failure, and the shipped C/N/F potentials behave the same.

Molecular, same grids as the TM pins (H₂ 0.74 Å at h = 0.25 Å, LiH 1.6 Å at
h = 0.30 Å, SZ basis, 4 qubits, ADAPT-VQE with the `qeb` pool, ≤ 4 iterations):

| | ONCV RHF | ONCV FCI = ADAPT | TM RHF | TM ADAPT | ONCV − TM |
|---|---|---|---|---|---|
| H₂ | −1.044179 | −1.058561 | −1.061096 | −1.075333 | +0.017 Ha |
| LiH | −0.774343 | −0.782341 | −0.729555 | −0.740839 | −0.042 Ha |

Both within the 0.05 Ha agreement asked of two independent LDA pseudizations
of the same atoms on a coarse grid; the LiH gap is the larger because the TM
lithium has $r_c = 3.59$ Bohr, wider than the Li–H bond, whereas the ONCV
value sits on a stable plateau ($r_c$ = 2.4–2.6 give −0.7747/−0.7743). The
nonlocal matrix is nonzero and Hermitian to 1e-17; four projectors per
molecule (two radial × one $m$ × two atoms) in two $2\times2$ blocks with
nonzero off-diagonal coupling. DZP on H₂ (20 qubits, RHF only) lowers the
RHF energy to −1.1428 Ha.

**Hardness at h = 0.25/0.30 Å** (`resolution_ratios` $T_{grid}/T_{exact}$
of the basis, `kb_resolution_ratios` grid/radial norm of the projectors):
H₂ basis 0.975 (TM 0.973), projectors 0.86/0.97; LiH basis 1.166/0.959 (TM
1.111/0.956), projectors 1.00/1.00/1.12/1.06 — all inside the ±25 % band the
driver warns at. The ONCV first zetas are as resolvable as TM's; TM has no
projectors on H/Li to compare, and the ONCV second projector (built from the
scattering wave) is the hardest object, still within 14 %.

### Library and files

`library/oncvpsp/{H,Li,C,N,O,F}.parquet` (70–220 kB each, decimated to 0.02
Bohr like the TM library), regenerated with
`build_oncv_library(["H", "Li", "C", "N", "O", "F"])`; `get_oncv(symbol,
directory)` is the family's loader (`directory` defaults to that
subdirectory, so `get(symbol, directory)` finds them and the TM files are
untouched). The record is `ONCVPseudoPotential` (a `PseudoPotential`
subclass: `channels[l]` are `ONCVChannel`s carrying `reference_energies`,
`wavevectors`, `wave_coefficients`, `pseudo_waves`, `projectors` (two),
`coupling`, `vanderbilt`, `residual_kinetic`; `projectors[l]` is the list of
the two radial projectors, `coupling[l]` the block, `v_local_screened`,
`r_cut_local`, `q_cut`, `energy_offset`). Files are the same Parquet/JSON
scheme with `"family": "oncvpsp"` (format version 2; radial tables under
`radial_tables` — `pseudo_wave_l{l}_{i}`, `projector_l{l}_{i}`,
`v_local_screened` — and the scalars in the metadata); `io.py` dispatches on
the family and loading a TM file is untouched. Generation takes 0.5 s (H) to
2.5 s (F).

### Not implemented

No nonlinear core correction, no scalar-relativistic or spin-orbit terms, no
projectors for angular momenta above the valence (those channels see the
local potential alone — a p projector at unbound energies, which Hamann adds
for H and O, is not built, so DZP polarization functions on H see only
$V_{loc}$), no GGA reference atom, and the second reference energy is a fixed
offset rather than Hamann's per-element tuned values. Departures from the
paper: the Bessel wave vectors are the interleaved zeros rather than Hamann's
own choice; the scattering tail entering the residual energy is tapered
between $3r_c$ and $5r_c$ (it does not decay); $r_{cl}$ is a fixed fraction
of the smallest $r_c$; and the raw Vanderbilt coupling is kept instead of the
orthogonalized projectors (equivalent operator).

## PAW: projector augmented waves

*(2026-09-14, step 3 of the family plan.)* The third shipped family is
`"paw"`, P. E. Blöchl's projector augmented-wave method, *Phys. Rev. B* **50**,
17953 (1994), in its **frozen-core, one-center-expansion** form with the
one-center energies **linearized around the reference atom** — a fixed
per-species coupling matrix $D^0$, which makes the dataset behave like an
ultrasoft pseudopotential with an exact PAW reconstruction of the atomic
partial waves. Written from scratch in
`carcara.experimental.pseudopotentials.paw` on the same LDA radial atom as
the other two families, reusing the Numerov partial waves, the Bessel
machinery and the polynomial local potential of the ONCVPSP module:

```python
atoms.calc = Carcara(method="adapt-vqe", pseudopotentials="paw", h=0.25)
atoms.calc = Carcara(method="vqe", pseudopotentials={"family": "paw", "size": "DZ"},
                     basis={"name": "PP", "size": "DZ"})
```

The name has no alias; `family_names()` lists `tm`, `oncvpsp`, `paw` (and the
unknown-family error names all three).

### The transformation

Blöchl's linear map between the smooth wave functions the grid sees and the
all-electron ones,

$$
|\psi\rangle = \mathcal T|\tilde\psi\rangle, \qquad
\mathcal T = 1 + \sum_i \big(|\varphi_i\rangle - |\tilde\varphi_i\rangle\big)\langle\tilde p_i| ,
$$

needs per atom and $l$ **all-electron partial waves** $\varphi_i$, **smooth
partial waves** $\tilde\varphi_i$ equal to them beyond the augmentation
radius $r_c$, and **projectors** $\tilde p_i$ inside $r_c$ dual to the smooth
waves, $\langle\tilde p_i|\tilde\varphi_j\rangle = \delta_{ij}$. The smooth
waves are *not* norm-conserving, so the transformation carries the overlap
operator $S = 1 + \sum_{ij}|\tilde p_i\rangle q_{ij}\langle\tilde p_j|$ with
$q_{ij} = \langle\varphi_i|\varphi_j\rangle_{r<r_c} -
\langle\tilde\varphi_i|\tilde\varphi_j\rangle_{r<r_c}$, and the valence
problem is the generalized eigenproblem

$$
\Big[T + \tilde v_{loc} + \sum_{ij}|\tilde p_i\rangle D_{ij}\langle\tilde p_j|\Big]\tilde\psi
= \varepsilon\,S\,\tilde\psi .
$$

This is exactly what the overlap hook of the general separable form was built
for: the family passes $D$ as `nonlocal_coupling` and $q$ as
`nonlocal_overlap`, and the basis overlap becomes $\tilde S + C q C^\dagger$
through Löwdin, RHF and UHF.

### Construction (`generate_paw`)

1. **Reference atom.** The self-consistent LDA atom; the frozen core density
   $n_c$ (every subshell below the valence) and, per valence $l$, two
   all-electron partial waves: the bound state (Numerov, `bound_state`) and
   the scattering state at $\varepsilon_1 + \Delta$ normalized to one inside
   $r_c$ — the ONCVPSP pair. $\Delta$ is 1 Ha except for H and Li (0.5 Ha,
   `DEFAULT_ENERGY_OFFSETS`): lithium's wave at +1 Ha sits at a pole of the
   logarithmic derivative ($L = +24$ at $r_c$) and the smooth pair then grows
   nodes and a ghost; for hydrogen +0.5 Ha gives a softer second projector.
2. **Smooth partial waves** (`smooth_partial_waves`). Inside $r_c$,
   $\tilde\varphi_i = \sum_{n=1}^{8} c_{in} j_l(q_n r)$ at the interleaved
   Bessel zeros, matched in value and first three derivatives, residual
   kinetic energy beyond $q_c = 5$ Bohr⁻¹ minimized. **The norm is not
   conserved**, but it is *controlled*: the inner-norm matrix is set to
   $(1-s)$ times the all-electron one (`optimize_pseudo_waves(...,
   norm_factor=1-s)`, the ONCVPSP optimizer with a scaled target), so
   $q = s\,\langle\varphi_i|\varphi_j\rangle_{r<r_c}$ is **positive
   definite by construction** and $S \ge 1$. The deficit $s$ is per element
   (`DEFAULT_NORM_DEFICITS`: H 0.05, Li 0.02, C 0.10, N/O/F 0.15; the
   derivative matching alone fixes most of the inner norm, so more is not
   reachable by the expansion). Free waves (`norm_deficit=None`, the residual
   energy minimized with no norm condition at all) are available and were
   tried first: the two reference waves are nearly proportional in the core,
   their dual projectors are large, and an *indefinite* $q$ then made
   $1 + \sum|\tilde p\rangle q\langle\tilde p|$ singular for Li and O at every
   cutoff, number of Bessel functions and $\Delta$ tried (smallest eigenvalue
   −0.03 … 0.03) — the generator now refuses such a dataset
   (`OVERLAP_MINIMUM`, the `overlap_minimum` diagnostic of each channel).
3. **Local potential.** The even-polynomial continuation of the screened AE
   potential inside $r_{cl} = 0.9\min_l r_c$ (`polynomial_local_potential`),
   the screened $\tilde v^{scr}$, **raised at the origin** by Hamann's
   `dvloc0` for the first row (`DEFAULT_LOCAL_SHIFTS`: C 12, N 10, O 6, F 8
   Ha). Without the raise the continued potential (O: −5.8 Ha at the origin)
   binds a spurious 1s-like state of its own in the s channel — a ghost 0.6–1
   Ha below $\varepsilon_{2s}$ — which the near-singular ONCVPSP coupling
   suppresses but the PAW projector term does not; the raise is chosen so the
   s spectrum has nothing between the bound state and the box states.
4. **Projectors** (`assemble_paw_channel`). $\chi_i = (\varepsilon_i - T -
   \tilde v^{scr})\tilde\varphi_i$ inside $r_c$ (analytic, $T j_l = \tfrac12
   q^2 j_l$), $B_{ij} = \langle\tilde\varphi_i|\chi_j\rangle$, and the dual
   set $\tilde p_i = \sum_k (B^{-1})_{ki}\chi_k$; the generator asserts
   $\langle\tilde p_i|\tilde\varphi_j\rangle = \delta_{ij}$ to 1e-8
   (achieved ~1e-14, `duality_error`).
5. **One-center matrices** on the fine inner grid: $q_{ij}$; $\Delta T_{ij}
   = \langle\varphi_i|T|\varphi_j\rangle - \langle\tilde\varphi_i|T|\tilde\varphi_j\rangle$
   (AE side from the radial equation, smooth side analytic); $\Delta V^{scr}_{ij}$;
   and the screened coupling $D^{scr}_{ij} = B_{ij} + \varepsilon_j q_{ij}$,
   which equals $\Delta T + \Delta V^{scr}$ to 1e-15 (`consistency_error`)
   and is symmetric by the generalized Wronskian identity $B_{ij} - B_{ji} =
   (\varepsilon_i - \varepsilon_j)q_{ij}$ (`asymmetry` ≤ 1e-9). It
   reproduces every reference energy exactly in the generalized problem.
6. **Compensation charge and unscreening.** The smooth reference density
   misses $\hat Q = \sum_l f_l q^l_{11}$ electrons (H 0.021, Li 0.0045, O
   0.64); the **monopole compensation charge** $\hat n = \hat Q\,g(r)$, $g
   \propto (1 - r^2/r_g^2)^3$ inside $r_g = \min_l r_c$
   (`compensation_shape`, analytic potential `compensation_potential`),
   restores neutrality with the ion outside the sphere. Unscreening follows
   the norm-conserving families: $\tilde v^{ion} = \tilde v^{scr} -
   v_H[\tilde n_v + \hat n] - v_{xc}[\tilde n_v]$ (→ $-Z_{ion}/r$ outside),
   and the coupling loses the Hartree screening of the augmentation,
   $D^{ion} = D^{scr} - q\int v_H[\tilde n_v + \hat n]\,g$
   (`hartree_screening`).
7. **Frozen one-center constant** (`one_center_energy`). With $D$
   linearized, the double counting of the one-center Hartree and xc energies
   at the reference is a per-species constant, fixed so that the LDA reference
   atom evaluated with the molecular machinery has exactly the all-electron
   valence energy in the norm-conserving convention, $E^{ref}_{val} = \sum_v
   f_v\varepsilon_v - E_H[n_v] - \int n_v v_{xc}[n_v] + E_{xc}[n_v]$:
   $E_{1c} = (E_H[\tilde n_v + \hat n] - E_H[n_v]) + (\int\tilde n v_{xc}[\tilde n]
   - E_{xc}[\tilde n]) - (\int n_v v_{xc}[n_v] - E_{xc}[n_v])$ — H +0.0021,
   Li +0.0006, C +0.030, N +0.053, O −0.107, F −0.570 Ha. It enters every
   molecular Hamiltonian through the new
   `MolecularIntegrals.constant_energy` (next to the nuclear repulsion, also
   in `hartree_fock_hamiltonian`), so PAW totals are comparable with TM/ONCV.

### In a molecule (`build_paw`, `PAWIntegrals`)

The basis is the bound smooth partial waves (first zeta; the `size`
hierarchy applies unchanged), the external potential $\sum_A\tilde v^{ion}_A$
through `Potentials.pseudopotential`, the nonlocal term $C D^{ion} C^\dagger$
and the overlap $\tilde S + C q C^\dagger$, and the constant the ion-ion
repulsion plus $\sum_A E^A_{1c}$. `PAWIntegrals` is a `MolecularIntegrals`
subclass with two additions:

* **Augmented two-body tensor.** Pair densities carry their compensation
  charge, $\rho_{pr} = \tilde\phi_p^*\tilde\phi_r + \sum_A Q^A_{pr}\,g_A$ with
  $Q^A = (C q C^\dagger)^A$ the monopole augmentation moments (the very blocks
  that augment the overlap), so through the new hook
  `MolecularIntegrals.two_body_augmentation()`
  $\langle pq|rs\rangle = \langle pq|rs\rangle_{grid} + \sum_A(Q^A_{pr}W^A_{qs}
  + W^A_{pr}Q^A_{qs}) + \sum_{AB}Q^A_{pr}U_{AB}Q^B_{qs}$, with
  $W^A_{qs} = \int\tilde\phi_q^*\tilde\phi_s V_{g_A}$ on the grid and
  $U_{AB}$ the compensation–compensation Coulomb integrals
  (`compensation_coulomb`: exactly $1/R$ for disjoint spheres, a
  Gauss–Legendre radial/angular quadrature when they overlap, as they do in
  H₂ and LiH). Hartree, exchange and correlation all see neutral atoms.
* **Exact on-site projections.** `PAWIntegrals.projections()` keeps the grid
  for a basis function projected on *another* atom's projectors but evaluates
  the on-site entries by radial quadrature (same center, same $Y_{lm}$). This
  is not cosmetic: the dual projectors are sharp (the two reference waves are
  nearly parallel in the core, $B^{-1}$ large), and the grid value of
  $\langle\tilde\varphi_1|\tilde p_i\rangle$ — exactly $\delta_{i1}$ — came out
  0.7–3× that depending on where the nucleus sat between nodes; the overlap
  correction turned that into a 0.2–0.6 Ha collapse of LiH at h = 0.25 and
  0.35 Å while h = 0.20 and 0.30 were fine. With the on-site entries exact the
  LiH energy drifts smoothly (−0.7467 / −0.7521 / −0.7605 / −0.7671 Ha at
  h = 0.20 / 0.25 / 0.30 / 0.35 Å), and H₂ is −1.0510 / −1.0533 / −1.0652 Ha at
  0.20 / 0.25 / 0.30 Å — more grid-stable than ONCV's −1.072 / −1.044 / −1.094.

The projector functions the grid samples can be either the dual $\tilde p_i$
(`projector_basis="dual"`) or the smooth raw $\chi_k$ with the transformed
blocks $B^{-1}DB^{-T}$, $B^{-1}qB^{-T}$ (`"raw"`, the default,
`PAWDataset.projector_set`). The sampled $\chi_k$ are exact linear
combinations of the sampled $\tilde p_i$, so the energies are **identical to
all digits** (tested); only the reported resolution ratio differs (H at
h = 0.25 Å: raw 0.90/0.78, dual 0.78/0.77).

### Validation (pinned by `test/experimental/test_paw.py`, 70 tests, 11.5 s, peak RSS 0.77 GB)

Atomic, freshly generated (`check_paw_channel`: `paw_spectrum` = the
generalized problem with the 3-point Laplacian on 0.01/0.02 Bohr grids,
Richardson-extrapolated; `paw_eigenstate` + `reconstruct_ae` at 0.005 Bohr for
the first row; `log_derivative_paw` = the ONCVPSP exact nonlocal Numerov
solve with the energy-dependent coupling $D^{scr} - Eq$):

| | $r_c$ | $s$ | $q_{11}$ | duality | lowest eigenvalue − $\varepsilon_1$ | next state | $\vert\varphi_{rec}-\varphi_{AE}\vert$ | $\vert\Delta L\vert$ at $\varepsilon_1$ / $\varepsilon_2$ / midpoint |
|---|---|---|---|---|---|---|---|---|
| H s | 1.30 | 0.05 | 0.0212 | 6.9e-15 | −2.3e-8 | +0.0068 | 1.1e-5 | 1.4e-6 / 4.3e-6 / 1.2e-5 |
| Li s | 2.60 | 0.02 | 0.0045 | 1.8e-14 | +7.5e-9 | +0.0002 | 7.2e-6 | 3.3e-8 / 1.9e-7 / 1.6e-4 |
| O s | 1.45 | 0.15 | 0.1142 | 1.3e-15 | −1.2e-7 | +0.0049 | 7.4e-6 | 8.0e-8 / 1.8e-4 / 7.5e-4 |
| O p | 1.45 | 0.15 | 0.1026 | 1.3e-15 | +3.1e-8 | +0.0324 | 6.5e-5 | 4.1e-7 / 1.0e-5 / 3.9e-3 (L = −4.5) |

The reconstruction is of the *lowest smooth eigenfunction of the generalized
problem*, not of the stored wave (that one reconstructs exactly by duality,
also tested): the smooth eigenfunction differs from the all-electron orbital
by 0.06 (H) to 0.57 (O 2s) and comes back to it to 1e-5 — the transformation
the method is named after. No ghost in any channel (the next state is a box
state above zero), $S$ bounded below by 1.02–1.35, $q$ positive definite and
equal to $s$ times the AE inner Gram matrix, $D^{scr} = \Delta T + \Delta V$
to 1e-15, smooth = AE beyond $r_c$ to 1e-14.

Molecular, same grids as the TM/ONCV pins (H₂ 0.74 Å at h = 0.25 Å, LiH 1.6 Å
at h = 0.30 Å, SZ basis, 4 qubits, ADAPT-VQE with the `qeb` pool, ≤ 4
iterations; energies in eV as the user sees them, Hartree in parentheses):

| | PAW RHF | PAW FCI = ADAPT | ONCV RHF / ADAPT | TM RHF / ADAPT | PAW − ONCV | PAW − TM |
|---|---|---|---|---|---|---|
| H₂ | −28.662 eV (−1.053292) | −29.046 eV (−1.067402) | −1.044179 / −1.058561 | −1.061096 / −1.075333 | −0.248 eV | +0.212 eV |
| LiH | −20.693 eV (−0.760451) | −20.924 eV (−0.768954) | −0.774343 / −0.782341 | −0.729555 / −0.740839 | +0.378 eV | −0.841 eV |

PAW lands between the two norm-conserving families on both molecules (all
three agree within 0.05 Ha, against the 0.1 Ha asked). On H₂ the augmented
overlap has eigenvalues 0.203 / 1.809 (bare 0.199 / 1.757), the
Löwdin-orthonormalized overlap is the identity to 1e-16, and the augmented
two-body tensor keeps the pair-density symmetries to 1e-12. DZ on H₂ (8
qubits, RHF −1.1399 Ha) is variational against SZ.

**Hardness at h = 0.25 Å** (`resolution_ratios` of the basis,
`kb_resolution_ratios` of the raw projectors): H₂ basis 0.98 (ONCV 0.975),
projectors 0.90 / 0.78; LiH basis 1.219 / 0.975 (ONCV 1.227 / 0.971),
projectors 1.00 / 1.00 / 0.88 / 0.85 — all inside the ±25 % band. The second
H projector, built from the scattering wave, is the hardest object; at
h = 0.35 Å (0.66 Bohr) neither family's H projectors are resolved and both
give nonsense.

### Library and files

`library/paw/{H,Li,C,N,O,F}.parquet` (100–385 kB, decimated to 0.02 Bohr;
9.6 s to regenerate with `build_paw_library()`; `get_paw(symbol, directory)`
is the family's loader, `paw_library_path()` its directory, TM and ONCVPSP
files untouched). The record is `PAWDataset` (a `PseudoPotential` subclass:
`channels[l]` are `PAWChannel`s with `reference_energies`, `ae_waves`,
`pseudo_waves`, `projectors` (dual), `raw_projectors`, `overlap_correction`
$q$, `kinetic_difference` $\Delta T$, `potential_difference`,
`coupling_screened` $D^{scr}$, `coupling` $D^{ion}$, `vanderbilt` $B$,
`duality_error`, `overlap_minimum`, `asymmetry`, `consistency_error`;
`coupling[l]` / `overlap_correction[l]` / `kinetic_difference[l]` the
blocks; `v_local` (ionic) / `v_local_screened`, `core_density`,
`smooth_core_density` (pseudized inside $r_g$, stored but unused — see
below), `compensation_radius`, `compensation_charge`, `hartree_screening`,
`one_center_energy`, `energies`, `norm_deficit`, `local_shift`, `q_cut`,
`energy_offset`). Files use the same Parquet/JSON scheme with `"family":
"paw"` (format version 2; every radial table under `radial_tables` —
`ae_wave_l{l}_{i}`, `pseudo_wave_l{l}_{i}`, `projector_l{l}_{i}`,
`raw_projector_l{l}_{i}`, the densities — and the scalars in the metadata).
`io.py` now dispatches the table families (`TABLE_FAMILIES`) by *record type*
on save and by the presence of `radial_tables` on load, so a plain TM record
merely carrying the name `"paw"` keeps the TM layout and the TM loader refuses
it as before. Round trips are lossless and idempotent (tested).

### What is frozen or omitted relative to Blöchl's full method

* **Linearized one-center terms.** The one-center Hartree and xc energies are
  expanded to first order in the density matrix $\rho_{ij}$ around the LDA
  reference atom: a fixed $D^0 = D^{ion}$ per species plus the constant
  $E_{1c}$. The full method recomputes $D_{ij}[\rho_{ij}]$ self-consistently
  from $n^1 - \tilde n^1$ every SCF step; here the error is second order in
  $\rho - \rho^{ref}$.
* **Monopole compensation only.** $\hat n$ carries the $l = 0$ moment of the
  augmentation; the higher multipoles $\hat n^{L}$ ($L \le 2l_{max}$) and the
  multipole moments of $n^1 - \tilde n^1$ beyond the monopole are omitted, so
  the electrostatics of overlapping spheres (H₂: $2r_c = 2.6 > R = 1.4$
  Bohr; LiH: $3.9 > 3.0$) is that of neutral spherical atoms.
* **Frozen core, no nonlinear core correction.** The core density is frozen
  (stored as `core_density`); the core-valence xc of the reference atom stays
  inside $\tilde v^{ion}$ and $D^{ion}$ (unscreened with $v_{xc}[\tilde n_v]$
  only, as TM/ONCVPSP do). The smooth core density is generated and stored
  but does not enter the Hamiltonian; the omitted core–valence xc
  correction is reported in `energies["core_valence_xc_omitted"]` (Li −1.54,
  O −4.60 Ha — the size of the core xc energy itself, which the linearized
  treatment keeps frozen).
* **HF/FCI valence with LDA-generated datasets.** The molecule's exchange and
  correlation are exact within the augmented Coulomb tensor, while the
  one-center xc corrections were linearized at the LDA level — the same
  inconsistency the norm-conserving families carry.
* **LDA only, no relativity, no projectors above the valence $l$, two partial
  waves per channel**, reference energies $\varepsilon_1$ and $\varepsilon_1 +
  \Delta$ rather than tuned per element, and a scaled-norm construction of the
  smooth waves rather than Blöchl's free polynomial pseudization (the price of
  a guaranteed positive definite overlap with this pair of reference waves).
* **Not norm-conserving, but by a controlled amount** ($s$ = 2–15 %): the
  softness gain over ONCVPSP is correspondingly modest (H₂ basis ratio 0.98
  vs 0.975).

## The nonlocal term: general separable form

Every family's nonlocal potential enters through one formula,

$$
H^{NL} = C\,D\,C^\dagger, \qquad C_{\mu p} = \langle\phi_\mu|\chi_p\rangle ,
$$

where $C$ (`MolecularIntegrals.projections()`, an $M\times P$ matrix) is the
only object that touches the grid (the C-accelerated `kb_projections` kernel)
and $D$ is a **block-diagonal** $P\times P$ coupling matrix. Each projector
carries three labels — `atom_index`, `channel = (l, m)` and a radial `index`
within that channel — and $D$ has one block per `(atom, l, m)`, of size
$n\times n$ for $n$ radial projectors in that channel. The family supplies the
blocks as `nonlocal_coupling={(atom, l, m): block}`.

For Troullier–Martins/Kleinman–Bylander there is one projector per channel and
the block is the $1\times1$ matrix $[E^{KB}_l]$ (`kb_coupling_blocks`), so
$C\,D\,C^\dagger$ reduces to the familiar $\sum_p |\chi_p\rangle E^{KB}_p
\langle\chi_p|$ — the test suite checks it agrees with the old rank-one formula
to $10^{-12}$. ONCVPSP and PAW fill $2\times2$ blocks; the machinery
(`projector_blocks`, `assemble_block_matrix` in `carcara.core.hamiltonian`)
validates and assembles them. `kb_nonlocal()` keeps its name (alias
`nonlocal_matrix()`), and the projector resolution check
(`kb_resolution_ratios`) is unchanged.

### Overlap correction

Families whose projectors also change the metric (PAW) pass the blocks of a
second matrix $Q$ in the same layout, `nonlocal_overlap={(atom, l, m): block}`.
The overlap the Löwdin orthonormalization uses then becomes

$$
S \;\to\; S + C\,Q\,C^\dagger ,
$$

(`MolecularIntegrals.overlap()`; the grid overlap alone is `bare_overlap()`),
so the orthonormalized one- and two-body integrals — and everything downstream,
RHF, the UHF natural orbitals, the qubit Hamiltonian — see the augmented
metric automatically. Norm-conserving families pass `None`; `Q = 0` reproduces
the plain Hamiltonian exactly. The PAW family is the first to use it (its $q$
blocks), together with two further hooks on `MolecularIntegrals`:
`two_body_augmentation()` (a correction added to the grid two-body tensor —
the compensation-charge terms) and `constant_energy` (an additive constant
next to the nuclear repulsion — the frozen one-center energies).

## Why they are not optional here

Carcará samples everything on a uniform real-space grid, and that grid must
resolve the shortest length scale in the problem. For an all-electron atom that
scale is the 1s cusp, $a_0/Z$ — 0.066 Å for oxygen, against a practical spacing
of 0.15–0.30 Å. The core is never resolved, and the error does not average out.

The sharpest symptom is the force on an **isolated** atom, whose exact value is
zero by symmetry:

| h (Å) | All-electron (frozen core) | Pseudopotential |
|-------|---------------------------|-----------------|
| 0.20 | 4820 eV/Å | 286 eV/Å |
| 0.15 | 6441 eV/Å | 65 eV/Å |
| 0.10 | 10041 eV/Å | 27 eV/Å |

The two columns behave *qualitatively* differently. Refining the grid makes the
all-electron number **worse**: the nearest node moves into an unresolved cusp
faster than the sampling improves. The pseudopotential column converges, because
there is no cusp left to resolve. Without pseudopotentials, geometry
optimization on this grid is not merely inaccurate — it does not converge.

## The bundled library

`src/carcara/experimental/pseudopotentials/library/` holds one subdirectory
per family. `library/ncpp/` ships norm-conserving Troullier–Martins
pseudopotentials for **every element with Z ≤ 92** (H through U), generated
from scratch by Carcará's own LDA radial atomic solver. They are loaded
automatically by symbol.

```python
from carcara.experimental.pseudopotentials.io import available_elements, get_pseudopotential

pp = get_pseudopotential("Fe")
pp.valence_charge     # 8.0  -- 3d^6 4s^2
sorted(pp.channels)   # [0, 2]
```

The valence includes semicore $(n-1)d$ and $(n-2)f$ shells, so iron is an
eight-electron atom with a d channel rather than a two-electron 4s² one.
Hydrogen and lithium carry a single valence channel, which is the local one, so
in this family H₂ and LiH have no projectors at all — their nonlocal term is
identically zero (the ONCVPSP family gives them two s projectors each). The
ONCVPSP potentials live in `library/oncvpsp/` and the PAW datasets in
`library/paw/` (H, Li, C, N, O, F in each). `io.library_root()` is the common
parent (overridden by `CARCARA_PSEUDO_PATH`); `io.default_library_path()` is
the `ncpp/` directory, `oncv_library_path()` / `paw_library_path()` the others.

To regenerate or extend the library:

```python
from carcara.experimental.pseudopotentials.io import build_library

written, failures = build_library()               # all of Z <= 92
written, failures = build_library(["Ti", "V"])    # or a subset
```

## File format

Pseudopotentials are stored as **Parquet by default**, with JSON retained as an
option — Parquet is about half the size for these radial tables, JSON can be read
without a Parquet engine.

```python
save_pseudopotential(pp, "Fe.parquet")               # Parquet (default)
save_pseudopotential(pp, "Fe.json")                  # JSON, from the extension
save_pseudopotential(pp, "Fe.dat", format="json")    # or stated explicitly
```

**Loading never needs to be told the format.** `load_pseudopotential` resolves it
from the extension, falling back to the file's leading bytes — `PAR1` for
Parquet, `{` for JSON — so a file with an unhelpful name still loads and the two
formats are freely interchangeable.

```python
load_pseudopotential("Fe.parquet")   # extension
load_pseudopotential("mystery.dat")  # magic bytes
```

Every file records its `family` (format version 2). A file without the field
is a version-1 file and loads as `"tm"`; the TM loader refuses a file that
declares another family.

```{note}
Saving is lossless and idempotent: `load` then `save` returns the same tables.
The library is decimated once at generation time (`build_library(stride=...)`,
4 by default) because the generation grid must resolve the all-electron core
while the smooth result does not need it. `save_pseudopotential` itself defaults
to `stride=1`, so repeated round trips never compound.
```

## Choosing the basis

A pseudopotential fixes its own first zeta: the Troullier–Martins construction
pseudizes each valence orbital inside its cutoff, and the Kleinman–Bylander
projectors are built from those specific pseudo-orbitals. Pairing the potential
with an unrelated all-electron radial function would be inconsistent, so an
all-electron family is **refused** rather than silently ignored:

```python
Carcara(method="adapt-vqe", basis="6-31G(d)",
                  pseudopotentials=True)   # ValueError
```

What you *can* vary is the size hierarchy, which refines that pseudo-orbital
instead of replacing it — the extra zetas are split-valence refinements of the
pseudized function, and the polarization shell is split from the outermost
channel:

```python
Carcara(method="adapt-vqe", basis={"name": "PP", "size": "DZP"},
                  pseudopotentials=True)
```

Sizes are the same names as for [the NAO family](basis_sets.md). The default
here stays `"SZ"` (the minimal valence set), because the valence-only space is
usually already at the edge of what a state-vector simulator can hold.

## Limits

The residual force on an isolated atom is still ~30 eV/Å at `h = 0.10 Å`. That
remainder is basis-set incompleteness — a minimal valence s+p shell per atom —
not the core, and it *shrinks* with grid refinement. A polarized multiple-zeta
basis addresses it directly; see [Basis sets](basis_sets.md).

```{warning}
**Geometry optimization does not work yet.** Relaxing H₂O with BFGS diverges:
the forces oscillate between 35 and 944 eV/Å over twelve steps and the molecule
destroys itself (bond angle 104.5° → 164.5°, one O–H contracting to 0.80 Å).

Two independent problems are responsible, and both are in the *energy*, not
only the gradient:

* **Egg-box.** Rigidly translating H₂O on a frozen grid changes the energy by
  ~160 eV/Å — the analytic net force reproduces that numerical derivative to
  within 4 %, so the gradient is faithfully differentiating a discretized energy
  that is itself not translation-invariant.
* **Residual gradient error.** Component by component against finite
  differences, the analytic forces are still too small by roughly a factor of
  eight on the hydrogens (H *y*: ±3.3 analytic vs ±28.4 numerical).

Use the forces for diagnostics and for single-point analysis, not for
relaxation.
```

## Tests and their budget

The experimental suite (`test/experimental/`) is resource-monitored by its
`conftest.py`: every test is timed, the peak RSS is read after each, and a
summary table is printed at the end of the session (also written to
`test/experimental/.resource_report.txt`). The budget is one test < 3 min,
the whole run < 10 min, peak RSS < 3 GB; shrink a test's grid or cell rather
than the limits. `test_ncpp_family.py` pins the TM energies of H₂ (0.74 Å,
h = 0.25 Å) and LiH (1.6 Å, h = 0.30 Å) measured before the nonlocal
generalization and exercises the general form with synthetic projectors;
`test_oncvpsp.py` validates the ONCVPSP family atomically (H, Li, O) and on
the same two molecules (50 tests, ~11 s, peak RSS 0.6 GB); `test_paw.py`
does the same for the PAW family, adding the overlap, on-site-projection,
compensation and grid-stability checks (70 tests, 11.5 s, peak RSS 0.77 GB;
the whole experimental suite: 338 tests, 48 s, peak RSS 2.2 GB).

See `examples/19_pseudopotential_generation.py` and
`examples/20_pseudopotential_calculations.py`.
