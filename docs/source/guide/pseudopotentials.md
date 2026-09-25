# Pseudopotentials

A pseudopotential replaces an atom's core electrons and the singular $-Z/r$
potential by a smooth, valence-only problem. In Mandacaru a pseudopotential
**family is a basis name**, selected exactly like `"HAO"` or `"cc-pVTZ"`:

```python
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis="NCPP",        # norm-conserving Troullier-Martins
                       h=0.15)
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis="ONCVPSP",     # Hamann's optimized norm-conserving Vanderbilt
                       h=0.25)
atoms.calc = Mandacaru(method="vqe",
                       basis={"name": "PAW-LCAO", "size": "DZP"},   # Bloechl's PAW-LCAO, polarized double zeta
                       h=0.25)
```

That basis turns an all-electron calculation into a valence-only one: the core
electrons are removed (oxygen keeps 6 of its 8), the basis becomes the family's
smooth pseudo-atomic orbitals — with the same `size` hierarchy as the
[NAO family](basis_sets.md) as its options — and the singular $-Z/r$ external
potential is replaced by a bounded local channel plus the family's projectors.
Every driver, `interaction_energy`, the periodic methods, the dry run and the
command line (`mandacaru H2O --cell 8 --basis PAW-LCAO --basis-option size=DZP`)
accept the names; `frozen_core` is refused with them as redundant. There is no
separate switch: the family is the basis, and the retired `"PP"` basis name
raises an error that names the families instead of aliasing to one.

## Families

Four families are shipped, all generated from scratch by Mandacaru's own LDA
radial atomic solver (`mandacaru.basis.atomic_solver`):

| Basis name | Aliases | Family | Projectors | Overlap | Library |
|---|---|---|---|---|---|
| `"NCPP"` | `"TM"`, `"NCPP-TM"` | Troullier–Martins norm-conserving, Kleinman–Bylander separable form | one per channel | none | `mandacaru-ncpp`, H–U |
| `"ONCVPSP"` | `"ONCV"` | Hamann's optimized norm-conserving Vanderbilt (below) | two per channel, $2\times2$ coupling | none | `mandacaru-oncvpsp`, H–U |
| `"PAW-LCAO"` | — | Blöchl's projector augmented wave (below) | two per channel, $2\times2$ coupling | $S + C\,q\,C^\dagger$ | `mandacaru-paw`, H–U |
| `"UPAW-LCAO"` | `"unitary-paw-lcao"` | the same, with a **unitary** transformation ($q = 0$, below) | two per channel, $2\times2$ coupling | $S$ (unaugmented) | generated on demand |

Names are case-insensitive. Each family accepts the options `size`,
`tail_norm` or `split_norm` (the split-valence scheme: GPAW's by default, or
the SIESTA-style one), `directory` (an alternative library folder) and `filter`
(*Fourier filtering*, below); PAW-LCAO and UPAW-LCAO also take
`projector_basis="raw"|"dual"`, `energy_shift`, `confinement` and
`polarization` (*Confined orbitals*, below). Any other key — or an all-electron option such
as `tier` — is refused before an integral is computed. A family may give an
option a **default of its own**: PAW-LCAO and UPAW-LCAO declare `filter=True`, the
norm-conserving families leave it off.

`"PAW-LCAO"` is the recommended pseudopotential family. `"UPAW-LCAO"` is an option — the
same construction with a unitary transformation — and the measurements that
decided that are in its own section below.

A **per-element basis** may give each atom its own size, as long as every
element uses the *same* family:

```python
basis={"O": {"name": "PAW-LCAO", "size": "DZP"}, "H": {"name": "PAW-LCAO"}}
basis={"O": {"name": "NCPP", "size": "DZP"}, "*": "NCPP"}     # "*" = every other element
```

Mixing a pseudopotential family with an all-electron family across elements
(`{"O": "PAW-LCAO", "H": "6-31G"}`), or two pseudopotential families, raises: a
pseudopotential replaces the core *and* the potential of its atom, so the
Hamiltonian is either valence-only or all-electron.

### The registry

The driver only ever talks to a family through its registry entry in
`mandacaru.pseudopotentials.families`:

```python
from mandacaru.pseudopotentials import (
    PSEUDO_FAMILIES, FamilySpec, family_names, lookup_family,
    register_family, resolve_family)

family_names()      # ['ncpp', 'oncvpsp', 'paw-lcao', 'upaw-lcao', 'ncpp-tm', 'oncv', 'tm', 'unitary-paw-lcao']
spec = resolve_family("TM")            # -> PSEUDO_FAMILIES["ncpp"]
spec.name, spec.aliases, spec.label    # "ncpp", ("tm", "ncpp-tm"), "NCPP"
spec.options       # ("size", "split_norm", "tail_norm", "directory", "filter")
spec.default_options       # {} -- PAW-LCAO/UPAW-LCAO declare {"filter": True, "energy_shift": 0.1}
spec.resolved_options({"size": "DZ"})  # the defaults with the user's options on top
spec.norm_conserving                   # True
spec.generate("O")                     # generate_pseudopotential("O")
spec.get("O")                          # the (cached) library loader
spec.build(atoms, grid, h, charge, spin, options, kinetic)
lookup_family("HAO")                   # None -- an all-electron basis name
```

`build` returns exactly the 5-tuple the all-electron path returns —
`(hamiltonian, num_particles, n_spatial_orbitals, integration_profile,
context)` — and `_pseudopotential_hamiltonian` in the driver is a thin
dispatcher on it. The family name is also **stored in every pseudopotential
file** (`family` field, format version 2; files written before the field
existed, and files spelled `"tm"`, load as `"ncpp"`).

A new family is added by registering a `FamilySpec`:

```python
register_family(FamilySpec(name="gth", description="...",
                           generate=..., get=..., build=...,
                           norm_conserving=True, aliases=("goedecker",),
                           options=("size", "split_norm", "tail_norm",
                                    "directory", "filter"),
                           default_options={"filter": True}))
```

and its name immediately works as a basis name — `basis="GTH"` — on every
driver, in the dry run and on the command line, with no change to any of them.

## ONCVPSP: optimized norm-conserving Vanderbilt potentials

*(2026-09-14, step 2 of the family plan.)* The second shipped family is
`"oncvpsp"` (alias `"oncv"`), D. R. Hamann's construction, *Phys. Rev. B*
**88**, 085117 (2013), written from scratch in
`mandacaru.pseudopotentials.oncv` on top of the same LDA radial
atom as the TM family:

```python
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis="ONCVPSP",                        # or "ONCV"
                       h=0.25)
atoms.calc = Mandacaru(method="vqe",
                       basis={"name": "ONCVPSP", "size": "DZP"},
                       h=0.25)
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

### Validation (pinned by `test/pseudopotentials/test_oncv.py`)

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

`$MANDACARU_ONCVPSP_PATH/lda/{H,Li,C,N,O,F}.parquet` (70–220 kB each, decimated
to 0.02 Bohr like the TM library), regenerated with
`build_oncv_library(["H", "Li", "C", "N", "O", "F"])`; `get_oncv(symbol,
directory)` is the family's loader (`directory` defaults to
`$MANDACARU_ONCVPSP_PATH/<xc>`, so `get_oncv(symbol, directory)` finds them and
the TM files, in their own repository, are untouched). The record is `ONCVPseudoPotential` (a `PseudoPotential`
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

### The reference atom and the channel set

Five arguments of `generate_oncv` change what the pseudopotential is built
*from*. They are generation-time only: they change the dataset, never the
calculation that later reads it.

| Argument | Default | What it does |
| --- | --- | --- |
| `relativity` | `"scalar"` | `"none"`, `"scalar"` (Koelling–Harmon) or `"dirac"` (each $j$ separately) |
| `xc` | `"lda"` | `"lda"` or `"pbe"`; screens the atom and unscreens the potential with the same functional |
| `nlcc` | `True` | Partial core density; `True` matches where $\rho_c = \rho_v$, a float sets the radius |
| `extra_l` | `0` | Channels above the highest valence $l$, two scattering references each |
| `points`, `r_max` | per element | The radial grid of the reference atom |

**`relativity="none"` with `nlcc=False` reproduces the pre-relativistic
construction bit for bit.** Both new defaults change every generated dataset.

#### Relativity

The Dirac radial pair collapses *exactly* onto one equation for the large
component, and the only place $j$ survives is a single $\kappa M'P/Mr$ term —
so one operator covers all three theories
(`mandacaru.basis.relativity`). The $(2j+1)$-weighted average of $\kappa$ is
$-1$ for every $l$, which is why the scalar-relativistic equation has no $l$
dependence in its spin-orbit term. For $l = 0$ there is one $j$ and
$\kappa = -1$ *is* its physical value, so the scalar equation **is** the Dirac
equation there.

With `relativity="dirac"` each $j$ gets its own full construction, and the two
are stored as their $(2j+1)$ average plus the difference
$\tfrac{2}{2l+1}(V_{l+1/2} - V_{l-1/2})$, which is the coefficient of
$\mathbf{L}\cdot\mathbf{S}$. Adding $\mathbf{L}\cdot\mathbf{S}$ back
reproduces each $j$ exactly — it is a $2\times2$ solve, not a fit. The average
sits in `channels` as usual, so a Dirac dataset is a drop-in replacement for a
scalar one that additionally *carries* spin-orbit coupling in `spin_orbit`.

```{note}
The pseudo partial waves stay non-relativistic — they are Bessel expansions,
meant for a Schrödinger calculation — so **the generalized norm condition
changes**. What has to be conserved is
$-W_{ij}(r_c)/2(\varepsilon_j-\varepsilon_i)$, a Wronskian, which equals the
inner overlap only as $M \to 1$. Conserving the overlap instead leaves the
Vanderbilt matrix asymmetric by $1.4\times10^{-4}$ Ha for oxygen against a
$10^{-5}$ tolerance, and refining the grid does not help. With the Wronskian
the logarithmic derivative of the *smooth* system reproduces the
*relativistic* all-electron one to $4\times10^{-7}$ at the reference energy.
```

Spin-orbit splittings of the reference atom, against experiment:

| Atom | Level | Computed | Measured |
| --- | --- | --- | --- |
| Ar | 3p | 0.1786 eV | 0.178 eV |
| Kr | 4p | 0.6482 eV | 0.666 eV |

```{note}
**Be honest about what the scalar-relativistic default buys for an $s$
channel.** A point-nucleus Dirac $s$ or $p_{1/2}$ state behaves like
$r^\gamma$ with $\gamma = \sqrt{\kappa^2 - (Z\alpha)^2} < 1$, and a uniform
radial grid cannot converge that cusp at second order -- the rate never
approaches 2, unlike every other state. On the *real* SCF potential of
oxygen and argon this stays harmless for the splittings above: every $p$
level (both $j$) converges at the clean rate 2.00, because the centrifugal
barrier regularizes $p_{1/2}$ despite it sharing $|\kappa| = 1$ with an $s$
state. It is not harmless for the **absolute size of an $s$-channel
relativistic shift**, which drifts with grid density instead of settling:

| points | O 2s shift (Ha) | O 2p shift (Ha) |
| --- | --- | --- |
| 3000 | −0.005298 | +0.000919 |
| 12000 (production for O) | −0.005047 | +0.000871 |
| 24000 | −0.004408 | +0.000754 |

A shift is a small difference of two large, separately-converging numbers, so
its relative error is amplified far past the underlying grid error -- about
400× for O 2p. **Treat a valence relativistic shift as good to about
10-15 % at production grid densities, not better.** That is enough for the
shift to be worth including by default (5 mHa on O 2s, and 85 % of a
correction beats none), and not enough to quote it past the first
significant digit. Spin-orbit splittings themselves are not affected by this:
$l = 0$ carries none, and every channel that does carry one is $p$ or higher,
converging at the clean rate.
```

#### Nonlinear core correction

Unscreening subtracts $V_{xc}[\tilde\rho_v]$, but the all-electron potential
was screened by $V_{xc}[\rho_c + \rho_v]$, and $V_{xc}$ is not linear. The
correction keeps a partial core density — the true core outside
$r_{\text{nlcc}}$, and $A\sin(Br)/r$ inside it, matched in value and slope —
and unscreens with $V_{xc}[\tilde\rho_c + \tilde\rho_v]$.

Mandacaru's many-body Hamiltonian is a wavefunction method with no
exchange-correlation functional at run time, so the *run-time* half of the
correction has nothing to act on and is not performed. The generation-time
half, which is where the nonlinearity is committed, is. `core_density` is
stored and reported so a DFT consumer of the dataset can do the rest.

### Still not implemented

The second reference energy is a fixed offset rather than Hamann's
per-element tuned values. Departures from the paper: the Bessel wave vectors
are the interleaved zeros rather than Hamann's own choice; the scattering tail
entering the residual energy is tapered between $3r_c$ and $5r_c$ (it does not
decay); $r_{cl}$ is a fixed fraction of the smallest $r_c$; and the raw
Vanderbilt coupling is kept instead of the orthogonalized projectors
(equivalent operator).

A point-nucleus Dirac s or p$_{1/2}$ state behaves like $r^\gamma$ with
$\gamma = \sqrt{\kappa^2 - (Z\alpha)^2} < 1$, and **a uniform radial grid
cannot converge that at second order**: gold's 1s is 1.6 % off and improves
only as $h^{1.5}$, against the clean rate 2.00 of its 2p$_{3/2}$ ($\gamma =
1.92$). It affects deep core levels, not the valence channels or the
splittings a pseudopotential is built from.

## PAW-LCAO: projector augmented waves

*(2026-09-14, step 3 of the family plan.)* The third shipped family is
`"paw-lcao"`, P. E. Blöchl's projector augmented-wave method, *Phys. Rev. B* **50**,
17953 (1994), in its **frozen-core, one-center-expansion** form with the
one-center energies **linearized around the reference atom** — a fixed
per-species coupling matrix $D^0$, which makes the dataset behave like an
ultrasoft pseudopotential with an exact PAW-LCAO reconstruction of the atomic
partial waves. Written from scratch in
`mandacaru.pseudopotentials.paw` on the same LDA radial atom as
the other two families, reusing the Numerov partial waves, the Bessel
machinery and the polynomial local potential of the ONCVPSP module:

```python
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis="PAW-LCAO",
                       h=0.25)
atoms.calc = Mandacaru(method="vqe",
                       basis={"name": "PAW-LCAO", "size": "DZ", "projector_basis": "raw"},
                       h=0.25)
```

The name has no alias; `family_names()` lists `ncpp`, `oncvpsp`, `paw` first
(and the unknown-family error names all three).

### The transformation

Blöchl's linear map between the smooth wave functions the grid sees and the
all-electron ones,

```{math}
|\psi\rangle = \mathcal T|\tilde\psi\rangle, \qquad
\mathcal T = 1 + \sum_i \big(|\varphi_i\rangle - |\tilde\varphi_i\rangle\big)\langle\tilde p_i| ,
```

needs per atom and $l$ **all-electron partial waves** $\varphi_i$, **smooth
partial waves** $\tilde\varphi_i$ equal to them beyond the augmentation
radius $r_c$, and **projectors** $\tilde p_i$ inside $r_c$ dual to the smooth
waves, $\langle\tilde p_i|\tilde\varphi_j\rangle = \delta_{ij}$. The smooth
waves are *not* norm-conserving, so the transformation carries the overlap
operator $S = 1 + \sum_{ij}|\tilde p_i\rangle q_{ij}\langle\tilde p_j|$ with
$q_{ij} = \langle\varphi_i|\varphi_j\rangle_{r<r_c} -
\langle\tilde\varphi_i|\tilde\varphi_j\rangle_{r<r_c}$, and the valence
problem is the generalized eigenproblem

```{math}
\Big[T + \tilde v_{loc} + \sum_{ij}|\tilde p_i\rangle D_{ij}\langle\tilde p_j|\Big]\tilde\psi
= \varepsilon\,S\,\tilde\psi .
```

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
   suppresses but the PAW-LCAO projector term does not; the raise is chosen so the
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
   0.64); the reference atom is spherical, so its compensation charge is the
   **monopole** $\hat n = \hat Q\,g(r)$, $g
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
   in `hartree_fock_hamiltonian`), so PAW-LCAO totals are comparable with TM/ONCV.

### In a molecule (`build_paw`, `PAWIntegrals`)

The basis is the bound smooth partial waves (first zeta; the `size`
hierarchy applies unchanged), the external potential $\sum_A\tilde v^{ion}_A$
through `Potentials.pseudopotential`, the nonlocal term $C D^{ion} C^\dagger$
and the overlap $\tilde S + C q C^\dagger$, and the constant the ion-ion
repulsion plus $\sum_A E^A_{1c}$. `PAWIntegrals` is a `MolecularIntegrals`
subclass with two additions:

* **Augmented two-body tensor.** Pair densities carry their compensation
  charge, $\rho_{pr} = \tilde\phi_p^*\tilde\phi_r + \sum_A Q^A_{pr}\,g_A$ with
  $Q^A = (C q C^\dagger)^A$ the augmentation moments, one per multipole channel
  $(L, M)$ (the $L = 0$ one is the very block that augments the overlap), so through the new hook
  `MolecularIntegrals.two_body_augmentation()`
  $\langle pq|rs\rangle = \langle pq|rs\rangle_{grid} + \sum_A(Q^A_{pr}W^A_{qs}
  + W^A_{pr}Q^A_{qs}) + \sum_{AB}Q^A_{pr}U_{AB}Q^B_{qs}$, with
  $W^A_{qs} = \int\tilde\phi_q^*\tilde\phi_s V_{g_A}$ on the grid and
  $U_{AB}$ the compensation–compensation Coulomb integrals
  (`compensation_coulomb`: exactly $1/R$ for disjoint spheres, a
  Gauss–Legendre radial/angular quadrature when they overlap, as they do in
  H₂ and LiH). Hartree, exchange and correlation all see neutral atoms.
* **Atom-centered projections.** `PAWIntegrals.projections()` evaluates every
  $C_{\mu p} = \langle\tilde\phi_\mu|\tilde p_p\rangle$ by a spherical product
  quadrature over the projector's sphere (`atom_centered_projections`:
  Gauss–Legendre in $r$ and $\cos\theta$, uniform in $\phi$), not on the grid.
  The dual projectors are sharp (the two reference waves are nearly parallel in
  the core, $B^{-1}$ large), and the grid values depended on where the nucleus
  sat between nodes: the on-site $\langle\tilde\varphi_1|\tilde p_i\rangle$ —
  exactly $\delta_{i1}$ — came out 0.7–3× that, and rigidly translating H₂ by
  one grid step moved the nonlocal energy by 3.9 eV at h = 0.25 Å. The
  quadrature depends only on the separation of function and projector, so it
  is exactly translation invariant; the grid ripple left (kinetic, local and
  Hartree terms) is 18 / 13 / 5 meV at h = 0.25 / 0.20 / 0.15 Å.
* **Range-separated local potential.** The same idea applied to
  $V_{pq} = \sum_A\int\tilde\phi^*_p\tilde\phi_q\,v_A$ — see below.

(paw-local-split)=
#### The local potential leaves the grid, in half

A projector vanishes outside its cutoff, which is what lets its integral live
on a sphere. The local channel does not: it decays as $-Z^{ion}_A/r$. So it is
split (`pseudopotentials/local_split.py`) into the potential of a **Gaussian
ion** and the remainder,

```{math}
v^{lr}_A(r) = -Z^{ion}_A\,\frac{\mathrm{erf}(r/\sqrt2\,\sigma)}{r},
\qquad v^{sr}_A = v_A - v^{lr}_A ,
```

with $\sigma = 1.4\,h$ (`SIGMA_FACTOR`), which puts
$e^{-(1.4\pi)^2/2}\approx 6\times10^{-5}$ of the Gaussian's weight past the
grid's Nyquist wave-vector $\pi/h$. $v^{lr}$ carries the whole tail and stays
on the grid — it is simply what `PAWIntegrals.external_potential()` hands the
engine. $v^{sr}$ vanishes beyond a few $\sigma$ (past the dataset's local
cutoff it is $-Z\,\mathrm{erfc}(r/\sqrt2\sigma)/r$, and
$\mathrm{erfc}(6.5/\sqrt2) = 7\times10^{-11}$), so its matrix is integrated on
an atom-centered sphere of radius $6.5\,\sigma$ and arrives through the new hook
`MolecularIntegrals.short_range_local()`, which returns `None` for every other
basis. `PAWIntegrals.exact_local_potential = False` puts the whole potential
back on the grid.

The split is exact by construction — $v^{sr}$ is evaluated as the *difference*
of the dataset's own `local_potential` and $v^{lr}$, never from the asymptotic
form — so only the quadrature order and the truncation approximate anything:
$3\times10^{-7}$ Ha on the hardest case measured (water PAW-LCAO-DZ, the oxygen
sphere holding both hydrogens) and better than $10^{-9}$ Ha on H₂.

**What it buys, measured.** Rigidly translating water (PAW-LCAO-SZ, 10 Å cell) by
fractions of $h$ along $(1,1,1)/\sqrt3$ with the orbitals and density frozen,
peak-to-peak per term in meV:

| $h$ (Å) | $S$ | $T$ | $V_{loc}$ grid → split | ERI | total grid → split |
|---|---|---|---|---|---|
| 0.30 | 288 | 1471 | 1102 → **2714** | 2390 | 3059 → **1470** |
| 0.25 | 104 | 50 | 634 → **683** | 533 | 61 → **57** |
| 0.20 | 8 | 86 | 55 → **14** | 26 | 105 → **44** |
| 0.16 | 3 | 23 | 24 → **23** | 19 | 26 → **24** |

Read this honestly. The nonlocal and compensation terms are already exactly
0.000 meV — those are the projector quadratures above. $V_{loc}$ is **not** the
dominant remaining term: the finite-difference kinetic stencil and the grid
Coulomb tensor are of the same size and larger at some spacings, and the total
is set by how the three happen to cancel. The short-range half that moves to
the sphere is exactly translation invariant ($3\times10^{-16}$), but the
long-range half stays on the grid and keeps an egg-box of its own — one that is
*larger* than the full potential's at $h \ge 0.25$ Å, because the Gaussian ion
is a narrower well than the pseudized channel it replaces ($-7.2$ Ha at the
origin against $-5.4$ Ha for oxygen at $h = 0.25$ Å). Widening $\sigma$ reduces
that term monotonically (oxygen at $h = 0.30$ Å: 3425 / 2714 / 2097 / 1501 /
1159 meV at $\sigma/h = 1/1.4/2/3/4$) but never to zero, because what remains is
the sampling of the *pair density*, not of the potential — and it degrades the
total by breaking the cancellation against the ERI term. The total egg-box is
1.0–2.1× smaller at $\sigma = 1.4h$ and never larger in the four spacings
measured; the energy itself moves by $\le 4\times10^{-5}$ Ha on the pinned H₂
and LiH cases and the convergence with $h$ is unchanged (split − grid is
$\le 1$ meV for $h \le 0.16$ Å on both H₂ and water). Treat it as what it is: one
term integrated exactly instead of approximately, not a cure for the egg-box.

The projector functions can be either the dual $\tilde p_i$
(`projector_basis="dual"`) or the smooth raw $\chi_k$ with the transformed
blocks $B^{-1}DB^{-T}$, $B^{-1}qB^{-T}$ (`"raw"`, the default,
`PAWDataset.projector_set`). The $\chi_k$ are exact linear combinations of the
$\tilde p_i$, so the energies are **identical to all digits** (tested); only
the reported grid resolution ratio differs (H at h = 0.25 Å: raw 0.90/0.78,
dual 0.78/0.77). The local potential is interpolated with a cubic spline
(`PAWDataset.local_potential`), so its derivatives are continuous.

### Forces (`algorithms/pseudo_forces.py`)

`atoms.get_forces()` with `basis="PAW-LCAO"` (or `"ONCVPSP"`) returns the
Hellmann–Feynman plus Pulay force of the converged state. With the reduced
density matrices $D$, $\Gamma$ and the molecular orbitals $V$ held fixed,

```{math}
E = \sum_{pq} D_{pq}\,h^{MO}_{pq} + \tfrac12\sum_{pqrs}\Gamma_{pqrs}\,g^{MO}_{pqrs}
  + E_{ion} + E_{1c}, \qquad h^{MO} = A^\dagger h A,\ A = S^{-1/2}V,
```

and every atomic-orbital matrix is differentiated. **Hellmann–Feynman:** the
atom's operators move — its local potential, its projectors (in
$C D C^\dagger$ and in $S = \tilde S + C q C^\dagger$), its compensation charge
($W$, $U$) and the ion–ion repulsion. **Pulay:** the basis functions centered
on the atom move — $\tilde S$, $T$, $V_{loc}$, $C$, the grid two-electron
tensor and $W$. The projection derivatives use the same atom-centered
quadrature ($+G$ for the basis function, $-G$ for the projector), so the two
parts cancel exactly under a rigid translation.

The [range-separated local potential](#paw-local-split) is differentiated the
same way. Writing $I^A_{pq} = \int\phi^*_p\phi_q\,v^{sr}_A$ and
$G^A_{pqk} = \int(\partial\phi_p/\partial R_{p,k})^*\phi_q\,v^{sr}_A$ (that
function's *own* center moving), the Pulay part for atom $B$ is
$\sum_A\big([p\in B]\,G^A_{pqk} + [q\in B]\,\overline{G^A_{qpk}}\big)$ and the
Hellmann–Feynman part for atom $A$ is
$-\big(G^A_{pqk} + \overline{G^A_{qpk}}\big)$ over its own sphere; summed over
a rigid translation of everything they cancel identically. The grid half of
the derivative differentiates the grid half of the operator —
`_atom_potentials` asks the integrals for
`local_potential_functions()`, which returns $v^{lr}$ when the split is on.
The legacy `force_method="scf-response"` rebuilds $T + V_{grid} + CDC^\dagger$
itself and knows nothing about the spheres, so it refuses a range-separated
potential outright rather than differentiating an energy nobody evaluated.

Holding $V$ fixed is exact for a state that is stationary under orbital
rotations, which a converged ADAPT-VQE run over all orbitals is;
`force_result.details["orbital_gradient"]` reports the residual. Because the
DZP basis has complex $Y_{lm}$ functions, the molecular orbitals are first made
**conjugation-real** (`core.hamiltonian.conjugation_real_orbitals`, same
determinant): with the phases the SCF happens to return, the MO integrals are
complex and the real excitation operators of every pool stall above the ground
state (62 meV on H₂ PAW-LCAO-DZP). H₂ and LiH in PAW-LCAO-DZP are 20-qubit problems,
solved exactly in their (1, 1) particle-number sector (`core.sector`, 100
states).

On H₂ (h = 0.25 Å) the analytic force agrees with a central difference of the
energy to 1e-4 eV/Å. Against VASP (PBE, plane waves, PAW-LCAO) the force curves
agree qualitatively: the H₂ minimum is near 0.81 Å instead of 0.750 Å — mostly
from the H augmentation radius, 1.30 bohr, which two atoms 0.75 Å apart overlap
almost entirely — while the LiH bond forces from 2.1 to 3.2 Å match within
0.15 eV/Å. Pinned by `test/algorithms/test_paw_forces.py` and `test/core/test_sector.py`.

### Validation (pinned by `test/pseudopotentials/test_paw.py`, 70 tests, 11.5 s, peak RSS 0.77 GB)

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

| | PAW-LCAO RHF | PAW-LCAO FCI = ADAPT | ONCV RHF / ADAPT | TM RHF / ADAPT | PAW-LCAO − ONCV | PAW-LCAO − TM |
|---|---|---|---|---|---|---|
| H₂ | −28.662 eV (−1.053292) | −29.046 eV (−1.067402) | −1.044179 / −1.058561 | −1.061096 / −1.075333 | −0.248 eV | +0.212 eV |
| LiH | −20.693 eV (−0.760451) | −20.924 eV (−0.768954) | −0.774343 / −0.782341 | −0.729555 / −0.740839 | +0.378 eV | −0.841 eV |

PAW-LCAO lands between the two norm-conserving families on both molecules (all
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

`$MANDACARU_PAW_PATH/lda/{H,Li,C,N,O,F}.parquet` (100–385 kB, decimated to
0.02 Bohr; 9.6 s to regenerate with `build_paw_library()`; `get_paw(symbol,
directory)` is the family's loader, `paw_library_path()` its directory
(`$MANDACARU_PAW_PATH/<xc>` by default), TM and ONCVPSP files, in their own
repositories, untouched). The record is `PAWDataset` (a `PseudoPotential` subclass:
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
"paw-lcao"` (format version 2; every radial table under `radial_tables` —
`ae_wave_l{l}_{i}`, `pseudo_wave_l{l}_{i}`, `projector_l{l}_{i}`,
`raw_projector_l{l}_{i}`, the densities — and the scalars in the metadata).
`io.py` now dispatches the table families (`TABLE_FAMILIES`) by *record type*
on save and by the presence of `radial_tables` on load, so a plain TM record
merely carrying the name `"paw-lcao"` keeps the TM layout and the TM loader refuses
it as before. Round trips are lossless and idempotent (tested).

### Relativity, GGA and spin-orbit coupling

`generate_paw` takes the same `xc`, `relativity`, `nlcc` and `extra_l`
arguments as `generate_oncv`, with the same defaults (`"lda"`, `"scalar"`,
`True`, `0`), and `relativity="none", nlcc=False` reproduces the
pre-relativistic dataset.

```{note}
The same honest limit applies here as for ONCVPSP: a relativistic $s$-channel
eigenvalue does not converge cleanly on this grid (the $r^\gamma$ cusp of a
point-nucleus Dirac state), so treat a valence relativistic shift as good to
about 10-15 % at production grid densities, not better -- see the
*Relativity* subsection under ONCVPSP, above, for the measured table. It does
not touch the spin-orbit splittings below, which come entirely from $p$ and
higher channels.
```

Two things differ from ONCVPSP, and both follow from PAW-LCAO's overlap operator.

**A Dirac PAW-LCAO dataset is scalar-relativistic partial waves plus a spin-orbit
term, not a $j$-resolved augmentation sphere.** $S = 1 + \sum|\tilde p\rangle q
\langle\tilde p|$ is the *metric* of the generalized eigenproblem, so a
$j$-dependent $q$ would give that metric an $\mathbf{L}\cdot\mathbf{S}$
structure and every consumer of $S$ — the Löwdin orthogonalization above all —
would have to learn about spin. Keeping one partial-wave set per $l$ leaves
$q$, $\Delta T$ and the compensation charges untouched, and carries spin-orbit
coupling where it belongs: as a one-center difference in the *Hamiltonian*,

$$D^{SO}_{ij} = \int_0^{r_c}\Big[\xi(r)\varphi_i\varphi_j
  - \tilde\xi(r)\tilde\varphi_i\tilde\varphi_j\Big]r^2\,dr,
\qquad \xi = \frac{1}{2c^2M^2 r}\frac{dV}{dr},$$

built exactly the way $D$ and $q$ are. For oxygen $\langle\xi\rangle(2l+1)/2$
reproduces the Dirac atom's own 2p splitting — 0.03675 eV against 0.03674 —
and 99.7 % of it sits inside $r_c$, which is why a one-center term captures
it. The smooth part is 0.5 % and is subtracted rather than assumed away.

**The overlap correction is built from the conserved norm, not the plain
overlap.** Relativistically they differ at $O(c^{-2})$, and the choice is not
free: the Wronskian at $r_c$ is a property of the all-electron equation, so
$B - B^\mathsf{T} = (\varepsilon_j - \varepsilon_i)(\text{achieved} -
\text{target})$ whatever constraint was imposed, and only
$q = \text{target} - \text{achieved}$ cancels it. The same quantity has to
appear in $\Delta T$, through
$\langle\varphi_i|T|\varphi_j\rangle = \varepsilon_j\,\text{target}_{ij}
- \langle\varphi_i|V|\varphi_j\rangle$ — which *is* the relativistic kinetic
matrix element, because the target is the $M$-weighted norm the relativistic
radial equation puts on the right-hand side. With both in place
$D^{scr} = \Delta T + \Delta V$ closes to $2\times10^{-15}$ and $D^{scr}$ is
symmetric to $10^{-9}$; getting either wrong breaks one of the two.

```{note}
UPAW-LCAO sets $q \equiv 0$, so it has no overlap correction left to absorb the
residual, and its hydrogen eigenvalue reproduction degrades from
$5.5\times10^{-8}$ to $1.3\times10^{-6}$ Ha (3.5e-5 eV) with the
scalar-relativistic default. That is the honest price of a smooth,
non-relativistic system reproducing a relativistic reference.
```

An `extra_l` channel has no bound state, so its references are normalized
inside $r_c$ by convention rather than by physics. Asking it to *also* give up
a fraction of that norm over-constrains the Bessel expansion and leaves the
minimization with no feasible point, so such channels conserve norm exactly —
which makes their $q$ identically zero. They hold no charge, so there is
nothing to correct.

### What is frozen or omitted relative to Blöchl's full method

* **Linearized one-center terms.** The one-center Hartree and xc energies are
  expanded to first order in the density matrix $\rho_{ij}$ around the LDA
  reference atom: a fixed $D^0 = D^{ion}$ per species plus the constant
  $E_{1c}$. The full method recomputes $D_{ij}[\rho_{ij}]$ self-consistently
  from $n^1 - \tilde n^1$ every SCF step; here the error is second order in
  $\rho - \rho^{ref}$.
* **Compensation multipoles up to $L = 2l_{max}$.** Each pair density carries
  the multipoles $\hat n^{LM}$ of its augmentation charge (`multipoles.py`; an
  s-valence atom has only $L = 0$, oxygen has $L = 0, 1, 2$), and the
  compensation charge is attracted to the other ions as well as repelled by
  the electrons. What is still omitted is the *one-center* two-body
  correction beyond the linearization -- measured at 0.08-0.33 eV for oxygen.
* **Frozen core.** The core density is frozen (stored as `core_density`).
  Since 2026-09-23 the unscreening *does* include the smooth core when
  `nlcc=True` (the default), which is the nonlinear core correction: PAW-LCAO
  already builds `smooth_core_density` for its one-center energies, so the
  correction here is a matter of including that density in $v_{xc}$ rather
  than constructing a second one. What remains frozen is the linearized
  treatment of the one-center core–valence xc, reported in
  `energies["core_valence_xc_omitted"]` (Li −1.54, O −4.60 Ha).
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

## UPAW-LCAO: a unitary transformation

`basis="UPAW-LCAO"` (alias `"unitary-paw-lcao"`) is the PAW construction with the
orthonormality constraint of Ivanov *et al.*,
[arXiv:2408.03159](https://arxiv.org/abs/2408.03159), imposed on the smooth
partial waves:

```{math}
O_{ij} \;=\; \langle\phi_i|\phi_j\rangle_{r_c}
           - \langle\tilde\phi_i|\tilde\phi_j\rangle_{r_c} \;=\; 0 ,
```

which is exactly Mandacaru's `norm_deficit = 0`. The transformation
$\mathcal{T} = 1 + \sum_i (|\phi_i\rangle - |\tilde\phi_i\rangle)\langle\tilde
p_i|$ is then **unitary**: the smooth states are orthonormal, the overlap
operator is the identity, and the whole augmentation of the metric disappears —

```{math}
S = \tilde S + C\,q\,C^\dagger \;\longrightarrow\; \tilde S , \qquad q = 0 .
```

Everything else is the ordinary PAW-LCAO dataset: the same two reference energies,
the same residual-kinetic-energy minimization of the smooth waves, the same
dual projectors, the same local potential and one-center linearization. Only
the norm constraint changes, so `generate_upaw(symbol, **options)` is
`generate_paw(symbol, norm_deficit=0.0, **options)` with the record tagged
`family="upaw-lcao"`, and it *refuses* an explicit `norm_deficit` — that number is
the definition of the family.

### Why it is attractive on a quantum computer

The motivation is not accuracy but the structure of the second-quantized
problem. With $q = 0$ the one-particle basis is orthonormal *before* the
Löwdin step, the overlap operator never enters the Hamiltonian, and the
generalized eigenproblem $Hc = \varepsilon S c$ becomes an ordinary one. In a
plane-wave PAW-LCAO code that removes a nontrivial metric from every algorithm built
on top; in Mandacaru it removes one matrix product, because $S^{-1/2}$ is
computed anyway for the grid basis.

### What it costs (measured, `mandacaru` env, h as noted)

| | PAW-LCAO | UPAW-LCAO |
|---|---|---|
| overlap correction $\max|q_{ij}|$, H | $2.12\times10^{-2}$ | $3.0\times10^{-14}$ |
| overlap minimum (H) | $1.055$ | $1.000000$ |
| $\max|S - \tilde S|$, H₂ (h = 0.3 Å) | $2.8\times10^{-2}$ | $1.0\times10^{-13}$ |
| O, $L=0$ compensation moment | $3.2\times10^{-2}$ | $7.7\times10^{-15}$ |
| O, $L=2$ compensation moment | $1.4\times10^{-3}$ | $2.2\times10^{-2}$ |
| H₂O net force, h = 0.25 Å | 0.380 eV/Å | 1.918 eV/Å |
| H₂O net force, h = 0.20 Å | 0.0415 eV/Å | 0.2227 eV/Å |
| LiH $d_{eq}$ / binding | 1.6782 Å / 3.2389 eV | 1.6802 Å / 3.1413 eV |
| H₂ binding at 0.85 Å | 2.640 eV | 2.386 eV |
| Hamiltonian one-norm $\lambda$, H₂ | 27.05 | 26.61 |

Three things to read out of that table.

**The constraint does what it claims, and only for the monopole.** $q$ and the
$L = 0$ augmentation vanish to machine precision. But the constraint is one
number per pair of partial waves, and the higher multipoles are not constrained
by it: on oxygen the $L = 2$ moment comes out **16× larger** than PAW-LCAO's. The
compensation machinery (`compensation_moments`, `compensation_potentials`,
`compensation_coulomb`, and the force derivatives of all three) therefore
cannot be deleted — the reason to want UPAW-LCAO is not realized in this code.

**The smooth waves get harder.** Forcing the inner norm to match the
all-electron one removes the freedom that the scaled-norm construction spends
on smoothness, so the pseudo waves carry more short-wavelength content. On a
uniform real-space grid that shows up immediately as a larger egg-box: the
translational residual on water is **≈ 5× worse at both spacings tested**, and
the residual is what limits how far a relaxation can be converged.

**Binding is slightly worse, and $\lambda$ barely moves.** UPAW-LCAO under-binds
H₂ by 0.25 eV and LiH by 0.10 eV relative to PAW-LCAO at the same grid and size,
while the LCU one-norm — the figure of merit for a qubitized phase estimation,
where the Toffoli count scales as $\lambda/\epsilon$ — improves by 1.6 %. The
quantum-resource argument for the unitary form is real but small here.

So UPAW-LCAO is available, tested, and not the default. It is worth revisiting if
the compensation charge ever grows its full multipole expansion (then $q = 0$
buys a genuinely simpler metric), or on a smooth basis where the egg-box
penalty does not apply — a plane-wave or Gaussian representation rather than a
uniform grid.

### Datasets

No UPAW-LCAO library is shipped, and it has no repository of its own:
`get_upaw(symbol)` looks in `$MANDACARU_PAW_PATH/upaw-lcao/<xc>/` and, finding
nothing there (including when `MANDACARU_PAW_PATH` is unset), **generates the
dataset on the fly**, caches it in memory and warns once — generation is
0.4–2.2 s per element, so an interactive run pays a fraction of a second and a
scan pays nothing after the first geometry. Naming a `directory` explicitly is
a statement that the library is there, and a missing element then raises with
the `build_upaw_library` recipe:

```python
from mandacaru.pseudopotentials.paw import build_upaw_library

build_upaw_library(("H", "C", "N", "O"))          # into $MANDACARU_PAW_PATH/upaw-lcao/lda/
build_upaw_library(("H", "O"), directory="/data/upaw")
```

or from the command line, which writes the same place:
`mandacaru-build --pp UPAW --install`.

The files use the PAW-LCAO layout (`TABLE_FAMILIES` maps both families to the same
codec) and record `family: "upaw-lcao"`, so a PAW-LCAO dataset is refused as UPAW-LCAO and
vice versa.

## The nonlocal term: general separable form

Every family's nonlocal potential enters through one formula,

```{math}
H^{NL} = C\,D\,C^\dagger, \qquad C_{\mu p} = \langle\phi_\mu|\chi_p\rangle ,
```

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
to $10^{-12}$. ONCVPSP and PAW-LCAO fill $2\times2$ blocks; the machinery
(`projector_blocks`, `assemble_block_matrix` in `mandacaru.core.hamiltonian`)
validates and assembles them. `kb_nonlocal()` keeps its name (alias
`nonlocal_matrix()`), and the projector resolution check
(`kb_resolution_ratios`) is unchanged.

### Overlap correction

Families whose projectors also change the metric (PAW-LCAO) pass the blocks of a
second matrix $Q$ in the same layout, `nonlocal_overlap={(atom, l, m): block}`.
The overlap the Löwdin orthonormalization uses then becomes

```{math}
S \;\to\; S + C\,Q\,C^\dagger ,
```

(`MolecularIntegrals.overlap()`; the grid overlap alone is `bare_overlap()`),
so the orthonormalized one- and two-body integrals — and everything downstream,
RHF, the UHF natural orbitals, the qubit Hamiltonian — see the augmented
metric automatically. Norm-conserving families pass `None`; `Q = 0` reproduces
the plain Hamiltonian exactly. The PAW-LCAO family is the first to use it (its $q$
blocks), together with two further hooks on `MolecularIntegrals`:
`two_body_augmentation()` (a correction added to the grid two-body tensor —
the compensation-charge terms) and `constant_energy` (an additive constant
next to the nuclear repulsion — the frozen one-center energies).

## Why they are not optional here

Mandacaru samples everything on a uniform real-space grid, and that grid must
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

## The pseudopotential libraries

None of the three generated families ships inside the package. Each lives in
a repository of its own — `mandacaru-ncpp`, `mandacaru-oncvpsp`,
`mandacaru-paw` — and an environment variable names the checkout Mandacaru
reads from:

| family | variable | set it with |
|---|---|---|
| `ncpp` | `MANDACARU_NCPP_PATH` | `mandacaru --set-ncpp DIR` |
| `oncvpsp` | `MANDACARU_ONCVPSP_PATH` | `mandacaru --set-oncvpsp DIR` |
| `paw-lcao` | `MANDACARU_PAW_PATH` | `mandacaru --set-paw DIR` |

```bash
git clone https://github.com/seixas-research/mandacaru-ncpp.git
mandacaru --set-ncpp mandacaru-ncpp

git clone https://github.com/seixas-research/mandacaru-oncvpsp.git
mandacaru --set-oncvpsp mandacaru-oncvpsp

git clone https://github.com/seixas-research/mandacaru-paw.git
mandacaru --set-paw mandacaru-paw

mandacaru --pseudo-status        # each variable, where it points, and how many datasets it serves
```

`--set-ncpp` / `--set-oncvpsp` / `--set-paw` write `export
MANDACARU_..._PATH=DIR` into `~/.zshrc` or `~/.bashrc` (whichever `$SHELL`
reads), asking `[Y/n]` before replacing a different value; open a new
terminal, or `source` the file, for the variable to take effect in your
shell. Inside a checkout the datasets sit one directory per
exchange-correlation functional — `<checkout>/lda/<Symbol>.parquet` today,
`<checkout>/pbe/` once there are PBE datasets — so one checkout serves every
functional.

```python
from mandacaru.pseudopotentials.io import available_elements, get_pseudopotential

pp = get_pseudopotential("Fe")   # reads $MANDACARU_NCPP_PATH/lda/Fe.parquet
pp.valence_charge     # 8.0  -- 3d^6 4s^2
sorted(pp.channels)   # [0, 2]
```

The NCPP datasets cover **every element with Z ≤ 92** (H through U), generated
from scratch by Mandacaru's own LDA radial atomic solver. The valence includes
semicore $(n-1)d$ and $(n-2)f$ shells, so iron is an eight-electron atom with a
d channel rather than a two-electron 4s² one. Hydrogen and lithium carry a
single valence channel, which is the local one, so in this family H₂ and LiH
have no projectors at all — their nonlocal term is identically zero (the
ONCVPSP family gives them two s projectors each). The ONCVPSP and PAW-LCAO
checkouts (all 92 elements, generated 2026-09-14) are about 110 MB and 190 MB;
NCPP's own datasets are about 11 MB.

A calculation that needs a variable that is unset, or that names something
that is not a directory, raises `LibraryPathError`
(`mandacaru.pseudopotentials.environment`) naming the `--set-*` command that
fixes it; the command line prints it and exits 2. A basis option
`directory=...` bypasses the variable for one run: it names the folder
holding the `<Symbol>.parquet` files itself, with no functional subdirectory,
and the loaders raise `FileNotFoundError` rather than `LibraryPathError` when
that folder is missing or empty.

UPAW-LCAO has no repository — it is generated on demand (*Datasets* above) —
but a library built with `mandacaru-build --pp UPAW --install` goes to
`$MANDACARU_PAW_PATH/upaw-lcao/<xc>/`, inside the PAW-LCAO checkout.

To regenerate or extend a library:

```python
from mandacaru.pseudopotentials.io import build_library

written, failures = build_library()               # all of Z <= 92, into $MANDACARU_NCPP_PATH/lda/
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
is a version-1 file and loads as `"ncpp"`, as does one spelled with the old
name `"tm"`; the NCPP loader refuses a file that declares another family.

```{note}
Saving is lossless and idempotent: `load` then `save` returns the same tables.
The library is decimated once at generation time (`build_library(stride=...)`,
4 by default) because the generation grid must resolve the all-electron core
while the smooth result does not need it. `save_pseudopotential` itself defaults
to `stride=1`, so repeated round trips never compound.
```

## The basis of a pseudopotential

A pseudopotential fixes its own first zeta: the construction pseudizes each
valence orbital inside its cutoff, and the projectors are built from those
specific pseudo-orbitals. Pairing the potential with an unrelated all-electron
radial function would be inconsistent — which is why the family *is* the basis
and an all-electron family cannot be combined with it (per element or
otherwise). What you *can* vary is the size hierarchy, which refines that
pseudo-orbital instead of replacing it — the extra zetas are split-valence
refinements of the pseudized function, and the polarization shell is split
from the outermost channel:

```python
Mandacaru(method="adapt-vqe",
          basis={"name": "NCPP", "size": "DZP"},
          h=0.15)
```

Sizes are the same names as for [the NAO family](basis_sets.md). The default
is `"SZ"` (the minimal valence set), because the valence-only space is usually
already at the edge of what a state-vector simulator can hold. The dry run
counts the valence functions of the family and size you ask for:

```console
$ mandacaru H2O --cell 8 --basis PAW-LCAO --basis-option size=DZP --dry-run
```

## Confined orbitals: `energy_shift`

Without further instruction the first zeta of a PAW-LCAO basis is the dataset's
bound smooth partial wave: the valence orbital of the **free** atom. It has no
range of its own -- a lithium 2s still carries 10⁻⁴ of its norm beyond 14 Bohr
-- and it is more diffuse than the same orbital inside a molecule. LCAO codes
(SIESTA, GPAW) use the orbital of the atom in a confining potential instead,
and name the confinement by what it costs: the **energy shift** is how far the
confined eigenvalue lies above the free one, and it *defines* the cutoff radius
of each orbital. One number gives every channel of every element a radius that
is tight for a compact orbital and generous for a diffuse one.

```python
Mandacaru(method="adapt-vqe",
          basis={"name": "PAW-LCAO", "size": "DZP", "energy_shift": 0.1},
          h=0.20)
```

`energy_shift` is in **eV** (as in GPAW and as for [the NAO
family](basis_sets.md)) and is accepted by `"PAW-LCAO"` and `"UPAW-LCAO"`. **The default
is 0.1 eV**, GPAW's default -- so a plain `basis="PAW-LCAO"` is a confined basis.
`None`, `False` or `0` switch the confinement off and restore the free-atom
orbitals; every PAW-LCAO energy quoted in this guide outside this section was
computed that way (it predates the default, 2026-09-20). A `{symbol: eV}`
mapping, or the per-element basis form, confines elements differently; an
element left out gets the 0.1 eV default.

Nothing about the *operator* changes. The projectors, the couplings `D`, the
overlap `q` and the local potential still come from the dataset, and no new
file is needed: the confined orbital is solved at run time
({mod}`mandacaru.pseudopotentials.confinement`, a second or two per channel,
cached per dataset) as the lowest solution of the same generalized radial
problem the stored wave solves, with a confining potential added,

```{math}
\Bigl(T_l + \tilde v^{scr} + v_{conf}
      + \sum_{ij}|\tilde p_i\rangle D^{scr}_{ij}\langle\tilde p_j|\Bigr)u
= \varepsilon\Bigl(1 + \sum_{ij}|\tilde p_i\rangle q_{ij}
      \langle\tilde p_j|\Bigr)u, \qquad u(r_c) = 0,
```

and the radius `r_c` found by a root search on
`ε(r_c) − ε_free = energy_shift`. The extra zetas and the polarization shell
are then split from the confined orbital, so the whole basis of an atom shares
its range.

### The same recipe as GPAW

The confining potential is the smooth one of Junquera *et al.* (PRB **64**,
235111),

```{math}
v_{conf}(r) = \frac{A}{r_c - r}\exp\Bigl(-\frac{r_c - r_i}{r - r_i}\Bigr)
\quad (r_i < r < r_c),
```

flat at `r_i` and divergent at `r_c`, so the orbital vanishes at its cutoff with
every derivative -- a hard wall leaves a kink there, and a real-space grid pays
for kinks. Mandacaru follows **GPAW's basis generator** (`BasisMaker.generate`
in `gpaw/atom/basis.py`) wherever it has a choice, so that an ADAPT-VQE
calculation here and a GPAW LCAO-DFT calculation can hold the basis recipe
fixed:

| | GPAW (`gpaw-basis`, `mode="lcao"`) | Mandacaru | default here |
| :--- | :--- | :--- | :--- |
| size | `basis="dzp"` | `"size": "DZP"` | `"SZ"` |
| energy shift | `energysplit=0.1` (eV) | `"energy_shift": 0.1` | **GPAW's** (0.1 eV) |
| confining potential | `vconf_args=(12.0, 0.6)` | `"confinement": (12.0, 0.6)` | **GPAW's** |
| split-valence zetas | `tailnorm=(0.16, 0.3, 0.6)` | `"tail_norm": (0.16, 0.3, 0.6)` | **GPAW's** |
| polarization function | quasi-Gaussian | `"polarization": "gaussian"` | **GPAW's** (when confined) |
| PAW-LCAO dataset | GPAW's own setups (LDA or PBE) | Mandacaru's own datasets (LDA) | -- |

Every default is GPAW's, so the basis closest to a GPAW `dzp` basis needs only
its size:

```python
Mandacaru(method="adapt-vqe",
          basis={"name": "PAW-LCAO", "size": "DZP"},
          h=0.20)
```

and Mandacaru's own earlier construction -- free-atom orbitals, the
SIESTA-style split, the `r · R_outer` polarization shell -- is

```python
Mandacaru(method="adapt-vqe",
          basis={"name": "PAW-LCAO",
                 "size": "DZP",
                 "energy_shift": None,
                 "split_norm": 0.15},
          h=0.20)
```

Only the last row cannot be made identical: each code pseudizes its own atom.
That leaves the radii close rather than equal. For the recipe above, GPAW's
generator (`BasisMaker.from_symbol(symbol, xc="LDA").generate(2, 1,
energysplit=0.1)`) and Mandacaru give, in Bohr:

| | GPAW | Mandacaru |
| :--- | ---: | ---: |
| H 1s cutoff / dz split radius | 6.64 / 3.67 | 6.68 / 3.68 |
| O 2s cutoff / dz split radius | 4.38 / 2.30 | 4.37 / 2.37 |
| O 2p cutoff / dz split radius | 5.34 / 2.89 | 5.35 / 2.87 |
| H p-polarization `r_char` | 1.395 | 1.396 |
| O d-polarization `r_char` | 1.125 | 1.128 |

(pinned by `test/pseudopotentials/test_paw_energy_shift.py::TestTheGPAWRecipe`). Compare trends
and differences between the two methods rather than absolute energies, use
`xc="LDA"` setups on the GPAW side, and remember that Mandacaru's Hamiltonian
has the bare Coulomb interaction -- the functional enters its datasets only.

#### Split-valence zetas: GPAW's scheme, or SIESTA's

The two codes share the split-valence polynomial -- inside a split radius the
orbital is replaced by `r^l (a − b r²)` matched in value and slope, and the
difference is the new zeta -- but **not the convention for where to split**, and
the numbers are not comparable digit for digit:

| | GPAW: `tail_norm` (**default**) | SIESTA-style: `split_norm` |
| :--- | :--- | :--- |
| what the number measures | the **norm** of the tail outside the split radius | the **squared norm** of that tail |
| default | `(0.16, 0.3, 0.6)` for the 2nd, 3rd, 4th zeta | `0.15`, halved for each further zeta |
| what each zeta splits | always the **first** zeta | the **previous** zeta |
| H 1s second-zeta radius (0.1 eV) | 3.68 Bohr | 2.52 Bohr |

A tail norm of 0.16 is a squared-norm fraction of 0.0256, so GPAW's second zeta
is considerably **longer-ranged** than one made with `split_norm = 0.15`.
`tail_norm` takes a number (the second zeta's; GPAW's values are kept for the
higher ones) or the whole sequence. Writing `split_norm` selects the
SIESTA-style scheme instead; giving both is refused. The choice applies to
every family with a size hierarchy (`"NCPP"`, `"ONCVPSP"`, `"PAW-LCAO"`, `"UPAW-LCAO"`
and the all-electron `"NAO"`), and the log's `[BASIS]` block names the scheme
in its `zeta_split:` line.

The scheme is part of the model. H₂ in PAW-LCAO-DZP (h = 0.25), with free-atom
orbitals and the `r · R_outer` shell on both sides: GPAW's split gives an
equilibrium distance of **0.760 Å** and the SIESTA-style one **0.711 Å**
(experiment 0.741, VASP-PBE 0.750), while the SIESTA-style basis is about
0.5 eV lower in absolute energy -- its tighter second zeta adds more freedom
near the nucleus. The full default basis (GPAW's split, the 0.1 eV confinement
and the Gaussian polarization shell) lands at **0.733 Å**. Every DZ/DZP/TZP energy quoted in this guide before
2026-09-20 was computed with `split_norm = 0.15`; pass it to reproduce them.

#### The polarization function

The default follows the confinement: `"gaussian"` wherever the orbital is
confined (so, by default, everywhere), and `"orbital"` for an unconfined
element -- the Gaussian takes its cutoff from the confined orbital, so it cannot
exist without one. `"polarization": "orbital"` raises the outermost valence orbital
by one unit of angular momentum, `r · R_outer(r)`, and adds one shell per
polarization count at `l_max + 1`, `l_max + 2`, ... `"gaussian"` is GPAW's
**quasi-Gaussian**,

```{math}
R(r) = r^{l}\Bigl[e^{-r^2/r_{char}^2} - (a - b r^2)\Bigr], \qquad r < r_{cut},
```

with `a`, `b` making the value and the slope vanish at `r_cut`. As in GPAW, the
shell takes the **first angular momentum missing** among the valence channels
(a 4s/3d metal is polarized with a p shell), `r_cut` is the cutoff of the
valence orbital one unit below at the basis's own `energy_shift`, and
`r_char = 0.25 · r_c(0.3 eV)` -- that reference cutoff always at 0.3 eV, whatever
the basis uses. Further polarization functions (`DZ2P`) are split-valence
refinements of the Gaussian with the *same* `l`, not shells of higher `l`. It
needs an `energy_shift` (an unconfined orbital has no cutoff to give it) and is
refused without one. The log's `polarization_shells` table records `l`,
`r_cut` and `r_char` per element.

What the run actually used is in the log's `[BASIS]` block: one row per orbital
with its `r_c`, the free and the confined eigenvalue, and the shift achieved
(see [Reading a run](run_output.md)).

### What it does to the energy

A mild confinement *lowers* the energy of a small basis, because it contracts
orbitals that were too diffuse for a molecule. H₂ (0.74 Å, 8 Å cell, h = 0.25,
ADAPT-VQE to the basis FCI), in eV:

| `energy_shift` (eV) | `r_c` of H 1s (Bohr) | SZ | DZP | DZP, Gaussian polarization | DZP, `split_norm=0.15` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| off | -- | −30.2425 | −32.8842 | -- | −33.3847 |
| 0.01 | 8.97 | −30.3292 | −32.9375 | −33.2528 | −33.4102 |
| 0.1 | 6.68 | −30.7093 | −33.1037 | −33.3748 | −33.4845 |
| 0.3 | 5.59 | −31.1507 | −33.1911 | −33.4228 | −33.5108 |

The gain shrinks as the basis grows: a second zeta already supplies part of the
contraction the confinement provides. GPAW's compact Gaussian is a better
polarization function than `r · R_outer` here (−0.27 eV at 0.1 eV). The forces
differentiate the confined basis exactly as they do the free one (analytic
against finite difference agree to 5 meV/Å on this system).

Two limits are refused with the usable range in the message: a shift so large
that the confinement would cut into the augmentation sphere (`r_i` must stay
outside `r_cut`), and one so small that the radius runs past the dataset's
radial table. The norm-conserving families do not take `energy_shift` (nor,
therefore, the Gaussian polarization): their confined radial solve has not been
written.

## Fourier filtering

Every integral here is a sum over a grid of spacing `h`, so the grid can only
represent wave-vectors up to its Nyquist value `k_N = π/h`. Whatever a basis
function carries above `k_N` is **aliased**: it does not vanish, it folds back
with a phase that depends on where the function's center falls between two
nodes. Translate a molecule by a fraction of `h` and the energy changes — the
*egg-box* — and `atoms.get_forces()` faithfully differentiates that artifact.

The cure is to remove those components from the **radial functions**, once,
before they are ever sampled (SIESTA's `FilterCutoff`; Anglada & Soler, PRB
73, 115122). Mandacaru does it per angular-momentum channel with the spherical
Bessel pair, multiplying `F_l(k)` by a raised-cosine window that is 1 below
`0.75 k_c` and 0 at `k_c`, then switching the result off smoothly beyond the
function's own support so a split zeta stays short-ranged
(`mandacaru.basis.filtering`).

```python
Mandacaru(method="adapt-vqe",
          basis={"name": "PAW-LCAO", "size": "DZ"},          # filtered: the default
          h=0.20)

Mandacaru(method="adapt-vqe",
          basis={"name": "PAW-LCAO", "filter": False},       # today's raw basis
          h=0.20)

Mandacaru(method="adapt-vqe",
          basis={"name": "NCPP", "filter": True},       # opt in, NCPP default off
          h=0.20)

Mandacaru(method="adapt-vqe",
          basis={"name": "PAW-LCAO", "filter": 800.0},       # explicit cutoff, in eV
          h=0.20)
```

`filter` takes `True` / `"auto"` (cutoff tied to the grid, `k_c = π/h`), a
positive **kinetic-energy cutoff in eV** (`k_c = sqrt(2E)` in atomic units),
or `False`. Anything else raises at construction. It is **on by default for
`PAW-LCAO` and `UPAW-LCAO`, off for `NCPP` and `ONCVPSP`** — declared once per family as
`FamilySpec.default_options`, so a new family states its own and the drivers
need no edit. `filter=False` reproduces the unfiltered basis byte for byte.

### Why the cutoff sits exactly at Nyquist

Remove what the grid cannot carry, keep everything it can. The measurement
agrees: water/PAW-LCAO-SZ, peak-to-peak RHF energy over one grid period under a
rigid shift along (1,1,1), against the rise in the energy itself.

| `k_c / k_N` | ripple, h = 0.25 | cost, h = 0.25 | ripple, h = 0.20 | cost, h = 0.20 |
|---|---|---|---|---|
| off  | 268.7 meV | — | 132.8 meV | — |
| 1.3  | 142.8 | 489 meV | 17.2 | 273 meV |
| 1.2  | 94.4  | 584 | 7.2 | 326 |
| 1.1  | 39.0  | 700 | **4.6** | 365 |
| **1.0** | **27.1** | 832 | 6.0 | 422 |
| 0.95 | 43.0  | 1020 | 6.6 | 477 |
| 0.9  | 75.9  | 1477 | 6.7 | 541 |
| 0.8  | 162.1 | 4240 | 6.8 | 679 |

The ripple has a minimum at 1.0–1.1 and rises on *both* sides: above, because
the unrepresentable components are still there; below, because the cutoff
starts eating the band the grid can carry, which distorts the orbital near the
core where the local potential varies fastest. Every fraction below 1.0 is
worse on both axes, so the only real choice is 1.0 or higher — and 1.0 is the
one that leaves nothing aliased.

Per term at h = 0.25, the ripple goes (unfiltered → filtered): kinetic
367.5 → 0.05 meV, local pseudopotential 1608.3 → 21.0, electron repulsion
1321.5 → 7.1. The total is far smaller than its parts because they partly
cancel. The nonlocal and compensation terms do not appear because they never
aliased: PAW-LCAO evaluates both by atom-centered quadrature, not on the grid.

**What is left is the local potential.** At h = 0.25 it is 21.0 of the
remaining 27.1 meV, and at h = 0.20 it is 5.8 of 6.0 — the filter band-limits
the *basis*, but the grid integral `∫ φ*φ v_loc` still aliases through
`v_loc`'s own high wave-vectors, which nothing here touches. Filtering the
local channel too, or evaluating that term by quadrature, is the obvious next
step and is independent of this one.

### What it buys, and what it costs

The thing filtering is *for* is the force. Water on a frozen grid,
`project_translation=False` (the net force must be zero for a free molecule,
so whatever is there is the egg-box):

| | net force, unfiltered | net force, filtered | largest force |
|---|---|---|---|
| h = 0.25 | 2.394 eV/Å | **0.160** | ~2.9 |
| h = 0.20 | 1.158 | **0.045** | ~2.4 |

The analytic gradient is still the derivative of the calculator's own energy:
central-difference agreement is 1.0e-3 (H₂, h = 0.25) and 7.5e-4 eV/Å (water,
h = 0.25) with the filter on, the same as without it.

The reported energy goes **up** — water/PAW-LCAO-SZ by 832 meV at h = 0.25 — and
most of that is not a loss. Hold the cutoff fixed at 602 eV and refine the
grid: the filtered-to-unfiltered gap is 530 / 422 / 294 / 237 meV at
h = 0.25 / 0.20 / 0.16 / 0.13, close to linear in `h`. It is the *unfiltered*
basis's own grid error — the finite-difference Laplacian under-estimates the
kinetic energy of a function the grid cannot resolve, so an unfiltered orbital
is reported too low. On H₂, where both bases are resolved by h = 0.12, the gap
is 2 meV and has the opposite sign.

And what a user actually reads off — geometry and binding — barely moves,
because the shift is nearly a constant offset that cancels in differences
(exact sector ground states, PAW-LCAO-SZ):

| | d_eq unfiltered | d_eq filtered | D_e unfiltered | D_e filtered |
|---|---|---|---|---|
| H₂ (h = 0.25)  | 0.9168 Å | 0.9167 Å | 2.801 eV | 2.807 eV |
| LiH (h = 0.30) | 1.6536 Å | 1.6538 Å | 1.879 eV | 1.876 eV |
| OH (h = 0.20)  | 1.0423 Å | 1.0442 Å | 3.466 eV | 3.429 eV |

OH's total energy moves 379 meV while its bond moves 0.002 Å and its binding
energy 0.037 eV.

```{note}
**Filtering changes the basis, so it changes the numbers.** It is a modeling
choice, not a numerical detail: the filtered first zeta is no longer exactly
the pseudo-orbital the projectors were built from, so the atomic reference is
no longer reproduced exactly. That is why the norm-conserving families leave
it off — their orbitals are not built band-limited — while PAW-LCAO and UPAW-LCAO turn
it on, since `optimize_pseudo_waves` already minimizes the kinetic energy
beyond `q_cut` and the filter has little left to take. Set `filter=False` to
compare against an older result.
```

## Limits

The residual force on an isolated atom is still ~30 eV/Å at `h = 0.10 Å`. That
remainder is basis-set incompleteness — a minimal valence s+p shell per atom —
not the core, and it *shrinks* with grid refinement. A polarized multiple-zeta
basis addresses it directly; see [Basis sets](basis_sets.md).

```{note}
**Relaxation works, and oxygen needs a finer grid than lithium.**
LiH relaxes cleanly (`examples/28_LiH_relaxation_PAW.py`: five BFGS steps from
2.0 Å to d = 1.6876 Å, final fmax 0.002 eV/Å), and so do H₂, OH, CH and H₂O.
Water from a 90°, 1.0 Å start converges in **three BFGS steps** (h = 0.16 Å,
PAW-LCAO-SZ, 123 s) to d = 0.992 Å and an angle of 117.3°, against 0.9572 Å and
104.52° in experiment: the bond length is good, and the 13° on the angle is
the minimal valence s+p shell, not the gradient — a polarized basis is what
addresses it (see [Basis sets](basis_sets.md)).

* **Check that the energy binds before relaxing.** A symmetric O–H scan at
  h = 0.16 Å gives −480.80, −481.92, −482.20, **−482.29**, −482.21, −481.63 eV
  at 0.85, 0.92, 0.96, **1.00**, 1.05 and 1.15 Å — a clean minimum. (Earlier
  releases did *not* bind water: the compensation charge was missing its
  electron–ion attraction, and every bond to a p-valence atom came out
  repulsive. If you are on an older version, scan before you relax.)
* **Egg-box.** Rigidly translating a molecule on a frozen grid changes the
  energy, so the forces of a free molecule do not sum to zero. The net force is
  a faithful derivative of that non-invariant discretized energy, and it is the
  honest measure of whether a force is good enough for geometry. Unfiltered,
  for water it falls as `|sum F|` = 1.85, 0.86, 0.41, 0.099, 0.025 eV/Å at
  h = 0.30, 0.25, 0.20, 0.16, 0.13 Å — oxygen then needs h ≤ 0.16 Å for
  geometry, finer than the h ≤ 0.25 Å that suffices for energies.
  *Fourier filtering* (above), on by default for PAW-LCAO, removes most
  of it (15–26× on water) and is the reason a coarser grid is now usable.
* **`project_translation=True`** subtracts the mean force so the molecule
  cannot drift. A free molecule's exact forces do sum to zero, so this enforces
  a symmetry rather than hiding an error — but it removes only the
  *translational* part of the egg-box, so it is not a substitute for a grid
  fine enough to trust. The unprojected residual stays on
  `force_result.details["translational_residual"]`.

The gradient itself is checked against a central difference of the same energy
on the same frozen grid: 1e-4 eV/Å on H₂, 3e-3 eV/Å on water (oxygen exercises
the L = 1 and L = 2 compensation multipoles). Mandacaru flags the two things that
most often say a force will not be usable — an unresolved projector, and a
state that is not stationary with respect to orbital rotations — as
`RuntimeWarning`s, and warns when the net force is a significant fraction of
the largest force.
```

## Tests and their budget

The test suite is resource-monitored by `test/conftest.py`: every test is
timed, the peak RSS is read after each, and a summary table of the slowest
tests is printed at the end of the session (the complete table is written to
`test/.resource_report.txt`). The budget — set for the pseudopotential tests,
the heaviest in the suite — is one test < 3 min, the whole run < 10 min, peak
RSS < 3 GB; shrink a test's grid or cell rather than the limits.
`pseudopotentials/test_ncpp_family.py` pins the NCPP energies of H₂ (0.74 Å, h = 0.25 Å) and
LiH (1.6 Å, h = 0.30 Å) measured before the nonlocal generalization and
exercises the general form with synthetic projectors; `pseudopotentials/test_oncv.py`
validates the ONCVPSP family atomically (H, Li, O) and on the same two
molecules (50 tests, ~11 s, peak RSS 0.6 GB); `pseudopotentials/test_paw.py` does the same for
the PAW-LCAO family, adding the overlap, on-site-projection, compensation and
grid-stability checks (70 tests, 11.5 s, peak RSS 0.77 GB);
`pseudopotentials/test_engine.py` covers the engine, the basis-name selector
and the per-element sizes.

See `examples/26_pseudopotential_generation.py` and
`examples/27_pseudopotential_calculations.py`.
