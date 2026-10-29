# Pseudopotentials

A pseudopotential replaces an atom's core electrons and the singular $-Z/r$
potential by a smooth, valence-only problem. In Mandacaru a pseudopotential
**family is a basis name**, selected exactly like `"HAO"` or `"cc-pVTZ"`:

atoms.calc = Mandacaru(method="adapt-vqe",
                       basis="PAW-LCAO",    # Bloechl's projector augmented wave
                       grid_spacing=0.25)
atoms.calc = Mandacaru(method="vqe",
                       basis={"name": "PAW-LCAO", "size": "DZP"},   # polarized double zeta
                       grid_spacing=0.25)
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis="UPAW-LCAO",   # the unitary PAW-LCAO variant
                       grid_spacing=0.25)

That basis turns an all-electron calculation into a valence-only one: the core
electrons are removed (oxygen keeps 6 of its 8), the basis becomes the family's
smooth pseudo-atomic orbitals — with the same `size` hierarchy as the
[NAO family](basis_sets.md) as its options — and the singular $-Z/r$ external
potential is replaced by a bounded local channel plus the family's projectors.
Every driver, `interaction_energy`, the periodic methods, the dry run and the
command line (`mandacaru H2O --cell 8 --basis PAW-LCAO --basis-option size=DZP`)
accept the names; `active_space`'s `"frozen"` is refused with them as
redundant. There is no
separate switch: the family is the basis, and the retired `"PP"` basis name
raises an error that names the families instead of aliasing to one.

## Families

Two families are shipped, all generated from scratch by Mandacaru's own LDA
radial atomic solver (`mandacaru.basis.atomic_solver`):

| Basis name | Aliases | Family | Projectors | Overlap | Library |
|---|---|---|---|---|---|
| `"PAW-LCAO"` | — | Blöchl's projector augmented wave (below) | two per channel, $2\times2$ coupling | $S + C\,q\,C^\dagger$ | `mandacaru-paw`, H–U |
| `"UPAW-LCAO"` | `"unitary-paw-lcao"` | the same, with a **unitary** transformation ($q = 0$, below) | two per channel, $2\times2$ coupling | $S$ (unaugmented) | generated on demand |

Names are case-insensitive. Each family accepts the options `size`,
`zeta_split` (`"first_zeta"` by default, or `"last_zeta"`; the scheme fixes
its own split radii), `paw_dataset` (an alternative dataset folder) and
`filter_cutoff` (*Fourier filtering*, below); PAW-LCAO and UPAW-LCAO also take
`projector_basis="raw"|"dual"`, `energy_shift`, `confinement_potential` and
`polarization_method` (*Confined orbitals*, below). Any other key — or an
all-electron option such as `tier` — is refused before an integral is
computed. A family may give an option a **default of its own**: PAW-LCAO and
UPAW-LCAO declare `filter_cutoff="auto"`.

`"PAW-LCAO"` is the recommended pseudopotential family. `"UPAW-LCAO"` is an option — the
same construction with a unitary transformation — and the measurements that
decided that are in its own section below.

A **per-element basis** may give each atom its own size, as long as every
element uses the *same* family:

```python
basis={"O": {"name": "PAW-LCAO", "size": "DZP"}, "H": {"name": "PAW-LCAO"}}
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

family_names()      # ['paw-lcao', 'upaw-lcao', 'unitary-paw-lcao']
spec = resolve_family("unitary-paw-lcao")   # -> PSEUDO_FAMILIES["upaw-lcao"]
spec.name, spec.aliases, spec.label    # "upaw-lcao", ("unitary-paw-lcao",), "UPAW-LCAO"
spec.options       # (..., "filter_cutoff", "energy_shift",
                   #  "confinement_potential", "polarization_method")
spec.default_options       # {"filter_cutoff": "auto", "energy_shift": 0.1}
spec.resolved_options({"size": "DZ"})  # the defaults with the user's options on top
spec.norm_conserving                   # False
spec.generate("O")                     # generate_upaw("O")
spec.get("O")                          # the (cached) library loader
spec.build(atoms, grid, grid_spacing, charge, spin, options, kinetic)
lookup_family("HAO")                   # None -- an all-electron basis name
```

`build` returns exactly the 5-tuple the all-electron path returns —
`(hamiltonian, num_particles, n_spatial_orbitals, integration_profile,
context)` — and `_pseudopotential_hamiltonian` in the driver is a thin
dispatcher on it. The family name is also **stored in every pseudopotential
file** (`family` field, format version 2); a file that does not carry the
field, or carries no radial tables, is refused rather than guessed at.

A new family is added by registering a `FamilySpec`:

```python
register_family(FamilySpec(name="gth", description="...",
                           generate=..., get=..., build=...,
                           norm_conserving=True, aliases=("goedecker",),
                           options=("size", "zeta_split", "paw_dataset",
                                    "filter_cutoff"),
                           default_options={"filter_cutoff": "auto"}))
```

and its name immediately works as a basis name — `basis="GTH"` — on every
driver, in the dry run and on the command line, with no change to any of them.

## The shared radial machinery

PAW-LCAO and UPAW-LCAO are generated by one radial construction, in
`mandacaru.pseudopotentials.partial_waves`, on Mandacaru's own LDA radial
atomic solver. The smooth partial waves follow D. R. Hamann's optimized
construction, *Phys. Rev. B* **88**, 085117 (2013); this section describes the
parts both families share, and the *PAW-LCAO* section below adds what makes
them projector augmented waves.

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
   (`energy_offset`), integrated outward and normalized to one inside $r_c$
   (at an absolute energy for the channels `second_reference` names).
2. **Smooth partial waves.** Inside $r_c$, $\tilde\varphi_i = \sum_{n=1}^{8}
   c_{in}\,j_l(q_n r)$ with the $q_n$ the interleaved zeros of $j_l$ and
   $j_l'$ at $r_c$ (`bessel_wavevectors`). The RRKJ choice — every basis
   function carrying the AE logarithmic derivative — is *not* usable here:
   with $q\,j_l'(qr_c) = L\,j_l(qr_c)$ the Bessel equation turns every
   derivative row into $j_l(q_n r_c)\times$ a polynomial in $q_n^2$, and the
   third-derivative condition becomes a combination of the value and
   second-derivative conditions. Constraints: value and first three
   derivatives at $r_c$ (Hamann's `ncon = 4`, the targets taken from the AE
   radial equation, `matching_targets`) and the inner norm matrix
   $\langle\tilde\varphi_i|\tilde\varphi_j\rangle_{r<r_c}$, which is set to a
   multiple of the all-electron one (`norm_factor`; 1 is Hamann's generalized
   norm conservation). In the
   remaining freedom the **residual kinetic energy**
   $E^r_i(q_c) = \tfrac12\int_{q_c}^\infty q^4|\tilde\varphi_i(q)|^2\,dq$
   ($q_c = 5$ Bohr⁻¹, `q_cut`; the transform of the whole wave, AE tail
   included) is minimized *exactly*: the linear constraints are eliminated
   through their null space and the one quadratic constraint (the wave's own
   norm) is a Lagrange multiplier found as a one-dimensional root on the
   Moré–Sorensen branch (`constrained_minimum`). No SLSQP; the norm matrix is
   satisfied to 1e-14 (`optimize_pseudo_waves`).
3. **Local potential.** An even polynomial continuation of the screened AE
   potential inside $r_{cl} = 0.9\,\max_l r_c$ (value and four derivatives
   matched, `polynomial_local_potential`; Hamann's `dvloc0` shift of the
   origin value is available as `local_shift`, default 0), unscreened with
   the Hartree and LDA xc potentials of the pseudo valence density.
4. **Ghost and scattering checks.** The generator checks that the local
   potential and the projectors bind no spurious state below the valence
   level and that the logarithmic derivative of the smooth system matches the
   all-electron one (`ghost_free`, `log_derivative_ae`) -- for every channel
   with projectors, and, within 0.2 rad, for every angular momentum without
   them up to one above the highest channel, which scatters off the local
   potential alone (`unprojected_scattering_errors`). Above the reference,
   a channel may not hold more bound levels (below −0.1 Ha) than the atom
   has valence states of its $l$: a compact semicore channel whose second
   reference energy is still bound can put a spurious level just above it,
   outside every scattering window (the 4f of W–Hg, the 4p of Tc and the
   LDA 5p of Ta, −0.15 to −2.5 Ha). These channels take their second
   reference at +0.25 Ha instead (`second_reference={l: energy}`, by default
   `DEFAULT_SECOND_REFERENCES`), and their datasets were rebuilt that way. A failing candidate is repaired first by a shorter local radius with no shift, then by local
   shifts and cutoff adjustments. The order matters: a shift acts on every
   angular momentum the projectors do not cover, and aluminum's former 5 Ha
   shift scattered d 0.52 rad off and put fcc Al 12 % too large. What cannot
   be repaired is
   refused, kept or flagged on the dataset as the `ghosts` mode says
   (`"repair"`, `"refuse"`, `"keep"`, `"flag"`), and a flagged dataset warns
   with `GhostStateWarning` when it is loaded.
5. **Completeness of the $s$ channel.** The $l = 0$ part $f$ of a hydrogen
   1s centered 0.2 Bohr outside $r_c$ is expanded in the channel's projectors
   and smooth partial waves, and the *intruding-1s miss*
   $\lVert f - \sum_i \langle\tilde p_i|f\rangle\tilde\varphi_i\rVert /
   \lVert f\rVert$ inside the sphere must stay below 1 (`intruding_s_miss`).
   It is a ratio of norms: 0.05 misses 5 % of the function, 1 misses as
   much as the function holds, and a miss of 1.76 is 176 % -- the expansion
   is farther from the function than zero is. One function at one distance:
   passing it does not certify every neighbor.

A check that cannot be evaluated is a failure, never a pass: a level or a
phase that is not finite raises `DiagnosticError` naming the element, channel
and energy, and a construction whose checks cannot be evaluated is not
written, not even flagged. The checks against the all-electron atom need the
atom, which a dataset read from a file does not carry: they raise
`DiagnosticError` ("not evaluated") rather than return an empty result.
`reference_atom(pp)` (`mandacaru.pseudopotentials.partial_waves`) solves the
generator's atom again for a loaded dataset (`pp.atom = reference_atom(pp)`),
and `ghost_errors(pp, levels, with_atom=False)` checks the projected
channels of one without it. `mandacaru-build` releases the file, not the
object in memory: the dataset is written beside its final name, read back,
and moved into place only when every channel's bound levels match the
generated ones to `RELOAD_LEVEL_TOLERANCE` (1e-5 Ha); its report is of the
file read back.

### The reference atom and the channel set

Several arguments of `generate_paw` change what the pseudopotential is built
*from*. They are generation-time only: they change the dataset, never the
calculation that later reads it.

| Argument | Default | What it does |
| --- | --- | --- |
| `relativity` | `"scalar"` | `"none"`, `"scalar"` (Koelling–Harmon) or `"dirac"` (each $j$ separately) |
| `xc` | `"lda"` | `"lda"` or `"pbe"`; screens the atom and unscreens the potential with the same functional |
| `nlcc` | `True` | Partial core density; `True` matches where $\rho_c = \rho_v$, a float sets the radius |
| `extra_l` | `0`, La: `1`; K, Ca, Rb, Sr: up to d | Empty channels above the highest occupied valence $l$; a bound non-core atom level is the first reference when present, otherwise both references scatter |
| `points`, `r_max` | per element | The radial grid of the reference atom |
| `frozen_subshells` | Tl–Rn: `4f`; otherwise none | Move selected occupied subshells into the pseudopotential core |
| `semicore_subshells` | Na, Mg, K, Ca, Rb, Sr, Sc, Ti, V, Y–Tc, Ta: $(n-1)s\,(n-1)p$; otherwise none | Move selected filled core subshells into the valence; `()` turns it off |

**Semicore valence.** An alkali or alkaline-earth atom with an $s$ channel
alone has no projectors for $p$, and its $p$ scattering -- the metallic bond
-- is the local potential's. No local potential is both ghost-free and right
for it: every sodium built that way put bcc Na 10–40 % off its lattice
constant, either way. The shipped datasets of Na, Mg, K, Ca, Rb and Sr
therefore carry the $(n-1)s\,(n-1)p$ shell in valence (sodium:
$2s^2 2p^6 3s^1$, 9 electrons). The $p$ channel gets the semicore $p$ as its
bound reference; the $s$ channel has two bound references, the semicore and
the valence $s$; the cutoff is 0.7 of the valence $s$ orbital's peak radius
(Na 2.3 Bohr, against 4.2 for the old $s$-only dataset), inside half of each
elemental crystal's nearest-neighbor distance. K, Ca, Rb and Sr also get an
empty d channel, and calcium's binds a 3d that joins the basis. Semicore
sodium puts bcc Na at 4.09 Å (all-electron LDA 4.05), fcc Ca 5.32 (5.33),
and
Na₂, NaH and NaCl within 1 % of all-electron LDA bond lengths. Cs, Ba, Fr
and Ra are not semicore yet -- cesium's d channel then binds a ghost and
barium's and francium's f scatter wrongly -- so they ship $s$-only and
flagged: their datasets warn on load that the local potential scatters
wrongly.

The early d block has the same gap in its empty $p$: with $s$ and $d$
(and $4f$) channels alone, the local potential of Sc, Ti, V, Y–Tc and Ta
scattered $p$ 0.2–1.5 rad off for every repair, in LDA and PBE alike; PBE
ScH came out 11 % long, bcc Nb 4 %, Mo 12 % and hcp Ti 6 %. With their
$(n-1)s\,(n-1)p$ in valence (scandium: $3s^2 3p^6 3d^1 4s^2$, 11 electrons)
each builds clean in both functionals (no ghost, phases within 0.04 rad,
the unprojected channel within 0.09 rad, the intruding-1s miss 0.01–0.08),
and PBE ScH, bcc Nb, Mo, Ta and hcp Ti come within 0.7 % of all-electron
PBE. Sc, Ti and V take larger $d$ spheres (2.07, 1.86 and 1.67 Bohr) so that their
filtered $3d$ still projects onto its own wave. Semicore Lu, Hf, W and Re
also pass every atomic check but not the solid (bcc W 4 % and hcp Re 5 %
short, hcp Hf and Lu without a converged SCF), and Ag, Cd and Ir lose their
$d$ projection; these keep their valence. The price is
8 more electrons per atom and four more basis functions (the semicore $s$
and $p$ are always single-zeta; only the valence $s$ is split and
polarized).

The `xc` argument belongs to the dataset. It is not the functional of a later
calculation: `Mandacaru(method="dft", xc=...)` chooses its own, and with it
the set it reads (see {doc}`dft`): the PBE set (*The pseudopotential
libraries*, below) for PBE, HSE06 and r2SCAN, the LDA set for LDA. Datasets
of another set, named with `paw_dataset=`, are warned about.

A gradient-corrected reference potential is a cubic spline through its
derivative nodes (`GGA_DERIVATIVE_STEP` apart), so the derivatives the local
potential is matched to at $r_{cl}$ (through the fourth) are read over a fit
window of $\pm 0.15\,r_{cl}$ (`GGA_LOCAL_FIT_FRACTION`) rather than the
$\pm 25$ grid points an LDA potential uses. Over the narrow window the
spline's kinks read as a fourth derivative: PBE aluminum's came out $-51$
against LDA's $-0.31$ at 3.09 Bohr, and the local potential sank to
$-13.4$ Ha at the origin (LDA $-1.5$) and bound ghost states that every PBE
dataset then had to be repaired around.

### Relativity

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
coupling matrix asymmetric by $1.4\times10^{-4}$ Ha for oxygen against a
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
### Nonlinear core correction

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


## PAW-LCAO: projector augmented waves

`"paw-lcao"` is P. E. Blöchl's projector augmented-wave method, *Phys. Rev. B* **50**,
17953 (1994), in its **frozen-core, one-center-expansion** form with the
one-center energies **linearized around the reference atom** — a fixed
per-species coupling matrix $D^0$, which makes the dataset behave like an
ultrasoft pseudopotential with an exact PAW-LCAO reconstruction of the atomic
partial waves. Written from scratch in
`mandacaru.pseudopotentials.paw` on the same LDA radial atom,
reusing the Numerov partial waves, the Bessel machinery and the polynomial
local potential of `mandacaru.pseudopotentials.partial_waves` (see
*The shared radial machinery*, above):

```python
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis="PAW-LCAO",
                       grid_spacing=0.25)
atoms.calc = Mandacaru(method="vqe",
                       basis={"name": "PAW-LCAO", "size": "DZ", "projector_basis": "raw"},
                       grid_spacing=0.25)
```

The name has no alias; `family_names()` lists `paw-lcao` first, then
`upaw-lcao` (and the unknown-family error names both).

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
   $r_c$. $\Delta$ is 1 Ha except for H and Li (0.5 Ha,
   `DEFAULT_ENERGY_OFFSETS`): lithium's wave at +1 Ha sits at a pole of the
   logarithmic derivative ($L = +24$ at $r_c$) and the smooth pair then grows
   nodes and a ghost; for hydrogen +0.5 Ha gives a softer second projector.
2. **Smooth partial waves** (`smooth_partial_waves`). Inside $r_c$,
   $\tilde\varphi_i = \sum_{n=1}^{8} c_{in} j_l(q_n r)$ at the interleaved
   Bessel zeros, matched in value and first three derivatives, residual
   kinetic energy beyond $q_c = 5$ Bohr⁻¹ minimized. **The norm is not
   conserved**, but it is *controlled*: the inner-norm matrix is set to
   $(1-s)$ times the all-electron one (`optimize_pseudo_waves(...,
   norm_factor=1-s)`, the same optimizer with a scaled target), so
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
   Ha below $\varepsilon_{2s}$ — which the projector term does
   not suppress; the raise is chosen so the
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
   the norm-conserving convention: $\tilde v^{ion} = \tilde v^{scr} -
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
   Li +0.0006, C +0.030, N +0.053, O −0.107, F −0.570 Ha. It enters the
   Kohn-Sham Hamiltonian through `MolecularIntegrals.constant_energy` (next
   to the nuclear repulsion). A Hartree-Fock or correlated Hamiltonian does
   not use it: see *The spheres of a wave-function method*, below.

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
  one grid step moved the nonlocal energy by 3.9 eV at grid_spacing = 0.25 Å. The
  quadrature depends only on the separation of function and projector, so it
  is exactly translation invariant; the grid ripple left (kinetic, local and
  Hartree terms) is 18 / 13 / 5 meV at grid_spacing = 0.25 / 0.20 / 0.15 Å
  (measured with `kinetic="fd"`).
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

with $\sigma = 1.4\,h$ (`SIGMA_FACTOR`; $h$ is the grid spacing,
`grid_spacing`), which puts
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
peak-to-peak per term in meV (measured with `kinetic="fd"`, the
finite-difference stencil):

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
*larger* than the full potential's at grid spacings of 0.25 Å and above, because the Gaussian ion
is a narrower well than the pseudized channel it replaces ($-7.2$ Ha at the
origin against $-5.4$ Ha for oxygen at a grid spacing of 0.25 Å). Widening $\sigma$ reduces
that term monotonically (oxygen at a grid spacing of 0.30 Å: 3425 / 2714 / 2097 / 1501 /
1159 meV at $\sigma$ = 1 / 1.4 / 2 / 3 / 4 grid spacings) but never to zero, because what remains is
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
the reported grid resolution ratio differs (H at grid_spacing = 0.25 Å: raw 0.90/0.78,
dual 0.78/0.77). The local potential is interpolated with a cubic spline
(`PAWDataset.local_potential`), so its derivatives are continuous.

### The spheres of a wave-function method

What a sphere carries depends on the method. Kohn-Sham DFT keeps the
dataset's one-center terms as generated: the functional's one-center
Hartree and exchange-correlation energies, linearized at the reference atom
inside $D^{ion}$ and the constant $E_{1c}$. Hartree-Fock (RHF, UHF, GHF),
FCI, VQE, ADAPT-VQE and every other wave-function method cannot use them: a
functional's exchange-correlation is not part of their Hamiltonian, and an
exchange operator frozen at the spherical neutral atom treats every
electron of a channel alike, which exact exchange does not. Their
Hamiltonian (`PAWIntegrals(..., one_center="exact")`, the default; Kohn-Sham
asks for `"frozen"`) therefore carries, per sphere $A$:

* **the one-center Coulomb tensor in the two-body operator.** Blöchl's
  electron-electron correction inside the sphere is
  $\Delta W_{abcd} = (\phi_a\phi_b|\phi_c\phi_d) -
  (\tilde\phi_a\tilde\phi_b + \hat n_{ab}|\tilde\phi_c\tilde\phi_d + \hat
  n_{cd})$ (`one_center_coulomb`), and it enters the two-body integrals as
  $\Delta\langle pq|rs\rangle = \sum_{abcd} C_{pa}C_{rb}\,\Delta W_{abcd}\,
  C_{qc}C_{sd}$, kept as the low-rank sum $\sum_k\lambda_k G^k\otimes G^k$
  with $G^k = C_A U^k C_A^T$ from the eigenvectors of $\Delta W$ as a
  pair-pair matrix (rank 33 for an s+p atom's eight projectors, 3 for
  hydrogen). Every consumer of the two-body operator -- the dense tensor,
  the integral-direct route, the Cholesky factors, the active space, DLPNO,
  the qubit Hamiltonian -- gets the sphere's Hartree and exchange exactly,
  at any density;
* **the dataset's frozen functional removed** from $D^{ion}$:
  $V_{ij} = \int v_{xc}[n_c + n^1_0]\,\phi_i\phi_j - \int v_{xc}[\tilde n^1_0
  + \tilde n_c]\,\tilde\phi_i\tilde\phi_j$, together with the linearized
  one-center Hartree term $\Delta W\cdot D^0$ that the quadratic one
  replaces;
* **the frozen core's exchange** added to $D^{ion}$,
  $K_{ij} = -\sum_c f_c(2l_c+1)\sum_L
  \begin{pmatrix} l_c & L & l\\ 0&0&0\end{pmatrix}^2 R^L(\phi_i\phi_c,
  \phi_c\phi_j)$ per spin, $f_c$ the per-spin occupation of each orbital of
  core subshell $c$. The files store the core density, not its orbitals, so
  the reference atom is solved again when the first Hamiltonian of a process
  needs a species (`partial_waves.reference_atom`, cached per dataset
  file): 0.5 s for H, 2-4 s for C-F, 8 s for Cl, 16-20 s for Fe and Cu,
  30-70 s for Br-I and about 3 minutes for Au;
* **no constant.** $E_{1c}$ is the double counting of the linearized
  Hartree term and of the functional; with the Hartree term quadratic and no
  functional left nothing remains, and `constant_energy` is zero.

`mandacaru.pseudopotentials.onecenter.OneCenterExact` builds the three
terms; the forces differentiate them through the projections. What this
changes (HISTORY.md, 2026-10-12):

| | frozen spheres | exact spheres | all-electron reference |
| :--- | ---: | ---: | ---: |
| H atom, UHF 1s (eV), spectral kinetic | -14.2 (`grid_spacing` → 0: -13.98) | -13.59 | -13.606 |
| F UHF electron affinity (eV), QZ2P + diffuse shells | 2.39 | 1.05-1.20 | 1.18 (aug-cc-pVQZ) |
| Cl UHF electron affinity (eV), same | 2.37 | 2.36 | 2.37 (aug-cc-pVQZ) |
| water RHF $\alpha$ (a.u.), DZP + diffuse shells, spectral kinetic | 7.41 | 8.60 | 8.64 (aug-cc-pVQZ) |
| hydrogen fluoride RHF bond length (Å), DZP | 0.853 | 0.906 | 0.897 (HF limit) |

The rows not marked "spectral kinetic" were measured with `kinetic="fd"`.

Cl does not move because its 3p lives outside its sphere, in the smooth
part, where exchange was always exact; F's compact 2p puts a third of an
hartree of exchange inside it.

### Forces (`algorithms/pseudo_forces.py`)

`atoms.get_forces()` with `basis="PAW-LCAO"` returns the
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
`force_result.details["orbital_gradient"]` reports the residual. The basis
functions use real spherical harmonics, so the SCF's molecular orbitals and
the MO integrals are real, which the real excitation operators of every pool
need (with complex harmonics, orbitals left with the SCF's phases stalled
ADAPT 62 meV above the ground state on H₂ PAW-LCAO-DZP). H₂ and LiH in PAW-LCAO-DZP are 20-qubit problems,
solved exactly in their (1, 1) particle-number sector (`core.sector`, 100
states).

On H₂ (grid_spacing = 0.25 Å) the analytic force agrees with a central difference of the
energy to 1e-4 eV/Å. Against a plane-wave PBE PAW calculation the force curves
agree qualitatively: the H₂ minimum is near 0.81 Å instead of 0.750 Å — mostly
from the H augmentation radius, 1.30 bohr, which two atoms 0.75 Å apart overlap
almost entirely — while the LiH bond forces from 2.1 to 3.2 Å match within
0.15 eV/Å. Pinned by `test/algorithms/test_paw_forces.py` and `test/core/test_sector.py`.

#### Mean-field forces

`Mandacaru(method="rhf" | "uhf" | "dft", basis={"name": "PAW-LCAO", ...})`
returns forces without a variational state. The converged orbitals are
orthonormal in a Löwdin frame that is rebuilt at every geometry, so the force
is the derivative of the total energy at fixed densities in that frame. Mandacaru
takes it as a Richardson difference (step $10^{-3}$ Bohr) of the same energy
expression on the fixed grid, using the retained PAW setup; no SCF is repeated.
All geometry dependence is differentiated together: the basis and projector
centers, the overlap and compensation augmentation, the compensation charge and
core density on the grid, the one-center terms, the closed-1s core exchange and
the ion–ion repulsion. Hartree–Fock, LDA, PBE and r²SCAN, restricted and
unrestricted, are supported. GHF, screened exchange (HSE06) and nonlocal
correlation (rVV10) raise. Because the grid does not follow the atoms, the force
is the exact derivative of the discretized energy, egg-box term included;
`project_translation` removes the net force as for the other methods. The run
log records the scheme, the step and the step-error estimate. Pinned by
`crates/mandacaru-core/tests/paw_forces.rs` and `test/migration/test_native_dft.py`.

### Validation (pinned by `test/pseudopotentials/test_paw.py`, 70 tests, 11.5 s, peak RSS 0.77 GB)

Atomic, freshly generated (`check_paw_channel`: `paw_spectrum` = the
generalized problem with the 3-point Laplacian on 0.01/0.02 Bohr grids,
Richardson-extrapolated; `paw_eigenstate` + `reconstruct_ae` at 0.005 Bohr for
the first row; `log_derivative_paw` = the exact nonlocal Numerov
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

Molecular (H₂ 0.74 Å at grid_spacing = 0.25 Å, LiH 1.6 Å
at grid_spacing = 0.30 Å, SZ basis, 4 qubits, ADAPT-VQE with the `qeb` pool, ≤ 4
iterations, unconfined; energies in eV as the user sees them, Hartree in
parentheses; measured 2026-10-13 with the exact one-center spheres and the
spectral kinetic default, and pinned by `test_paw.py`):

| | PAW-LCAO RHF | PAW-LCAO FCI = ADAPT |
|---|---|---|
| H₂ | −28.706 eV (−1.054916) | −29.084 eV (−1.068832) |
| LiH | −19.812 eV (−0.728080) | −20.160 eV (−0.740849) |

On H₂ the augmented
overlap has eigenvalues 0.203 / 1.809 (bare 0.199 / 1.757), the
Löwdin-orthonormalized overlap is the identity to 1e-16, and the augmented
two-body tensor keeps the pair-density symmetries to 1e-12. DZ on H₂ (8
qubits) is variational against SZ.

**Hardness at grid_spacing = 0.25 Å** (`resolution_ratios` of the basis,
`kb_resolution_ratios` of the raw projectors; the basis ratios with the
spectral kinetic default, 0.98 and 1.219 / 0.975 on the stencil): H₂ basis
1.001, projectors 0.90 / 0.78; LiH basis 1.072 / 1.000,
projectors 1.00 / 1.00 / 0.88 / 0.85 — all inside the ±25 % band. The second
H projector, built from the scattering wave, is the hardest object; at
grid_spacing = 0.35 Å (0.66 Bohr) the H projectors are not resolved and the result is
nonsense.

### Library and files

`$MANDACARU_PAW_PATH/lda-sr/{H,Li,C,N,O,F}.parquet` (100–385 kB, decimated to
0.02 Bohr; 9.6 s to regenerate with `build_paw_library()`; `get_paw(symbol,
paw_dataset, xc)` is the family's loader, `paw_library_path()` its folder
(`$MANDACARU_PAW_PATH/lda-sr` by default, `lda-fr` for
`relativity="dirac"`, `pbe-sr` for `xc="pbe"`, `pbe-fr` for both). The record is `PAWDataset` (a `PseudoPotential` subclass:
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
`io.py` dispatches the table families (`TABLE_FAMILIES`) by *record type*
on save and by the family declared in the payload on load; a file without a
recognized `family`, or without `radial_tables`, is refused outright. Round
trips are lossless and idempotent (tested).

### Relativity, GGA and spin-orbit coupling

`generate_paw` takes the `xc`, `relativity`, `nlcc` and `extra_l` arguments
described under *The shared radial machinery*, with the defaults `"lda"`,
`"scalar"`, `True` and `0`; `relativity="none", nlcc=False` gives the plain
non-relativistic construction without a partial core.

**A relativistic reference atom also uses relativistic exchange.** Near a
heavy nucleus the Fermi momentum $k_F = (3\pi^2\rho)^{1/3}$ is no longer small
against $c$, and the exchange of the relativistic uniform gas is weaker
(MacDonald and Vosko, *J. Phys. C* **12**, 2977 (1979)). With
$\beta = k_F/c$, both relativistic treatments (`"scalar"` and `"dirac"`)
scale the exchange energy per electron by
$\Phi_E = 1 - \tfrac32\big(\sqrt{1+\beta^2}/\beta
- \operatorname{asinh}\beta/\beta^2\big)^2$ and the exchange potential by its
exact derivative,
$\Phi_V = -\tfrac12 + \tfrac32\operatorname{asinh}\beta/(\beta\sqrt{1+\beta^2})$,
for LDA and PBE exchange alike. The reference atom, the unscreening, the
one-center energies and the rescreening of a loaded file all use the same
functional. A dataset records it (`relativistic_exchange`), so a file written
before the correction existed still rescreens the way it was unscreened. On
Au and Bi the correction moves the 6s and 5d levels by 1–2.5 mHa and the
semicore 5p and 4f by 6–20 mHa. It lowers the core's exchange by tens of
Hartree, and changes the spin-orbit splittings by at most 1.3 %.

```{note}
A relativistic $s$-channel
eigenvalue does not converge cleanly on this grid (the $r^\gamma$ cusp of a
point-nucleus Dirac state), so treat a valence relativistic shift as good to
about 10-15 % at production grid densities, not better -- see the
*Relativity* subsection under *The shared radial machinery*, above, for the
measured table. It does
not touch the spin-orbit splittings below, which come entirely from $p$ and
higher channels.
```

Two things follow from PAW-LCAO's overlap operator.

**A Dirac PAW-LCAO dataset is scalar-relativistic partial waves plus a
$j$-resolved spin-orbit term, not a $j$-resolved augmentation sphere.**
$S = 1 + \sum|\tilde p\rangle q\langle\tilde p|$ is the *metric* of the
generalized eigenproblem, so a $j$-dependent $q$ would give that metric an
$\mathbf{L}\cdot\mathbf{S}$ structure and every consumer of $S$ — the Löwdin
orthogonalization above all — would have to learn about spin. The scalar
channels, $q$, $\Delta T$ and the compensation charges therefore stay as they
are. Next to them, each $l \geq 1$ gets two unitary branches, one per
$j = l \mp 1/2$, built from the Dirac atom's own reference waves with their
own $\kappa$. Any $j$-dependent separable operator is two terms,
$V_j = V^{avg} + V^{SO}\,\mathbf{L}\cdot\mathbf{S}$, so the dataset stores
the $(2j+1)$ average and the $\mathbf{L}\cdot\mathbf{S}$ difference on the
union of the two branches' projectors (`spin_orbit`). Each $j$ is then exact:
on O, Ar, Kr, I, Xe, Au, Tl, Pb, Bi and U every valence level is within
0.05 mHa of the Dirac atom's and every splitting within 0.3 % (the 6p of
Tl–Bi within 0.01 %, where a first-order $\int\xi\varphi\varphi$ term was
19–20 % short, because one $j$-averaged partial wave cannot hold the
$p_{1/2}$ contraction).

In a calculation the $\mathbf{L}\cdot\mathbf{S}$ term acts through that union
of projectors, added to the scalar Hamiltonian; the scalar channel stands in
for the $j$ average. Both $j$ levels are then shifted together by the gap
between the scalar-relativistic level and the Dirac atom's $j$ average: below
0.5 mHa for light and 4d/5f shells, 1–2 mHa for 5p and 5d, 3.7–6.8 mHa for the
6p of Tl–Bi. The splitting itself is kept. The installed Dirac datasets are
read with `paw_dataset="lda-fr"`. The term breaks $S_z$, so among the quantum
drivers only ADAPT-VQE with `pool="spin-orbit"` runs it, and not with the
`parity_reduced` mapping; `method="dft"` solves a molecule or a crystal on it
in two-component spinors ({doc}`dft`).
The molecular orbitals are then the GHF spinors, and an `active_space`
freezes and deletes Kramers pairs of them (see the active-space guide).

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

* **Linearized one-center terms (Kohn-Sham DFT).** The one-center Hartree
  and xc energies are expanded to first order in the density matrix
  $\rho_{ij}$ around the reference atom: a fixed $D^0 = D^{ion}$ per species
  plus the constant $E_{1c}$. The full method recomputes
  $D_{ij}[\rho_{ij}]$ self-consistently from $n^1 - \tilde n^1$ every SCF
  step; here the error is second order in $\rho - \rho^{ref}$. A
  wave-function method does not linearize: its spheres are exact (below).
* **Compensation multipoles up to $L = 2l_{max}$.** Each pair density carries
  the multipoles $\hat n^{LM}$ of its augmentation charge (`multipoles.py`; an
  s-valence atom has only $L = 0$, oxygen has $L = 0, 1, 2$), and the
  compensation charge is attracted to the other ions as well as repelled by
  the electrons. Kohn-Sham DFT still omits the *one-center* two-body
  correction beyond the linearization -- measured at 0.08-0.33 eV for oxygen;
  a wave-function Hamiltonian carries it in full.
* **Frozen core.** The core density is frozen (stored as `core_density`).
  The unscreening includes the smooth core when
  `nlcc=True` (the default), which is the nonlinear core correction: PAW-LCAO
  already builds `smooth_core_density` for its one-center energies, so the
  correction here is a matter of including that density in $v_{xc}$ rather
  than constructing a second one. What remains frozen is the linearized
  treatment of the one-center core–valence xc, reported in
  `energies["core_valence_xc_omitted"]` (Li −1.54, O −4.60 Ha).
* **Wave-function methods on datasets generated with a functional.** The
  partial waves, projectors and frozen core come from the LDA (or PBE)
  reference atom; everything electron-electron in the Hamiltonian of
  Hartree-Fock, FCI, VQE and the other wave-function methods is exact, inside
  the spheres too (*The spheres of a wave-function method*, below). The
  remaining error of this transfer is the partial waves' and the frozen
  core's: the F⁻ affinity comes within 0.0-0.13 eV of the all-electron UHF
  value.
* **Two partial waves per channel**, at $\varepsilon_1$ and
  $\varepsilon_1 + \Delta$ (a per-dataset $\Delta$ only where a repair
  needed one; a semicore $s$ channel's are both bound states), no
  projectors above the valence $l$ unless `extra_l` asks for
  them (or a frozen 4f leaves an empty f channel, Tl–Rn), and a scaled-norm
  construction of the smooth waves rather than Blöchl's free polynomial
  pseudization (the price of a guaranteed positive definite overlap with this
  pair of reference waves). The shipped sets are LDA, scalar-relativistic or
  Dirac (*Relativity, GGA and spin-orbit coupling*, above).
* **Not norm-conserving, but by a controlled amount** ($s$ = 2–15 %).

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
| overlap correction $\max\vert q_{ij}\vert$, H | $2.12\times10^{-2}$ | $3.0\times10^{-14}$ |
| overlap minimum (H) | $1.055$ | $1.000000$ |
| $\max\vert S - \tilde S\vert$, H₂ (grid_spacing = 0.3 Å) | $2.8\times10^{-2}$ | $1.0\times10^{-13}$ |
| O, $L=0$ compensation moment | $3.2\times10^{-2}$ | $7.7\times10^{-15}$ |
| O, $L=2$ compensation moment | $1.4\times10^{-3}$ | $2.2\times10^{-2}$ |
| H₂O net force, grid_spacing = 0.25 Å | 0.380 eV/Å | 1.918 eV/Å |
| H₂O net force, grid_spacing = 0.20 Å | 0.0415 eV/Å | 0.2227 eV/Å |
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

No UPAW-LCAO library is shipped, and its repository, `mandacaru-upaw`, is the
one optional one: `get_upaw(symbol)` looks in `$MANDACARU_UPAW_PATH/lda-sr/`
and, finding nothing there (including when `MANDACARU_UPAW_PATH` is unset),
**generates the dataset on the fly**, caches it in memory and warns once —
generation is 0.4–2.2 s per element, so an interactive run pays a fraction of
a second and a scan pays nothing after the first geometry. Naming a
`paw_dataset` explicitly is a statement that the library is there, and a
missing element then raises with the `build_upaw_library` recipe:

```python
from mandacaru.pseudopotentials.paw import build_upaw_library

build_upaw_library(("H", "C", "N", "O"))          # into $MANDACARU_UPAW_PATH/lda-sr/
build_upaw_library(("H", "O"), paw_dataset="/data/upaw")
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

where $C$ (`MolecularIntegrals.projections()`, an $M\times P$ matrix) contains
the basis–projector overlaps and $D$ is a **block-diagonal** $P\times P$
coupling matrix. PAW-LCAO integrates $C$ on atom-centered spheres.
Each projector
carries three labels — `atom_index`, `channel = (l, m)` and a radial `index`
within that channel — and $D$ has one block per `(atom, l, m)`, of size
$n\times n$ for $n$ radial projectors in that channel. The family supplies the
blocks as `nonlocal_coupling={(atom, l, m): block}`.

PAW-LCAO fills $2\times2$ blocks; the machinery
(`projector_blocks`, `assemble_block_matrix` in `mandacaru.core.hamiltonian`)
validates and assembles them. A family with a single Kleinman–Bylander
projector per channel would instead supply the $1\times1$ block $[E^{KB}_l]$,
reducing $C\,D\,C^\dagger$ to the familiar separable form $\sum_p
|\chi_p\rangle E^{KB}_p\langle\chi_p|$. `kb_nonlocal()` keeps its name (alias
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
metric automatically. Passing `None` (or `Q = 0`) reproduces
the plain Hamiltonian exactly. The PAW-LCAO family uses it (its $q$
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

None of the generated families ships inside the package. Each lives in
a repository of its own — `mandacaru-paw`, `mandacaru-upaw` — and an
environment variable names the checkout Mandacaru reads from:

| family | variable | set it with |
|---|---|---|
| `paw-lcao` | `MANDACARU_PAW_PATH` | `mandacaru-paw --set DIR` |
| `upaw-lcao` | `MANDACARU_UPAW_PATH` | `mandacaru-paw --set-upaw DIR` (optional) |

```bash
git clone https://github.com/seixas-research/mandacaru-paw.git
mandacaru-paw --set mandacaru-paw

mandacaru-paw --status           # a table: one row per dataset set each variable reaches
```

`--set` and `--set-upaw` write `export MANDACARU_..._PATH=DIR` (as an absolute
path) into `~/.zshrc` or `~/.bashrc` (whichever `$SHELL` reads), asking
`[Y/n]` before replacing a different value; open a new terminal, or `source`
the file, for the variable to take effect in your
shell. Inside a checkout the datasets sit one folder per set:

| folder | set | elements |
|---|---|---|
| `lda-sr/` | scalar-relativistic LDA, the default for LDA and every method without a functional | H–U (92) |
| `lda-fr/` | LDA with the spin-orbit term | H–U (92) |
| `pbe-sr/` | scalar-relativistic PBE, the default for PBE, HSE06 and r2SCAN | H–U (92) |
| `pbe-fr/` | PBE with the spin-orbit term | H–U (92) |

`mandacaru-paw --status` lists each of those sets as a row, with its functional,
relativistic treatment, number of datasets, which calculations read it by
default, and a state: green `complete` (all 92 elements), yellow `partial` or
an unset optional library, red for a required library that is unset, names no
folder or holds no dataset folder. It exits 1 in the red cases, so a script can
test it. Color is on when stdout is a terminal; `--color never|always` or the
`NO_COLOR` / `FORCE_COLOR` environment variables override that.

`pbe-fr/` is the PBE counterpart of `lda-fr/` (`mandacaru-paw/FR.md`): the
reference atom is the radial Dirac atom in PBE. It is read like `lda-fr/`,
with `paw_dataset="pbe-fr"` and `method="ghf"` or a spin-orbit pool, or
with `method="dft"`, which then solves a molecule or a crystal in
two-component spinors with the spin-orbit term in its Kohn-Sham operator
({doc}`dft`, *Spin-orbit coupling: two-component Kohn-Sham*; forces and
stress are refused there). Every other consumer of a crystal on a spin-orbit
set refuses it rather than drop the term.
Without `paw_dataset=` a Kohn-Sham PBE calculation reads `pbe-sr/`.
Protactinium is built in both Dirac sets on the
reference $5f^2 6d^1 7s^2$, where all its $j$ branches are bound (the scalar
$5f^3 7s^2$ leaves one unbound). `pbe-fr/` flags 24 elements, none for a ghost: Ag, Cd, Cs, Ba, Fr, La, Ce–Yb, Lu,
Hf, W, Ir and Ac, for the same structural reasons as `pbe-sr/`.

In the PBE sets the reference atom, the
unscreening and the one-center energies are PBE (Perdew-Wang 1992 correlation,
MacDonald-Vosko relativistic exchange), so `pbe-sr/` is the set for
`Mandacaru(method="dft", xc="pbe")` and the hybrid built on PBE, which read
it by default (r2SCAN too, having no set of its own); on the LDA set, named
with `paw_dataset="lda-sr"`, a PBE calculation runs but warns that the
functionals differ, and the warning names `paw_dataset="pbe-sr"`. `pbe-sr/` flags 27
elements, none for a ghost: La–Yb, Ac, Pa and U (no projectors under their
$4f$/$5f$), Cs, Ba, Fr and Ra (no $p$ or $f$), and Lu, Hf, W, Ag, Cd and Ir
(an empty $p$ 0.21–0.67 rad off, Lu also an intruding-1s miss of 2.25);
`mandacaru-paw/PBE.md` lists them. Light elements are built exactly as in
LDA. With semicore Na or Mg, converge the grid: PBE MgO and NaCl are 1.2 %
and 1.4 % long at grid_spacing = 0.25 Å and within 0.3 % of all-electron PBE at 0.18 Å.
`MANDACARU_UPAW_PATH` is the one optional variable: UPAW-LCAO is generated on
demand without it (*Datasets* above).

A calculation reads the set of its functional -- `pbe-sr/` for
`method="dft"` with PBE, HSE06 or r2SCAN, `lda-sr/` for LDA and every other
method -- unless the calculator names another folder with `paw_dataset=`,
resolved against the family's own variable (`paw_dataset="pbe-fr"` is the
PBE set with spin-orbit coupling):

```python
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis={"name": "PAW-LCAO", "size": "DZP"},
                       paw_dataset="lda-fr")   # $MANDACARU_PAW_PATH/lda-fr/

atoms.calc = Mandacaru(method="dft", xc="pbe",
                       basis={"name": "PAW-LCAO", "size": "DZP"},
                       paw_dataset="lda-sr")      # LDA datasets under PBE: warns
```

On the command line the same folder is `--paw-dataset lda-sr`.

A per-element basis mapping has each of its library
entries pointed at the same folder name; with the `lda-sr` set every entry
reads the scalar-relativistic LDA set unchanged. `paw_dataset=` takes the
name of one folder inside the checkout, not a path; a missing folder raises
`LibraryPathError` listing the ones present, and naming a folder with a basis
that reads no library is refused (the functional's own set is simply unused
by such a basis).

Each family's own module is the loader —
`get_paw(symbol, paw_dataset, xc)`, `get_upaw(symbol, paw_dataset, xc)` — both
accepting `paw_dataset=None` to fall back to the library variable and its
default `lda-sr/` set (`get_paw(symbol, xc="pbe")` reads `pbe-sr/`); `available_elements(paw_dataset)` (from
`mandacaru.pseudopotentials.io`) lists what a dataset folder holds. The PAW-LCAO
datasets cover **every element with Z ≤ 92** (H through U),
generated from scratch by Mandacaru's own radial atomic solver, in LDA and
in PBE.

A folder named for a set is a promise about its files: a dataset whose
functional or relativistic treatment is not the one the name says (a scalar
file copied into `lda-fr/`, which would silently drop the spin-orbit term)
is refused with the folder named; a folder with any other name is the
caller's own and is read as it is. A loaded dataset is cached for the
process by its file and that file's modification time and size, so a file
replaced on disk -- a set rebuilt under a running session -- is read again.

A calculation that needs a variable that is unset, or that names something
that is not a folder, raises `LibraryPathError`
(`mandacaru.pseudopotentials.environment`) naming the `--set-*` command that
fixes it; the command line prints it and exits 2. A basis option
`paw_dataset=...` (unlike the calculator's) bypasses the variable for one run,
and wins over the calculator's folder: it names the folder
holding the `<Symbol>.parquet` files itself, with no functional subfolder,
and the loaders raise `FileNotFoundError` rather than `LibraryPathError` when
that folder is missing or empty.

UPAW-LCAO has its own repository, `mandacaru-upaw`, but it is the one
optional library: without `MANDACARU_UPAW_PATH` a missing dataset is
generated on demand (*Datasets* above) rather than raising
`LibraryPathError`. A library built with `mandacaru-build --pp UPAW
--install` goes to `$MANDACARU_UPAW_PATH/lda-sr/`, inside that checkout. The
PAW-LCAO library is regenerated or extended the same way,
with `mandacaru-build --pp PAW --all --install`
(or `build_paw_library` from Python, above).

### The solid-state gate

A dataset that passes every atomic check can still fail to hold its own crystal
together, so `test/pseudopotentials/test_solid_state_gate.py` scans the
elemental crystal (or a proxy at its atomic volume) over lattice constants from
-12 % to +12 % at a fixed number of grid points per primitive vector, with the
Fourier-filter cutoff **pinned** at the value of the coarsest point (the default
cutoff follows the grid, which changes the basis function itself along the
scan and can create or hide a minimum), and asks one question: is there an
interior energy minimum? Alongside it run rule tests (the OK < 3 %, SUSPECT
3-6 %, FAIL flags and the quartic and Birch-Murnaghan fits against analytic
curves), a library check that every dataset declaring a semicore shell has its
channel, and a bound (1e3 Ha) on the one-center integrals, which blow up when a
scattering reference grows outside the sphere. The library checks need
`MANDACARU_PAW_PATH`; `-m slow` adds three SZP sentinels (Al, V, Ta) and the
bound for Cu. The sweep over every element of `lda-sr` and `pbe-sr` (DZP, seven
points) and the one-center bound for every dataset (about 20-30 s per element)
take hours, so they run only with `MANDACARU_LIBRARY_AUDIT=1`, on an idle
machine:

```bash
MANDACARU_LIBRARY_AUDIT=1 MANDACARU_PAW_PATH=/path/to/mandacaru-paw OMP_NUM_THREADS=2 \
  python -m pytest test/pseudopotentials/test_solid_state_gate.py -m "slow or not slow" \
  -k "every_element or unbounded_one_center" -p no:cacheprovider -rxX --durations=20
```

Elements listed in `KNOWN_FAILURES` (one table in the test module, with the
reason and the date of the measurement) are expected failures: an `XPASS` in the
report means a dataset has been fixed and its row should be deleted. The
sweep checks that a minimum exists, not that the lattice constant is right; for
that, compare with the all-electron value at the converged settings.

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

Every file records its `family` (format version 2); a file without the
field, or without `radial_tables`, is refused rather than guessed at — this
build reads only the PAW-LCAO table layout (UPAW-LCAO sharing it).

```{note}
Saving is lossless and idempotent: `load` then `save` returns the same tables.
The library is decimated once at generation time
(`build_paw_library`/`build_upaw_library(..., stride=...)`,
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
          basis={"name": "PAW-LCAO", "size": "DZP"},
          grid_spacing=0.15)
```

Sizes are the same names as for [the NAO family](basis_sets.md). The default
is `"SZ"` (the minimal valence set), because the valence-only space is usually
already at the edge of what a state-vector simulator can hold. The dry run
counts the valence functions of the family and size you ask for:

```console
$ mandacaru H2O --cell 8 --basis PAW-LCAO --basis-option size=DZP --dry-run
```

## Confined orbitals: `energy_shift`

Without confinement the first zeta of a pseudopotential basis is the dataset's
bound smooth partial wave: the valence orbital of the **free** atom. It has no
range of its own -- a lithium 2s still carries 10⁻⁴ of its norm beyond 14 Bohr
-- and it is more diffuse than the same orbital inside a molecule. LCAO codes
commonly use the orbital of the atom in a confining potential instead,
and name the confinement by what it costs: the **energy shift** is how far the
confined eigenvalue lies above the free one, and it *defines* the cutoff radius
of each orbital. One number gives every channel of every element a radius that
is tight for a compact orbital and generous for a diffuse one.

```python
Mandacaru(method="adapt-vqe",
          basis={"name": "PAW-LCAO", "size": "DZP", "energy_shift": 0.1},
          grid_spacing=0.20)
```

`energy_shift` is in **eV** (as for [the NAO family](basis_sets.md)) and is accepted by every pseudopotential family:
`"PAW-LCAO"` and `"UPAW-LCAO"`, together with
`confinement_potential` and `polarization_method`. **The default is 0.1 eV**
for both
-- so a plain `basis="PAW-LCAO"` is a
confined basis.
`None`, `False` or `0` switch the confinement off and restore the free-atom
orbitals; every PAW-LCAO energy quoted in this guide outside this section was
computed that way, with confinement off rather than at the default. A `{symbol: eV}`
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
`ε(r_c) − ε_free = energy_shift`. The overlap
is the identity for UPAW-LCAO (`q = 0`).

The extra zetas and the polarization shell are then split from the confined
orbital, so the whole basis of an atom shares its range.

### A common LCAO recipe

The confining potential is the smooth one of Junquera *et al.* (PRB **64**,
235111),

```{math}
v_{conf}(r) = \frac{A}{r_c - r}\exp\Bigl(-\frac{r_c - r_i}{r - r_i}\Bigr)
\quad (r_i < r < r_c),
```

flat at `r_i` and divergent at `r_c`, so the orbital vanishes at its cutoff with
every derivative -- a hard wall leaves a kink there, and a real-space grid pays
for kinks. The defaults follow a widely used LCAO basis recipe, so that an
ADAPT-VQE calculation here and an LCAO-DFT calculation elsewhere can hold the
basis recipe fixed:

| | option | default here |
| :--- | :--- | :--- |
| size | `"size": "DZP"` | `"SZ"` |
| energy shift | `"energy_shift": 0.1` | 0.1 eV |
| confining potential | `"confinement_potential": (12.0, 0.6)` | `(12.0, 0.6)` |
| split-valence zetas | `"zeta_split": "first_zeta"` (tail norms `(0.16, 0.3, 0.6)`) | as written |
| polarization function | `"polarization_method": "gaussian"` (quasi-Gaussian) | when confined |
| PAW-LCAO dataset | Mandacaru's own datasets (LDA) | -- |

So the double-zeta-polarized basis of that recipe needs only its size:

```python
Mandacaru(method="adapt-vqe",
          basis={"name": "PAW-LCAO", "size": "DZP"},
          grid_spacing=0.20)
```

and Mandacaru's own earlier construction -- free-atom orbitals, the
`last_zeta` split, the `r · R_outer` polarization shell -- is

```python
Mandacaru(method="adapt-vqe",
          basis={"name": "PAW-LCAO",
                 "size": "DZP",
                 "energy_shift": None,
                 "zeta_split": "last_zeta"},
          grid_spacing=0.20)
```

Only the dataset cannot be made identical: each code pseudizes its own atom.
That leaves the radii close rather than equal. For the recipe above (LDA,
double zeta plus polarization, 0.1 eV), an independent LCAO basis generator and
Mandacaru give, in Bohr:

| | reference generator | Mandacaru |
| :--- | ---: | ---: |
| H 1s cutoff / dz split radius | 6.64 / 3.67 | 6.68 / 3.68 |
| O 2s cutoff / dz split radius | 4.38 / 2.30 | 4.37 / 2.37 |
| O 2p cutoff / dz split radius | 5.34 / 2.89 | 5.35 / 2.87 |
| H p-polarization `r_char` | 1.395 | 1.396 |
| O d-polarization `r_char` | 1.125 | 1.128 |

(pinned by `test/pseudopotentials/test_paw_energy_shift.py::TestTheReferenceRecipe`).
Compare trends and differences between the two methods rather than absolute
energies, use LDA datasets on the other side, and remember that Mandacaru's
wave-function Hamiltonian
has the bare Coulomb interaction, inside the augmentation spheres too -- the
functional enters its datasets' partial waves and frozen core only.

#### Split-valence zetas: `first_zeta` or `last_zeta`

The two schemes share the split-valence polynomial -- inside a split radius the
orbital is replaced by `r^l (a − b r²)` matched in value and slope, and the
difference is the new zeta -- but **not the convention for what and where to
split**, and the numbers are not comparable digit for digit:

| | `"first_zeta"` (**default**): tail norms | `"last_zeta"`: split norm |
| :--- | :--- | :--- |
| what the number measures | the **norm** of the tail outside the split radius | the **squared norm** of that tail |
| default | `(0.16, 0.3, 0.6)` for the 2nd, 3rd, 4th zeta | `0.15`, halved for each further zeta |
| what each zeta splits | always the **first** zeta | the **previous** zeta |
| H 1s second-zeta radius (0.1 eV) | 3.68 Bohr | 2.52 Bohr |

A tail norm of 0.16 is a squared-norm fraction of 0.0256, so `first_zeta`'s
second zeta is considerably **longer-ranged** than one made with a split norm
of 0.15. Each scheme carries its parameter: `zeta_split` alone fixes the split
radii, and neither number is a basis option of its own.
`"zeta_split": "last_zeta"` selects the other scheme. The choice applies to
every family with a size hierarchy (`"PAW-LCAO"`, `"UPAW-LCAO"`
and the all-electron `"NAO"`), and the log's `[BASIS]` block names the scheme
in its `zeta_split:` line.

The scheme is part of the model. H₂ in PAW-LCAO-DZP (grid_spacing = 0.25), with free-atom
orbitals and the `r · R_outer` shell on both sides: `first_zeta` gives an
equilibrium distance of **0.760 Å** and `last_zeta` **0.711 Å**
(experiment 0.741, plane-wave PBE 0.750), while the `last_zeta` basis is about
0.5 eV lower in absolute energy -- its tighter second zeta adds more freedom
near the nucleus. The full default basis (`first_zeta`, the 0.1 eV confinement
and the Gaussian polarization shell) lands at **0.733 Å**. Every DZ/DZP/TZP energy quoted in this guide outside
this section was computed with `"zeta_split": "last_zeta"` (split norm
0.15) rather than the current `first_zeta` default; pass it to reproduce
them.

#### The polarization function

The default follows the confinement: `"gaussian"` wherever the orbital is
confined (so, by default, everywhere), and `"orbital"` for an unconfined
element -- the Gaussian takes its cutoff from the confined orbital, so it cannot
exist without one. `"polarization_method": "orbital"` raises the outermost valence orbital
by one unit of angular momentum, `r · R_outer(r)`, and adds one shell per
polarization count at `l_max + 1`, `l_max + 2`, ... `"gaussian"` is a
**quasi-Gaussian**,

```{math}
R(r) = r^{l}\Bigl[e^{-r^2/r_{char}^2} - (a - b r^2)\Bigr], \qquad r < r_{cut},
```

with `a`, `b` making the value and the slope vanish at `r_cut`. The shell
takes the **first angular momentum missing** among the valence channels
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
orbitals that were too diffuse for a molecule. H₂ (0.74 Å, 8 Å cell, grid_spacing = 0.25,
ADAPT-VQE to the basis FCI), in eV:

| `energy_shift` (eV) | `r_c` of H 1s (Bohr) | SZ | DZP | DZP, Gaussian polarization | DZP, `"zeta_split": "last_zeta"` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| off | -- | −30.2425 | −32.8842 | -- | −33.3847 |
| 0.01 | 8.97 | −30.3292 | −32.9375 | −33.2528 | −33.4102 |
| 0.1 | 6.68 | −30.7093 | −33.1037 | −33.3748 | −33.4845 |
| 0.3 | 5.59 | −31.1507 | −33.1911 | −33.4228 | −33.5108 |

The gain shrinks as the basis grows: a second zeta already supplies part of the
contraction the confinement provides. The compact Gaussian is a better
polarization function than `r · R_outer` here (−0.27 eV at 0.1 eV). The forces
differentiate the confined basis exactly as they do the free one (analytic
against finite difference agree to 5 meV/Å on this system).

Two limits are refused with the usable range in the message: a shift so large
that the confinement would cut into the augmentation sphere (`r_i` must stay
outside `r_cut`), and one so small that the radius runs past the dataset's
radial table.

## Diffuse functions: `diffuse`

No `energy_shift` makes a pseudopotential basis diffuse. The first zeta is at
most the free atom's orbital: the rms radius of hydrogen's 1s is 1.83, 1.94
and 1.97 Bohr at 0.1 eV, at 0.01 eV and unconfined. A response property (the
polarizability), a bound anion or a weakly bound complex needs the density
further out than that. The `diffuse` option **adds** functions there and leaves
the ordinary basis (`size`, `energy_shift`, the polarization shell) as it was:

```python
from ase.build import molecule
from mandacaru import Mandacaru

water = molecule("H2O")
water.center(vacuum=7.3)                 # the box must hold the shells (below)
water.calc = Mandacaru(method="rhf",
                       basis={"name": "PAW-LCAO", "size": "DZP",
                              "diffuse": True},
                       grid_spacing=0.2)
```

It is off by default for every family and geometry, and `False`, `None` and
`0` all mean off. PAW-LCAO and UPAW-LCAO both accept it. Diffuse functions
belong to the trial space only: they are projected on the dataset's projectors
and enter the augmented overlap `S + C Q C†` like every other function, and
they add no projector, occupation or dataset change.

### Construction

Each diffuse shell is the quasi-Gaussian of the polarization function,

```{math}
R(r) = N\,r^{l}\Bigl[e^{-\alpha r^2} - (a - b r^2)\Bigr], \qquad r < r_{cut},
```

whose value and slope vanish at `r_cut`. Its width and its support are set
separately:

* **The width** is an rms radius, `sqrt(<r²>)` of the cut function:
  `rms_ratio` times the rms radius of the atom's **free** valence orbital of
  the same `l`. That orbital is the dataset's stored bound wave, unconfined, so
  the scale does not depend on `size`, `energy_shift` or the polarization
  method. An `l` with no valence state (hydrogen's p, oxygen's d, lithium's p)
  takes the atom's outermost valence orbital. `rms_radius` (Å) gives the
  width directly instead.
* **The support** `r_cut` is where the *uncut* Gaussian `r^l e^{-αr²}` leaves
  `cutoff_norm` of its squared norm outside. For a Gaussian the tail fraction
  is the regularized gamma function `Q(l + 3/2, 2α r_cut²)`, and both the cut
  function's rms radius and `r_cut` scale as `α^{-1/2}`, so `α` and `r_cut`
  follow from the target in closed form.

`count` shells per channel follow a geometric sequence of rms radii, `ratio`
apart. Every shell is tabulated on **its own radial grid**: the dataset's
grid, continued past its 30-Bohr end when the support needs it (lithium's s
shell reaches 29.7 Bohr). It is Fourier-filtered like the rest of the basis
and then **Gram-Schmidt orthogonalized on its atom** against that atom's
functions of the same `l`, compact ones first. A shell that keeps less than
10⁻⁴ of its squared norm after the projection is dropped as redundant, and the
log records it as `rejected`. This step depends only on the atom, never on the
geometry, so it cannot make an energy scan jump. Dependence *between* atoms is
left to the Löwdin step, which warns below an overlap eigenvalue of 10⁻⁶ and
refuses at 10⁻¹⁰. Diffuse functions come after the polarization shells of
their atom, in the atom-major order that the populations rely on; a basis
function reports `kind == "diffuse"`.

| key | meaning | default |
| :--- | :--- | :--- |
| `shells` | `"present"` (every `l` of the atom's ordinary basis, valence and polarization, up to d), a string of letters (`"sp"`), or a list of `l` or letters | `"present"` |
| `count` | shells per `l` | 1 |
| `rms_ratio` | rms radius of the first shell over the free-atom valence rms radius of its `l` | 2.5 |
| `ratio` | factor between the rms radii of successive shells (`count > 1`) | √3.5 = 1.87 |
| `rms_radius` | the first shell's rms radius in **Å**, instead of `rms_ratio` | -- |
| `cutoff_norm` | squared norm of the Gaussian tail beyond `r_cut` | 10⁻⁴ |

`True` is the default recipe, an `int` is that many shells per `l` with it,
and a dict sets any of the keys above. A `{symbol: spec}` mapping, with an
optional `"*"` default, gives elements different recipes. The per-element basis
form takes it inside each element's dict:

```python
basis={"O": {"name": "PAW-LCAO", "size": "DZP", "diffuse": True},
       "H": {"name": "PAW-LCAO", "size": "DZP"}}     # H: no diffuse shells
```

An element that does not write `diffuse` gets no diffuse shells. Leaving an
option out means its default, and here the default is off.

The default recipe (one shell per `l` present, `rms_ratio` 2.5, a 10⁻⁴ tail)
was chosen from the measurements below. Compared with 2.0, a ratio of 2.5 binds
the fluorine and hydrogen anions better (F 1.17 against 1.00 eV, H⁻ 0.59
against 0.54), brings water and HF (and NH₃ at PBE) closer in polarizability and doubles
the smallest overlap eigenvalue. It does worse on the hydrogen-rich H₂ and CH₄. At 3.0 a gap opens between the valence
orbital and the shell, and chlorine's affinity falls from 2.36 to 2.25 eV.
The diffuse primitives of aug-cc-pVDZ sit at 2.2-2.8 times the same free-atom
radii. Shells on the valence `l` alone do little for the polarizability: the
response needs the shell *above* the occupied `l` (oxygen's d, hydrogen's p).
A 10⁻⁴ tail instead of 10⁻⁶ shortens every support by 14 % and moved the H⁻
and F affinities by 0.01 eV.

### The box

A molecule's box is its cell, and nothing pads the box from the basis. A
diffuse shell that the cell cuts is a different, more compact function, and
the energy it gives is wrong rather than merely unconverged. H₂ (SZ with the
default shells, RHF) comes out 60 meV *lower* in 3 Å of vacuum than in 7.5 Å,
so a lower energy does not show that a box is large enough. Once the cell holds
the shells, 7.5 Å and 9 Å agree to 10⁻⁹ eV.
At the default recipe, with the DZP shell set (`shells="present"`):

| element | shells (DZP) | largest `r_cut` (Bohr) | vacuum needed (Å) |
| :--- | :--- | ---: | ---: |
| H | s p | 13.64 | 7.2 |
| Li | s p | 29.73 | 15.7 |
| C | s p d | 12.33 | 6.5 |
| N | s p d | 10.52 | 5.6 |
| O | s p d | 9.09 | 4.8 |
| F | s p d | 8.00 | 4.2 |
| Cl | s p d | 11.83 | 6.3 |

Every run checks this. Each diffuse function's norm on the grid is compared
with its radial norm, and a function that keeps less than `1 − 10⁻⁴` of it
raises a `RuntimeWarning`. The `[BASIS]` block's `diffuse_box` table lists each
atom's largest `r_cut` against its distance to the nearest face of the grid,
in Bohr and Å, with the worst grid norm. A larger box is the cost: water with
the default shells needs 0.47 M grid points at 0.2 Å, against 0.16 M for DZP
in 5 Å of vacuum.

### Active spaces

Every diffuse function is one more spatial orbital, so two more qubits. With
the default recipe on a DZP basis, a hydrogen or lithium atom gains 4
functions (s and p: 8 qubits), and a C, N, O, F or Cl atom gains 9 (s, p and d:
18 qubits). Water goes from 23 to 40 functions, from 46 to 80 unreduced qubits.
The basis cost is paid before the active space and is not reduced by it: the
SCF, the integrals and the MP2 ranking all see the full basis.

Diffuse shells make the *lowest* virtual orbitals, and the default
`active_space` ranking, `"energy"`, keeps the lowest first. A truncated
register would then fill with diffuse orbitals whatever they contribute, so a
basis with diffuse functions **refuses** an `"energy"`-ranked truncation and
names the remedy: rank the virtuals by occupation with `"method": "mp2"` (or
`"natural"` for an open shell). An untruncated space and explicit orbital
indices are not ranked, so they are accepted. For H⁻, where the diffuse
virtuals carry the extra electron, the MP2 ranking keeps fewer of them and
recovers more of the correlation than the energy ranking does:

H⁻ electron affinity (eV): the two-electron FCI energy of the anion
(what UCCSD or ADAPT-VQE reach), the H atom exact at UHF, `grid_spacing` 0.25
(0.3 with two shells), box per the table above. The kept-diffuse column is the
Löwdin population of the active virtual orbitals on the diffuse functions, in
orbitals.

| basis | full | `"mp2"` 12 / 16 / 20 qubits | `"energy"` 12 / 16 / 20 qubits | diffuse kept, `"mp2"` / `"energy"` at 12 qubits |
| :--- | ---: | :--- | :--- | :--- |
| DZP, no diffuse (5 functions) | −0.66 | -- | -- | -- |
| QZ2P unconfined (12) | 0.26 | 0.24 / 0.24 / 0.24 | −0.31 / −0.16 / 0.10 | 0 / 0 |
| DZP + `diffuse` (9) | 0.59 | 0.57 / 0.58 / -- | 0.38 / 0.52 / -- | 2.3 / 3.8 |
| QZ2P + `diffuse` (14) | 0.61 | 0.58 / 0.59 / 0.61 | 0.32 / 0.46 / 0.59 | 1.9 / 3.6 |
| QZ2P + `diffuse=2` (18) | 0.65 | 0.61 / 0.62 / 0.64 | −0.04 / 0.02 / 0.38 | 1.9 / 4.7 |
| all-electron FCI, cc-pVQZ / aug-cc-pVQZ | 0.18 / 0.74 | | | |

Hartree-Fock leaves H⁻ unbound in every basis (−0.33 to −0.40 eV with the
diffuse shells), so all of the binding is correlation. The MP2 ranking
recovers 96-99.5 % of that correlation contribution at 12-20 qubits, and the
energy ranking 30-98 %. The energy ranking also gets *worse* as diffuse shells
are added: at 12 qubits the second shell makes the anion unbound again
(−0.04 eV). ADAPT-VQE in the 16-qubit MP2 space of DZP + `diffuse` gives
0.581 eV, the FCI value of that space. What is left against the
all-electron 0.74 eV is angular: hydrogen's basis stops at p here, while
aug-cc-pVQZ carries d, f and g on it. Some register sizes split a
degenerate p set, and Mandacaru warns about it; `"symmetry": True` keeps the
sets whole.

### Measured

All on DZP at 0.1 eV, `grid_spacing` 0.2 Å (0.3 for LiH), in a box that holds
the shells. Wave-function methods use `lda-sr/` with the exact spheres, PBE
uses `pbe-sr/`, and the kinetic operator is the spectral default. The references
are all-electron finite-field values (aug-cc-pVQZ), which serve only as
measured reference numbers; nothing in Mandacaru reads them.

Static polarizability, `alpha_iso` (a.u.), DZP → DZP + `diffuse` (reference):

| molecule | RHF | PBE |
| :--- | :--- | :--- |
| H₂O | 5.70 → 8.58 (8.64) | 5.98 → 10.40 (10.58) |
| HF | 2.94 → 4.93 (4.95) | 3.16 → 6.15 (6.17) |
| HCl | 8.59 → 15.77 (16.76) | 8.67 → 17.62 (18.29) |
| NH₃ | 9.01 → 12.77 (13.02) | 9.30 → 15.06 (15.40) |
| CH₄ | 12.89 → 15.35 (16.08) | 13.17 → 16.60 (17.52) |
| H₂ | 3.19 → 4.47 (5.19) | 3.21 → 4.99 (5.64) |
| LiH | 12.66 → 25.32 (24.94) | 16.81 → 34.57 (34.75) |

Without diffuse shells the confined DZP basis misses 20-53 % of every
polarizability. With them it comes within 5 % of the reference except for H₂ (−14 %
RHF, −12 % PBE), HCl at RHF (−6 %) and CH₄ at PBE (−5 %). The worst cases are
the hydrogen-rich ones: hydrogen's diffuse p shell, sized from its 1s, is
wider than the response needs. `rms_ratio` 2.0 does better there (H₂ 4.93
RHF, 5.40 PBE; CH₄ 15.82, 17.06) and worse on water, HF and the anions, so the
default stays at 2.5. Water at TZ2P + `diffuse` gives 8.56 (RHF), so the size
is not what limits it. Halving the field moves PBE water by 10⁻³ a.u.
Unconfined DZP and TZP, with no diffuse shells, give 8.15 for RHF water, and
unconfined TZP gives 9.49 for PBE: the free-atom orbitals recover most of
the RHF response but only three quarters of the PBE one, and the diffuse
shells close what is left.

Electron affinities (eV), UHF ΔSCF for F and Cl, FCI for H⁻:

| | F | Cl | H⁻ |
| :--- | ---: | ---: | ---: |
| DZP | −0.04 | 1.35 | −0.66 |
| DZP + `diffuse` | 1.17 | 2.37 | 0.59 |
| QZ2P + `diffuse` | -- | -- | 0.61 |
| reference (aug-cc-pVQZ; cc-pVQZ) | 1.18 (0.84) | 2.37 (2.28) | 0.74 (0.18) |

The confined basis cannot hold an anion; the diffuse shells can. F carries a
grid uncertainty of about 0.04 eV per 0.05 Å of `grid_spacing`, and at h → 0
DZP + `diffuse` extrapolates to about 1.10 eV, TZ2P + `diffuse` to 1.21.

Water dimer (S22, PBE, kcal/mol): DZP raw −8.24, counterpoise −4.56;
DZP + `diffuse` raw −5.41, counterpoise −4.72; reference −4.98 / −4.95. The
diffuse shells cut the superposition error from 3.7 to 0.7 kcal/mol. The
counterpoise monomers carry their partner's diffuse shells on the ghosts.

Conditioning: water's smallest diagonal-normalized overlap eigenvalue is
6.3·10⁻³ at DZP, 4.1·10⁻⁴ with the shells and 1.6·10⁻⁴ at TZ2P with them
(CH₄: 8.8·10⁻⁵), all well above the 10⁻⁶ warning. Analytic forces match a
central difference (H₂, `test_paw_forces.py`). An RHF scan of HF from 0.80 to
1.05 Å stays as smooth with the shells as without them (quartic-fit residual
0.54 against 0.55 meV), and the bond moves by −0.002 Å.

Cost, water, `grid_spacing` 0.2: the PBE `polarizability()` (seven runs)
takes 149 s and 2.1 GB with the shells, against 46 s and 0.8 GB without; the
PBE water dimer takes 250 s and 3.8 GB, against 86 s and 1.3 GB. LiH needs
16 Å of vacuum (1.3 M points at 0.3 Å).

### Not for crystals

A periodic geometry with `diffuse` is refused with a `ValueError`. In a solid,
a diffuse shell is nearly linearly dependent on the neighbors' valence
functions. Its Bloch sum also runs over several times the images of a confined
orbital: for bcc lithium, an estimated 330 images per function for a 20-Bohr
shell, against 70 for the confined 2s. Slabs, where diffuse functions would describe
the vacuum side, are a separate extension.

## Fourier filtering

Every integral here is a sum over a grid of spacing `grid_spacing`, so the grid can only
represent wave-vectors up to its Nyquist value `k_N = π/h`. Whatever a basis
function carries above `k_N` is **aliased**: it does not vanish, it folds back
with a phase that depends on where the function's center falls between two
nodes. Translate a molecule by a fraction of `grid_spacing` and the energy changes — the
*egg-box* — and `atoms.get_forces()` faithfully differentiates that artifact.

The cure is to remove those components from the **radial functions**, once,
before they are ever sampled (Anglada & Soler, PRB 73, 115122). Mandacaru does it per angular-momentum channel with the spherical
Bessel pair, multiplying `F_l(k)` by a raised-cosine window that is 1 below
`0.75 k_c` and 0 at `k_c`, then switching the result off smoothly beyond the
function's own support so a split zeta stays short-ranged
(`mandacaru.basis.filtering`).

```python
Mandacaru(method="adapt-vqe",
          basis={"name": "PAW-LCAO", "size": "DZ"},          # filtered: the default
          grid_spacing=0.20)

Mandacaru(method="adapt-vqe",
          basis={"name": "PAW-LCAO", "filter_cutoff": False},  # raw basis
          grid_spacing=0.20)

Mandacaru(method="adapt-vqe",
          basis={"name": "PAW-LCAO", "filter_cutoff": 800.0},  # in eV
          grid_spacing=0.20)
```

`filter_cutoff` takes `"auto"` / `True` (cutoff tied to the grid,
`k_c = π/grid_spacing`), a
positive **kinetic-energy cutoff in eV** (`k_c = sqrt(2E)` in atomic units),
or `False`. Anything else raises at construction. It is **on by default for
`PAW-LCAO` and `UPAW-LCAO`** — declared once per family as
`FamilySpec.default_options`, so a new family states its own and the drivers
need no edit. `filter_cutoff=False` reproduces the unfiltered basis byte for
byte. The measurements in the rest of this section were made with
`kinetic="fd"`, the finite-difference stencil.

**The `auto` cutoff is resolved once per calculator.** A crystal rebuilds its
grid whenever its cell changes, and a node count that steps, or a cell that
stretches at a fixed count, moves `π/h`; a basis filtered at each cell's own
`π/h` is a different basis at every cell, which acts as a spurious pressure
(lattice constants of the transition metals came out 2-5 % small). So the
first crystal build of a `Mandacaru` calculator resolves `"auto"` from its own
grid, exactly as before, and every later cell of that calculator (a cell-filter
relaxation, a stress or `ase` strain loop, a lattice scan that reuses the
calculator by setting `grid_spacing`, the strained cells of the piezoelectric
tensor) keeps that cutoff. A single point is unchanged. `calc.filter_cutoff_ev`
reports the held value (`None` before the first build, and for an explicit
`filter_cutoff` or `False`, which are never held), `calc.release_filter_cutoff()`
forgets it, and it is dropped by itself when the elements change. The run log's
`[BASIS]` block (its only owner) marks a held value. A scan that builds a
*new* calculator per cell still resolves `"auto"` per cell: give the cutoff in
eV, taken from the coarsest cell, to keep one basis; on one calculator, start
a scan at its coarsest (largest) cell, since the first build sets the cutoff. A molecule needs none of
this: its grid is its fixed cell.

### Why the cutoff sits exactly at Nyquist

Remove what the grid cannot carry, keep everything it can. The measurement
agrees: water/PAW-LCAO-SZ, peak-to-peak RHF energy over one grid period under a
rigid shift along (1,1,1), against the rise in the energy itself.

| `k_c / k_N` | ripple, `grid_spacing=0.25` | cost, `grid_spacing=0.25` | ripple, `grid_spacing=0.20` | cost, `grid_spacing=0.20` |
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

Per term at grid_spacing = 0.25, the ripple goes (unfiltered → filtered): kinetic
367.5 → 0.05 meV, local pseudopotential 1608.3 → 21.0, electron repulsion
1321.5 → 7.1. The total is far smaller than its parts because they partly
cancel. The nonlocal and compensation terms do not appear because they never
aliased: PAW-LCAO evaluates both by atom-centered quadrature, not on the grid.

**What is left is the local potential.** At grid_spacing = 0.25 it is 21.0 of the
remaining 27.1 meV, and at grid_spacing = 0.20 it is 5.8 of 6.0 — the filter band-limits
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
| grid_spacing = 0.25 | 2.394 eV/Å | **0.160** | ~2.9 |
| grid_spacing = 0.20 | 1.158 | **0.045** | ~2.4 |

The analytic gradient is still the derivative of the calculator's own energy:
central-difference agreement is 1.0e-3 (H₂, grid_spacing = 0.25) and 7.5e-4 eV/Å (water,
grid_spacing = 0.25) with the filter on, the same as without it.

The reported energy goes **up** — water/PAW-LCAO-SZ by 832 meV at grid_spacing = 0.25 — and
most of that is not a loss. Hold the cutoff fixed at 602 eV and refine the
grid: the filtered-to-unfiltered gap is 530 / 422 / 294 / 237 meV at
grid_spacing = 0.25 / 0.20 / 0.16 / 0.13 (measured with `kinetic="fd"`), close to linear in `grid_spacing`. It is the *unfiltered*
basis's own grid error — the finite-difference Laplacian under-estimates the
kinetic energy of a function the grid cannot resolve, so an unfiltered orbital
is reported too low. On H₂, where both bases are resolved by grid_spacing = 0.12, the gap
is 2 meV and has the opposite sign.

And what a user actually reads off — geometry and binding — barely moves,
because the shift is nearly a constant offset that cancels in differences
(exact sector ground states, PAW-LCAO-SZ):

| | d_eq unfiltered | d_eq filtered | D_e unfiltered | D_e filtered |
|---|---|---|---|---|
| H₂ (grid_spacing = 0.25)  | 0.9168 Å | 0.9167 Å | 2.801 eV | 2.807 eV |
| LiH (grid_spacing = 0.30) | 1.6536 Å | 1.6538 Å | 1.879 eV | 1.876 eV |
| OH (grid_spacing = 0.20)  | 1.0423 Å | 1.0442 Å | 3.466 eV | 3.429 eV |

OH's total energy moves 379 meV while its bond moves 0.002 Å and its binding
energy 0.037 eV.

```{note}
**Filtering changes the basis, so it changes the numbers.** It is a modeling
choice, not a numerical detail: the filtered first zeta is no longer exactly
the pseudo-orbital the projectors were built from, so the atomic reference is
no longer reproduced exactly. It is on by default for PAW-LCAO and UPAW-LCAO
because `optimize_pseudo_waves` already minimizes the kinetic energy beyond
`q_cut` and the filter has little left to take. Set `filter_cutoff=False` to
compare against an unfiltered result.
```

## Limits

The residual force on an isolated atom is still ~30 eV/Å at `grid_spacing = 0.10 Å`. That
remainder is basis-set incompleteness — a minimal valence s+p shell per atom —
not the core, and it *shrinks* with grid refinement. A polarized multiple-zeta
basis addresses it directly; see [Basis sets](basis_sets.md).

```{note}
**Relaxation works, and oxygen needs a finer grid than lithium.**
LiH relaxes cleanly: `examples/relaxation/02_lih_relaxation.py` takes five
BFGS steps from 2.0 Å to d = 1.7383 Å, final fmax 0.014 eV/Å (2026-10-13,
the spectral kinetic default and the exact one-center spheres; September's
1.6876 Å was measured with `kinetic="fd"` and the frozen spheres),
and so do H₂, OH, CH and H₂O.
Water from a 90°, 1.0 Å start converges in **three BFGS steps** (grid_spacing = 0.16 Å,
PAW-LCAO-SZ, 123 s, measured with `kinetic="fd"`) to d = 0.992 Å and an angle of 117.3°, against 0.9572 Å and
104.52° in experiment: the bond length is good, and the 13° on the angle is
the minimal valence s+p shell, not the gradient — a polarized basis is what
addresses it (see [Basis sets](basis_sets.md)).

* **Check that the energy binds before relaxing.** A symmetric O–H scan at
  grid_spacing = 0.16 Å (`kinetic="fd"`) gives −480.80, −481.92, −482.20, **−482.29**, −482.21, −481.63 eV
  at 0.85, 0.92, 0.96, **1.00**, 1.05 and 1.15 Å — a clean minimum. (Earlier
  releases did *not* bind water: the compensation charge was missing its
  electron–ion attraction, and every bond to a p-valence atom came out
  repulsive. If you are on an older version, scan before you relax.)
* **Egg-box.** Rigidly translating a molecule on a frozen grid changes the
  energy, so the forces of a free molecule do not sum to zero. The net force is
  a faithful derivative of that non-invariant discretized energy, and it is the
  honest measure of whether a force is good enough for geometry. Unfiltered,
  for water it falls as `|sum F|` = 1.85, 0.86, 0.41, 0.099, 0.025 eV/Å at
  grid_spacing = 0.30, 0.25, 0.20, 0.16, 0.13 Å — oxygen then needs grid_spacing ≤ 0.16 Å for
  geometry, finer than the grid_spacing ≤ 0.25 Å that suffices for energies.
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
`pseudopotentials/test_paw.py` validates the PAW-LCAO family atomically (H,
Li, O) and on H₂ and LiH, adding the overlap, on-site-projection, compensation
and grid-stability checks (70 tests, 11.5 s, peak RSS 0.77 GB);
`pseudopotentials/test_families.py` covers the basis-name selector and its
aliases.
