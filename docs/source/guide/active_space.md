# Fitting a large basis on a small register

A basis set buys accuracy with virtual orbitals, and a qubit register pays for
them at two qubits each. Those are not the same currency, and `active_space`
is where they are exchanged.

Water in PAW-LCAO-TZP has 29 spatial orbitals — 4 occupied and 25 virtual —
which is 56 qubits after the parity reduction. The same molecule in PAW-LCAO-SZ
has 10. The first is unreachable on hardware and the second is a poor
description of the molecule, and the way out is not a third basis: it is a
**large basis with a small active space**. Basis-set quality lives in the
*shape* of the orbitals; correlation lives in the few of them the wavefunction
actually mixes.

```{code-block} python
calc = Mandacaru(method="adapt-vqe",
                 basis={"name": "PAW-LCAO", "size": "TZP"},
                 active_space={"orbitals": 8, "method": "mp2"})
```

```text
H2O / PAW-LCAO-TZP                     29 spatial orbitals    56 qubits
  + active_space={"orbitals": 10}      10 spatial orbitals    18 qubits
```

Unlike a `"frozen"` core, dropping virtuals works with a pseudopotential basis.
A valence-only basis has no core left to freeze — `active_space`'s `"frozen"`
is refused for it — but it has plenty of virtual orbitals to drop, and dropping
them is what fits the register.

## The three places an orbital can go

| | what happens to it | who may be one |
|---|---|---|
| **frozen** | replaced by its mean field: a constant core energy plus an effective one-body potential on the rest | doubly occupied only |
| **active** | represented by qubits | anything |
| **deleted** | dropped; nothing is folded in | virtual only |

Deletion needs no fold-in, and that is the whole reason it is cheap: an orbital
that is empty in the reference contributes neither to the core energy nor to the
effective potential, so deleting it costs exactly the correlation it would have
carried and nothing else.

An **occupied** orbital is never deleted. It would take its electrons with it,
which is a different molecule rather than a smaller correlation treatment, so
that request is refused by name and points at `active_space`'s `"frozen"` key
instead (e.g. `"frozen": [0, 1]`).

## One option, six keys

`active_space` is a single dict with up to six keys:

| key | meaning |
|---|---|
| `"frozen"` | the frozen-core approximation (below) |
| `"orbitals"` | how many orbitals reach the register, by count |
| `"threshold"` | how many virtual orbitals to keep, by an occupation criterion |
| `"method"` | how the kept virtuals are ranked |
| `"correlating_pairs"` | keep each active occupied orbital's correlating partner (default `False`) |
| `"symmetry"` | keep degenerate sets whole and the symmetries the target states need (default `False`) |

Omitting `active_space` (the default, `None`) keeps every orbital. At least one
of `"orbitals"`, `"threshold"` or `"frozen"` is required when the option is
given — `"method"` alone ranks the virtuals but does not say how many to keep,
so it is refused on its own.

## Saying how many: `"orbitals"`

`active_space`'s `"orbitals"` takes one meaning per type, so there is nothing to
disambiguate:

| spec | meaning |
|---|---|
| *(key absent)* | every orbital on the register (subject to `"frozen"` / `"threshold"`) |
| `12` | twelve spatial orbitals in total. Never removes an occupied orbital: the virtuals take whatever is left |
| `{"occupied": 3, "virtual": 9}` | keep three doubly occupied and nine virtual orbitals; the rest of the occupied space is frozen |
| `[0, 1, 4, 7]` | exactly these spatial MO indices |

## Saying which are worth keeping: `"threshold"`

A count fixes the register width. The other way round is to fix the *criterion*
and let the width follow, which is what you want when the question is "how small
can this get" rather than "what fits my processor":

```{code-block} python
calc = Mandacaru(method="adapt-vqe",
                 basis={"name": "PAW-LCAO", "size": "TZP"},
                 active_space={"threshold": 1e-3,   # or True, for the default
                              "method": "mp2"})
```

`"threshold"` keeps the virtual natural orbitals whose **occupation number**
(NOON) is at least the value given. `True` takes the default, `1e-3`. It needs a
`"method"` that computes occupations, so it is refused with
`"method": "energy"` (the default) — orbital energies are not occupations.

`"orbitals"` and `"threshold"` compose. Given both, the threshold decides which
orbitals earn a place and the count caps how many there is room for, and
whichever binds first wins:
`active_space={"orbitals": 8, "threshold": 1e-5, "method": "mp2"}` means "the
orbitals worth keeping, but never more than eight".

### Pick the number from the spectrum, not from intuition

**A virtual natural occupation is the small charge that correlation promotes into
an orbital — not an electron count.** It is of order 10⁻² at the very most, and
falls away fast. Measured MP2 spectra:

```text
LiH / PAW-LCAO-TZP, 11 virtual orbitals, 2.37e-2 electrons promoted in total
  1.33e-2  3.12e-3  3.00e-3  3.00e-3  4.80e-4  3.22e-4  3.22e-4
  1.10e-4  2.38e-5  4.62e-6  6.76e-7
```

| threshold | LiH / TZP | H2O / DZP |
|---|---|---|
| `2e-2` | **keeps nothing** | **keeps nothing** |
| `1e-3` | 4 of 11 · 94.8 % of the charge · 10 qubits | 9 of 19 · 96.1 % · 26 qubits |
| `1e-4` | 8 of 11 · 99.88 % · 18 qubits | 17 of 19 · 99.94 % · 42 qubits |
| `1e-5` | 9 of 11 · 99.98 % · 20 qubits | 19 of 19 · 100 % · 46 qubits |

So the useful range is roughly **1e-3 to 1e-5**, and `1e-3` is the default
because it is the knee of both curves — past it, each further orbital costs two
qubits and returns a few parts per thousand of the correlation.

```{warning}
A threshold like `0.02` sounds small and is **above the entire spectrum**: it
keeps no virtual orbital at all, leaving an active space that cannot correlate
anything and would return the Hartree-Fock energy through a variational solver.
Mandacaru refuses it by name and quotes the largest occupation it found, so the
threshold can be chosen from the data rather than guessed again.
```

A dry run cannot resolve a threshold — how many orbitals clear it is a property
of the second-order density, which a dry run does not compute. It therefore
reports the untruncated width as an explicit **upper bound** and says so, rather
than printing a number the run will not use. Pass `"orbitals"` as well if
you need the width known in advance.

On the command line:

```bash
mandacaru H2O --cell 12 --basis PAW-LCAO --basis-option size=TZP \
    --active-orbitals 10 --active-method mp2 --dry-run
```

`--active-orbitals` reads `10` as a total, `3,9` as the `occupied,virtual`
split, and `'[0,1,4,7]'` as an explicit list. `--active-threshold 1e-3` uses the
occupation criterion instead. These two flags and `--active-method` are combined
by `mandacaru` into the single `active_space` dict the calculator sees.

## Saying which — and why energy ordering is the wrong answer

`active_space`'s `"method"` ranks the **virtual** orbitals.

`"energy"` (the default)
: Canonical order, lowest orbital energy first. Free. It asks *which orbital is
  cheapest to excite into*.

`"mp2"`
: The eigenvalues of the second-order density's virtual block — the **frozen
  natural orbitals**. It asks *which orbital the correlated wavefunction
  actually occupies*, which is the question an active space is asking, and it
  answers it in a **rotated** virtual basis, so a few orbitals can gather up
  correlation that canonical ordering leaves spread thinly over many. Costs one
  MP2 calculation in the full virtual space. Open shells (odd electron counts,
  and any `n_alpha != n_beta`, such as triplet O₂) use the open-shell
  expression: separate alpha and beta Fock operators of the reference
  determinant, semicanonicalized per spin, with the singles that a
  non-stationary reference brings. Only the orbitals empty in both spins are
  ranked; the singly occupied ones stay occupied.

`"dlpno-mp2"`
: The same natural orbitals from **local** MP2, built **integral-direct**: the
  basis's two-electron tensor is never formed. See
  [Large bases: `"dlpno-mp2"`](#large-bases-dlpno-mp2) below. Closed shell.

`"natural"`
: The occupations of the **reference** natural orbitals. Open-shell references
  only — see the warning below.

The difference is not cosmetic. H₂ in 6-31G on a real-space grid has one
occupied and three virtual orbitals, with energies 0.156, 0.278 and 0.739 Ha.
The first-order amplitudes are

```text
t_00^aa  =  -0.0268   -0.0285   -0.0520
              (0.156)  (0.278)  (0.739 Ha)
```

The **largest** amplitude sits on the **highest** virtual, and there is a strong
off-diagonal coupling between the first and third. The leading natural orbital
is `-0.56 v1 + 0.83 v3`, with an occupation 6.5 times the next one; energy
ordering keeps `v1`, which carries the least of the three.

This is not a quirk of that basis. Diffuse functions produce virtuals that are
*low in energy and spatially wrong* for correlation — the orbital that
correlates an occupied orbital has to have its spatial extent, which puts it
high. **Orbital energy is a poor proxy for correlation importance in any basis
with diffuse or polarization functions**, which is every basis worth using a
large one for.

### What it recovers

Exact diagonalization in the particle-number sector, correlation energy
recovered against the untruncated basis:

| system | active / total | energy order | mp2 order |
|---|---|---|---|
| H₂ / 6-31G | 2 of 4 | 6 % | **92 %** |
| H₂ / 6-31G | 3 of 4 | 15 % | **99.7 %** |
| LiH / 6-31G | 3 of 5 | 9 % | **89 %** |
| LiH / 6-31G | 4 of 5 | 18 % | **97 %** |

And through the full ADAPT-VQE stack on the basis this was built for, LiH in
PAW-LCAO-TZP at 1.60 Å:

| active orbitals | qubits | ADAPT operators | error vs. the full basis |
|---|---|---|---|
| 12 (all) | 24 | 50 | — |
| 8, `energy` | 16 | 33 | 412 meV |
| 8, `mp2` | 16 | **13** | **1.4 meV** |
| 4, `energy` | 8 | 9 | 784 meV |
| 4, `mp2` | 8 | 6 | 138 meV |

At 16 qubits the second-order ranking reproduces the 24-qubit TZP energy to
1.4 meV — inside chemical accuracy — with 13 operators instead of 50. The
shorter ansatz matters as much as the narrower register: two-qubit gate count is
what sets the fidelity of a hardware run, and it falls with the operator count.
See {doc}`measurement_cost`.

```{note}
**Read those errors carefully.** Deleting a virtual orbital removes variational
freedom, so the *exact* energy of a truncated space is always above the
untruncated one and falls monotonically as orbitals return — by exact
diagonalization on LiH/TZP, +22.65, +3.69 and +0.24 meV at thresholds 1e-3, 1e-4
and 1e-5.

The errors in the tables above come from ADAPT-VQE runs, and at these widths the
truncation error is *smaller than the optimizer's own residual on the
untruncated run*, which has fifty-odd operators to converge against a handful.
So a truncated run can come out a few meV **below** the full one without
anything being wrong. Compare truncations by exact diagonalization, or with a
convergence tolerance tight enough that the solver is not the largest error in
the comparison.
```

```{warning}
`active_space={"method": "natural", ...}` carries **no information** for a closed-shell
reference and is refused for one. The RHF density is idempotent, so its natural
occupations are exactly 2 and 0 and every ordering of the virtuals is as good as
every other; returning the energy ordering under another name would be a no-op
wearing a physical label. Reference natural orbitals do say something for an
**open-shell (UHF)** reference, and that combination is allowed. `"mp2"` ranks
both: it is the selector that actually ranks a closed shell, and for an open
shell it measures correlation rather than the reference's own spin
polarization. For triplet O₂ in PAW-LCAO-SZP at the same eight orbitals, the
MP2-selected space is 278 meV lower in exact energy than the natural or energy
selection.
```

(large-bases-dlpno-mp2)=
## Large bases: `"dlpno-mp2"`

`"mp2"` itself is cheap. What it needs first is not: the RHF reference and the
MP2 amplitudes read the basis's full two-electron tensor, about $48 M^4$ bytes
at its peak and $M^2/2$ Poisson solves plus an $M^4$ contraction to build. On a
17 GB machine that stops near $M = 135$ functions, and benzene in
PAW-LCAO-DZP ($M = 108$) did not finish building it in 25 minutes. The MP2
step itself took under 10 ms on every molecule measured.

`"dlpno-mp2"` never forms that tensor:

1. **RHF with direct Fock builds.** $J$ is one Poisson solve of the density;
   $K$ is the potentials of the pair densities $\phi_i^*\chi_q$ for every
   occupied orbital, $oM$ solves. PAW-LCAO's compensation charges enter as a
   few matrix products. Same SCF, same orbitals to round-off.
2. **Local MP2 for the natural orbitals** (Pinski, Riplinger, Valeev and
   Neese, 2015). The active occupied orbitals are localized (Foster–Boys); the
   virtual space is spanned by projected atomic orbitals; each localized
   orbital gets a domain of atoms by differential overlap (above $10^{-2}$);
   every pair gets its pair natural orbitals (kept above occupation
   $10^{-8}$) and the local MP2 equations are solved in them. Only the
   exchange integrals of each pair's domain are computed.
3. **The active-space Hamiltonian from the active orbitals alone**: the frozen
   core from its own Coulomb and exchange, and the two-electron integrals of
   the $n$ active orbitals, $n^2/2$ Poisson solves.

Measured:

| | |
|---|---|
| Direct RHF vs tensor RHF | equal to $10^{-14}$ Ha (H₂O, NH₃, C₂H₄ DZP) |
| Full domains, no PNO cutoff | canonical MP2 exactly: energy $2\times10^{-10}$ Ha, density $3\times10^{-10}$ (C₂H₆ DZP) |
| Default domains and cutoff | 99.986% of the MP2 correlation energy (C₂H₆ DZP) |
| Domains by Löwdin population alone | 96.6% — why the differential-overlap criterion |
| C₂H₄ DZP, $M = 46$ | direct RHF 16.6 s, tensor RHF 16.8 s |
| C₆H₆ DZP, $M = 108$ | direct RHF 4.2 min; the tensor did not finish |
| C₆H₆ DZP, whole active-space build | 6.3 min, 2.5 GB peak, no tensor formed |

For a small molecule it is no faster than `"mp2"`. Above roughly 50 basis
functions it is the only one that finishes, and its memory is $M^2$ plus the
grid instead of $M^4$.

```{code-block} python
calc = Mandacaru(method="adapt-vqe",
                 basis={"name": "PAW-LCAO", "size": "DZP"},
                 active_space={"orbitals": 8, "method": "dlpno-mp2"})
```

Limits of this version:

- **Closed shell only**, and the requested `"frozen"` core is not correlated
  (frozen-core MP2, the usual practice).
- **Local, but not yet near-linear.** The pieces of a near-linear method are
  in place and on by default:
  - the SCF's exchange is solved on local boxes around Foster–Boys orbitals,
    keeping only the basis functions that overlap each one, then finished on
    the exact exchange, so the orbitals are the exact SCF's (C₁₆H₃₄ SZ:
    584 s against 866 s, same energy);
  - distant pairs are prescreened by a dipole estimate, the amplitude
    equations couple only nearby orbitals, and the pair exchange blocks use
    local boxes;
  - the basis and the atom-centered quadratures evaluate a function only
    where it is nonzero.

  What still grows faster than linearly: the PAW compensation-charge setup
  (an atom-pair loop), the dense $M$-length vectors DLPNO stores for every
  PAO and PNO, and the boxes themselves, which on these chains span about
  25 Å (a localized orbital's tail plus the 4–5 Å reach of each basis
  function). On the chains measured the gain is 1.5–1.9×; near-linear
  scaling needs molecules much longer than that, and the remaining setup
  work.
- **Forces** are the central difference of the rebuilt reduced Hamiltonian,
  every displaced geometry going through the same integral-direct build and
  selection: $2(3N-6)$ builds, each as expensive as the first. On H₂O DZP
  they equal the `"mp2"` active space's forces to $10^{-5}$ eV/Å and a
  total-energy difference along an internal direction to
  $3.5\times10^{-3}$ eV/Å. The Hellmann–Feynman/Pulay breakdown is not
  reported: it needs the full two-body tensor.

## Occupied orbitals are chosen by energy, always

`active_space`'s `"method"` never decides *which* occupied orbitals are frozen:
that is always the lowest-energy ones. Two things depend on that.

The first is bookkeeping with teeth: `"frozen": "auto"` resolves to "the
lowest so many MOs", which is only the chemical core while the orbitals are
energy-ordered. A selector that reordered the occupied block would leave those
indices naming different orbitals, and the run would silently freeze the wrong
ones. (If the incoming orbitals ever fail to diagonalize the Fock matrix, the
selection is refused rather than applied to labels it cannot trust.)

The second is physics: removing an occupied orbital is a far coarser
approximation than removing a virtual one, and it belongs to an explicit request
— `"frozen": "auto"`, `"frozen": [i, j]`, or the `"orbitals": {"occupied": ...}`
form — not to an automatic selector.

Once the core is settled, `"mp2"` does rotate the **active** doubly occupied
orbitals among themselves into natural orbitals of the same density. A
rotation inside the doubly occupied space leaves the reference determinant,
and so the Hartree-Fock energy, unchanged; it touches no frozen orbital, so
the frozen-core indices keep naming the core; and it makes every active
orbital's occupation an eigenvalue of the MP2 density, as the virtual ones
already were. Singly occupied orbitals are never mixed in, since that would
change an open-shell determinant. Frozen orbitals keep their canonical form,
so their occupations in the run log are diagonal elements of the density, not
eigenvalues.

## Correlating pairs: `"correlating_pairs"`

With `"correlating_pairs": True` every active occupied orbital keeps its
**correlating partner** active: the virtual orbital its electron pair is
promoted into. The weight of a pair is the first-order amplitude of the pair
excitation $i^2 \to a^2$, the excitation that turns a bonding pair into its
antibonding one,

$$
w_{ia} = \frac{|\langle ii|aa\rangle|}{F_{aa} - F_{ii}} ,
$$

an exchange-type integral, large when $i$ and $a$ occupy the same region of
space with a node between them, over the energy it costs. The pairs are the
one-to-one assignment of maximum total weight, so two occupied orbitals never
share a partner. It works the other way too: an occupied orbital the count
froze (the `{"occupied": n}` form) keeps its partner out of the register.
Orbitals named by `"frozen"` are the chemical core, not bonding orbitals, and
are never paired; nor is a singly occupied orbital, which has no electron pair
to promote.

For water in PAW-LCAO-DZP every pair comes out within one irreducible
representation, as a pair excitation must for the product to be totally
symmetric, though the assignment knows nothing about symmetry: the in-plane
O–H bonding orbital 1b2 pairs with the antibonding 2b2, the lone pair 1b1
with 2b1, and so on. A register needs at least one virtual orbital per active
occupied orbital for this; a count too small is refused, naming how many more
are needed.

## Symmetry: `"symmetry"`

With `"symmetry": True` the molecule's point group is found and every orbital
is classified by how it transforms (see
{mod}`mandacaru.algorithms.orbital_symmetry`). The active space then

- keeps **degenerate sets whole**: a π pair, an e pair or a t triple enters
  or leaves the register together. Splitting one breaks the molecule's
  symmetry, so the energy depends on which component was kept;
- refuses a frozen core that takes part of a degenerate occupied set;
- for a subspace method (`num_states=`), keeps the symmetries its excited
  states need. Those are not known before the states are solved for, so the
  target is the lowest `num_states - 1` single excitations of the canonical
  orbitals: for each, some active occupied and active virtual orbital must
  reach its symmetry.

A ground-state method targets the ground state only. `energy_levels` (excited
states by deflation) is called after the active space is built, so it warns
when asked for more states than the space was chosen for.

The point group is found from the atoms, the operations represented in the
basis numerically, and characters read from the orbitals. A linear molecule's
infinite group is represented by its order-16 (`Cinfv`) or order-32 (`Dinfh`)
subgroup on an 8-fold axis, which keeps σ, π, δ and φ apart. Labels follow
Mulliken; where the convention depends on orientation, the plane or axis with
the most atoms is $yz$ or $z$ and a planar molecule's normal is $x$, so
water's out-of-plane lone pair is 1b1 and ethylene's π orbital is 1b3u. Orbitals
are numbered within this basis, so with a pseudopotential the core is not
counted: water's first valence orbital is 1a1, not 2a1.

If the Hartree-Fock reference itself breaks the symmetry (a doubly occupied
and a virtual orbital in one degenerate set), the run is refused: no active
space can be chosen by a symmetry the determinant does not have.

### Both constraints keep the register's size

The virtual orbitals are filled unit by unit: a single orbital, or a whole
degenerate set. The units the constraints require go in first, then the rest
in rank order while they fit. A unit that does not fit is skipped, and a
smaller one further down may take its place. The register keeps the size the
count asked for, except when no whole set fits the last slots; then they stay
empty and the block says so. For LiH in PAW-LCAO-TZP on four orbitals:

```text
    symmetry: on (brought in [5]; displaced [3])
    ...
        index  occupancy  role     irrep    occupation
        ----------------------------------------------
        0      occupied   active   1sigma+  1.97614864
        1      virtual    active   2sigma+  0.01343098
        2      virtual    active   3sigma+  0.00313653
        3      virtual    deleted  1pi      0.00302569
        4      virtual    deleted  1pi      0.00302569
        5      virtual    active   4sigma+  0.00046910
```

Without the constraint orbital 3 is kept and its partner 4 deleted.

## Forces and remaining limits

**Nuclear forces.** Molecular PAW-LCAO and other atom-centered bases support
forces with a reduced active space. The force path rebuilds the reduced
Hamiltonian on the same frozen grid at symmetric nuclear displacements and
contracts it with the converged active-space RDMs. This includes the motion of
the selected orbitals, any MP2 or natural-orbital rotation, and the frozen-core
constant. The displaced orbitals are aligned to the original orbital gauge
before contraction. This costs two integral and SCF builds per Cartesian
coordinate, without additional VQE optimizations or hardware jobs. It requires
`force_method="rdm"` and `include_pulay=True`.

**Periodic stress and periodic active-space forces.** These remain unavailable
with deleted virtual orbitals.

**The plane-wave basis.** There the register *is* the cutoff, which
`energy_cutoff` sets; truncating the mean-field orbitals on top of it would be a
second, hidden cutoff with no convergence handle.

**Spin-orbit coupling.** With spin-orbit coupling the two spins of a spatial
orbital are not partners, so the molecular orbitals are the GHF spinors
instead, and the active space works on **Kramers pairs** of them. Spinors
$2k$ and $2k+1$ (by energy) form pair $k$, and the pair occupies the two slots
a spatial orbital would. `"frozen"` counts the lowest pairs and `"orbitals"`
the pairs kept, lowest first. The spinor energy is the only ranking:
`"method": "mp2"`, `"threshold"` and the occupied/virtual forms rank spatial
orbitals and are refused. On HI (PAW-LCAO-SZ, `directory="lda-dirac"`, 18
electrons in 10 pairs), freezing one pair costs 0.11 mHa, two pairs 0.11 mHa
and four pairs 0.23 mHa. Keeping only the occupied pairs leaves the GHF
determinant, 17.8 mHa above the full space. Forces are available with a
frozen core. With deleted pairs they are refused, because the
finite-difference path aligns spatial orbitals between geometries.

## What the density still does

Everything: the cube writer, the atomic charges and the natural orbitals all
work with a truncated space, and the electron count comes out exact. A deleted
virtual is empty in the model, so it contributes nothing and stays zero — but it
is neither frozen nor active, so the density path is told the active set
explicitly rather than deriving it as the complement of the frozen one.

## Reading it back

The terminal header reports the partition that was actually built in one line:

```text
frozen core              : none
active space             : 4 active, 8 deleted of 12 spatial orbitals
                           (virtuals ranked by mp2), MP2 E_corr -0.021895 Ha
                           in the full virtual space
```

The `output.txt` log gives it a block of its own, `[ACTIVE SPACE]`, written
only when orbitals were frozen or deleted. It lists every spatial orbital with
its reference occupancy (`occupied`, `singly` or `virtual`), its role
(`frozen`, `active` or `deleted`) and the selector's occupation number, so the
cut can be checked against the spectrum it was made on:

```text
[ACTIVE SPACE]
    method: mp2
    orbitals_requested: 4
    frozen_requested: none
    correlating_pairs: off
    symmetry: off
    spatial_orbitals: 12 (0 frozen, 4 active, 8 deleted)
    mp2_correlation_energy_Ha: -0.02189456
    occupation: eigenvalues of the MP2 one-particle density (natural orbitals); frozen rows are its diagonal, since frozen orbitals are not rotated
    orbitals:
        index  occupancy  role     occupation
        -------------------------------------
        0      occupied   active   1.97614864
        1      virtual    active   0.01343098
        2      virtual    active   0.00313653
        3      virtual    active   0.00302569
        4      virtual    deleted  0.00302569
        5      virtual    deleted  0.00046910
        ...
```

Rows 3 and 4 are worth a look: they have the same occupation -- the two
components of a degenerate π pair -- and a four-orbital register keeps one and
deletes the other. That is what `"orbitals": 4` asks for, but it breaks the
molecule's symmetry; `"symmetry": True` (above) prevents it. With either
constraint on, the block gains an `irrep` column (and a `point_group` line) or
a `partner` column. See {doc}`run_output` for the rest of the log.

The reported correlation energy is the **untruncated** one, on purpose: it is
what the truncation is measured against. In code it is on the integral object as
`calc.solver._gradient_context["active_space"]`, an
{class}`~mandacaru.algorithms.active_space.ActiveSpace`.

A dry run reports the register width without computing any integrals. It cannot
know *which* orbitals a selector will keep — that needs the integrals — but the
width does not depend on the choice, so the number is exact (with one
exception: `"symmetry"` may leave slots empty when no whole degenerate set
fits them, and only the integrals show that):

```text
  basis functions   : 29  (O:17, H:6, H:6)
  deleted virtuals  : 19 spatial orbital(s), ranked by mp2
  active orbitals   : 10 spatial / 20 spin
  QUBITS REQUIRED   : 18
```
