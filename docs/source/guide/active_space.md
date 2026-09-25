# Fitting a large basis on a small register

A basis set buys accuracy with virtual orbitals, and a qubit register pays for
them at two qubits each. Those are not the same currency, and `active_orbitals`
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
                 active_orbitals=8, active_selection="mp2")
```

```text
H2O / PAW-LCAO-TZP        29 spatial orbitals    56 qubits
  + active_orbitals=10    10 spatial orbitals    18 qubits
```

Unlike `frozen_core`, this works with a pseudopotential basis. A valence-only
basis has no core left to freeze — `frozen_core` is refused for it — but it has
plenty of virtual orbitals to drop, and dropping them is what fits the register.

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
that request is refused by name and points at `frozen_orbitals` instead.

## Saying how many

`active_orbitals` takes one meaning per type, so there is nothing to
disambiguate:

| spec | meaning |
|---|---|
| `None` | every orbital on the register (the default) |
| `12` | twelve spatial orbitals in total. Never removes an occupied orbital: the virtuals take whatever is left |
| `{"occupied": 3, "virtual": 9}` | keep three doubly occupied and nine virtual orbitals; the rest of the occupied space is frozen |
| `[0, 1, 4, 7]` | exactly these spatial MO indices |

## Saying which are worth keeping: `active_threshold`

A count fixes the register width. The other way round is to fix the *criterion*
and let the width follow, which is what you want when the question is "how small
can this get" rather than "what fits my processor":

```{code-block} python
calc = Mandacaru(method="adapt-vqe",
                 basis={"name": "PAW-LCAO", "size": "TZP"},
                 active_threshold=1e-3,        # or True, for the default
                 active_selection="mp2")
```

`active_threshold` keeps the virtual natural orbitals whose **occupation number**
(NOON) is at least the value given. `True` takes the default, `1e-3`. It needs a
selector that computes occupations, so it is refused with
`active_selection="energy"` — orbital energies are not occupations.

The two compose. Given both, the threshold decides which orbitals earn a place
and the count caps how many there is room for, and whichever binds first wins:
`active_orbitals=8, active_threshold=1e-5` means "the orbitals worth keeping, but
never more than eight".

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
than printing a number the run will not use. Pass `active_orbitals` as well if
you need the width known in advance.

On the command line:

```bash
mandacaru H2O --cell 12 --basis PAW-LCAO --basis-option size=TZP \
    --active-orbitals 10 --active-selection mp2 --dry-run
```

`--active-orbitals` reads `10` as a total, `3,9` as the `occupied,virtual`
split, and `'[0,1,4,7]'` as an explicit list. `--active-threshold 1e-3` uses the
occupation criterion instead.

## Saying which — and why energy ordering is the wrong answer

`active_selection` ranks the **virtual** orbitals.

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
`active_selection="natural"` carries **no information** for a closed-shell
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

## Occupied orbitals are ranked by energy, always

`active_selection` does not touch the occupied space, and the MP2 selector
deliberately leaves the occupied block of its rotation as the identity. Two
things depend on that.

The first is bookkeeping with teeth: `frozen_core="auto"` resolves to "the
lowest so many MOs", which is only the chemical core while the orbitals are
energy-ordered. A selector that reordered the occupied block would leave those
indices naming different orbitals, and the run would silently freeze the wrong
ones. (If the incoming orbitals ever fail to diagonalize the Fock matrix, the
selection is refused rather than applied to labels it cannot trust.)

The second is physics: removing an occupied orbital is a far coarser
approximation than removing a virtual one, and it belongs to an explicit request
— `frozen_core`, `frozen_orbitals`, or the `{"occupied": ...}` form — not to an
automatic selector. This is the classic frozen-natural-orbital scheme, and it is
deliberately the conservative half of it.

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

**Spin-orbit coupling.** The selector ranks spatial orbitals, and with
spin-orbit coupling the two spins of a spatial orbital are not degenerate
partners, so keeping or deleting them together is not a choice the Hamiltonian
permits.

## What the density still does

Everything: the cube writer, the atomic charges and the natural orbitals all
work with a truncated space, and the electron count comes out exact. A deleted
virtual is empty in the model, so it contributes nothing and stays zero — but it
is neither frozen nor active, so the density path is told the active set
explicitly rather than deriving it as the complement of the frozen one.

## Reading it back

The run log reports the partition that was actually built, next to the frozen
core:

```text
frozen core              : none
active space             : 4 active, 8 deleted of 12 spatial orbitals
                           (virtuals ranked by mp2), MP2 E_corr -0.021800 Ha
                           in the full virtual space
```

The reported correlation energy is the **untruncated** one, on purpose: it is
what the truncation is measured against. In code it is on the integral object as
`calc.solver._gradient_context["active_space"]`, an
{class}`~mandacaru.algorithms.active_space.ActiveSpace`.

A dry run reports the register width without computing any integrals. It cannot
know *which* orbitals a selector will keep — that needs the integrals — but the
width does not depend on the choice, so the number is exact:

```text
  basis functions   : 29  (O:17, H:6, H:6)
  deleted virtuals  : 19 spatial orbital(s), ranked by mp2
  active orbitals   : 10 spatial / 20 spin
  QUBITS REQUIRED   : 18
```
