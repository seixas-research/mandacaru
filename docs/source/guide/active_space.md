# Fitting a large basis on a small register

A basis set buys accuracy with virtual orbitals, and a qubit register pays for
them at two qubits each. `active_space` is where the two currencies are
exchanged: keep a large basis for the shape of the orbitals, and let only a few
of them reach the register.

## The three places an orbital can go

| | what happens to it | who may be one |
|---|---|---|
| **frozen** | replaced by its mean field: a constant energy plus an effective one-body potential on the rest | doubly occupied only |
| **active** | represented by qubits | anything |
| **deleted** | dropped; nothing is folded in | virtual only |

An orbital that is empty in the reference adds nothing to the constant energy or
to the effective potential, so deleting it costs exactly the correlation it
would have carried. Every occupied orbital must be frozen or active; a request
that deletes one is refused, because it would take its electrons with it.

## One option, six keys

`active_space` is a dictionary. The keys are exactly these; any other key is
refused.

| key | meaning |
|---|---|
| `"method"` | how the virtual orbitals are ranked: `"energy"` (the default), `"mp2"` or `"dlpno-mp2"` |
| `"orbitals"` | how many orbitals reach the register, by count, by `{"occupied": n, "virtual": m}`, or by an explicit list of indices |
| `"threshold"` | keep the virtual natural orbitals whose occupation is at least this value; `True` means `1e-3` |
| `"frozen"` | the frozen core: `"auto"`, a list of orbital indices, or a count |
| `"symmetry"` | keep degenerate orbital sets whole (closed shell only) |
| `"correlating_pairs"` | not implemented; anything but `False` or `None` raises |

Omitting `active_space` keeps every orbital.

```python
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("LiH")
atoms.center(vacuum=3.0)
for spec in (None,
             {"frozen": "auto"},
             {"frozen": "auto", "method": "mp2", "orbitals": 2}):
    atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO",
                           grid_spacing=0.25, active_space=spec)
    energy = atoms.get_potential_energy()
    print(spec, atoms.calc.n_qubits, "qubits", f"{energy:.6f} eV")
```

`"frozen": "auto"` freezes one orbital per atom from lithium to neon, five from
sodium to argon, and so on. With a PAW-LCAO basis it freezes nothing: a
valence-only basis has no core left to freeze, and the virtual orbitals are what
fits the register.

## Choosing the orbitals

`"orbitals"` and `"threshold"` answer different questions. A count fixes the
width of the register; a threshold fixes the criterion and lets the width
follow. Given both, the threshold decides which virtual orbitals earn a place
and the count caps how many.

* **A count.** `12` keeps the first twelve non-frozen orbitals in the order the
  method gives. `{"occupied": 3, "virtual": 9}` keeps three doubly occupied and
  nine virtual orbitals, freezing the rest of the occupied space. An index list
  names the orbitals exactly and needs `"method": "energy"`.
* **A threshold.** It needs an MP2 ranking, so it is refused with
  `"method": "energy"`. It must lie between 0 and 2. A virtual natural
  occupation is the small charge that correlation promotes into the orbital,
  of order `1e-2` at the very most, so a threshold such as `0.02` can sit above
  the whole spectrum. Choose it from `selection_occupations` (below), not by
  guess. With a threshold, a doubly occupied orbital whose natural occupation
  is within the threshold of 2 is frozen.

The ranking methods differ in what they cost and what they ask:

`"energy"`
: Canonical order, lowest orbital energy first. Free, and valid for open
  shells. Orbital energy is a poor proxy for correlation importance in a basis
  with diffuse or polarization functions.

`"mp2"`
: The frozen natural orbitals: the eigenvectors of the second-order density's
  virtual block, which rotate the virtual space so a few orbitals gather the
  correlation that canonical ordering spreads thinly. It needs a closed-shell
  Hartree-Fock reference and real orbitals; an open shell is refused.

`"dlpno-mp2"`
: The same natural orbitals from local MP2 (Boys localization, projected
  atomic orbital domains, pair natural orbitals), without forming the full
  four-index tensor. Closed shell only.

The frozen orbitals are always the lowest-energy occupied ones; the ranking
method only decides the virtual space (and rotates the active doubly occupied
orbitals among themselves, which leaves the reference determinant and its
energy unchanged).

The remaining open items are listed in [what is not yet available](#not-yet-available):
open-shell MP2 selection and correlating-pair constraints.

## Reading the selection back

The reduced Hamiltonian is explicit on the result. It carries the selector's
bookkeeping:

```python
model = atoms.calc.result.hamiltonian
model.n_spatial_orbitals          # active spatial orbitals
model.frozen_indices              # the frozen core
model.mp2_correlation_energy      # untruncated, Hartree (None for "energy")
model.selection_occupations       # the spectrum the cut was made on
model.orbital_rotation            # the rotation applied to the canonical orbitals
```

The correlation energy is the untruncated one on purpose: it is what the
truncation is measured against. Deleting a virtual orbital removes variational
freedom, so the exact energy of a truncated space lies above the untruncated
one and falls as orbitals return. A truncated ADAPT-VQE run can still come out
a few meV below an untruncated one, when the solver's own residual on the
larger space exceeds the truncation error. Compare truncations by exact
diagonalization, or with a tolerance tight enough that the solver is not the
largest error.

Densities, cube files and natural orbitals work with a truncated space: frozen
occupied orbitals are included, deleted virtual ones are not, and the electron
count is exact. Variational forces of `vqe` and `adapt-vqe` include the response
of the active-space selection.

A spatial active space does not apply to a `ghf` reference, whose orbitals are
spinors; it is refused.
