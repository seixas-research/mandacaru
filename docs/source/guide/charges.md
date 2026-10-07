# Partial charges and magnetic moments

A converged run knows the electron density exactly. Turning it into *per-atom*
numbers needs one more thing the physics does not supply — a boundary between
atoms — so Mandacaru offers three conventions and lets you see them disagree.

```python
atoms.get_potential_energy()                      # run first

atoms.get_charges()                               # Hirshfeld, the default
atoms.calc.get_charges(method="bader")
atoms.get_magnetic_moments()                      # per atom
atoms.get_magnetic_moment()                       # the total
```

All three are ASE properties, so `atoms.get_charges()` and
`atoms.get_magnetic_moments()` work through the usual `Atoms` accessors and
land in `calc.results` as `charges`, `magmoms` and `magmom`.
`calc.get_total_magnetic_moment()` is the explicit name for the last one.

## What is a convention and what is not

| | fixed by the state | a choice |
| :--- | :--- | :--- |
| `sum(get_charges())` | the system's total charge | — |
| `get_magnetic_moment()` | $N_\alpha - N_\beta$ | — |
| `get_charges()` per atom | — | the partition |
| `get_magnetic_moments()` per atom | — | the partition |

The totals are integrals of the whole density and come out identical whichever
method is asked for — the tests pin them to 1e-8. Only the split between atoms
moves. That is why **`get_total_magnetic_moment()` takes no `method`**: there
is nothing to choose.

## The three partitions

Each defines a weight $w_A(\mathbf r)$ with $\sum_A w_A = 1$, giving
$N_A = \int w_A n\, d^3r$ and $q_A = Z_A - N_A$.

`hirshfeld` (default)
: $w_A = n_A^0 / \sum_B n_B^0$ — each atom takes the share its own free-atom
  density contributes to the superposition of free atoms. The reference atoms
  are solved here by the same LDA radial solver the bases are generated with,
  so nothing is tabulated. Smooth, insensitive to the grid, and famously
  **small**: the reference is always a neutral atom, so Hirshfeld understates
  ionicity.

`voronoi`
: $w_A = 1$ where $A$ is the nearest nucleus. Purely geometric — it knows
  nothing about the density it cuts, which makes it a useful control. A charge
  that moves a lot between Voronoi and Hirshfeld is one the partition is
  deciding, not the physics.

`bader`
: The zero-flux partition: every point climbs the density along its steepest
  gradient, and the points reaching the same nucleus are one basin. The
  ascent is **continuous** -- on a cubic-spline interpolation of the density,
  from every point of a twice-refined grid -- and the points next to a basin
  boundary are traced again from eight sub-voxel offsets and averaged, so the
  zero-flux surface is resolved below the grid spacing. It follows the
  density rather than a reference and gives the largest charges of the
  three. (An ascent that walks from node to node steers along the grid's own
  directions: it underestimated cation charges by 0.04--0.2 e at ordinary
  spacings and moved them with a shift of the atoms against the grid.)

LiH at 1.6 Å, HAO, `h=0.20` — same state, three answers:

| method | q(Li) | q(H) |
| --- | --- | --- |
| hirshfeld | +0.689 | −0.689 |
| voronoi | +0.803 | −0.803 |
| bader | +0.921 | −0.921 |

They agree on the sign and on the total, and that is the honest extent of the
agreement. Quote the method with the number.

## Grid sensitivity

Hirshfeld's weights are smooth, so it is exact on a symmetric molecule at any
spacing. The hard partitions cut at a surface the grid has to resolve. H₂,
whose charges must both be zero:

| `h` (Å) | hirshfeld | voronoi | bader |
| --- | --- | --- | --- |
| 0.30 | 0 | 0 | 0 |
| 0.25 | 0 | 0 | 0 |
| 0.20 | 0 | 0 | 0 |

Voronoi is exact because a node lying exactly on the dividing plane is **split
between the tied atoms** rather than given to one of them; without that it
charges H₂ by 0.16 e at some spacings and by nothing at others. Bader's
boundary refinement does the same for the points on its zero-flux surface: a
symmetric grid puts a whole plane of them exactly on the bond midplane, where
round-off alone would pick the side (0.08 e of charge before the refinement).

A partition can only be as converged as the density it cuts: in the
all-electron `HAO` basis LiH's density itself moves with `h` (Hirshfeld
+0.68 / +0.63 / +0.69 at h = 0.30 / 0.25 / 0.20), and Bader follows it. In a
crystal at fixed density, the continuous ascent is converged by about ten
nodes per lattice vector (rocksalt LiH +0.843 / +0.847 / +0.847 at 10 / 12 / 16)
and moves by about 1e-3 e with a shift of the atoms against the grid.

```{note}
An atom that no trajectory reaches -- possible only on a grid far too coarse
for its density -- would get no basin and no electrons; it raises a
`RuntimeWarning` naming the atom instead.
```

## Pseudopotentials

`Z_A` is the charge the *Hamiltonian* carries, so for a pseudopotential run it
is the valence charge (O is 6, not 8) and `q_A` is still the physical partial
charge. Three things follow automatically:

* the Hirshfeld reference becomes the **valence** free atom, so the
  stockholder fractions are not weighted by core density that is not on the
  grid to share;
* Bader's ascent climbs the **all-electron density**: the smooth valence
  density, plus inside each PAW sphere the spherical part of the
  reconstruction $n^1_A - \tilde n^1_A$ (the all-electron partial waves in
  place of the smooth ones), plus the free atoms' frozen cores, both added
  analytically. Both matter. A valence density need not peak at the nuclei:
  covalent silicon's maxima sit at its bond centers, and a valence-only
  ascent gave those basins to one atom. And the smooth density has no cusp
  inside a sphere, so on it hydrogen's zero-flux surface moves toward the
  proton. Water's oxygen came out at −1.68 e (PBE, DZP) on the smooth density
  and is −1.24 e on the reconstructed one, against about −1.2 in the
  literature; HF's fluorine is −0.74. The reconstruction only moves the
  basin boundaries: the electrons counted are still the smooth density in
  each basin plus each sphere's own augmentation charge (next item), and the
  reconstruction integrates to exactly that charge. Its non-spherical part,
  left out, moved water's charges by 0.003 e;
* for PAW-LCAO the grid holds the *smooth* density, and the charge inside each
  augmentation sphere is added back to its own atom from
  $C_A q_A C_A^\dagger$ — that term is block-diagonal per atom, so the
  decomposition is exact rather than a sharing rule. `AtomicPartition.augmentation`
  reports it and `grid_electrons` comes out at the full electron count.

A frozen core is refilled the same way the forces and the density refill it,
so `sum(populations)` is the molecule's electron count however the active
space was truncated.

## Writing them to a file

ASE's extxyz writer turns the per-atom entries of `calc.results` into
**columns** and the scalars into **header** keys, so the whole trick is having
them there when the file is written. `population=` does that on every
evaluation:

```python
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis="HAO",
                       h=0.20,
                       population="hirshfeld")

atoms.get_potential_energy()
write("out.extxyz", atoms)
```

```text
3
Lattice="8.0 0.0 ..." Properties=species:S:1:pos:R:3:charge:R:1:magmoms:R:1 energy=-37.460006 free_energy=-37.460006 magmom=1.0000000 pbc="F F F"
H   4.00000000  3.52500000  3.55000000   0.04549504   0.17184696
H   4.00000000  3.52500000  4.45000000   0.04549504   0.17184696
H   4.00000000  4.47500000  4.00000000  -0.09099008   0.65630608
```

The partial charges and the local moments are columns; the **total** moment
sits in the header beside the energy, which is where a scalar belongs.
`ase.io.read` recovers all four.

```{note}
ASE names the per-atom charge column **`charge`**, singular, while the moments
keep the plural `magmoms`. That is ASE's spelling of the `charges` property,
not a typo here.
```

`population=` takes a method name, `True` for the default, or `None` (the
default) to switch it off — the partition costs a grid pass, and a basin
search for Bader, that a relaxation has no use for. It also becomes the
default for `get_charges()` and `get_magnetic_moments()` on that calculator,
so a run configured for Bader does not quietly report Hirshfeld. An explicit
`method=` still wins.

Without it, calling the getters once before writing has the same effect —
they store what they compute:

```python
atoms.get_potential_energy()
atoms.get_charges()               # -> results["charges"]
atoms.get_magnetic_moments()      # -> results["magmoms"]
atoms.get_magnetic_moment()       # -> results["magmom"]
write("out.extxyz", atoms)
```

## The whole table at once

`calc.atomic_partition(method)` returns an
{class}`~mandacaru.algorithms.charges.AtomicPartition` — populations, charges,
moments, reference charges and augmentation in one object, computed once:

```python
partition = atoms.calc.atomic_partition("bader")
print(partition.summary())
```

```text
bader partition
 atom   Z_eff   electrons    charge    moment
    0    3.00    2.075809 +0.924191 -0.000000
    1    1.00    1.924191 -0.924191 +0.000000
total            4.000000 +0.000000 +0.000000
```
