# Pseudopotentials

```python
atoms.calc = ADAPTVQE(basis="FAO", pseudopotentials=True, h=0.15)
```

That switch turns an all-electron calculation into a valence-only one: the core
electrons are removed, the basis becomes smooth pseudo-atomic orbitals, and the
singular $-Z/r$ external potential is replaced by a bounded local channel plus
Kleinman–Bylander projectors.

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

`pseudos/` ships norm-conserving Troullier–Martins pseudopotentials for **every
element with Z < 90** (H through Ac), generated from scratch by Carcará's own
LDA radial atomic solver. They are loaded automatically by symbol.

```python
from carcara.basis.pseudo_io import available_elements, get_pseudopotential

pp = get_pseudopotential("Fe")
pp.valence_charge     # 8.0  -- 3d^6 4s^2
sorted(pp.channels)   # [0, 2]
```

The valence includes semicore $(n-1)d$ and $(n-2)f$ shells, so iron is an
eight-electron atom with a d channel rather than a two-electron 4s² one.

To regenerate or extend the library:

```python
from carcara.basis.pseudo_io import build_library

written, failures = build_library()               # all of Z < 90
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
ADAPTVQE(basis="6-31G(d)", pseudopotentials=True)   # ValueError
```

What you *can* vary is the size hierarchy, which refines that pseudo-orbital
instead of replacing it — the extra zetas are split-valence refinements of the
pseudized function, and the polarization shell is split from the outermost
channel:

```python
ADAPTVQE(basis={"name": "PP", "size": "DZP"}, pseudopotentials=True)
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

See `examples/19_pseudopotential_generation.py` and
`examples/20_pseudopotential_calculations.py`.
