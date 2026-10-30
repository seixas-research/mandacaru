(nao-ae)=
# All-electron numerical atomic orbitals (NAO-AE)

`basis="NAO-AE"` selects an **all-electron numerical atomic orbital** basis:
every function is a numerically tabulated radial part times a spherical harmonic,
generated for each element by solving one-dimensional radial problems. Like every
other family it carries no tabulated basis-set data.

```python
from mandacaru import Mandacaru

Mandacaru(method="rhf", basis={"name": "NAO-AE", "tier": 1, "onset": 3.0})
```

The dictionary accepts `tier`, `onset`, `width`, `scale`, `extra`,
`linear_dependence_tol`, `tail_norm`, `points` and `r_max`, and nothing else.
`onset` and `width` are in ångström; `points` and `r_max` set the radial grid of
the reference atom.

## What "all-electron" means here

The **minimal basis is the atom itself**. A spherical, self-consistent LDA atom
provides the effective potential $v_\text{free}(r)$, and *every* occupied shell
$(n, l)$, core shells included, is re-solved as a bound state of the
basis-defining potential

```{math}
v_\text{basis}(r) = v_\text{free}(r) + v_\text{cut}(r).
```

Oxygen therefore brings its own 1s, 2s and 2p; nothing is pseudized and nothing
is frozen at the basis level.

## Localization by a smooth wall

Where the `NAO` family confines orbitals in a hard sphere, NAO-AE uses an
*exponential wall* that is exactly zero up to an **onset** $r_0$, rises
smoothly, and diverges at $r_0 + w$:

```{math}
v_\text{cut}(r) = s\,\frac{\exp\!\big[-w/(r - r_0)\big]}{(r_0 + w - r)^2},
\qquad r_0 < r < r_0 + w .
```

All derivatives vanish at the onset, so an orbital is untouched inside $r_0$ and
its tail is bent to zero over the width $w$ without a kink. Every function is
**strictly zero** beyond $r_0 + w$. The defaults are `onset=3.0` and `width=1.0`
Å; the cell must reach at least `onset + width` beyond every atom, or the
functions are truncated by the box rather than by the wall.

## Tiers: hydrogen-like functions sized from the atom

Radial and angular flexibility come from **hydrogen-like functions**, bound
states of $-z/r + v_\text{cut}(r)$ whose effective charge $z$ sets their size.
The value is derived from the atom rather than tabulated: a hydrogen-like
$(n, l)$ state has mean radius

```{math}
\langle r \rangle = \frac{3n^2 - l(l+1)}{2z},
```

so $z$ is chosen to give the function a prescribed extent relative to the
valence shell's own $\langle r \rangle$. The functions are organized in tiers:

| `tier` | Added on top of the previous one |
| :--- | :--- |
| `0` | nothing: the all-electron minimal basis |
| `1` (default) | one **polarization** shell at $l_\max + 1$ (nodeless, as compact as the valence shell) and one **diffuse** function per valence channel (one more node than the valence shell, twice its extent) |
| `2` | a second polarization channel at $l_\max + 2$, a noded polarization function at $l_\max + 1$, and one **contracted** function per valence channel (0.6 of the valence extent) |

`extra=[(n, l, z), ...]` adds explicit hydrogen-like functions for full control.
Within each $l$ channel the functions are **Gram–Schmidt orthonormalized**
(minimal first, then the most compact tier functions), and a candidate whose norm
after projection falls below `linear_dependence_tol` (default `1e-4`) is dropped.
Tier functions are also shortened until at most `tail_norm` (default `1e-4`) of
their norm lies beyond the onset, so compact functions stay compact.

## What to expect

- The tier-0 functions are LDA atomic orbitals, which need not coincide with the
  analytic `HAO` minimal basis; tier 1 is the sensible default.
- The core functions are as sharp as the atom's own core. The uniform real-space
  grid resolves them no better than it resolves the `HAO` core, so refine the
  grid and check convergence, or use a [PAW-LCAO basis](pseudopotentials.md) when
  forces or geometry matter ([Forces](forces_relaxation.md)).

`examples/basis_nao/02_nao_ae_tiers_h2.py` runs H₂ over the tiers.
