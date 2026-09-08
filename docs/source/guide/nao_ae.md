# All-electron numerical atomic orbitals (NAO-AE)

`basis="NAO-AE"` selects an **all-electron numerical atomic orbital** basis:
every function is a numerically tabulated radial part times a spherical
harmonic, generated from scratch for each element by solving one-dimensional
radial problems. Like every other family in Carcará it carries no tabulated
basis-set data.

```python
atoms.calc = QuantumCalculator(basis={"name": "NAO-AE", "tier": 1, "onset": 3.0})
```

## What "all-electron" means here

The **minimal basis is the atom itself**. A spherical, self-consistent LDA atom
({func}`~carcara.basis.atomic_solver.solve_atom`) provides the effective
potential $v_\text{free}(r)$, and *every* occupied shell $(n, l)$ — core shells
included — is re-solved as a bound state of the basis-defining potential

$$v_\text{basis}(r) = v_\text{free}(r) + v_\text{cut}(r).$$

Oxygen therefore brings its own 1s, 2s and 2p; nothing is pseudized and nothing
is frozen at the basis level (the `frozen_core` approximation still applies
afterwards, at the Hamiltonian level, exactly as for `FAO`).

## Localization by a smooth wall

Where the SIESTA-type `NAO` family confines orbitals in a hard sphere, NAO-AE
uses an *exponential wall* that is exactly zero up to an **onset** $r_0$, rises
smoothly, and diverges at $r_0 + w$:

$$v_\text{cut}(r) = s\,\frac{\exp\!\big[-w/(r - r_0)\big]}{(r_0 + w - r)^2},
\qquad r_0 < r < r_0 + w .$$

All derivatives vanish at the onset, so an orbital is untouched inside $r_0$ and
its tail is bent to zero over the width $w$ without a kink. Every function is
**strictly zero** beyond $r_0 + w$. The defaults are `onset=3.0`, `width=1.0`
(Ångström); the real-space box must reach at least `onset + width` beyond every
atom, or the functions are truncated by the box rather than by the wall.

## Tiers: hydrogen-like functions sized from the atom

Radial and angular flexibility come from **hydrogen-like functions**, bound
states of $-z/r + v_\text{cut}(r)$ whose effective charge $z$ sets their size.
Instead of a table of $z$ per element, the value is derived from the atom: a
hydrogen-like $(n, l)$ state has mean radius

$$\langle r \rangle = \frac{3n^2 - l(l+1)}{2z},$$

so $z$ is chosen to give the function a prescribed extent relative to the
valence shell's own $\langle r \rangle$
({func}`~carcara.basis.nao_ae.effective_charge_for_radius`). The functions are
organised in tiers:

| `tier` | Added on top of the previous one |
| :--- | :--- |
| `0` | nothing — the all-electron minimal basis |
| `1` (default) | one **polarization** shell at $l_\max + 1$ (nodeless, as compact as the valence shell) and one **diffuse** function per valence channel (one more node than the valence shell, twice its extent) |
| `2` | a second polarization channel at $l_\max + 2$, a noded polarization function at $l_\max + 1$, and one **contracted** function per valence channel (0.6 of the valence extent) |

Explicit `extra=[(n, l, z), ...]` triples can be added for full control.

| Element | tier 0 | tier 1 | tier 2 |
| :--- | ---: | ---: | ---: |
| H | 1 | 5 | 14 |
| O | 5 | 14 | 30 |
| Li | 2 | 6 | — |

Within each $l$ channel the functions are **Gram–Schmidt orthonormalized**
(minimal first, then the most compact tier functions), and a candidate whose
norm after projection falls below `linear_dependence_tol` (`1e-4`) is dropped.
Tier functions are also shortened until at most `tail_norm` (`1e-4`) of their
norm lies beyond the onset — the wall is re-centred and the state re-solved —
so compact functions stay compact.

```python
from carcara.basis import BasisSet

bset = BasisSet.build("NAO-AE", tier=1)
print(bset.describe("O"))
#   1s (atomic)                  eps =  -18.7413 Ha   <r> =  0.20 a0 ...
#   2s (atomic)                  eps =   -0.8701 Ha   <r> =  1.15 a0 ...
#   2p (atomic)                  eps =   -0.3378 Ha   <r> =  1.26 a0 ...
#   3d (polarization, z=8.31)    ...
#   3s (diffuse, z=5.87)         ...
#   3p (diffuse, z=4.95)         ...
```

## What to expect

- The minimal (tier 0) functions are LDA atomic orbitals — for hydrogen
  slightly more diffuse than the exact 1s, so on H₂ tier 0 sits *above* the
  analytic `FAO` minimal basis; tier 1 recovers a chemically significant amount
  of energy (polarization and radial breathing) and is the sensible default.
- The core functions are as sharp as the atom's own core (the oxygen 1s has
  $\langle r\rangle \approx 0.2\,a_0$). The uniform real-space grid resolves them
  no better than it resolves the `FAO` core, so use `frozen_core=True` for
  anything heavier than the first row, exactly as with `FAO`.
- NAO-AE is all-electron by definition and is refused together with
  `pseudopotentials=True`; use `basis={"name": "PP", "size": ...}` there.

See `examples/23_NAO_AE_basis.py` and {mod}`carcara.basis.nao_ae`.
