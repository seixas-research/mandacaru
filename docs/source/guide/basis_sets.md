# Basis sets: multiple zeta and polarization

Every basis in Carcará is generated from scratch — there are no tabulated
exponents anywhere in the package. For the numerical atomic orbitals (NAO) that
generation is now controlled by a `size` argument, which selects how much
variational freedom each valence shell gets.

```python
atoms.calc = VQE(basis={"name": "NAO", "size": "DZP"})
```

## Why a single zeta is not enough

A single-zeta (SZ) basis gives each occupied valence subshell exactly one radial
function. That orbital can be occupied or emptied, but never *reshaped*: it has
the width the isolated atom gave it, and bonding cannot change it. Two distinct
freedoms are missing.

**Radial.** An atom in a molecule is not the atom in vacuum — its valence shell
contracts under a more positive environment and expands under a more negative
one. Describing that needs a second function of the same symmetry but a
different width.

**Angular.** A bond pulls charge off-centre in a direction the shell's own
angular momentum cannot express. A hydrogen 1s is spherical; the moment it bonds,
the density is not. Fixing that requires $l+1$ character.

## The sizes

| Name | Zetas | Polarization shells |
|------|-------|---------------------|
| `SZ`   | 1 | – |
| `SZP`  | 1 | 1 |
| `DZ`   | 2 | – |
| `DZP`  | 2 | 1 |
| `DZ2P` | 2 | 2 |
| `TZ`   | 3 | – |
| `TZP`  | 3 | 1 |
| `TZ2P` | 3 | 2 |
| `QZ`   | 4 | – |
| `QZP`  | 4 | 1 |
| `QZ2P` | 4 | 2 |

An explicit `(n_zeta, n_polarization)` pair works too. **`size` defaults to
`"DZP"`** — single zeta has neither radial nor angular freedom, so it is a poor
default for chemistry. Pass `size="SZ"` for the older minimal basis, which is
far cheaper and is still what the pseudopotential path uses unless you ask
otherwise.

Other NAO options combine freely in the same dict:

```python
basis={"name": "NAO", "size": "DZP", "energy_shift": 0.03}
```

`energy_shift` (eV, default 0.03) sets the confinement radius
$r_c = \pi/\sqrt{2\delta E}$; a smaller shift means a longer-ranged, more
diffuse orbital. `split_norm` (default 0.15) controls how much norm each extra
zeta leaves outside its split radius.

The cost is set by how many functions each size produces, and **every function
becomes two spin orbitals, hence two qubits**:

| Atom | Valence | SZ | DZ | DZP | TZP | QZP |
|------|---------|----|----|-----|-----|-----|
| H  | 1s    | 1 | 2  | 5  | 6  | 7  |
| C  | 2s 2p | 4 | 8  | 13 | 17 | 21 |
| O  | 2s 2p | 4 | 8  | 13 | 17 | 21 |
| Fe | 3d 4s | 6 | 12 | 19 | 25 | 31 |

Polarization is not a small addition: it adds a whole $l+1$ shell — 3 functions
for an s-valence atom, 5 for a p-valence atom, 7 for a d-valence one. On a
state-vector simulator `DZP` water is already out of reach; the sizes exist so
that the basis is not the limiting approximation when the system is small enough
to afford them.

## How the extra zetas are built

They are *not* new eigenvalue problems. Carcará uses the SIESTA split-valence
construction (Artacho *et al.*, 1999). Given the first-zeta radial function
$R_1$, pick a split radius $r_s$ leaving a prescribed fraction of the norm
outside it (`split_norm`, 0.15 by default):

$$\int_{r_s}^{r_c} |R_1(r)|^2 r^2\,dr = \texttt{split\_norm}.$$

Inside $r_s$, replace the orbital by the smooth polynomial $r^l(a - b r^2)$
matched in value *and* slope at $r_s$, and keep the difference:

$$R_2(r) = \begin{cases} R_1(r) - r^{l}(a - b r^{2}), & r < r_s \\ 0, & r \ge r_s. \end{cases}$$

$R_2$ is strictly shorter-ranged than $R_1$ and vanishes smoothly at $r_s$, so it
is cheap to integrate and injects no discontinuity. Higher zetas repeat the
construction on the previous one with a halved split norm. For the hydrogen 1s:

| Zeta | Range (a₀) | Norm |
|------|-----------|------|
| 1 | 3.99 | 1.00 |
| 2 | 2.09 | 1.3e-1 |
| 3 | 1.22 | 1.7e-2 |
| 4 | 0.81 | 2.9e-3 |

Polarization shells are solved in the same confining sphere at $l_{\max}+1$,
using the lowest principal quantum number that angular momentum allows, so they
stay compact.

## What it buys, and the catch

Hartree–Fock total energy of H₂ at 0.74 Å, on a uniform real-space grid:

| Size | Orbitals | E (h=0.20 Å) | E (h=0.12 Å) |
|------|----------|--------------|--------------|
| SZ  | 2  | −1.064037 | −1.078832 |
| DZ  | 4  | −1.092314 | −1.110606 |
| TZ  | 6  | −1.092315 | −1.110642 |
| DZP | 10 | −1.093031 | −1.111294 |

Double zeta is worth ~28 mHa, polarization a little more. But the third zeta
adds essentially nothing — and the reason is not that it is redundant.

```{warning}
Extra zetas are short-ranged **by construction**, and Carcará samples every
basis function on a uniform real-space grid. The third hydrogen zeta extends to
0.65 Å, which is 3.2 grid points at `h = 0.20 Å`. A function three points wide is
not represented, it is aliased.

Its contribution does grow as the grid is refined (1e-6 Ha at `h = 0.20 Å`,
3.6e-5 Ha at `h = 0.12 Å`), so this is a resolution limit rather than a defect.
On a uniform mesh, though, it is the binding constraint: **there is no point
paying for TZ or QZ unless the grid can resolve them.** DZP is the practical
sweet spot.
```

Multiple zeta also risks linear dependence, since the added functions overlap the
ones they were split from. The overlap matrix stays comfortably invertible at DZP
(condition number ~170 for H₂), but it grows with every zeta — another reason the
high sizes are specialist tools.

See `examples/21_multizeta_basis.py`, which reproduces every table above and
plots the zeta hierarchy.
