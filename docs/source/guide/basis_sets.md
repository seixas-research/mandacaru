# Basis sets: multiple zeta, polarisation, and the named Gaussian families

Every basis in Carcará is generated from scratch — there are no tabulated
exponents anywhere in the package. This page covers the two ways to buy
variational freedom: the `size` argument of the numerical atomic orbitals
(NAO), which selects how much radial and angular freedom each valence shell
gets, and — at the end — the standard **named Gaussian families** (Pople,
Dunning, Karlsruhe), which are accepted by name and generated with the
published shell structure.

```python
atoms.calc = Carcara(method="vqe",
                     basis={"name": "NAO", "size": "DZP"})
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

| Name | Zetas | Polarisation shells |
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
far cheaper.

Other NAO options combine freely in the same dict:

```python
basis={"name": "NAO", "size": "DZP", "energy_shift": 0.03}
```

`energy_shift` (eV, default 0.03) sets the confinement radius
$r_c = \pi/\sqrt{2\delta E}$ in atomic units (convert the energy shift to
Hartree to evaluate this expression); a smaller shift means a longer-ranged, more
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

Polarisation is not a small addition: it adds a whole $l+1$ shell — 3 functions
for an s-valence atom, 5 for a p-valence atom, 7 for a d-valence one. On a
state-vector simulator `DZP` water is already out of reach; the sizes exist so
that the basis is not the limiting approximation when the system is small enough
to afford them.

## How the extra zetas are built

They are *not* new eigenvalue problems. Carcará uses the SIESTA split-valence
construction (Artacho *et al.*, 1999). Given the first-zeta radial function
$R_1$, pick a split radius $r_s$ leaving a prescribed fraction of the norm
outside it (`split_norm`, 0.15 by default):

```{math}
\int_{r_s}^{r_c} |R_1(r)|^2 r^2\,dr = \texttt{split\_norm}.
```

Inside $r_s$, replace the orbital by the smooth polynomial $r^l(a - b r^2)$
matched in value *and* slope at $r_s$, and keep the difference:

```{math}
R_2(r) = \begin{cases} R_1(r) - r^{l}(a - b r^{2}), & r < r_s \\ 0, & r \ge r_s. \end{cases}
```

$R_2$ is strictly shorter-ranged than $R_1$ and vanishes smoothly at $r_s$, so it
is cheap to integrate and injects no discontinuity. Higher zetas repeat the
construction on the previous one with a halved split norm. For the hydrogen 1s:

| Zeta | Range (a₀) | Norm |
|------|-----------|------|
| 1 | 3.99 | 1.00 |
| 2 | 2.09 | 1.3e-1 |
| 3 | 1.22 | 1.7e-2 |
| 4 | 0.81 | 2.9e-3 |

Polarisation shells are solved in the same confining sphere at $l_{\max}+1$,
using the lowest principal quantum number that angular momentum allows, so they
stay compact.

## What it buys, and the catch

Hartree–Fock total energy of H₂ at 0.74 Å (eV), on a uniform real-space grid:

| Size | Orbitals | E (h=0.20 Å) | E (h=0.12 Å) |
|------|----------|--------------|--------------|
| SZ  | 2  | −28.953922 | −29.356514 |
| DZ  | 4  | −29.723378 | −30.221129 |
| TZ  | 6  | −29.723405 | −30.222108 |
| DZP | 10 | −29.742889 | −30.239850 |

Double zeta is worth ~0.76 eV, polarisation a little more. But the third zeta
adds essentially nothing — and the reason is not that it is redundant.

```{warning}
Extra zetas are short-ranged **by construction**, and Carcará samples every
basis function on a uniform real-space grid. The third hydrogen zeta extends to
0.65 Å, which is 3.2 grid points at `h = 0.20 Å`. A function three points wide is
not represented, it is aliased.

Its contribution does grow as the grid is refined (3e-5 eV at `h = 0.20 Å`,
1e-3 eV at `h = 0.12 Å`), so this is a resolution limit rather than a defect.
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


## Named Gaussian families: Pople, Dunning, Karlsruhe

The standard Gaussian basis-set names are accepted directly, and every one of
them is **generated natively** the same way 6-31G(d) already is: the name is
parsed into its *structure* — core contraction length, valence split,
polarisation, diffuse and core-correlating functions — and the exponents and
contraction coefficients are produced for the atom at hand from the cached
Slater-orbital fits and Slater's rules
({mod}`carcara.basis.gaussian_families`). No basis-set table is read.

```python
atoms.calc = Carcara(basis="cc-pVDZ")
atoms.calc = Carcara(basis="6-311+G(2df,2p)")
atoms.calc = Carcara(basis="def2-TZVP")
```

| Family | Names |
| :--- | :--- |
| Minimal | `STO-3G`, `STO-4G`, `STO-5G`, `STO-6G` |
| Pople | `3-21G`, `3-21G*`, `3-21G**`, `3-21+G`, `3-21++G`, `3-21+G*`, `3-21+G**`, `4-21G`, `4-31G`, `6-21G`, `6-31G`, `6-31G*`, `6-31+G*`, `6-31G(3df,3pd)`, `6-311G`, `6-311G*`, `6-311+G*`, `6-311+G(2df,2p)` — and any other name of the form `K-NL[M][+|++]G[*|**|(…)]` |
| Dunning | `cc-pVDZ`, `cc-pVTZ`, `cc-pVQZ`, `cc-pV5Z`, `aug-cc-pVDZ`, `cc-pCVDZ` — and any `[aug-]cc-p[C]VXZ` |
| Karlsruhe | `def2-SV(P)`, `def2-SVP`, `def2-SVPD`, `def2-TZVP` (also spelled `def2-TZDP`), `def2-TZVPD`, `def2-TZVPP`, `def2-TZVPPD`, `def2-QZVP`, `def2-QZVPD`, `def2-QZVPP`, `def2-QZVPPD` |

What is reproduced is the **shell structure and function count** of the
published set — `cc-pVTZ` carbon is `[4s3p2d1f]` (30 functions), `def2-TZVP`
carbon `[5s3p2d1f]`, `6-311+G(2df,2p)` hydrogen `[3s2p]`:

```python
from carcara.basis import BasisSet, parse_basis_name, shell_notation

bset = BasisSet.build("aug-cc-pVDZ")
bset.notation("C")                       # '[4s3p2d]'
len(bset.atom("C"))                      # 23
parse_basis_name("6-31+G*").summary()
```

What is *not* reproduced are the published exponents, which come from
molecular energy optimisations. Carcará's are its own: a Slater-orbital fit
partitioned tightest-first for the contracted and split-valence functions,
polarisation exponents `f_l · ζ_val²` spread geometrically, diffuse functions
at the atom's most diffuse exponent divided by 3.5, and tight
(core-correlating) functions at three times the tightest valence exponent.
Treat the result as a self-contained basis *of the same size and shape* as its
namesake — the right thing for qubit-count planning and for comparing solvers
across basis sizes — not as a drop-in for a literature number quoted in that
basis.

Two conventions to know. Shells are **spherical** (5 `d`, 7 `f`), so `6-31G*`
carbon has 14 functions, not the 15 of a Cartesian-`d` program. And Pople's
`3-21G*` puts its `d` functions on second-row atoms only, exactly as
published, while `6-31G*` and `6-311G*` polarise every atom beyond helium.


## Virtual levels for the FAO basis

`FAO` is the cheapest basis in Carcará: one analytic hydrogenic orbital per
**occupied** subshell, carrying the atom's bare nuclear charge (H → 1s; Li →
1s, 2s; C → 1s, 2s, 2p). That minimality is also its limit — a correlated
method has almost nothing to correlate *into*. H₂ in the occupied-only FAO
basis is 2 spatial orbitals, 4 qubits, and a single double excitation.

`virtual_orbitals` buys room above the occupied set:

```python
atoms.calc = Carcara(method="adapt-vqe",
                     basis={"name": "FAO", "virtual_orbitals": 1},
                     h=0.30)
```

It appends the *k* lowest **unoccupied** subshells of each atom, in aufbau
order and with the same bare Z: hydrogen gains `2s`, carbon `3s`, iron `4p`. A
partially filled subshell counts as occupied — carbon's `2p²` is already a
basis shell, so carbon's first virtual level is `3s`, not the empty half of
`2p`. The default is `0`, which is the historical minimal basis exactly.

**A level is a whole subshell.** `virtual_orbitals` counts levels, not
functions, so one virtual `p` level adds three functions. Half a shell would
break the atom's spherical symmetry and make the energy depend on how the
molecule happens to be oriented in its box, so every appended level carries all
its `m` components. What that costs on hydrogen:

| `virtual_orbitals` | levels added | functions per H | H₂ qubits |
|---|---|---|---|
| `0` | – | 1 | 4 |
| `1` | 2s | 2 | 8 |
| `2` | 2s, 2p | 5 | 20 |
| `3` | 2s, 2p, 3s | 6 | 24 |

`BasisSet.build("FAO", virtual_orbitals=k).function_count("H")` reports it for
any element, and the [dry run](dry_run.md) reports the qubit total before
anything is integrated.

The payoff is variational: the smaller basis is a strict subset of the larger,
so the correlated energy must fall. H₂ at 0.74 Å in an 8 Å cell (`h = 0.30`,
ADAPT-VQE to its basis FCI):

| `virtual_orbitals` | qubits | E (eV) | gain |
|---|---|---|---|
| `0` | 4 | −28.1619 | – |
| `1` | 8 | −28.3009 | −0.139 |
| `2` | 20 | −28.3384 | −0.177 |

```{note}
A virtual hydrogenic orbital is **diffuse** — H `2s` has ⟨r⟩ = 6 a₀ ≈ 3.2 Å —
so the cell has to be large enough to contain it or the grid clips its tail,
and the energies above move with the box like every other FAO number. The
engine's resolution check warns when a function is not represented on the grid.
```

## A different basis on different elements

The `basis` argument also takes a **per-element mapping**: a dict keyed by
chemical symbol (plus an optional `"*"` default), each entry a basis spec of
its own. The typical use is a polarised basis on the atoms whose chemistry
matters next to a minimal one on a spectator ion — the qubit count is the sum
over atoms, so this is how a large system is kept inside a state-vector
budget:

```python
basis = {"O": {"name": "NAO", "size": "DZP"},
         "H": {"name": "NAO", "size": "DZP"},
         "*": "FAO"}                              # every other element
atoms.calc = Carcara(method="adapt-vqe",
                     basis=basis,
                     frozen_core=True)
```

Every driver, the dry run and `BasisSet.build(mapping)` accept it; an element
without an entry and without a `"*"` default is an error rather than a silent
fallback, and the plane-wave family, which is not atom-centred, cannot be
assigned to one element. With a [pseudopotential family](pseudopotentials.md)
the same mapping selects a per-element *size* — every element must use the
same family, since the radial functions there come from each potential.
