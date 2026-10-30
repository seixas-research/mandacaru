(basis-sets)=
# Basis sets: multiple zeta, polarization, and the named Gaussian families

Every basis in Mandacaru is generated from scratch; no tabulated exponents are
read. The families are:

| `basis=` | Family |
| --- | --- |
| `"HAO"` | one analytic hydrogenic orbital per occupied subshell, the cheapest basis |
| `"NAO"` | numerical atomic orbitals with multiple zeta and polarization (this page) |
| `"NAO-AE"` | all-electron numerical atomic orbitals in tiers ([NAO-AE](nao_ae.md)) |
| `"STO-3G"`, `"cc-pVDZ"`, `"def2-SVP"`, … | named Gaussian families, generated natively (below) |
| `"PAW-LCAO"`, `"UPAW-LCAO"` | valence-only PAW bases ([PAW-LCAO bases](pseudopotentials.md)) |

```python
from mandacaru import Mandacaru

Mandacaru(method="rhf", basis={"name": "NAO", "size": "DZP"})
```

The number of basis functions is the number of spatial orbitals, and every
spatial orbital is two spin orbitals, hence two qubits for a quantum method.
`estimate_qubits` counts them without computing anything ([dry run](dry_run.md)).

(basis-sizes)=
## The NAO sizes

A single-zeta (SZ) basis gives each occupied valence subshell one radial
function: it can be occupied or emptied but not reshaped, so bonding cannot
change its width (**radial** freedom is missing), and a function of the shell's
own angular momentum cannot bend into a bond (**angular** freedom is missing).
Extra zetas supply the first, polarization shells at $l+1$ the second.

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

An explicit `(n_zeta, n_polarization)` pair works too. Sizes are
case-insensitive. **The default `size` of `"NAO"` is `"DZP"`**; that of a
PAW-LCAO basis is `"SZ"`.

The `NAO` dictionary accepts `size`, `r_c`, `n_grid`, `energy_shift` and
`zeta_split`, and nothing else. `energy_shift` (eV, default 0.03) sets the
confinement radius $r_c = \pi/\sqrt{2\,\delta E}$ with $\delta E$ in Hartree; a
smaller shift means a longer-ranged, more diffuse orbital. `r_c` (Bohr) gives
the radius directly; `n_grid` (default 2000) is the radial grid.

Polarization adds a whole $l+1$ shell: 3 functions for an s-valence atom, 5 for
a p-valence atom, 7 for a d-valence one. The resulting counts:

| Atom | Valence | SZ | DZ | DZP | TZP | QZP |
|------|---------|----|----|-----|-----|-----|
| H  | 1s    | 1 | 2  | 5  | 6  | 7  |
| C, O | 2s 2p | 4 | 8  | 13 | 17 | 21 |
| Fe | 3d 4s | 6 | 12 | 19 | 25 | 31 |

Diffuse functions are a third, independent control, available for the PAW-LCAO
families only ([diffuse functions](#paw-lcao-diffuse)).

(basis-zeta-split)=
## How the extra zetas are built

The extra zetas are not new eigenvalue problems. Mandacaru uses the
split-valence construction (Artacho *et al.*, 1999): given a radial function $R$
and a split radius $r_s$, replace the orbital inside it by the smooth polynomial
$r^l(a - b r^2)$ matched in value and slope at $r_s$, and keep the difference,

```{math}
R_2(r) = \begin{cases} R(r) - r^{l}(a - b r^{2}), & r < r_s \\ 0, & r \ge r_s. \end{cases}
```

$R_2$ is strictly shorter-ranged than $R$ and vanishes smoothly at $r_s$.
*What* to split and *where* is a convention. `zeta_split` names the two schemes,
and each carries its own parameter, so the split radii follow from `zeta_split`
alone. This page owns the option; the PAW-LCAO families take the same key.

| | `"first_zeta"` (**default**) | `"last_zeta"` |
| :--- | :--- | :--- |
| the number is | the **norm** of the tail, $\bigl(\int_{r_s}^{r_c}\lvert R\rvert^2 r^2dr\bigr)^{1/2}$ | its **squared norm** |
| default | `(0.15, 0.31, 0.54)` for zetas 2, 3, 4 | `0.15`, halved for each further zeta |
| each zeta splits | the **first** zeta | the **previous** zeta |

```python
basis = {"name": "NAO", "size": "TZP"}                             # first_zeta
basis = {"name": "NAO", "size": "TZP", "zeta_split": "last_zeta"}  # last_zeta
```

The two sets of numbers are not comparable digit for digit: a tail norm of
$0.15$ is a squared-norm fraction of $0.0225$, so the extra zetas of `first_zeta`
are longer-ranged. For `"first_zeta"`, `zeta_split` also takes a number or a
sequence of tail norms in $(0, 1)$. The scheme is part of the model: it changes
the energy and the equilibrium geometry. Polarization shells are solved in the
same confining sphere at $l_{\max}+1$, using the lowest principal quantum number
that angular momentum allows, so they stay compact.

```{warning}
Extra zetas are short-ranged by construction, and every basis function is
sampled on the same uniform real-space grid. A third or fourth zeta of hydrogen
is only a few grid points wide at ordinary spacings; a function that narrow is
aliased, not represented. Refine `grid_spacing` before paying for `TZ` or `QZ`;
`DZP` is the practical choice. Multiple zeta also approaches linear dependence,
since each added function overlaps the one it was split from.
```

`examples/basis_nao/01_nao_multizeta_h2.py` runs H₂ at Hartree–Fock in SZ, DZ,
TZ and DZP; read its output for the energy ladder at the grid you choose.

(basis-gaussian-families)=
## Named Gaussian families

The standard names are accepted directly, and each is generated natively: the
name is parsed into its structure (core contraction length, valence split,
polarization, diffuse and core-correlating functions) and the exponents and
contraction coefficients are produced for the atom at hand from cached
Slater-orbital fits. No basis-set table is read.

```python
Mandacaru(method="rhf", basis="cc-pVDZ")
Mandacaru(method="rhf", basis="6-311+G(2df,2p)")
```

| Family | Names |
| :--- | :--- |
| Minimal | `STO-3G`, `STO-4G`, `STO-5G`, `STO-6G` |
| Pople | `3-21G`, `6-31G`, `6-31G*`, `6-31+G*`, `6-311G`, `6-311+G(2df,2p)`, and other names of the same pattern |
| Dunning | `cc-pVDZ`, `cc-pVTZ`, `cc-pVQZ`, `cc-pV5Z`, `aug-cc-pVDZ`, `cc-pCVDZ`, and other `cc-p[C]VXZ` names |
| Karlsruhe | `def2-SV(P)`, `def2-SVP`, `def2-SVPD`, `def2-TZVP`, `def2-TZVPD`, `def2-TZVPP`, `def2-QZVP`, and their `D`/`PP` variants |

What is reproduced is the **shell structure and function count** of the
published set: `cc-pVTZ` carbon is `[4s3p2d1f]` (30 functions), `def2-TZVP`
carbon `[5s3p2d1f]`. The published exponents, which come from molecular energy
optimizations, are not reproduced: Mandacaru's own are a partitioned
Slater-orbital fit with geometrically spread polarization exponents. Treat a
named basis as a self-contained basis *of the same size and shape* as its
namesake, right for qubit-count planning and for comparing solvers across
sizes, and not as a drop-in for a literature number quoted in that basis. Shells
are spherical (5 `d`, 7 `f`), so `6-31G*` carbon has 14 functions, not the 15 of
a Cartesian-`d` program. A named recipe takes no extra options.

(basis-hao-virtual)=
## Virtual levels for the HAO basis

`HAO` has one analytic hydrogenic orbital per **occupied** subshell, carrying the
atom's bare nuclear charge (H → 1s; Li → 1s, 2s; C → 1s, 2s, 2p). A correlated
method then has almost nothing to correlate *into*: H₂ is 2 spatial orbitals,
4 qubits and a single double excitation. `virtual_orbitals` adds the $k$ lowest
**unoccupied** subshells of each atom, in aufbau order and with the same bare
charge; it is the only option `HAO` accepts:

```python
Mandacaru(method="adapt-vqe", basis={"name": "HAO", "virtual_orbitals": 1},
          grid_spacing=0.30)
```

A partially filled subshell counts as occupied, so carbon's first virtual level
is `3s`. A level is a whole subshell, because half a shell would break the
atom's spherical symmetry and make the energy depend on the orientation in the
box, so one virtual `p` level adds three functions. On hydrogen:

| `virtual_orbitals` | levels added | functions per H | H₂ qubits |
|---|---|---|---|
| `0` | – | 1 | 4 |
| `1` | 2s | 2 | 8 |
| `2` | 2s, 2p | 5 | 20 |
| `3` | 2s, 2p, 3s | 6 | 24 |

A smaller basis is a strict subset of the larger, so the correlated energy can
only fall. A virtual hydrogenic orbital is **diffuse**: H `2s` has
$\langle r\rangle = 6\,a_0$, so the cell must contain its tail or the grid clips
it.

(basis-per-element)=
## A different basis on different elements

The `basis` argument also takes a **per-element mapping** of the all-electron
families: a dictionary keyed by chemical symbol (or atomic number), plus an
optional `"*"` default, each entry a basis specification of its own. The qubit
count is the sum over atoms, so a polarized basis on the atoms whose chemistry
matters and a minimal one on spectators keeps a large system inside a
state-vector budget:

```python
basis = {"O": {"name": "NAO", "size": "DZP"},
         "H": {"name": "NAO", "size": "DZP"},
         "*": "HAO"}                              # every other element
Mandacaru(method="adapt-vqe", basis=basis)
```

PAW-LCAO bases cannot appear in a mapping.
