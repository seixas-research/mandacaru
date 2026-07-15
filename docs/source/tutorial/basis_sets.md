# Localized basis sets: NAO and GTO

Beyond the analytic `HydrogenicOrbital`, Carcará provides two families of
localized basis functions, both implementing the same `BasisFunction` contract
so they plug straight into `IntegralEngine`:

* **NAO** -- confined *Numerical Atomic Orbitals* (Sankey/SIESTA-type);
* **GTO** -- *Contracted Gaussian-Type Orbitals* from standard families.

The `BasisSet` factory builds either by name. Lengths are in Ångström and
energies in eV (the user-facing units).

```python
from carcara.basis import BasisSet

nao = BasisSet.build(method="NAO", energy_shift=0.03)
gto = BasisSet.build(method="GTO", name="6-31G(d)")
```

## Numerical Atomic Orbitals

An NAO is a numerical radial function times a spherical harmonic,
$\psi_{nlm} = R_{nl}(r)\,Y_l^m$, where $R_{nl}$ is obtained by solving the
radial Schrödinger equation **inside a hard-wall sphere** of radius $r_c$, so
the orbital is strictly zero for $r \ge r_c$. The cutoff follows from a
user-specified *energy shift* — the amount the confinement raises the free-atom
level:

```{math}
\delta E = \frac{\pi^2 \hbar^2}{2 m r_c^2}
\quad\Longrightarrow\quad
r_c = \frac{\pi}{\sqrt{2\,\delta E}} \;\;(\text{atomic units}).
```

`energy_shift` defaults to `0.03` eV; `energy_shift_to_rc` performs the
conversion. The radial equation is solved by finite differences on a uniform
grid with a *screened* (Slater) charge, and the samples are cubic-spline
interpolated up to $r_c$, where they taper smoothly to zero.

```python
from carcara.basis import energy_shift_to_rc

energy_shift_to_rc(0.03)          # ~66.9 Bohr
orbitals = nao.atom("C")          # valence 2s + 2p NAOs for carbon
orbitals[0].r_c, orbitals[0].energy   # cutoff (Bohr) and confined energy (Ha)
```

## Gaussian-Type Orbitals

A contracted Gaussian of angular momentum $l$ is
$\chi(\mathbf r) = R(r)\,Y_l^m$ with
$R(r) = \sum_i c_i\, N(\alpha_i, l)\, r^l e^{-\alpha_i r^2}$; each primitive is
normalized and the contraction renormalized so $\int |R|^2 r^2 dr = 1$.

Seven families spanning the main design philosophies ship built in, each
covering the elements **H–Ar** (spherical harmonics, from the
[Basis Set Exchange](https://www.basissetexchange.org)):

| Group | Families | Design |
|---|---|---|
| Pople | `STO-3G`, `6-31G(d)`, `6-311G(d,p)` | minimal / split-valence + polarization |
| Dunning | `cc-pVDZ`, `cc-pVTZ` | correlation-consistent (post-HF) |
| Karlsruhe | `def2-SVP`, `def2-TZVP` | DFT-optimized |

List them with `carcara.basis.available_bases()`. The parser handles segmented
*and* general contractions and shells up to `f`, so triple-zeta sets work too:

```python
BasisSet.build(method="GTO", name="6-31G(d)").atom("C")   # 14 funcs: 3s+6p+5d
BasisSet.build(method="GTO", name="cc-pVTZ").atom("C")    # 30 funcs: +f shell
```

Any other element/family can be added from a Basis Set Exchange download in
NWChem/Gaussian94 format:

```python
from carcara.basis import register

register("my-basis", nwchem_text)     # then BasisSet.build(method="GTO", name="my-basis")
```

```{note}
Real-space grids cannot resolve the ultra-tight *core* Gaussians of
all-electron triple-zeta sets (exponents in the thousands); those functions are
still analytically normalized but under-sampled on a coarse grid. Diffuse and
valence functions integrate accurately.
```

## Feeding the engine

Build a whole geometry with `molecule` and hand the list to `IntegralEngine`
exactly as with any other basis:

```python
from carcara.integrals import Grid, IntegralEngine, Potentials

basis = gto.molecule(["H", "H"], [[0, 0, 0], [0, 0, 0.74]])
grid = Grid(center=[0.0, 0.0, 0.0], box_size=6.0, h=0.12)
engine = IntegralEngine(basis, grid)

T, V = engine.one_body(Potentials([(1.0, [0, 0, -0.37]),
                                   (1.0, [0, 0, 0.37])]).nuclear_potential)
eri = engine.two_body(method="fft")
```

The complete runnable script is in
[`examples/basis_sets.py`](https://github.com/seixas-research/carcara/blob/main/examples/basis_sets.py).
