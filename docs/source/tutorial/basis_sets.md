# Localized basis sets: NAO and GTO

Beyond the analytic `HydrogenicOrbital`, Carcará provides two families of
localized basis functions, both implementing the same `BasisFunction` contract
so they plug straight into `IntegralEngine`:

* **NAO** -- confined *Numerical Atomic Orbitals* (Sankey/SIESTA-type);
* **GTO** -- a minimal *STO-nG Contracted Gaussian* basis, generated from
  scratch (no tabulated basis-set data).

Both are built by the `BasisSet` factory and generated natively. Lengths are in
Ångström and energies in eV (the user-facing units).

```python
from carcara.basis import BasisSet

nao = BasisSet.build(method="NAO", energy_shift=0.03)
gto = BasisSet.build(method="GTO", n_gaussians=3)   # STO-3G-like
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

The GTO family is a **minimal STO-nG basis generated from scratch** — it needs
no tabulated basis-set data. For every occupied subshell $(n, l)$ of the atom
(core *and* valence) a Slater-type orbital
$S_{nl}(r) \propto r^{\,n-1} e^{-\zeta r}$, with Slater's-rules exponent
$\zeta = (Z - s)/n^*$, is approximated by a fixed contraction of `n_gaussians`
Gaussians,

```{math}
S_{nl}(r) \;\approx\; \sum_{i=1}^{n} c_i\, N(\alpha_i, l)\, r^l e^{-\alpha_i r^2},
```

whose exponents and coefficients come from a least-squares fit to $S_{nl}$
(weighted by $r^2\,dr$). The fit is scale-covariant, so it is done once at the
reference $\zeta = 1$ (cached) and the exponents are rescaled by $\zeta^2$ for
the actual atom. This reproduces the standard published STO-nG contractions
closely — e.g. the hydrogen 1s STO-3G exponents $2.22766, 0.40577, 0.10982$ to
five figures.

```python
BasisSet.build(method="GTO", n_gaussians=3).atom("C")   # 5 funcs: 1s+2s+2p
BasisSet.build(method="GTO", n_gaussians=6).atom("H")   # 1 func, 6 primitives
```

The count is the minimal-basis size (one contraction per occupied subshell);
`n_gaussians` only sets how many primitives each contraction uses. Any atom whose
occupied shells have a defined Slater $n^*$ ($n \le 6$) is supported.

```{note}
Real-space grids cannot fully resolve the ultra-tight *core* Gaussians of heavy
atoms (e.g. the argon 1s); those functions are still analytically normalized but
under-sampled on a coarse grid. Diffuse and valence functions integrate
accurately.
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
