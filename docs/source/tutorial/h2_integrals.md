# One- and two-body integrals for H2

Carcará ships a basis-agnostic real-space integral engine in
`carcara.integrals`. Given any set of localized basis functions and a
uniform grid, it produces the one-body (kinetic and external-potential) matrices
and the two-body electron-repulsion tensor that define a second-quantized
Hamiltonian.

This tutorial builds the smallest chemically meaningful example: the hydrogen
molecule H2 in a minimal basis of one hydrogen 1s orbital on each proton.

## Geometry and external potential

We work in atomic units (Bohr, Hartree). The two protons sit symmetrically
about the origin at the equilibrium bond length `R = 1.4 a0`. The external
potential felt by the electrons is the sum of the two nuclear Coulomb wells:

```{math}
V(\mathbf r) = -\sum_A \frac{Z}{|\mathbf r - \mathbf R_A|}.
```

```python
import numpy as np

Z, R = 1.0, 1.4
nuclei = np.array([[0.0, 0.0, -R / 2], [0.0, 0.0, +R / 2]])

def nuclear_potential(x, y, z):
    v = np.zeros_like(x, dtype=float)
    for Rx, Ry, Rz in nuclei:
        r = np.sqrt((x - Rx) ** 2 + (y - Ry) ** 2 + (z - Rz) ** 2)
        v -= Z / np.maximum(r, 1e-12)
    return v
```

## Grid, basis and engine

The `Grid` is a uniform cubic box; it must be large enough to contain the
orbital tails. The basis is a list of `HydrogenicOrbital` objects, one centered
on each proton.

```python
from carcara.basis import HydrogenicOrbital
from carcara.integrals import Grid, IntegralEngine

grid = Grid(center=[0.0, 0.0, 0.0], box_size=10.0, points=64)
basis = [HydrogenicOrbital(1, 0, 0, Z=Z, center=nuclei[0]),
         HydrogenicOrbital(1, 0, 0, Z=Z, center=nuclei[1])]

engine = IntegralEngine(basis, grid)
```

## One-body integrals

`IntegralEngine.one_body` returns the kinetic matrix
`T[a,b] = <a| -1/2 nabla^2 |b>` and the external-potential matrix
`V[a,b] = <a| V |b>`. Their sum is the one-body core Hamiltonian.

```python
T, V = engine.one_body(nuclear_potential)
h_core = T + V
```

## Two-body integrals

`IntegralEngine.two_body` returns the electron-repulsion tensor `(ab|cd)` in
the chemists' convention. The default
`method="fft"` uses an O(N log N) FFT Poisson solver.

```python
eri = engine.two_body(method="fft")

print("Core Hamiltonian h = T + V (Ha):")
print(h_core.real)
print(f"(00|00) on-site repulsion = {eri[0, 0, 0, 0].real:.4f} Ha")
print(f"(00|11) inter-site Coulomb = {eri[0, 0, 1, 1].real:.4f} Ha")
```

## Checking the result

The on-site integral `(00|00)` is the self-repulsion of a single hydrogen 1s
orbital, whose exact value is `5/8 = 0.625 Ha`. On the grid above the engine
returns `~0.62 Ha`, and it converges toward the exact value as the grid is
refined. The complete runnable script is available in
[`examples/h2_integrals.py`](https://github.com/seixas-research/carcara/blob/main/examples/h2_integrals.py).

The resulting `T`, `V` and `(ab|cd)` arrays are exactly the inputs needed to
assemble the fermionic Hamiltonian and map it to qubits for a VQE calculation.
