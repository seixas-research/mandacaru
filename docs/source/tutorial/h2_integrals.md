# One- and two-body integrals for H2

Carcará ships a basis-agnostic real-space integral engine in
`carcara.integrals`. Given any set of localized basis functions and a
uniform grid, it produces the one-body (kinetic and external-potential) matrices
and the two-body electron-repulsion tensor that define a second-quantized
Hamiltonian.

This tutorial builds the smallest chemically meaningful example: the hydrogen
molecule H2 in a minimal basis of one hydrogen 1s orbital on each proton.

## Units

Carcará computes internally in atomic units (Bohr, Hartree), but the user-facing
API prefers the everyday chemistry units: **lengths in Ångström** and
**energies in electronvolts**. Every class below therefore takes coordinates in
Ångström and the engine returns energies in eV by default. Pass
`units="bohr"` / `energy_units="Ha"` to work in atomic units instead.

## Geometry and external potential

The two protons sit symmetrically about the origin at the equilibrium bond
length `R = 0.74 Å`. The external potential felt by the electrons is the sum of
the two nuclear Coulomb wells:

```{math}
V(\mathbf r) = -\sum_A \frac{Z}{|\mathbf r - \mathbf R_A|}.
```

The `Potentials` class builds this callable from a list of `(Z, center)` pairs;
its `nuclear_potential` method is exactly the `V(x, y, z)` signature the engine
expects.

```python
import numpy as np

from carcara.integrals import Potentials

Z, R = 1.0, 0.74  # Angstrom
proton_a = np.array([0.0, 0.0, -R / 2])
proton_b = np.array([0.0, 0.0, +R / 2])

potentials = Potentials([(Z, proton_a), (Z, proton_b)])
```

## Grid, basis and engine

The `Grid` is a uniform cubic box; it must be large enough to contain the
orbital tails. It is specified by a physical spacing `h` (in Ångström, default
`0.20`) rather than a node count -- the number of points per dimension is
derived from `h` and `box_size`. The basis is a list of `HydrogenicOrbital`
objects, one centered on each proton.

```python
from carcara.basis import HydrogenicOrbital
from carcara.integrals import Grid, IntegralEngine

grid = Grid(center=[0.0, 0.0, 0.0], box_size=5.0, h=0.10)  # Angstrom
basis = [HydrogenicOrbital(1, 0, 0, Z=Z, center=proton_a),
         HydrogenicOrbital(1, 0, 0, Z=Z, center=proton_b)]

engine = IntegralEngine(basis, grid)
```

## One-body integrals

`IntegralEngine.one_body` returns the kinetic matrix
`T[a,b] = <a| -1/2 nabla^2 |b>` and the external-potential matrix
`V[a,b] = <a| V |b>`. Their sum is the one-body core Hamiltonian.

```python
T, V = engine.one_body(potentials.nuclear_potential)
h_core = T + V
```

## Two-body integrals

`IntegralEngine.two_body` returns the electron-repulsion tensor `(ab|cd)` in
the chemists' convention. The default
`method="fft"` uses an O(N log N) FFT Poisson solver.

```python
eri = engine.two_body(method="fft")

print("Core Hamiltonian h = T + V (eV):")
print(h_core.real)
print(f"(00|00) on-site repulsion = {eri[0, 0, 0, 0].real:.3f} eV")
print(f"(00|11) inter-site Coulomb = {eri[0, 0, 1, 1].real:.3f} eV")
```

## Checking the result

The on-site integral `(00|00)` is the self-repulsion of a single hydrogen 1s
orbital, whose exact value is `5/8 Ha = 17.007 eV`. On the grid above the engine
returns `~17.0 eV`, and it converges toward the exact value as the grid is
refined. The complete runnable script is available in
[`examples/h2_integrals.py`](https://github.com/seixas-research/carcara/blob/main/examples/h2_integrals.py).

The resulting `T`, `V` and `(ab|cd)` arrays are exactly the inputs needed to
assemble the fermionic Hamiltonian and map it to qubits for a VQE calculation.
