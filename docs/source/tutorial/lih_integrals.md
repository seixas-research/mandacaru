# One- and two-body integrals for LiH

The [H2 tutorial](h2_integrals.md) used a two-orbital homonuclear basis. Here we
scale the same `carcara.integrals` machinery to a multi-orbital, *heteronuclear*
system: lithium hydride, LiH. This introduces two new ingredients -- different
nuclear charges on the two centers, and more than one orbital per atom.

## Geometry and external potential

As in the [H2 tutorial](h2_integrals.md) we use the user-facing units: lengths
in Ångström, energies in eV. Lithium and hydrogen sit along the z-axis about the
origin at the equilibrium bond length `R = 1.595 Å`. Because the nuclei carry
different charges, the external potential sums Coulomb wells with the *true*
charges `Z_Li = 3` and `Z_H = 1`:

```{math}
V(\mathbf r) = -\sum_A \frac{Z_A}{|\mathbf r - \mathbf R_A|}.
```

The `Potentials` class takes a list of `(Z, center)` pairs -- here with the
*true* charges on each center -- and exposes the sum of Coulomb wells as its
`nuclear_potential` method.

```python
import numpy as np

from carcara.integrals import Potentials

Z_LI, Z_H, R = 3.0, 1.0, 1.595  # R in Angstrom
li_pos = np.array([0.0, 0.0, -R / 2])
h_pos = np.array([0.0, 0.0, +R / 2])

potentials = Potentials([(Z_LI, li_pos), (Z_H, h_pos)])
```

## Grid, basis and effective charges

The minimal basis carries three orbitals on lithium -- the 1s core plus the 2s
and 2p_z valence orbitals -- and one 1s orbital on hydrogen. A subtlety appears
here: a hydrogenic Li 1s built with the bare charge `Z = 3` is extremely
contracted and hard to resolve on a modest grid. We therefore give the *basis*
orbitals *effective* charges, while the *potential* above keeps the true
nuclear charges.

`FullAtomicOrbital.from_slater` derives the effective charge from **Slater's
rules** given the atomic number of the center (e.g. `Z_eff = 2.70` for Li 1s,
`1.30` for Li 2s/2p, `1.00` for H 1s). The static method
`FullAtomicOrbital.slater_effective_charge(atomic_number, n, l)` exposes the
same value directly.

```python
from carcara.basis import FullAtomicOrbital
from carcara.integrals import Grid, IntegralEngine

# A fine spacing (h = 0.10 Angstrom) resolves the contracted Li 1s core.
grid = Grid(center=[0.0, 0.0, 0.0], box_size=4.8, h=0.10)

labels = ["Li 1s", "Li 2s", "Li 2pz", "H 1s"]
basis = [FullAtomicOrbital.from_slater(1, 0, 0, atomic_number=3, center=li_pos),
         FullAtomicOrbital.from_slater(2, 0, 0, atomic_number=3, center=li_pos),
         FullAtomicOrbital.from_slater(2, 1, 0, atomic_number=3, center=li_pos),
         FullAtomicOrbital.from_slater(1, 0, 0, atomic_number=1, center=h_pos)]

engine = IntegralEngine(basis, grid)
```

## One-body integrals

`IntegralEngine.one_body` returns the kinetic matrix
`T[a,b] = <a| -1/2 nabla^2 |b>` and the nuclear-attraction matrix
`V[a,b] = <a| V |b>`. With four basis functions both are `4 x 4`; their sum is
the one-body core Hamiltonian.

```python
T, V = engine.one_body(potentials.nuclear_potential)
h_core = T + V
```

## Two-body integrals

`IntegralEngine.two_body` returns the electron-repulsion tensor `<ab|cd>` in the
physicists' convention -- now a `4 x 4 x 4 x 4` array. The default `method="fft"`
uses an O(N log N) FFT Poisson solver.

```python
eri = engine.two_body(method="fft")

print("Core Hamiltonian h = T + V (eV):")
print(h_core.real)
print(f"<00|00> Li 1s on-site  = {eri[0, 0, 0, 0].real:.3f} eV")
print(f"<33|33> H 1s on-site   = {eri[3, 3, 3, 3].real:.3f} eV")
print(f"<13|13> Li 2s - H 1s J = {eri[1, 3, 1, 3].real:.3f} eV")
```

## Checking the result

The hydrogen 1s on-site integral `<33|33>` is unaffected by its heteronuclear
neighbour and recovers the exact self-repulsion `5/8 Ha = 17.007 eV` (the engine
returns `~17.0 eV`). The lithium core integral `<00|00>` is larger because the
1s orbital is far more contracted, and the inter-site `<13|13>` term measures
the Coulomb repulsion between the Li 2s and H 1s charge clouds. The complete
runnable script is available in
[`examples/lih_integrals.py`](https://github.com/seixas-research/carcara/blob/main/examples/lih_integrals.py).

As in the H2 case, the resulting `T`, `V` and `<ab|cd>` arrays are exactly the
inputs needed to assemble the fermionic Hamiltonian and map it to qubits for a
VQE calculation.
