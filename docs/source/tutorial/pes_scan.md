# A native potential-energy scan

A potential-energy curve compares independently solved geometries with the
same basis, integration grid and electronic approximation. This short H2
example uses native PBE DFT; it checks the ASE interface without a long quantum
optimization at each point.

```python
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from mandacaru import Mandacaru
from mandacaru.integrals import Grid

# Explicitly retain one integration box throughout the scan.
grid = Grid(center=[2.5, 2.5, 2.5], cell=np.diag([5., 5., 5.]), grid_spacing=.35)
distances = np.linspace(.6, 1.4, 9)
energies = []
for distance in distances:
    atoms = Atoms("H2", positions=[[2.5, 2.5, 2.5-distance/2],
                                  [2.5, 2.5, 2.5+distance/2]], cell=[5.]*3)
    atoms.calc = Mandacaru(method="dft", xc="pbe", basis="HAO",
                           grid=grid, trace=False)
    energies.append(atoms.get_potential_energy())

np.savetxt("h2-pbe.csv", np.column_stack([distances, energies]),
           delimiter=",", header="distance_angstrom,energy_ev", comments="")
plt.plot(distances, energies, "o-")
plt.xlabel("H–H distance (angstrom)")
plt.ylabel("Total energy (eV)")
plt.tight_layout()
plt.savefig("h2-pbe.png", dpi=180)
```

Each point builds and solves a new native electronic problem. The lines only
connect sampled values; they are not a fitted spectroscopic potential. Repeat
with finer grid spacing, a larger cell and a more flexible basis before
interpreting the minimum or a dissociation energy.

Moving atoms relative to grid nodes can produce sampling oscillations. A fixed
box makes these visible instead of mixing them with changes in box geometry.
Check analytic forces on a fixed grid against sufficiently small displacements
when assessing a derivative.

A quantum scan follows the same geometry loop with `method="vqe"` or
`method="adapt-vqe"`. Converge every optimizer and outer adaptive loop before
comparing its energies. A cached Hamiltonian belongs to one geometry and must
not be carried across this scan.
