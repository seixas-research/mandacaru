# Potential-energy surfaces and basis sets

A potential-energy surface (PES) is the molecular energy as a function of
geometry. Scanning a diatomic's bond length across several basis sets shows both
the bonding well and how the basis-set choice shifts it. Carcará builds these
curves from its real-space integral engine and restricted Hartree-Fock (RHF)
solver, over three **natively generated** basis sets:

* `hydrogenic` — one analytic hydrogen-like orbital per occupied subshell;
* `STO-3G` — the minimal Gaussian basis (`GTO`, three primitives);
* `6-31G(d)` — the native Pople split-valence basis with `d` polarization.

None of these read tabulated basis-set data — the Gaussian families are least-
squares fits to Slater-type orbitals (see {doc}`basis_sets`).

## Building one point of the curve

Each geometry is a self-consistent-field calculation over the chosen basis:

```python
import numpy as np

from carcara.algorithms import RHF
from carcara.basis import BasisSet
from carcara.core import HydrogenicIntegrals
from carcara.integrals import Grid

sto3g = BasisSet.build("GTO", n_gaussians=3)

R = 0.74
positions = [np.array([0.0, 0.0, -R / 2]), np.array([0.0, 0.0, +R / 2])]
basis = sto3g.atom("H", center=positions[0]) + sto3g.atom("H", center=positions[1])

grid = Grid(center=[0.0, 0.0, 0.0], box_size=7.0, h=0.12)
integrals = HydrogenicIntegrals([(1.0, positions[0]), (1.0, positions[1])],
                                basis, grid)
rhf = RHF(integrals.one_body(), integrals.two_body(), n_electrons=2).run()
energy = rhf.electronic_energy + integrals.nuclear_repulsion
```

## The scripts

Two turnkey scripts scan the whole curve and export a tidy CSV, and a separate
plotting script renders the multi-basis comparison:

```console
python examples/generate_h2_pes.py     # -> data/h2_pes_data.csv
python examples/generate_lih_pes.py    # -> data/lih_pes_data.csv
python examples/plot_pes.py            # -> data/h2_pes.png, data/lih_pes.png
```

Each curve is referenced to the sum of isolated-atom energies (computed with
unrestricted Hartree-Fock, so open-shell atoms like H and Li are handled), so
`E = 0` is the separated-atom limit and the well depth is a binding energy. For
H₂ the curves show the expected ordering — the larger `6-31G(d)` basis reaches a
deeper (lower) energy than minimal `STO-3G`.

## A note on the real-space grid

The engine integrates on a *uniform* grid, so a nucleus that sits between grid
nodes samples its $-Z/r$ cusp slightly differently than one on a node (the
"egg-box" effect). For a light, core-less atom (H) this is negligible and H₂ is
quantitative. For a tight $1s$ core (Li) it is large, so the LiH scripts

* step the bond length by **twice the grid spacing**, keeping both nuclei on a
  fixed sub-node alignment along the scan, and
* place the isolated-atom references at their molecular grid positions,

which together yield a smooth curve with a bound minimum near the experimental
1.6 Å. The absolute LiH well depth remains grid-limited by the under-resolved Li
core and should be read qualitatively.

```{note}
These PES curves use RHF because it scales to every basis set (an exact
state-vector VQE/ADAPT run is limited to a handful of qubits). The correlated
solvers ({doc}`vqe`, {doc}`adapt_vqe`) target the same Hamiltonians on small
molecules.
```
