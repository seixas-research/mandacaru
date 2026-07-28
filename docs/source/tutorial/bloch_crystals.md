# Periodic Systems with the Bloch Calculator

Molecules are open-boundary systems, but crystals are periodic. Carcará extends to
**1-, 2- and 3-dimensional crystals** through a single Bloch / k-point calculator,
{class}`~carcara.algorithms.BlochCalculator`, whose correlated supercell solver is
selected with `method=`:

| `method=` | Supercell solver |
| :--- | :--- |
| `"vqe"` | fixed UCCSD ansatz (VQE) |
| `"adapt-vqe"` | adaptive ansatz (ADAPT-VQE) |
| `"vasqe"` | stochastic, annealed selection (VASQE) |

The calculator does two things: it solves the single-particle **Bloch Hamiltonian**
at each k-point for the band structure, and it computes a **total energy that uses
all k-points** with the selected molecular method. The band structure is
single-particle, so it is **identical across all three methods** — it is
method-independent. Only the correlated total energy differs by method.

The crystal is given as an ASE `Atoms` **primitive cell** — `atoms.cell` sets the
lattice vectors and `atoms.pbc` selects which directions are periodic.

---

## A one-dimensional hydrogen chain

```python
import numpy as np
from ase import Atoms
from carcara.algorithms import BlochCalculator

# One H per cell; periodic along x, vacuum in y and z.
atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]],
              cell=[[1.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
              pbc=[True, False, False])

bloch = BlochCalculator(atoms, method="vqe", basis="FAO", mapping="jordan_wigner",
                        n_cells=4, n_images=7, h=0.20)
print(bloch.dimension, "D crystal,", bloch.n_bands, "band(s)")
```

`n_cells` sets how many lattice translations enter the Bloch sum, `n_images` the
width of the nuclei window used to build the periodic potential, and `h` the
real-space grid spacing (Ångström). The constructor is the same for every method —
swap `method="vqe"` for `method="adapt-vqe"` or `method="vasqe"`.

---

## Band structure

The band structure is the generalized Bloch eigenproblem

$$H(\mathbf{k})\,c = \varepsilon(\mathbf{k})\,S(\mathbf{k})\,c , \qquad
H(\mathbf{k}) = \sum_{\mathbf{R}} e^{2\pi i\,\mathbf{k}\cdot\mathbf{n}}\, h^{(\mathbf{R})} ,$$

solved at each k-point from the real-space cell-to-cell overlap/one-body blocks.
Because it is single-particle, **every method returns the same bands**. Pass
**fractional** k-points to `bands`, or use `band_structure` for a high-symmetry
path (k-points generated with ASE):

```python
# Energies (eV) at explicit fractional k-points.
energies = bloch.bands([[0.0, 0, 0], [0.25, 0, 0], [0.5, 0, 0]])   # (3, n_bands)

# Or a full path with ASE high-symmetry labels.
bs = bloch.band_structure("GX", npoints=201)
print(bs.energies.shape, bs.labels)        # (201, n_bands), ['G', 'X']

# A Monkhorst-Pack mesh (via ASE), Gamma-centered.
mesh = bloch.monkhorst_pack((10, 1, 1))
band_mp = bloch.bands(mesh)[:, 0]
```

`band_structure` returns a {class}`~carcara.algorithms.BandStructure` with the
cumulative k-axis (`x`), band `energies` (eV), and the `xticks`/`labels` of the
high-symmetry points — ready to save to CSV or plot.

---

## Total energy using all k-points

A correlated solver **cannot** be run independently per k-point and summed: the
two-electron interaction couples crystal momenta ($\mathbf{k}_1+\mathbf{k}_2 =
\mathbf{k}_3+\mathbf{k}_4+\mathbf{G}$). The exact way to fold **all** k-points into
a correlated total energy is the **Born–von Kármán equivalence**: an
$(n_1, n_2, n_3)$ Monkhorst-Pack mesh is a $\Gamma$-point calculation on the
$(n_1, n_2, n_3)$ supercell, so

$$E_\text{cell} = \frac{E_\Gamma\big(\text{supercell}\big)}{n_\text{cells}} .$$

`total_energy` builds that supercell with `atoms.repeat`, runs it through the
selected molecular method (the box is the supercell's own cell), and returns
`(energy_per_cell_eV, result)` — the `result` being a `VQEResult`,
`ADAPTVQEResult`, or `VASQEResult` depending on the method:

```python
# Fixed-ansatz VQE.
e_cell, res = BlochCalculator(atoms, method="vqe", h=0.20).total_energy(
    (4, 1, 1), optimizer="L-BFGS-B")

# Adaptive ADAPT-VQE (extra adaptive controls).
e_cell, res = BlochCalculator(atoms, method="adapt-vqe", h=0.20).total_energy(
    (4, 1, 1), max_iterations=10, gradient_tolerance=1e-3)
print(res.num_operators, "operators grown")
```

Extra keyword arguments are forwarded to the supercell solver, so which are
valid depends on the selected method (`optimizer`/`h` for all;
`pool`/`max_iterations`/`gradient_tolerance` for the adaptive ones). This is a finite-supercell estimate
that converges to the bulk total energy as the mesh is refined (exact in the
infinite-mesh limit). Note the supercell size — and therefore the qubit count —
grows with the mesh, so a dense mesh such as `(10, 1, 1)` (a 20-qubit supercell) is
heavy for exact state-vector simulation.

### Stochastic selection with method="vasqe"

`method="vasqe"` runs the crystal supercell through VASQE, so the ansatz grows by
**stochastic softmax selection** with an optional temperature **annealing**
schedule — useful for exploring the operator space on a larger supercell before
settling on the greedy (ADAPT) choice:

```python
e_cell, res = BlochCalculator(atoms, method="vasqe", h=0.20).total_energy(
    (4, 1, 1), temperature=2.0, final_temperature=0.02, schedule="exponential",
    max_iterations=10, gradient_tolerance=1e-3, seed=1)
print(res.temperatures)          # the selection temperature at each growth step
```

---

## Higher dimensions

The same interface handles 2-D and 3-D crystals — only `atoms.pbc` and the
k-points change. A square lattice of hydrogen:

```python
square = Atoms("H", positions=[[0.0, 0.0, 0.0]],
               cell=[[2.0, 0, 0], [0, 2.0, 0], [0, 0, 10.0]],
               pbc=[True, True, False])
bloch2d = BlochCalculator(square, method="vqe", basis="FAO",
                          n_cells=2, n_images=3, h=0.35)

# Bands along Gamma -> X -> M of the square Brillouin zone.
gamma, X, M = [0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0]
print(bloch2d.bands([gamma, X, M])[:, 0])
```

Runnable end-to-end scripts: `examples/07_ADAPTVQE_H_chain_bands.py` (band structure
+ total energy to CSV, with a separate plotting script) and
`examples/11_Bloch_crystals.py` (the three methods compared on the H chain).
