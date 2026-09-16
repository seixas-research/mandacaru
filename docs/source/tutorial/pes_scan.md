# LiH potential-energy curves: basis sets and operator pools

A potential-energy curve repeats a molecular energy calculation at several bond
lengths. For each Li–H distance $R$, we optimise the electronic state while
holding the nuclei fixed:

```{math}
E_{b,p}(R) = \min_{\boldsymbol{\theta}}
\langle\psi_p(\boldsymbol{\theta};R)|\hat H_b(R)
|\psi_p(\boldsymbol{\theta};R)\rangle.
```

Here $b$ labels the basis set and $p$ labels the operator pool. In practice the
optimiser returns an approximation to this minimum. We retain its convergence
information alongside every energy.

## What the comparison holds fixed

The supplied script calculates neutral LiH with a frozen lithium core, two
active electrons, the Jordan–Wigner mapping and local state-vector evaluation.
It compares Carcará's `STO-3G` and `3-21G` constructions, each with the
`fermionic`, `qubit`, `qeb` and `ceo` pools.

| Basis | Spatial orbitals before freezing | Active spatial orbitals | Qubits |
| :--- | ---: | ---: | ---: |
| `STO-3G` | 3 | 2 | 4 |
| `3-21G` | 5 | 4 | 8 |

The second basis adds radial flexibility. These native constructions omit
lithium p functions and use internally generated Gaussian exponents. They are
not interchangeable with published basis tables carrying the same names.
See [basis sets](../guide/basis_sets.md).

At each geometry, the script builds one Hamiltonian per basis, saves it to a
temporary JSON cache and reuses it across pools. It creates a new Hamiltonian
when either the distance or basis changes.

## Keep the grid fixed

The scan uses a single box extending from −4.8 to +4.8 Å along each axis and a
requested spacing of 0.12 Å. With symmetric nuclear positions at $z=\pm R/2$,
the distance step is twice the **actual** grid spacing:

```python
import numpy as np
from carcara.integrals import Grid
from carcara.units import BOHR_TO_ANGSTROM

grid = Grid(center=[0.0, 0.0, 0.0], box_size=4.8, h=0.12)
spacing = grid.dz * BOHR_TO_ANGSTROM  # Grid stores its spacing in bohr.
distances = 1.12 + 2 * spacing * np.arange(8)
```

Each nucleus therefore moves by one whole grid interval between successive
geometries. This preserves its alignment relative to the nodes and reduces
changes in core sampling, often called the *egg-box effect*. It does not remove
integration error or establish convergence. Freezing the core reduces the
variational problem, but its integrals still need adequate resolution.

## Reproduce the data and PNG

From a source checkout with Carcará installed, run:

```bash
python examples/30_LiH_basis_pool_scan.py
```

The script writes these files into `docs/source/_static/lih/`:

- `energies.csv`: total energy, reference energy, operator count, qubit count
  and convergence information for every basis, pool and distance, plus the
  probability of finding one active electron of each spin.
- `metadata.json`: package versions, grid settings, distances and solver controls.
- `basis_pool_scan.png`: the comparison shown below.

To redraw the saved CSV without rerunning any calculations:

```bash
python examples/30_LiH_basis_pool_scan.py --plot-only
```

To test another grid without overwriting the documentation figure:

```bash
python examples/30_LiH_basis_pool_scan.py --spacing 0.10 --output-dir /tmp/lih-finer-grid
```

Changing the spacing also changes the grid-aligned distances. Compare only
matching geometries, or adapt the distance list when performing a systematic
convergence study. Check a larger box independently of the grid spacing.

```{figure} ../_static/lih/basis_pool_scan.png
:name: lih-basis-pool-comparison
:alt: LiH total energy against Li–H distance for two generated basis sets and four ADAPT-VQE pools, with differences from the fermionic pool below.
:class: molecular-plot
:width: 100%

Calculated LiH energies for two basis sets and four operator pools. The upper
panels show total energies; the lower panels show differences from the
fermionic-pool calculation in the same basis at the same distance. Curves may
overlap. Black crosses identify an unmet outer convergence criterion or an
unsuccessful inner optimisation. Lines connect calculated points only.
```

Download the {download}`PNG <../_static/lih/basis_pool_scan.png>`,
{download}`CSV <../_static/lih/energies.csv>` and
{download}`calculation settings <../_static/lih/metadata.json>`.

## Read the figure

First compare pools **within one column**: the Hamiltonian is identical there.
The lower panel makes small differences visible; its fermionic reference is a
variational calculation, not an exact-diagonalisation benchmark. `qeb` and `ceo`
are expected to coincide for the Jordan–Wigner construction used here.

Then compare the two upper panels: a different basis changes the approximate
Hamiltonian and the number of qubits. A lower energy alone does not establish
better physical accuracy when the numerical integrals are not converged.

These are calculated results for a small teaching model, **not a converged
spectroscopic potential**. The figure does not establish the experimental
bond length, dissociation energy or chemical accuracy. Check the basis, grid,
box size, frozen-core approximation, final electron number and optimisation
before making those claims.

The plotted quantity is the total energy. If you need a binding curve, define
consistent fragment calculations and subtract their energies:

```{math}
\Delta E_b(R) = E_{\mathrm{LiH},b}(R)
- E_{\mathrm{Li},b} - E_{\mathrm{H},b}.
```

Use compatible basis, grid, potential and core conventions, and account for the
open-shell isolated atoms. `reference_energy` is a molecular reference
determinant, not the sum of isolated-atom energies. The helper
`examples/pes_utils.py::atomic_reference` already returns **eV**; do not convert
it from Hartree a second time.

The supplied `3-21G` calculation reports grid-resolution warnings for a compact
Gaussian function; freezing the core does not remove that numerical issue.
The qubit-pool searches can also exhaust the 40-step growth budget. Both are
limitations of this demonstration. The CSV's `particle_sector_weight` should
be close to one for the intended two-electron problem; inspect it separately
from the energy and optimiser flags.

## Embed the plot in your own documentation

In MyST Markdown, use a figure directive so the caption and alternative text
remain attached to the image:

````markdown
```{figure} ../_static/lih/basis_pool_scan.png
:alt: LiH energy against bond length for two basis sets and four operator pools.
:width: 100%

LiH basis-set and operator-pool comparison; energies in eV and distances in Å.
```
````

The equivalent reStructuredText is:

```rst
.. figure:: ../_static/lih/basis_pool_scan.png
   :alt: LiH energy against bond length for two basis sets and four operator pools.
   :width: 100%

   LiH basis-set and operator-pool comparison; energies in eV and distances in Å.
```

Paths are relative to the page containing the directive. The image is committed
with the documentation; Read the Docs does not run the scientific calculation.

## Complete script

```{literalinclude} ../../../examples/30_LiH_basis_pool_scan.py
:language: python
```
