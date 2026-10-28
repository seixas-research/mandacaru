# Kohn–Sham density functional theory

The Rust molecular solver computes spin densities, Hartree convolution,
exchange-correlation energy and potentials, and the generalized Kohn–Sham
matrix. Python supplies geometry values, an optional ASE adapter and unit conversion.

## Molecular calculation

```python
from ase import Atoms
from mandacaru import Mandacaru

atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, .74]], cell=[5, 5, 5])
atoms.center()
from mandacaru.integrations.ase import ASECalculator

atoms.calc = ASECalculator(
    method="dft", xc="r2scan", basis="HAO",
    grid_spacing=.35, project_translation=False, trace=False,
    resources={"threads": 2, "memory_limit_bytes": 512 * 1024**2},
)
energy = atoms.get_potential_energy()  # eV, including nuclear repulsion
forces = atoms.get_forces()            # eV / angstrom
```

Use `spin=True` for an unrestricted collinear calculation and set the initial
magnetic moments on the ASE atoms. Converge both the basis and grid before
interpreting energy differences. `HAO` is a generated hydrogenic basis, not a
claim of basis-set completeness.

Gradient-dependent molecular functionals currently require an orthorhombic
isolated grid. Nonlocal field convolution itself also supports skew periodic
cells, but that does not enable their complete advanced-functional SCF path.

## Canonical functionals

Only the exact names below are accepted; alternative spellings and aliases
are rejected.

| Name | Native molecular contribution |
| --- | --- |
| `lda` | Slater exchange and continuous Perdew–Zunger correlation |
| `pbe` | PBE exchange and correlation, with the Perdew–Wang uniform-gas reference |
| `r2scan` | Spin-resolved r²SCAN exchange and correlation with kinetic-density response |
| `hse06` | Exchange-hole semilocal contribution and 25% screened exact exchange, with screening 0.11 Bohr⁻¹ |
| `pbe+rvv10` | PBE plus native rVV10 correlation, with b = 6.3 |
| `r2scan+rvv10` | r²SCAN plus native rVV10 correlation, with b = 11.95 |

The native point kernels compute an energy per volume and its seven partials
with respect to spin densities, three density-gradient contractions and two
kinetic-energy densities. The molecular solver forms these fields from its
current density matrices on every SCF evaluation.

### Meta-GGA response

For each spin channel,

$$
\tau_\sigma(\mathbf r)=\frac12\sum_{ab}D^\sigma_{ab}
\nabla\phi_a(\mathbf r)\cdot\nabla\phi_b(\mathbf r).
$$

The matrix includes the corresponding kinetic-density potential:

$$
V^\tau_{ab}=\frac12\int v_\tau(\mathbf r)
\nabla\phi_a(\mathbf r)\cdot\nabla\phi_b(\mathbf r)\,d\mathbf r.
$$

A fixed or guessed tau would define a different calculation. Native r²SCAN
uses differentiated orbital samples consistently in its energy, matrix and
stationary gradient. The discretized iso-orbital indicator has a smooth floor
below zero to keep noisy density tails bounded. Points below the grid density
threshold are excluded from the meta-GGA contribution.

### Screened hybrid

HSE06 needs two matching pieces: semilocal short-range exchange subtraction
and the screened exact-exchange operator. The Rust implementation integrates
the exchange hole with erfc screening and differentiates that quadrature.
The molecular solver builds screened Cholesky factors and contracts them into the
exchange matrix and includes the same operator in stationary gradients.
Evaluating a semilocal point alone is not a full hybrid SCF calculation.

### Nonlocal correlation

rVV10 uses logarithmic interpolation in a local inverse-length variable,
cardinal cubic splines and FFT convolution. The reciprocal-space kernel is
analytic. The transform is tabulated on a uniform wave-number mesh. The energy
and density/gradient partials use the same kernel, so the potential differentiates the discrete energy.

An isolated calculation zero-pads each dimension before convolution. Periodic
field evaluation convolves on the cell, with reciprocal aliases averaged over
the grid's lattice operations. All interpolation and FFT workspaces belong to
the Rust context. The periodic *SCF* integration currently accepts only LDA and
PBE; the nonlocal field kernel does not imply periodic SCF support.

## Forces

Native stationary molecular gradients include nuclear and external-potential
terms, basis response, overlap response and the functional's generalized
Kohn–Sham action. They require a converged, real collinear state and spectral
kinetic energy. They do not run displaced SCFs. Independent finite differences
of fixed-grid total energies check all four advanced molecular functional
choices, including the hybrid exchange and nonlocal terms.

Use a common grid when checking a geometry derivative. Re-centering or resizing
the integration grid between displacements changes the numerical problem.

## Periodic calculations

The native periodic engine supports LDA/PBE iterations, collinear spin,
metallic occupations and spectral observables. PAW geometry and dataset
assembly still use Python and are part of the remaining migration. Advanced
functionals, spinor response and complete electronic stress are not enabled
on this path. See [coverage](native_engine.md) before choosing a crystal
workflow.

## Dispersion correction

A complete native D4 implementation remains pending. It needs the published
reference dynamic polarizabilities and fitted element parameters, coordination
interpolation, charge equilibration and its geometry response, pair damping,
neutral-reference three-body coefficients and periodic sums. A pair potential
with guessed coefficients is not an implementation of that model.

The Rust calculator does not silently replace D4 with rVV10 or another
correction. The independent nonlocal correlation choices above are available
for native molecular calculations. The cutover audit tracks the D4 work.

## Low-level fields

```python
import numpy as np
from mandacaru import _native

context = _native.Context(threads=1)
# rho_up, rho_down, sigma_uu, sigma_ud, sigma_dd, tau_up, tau_down
variables = np.array([[.12, .08, .004, -.001, .003, .08, .06]])
points = context.xc_points("r2scan", variables)
# Columns: energy per volume, then seven partials; atomic units throughout.
```

`Context.exchange_correlation(grid, densities, functional=...,
kinetic_densities=...)` returns energy and two spin potentials. Meta-GGAs also
return `tau_potentials`. Kinetic densities must have shape `(2, ngrid)`.
`Context.nonlocal_correlation` returns energy and density/sigma partials for
the nonlocal contribution. Arrays must have the documented dimensions and
finite physical inputs; native validation rejects invalid fields.

## Scientific references

- Perdew and Zunger, *Physical Review B* **23**, 5048 (1981),
  [doi:10.1103/PhysRevB.23.5048](https://doi.org/10.1103/PhysRevB.23.5048).
- Perdew and Wang, *Physical Review B* **45**, 13244 (1992),
  [doi:10.1103/PhysRevB.45.13244](https://doi.org/10.1103/PhysRevB.45.13244).
- Perdew, Burke and Ernzerhof, *Physical Review Letters* **77**, 3865 (1996),
  [doi:10.1103/PhysRevLett.77.3865](https://doi.org/10.1103/PhysRevLett.77.3865).
- Furness and colleagues, *Journal of Physical Chemistry Letters* **11**, 8208
  (2020), [arXiv:2008.03374](https://arxiv.org/abs/2008.03374).
- Heyd, Scuseria and Ernzerhof, *Journal of Chemical Physics* **118**, 8207
  (2003), [doi:10.1063/1.1564060](https://doi.org/10.1063/1.1564060).
- Ernzerhof and Perdew, *Journal of Chemical Physics* **109**, 3313 (1998),
  [doi:10.1063/1.476928](https://doi.org/10.1063/1.476928).
- Vydrov and Van Voorhis, *Journal of Chemical Physics* **133**, 244103 (2010),
  [arXiv:1009.1421](https://arxiv.org/abs/1009.1421).
- Sabatini, Gorni and de Gironcoli, *Physical Review B* **87**, 041108 (2013),
  [doi:10.1103/PhysRevB.87.041108](https://doi.org/10.1103/PhysRevB.87.041108).
- Román-Pérez and Soler, *Physical Review Letters* **103**, 096102 (2009),
  [doi:10.1103/PhysRevLett.103.096102](https://doi.org/10.1103/PhysRevLett.103.096102).
- Caldeweyher and colleagues, *Journal of Chemical Physics* **150**, 154122
  (2019), [doi:10.1063/1.5090222](https://doi.org/10.1063/1.5090222), and
  *Physical Chemistry Chemical Physics* **22**, 8499 (2020),
  [doi:10.1039/D0CP00502A](https://doi.org/10.1039/D0CP00502A).
