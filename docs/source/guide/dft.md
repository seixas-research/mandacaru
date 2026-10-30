(dft)=
# Kohn–Sham density functional theory

The Rust molecular solver computes spin densities, the Hartree convolution, the
exchange–correlation energy and potentials, and the generalized Kohn–Sham matrix.
Python supplies geometry values, the ASE interface and unit conversion. Select it
with `method="dft"` and a functional `xc`; scripts are in `examples/dft/`.

## Molecular calculation

```python
from ase import Atoms
from mandacaru import Mandacaru

atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, .74]], cell=[5, 5, 5])
atoms.center()
atoms.calc = Mandacaru(method="dft", xc="r2scan", basis="HAO", grid_spacing=.35,
                       resources={"threads": 2, "memory_limit_bytes": 512 * 1024**2})
energy = atoms.get_potential_energy()   # eV, including nuclear repulsion
atoms.calc.close()
```

The functional is chosen by `xc` and defaults to `"lda"`. The calculation is
restricted unless the electron counts differ or you pass `spin=True`
(`unrestricted=True`); the spin state follows the initial magnetic moments, as in
[Open-shell systems](open_shell.md). Converge both the basis and the grid before
interpreting energy differences: `HAO` is a generated hydrogenic basis, not a
claim of basis-set completeness. `scf_options` passes `max_iterations` (300),
`tolerance` (`1e-9`), `diis` (`True`) and `guesses` (3) to the SCF.

With a PAW-LCAO basis the molecular Kohn–Sham solver is semilocal (`lda`, `pbe`,
`r2scan`) and reads the dataset set of its functional
([PAW-LCAO bases](pseudopotentials.md)). Gradient-dependent molecular
functionals currently require an orthorhombic isolated grid. Nonlocal field
convolution itself supports skew periodic cells, but that does not enable the
complete advanced-functional SCF path.

(dft-functionals)=
## Canonical functionals

Only the exact names below are accepted; alternative spellings and aliases are
rejected.

| Name | Native molecular contribution |
| --- | --- |
| `lda` | Slater exchange and continuous Perdew–Zunger correlation |
| `pbe` | PBE exchange and correlation, with the Perdew–Wang uniform-gas reference |
| `r2scan` | Spin-resolved r²SCAN exchange and correlation with kinetic-density response |
| `hse06` | Exchange-hole semilocal contribution and 25% screened exact exchange, with screening 0.11 Bohr⁻¹ |
| `pbe+rvv10` | PBE plus native rVV10 correlation, with b = 6.3 |
| `r2scan+rvv10` | r²SCAN plus native rVV10 correlation, with b = 11.95 |

The point kernels compute an energy per volume and its seven partials with
respect to the two spin densities, three density-gradient contractions and two
kinetic-energy densities. The molecular solver forms these fields from its
current density matrices on every SCF evaluation.

### Meta-GGA response

For each spin channel,

$$
\tau_\sigma(\mathbf r)=\frac12\sum_{ab}D^\sigma_{ab}
\nabla\phi_a(\mathbf r)\cdot\nabla\phi_b(\mathbf r),
$$

and the matrix includes the corresponding kinetic-density potential

$$
V^\tau_{ab}=\frac12\int v_\tau(\mathbf r)
\nabla\phi_a(\mathbf r)\cdot\nabla\phi_b(\mathbf r)\,d\mathbf r.
$$

A fixed or guessed tau would define a different calculation. Native r²SCAN uses
differentiated orbital samples consistently in its energy, matrix and stationary
gradient. The discretized iso-orbital indicator has a smooth floor below zero to
keep noisy density tails bounded, and points below the grid density threshold are
excluded from the meta-GGA contribution.

### Screened hybrid

HSE06 needs two matching pieces: the semilocal short-range exchange subtraction
and the screened exact-exchange operator. The Rust implementation integrates the
exchange hole with erfc screening and differentiates that quadrature. The
molecular solver builds screened Cholesky factors, contracts them into the
exchange matrix, and includes the same operator in the stationary gradient.
Evaluating a semilocal point alone is not a hybrid SCF calculation. Screened
exchange with a PAW-LCAO basis is not yet available
([open items](#not-yet-available)).

### Nonlocal correlation

rVV10 uses logarithmic interpolation in a local inverse-length variable, cardinal
cubic splines and FFT convolution; the reciprocal-space kernel is analytic and
tabulated on a uniform wave-number mesh. The energy and the density and gradient
partials use the same kernel, so the potential differentiates the discrete
energy. An isolated calculation zero-pads each dimension before convolution;
a periodic one convolves on the cell, with reciprocal aliases averaged over the
grid's lattice operations.

## Forces

All-electron molecular gradients are analytic and stationary: nuclear and
external-potential terms, basis response, overlap response and the functional's
generalized Kohn–Sham action, with no displaced SCF. They are checked against
finite differences of fixed-grid total energies for the four advanced
functionals (`r2scan`, `hse06` and the two rVV10 pairings), including the hybrid
exchange and nonlocal terms. Their use, the PAW-LCAO force
scheme and the grid requirements are in [Forces](forces_relaxation.md), which
owns them; for geometry work use a PAW-LCAO basis.

## Periodic calculations

A periodic structure (`pbc=True` on the atoms) runs the periodic engine: `lda`,
`pbe`, `r2scan`, `pbe+rvv10` and `r2scan+rvv10`, collinear spin, `kpts` meshes,
`smearing` for metals, and spectral observables (`band_structure()`, `dos()`,
`pdos()`, `get_fermi_level()` on the calculator). `hse06` raises
`NotImplementedError` for a crystal. PAW geometry and dataset assembly still go
through a Python adapter. Forces and stress of crystals, screened exchange and
spinor meta-GGA are not yet available ([open items](#not-yet-available)); see
[Native calculations](native_engine.md) before choosing a crystal workflow.

## Dispersion correction

A dispersion correction (D4) is not yet available. Native building blocks exist,
but a complete parameterized model needs reference polarizabilities, fitted
element parameters, coordination interpolation, charge equilibration with its
geometry response, pair damping, three-body terms and periodic sums, and a pair
potential with guessed coefficients is not that model. The calculator does not
silently replace it with rVV10 or any other correction; the rVV10 choices above
are independent nonlocal correlation functionals ([open
items](#not-yet-available)).

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
