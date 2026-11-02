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

With a PAW-LCAO basis the molecular Kohn–Sham solver runs `lda`, `pbe`, `r2scan`
and the screened hybrid `hse06` ([Screened hybrid](#dft-screened-hybrid)) and reads
the dataset set of its functional ([PAW-LCAO bases](pseudopotentials.md)). Gradient-dependent molecular
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

(dft-screened-hybrid)=
### Screened hybrid

HSE06 needs two matching pieces: the semilocal short-range exchange subtraction
and the screened exact-exchange operator. The Rust implementation integrates the
exchange hole with erfc screening and differentiates that quadrature. The
molecular solver builds screened Cholesky factors, contracts them into the
exchange matrix, and includes the same operator in the stationary gradient.
Evaluating a semilocal point alone is not a hybrid SCF calculation. With a
PAW-LCAO basis, the hybrid carries the PAW terms of the exchange: the screened
pair densities include the compensation charges, and each atom's sphere adds its
one-center screened exchange, the frozen short-range semilocal term that the exact
exchange replaces, and the constant of that term (once per atom). In an open-shell
molecule the spheres' semilocal short-range exchange follows the two spin densities
rather than their sum; closed-shell results are unchanged by that term. A periodic
crystal carries the same terms in the self-consistent iterations, the bands and the
Wannier frames, plus the same spin-scaling remainder when it is unrestricted; a
hybrid crystal without its spheres raises at construction (see
[Periodic calculations](#dft-periodic)).

The hybrid energies are not yet validated against the all-electron reference, and
the open items are measured, not closed. For H₂O the PAW HSE06 minus PBE energy is
+13.5 mHa (9 Å box, DZP, grid spacing 0.3 Å), against +20.98 mHa for the
all-electron valence-only value. The deficit grows with the number of $p$ states of
the heavy atom, and it has parts that are known: the grid resolution of the hard
smooth $p$ pair densities, the one-center correction of the hole-model exchange
(which the spheres do not carry), and the first-order linearization of the frozen
term. A crystal HSE06 energy and gap are reported by the code but not pinned
against a literature value: the whole-mesh requirement below limits a hybrid crystal
to small $k$ meshes, and on them the two divergence treatments differ by more than
1 eV in the total energy of a two-atom silicon cell. A molecule in a periodic box at
the $\Gamma$ point reproduces the molecular HSE06 for H and H₂ once the box is large
enough, but not for water, which stays about 1 mHa apart; that difference is not yet
explained ([open items](#not-yet-available)).

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

(dft-periodic)=
## Periodic calculations

A periodic structure (`pbc=True` on the atoms) runs the periodic engine: `lda`,
`pbe`, `r2scan`, `pbe+rvv10`, `r2scan+rvv10`, and `hse06`, collinear spin, `kpts` meshes,
`smearing` for metals, and spectral observables (`band_structure()`, `dos()`,
`pdos()`, `get_fermi_level()` on the calculator). Screened hybrid calculations
(`xc="hse06"`) evaluate the non-singular reciprocal screened exchange operator across the
Brillouin zone in Rust (self-consistent crystal SCF, band structures, DOS/PDOS, and
two-component spinor calculations with spin-orbit coupling). The exact exchange sums over
the listed $k$-points, so a hybrid crystal needs the **whole** $k$ mesh: a mesh reduced by
space-group symmetry (the default `symmetry=True` on a crystal that has symmetry) or
halved by time reversal raises `NotImplementedError`. Give `symmetry=False` together
with a Gamma-centered mesh of one or two points per direction (all of its points are
time-reversal invariant, so nothing is halved). On any other mesh give `symmetry=False`
with `unrestricted=True` and **nonzero** initial magnetic moments: only a spin-polarized run
keeps the whole mesh, and zero moments do not (a spin-orbit crystal has no such option).
One Gamma point needs neither. PAW geometry
and dataset assembly still go through a Python adapter. Forces and stress of crystals are computed via `atoms.get_forces()` and
`atoms.get_stress()` as central finite differences of the energy through ASE (one SCF per
displacement, so they cost many SCFs and are not analytic); for a hybrid the energy
includes the one-center screened terms, and there is no analytic hybrid derivative.

A hybrid crystal is built from its PAW spheres. The setup records the atom centers,
the projector basis and, per atom, the dataset file with its SHA-256 digest; Rust
reads each file again when it builds the crystal. A saved crystal (`load_crystal=`)
therefore needs its dataset files at load time, and a file whose digest differs from
the recorded one raises. A hybrid setup without that record, such as one written
before the one-center terms existed, raises as well, and so does a magnetic
(non-time-reversal) spinor hybrid; a spinor hybrid with time-reversal symmetry runs.

The exact exchange enters the self-consistent mixing with the density. The
`tolerance` of `periodic_scf_options` (default `1e-7` Ha) therefore bounds the change
of the hybrid exchange operator between iterations as well as the energy change: the
SCF bands are then within `tolerance` of the bands of the converged density. The
snapshot of a hybrid run carries `exchange_residuals`, one value per iteration, in
Hartree. The first iteration diagonalizes without the exchange operator.

`resources={"threads": N}` does not yet bound the Bloch projection kernels that the
Python PAW setup calls from a shared library; they take the number of threads from
the environment or the core count until that setup moves into Rust.

For HSE06 crystal calculations, the exact exchange divergence treatment can be
selected via `exxdiv_treatment="gygi-baldereschi"` (`gygi_baldereschi=True` is the one
shorthand for it). Any other spelling raises `ValueError`, and so does a divergence
treatment given with a functional that is not a hybrid or for a molecule. The Gygi–Baldereschi scheme integrates the
screened kernel over the equivalent $k$-point Voronoi sphere of radius
$q_c = (6\pi^2 w_q / \Omega_{\text{cell}})^{1/3}$. It averages a smooth kernel over the
$k$-cell, so it shares the isolated-system limit of the default but approaches it from
the other side. For H₂ at the $\Gamma$ point of a periodic box the two treatments
differ from the molecular HSE06 by 0.9 mHa (`none`, 9 Å box) and 2.5 mHa
(Gygi–Baldereschi); at 13 Å the default is within 0.02 mHa, while Gygi–Baldereschi is
still not converged at box sizes of 9 to 13 Å. Which treatment is better on a real crystal mesh has not
been decided. The default is `exxdiv_treatment="none"`, using the point limit $\pi/\omega^2$.
The on-site Hubbard correction of a `d` or `f` shell (`hubbard_u`, DFT+U) is an
option of this periodic path and has [its own guide](dft_u.md).

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
- Gygi and Baldereschi, *Physical Review B* **34**, 4405 (1986),
  [doi:10.1103/PhysRevB.34.4405](https://doi.org/10.1103/PhysRevB.34.4405).
- Ernzerhof and Perdew, *Journal of Chemical Physics* **109**, 3313 (1998),
  [doi:10.1063/1.476928](https://doi.org/10.1063/1.476928).
- Vydrov and Van Voorhis, *Journal of Chemical Physics* **133**, 244103 (2010),
  [arXiv:1009.1421](https://arxiv.org/abs/1009.1421).
- Sabatini, Gorni and de Gironcoli, *Physical Review B* **87**, 041108 (2013),
  [doi:10.1103/PhysRevB.87.041108](https://doi.org/10.1103/PhysRevB.87.041108).
- Román-Pérez and Soler, *Physical Review Letters* **103**, 096102 (2009),
  [doi:10.1103/PhysRevLett.103.096102](https://doi.org/10.1103/PhysRevLett.103.096102).
