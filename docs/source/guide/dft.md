# Kohn-Sham density functional theory

`method="dft"` solves the Kohn-Sham equations (Kohn and Sham 1965)
in the same basis, on the same real-space grid and with the same external
potential as every other method. A DFT energy and an ADAPT-VQE energy of the
same system therefore differ only in how the electrons are treated: a mean-field
functional on one side, the many-body Hamiltonian on the other. Like `"rhf"`
and `"uhf"`, the method is purely classical: it builds no ansatz, pool, circuit
or qubit Hamiltonian.

```python
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("H2O")
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="dft",
                       xc="pbe",                  # "lda" (default) | "pbe" | "r2scan" | "hse06"
                       vdw="d4",                  # None (default) | "rvv10" | "d4"
                       basis={"name": "PAW-LCAO", "size": "DZP"},
                       grid_spacing=0.2)
energy_ev = atoms.get_potential_energy()
```

The same calculation from the command line:

```bash
mandacaru H2O --method dft --xc pbe --vdw d4 --cell 10
```

A crystal read from a file with a periodic cell (CIF, POSCAR, extended XYZ)
takes its k-point mesh and smearing as flags (see *Crystals* below):

```bash
mandacaru Al.cif --method dft --basis PAW-LCAO --basis-option size=DZP \
    --kpts 8 --gamma-centered --smearing methfessel-paxton 0.2
```

`--kpts` takes one mesh size or three, `--gamma-centered` shifts an even mesh
onto Gamma, and `--smearing` takes a width in eV (Fermi-Dirac), a method, or
a method and a width. Both are refused for a molecule (a g2 name boxed by
`--cell` has no periodic boundary conditions), and `--smearing` is refused by
every method but `dft`.

`atoms.get_potential_energy()` returns the Kohn-Sham total energy in eV: the
electronic energy, the ion-ion repulsion, the dispersion energy when one is
asked for, and, for a pseudopotential basis, the per-atom offset that keeps the
valence-only energy zero. `device=` is accepted and ignored, since the
calculation runs wherever Python runs. Options that need a circuit or a mapped
Hamiltonian (`optimizer`, `shots`, `taper`, `checkpoint`, `load_hamiltonian`,
`save_hamiltonian`, `verbose_operators`, `verbose_hamiltonian`) are refused, as
is `active_space`, which belongs on the quantum calculation that follows (see
the handoff below).

## The functionals

| `xc` | Functional | Reference |
| :--- | :--- | :--- |
| `"lda"` (default) | Local density approximation, the Ceperley-Alder correlation as parameterized by Perdew and Zunger | Perdew and Zunger 1981 |
| `"pbe"` | Generalized-gradient approximation | Perdew, Burke and Ernzerhof 1996 |
| `"r2scan"` | Meta-GGA, depending on the kinetic-energy density | Furness *et al.* 2020 |
| `"hse06"` | Screened hybrid: a quarter of the short-range exchange is exact (molecules and crystals; crystal forces and stress) | Heyd, Scuseria and Ernzerhof 2003; Krukau *et al.* 2006 |
| `"pbe+rvv10"` | PBE plus rVV10 nonlocal correlation, $b = 6.3$ (molecules and crystals, forces and stress, with or without spin-orbit coupling) | Vydrov and Van Voorhis 2010; Sabatini *et al.* 2013 |
| `"r2scan+rvv10"` | r2SCAN plus rVV10 nonlocal correlation, $b = 11.95$: dispersion from the density (molecules and crystals, forces and stress, with or without spin-orbit coupling) | Vydrov and Van Voorhis 2010; Sabatini *et al.* 2013; Ning *et al.* 2022 |

`vdw="rvv10"` is the same choice made on the side: `xc="pbe", vdw="rvv10"`
runs `"pbe+rvv10"` and `xc="r2scan", vdw="rvv10"` runs `"r2scan+rvv10"`. It
is refused for any other functional.

PBE and r2SCAN use the Perdew-Wang 1992 parameterization of the uniform-gas
correlation energy. The exchange-correlation energy and its potential are
evaluated on the grid from the density, with spectral gradients on a crystal's
grid and central-difference ones on a molecule's (below); r2SCAN also
builds the kinetic-energy density
$\tau = \tfrac12\sum_{pq}D_{pq}\nabla\phi_p\cdot\nabla\phi_q^*$ from the
gradients of the orbitals and feeds its derivative back into the Kohn-Sham
operator.

- **LDA** is the default and the cheapest. With PAW-LCAO it reads the LDA
  datasets (`lda-sr/`), so the molecule and its datasets use the same
  functional.
- **PBE** adds the density gradient. With PAW-LCAO it reads the PBE datasets
  (`pbe-sr/`, below), again the same functional as the molecule.
- **r2SCAN** is the most expensive of the three because it needs the orbital
  gradients on the grid. It shifts the orbital energies noticeably from PBE:
  for H$_2$O the HOMO-LUMO gap is about 1.4 eV larger with r2SCAN than with
  PBE, the same shift an independent all-electron Gaussian-basis calculation
  gives.

- **HSE06** replaces a quarter of the short-range exchange by exact
  exchange (below). It builds a second two-electron tensor, so it costs
  about twice the integrals of the others.

### The hybrid HSE06

`xc="hse06"` splits the Coulomb operator as
$1/r = \operatorname{erfc}(\omega r)/r + \operatorname{erf}(\omega r)/r$
with $\omega = 0.11\ a_0^{-1}$ and uses

$$E_{xc} = E_x^{\omega\rm PBE}(0) - \tfrac14 E_x^{\omega\rm PBE,SR}(\omega)
  + \tfrac14 E_x^{\rm HF,SR}(\omega) + E_c^{\rm PBE},$$

the semilocal exchange being the screened exchange hole of Heyd, Scuseria
and Ernzerhof. The exact exchange is contracted from the short-range
two-electron tensor -- the full tensor minus the same pair densities under
$\operatorname{erf}(\omega r)/r$ -- and enters the Kohn-Sham operator as the
nonlocal $-\tfrac18 K^{\rm SR}[D]$: a generalized Kohn-Sham problem, like
r2SCAN's. The result carries the exact part of the exchange-correlation
energy as `scf.exact_exchange_energy`, and the run log reports it.

- **Crystals too** -- see *HSE06 in a crystal* below.
- **Forces** (`get_forces()`) are analytic, restricted and unrestricted:
  the short-range tensor's nuclear derivative is the full tensor's minus
  the long-range one's, through the same moving basis functions (and, on
  PAW-LCAO, the moving compensation charges and projections). A crystal's
  forces and stress are analytic too (below).
- **The hole model's reduced gradient bends continuously.** Above $s = 1$
  the hole model bends $s$ towards a maximum; the reference implementation
  starts that bend $5.2\times10^{-4}$ below $s = 1$ (and caps it at
  $s = 15$), which leaves steps in the energy that a finite-difference force
  sees (2e-3 eV/Angstrom on H$_2$O, 1e-2 on OH). Here the bend starts at
  $s = 1$ and needs no cap: identical below $s = 1$, within $2\times10^{-4}$
  of the reference above it, and the forces are the energy's derivative.
- **PAW-LCAO** takes the exact exchange over the augmented pair densities,
  as the Hartree term does, and each augmentation sphere adds its
  one-center correction. The datasets freeze their one-center semilocal
  energies at the reference atom; the share of them the exact exchange
  replaces is frozen the same way and removed.
- **The Coulomb kernels are spectral.** A molecule's Hartree and exact
  exchange kernels apply their singular part in reciprocal space on the
  zero-padded grid, so neither carries a grid-sampling error of the $1/r$
  singularity. A molecule in a 12 Angstrom periodic box and the same
  molecule isolated then agree on the HSE06 - PBE energy to 0.1 mHa.
- **Shifts against references** (DZP, grid_spacing = 0.2 Angstrom, 3 Angstrom of
  vacuum; HSE06 - PBE): H$_2$O HOMO -1.73 eV (PAW-LCAO) against -1.45 eV for an
  independent norm-conserving pseudopotential calculation and -1.42 eV
  all-electron (aug-cc-pVTZ; -1.51 eV in def2-TZVP); CH$_4$ -1.18 eV
  against -1.14 eV. The excess follows second-row lone-pair HOMOs
  (NH$_3$ too, CO$_2$ not), which points at the confined basis rather than
  at the dataset (open). Converge h as well: a molecule's HOMO shift still
  moves by ~0.03 eV per 0.05 Angstrom here, and its PBE HOMO by ~0.07 eV.
- **A partial core keeps its semilocal exchange.** The exact exchange
  replaces the short-range semilocal exchange of the valence density only;
  the core has no orbitals to take exact exchange from.
- The relativistic exchange factor of a relativistic dataset is not applied,
  as for r2SCAN.

A molecule's grid (orthorhombic, not periodic) takes those gradients and the
divergence of the potential with the symbol of an order-8 central difference
(nine points per axis) in place of $i\mathbf{k}$. A spectral derivative of a
density the grid does not resolve rings over the whole box, and the gradient
functionals weight that ringing most where the density is small: on LiH at
`grid_spacing = 0.3` Å it put a 3e-4 eV/Å ripple on the PBE force surface at the
1e-3 Å scale, so the force and the finite difference of the energy disagreed
by 5.5e-4 eV/Å. With the stencil the ripple is 1.6e-7 eV/Å and the
disagreement 1e-6 eV/Å. The energy, the potential, the forces and rVV10 all
differentiate the same operator, so the forces remain the derivative of the
reported energy. Every molecular GGA energy moved by a fraction of a
milli-Hartree against the spectral gradient (H$_2$O, DZP, `grid_spacing = 0.2` Å:
PBE+D4 -0.26 mHa; LDA has no gradient and is unchanged). Crystals keep the
spectral derivative described next.

r2SCAN's density gradient on a molecule is not the derivative of the sampled
density but the product rule on the orbitals,
$\nabla n = 2\,\mathrm{Re}\sum_{pq}D_{pq}\,\phi_p\nabla\phi_q$, built
from the same stencil gradients of the orbitals that give $\tau$. The density
gradient (so $p$) and $\tau$ then come from one set of orbital gradients; before,
they did not, and the iso-orbital indicator $\alpha$ of a one-orbital density was
not 0. (rVV10 keeps the grid's derivative of the density.) The roughness of the
H$_2$O energy against a rigid shift of oxygen (rms residual of a cubic fit, PAW-LCAO
SZ, `grid_spacing = 0.3` Å, seven points over $\pm0.0045$ Å) fell from 0.49 µHa with the
spectral density gradient to 0.016 µHa with the stencil of the density and to
0.0003 µHa with the product rule (PBE: 0.0001 µHa); on DZP at `grid_spacing = 0.25` Å
(17 points over $\pm0.04$ Å, quartic) it is 0.021 µHa against 0.16 before, PBE 0.007 and LDA
0.0045. Those scans let the grid follow the molecule, as a single-point energy does; on
the grid frozen at the first geometry, which forces and relaxations use, the
ripple is the egg-box's (next to the forces below) and the product rule does not
change it: water 0.15 µHa against PBE's 0.015, and 0.05 to 0.4 µHa for CO and
N$_2$ with every functional. What the product rule does remove is the
disagreement between force and energy (below). The r2SCAN total energy moved with it, by an amount that falls with the
grid (H$_2$O, DZP, new minus old): +91 meV at `grid_spacing = 0.30` Å, +18.5 meV
at 0.25, +5.0 meV at 0.20, +1.2 meV at 0.15 and +0.33 meV at 0.12. With the basis
held fixed (`filter_cutoff` 600 eV) both discretizations converge to the same
energy, -466.828 eV, within the 3 meV egg-box scatter.

The density gradients PBE and r2SCAN need on a crystal are spectral derivatives built
from the reciprocal vectors of the cell, so a skewed cell -- hexagonal,
monoclinic, triclinic -- is handled as exactly as an orthogonal one. The
FFT box's wave-vectors are averaged over the point operations of the grid's
lattice (each operation's image of a frequency is pulled back to an alias of
it), so the gradients commute with every space-group operation that maps
the grid onto itself: on a hexagonal or fcc grid a gradient-dependent
functional is as symmetric as LDA. The average is taken on the integer
frequencies, so a strain moves the derivatives linearly, as the box's do,
and the finite-strain stress is unaffected. Only frequencies near the faces
of the box change; on an orthorhombic box, which is every molecule's grid,
that is the Nyquist plane of an even axis, whose derivative along that axis
a real field never had, and nothing changes. The kinetic energy of the Bloch
sums, the Coulomb kernel and the compensation and ion charges on the grid
are averaged over the same aliases (the kinetic multiplier as the mean of
|G + k|^2, alias by alias), so every operator commutes with the operations
and with time reversal: the irreducible wedge reproduces the full k-mesh to
1e-13 Ha on hcp Mg and wurtzite AlN, and to 2e-10 Ha on diamond Si (the
short-range spheres' quadrature is not invariant under the cube's
three-fold axes).

### rVV10: dispersion from the density

`xc="r2scan+rvv10"` (or `vdw="rvv10"`) adds the rVV10 nonlocal correlation to r2SCAN, so the
van der Waals attraction comes from the density itself rather than from an
atom-pairwise correction:

$$E_c^{nl} = \int n(\mathbf r)\Big[\beta + \tfrac12\int n(\mathbf r')
\Phi(\mathbf r, \mathbf r')\,d\mathbf r'\Big]d\mathbf r,\qquad
\Phi = -\frac{3}{2}\frac{1}{(\kappa\kappa')^{3/2}}
\frac{1}{(qR^2+1)(q'R^2+1)(qR^2+q'R^2+2)},$$

with $q = \omega_0/\kappa$, $\omega_0^2 = C|\nabla n/n|^4 + 4\pi n/3$ and
$\kappa = \tfrac{3\pi}{2} b\,(n/9\pi)^{1/6}$ (Vydrov and Van Voorhis 2010;
the separable rVV10 kernel of Sabatini *et al.* 2013). The parameters are
r2SCAN's, $b = 11.95$ and $C = 0.0093$ (Ning *et al.* 2022, fitted to the
argon dimer), and PBE's, $b = 6.3$ and the same $C = 0.0093$ (Sabatini *et al.*
2013); $\beta = (3/b^2)^{3/4}/32$ makes the term vanish for the
uniform gas.

- **How it is evaluated.** The kernel is interpolated over $q$ on 16 points
  (Roman-Perez and Soler 2009) and applied in reciprocal space, where its
  transform is analytic. A molecule is convolved on the zero-padded grid,
  a crystal on its cell. The term enters the Kohn-Sham operator, the forces,
  the spin channels (through the total density) and the crystal stress like
  a gradient correction. It adds well under a second per SCF iteration for a
  small molecule.
- **Checked.** The energy against the exact double sum of the kernel to
  3e-7 Ha; its potentials against finite differences; molecular and crystal
  forces and crystal stress against finite differences. Two free argon atoms
  reach the asymptotic $C_6$ of the model (76.3 a.u. for this argon density;
  the reference is 64.3, an overestimate of the VV10 model itself).
- **Argon dimer** (TZP, unconfined basis, grid_spacing = 0.2 Angstrom; reference
  CCSD(T) $-12.3$ meV at 3.76 Angstrom): $-12.2$ meV at 3.76, $-11.1$ at
  4.0, $-5.9$ at 4.5, $-3.5$ at 5.0 and $-0.7$ meV at 6.0 Angstrom;
  r2SCAN alone gives $-6.3$ meV at the minimum.
- **Use an unconfined basis for weak binding.** The default confinement
  (`energy_shift` 0.1 eV) makes the isolated fragments' orbitals too compact,
  and each fragment then borrows its neighbor's functions: argon dimer
  binding with PBE at 3.76 Angstrom is $-18.2$ meV confined and $-4.2$ meV
  unconfined, at any basis size. Pass `basis={"name": "PAW-LCAO",
  "energy_shift": None}` (or a much smaller shift) for interaction energies
  of this size, with any functional, and check them with the counterpoise
  correction (`interaction_energy(..., counterpoise=True)`, see
  {doc}`interaction_energy`).
- D4 is refused with it: the dispersion would be counted twice.

## D4 dispersion

`vdw="d4"` adds the D4 correction (Caldeweyher *et al.* 2019), computed
for the functional that ran, with its gradient in the forces. On a crystal
it is the lattice sum -- the pair dispersion and the coordination numbers
that set the C6 coefficients both run over the periodic images -- in the
energy, the forces and the stress (its virial over the cell volume). Ghost
atoms carry none: a counterpoise fragment's D4 is that of its real atoms in
the same cell, molecule or crystal. It is taken from the external `dftd4`
package, an optional dependency:

```bash
pip install 'mandacaru[dispersion]'
```

D4 has parameters for PBE, r2SCAN and HSE06 and none for LDA, so `vdw="d4"`
with `xc="lda"` is refused, as it is with `xc="pbe+rvv10"` or
`xc="r2scan+rvv10"`, which carry their own van der Waals correlation. Without `dftd4` installed the calculation stops
with an error naming the extra.

## Pseudopotential bases and their functional

The all-electron bases (`"HAO"`, `"NAO"`, the Gaussian families) and the
pseudopotential families (`"PAW-LCAO"`, `"UPAW-LCAO"`) all work, with every size
from `SZ` to `QZP` and `energy_shift` as in the other methods.

The PAW-LCAO datasets linearize the one-center Hartree and exchange-correlation
energies around their reference atom, so the Kohn-Sham energy on them needs no
exchange-correlation evaluation inside the augmentation spheres: it is the
smooth energy, with the compensation charges in the Hartree term and the
dataset's smooth core density in the functional, the way the dataset was
unscreened. The library ships a scalar-relativistic **LDA** set and a
scalar-relativistic **PBE** set, and each functional reads its own set
unless `paw_dataset=` names another:

| `xc` | datasets read by default |
|---|---|
| `"lda"` | `lda-sr/` |
| `"pbe"` | `pbe-sr/` |
| `"hse06"` | `pbe-sr/` (the hybrid built on PBE) |
| `"r2scan"`, `"pbe+rvv10"`, `"r2scan+rvv10"` | `pbe-sr/` (no r2SCAN datasets are generated) |

```python
atoms.calc = Mandacaru(method="dft", xc="pbe",       # reads pbe-sr/
                       basis={"name": "PAW-LCAO", "size": "DZP"})
```

HSE06 and r2SCAN on the PBE datasets still use a functional in the molecule
that is not the one in the datasets' cores and one-center terms; PBE is the
nearest set, not an exact match. A folder of another set
(`paw_dataset="lda-sr"` with PBE, `paw_dataset="pbe-sr"` with LDA) mixes the two
further, and the calculation emits a `RuntimeWarning` naming the folder of
the functional's own set. The mixed numbers are still useful for trends, but
they are not the answer of a consistent calculation.

## The grid

Converge the grid before trusting a difference of functionals. A spacing of
`grid_spacing` between 0.15 and 0.2 Å is a sensible range for H$_2$O and similar light
molecules. Three points to keep in mind:

- An absolute total energy on a real-space grid drifts with `grid_spacing`: for H$_2$O it
  moves by about 0.3 Ha between `grid_spacing = 0.25` and `0.12` Å, for Hartree-Fock and
  DFT alike, while the difference between the two stays constant to about
  8 mHa. Compare energies at one fixed `grid_spacing`.
- The kinetic-energy operator is spectral by default (`kinetic="spectral"`):
  the Laplacian is evaluated in reciprocal space, the operator a crystal
  uses, for molecules and crystals alike. `kinetic="fd"` selects the 3-point
  finite-difference stencil instead, which under-estimates the kinetic energy
  of a hard smooth wave: for the He atom the missing energy at
  `grid_spacing = 0.20` Å is about 108 mHa with the stencil and about 9 mHa
  with the spectral operator. The choice is shared by every method, not only
  DFT, and the analytic forces differentiate either operator (see *Forces and
  relaxations*). The spectral operator converges from above and much faster
  with `grid_spacing`: for H$_2$O (PAW-LCAO DZP, LDA) the energy moves by
  1.5 eV from `grid_spacing = 0.30` to `0.15` Å against 14 eV with the
  stencil, and the oxygen force stays within 0.14 eV/Å while the stencil's
  changes sign. The other molecular numbers in this guide were measured with
  `kinetic="fd"`, unless they say otherwise. The semilocal functionals' molecular numbers
  dated before the stencil gradient (v26.10.16 and earlier) also took their
  density gradient spectrally; the stencil moves a GGA total by a fraction of a
  milli-Hartree, and r2SCAN's by more on a coarse grid (the product-rule gradient
above; +5.0 meV on H$_2$O at `grid_spacing = 0.2` Å); the H$_2$O table below was
re-measured.
- The noble-gas atoms Ne and Ar in a PAW-LCAO basis need a spacing of about
  0.06 Å or smaller, well below the range above.

## Analytic two-center integrals

`two_center="analytic"` (command line `--two-center analytic`) integrates the
matrices that involve only two centers analytically instead of sampling them
on the grid: the basis overlap $S$, the kinetic energy $T$ and the
projections $P_{p\mu} = \langle\tilde p_p|\phi_\mu\rangle$ on the PAW-LCAO
projectors. Each radial function (the one the basis really uses, Fourier
filter included) is Fourier-Bessel transformed once; the integral at a
separation $\mathbf R$ is a sum over multipoles $L$ of a radial table
$s_L(|\mathbf R|)$, kept as a cubic spline, times a real Gaunt coefficient
times $S_{LM}(\hat{\mathbf R})$, for the atom pairs within the sum of the
cutoffs only. The forces differentiate the same splines, so they are the
exact derivative of the energy computed and the Pulay terms of $S$, $T$ and
$P$ are analytic. The three-center terms (the local potential, the Hartree
and exchange-correlation potentials, the two-electron tensor) stay on the
grid, as does every method reading the one-body integrals: Hartree-Fock, VQE
and ADAPT-VQE use the same matrices.

The molecular DFT default is `two_center="analytic"` for scalar-relativistic
PAW-LCAO and UPAW-LCAO bases when `kinetic` is omitted. Use
`two_center="grid"` to recover the sampled operator. Crystals, spin-orbit
datasets, other basis families and an explicit `kinetic` keep the grid route.
The integration route can also be selected explicitly. The analytic route computes its own kinetic energy, so `kinetic=`
with it is an error, and the `kinetic operator` line of `[ELECTRONS]` reads
`analytic (two-center)`. It is refused for crystals, for the spin-orbit
`lda-fr` and `pbe-fr` sets, for any basis but PAW-LCAO and UPAW-LCAO, and for
`force_method="scf-response"`.

The matrices agree with the grid's as the grid is refined (1e-6 and better
from `grid_spacing` = 0.20 Å for a Fourier-filtered basis), and the energies
of the two routes converge to the same limit. It removes the two-center part
of the egg-box, which is only part of it: water's translation ripple is
unchanged, because the local, Hartree and exchange-correlation terms keep
theirs. The option is a gain in accuracy and a modest one in time, not a
cure for the egg-box; the measurements are in `HISTORY.md`.

## The Hartree term

For a molecule, the Hartree matrix $J[D]$ is by default the potential of
the density from one Poisson solve per SCF iteration (`hartree="poisson"`),
with the same spectral isolated Coulomb kernel the two-body tensor is built
with. On a PAW-LCAO basis the compensation charges, every multipole the
datasets carry, enter as their low-rank matrices, exactly as they enter the
tensor. A semilocal Kohn-Sham run therefore never forms the $M^4$ two-body
tensor; it is built only when something reads it: a hybrid's forces and
exact exchange (its short-range tensor), or the export to a quantum method.
The forces of a semilocal functional read the Hartree potential of the
converged density instead (*Forces and relaxations*).
The reference energy of the exported Hamiltonian needs only the occupied
orbitals' integrals, which come from their own pair solves.

`hartree="tensor"` contracts the two-body tensor instead. The two routes
give the same Fock matrix, energy, reference energy and forces to round-off
(about $10^{-13}$ Ha), so the option exists to cross-check, not to choose an
accuracy. The Poisson route is faster where the tensor dominates: on
ethylene in PAW-LCAO DZP (46 functions, grid_spacing = 0.2 Angstrom) the energy took
42 s instead of 162 s and 0.40 GB instead of 0.71 GB, and benzene in DZP
(108 functions), whose tensor alone needs about 6.5 GB, now runs in about
2 minutes and 1.1 GB. A crystal's Hartree
potential is always the periodic Poisson solve, and `hartree="tensor"` is
refused for one.

## What the result holds

After `atoms.get_potential_energy()`, `atoms.calc.result` is a
`MeanFieldResult`, the same type `"rhf"` returns. Its `scf` field holds the
converged Kohn-Sham determinant:

| `result.scf` | Meaning |
| :--- | :--- |
| `mo_energies` | Kohn-Sham eigenvalues, ascending, in Hartree |
| `n_occupied` | Doubly occupied spatial orbitals |
| `homo_lumo_gap` | $\varepsilon_{\mathrm{LUMO}}-\varepsilon_{\mathrm{HOMO}}$ in Hartree |
| `hartree_energy` | Hartree energy, in Hartree |
| `xc_energy` | Exchange-correlation energy, in Hartree |
| `core_correction_energy` | Per-atom offset restoring the valence-only zero (pseudopotential bases), in Hartree |
| `dispersion_energy` | The D4 energy, in Hartree (zero without `vdw="d4"`) |

The HOMO-LUMO gap is a difference of Kohn-Sham eigenvalues, an orbital-energy
gap, not an excitation energy or a quasiparticle gap.

`result.reference_energy` is **not** the Kohn-Sham energy. It is the energy of
the Kohn-Sham determinant inside the exported many-body Hamiltonian, which has
neither the dispersion term nor the core-correction offset.
`result.optimal_energy` is the Kohn-Sham total.

On H$_2$O in a PAW-LCAO DZP basis at a grid spacing of 0.2 Å with 3 Å of vacuum, the three
functionals of the first example give these totals and gaps with the
spectral kinetic default (the setup of
[the functionals example](../../../examples/dft/01_dft_h2o_functionals.py)):

| Functional | Total energy (Ha) | HOMO-LUMO gap (eV) |
| :--- | ---: | ---: |
| LDA | -17.133249 | 7.66 |
| PBE + D4 | -17.172320 | 7.65 |
| r2SCAN + D4 | -17.164274 | 9.05 |

The D4 term is $-1.97\times10^{-4}$ Ha for PBE and $-4.7\times10^{-5}$ Ha for
r2SCAN. The PBE and r2SCAN rows read the PBE datasets, so neither warns.

The Kohn-Sham eigenvalues also give the molecular density of states: the
levels, each broadened by a Gaussian, of one isolated molecule. This is a
picture of its orbital energies at the Gamma point, not a band structure; see
the "Bands and densities of states" section below and
[the density-of-states example](../../../examples/dft/02_dft_h2o_dos.py).

## Populations, cube files and dipoles

For a molecule, `method="dft"` (like `"rhf"` and `"uhf"`) gives the one-particle
analyses of the converged state: populations (`population="hirshfeld"`,
`"voronoi"` or `"bader"`), `atoms.calc.get_charges()`, `atomic_partition()`,
`write_cube`, `get_dipole_moment()` and `natural_orbitals`.

A crystal has the populations too. The same three partitions are made
periodic: Voronoi cells and Bader basins use the nearest periodic image of
each atom, the Hirshfeld promolecule is the periodic superposition of free
atoms, and the charge inside each PAW sphere is added to its atom exactly.
Bader needs one more ingredient in a crystal: a pseudopotential's valence
density need not peak at the nuclei (covalent silicon's maxima sit at the bond
centers), so the basins are found on the all-electron density: the valence
density with the PAW reconstruction inside each sphere, **plus the free
atoms' frozen cores**. Only the valence density is integrated (see
[Charges](charges.md)). Every
partition accounts for all the valence electrons; the moments are zero for a
spin-restricted run and the magnetization's share otherwise.

```python
si.calc = Mandacaru(method="dft", basis={"name": "PAW-LCAO", "size": "DZP"},
                    kpts={"size": (4, 4, 4), "gamma": True},
                    population="hirshfeld")
si.get_potential_energy()
print(si.calc.atomic_partition("bader").summary())
```

As for a molecule, the split is a convention: rocksalt LiH (SZ, 2x2x2 mesh) gives
Li +0.24 (Hirshfeld), +0.66 (Voronoi) and +0.89 (Bader). Two notes:

- **Bader converges quickly with the grid**: the basins are traced by a
  continuous ascent on an interpolation of the density (see
  [Charges](charges.md)), so rocksalt LiH's Li is +0.852 at 10, 12 and 16 nodes
  per lattice vector (4x4x4) and moves by ~1e-3 e when the
  atoms are shifted against the grid; NaCl's Na is +0.85 and MgO's Mg +1.64
  (SZ, LDA).
- **Use a real k-mesh.** A Gamma-only crystal's density is not converged:
  LiH's Bader charge is +0.71 at Gamma, +0.85 at 3x3x3 and 6x6x6.

Equivalent atoms come
out equal only on a grid that keeps the operations relating them: on a grid
whose nodes miss diamond's quarter translations, the two silicon atoms of the
discretized crystal are not equivalent, and a Voronoi split shows it.

### Cube files and natural orbitals of a crystal

A crystal writes its densities and orbitals through the same calls as a
molecule, on the cell grid the SCF ran on. The `.xsf` file is a `CRYSTAL`
with the lattice, its datagrid closed by the periodic copy of the first
planes; the cube file holds the cell's nodes once. `repeat=(n1, n2, n3)`
tiles the cell.

```python
si.calc.write_cube("density.xsf", repeat=(2, 2, 2))
si.calc.write_cube("ae.cube", quantity="reconstructed_density")
si.calc.write_cube("vbm.cube", quantity="bloch_orbital", index=3,
                   kpoint=(0, 0, 0))                # fractional; or an index
si.calc.write_cube("k.cube", quantity="bloch_orbital", index=0,
                   kpoint=(0.1, 0.2, 0.3), component="modulus")
orbitals = si.calc.natural_orbitals()               # per SCF k-point
```

| `quantity` | What the file holds |
| --- | --- |
| `density` | the smooth valence density; the augmentation charge it lacks is reported, and `integral + augmentation_charge` is the valence count |
| `alpha_density`, `beta_density`, `spin_density` | a spin-polarized crystal's channels and the magnetization, which integrates (with its augmentation) to the moment |
| `reconstructed_density` | the all-electron valence density: the smooth density plus the spherical part of each sphere's one-center correction, the density the Bader partition climbs (no core). It is written as its average over each voxel, so the file integrates to the valence count: silicon 8.00004, triplet O2 12.0005 |
| `bloch_orbital` | Kohn-Sham band `index` at `kpoint` (any k-point; `spin=0` or `1` for a spin-polarized crystal), its smooth part |
| `natural_orbital` | natural orbital `index` (descending occupation) of the spin-summed density matrix at an SCF k-point |

These are one-particle quantities. A Bloch orbital is complex: at a
time-reversal invariant k-point (Gamma, or 2k a reciprocal-lattice vector)
each degenerate set is rotated to be real up to a phase, and the default
`component="auto"` writes it real (silicon at Gamma: 1e-15, at X: 2e-10);
elsewhere ask for `component="real"`, `"imag"` or `"modulus"`. Repeated
cells carry the Bloch phase, psi(r + R) = exp(ik.R) psi(r).

`natural_orbitals()` returns the eigenvectors of the spin-summed density
matrix at each SCF k-point, in the overlap of the Bloch sums
(`kpoints`, `weights`, `occupations`, `coefficients`, `n_electrons`): the
Kohn-Sham orbitals and their smeared occupations for a spin-restricted
crystal, the natural orbitals proper for a spin-polarized one (triplet O2
in a box: two orbitals at exactly 1). `natural_orbitals(method="rpa")` is
the direct-RPA density matrix's ({doc}`target_space`).

A crystal has no dipole moment: the first moment of a periodic density
depends on where the cell is cut. `get_dipole_moment()` refuses and points
to `get_polarization()`, the Berry-phase polarization (below, "Electric
polarization: the Berry phase").

## Reading the run log

A run writes the shared `[SYSTEM]`, `[BASIS]` and `[ELECTRONS]` blocks and then
three blocks of its own, read back by `parse_output` as `"setup"`,
`"scf_iterations"` and `"summary"` (see [Reading a run](run_output.md)):

- `[SCF SETUP]` names the functional (`xc_functional`), the dispersion
  treatment (`vdw`, `NONE` when there is none), how the Hartree term
  was built (`hartree`, see *The Hartree term* above) and the
  convergence
  criteria: at most 200 iterations, converged when both the energy change and
  the largest density-matrix change are below $10^{-8}$ Hartree, with DIIS and
  a level shift that is on only while the energy rises by more than
  $10^{-6}$ Hartree.
- `[SCF ITERATIONS]` has one row per iteration: the time, the electronic
  energy of a molecule (`electronic_energy_eV`, or `_Ha` in atomic units)
  or the free energy of a crystal (`free_energy_eV`), its change, the residual the convergence test
  measures and, for a spin-polarized crystal, the moment per cell.
- `[SCF SUMMARY]` gives the total and, for a molecule, its additive
  contributions: `electronic_energy`, `ion_ion_energy`, `one_center_energy`
  (the frozen PAW spheres), `core_correction_energy` and `dispersion_energy`,
  each suffixed with the energy unit. Zero contributions are included.
  Any other density-independent term, such as the ions' coupling to an
  electric field, appears as `other_constant_energy`. Their sum is
  `optimal_energy`. The Hartree and exchange-correlation energies are
  components of `electronic_energy`; a hybrid's `exact_exchange_energy`
  is already part of `xc_energy`. The reference energy of the exported
  Hamiltonian is reported separately.

The run also writes a `references.bib`: the Hohenberg-Kohn and Kohn-Sham
papers, the functional that ran and, with `vdw="d4"`, the D4 paper.

## From Kohn-Sham to ADAPT-VQE

The converged orbitals are exported as the Hartree-Fock ones are: the result
carries the many-body Hamiltonian, with the bare Coulomb interaction, written
in the Kohn-Sham orbitals, and its reference state is the Kohn-Sham
determinant. With a PAW-LCAO basis that Hamiltonian is built from the
Kohn-Sham run's integrals, so its augmentation spheres are the *frozen* ones
(the dataset's functional, linearized), not the exact one-center Coulomb and
exchange a wave-function method builds for itself
([PAW-LCAO](pseudopotentials.md)); a correlated energy on Kohn-Sham orbitals
and one on Hartree-Fock orbitals therefore differ by more than the orbitals.

```python
dft = atoms.calc.result
post_dft = Mandacaru(method="adapt-vqe", **dft.as_quantum_problem())
correlated = post_dft.run()
```

The ADAPT-VQE energy lowers the energy of this reference, so compare it with
`dft.reference_energy`, not with the Kohn-Sham total: the reference is the
Hartree-Fock functional of the Kohn-Sham orbitals, which is not the Kohn-Sham
energy, and it carries no dispersion energy. If the dispersion correction is
wanted in the final number, add `dft.scf.dispersion_energy` back by hand.
Reduce the register on the `adapt-vqe` call, not on the `dft` one.

## Forces and relaxations

`atoms.get_forces()` returns analytic Kohn-Sham forces (eV/Angstrom), so a DFT
surface can be relaxed with any ASE optimizer:

```python
from ase.optimize import BFGS

atoms.calc = Mandacaru(method="dft", xc="pbe", vdw="d4", grid_spacing=0.2,
                       basis={"name": "PAW-LCAO", "size": "DZP"})
BFGS(atoms).run(fmax=0.05)
```

The Kohn-Sham energy is stationary in its density matrix, so the gradient
needs no orbital response: it is the Hellmann-Feynman term (the operators of
each atom moving) plus the Pulay term (its basis functions moving), with the
exchange-correlation potential acting on the moving basis functions, the
partial core density moving with its atom, and the D4 gradient when the
dispersion correction is on. `atoms.calc.force_result` keeps the breakdown, as
for the other methods. With forces requested the grid is frozen along the
trajectory, so a finite difference of the reported energy on it reproduces the
forces: to 1e-6 eV/Angstrom for LiH with LDA and with PBE+D4 (PAW-LCAO DZP,
grid_spacing = 0.25 Angstrom).

The Hartree term of a semilocal functional (LDA, PBE, r2SCAN) enters the
gradient the way the exchange-correlation potential does: linearized about the
converged density. Its potential, the one Poisson solve the SCF already
performs, acts on the moving basis functions, the PAW-LCAO compensation
charges' matrix joins the fixed one-body matrix, and what is left of the
two-electron tensor's derivative is contracted with the fixed density matrix
to a handful of scalars. The $M^4$ tensor, its pair-density derivatives and
the $M^2$ Poisson solves behind them are never formed, so a force costs
$O(M^2 G)$ and $O(M G)$ memory ($M$ basis functions, $G$ grid points) instead
of $O(M^4 G)$ and $O(M^4)$. A hybrid keeps the tensor route, since its
short-range exchange needs the tensor anyway. Forces asked for right after
the energy of the same geometry read the converged state instead of repeating
the SCF; a moved atom or a changed cell converges its own.

The forces differentiate the kinetic operator the energy used, the spectral
default or the stencil (`kinetic="fd"`): for the spectral one the Pulay term
contracts the moving basis functions against the reciprocal-space Laplacian
of the fixed ones. With either operator the same checks hold to the same precision (LiH, PAW-LCAO DZP: 1e-6
eV/Angstrom with LDA and PBE+D4; HSE06, UPAW-LCAO, the unrestricted OH
radical and the correlated `force_method="rdm"` states alike), and a
molecule in a box has the forces of the same box solved as a crystal at
Gamma, to the grid's egg-box (water, SZ, grid_spacing = 0.3 Angstrom: 0.05
eV/Angstrom on forces of 5; the stencil's differ by 6.5). The egg-box itself
is about the same with either operator: a rigid translation at grid_spacing = 0.2
Angstrom ripples water's energy by 3.5 meV and CO's by 1.0 meV with both,
while LiH's net force is about 1.5 times larger with the spectral operator
(0.021 against 0.014 eV/Angstrom at grid_spacing = 0.25).

r2SCAN forces are analytic too, and on a molecule they are the derivative of
the energy on the frozen grid, as PBE's are: against a Richardson central
difference of the energy the force agrees to 4e-5 eV/Angstrom for H$_2$ (NAO DZ,
grid_spacing = 0.3 Angstrom), 4e-6 eV/Angstrom for water (DZP at grid_spacing = 0.25
Angstrom, SZ at 0.3; the oxygen's z component was 5.3e-4 there before the density
gradient came from the orbitals), for CO and N$_2$ (SZ, 0.3), and 4e-7 eV/Angstrom for LiH (the H
component).

What is not removed is the egg-box, the ripple of the energy when the molecule
is translated rigidly across one grid period, and it is larger for r2SCAN than
for the gradient functionals. Water
(PAW-LCAO DZP), peak to peak in micro-Hartree, and the largest net force it puts
on a rigidly translated molecule, $\pi\,(\text{peak to peak})/h$:

| `grid_spacing` (Angstrom) | r2SCAN | PBE | LDA | r2SCAN net force | PBE net force |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.30 | 6730 | 1350 | 516 | 2 eV/Angstrom | 0.4 eV/Angstrom |
| 0.25 | 3620 | 46 | 399 | 1.2 eV/Angstrom | 0.016 eV/Angstrom |
| 0.20 | 578 | 26 | 87 | 0.25 eV/Angstrom | 0.011 eV/Angstrom |
| 0.15 | 156 | 8.6 | 6.2 | 0.089 eV/Angstrom | 0.005 eV/Angstrom |
| 0.12 | 38 | 5.3 | - | 0.027 eV/Angstrom | 0.004 eV/Angstrom |
| 0.10 | 31 | 5.6 | - | 0.027 eV/Angstrom | 0.005 eV/Angstrom |

A relaxation with r2SCAN therefore needs a finer grid than one with PBE, and
neither functional's forces are better than a few 1e-3 eV/Angstrom (PBE) or a few
1e-2 eV/Angstrom (r2SCAN) from the converged ones on any grid tried: take
`grid_spacing` of 0.12 Angstrom or less for water with r2SCAN, and do not set
`fmax` below the net-force noise of the grid you chose. This is a property of
the grid integration, not of the functional's implementation.
Second-row elements (HCl, SiH$_4$) show a 1-3 micro-Hartree ripple on a bond
stretch that is the same for r2SCAN, PBE and LDA.

## Crystals: Bloch states, k-points and smearing

A periodic geometry -- any `atoms.pbc` set -- is solved as a crystal: Bloch
states on a Monkhorst-Pack mesh, with fractional occupations from a smearing
function, so metals and insulators are handled alike.

```python
from ase.build import bulk
from mandacaru import Mandacaru

al = bulk("Al", "fcc", a=4.05)
al.calc = Mandacaru(method="dft", xc="lda", grid_spacing=0.25,
                    kpts={"size": (6, 6, 6), "gamma": True},
                    smearing={"method": "methfessel-paxton", "width": 0.2},
                    basis={"name": "PAW-LCAO", "size": "DZP"})
energy = al.get_potential_energy()          # eV per cell
result = al.calc.result
result.fermi_level, result.free_energy, result.scf.band_gap
```

- **Basis.** PAW-LCAO or UPAW-LCAO, with the same radial functions as a
  molecule: every atom-centered function becomes a Bloch sum
  $\chi_{\mu\mathbf k} = \sum_{\mathbf R} e^{i\mathbf k\cdot\mathbf R}
  \phi_\mu(\mathbf r - \mathbf R)$, and every matrix element is an integral
  over one cell.
- **k-points.** `kpts` as for the other periodic methods (a size or
  `{"size": ..., "gamma": True}`). The mesh is reduced to its irreducible
  wedge by the crystal's space-group operations and time reversal, and the
  density summed over the wedge is symmetrized back to the full mesh's -- Al
  on 6x6x6 solves 16 k-points instead of 112, with the same energy. Only
  operations that map both the real-space grid and the k-mesh onto themselves
  are used: on 10 nodes per lattice vector silicon keeps 24 of its 48 (the
  quarter translations miss the nodes), and a 2x1x1 mesh keeps only the
  operations a 2x1x1 mesh has. `[SCF SETUP]` says how many were used.
- **Smearing.** `smearing={"method": ..., "width": eV}` with `"fermi-dirac"`
  (default, 0.1 eV), `"gaussian"` or `"methfessel-paxton"`. The reported
  energy is the $\sigma \to 0$ estimate -- $\tfrac12(E + F)$ for Fermi-Dirac
  and Gaussian, $F$ itself for Methfessel-Paxton -- and `result.free_energy`
  is $F = E - \sigma S$. On a coarse mesh Methfessel-Paxton is the more
  forgiving choice: its $F$ has no $\sigma^2$ term, so it barely moves with
  the width, while the Fermi-Dirac and Gaussian free energies bend away
  quadratically and only their $\tfrac12(E + F)$ estimate stays put.
  [The aluminum smearing example](../../../examples/dft/04_dft_al_smearing.py)
  sweeps the three methods over four widths and plots both quantities.
- **Electrostatics without finite-grid effects.** The smooth density, the
  compensation charges and the ions are combined into one neutral charge whose
  Coulomb energy is summed in reciprocal space; the compact charges are
  Fourier-filtered where they meet the grid and summed exactly among
  themselves. The energy is then nearly independent of the grid: H2 in an
  8 Angstrom cell changes by 0.08 mHa between grid_spacing = 0.25 and 0.18 Angstrom and
  reproduces the converged molecular limit, which the molecular solver itself
  reaches only by extrapolating from grid_spacing <= 0.12 Angstrom.
- **Eigenvalues, the Fermi level and the gap.** Eigenvalues and
  `result.fermi_level` are measured from the cell average of the electrostatic
  potential of the electrons and point ions (each atom's non-Coulomb local
  potential included), the usual plane-wave zero, so they do not move with the
  grid spacing. `result.scf.band_gap` is taken over the k-points of the mesh
  only: it is an upper bound whenever a band edge lies between them, as
  silicon's conduction-band minimum does on any small Gamma-centered mesh.
- **Checks you can rely on.** A k-point mesh reproduces the corresponding
  supercell at Gamma exactly (diamond Si, 2x2x2 mesh against the 16-atom
  supercell: equal to 1e-8 Ha per cell).
- **A molecule in a box is the same Hamiltonian.** Solved as a crystal at
  Gamma, a molecule in a box reproduces the molecular run on the same nodes,
  basis and kinetic operator. A crystal's kinetic energy is always spectral
  (it refuses `kinetic="fd"`), and so is the molecular default: Ne with a
  150 eV basis filter agrees to 1 uHa, and
  water, CO and HF in a 9 Angstrom box (SZP, grid_spacing = 0.3 Angstrom) agree on the
  level spacings to 0.4 mHa, on the energy to 0.7 mHa and on the Hirshfeld
  charges to 0.001 e. The finite-difference stencil (`kinetic="fd"`)
  differs from it by far more than that on these grids: 0.80 Ha for water,
  with a dipole 1.6 times larger. Water's density cube files agree node
  for node to 6e-4 of the peak.

### HSE06 in a crystal

`xc="hse06"` on a periodic geometry adds the short-range exact exchange of
the whole k-mesh to every k-point's Kohn-Sham matrix,

$$K^{\mathbf k}_{\mu\nu} = \sum_{\mathbf q} w_{\mathbf q}\sum_m f_{m\mathbf q}
\langle\chi_{\mu\mathbf k}\,\phi_{m\mathbf q}|\,v_{\rm SR}\,|
\phi_{m\mathbf q}\,\chi_{\nu\mathbf k}\rangle,$$

the sum running over every point of the mesh, not only the irreducible ones
(the others are symmetry images and time-reversed copies of the wedge's
states). Each pair density is one FFT with the short-range kernel at
$|\mathbf G + \mathbf k - \mathbf q|$; that kernel is finite where
$\mathbf G + \mathbf k - \mathbf q = 0$, so the screened hybrid needs no
treatment of the Coulomb singularity.

- **The exchange needs a denser mesh than the density.** The q-mesh is the
  k-mesh. For silicon (PAW-LCAO DZP, grid_spacing = 0.3 Angstrom) the HSE06 - PBE gap
  shift is +0.66 and +0.57 eV on 3x3x3 and 4x4x4 meshes (a 2x2x2 mesh
  overshoots by about a factor of two), against about +0.55-0.6 eV in the
  literature (HSE06 1.17 eV against PBE 0.59-0.62 eV): converge the shift,
  not just the energy.
- **Cost.** One FFT per occupied state, basis function and pair of k-points
  per SCF iteration: on the 4x4x4 silicon mesh the hybrid SCF took about
  nine times the PBE one (88 against 10 s).
- **Forces and stress.** The band structure, DOS and PDOS include the
  exchange, and so do the forces and the stress. The exchange is symmetric
  between its two k-points, so moving the basis functions is the Pulay term
  of $-\tfrac a2 K$ built over the basis and its derivatives with the
  occupied states held fixed; the pairs' compensation charges move with
  their projectors (the same doubling) and their shapes (once); the
  spheres' one-center operator moves like the nonlocal projector term; the
  stress carries the converged state to strained cells as for the other
  functionals. Both agree with finite differences of the free energy
  (LiH at Gamma, Si on a k-mesh, the irreducible k-points against the full
  mesh). A partial core moves under the semilocal potential without the
  short-range part the exact exchange replaces, which is the valence's.
- **PAW-LCAO, as for molecules.** The pair densities are augmented by
  their compensation charges -- their moments from the pairs' projections,
  the shapes' transforms at $\mathbf G + \mathbf k - \mathbf q$, and the
  compensation charges between themselves on the same dense reciprocal set
  as the Hartree term's -- and each augmentation sphere adds its one-center
  correction, from the projected density matrix of the whole mesh. The
  states at the mesh points outside the irreducible wedge are images of the
  wedge's; their projections come from the Bloch sums solved once at each
  image point.
- **Spin polarization.** A spin-polarized crystal takes each channel's
  exchange ($-aK_\sigma$) and the spheres' terms per channel.

## Spin polarization

A molecule with `n_alpha != n_beta` -- an odd electron count, or initial
magnetic moments on the atoms (`atoms.set_initial_magnetic_moments`) -- is
solved **spin-unrestricted**: one determinant per spin, with the
spin-polarized LDA, PBE, r2SCAN or HSE06. A crystal whose atoms carry nonzero
initial moments is solved **spin-polarized**: two potentials, one Fermi
level over both channels, the total density and the magnetization mixed
together (only the total is Kerker-damped).

```python
from ase.build import bulk, molecule

o2 = molecule("O2")                       # ASE gives it moments of 1 per atom
o2.calc = Mandacaru(method="dft", xc="pbe", grid_spacing=0.2,
                    basis={"name": "PAW-LCAO", "size": "DZP"})
o2.get_potential_energy()                 # the triplet: moment 2.000

fe = bulk("Fe", "bcc", a=2.87)
fe.set_initial_magnetic_moments([2.5])
fe.calc = Mandacaru(method="dft", xc="lda", grid_spacing=0.2,
                    basis={"name": "PAW-LCAO", "size": "DZP"},
                    kpts={"size": (8, 8, 8), "gamma": True})
fe.get_potential_energy()
fe.calc.get_total_magnetic_moment()       # N_up - N_down per cell
```

- **The functionals** follow the unpolarized ones exactly at zero
  polarization (to round-off) and are checked against an independent
  library at partial and full polarization. Exchange is the exact spin
  scaling of the unpolarized one; LDA correlation joins the paramagnetic and
  ferromagnetic fits with the von Barth-Hedin interpolation; PBE and r2SCAN
  carry the polarization through the Perdew-Wang uniform gas and their
  gradient and iso-orbital terms. The partial core is spin-unpolarized and
  split evenly between the channels.
- **What you get**: `get_number_of_spins()` is 2; `get_eigenvalues(kpt,
  spin)`; `get_total_magnetic_moment()` (from the occupations) and the
  per-atom moments of `population=` / `get_magnetic_moments()`; `dos`,
  `pdos` and the bands carry both channels (`band_structure` has ASE's
  `(n_spins, n_kpoints, n_bands)` energies). A molecule's many-body
  Hamiltonian is exported in the natural orbitals of the total density, as
  for UHF, and its forces are analytic (the magnetization term included).
  A spin-polarized crystal has analytic forces and the stress too, for
  every functional, HSE06 included: each channel's Pulay term is taken at
  its own potential (and, for the hybrid, its own exchange), the partial
  core moves under the mean of the two channels' potentials, and the
  stress carries both channels' states to the strained cells. Both agree
  with finite differences of the free energy (triplet O2 in a box, at Gamma
  and on a k-mesh), and a crystal that loses its moment gives the
  restricted forces.
- **Symmetry**: atoms are equivalent only with the same element *and*
  initial moment, so an antiferromagnetic arrangement keeps the operations
  that preserve it.
- **Checked**: O2's triplet is 0.78 eV (LDA) / 0.92 eV (PBE) below the
  spin-restricted state, which holds the two π* electrons as an ensemble
  (SZ, grid_spacing = 0.3 Å); O2 in a periodic box reproduces the molecular
  unrestricted run (moment 2.000, 1.000 per atom, pi* exchange splitting
  1.555 vs 1.558 eV).

## Spin-orbit coupling: two-component Kohn-Sham

A molecule or a crystal on a spin-orbit set -- `paw_dataset="lda-fr"` with
LDA, `paw_dataset="pbe-fr"` with PBE -- is solved in **two-component
spinors** (crystals: *Crystals* below).
No other option is needed: the dataset carries the spin-orbit term, and the
Kohn-Sham operator then contains it, exactly the one-body term
`method="ghf"` adds.

```python
from ase import Atoms

hi = Atoms("HI", positions=[[0, 0, 0], [0, 0, 1.609]], cell=[9.0] * 3)
hi.center()
hi.calc = Mandacaru(method="dft", xc="pbe", paw_dataset="pbe-fr",
                    grid_spacing=0.25, basis={"name": "PAW-LCAO", "size": "SZ"})
hi.get_potential_energy()
scf = hi.calc.result.scf
scf.spin_orbit_energy            # against the collinear run, Hartree
scf.kramers_pairing              # 0 for a closed shell: Kramers pairs
```

- **The operator** is $2M \times 2M$ over the Loewdin basis, alpha block
  first: the scalar core Hamiltonian on both spin blocks, the spin-orbit
  term (which couples them), the Hartree matrix of the total density on
  both diagonal blocks (the same Poisson solve) and the exchange-correlation
  potential and magnetic field, $v\,\mathbb 1 +
  \mathbf B\cdot\boldsymbol\sigma$.
- **Noncollinear functionals, local spin axis.** At each point the spin
  density matrix gives the density $n$ and the magnetization vector
  $\mathbf m$, and the collinear spin-polarized functional is evaluated on
  $n_\pm = (n \pm |\mathbf m|)/2$, the spin densities along the local axis
  $\hat{\mathbf m}$ (Kübler et al. 1988). LDA, PBE, r2SCAN, r2SCAN+rVV10
  and HSE06 are supported. PBE (and every gradient functional here)
  takes the gradients along the same axis: $\nabla n$ and the projection
  $\hat{\mathbf m}\cdot\nabla\mathbf m$, which is $\nabla|\mathbf m|$
  wherever $\mathbf m$ is smooth and nonzero but has no kink where a
  collinear moment changes sign. The energy is invariant under a global
  spin rotation, a collinear state gives back the collinear functional and
  potentials to round-off, and the field is the exact derivative of the
  energy (the gradient-corrected form follows Hobbs, Kresse and Hafner
  2000). Where $\mathbf m$ vanishes exactly on a grid point -- a collinear
  moment's nodal plane through the grid -- the axis is taken from the
  gradient of $\mathbf m$, so that point keeps its spin gradient and the
  collinear limit stays exact there too. **Its limits**: the gradient of the axis itself
  ($\nabla\hat{\mathbf m}$) does not enter the energy, so a twisting
  magnetization has no spin stiffness from the GGA, and the field's
  axis-rotation term carries $1/|\mathbf m|$, large where a noncollinear
  moment passes through zero (the known instability of local-axis GGAs; it
  is zero for collinear and for time-reversal symmetric states).
- **r2SCAN** takes the kinetic-energy density matrix of the spinors,
  $\tau_{\sigma\sigma'} = \tfrac12(\tau\,\delta_{\sigma\sigma'} +
  \boldsymbol\tau_m\cdot\boldsymbol\sigma_{\sigma\sigma'})$, along the same
  axis, $\tau_\pm = \tfrac12(\tau \pm \hat{\mathbf m}\cdot\boldsymbol\tau_m)$,
  and adds $\tfrac12\nabla\phi^*\cdot(v_\tau\mathbb 1 + \mathbf b_\tau\cdot
  \boldsymbol\sigma)\nabla\phi$ to the operator (the local-axis form; a
  noncollinear meta-GGA's spin-current term is not included).
  **r2SCAN+rVV10**'s nonlocal correlation sees the total density only.
  **HSE06** adds the short-range exact exchange of every spin block of the
  spinor density matrix (generalized Hartree-Fock's exchange, with the
  collinear hybrid's kernel and one-center terms), and its semilocal part
  gives up the short-range exchange channel by channel along the axis. A
  magnetization at round-off (a collinear moment's nodal plane, seen from a
  turned frame) counts as zero: its axis is taken from $\boldsymbol\tau_m$
  or the gradient of $\mathbf m$, as at an exact zero, and a channel the
  functional clips at zero has no derivative there, as in the collinear
  evaluator; with both, a collinear state turned to $x$ is the collinear
  energy to round-off for every functional.
- **Occupations** are aufbau over the $2M$ spinor levels, with a degenerate
  shell at the Fermi level filled equally, as in the collinear solvers: a
  Kramers pair holding one electron, or an open $p_{3/2}$ quartet.
- **The SCF starts from the collinear solution** of the same integrals
  without the spin-orbit term (restricted, or unrestricted for an open
  shell), so a spin-polarized open shell stays polarized; its energy is kept
  (`scalar_energy`), and `spin_orbit_energy` (also in `[SCF SUMMARY]`) is
  what the spinors lowered it by.
- **What you get**: the spinor eigenvalues (`get_eigenvalues()`, one set,
  `get_number_of_spins()` is 1), `kramers_pairing` (the largest splitting
  within the occupied pairs, measured, not imposed), the **spin** moment
  vector `spin_moment`, $2\langle\mathbf S\rangle$ in Bohr magnetons
  (`get_total_magnetic_moment()` is its $z$ component). It is not the total
  moment $\langle\mathbf L + 2\mathbf S\rangle$: the orbital moment is not
  computed, and the two can differ in sign (Tl $6p^1$: spin $+1/3$, total
  $-1/3$). Also `dos`/`pdos` with one electron per level, populations, cube
  files and dipoles from the spinor density, and the many-body Hamiltonian
  exported in the spinors (the GHF layout: `pool="spin-orbit"` for ADAPT-VQE,
  and not the `parity_reduced` mapping, which is refused).
- **Checked**: without the spin-orbit term the spinor problem gives the
  collinear energy to round-off -- OH with its UKS moment turned from $z$
  to $x$ (LDA 1e-14, PBE 2.4e-13 Ha; moment 1.000 along $x$) and closed-shell
  water and HI with zero moment vectors (below; 1e-14 Ha). HI's occupied
  spinors pair to 2e-15 Ha. Xe's $5p$ splitting is 1.1 % (LDA) and 1.0 %
  (PBE) below the radial Dirac atom of the same functional in SZ at
  grid_spacing 0.25 Å (0.9 % at 0.2 Å, extrapolating to about 0.5 %; the
  dataset's own operator is 0.4 % short, the rest is the basis and the
  grid). HI's spin-orbit lowering is 53.5 meV (LDA) and 50.5 meV (PBE),
  against 46.6 and 42.8 meV between GHF+SO and the RHF determinant of the
  same integrals; TlH's is 115 meV against 147 meV. The open-shell Tl atom
  ($6p^1$) puts its electron in a $6p_{1/2}$ spinor: moment 1/3 Bohr
  magneton (the spin of $p_{1/2}$), 633 meV below the collinear UKS.

### A noncollinear start

Without a spin-orbit term the collinear solution is a fixed point of the
spinor SCF, and a stable one: a noncollinear ground state is **not** found
from the collinear start, and a slightly canted collinear start falls back to
it. Equilateral H$_3$ (a frustrated triangle) has a coplanar state with its
site moments at 120°, 37.8 meV below the UKS doublet at $R = 2.5$ Å (LDA,
SZ), that the collinear start never reaches. Ask for it with ASE's
noncollinear moments, one vector per atom:

```python
import numpy as np
from ase import Atoms

R = 2.5
h3 = Atoms("H3", positions=R * np.array([[0, 0, 0], [1, 0, 0],
                                         [0.5, np.sqrt(3) / 2, 0]]),
           cell=[R + 6, R + 6, 6])
h3.center()
angles = 2 * np.pi * np.arange(3) / 3
h3.set_initial_magnetic_moments(np.c_[np.cos(angles), np.sin(angles),
                                      np.zeros(3)])
h3.calc = Mandacaru(method="dft", xc="lda", grid_spacing=0.3,
                    basis={"name": "PAW-LCAO", "size": "SZ"})
h3.get_potential_energy()
scf = h3.calc.result.scf
scf.noncollinear_energy          # against the collinear start, Hartree
scf.spin_moment                  # (0, 0, 0): the site moments cancel
```

- `(N, 3)` moments select the two-component solver on **any** dataset: on a
  scalar set it runs without the spin-orbit term (`spin_orbit_coupling:
  False` in `[SCF SETUP]`, `noncollinear_energy_<unit>` in `[SCF SUMMARY]`),
  on a spin-orbit set with it. `(N,)` moments behave as before.
- The collinear run first takes every moment along one axis, with the
  vectors' lengths (their rounded sum is the number of unpaired electrons,
  so it must match the electron count's parity); then each atom's share of
  that magnetization -- the Loewdin populations of its basis functions -- is
  turned to its own vector, and the spinor SCF starts there. An atom given a
  zero vector has its share dropped. The charge density is the collinear
  one.
- For H$_3$ it lands on the 120° state: site moments 0.912 Bohr magneton
  (Loewdin), 120.00° apart, the total zero to 2e-4 (the cubic grid breaks
  the threefold axis), 37.83 meV below UKS. Turning OH's moment to $x$
  gives back the UKS energy (LDA 1e-14, PBE 2.4e-13 Ha) with the moment along
  $x$.
- Refused: `(N, 3)` moments on any other method; forces, as for every
  two-component run.

### Crystals

A periodic geometry on a spin-orbit set, or with `(N, 3)` moments, is
solved in **two-component Bloch spinors**, with the options of any crystal
(k-points, smearing, grid, basis):

```python
from ase.build import bulk

gaas = bulk("GaAs", "zincblende", a=5.653)
gaas.calc = Mandacaru(method="dft", xc="lda", paw_dataset="lda-fr",
                      grid_spacing=0.15, kpts={"size": (4, 4, 4), "gamma": True},
                      basis={"name": "PAW-LCAO", "size": "DZP"})
gaas.get_potential_energy()
gamma = gaas.calc.get_eigenvalues(kpt=0)   # 2M spinor bands at Gamma, eV
scf = gaas.calc.result.scf
scf.spin_orbit_energy                      # against the collinear crystal, Hartree
scf.terms["spin_orbit"]                    # <h_SO>, a term of the energy
```

- **The operator** at each k-point is the collinear crystal's Kohn-Sham
  matrix of the total density on both spin blocks, plus the molecule's
  spin-orbit term summed over the lattice -- the projections of the Bloch
  sums on the home cell's spin-orbit projectors, as the scalar nonlocal
  term -- plus $\int\chi^*\,\mathbf B_{xc}\cdot\boldsymbol\sigma\,\chi$ of
  the noncollinear LDA or PBE on the cell grid. The metric is
  $S(\mathbf k)\otimes\mathbb 1$. Each spinor band holds one electron
  around one Fermi level; smearing, mixing and the energy conventions are
  the collinear crystal's.
- **The k-point mesh.** The spin is not a separate quantum number, but time
  reversal still maps the states at $\mathbf k$ onto those at
  $-\mathbf k$ with the same energy and density and the *opposite*
  magnetization, and a space-group operation maps them, spin rotated, onto
  those at $R\mathbf k$, the magnetization turning as an axial vector. So:
  a crystal given **no initial moments** is solved as a non-magnetic one,
  with time-reversal symmetry imposed -- its magnetization is zero and not
  computed, the functional is the unpolarized one -- and its density is a
  scalar, so the mesh is reduced by the space group and time reversal as for
  a collinear crystal. That is an assumption, as a collinear crystal
  without moments is restricted: a crystal that would magnetize must be
  given moments (a free Tl atom in a box without them lands on a half-filled
  Kramers pair, 122 meV above its magnetic $6p_{1/2}$ state). A
  **magnetic** one (`(N,)` moments with the spin-orbit term, or `(N, 3)`
  moments) is solved on the **whole mesh**, the magnetization a vector
  field of every k-point, and its spectra (DOS, Fermi surface) are
  diagonalized on the whole mesh too. `[SCF SETUP]` says which
  (`k_points_irreducible`, `spin`).
- **The SCF continues the collinear crystal**: the collinear problem of the
  same crystal and mesh without the spin-orbit term is solved first
  (restricted, or spin-polarized with the moments' lengths along $z$), and
  the spinor SCF starts from the density its last diagonalization was built
  from -- the magnetization along $z$, or, with `(N, 3)` moments, each
  atom's Hirshfeld share of it turned to its vector. Without the
  spin-orbit term and with every moment along one axis its first step *is*
  the collinear step, so it stops there, on the collinear energy.
- **Meta-GGA and hybrid.** r2SCAN and r2SCAN+rVV10 run on the wedge
  like LDA and PBE. HSE06's exchange takes every occupied spinor of the
  mesh, and the images of a wedge's spinors would need their spin
  rotations, so a hybrid spinor crystal is solved on the **whole mesh**;
  its exchange operator is mixed with the density as the collinear
  crystal's is, and seeded from the collinear start's (turned with the
  moments' mean direction for `(N, 3)` moments).
- **What you get**: the spinor bands (`get_eigenvalues`, one set of $2M$
  per k-point, `get_number_of_spins()` 1), `band_structure` and
  `fat_bands`, `dos` and `pdos` with either integration (one electron per
  spinor band: the DOS integrates to the number of spinor bands, $2M$),
  `fermi_surface` and `write_fermi_surface`, charges (`get_charges`,
  `atomic_partition`; local moments are the $z$ components of a magnetic
  crystal's spin moments, zero for a non-magnetic one) and the spin moment
  per cell `spin_moment`, $2\langle\mathbf S\rangle$ (augmentation
  included; `get_total_magnetic_moment()` is its $z$ component).
- **The Berry-phase polarization** (`get_polarization`) takes the strings
  of the $N$ occupied spinor bands with the spin-summed overlaps: one
  electron per band, so the electrons' quantum is $e\,\mathbf a_j/\Omega$,
  not $2e\,\mathbf a_j/\Omega$. The Born effective charges and the
  clamped-ion piezoelectric tensor follow
  (`born_effective_charges`, `piezoelectric_tensor`); the relaxed-ion
  tensor needs the forces and is refused.
- **Wannier functions** (`wannier`) of spinor bands are spinors, one
  electron each: every trial orbital of the guess is taken twice, spin up
  and spin down along `spin_axis` (default $z$), the overlaps sum both spin
  components, and spreads, centers, the disentanglement in energy windows
  and the interpolation are the collinear path's. The disentangled
  subspace is not symmetrized (the operations' spin rotations are not
  represented), the state selections by projectability (`"auto"`,
  `"scdm"`, `"rpa"`) are refused, and so is the downfolded problem. A
  spinor function is written component by component
  (`write(..., component="up")`) or as the density it carries
  (`write(..., part="density")`, $|w_\uparrow|^2 + |w_\downarrow|^2$).
- **Refused**: forces and stress (the crystal gradient has no spin-orbit or
  noncollinear term), projectabilities, natural orbitals and volumetric
  fields of the crystal (cube files), which read collinear Bloch states,
  and a finite electric field. A crystal on a spin-orbit set is never read
  as a scalar-relativistic one: every consumer without spinor states
  refuses it.
- **Checked**: with the spin-orbit term zeroed, a spin-orbit crystal is its
  collinear start to round-off (Si, LDA 0.0 and PBE 1.8e-15 Ha; fcc Al
  8.9e-16 Ha), and so are the O$_2$ triplet's moment vectors along $z$ in
  a periodic box (1.4e-14 Ha) and turned to $x$ (SU(2) invariance, moment
  $(2, 0, 0)$). Si on `lda-fr`: Kramers pairs at $\Gamma$ to 2e-16 Ha,
  $E(\mathbf k) = E(-\mathbf k)$ to 1e-15, and the whole mesh with zero
  moment vectors reproduces the wedge to the collinear starts' convergence
  (8e-9 Ha). The matrix is the derivative of the energy, spin-orbit term and
  noncollinear field included (finite differences, 4e-10 relative, LDA and
  PBE). In a large cell at $\Gamma$ a crystal is the molecule: HI's
  spin-orbit lowering 52.94 against 53.04 meV at grid_spacing 0.18 Å (52.15
  against 53.46 at 0.25: the molecule samples the iodine projectors on its
  grid, the crystal on spheres, and the molecule's converges toward the
  crystal's), the Tl atom's $6p_{1/2}$ lowering 632.6 against 633.4 meV,
  and equilateral H$_3$'s 120° state to 0.13 meV.
  The split-off gap $\Delta_{so}$ at $\Gamma$ (LDA, DZP, experimental
  lattices): Si 50.4 meV (h 0.15 Å, 4$^3$; experiment 44), AlAs 303 meV
  (h 0.15; experiment 0.28 eV), AlSb 678 meV (h 0.15; experiment 0.676
  eV), GaAs 334 meV (h 0.15, 4$^3$; 341 at h 0.11; experiment 0.341 eV; the
  III-V values from Vurgaftman, Meyer and Ram-Mohan 2001); PBE gives Si
  49.1 and GaAs 320 meV. GaAs comes
  out inverted in this basis and grid ($\Gamma_6$ below $\Gamma_8$): its
  Ga 3d sits far too shallow and sinks only slowly with the grid, so its
  gap is not to be trusted there.
- **Checked, meta-GGA and hybrid**: with the spin-orbit term zeroed, or
  the O$_2$ triplet's moment vectors along $z$, $x$ or $y$ (a complex spin
  rotation), r2SCAN, r2SCAN+rVV10 and HSE06 crystals stop at their first
  step on the collinear energy (1e-15 to 2e-14 Ha); OH's moment turned to
  $x$ or $y$ is its UKS energy (r2SCAN, HSE06: 1e-14 Ha). The matrix is the
  energy's derivative (r2SCAN 3e-10; HSE06 3e-8, the grid exchange's
  1e-7-level asymmetry, which the collinear crystal shares). Without the
  term the spinor Wannier functions are two copies of the collinear ones
  (spreads 3e-12) and the Berry phases twice the collinear phases (modulo
  $2\pi$); AlSb's Born charge is 1.898 with the term, 1.8995 without.
  $\Delta_{so}$ at $\Gamma$ (DZP): Si r2SCAN 49.0 meV (h 0.2 Å, 4$^3$; PBE
  49.1), HSE06 53.2 meV (3$^3$; PBE 49.1); GaAs r2SCAN 324 meV (h 0.15 Å,
  4$^3$; PBE 320), HSE06 334 meV (2$^3$; PBE 316), where HSE06 also lifts
  $\Gamma_6$ above $\Gamma_8$ (gap 0.99 eV). The exact exchange of the
  spin-orbit-split occupied states raises $\Delta_{so}$; a scalar
  functional cannot (in SZ silicon, where symmetry fixes the
  $\Gamma_{25'}$ states, PBE, r2SCAN and Gamma-only HSE06 give the same
  55.7 meV and HSE06 on 2$^3$ 58.3). The hybrid's value moves with the
  k-mesh; the hybrid on the whole mesh is costly (Si DZP 3$^3$: 19 min,
  GaAs DZP 2$^3$: 50 min, 2.7 GB).

Refused for every two-component run, each with its own message:
**forces** (the analytic gradient has no spin-orbit or noncollinear term;
relax on a scalar set).

## Crystal forces, stress and cell relaxation

For a crystal, `atoms.get_forces()` and `atoms.get_stress()` are the
derivatives of the **free energy** $F = E - \sigma S$ (what smearing
makes variational), and `atoms.get_potential_energy(force_consistent=True)`
returns that F, so ASE's optimizers and cell filters relax consistently:

```python
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS

si = bulk("Si", "diamond", a=5.50)
si.calc = Mandacaru(method="dft", xc="lda", grid_spacing=0.25,
                    basis={"name": "PAW-LCAO", "size": "DZP", "filter_cutoff": 300},
                    kpts={"size": (4, 4, 4), "gamma": True})
BFGS(FrechetCellFilter(si)).run(fmax=0.01)
```

- **Forces** are analytic: the Hellmann-Feynman term (each atom's projectors,
  short-range local potential, compensation charges, Gaussian ion and partial
  core moving) plus the Pulay term (its basis functions moving: the
  Cartesian derivatives of the basis functions, evaluated analytically), with
  the energy-weighted density matrix for the overlap. `atoms.calc.force_result`
  keeps the breakdown. They match a finite difference of the self-consistent
  F to better than 1e-5 eV/Angstrom (LDA, PBE and r2SCAN). Their sum over the
  atoms is the grid's egg-box, not zero (about 1e-3 eV/Angstrom for LDA,
  1e-2 for the gradient functionals at grid_spacing = 0.25 Angstrom).
- **The stress** (eV/Angstrom^3, ASE's sign: negative for a compressed cell)
  strains the cell, the atoms and the grid together, with the converged
  states carried along -- no new SCF. Only the strains the space group
  leaves unchanged are applied, since the stress lies in their span: two
  rebuilds of a cubic crystal, twelve without symmetry. The
  basis filter is held at the cutoff the run used, so the stress never
  includes the basis changing with the grid spacing. The default grid-tied
  filter (`"auto"`) is held across a relaxation the same way: see
  [Fourier filtering](pseudopotentials.md) ("The `auto` cutoff is resolved
  once per calculator").
- **Fixed node counts.** The grid follows the cell during a relaxation with
  the node counts set by `grid_spacing`; when a count changes, the energy steps. Choose
  `grid_spacing` so the counts stay put over the range the cell explores (the check
  below kept 15 nodes from a = 5.50 to 5.80 Angstrom with grid_spacing = 0.265).
- **Checked**: Si (SZ, 2x2x2) relaxed from a = 5.50 Angstrom with a distorted
  basis to a = 5.7179 Angstrom in 9 BFGS steps, where an E(a) scan on the
  same settings has its minimum at 5.7180 (an SZ number, not silicon's
  lattice constant).
- **Cost**: both run on the SCF's irreducible k-points and are symmetrized
  (the force by the space group with its atom map); they equal the full
  mesh's to round-off. Diamond Si, DZP, 4x4x4, grid_spacing = 0.2 Angstrom: forces
  29 s, stress 7.5 s on 8 cores.
- **A molecule has no stress.** Kohn-Sham is periodic only when `atoms.pbc`
  has a periodic direction, and `calc.implemented_properties` lists `"stress"`
  only then. On a molecule, `get_stress()` raises ASE's
  `PropertyNotImplementedError`, as ASE's own calculators do for a cell-less
  geometry. So a cell filter cannot wrap a molecule, and neither can
  `ase.calculators.fd.FiniteDifferenceCalculator(..., eps_strain=None)`:
  `None` there means "use the calculator's analytic stress", not "skip the
  stress". A plain `BFGS(atoms)` never asks for the stress. For
  finite-difference forces on a molecule, call
  `ase.calculators.fd.calculate_numerical_forces(atoms, eps=0.01)`, or keep
  the wrapper's default `eps_strain`. The wrapper then strains the box
  numerically, at twelve extra energies per call, and that stress means
  nothing for a free molecule.

## Bands and densities of states

After `atoms.get_potential_energy()`, the converged potential can be reused
without another SCF. Every method below is reached as `atoms.calc.<name>`.

```python
from ase.build import bulk
from mandacaru import Mandacaru

si = bulk("Si", "diamond", a=5.43)
si.calc = Mandacaru(method="dft", xc="lda", grid_spacing=0.25,
                    kpts={"size": (4, 4, 4), "gamma": True},
                    basis={"name": "PAW-LCAO", "size": "DZP"})
si.get_potential_energy()

bands = si.calc.band_structure(path="LGXWKG", npoints=80)
bands.plot()                                      # ASE BandStructure

structure, weights = si.calc.fat_bands(path="LGXWKG", npoints=80)
energies, dos = si.calc.dos(width=0.1, kpts=(8, 8, 8))
energies, pdos = si.calc.pdos(width=0.1, kpts=(8, 8, 8))
energies = energies - si.calc.get_fermi_level()   # center on the Fermi level
```

- **`band_structure(path=None, npoints=100)`** freezes the converged potential
  and diagonalizes $H(\mathbf k)$ at every point of `path`: a string of
  high-symmetry labels such as `"LGXWKG"`, or `None` for the lattice's own
  path. It is non-self-consistent, so the bands are continuous along the path
  and not limited to the SCF mesh. It returns an ASE `BandStructure` (eV, one
  spin channel, `reference` the Fermi level) and `.plot()` draws it. Crystals
  only: a molecule has no Brillouin zone and refuses it.
- **`fat_bands(path=None, npoints=100)`** returns `(band_structure, weights)`.
  `weights[(atom, l)]` is a `(n_kpoints, n_bands)` array of the Loewdin weight
  of each state on the shell of angular momentum `l` of atom `atom`; the weights
  of one state sum to one over every `(atom, l)`.
- **`dos(width=0.1, npoints=2001, kpts=None, energies=None)`** returns
  `(energies, dos)`. Energies are in eV and `dos` is in states/eV per cell (per
  molecule), counting both spins, so it integrates to twice the number of
  bands. `width` is the standard deviation of the Gaussian, in eV. `kpts=` takes
  a mesh like the calculator's own and diagonalizes it non-self-consistently
  after the same symmetry reduction, so a mesh denser than the SCF one gives a
  smoother curve; `None` uses the SCF mesh. A molecule has no `kpts`: passing
  one is refused.
- **`pdos(...)`** takes the same arguments and returns `(energies, pdos)` with
  `pdos[(atom, l)]`, the part of the total carried by the Loewdin-orthogonalized
  orbitals of angular momentum `l` on atom `atom` (an index into the `Atoms`).
  The shells sum to the total. On a symmetry-reduced mesh each atom's share is
  averaged over the atoms the operations map it to, which makes it the full
  mesh's. A molecule works the same way, with one "k-point" and the squared
  orbital coefficients in the Loewdin basis. Like any division of a state
  between atoms, the shares depend on the basis: diffuse functions on one
  atom overlap its neighbors, and the symmetric orthogonalization spreads
  that overlap evenly. Water's lowest level is 68 % oxygen in an SZ basis
  and 50 % in DZP (Mulliken: 80 % and 64 %), while the lone pair stays
  oxygen's. Compare shares within one basis, not across bases.
- **`integration="tetrahedron"`** (on `dos` and `pdos`; the default is
  `"gaussian"`) replaces the Gaussians by the linear tetrahedron method:
  every band is interpolated linearly inside the six tetrahedra that cut
  each cell of the k-mesh along its shortest diagonal, and the interpolant
  is integrated exactly. There is no width (`width` then only pads the
  default energy axis), band edges and van Hove singularities stay sharp,
  and a gap is exactly empty. Each value is the DOS averaged over its
  energy's cell of the axis, taken from the exact number of states at the
  cell's edges, so the values times the spacing add up to the count and a
  band flat across a tetrahedron is not lost; pass distinct `energies`. In
  `pdos` each shell's weight is interpolated with its band, corner by
  corner, rather than averaged over a tetrahedron, so the shells still sum
  to the total. The bands are diagonalized on the irreducible wedge and
  unfolded onto the whole mesh: a band energy is the same at every image of
  a k-point, and a shell's weight summed over the atoms the operations
  permute is too, so that sum is unfolded and shared equally among those
  atoms. Degenerate states at one k-point (within 1 meV) share their
  weights, since which of them carries which atom is the eigensolver's
  arbitrary choice.

  ```python
  energies, dos = si.calc.dos(integration="tetrahedron", kpts=(8, 8, 8))
  energies, pdos = si.calc.pdos(integration="tetrahedron", kpts=(8, 8, 8))
  ```

  `curvature_correction=True` adds Bloechl's correction, which restores to
  leading order the curvature the linear interpolation misses. It moves
  weight between the corners of each tetrahedron without changing their
  sum, so `dos` is the same with it or without it, and it acts on `pdos`
  only -- where it is **off by default**. It is meant for the Fermi-sea
  integrals of a metal at a fixed count (30 to 1000 times closer with it on
  free electrons), not for a projected DOS: a corrected shell goes negative
  where its projection is small at a band extremum on a mesh point, and its
  error at a band onset does not shrink with the mesh (free electrons
  projected on $\sin^2 \pi k_x$: a relative error of 0.4 at 8, 16 and 32
  points per axis, against 0.18 to 0.15 uncorrected). For an insulator's
  shell populations it changes nothing (a full band's correction is zero),
  while it reshapes silicon's s-PDOS by 38 % of its peak on a 4x4x4 mesh.
  Uncorrected, every shell is non-negative by construction.

  Near a band edge, mind the mesh: within the first k-shell of a band
  extremum that sits on a mesh point the linear tetrahedron DOS rises
  quadratically, $(E - E_0)^2$, not as $\sqrt{E - E_0}$; the square root
  appears only beyond it. Do not fit edge exponents (or effective masses)
  inside that first shell.

  The tetrahedron method needs a crystal and a uniform mesh,
  `kpts=(n1, n2, n3)` or `{"size": ..., "gamma": ...}`, with at least two
  points along every periodic direction: a molecule, a band path, an
  explicit list of k-points and a Gamma-only mesh are refused.

  Use the tetrahedra for the spectrum itself: band edges, gaps, the DOS at
  the Fermi level of a metal, and a coarse mesh, where a Gaussian must be
  wide enough to bridge the mesh spacing. On a 6x6x6 mesh of the
  simple-cubic tight-binding band, broadening the tetrahedron DOS by a
  Gaussian of 0.2 t comes 6 times closer to the converged spectrum at that
  resolution than broadening the levels by the same Gaussian. Keep the
  Gaussian for a spectrum meant to be smooth at a chosen resolution on a
  mesh that resolves it, and for molecules.
- **`fermi_surface((n1, n2, n3))`** returns the band energies (eV) on the
  full Gamma-centered mesh `k = (i1/n1, i2/n2, i3/n3)`, shaped
  `(n_bands, n1, n2, n3)` (`(2, n_bands, ...)` spin-polarized). Only the
  irreducible points are diagonalized and the rest are filled in by symmetry,
  which a band energy keeps exactly. **`write_fermi_surface(path, (n1, n2,
  n3), bands=None, spin=None)`** writes them as an XCrySDen `.bxsf` file,
  with the Fermi level in its header; XCrySDen and FermiSurfer draw the
  surface from it. A spin-polarized crystal has one surface per channel, so
  pass `spin=0` or `1`. Pick the bands that cross the Fermi level with
  `bands=[...]` to keep the file small.

  ```python
  al.calc.write_fermi_surface("al.bxsf", (24, 24, 24))
  ```

- **The energy reference.** The energies are on the eigenvalues' own reference
  (the plane-wave zero for a crystal, see above), not on the Fermi level.
  Subtract `atoms.calc.get_fermi_level()` to center a plot. For a molecule the
  Fermi level lies midway between the HOMO and the LUMO.

The ASE getters are available too: `get_eigenvalues(kpt=0, spin=0)` in eV at
the `kpt`-th irreducible SCF k-point (`spin` 0 or 1 for a spin-polarized run),
`get_fermi_level()` in eV, `get_ibz_k_points()` (fractional, reduced),
`get_k_point_weights()` (summing to one) and `get_number_of_spins()` (1, or 2
for a spin-polarized run; see *Spin polarization*).

The mesh band gap in `result.scf.band_gap` is an upper bound whenever a band
edge lies between the mesh points; the gap along a `band_structure` path shows
how much. See [the silicon bands example](../../../examples/dft/03_dft_si_bands.py)
and, for a molecule, [the density-of-states
example](../../../examples/dft/02_dft_h2o_dos.py).

## Electric polarization: the Berry phase

The polarization of a crystal is not the dipole of its cell, which depends
on where the cell is cut. It is a Berry phase of the occupied Bloch states
(King-Smith and Vanderbilt; Resta). For an insulating crystal:

```python
from ase.build import bulk
from mandacaru import Mandacaru

lih = bulk("LiH", "rocksalt", a=4.0)
lih.calc = Mandacaru(method="dft", basis={"name": "PAW-LCAO", "size": "SZP"},
                     kpts={"size": (4, 4, 4), "gamma": True},
                     smearing={"method": "fermi-dirac", "width": 0.001})
lih.get_potential_energy()
lih.calc.get_polarization()          # C/m^2, the branch nearest zero
print(lih.calc.polarization_result.summary())
```

- **Strings.** Along each reciprocal vector the k-points of the mesh form
  closed strings. The phase $\phi = -\operatorname{Im}\ln\prod\det M$ of
  the overlaps $M_{mn} = \langle u_{m\mathbf k}|u_{n\mathbf k+\mathbf b}
  \rangle$ between the occupied bands is averaged over the strings.
  `get_polarization(kpts=...)` names the mesh, by default the SCF's. Every
  point is diagonalized at the converged potential.
- **The PAW-LCAO overlaps** add, inside each sphere, the augmentation's
  charge and dipole, to first order in the string step. The neglected term
  shrinks as the strings get denser.
- **The quantum.** The Berry phase is defined modulo $2\pi$, and an ion can
  be moved by a lattice vector, so $\mathbf P$ is defined modulo
  $g\,e\,\mathbf a_j/\Omega$ along each lattice vector, with $g$ the
  greatest common divisor of the band occupation (2) and the valence
  charges. The result reports $\mathbf P$ on the branch nearest zero, its
  coordinates in quanta (`fractional`), the quanta and the unreduced
  coordinates (`raw`). Only *changes* of $\mathbf P$ are physical: compare
  `raw` along a path and wrap the difference to the nearest quantum. A
  centrosymmetric crystal sits at 0 or half a quantum.
- **Insulators only.** A partly filled state, from a metal or from a
  smearing wider than the gap, is refused, and so is a band count that
  changes across the zone. Use a narrow smearing (`width` 0.001 eV) for an
  insulator.

**Born effective charges.**
$Z^*_{A,ij} = (\Omega/e)\,\partial P_i/\partial u_{Aj}$ comes from central
differences, with every atom moved by $\pm$`delta` (0.01 Å) along each
axis, which is $6N$ self-consistent runs:

```python
z = lih.calc.born_effective_charges(lih)
z.tensors        # (n_atoms, 3, 3), units of e
z.sum_rule       # sum over atoms: zero up to the k-point convergence
```

**Piezoelectric constants.** `piezoelectric_tensor` gives the *proper*
tensor $e_{i\mu} = \partial P_i/\partial\eta_\mu$ (C/m², Voigt order
xx, yy, zz, yz, xz, xy). Each strain is applied at $\pm\eta$ (0.005). The
raw polarization coordinates are differenced against the unstrained
lattice, which removes the improper part a strained quantum would add.
Two variants:

- **Clamped-ion:** the atoms move with the cell, in fractional coordinates.
- **Relaxed-ion** (`relax_ions=True`): the internal coordinates are relaxed
  at every strain with the crystal forces, to `fmax` (0.01 eV/Å). The
  centroid of the atoms is held fixed and the mean force removed. A
  crystal's exact forces sum to zero, but the real-space grid gives them a
  net part (0.11 eV/Å for AlN at `grid_spacing = 0.25` Å). Left in, that net force
  drags the whole crystal along the grid and corrupts the internal
  coordinates.

`components=` limits the strains, for example `(0, 2)` for a wurtzite's
$e_{31}$ and $e_{33}$:

```python
e = atoms.calc.piezoelectric_tensor(atoms, components=(0, 2))
e.tensor        # (3, 6), C/m^2
```

**A finite electric field and the dielectric tensor.** In a crystal,
`electric_field=(Fx, Fy, Fz)` (Hartree per e Bohr) couples the field through
the Berry phase, with the ions clamped (Souza, Íñiguez and Vanderbilt;
Umari and Pasquarello). The functional
$E_{KS} - \Omega\,\boldsymbol{\mathcal E}\cdot\mathbf P$ is minimized on the
full k-mesh, unreduced, because the field breaks the symmetry. Each
k-point's Kohn-Sham matrix gains a term built from its neighbors along the
strings. `get_polarization()` then reads $\mathbf P$ from the converged
states themselves. `dielectric_tensor` takes $\pm\mathcal E$ along each
axis, seven runs in all:

```python
eps = atoms.calc.dielectric_tensor(atoms)     # field 2e-4 a.u. by default
eps.tensor          # epsilon_inf = 1 + 4 pi chi
eps.energy_diagonal # chi_jj again, from the energy's curvature
```

The response from the polarization and from the energy curvature agree to
every printed digit: water in a 9 Å box gives $\Omega\chi_{yy}$ = 8.298 a.u.
both ways. There are limits on the field and the system:

- **The Zener limit.** Keep the field below
  $e\,\mathcal E\,N_j a_j < E_{gap}$, the field times the length of the
  k-point supercell. Beyond it the occupied states of neighboring k-points
  stop overlapping, and the run is refused.
- **Restricted, semilocal crystals only.** A spin-polarized crystal and a
  hybrid functional are refused, and so are forces and the stress in a
  field.

**Not yet validated against experiment.** The same atom in a box gives
different levels and densities in the crystal and the molecular paths
(K20), so absolute $\varepsilon^\infty$ values for real solids wait for that
to be resolved.

Checks (LDA, PAW-LCAO SZP, grid_spacing = 0.25 Å, a 4³ or 4×4×3 mesh):

| | Mandacaru | literature |
| :--- | ---: | ---: |
| $Z^*$, rocksalt LiH | Li +1.097, H −1.097 | about 1.0–1.1 |
| $Z^*$, rocksalt NaCl | Na +1.100, Cl −1.100 | about 1.1 |
| $Z^*_{zz}$, wurtzite AlN | Al +2.682, N −2.689 | 2.65–2.70 |
| spontaneous $P$ of AlN (relative to zinc blende) | −0.089 C/m² | −0.08 to −0.09 |

The acoustic sum rule holds to 0.01 e, and the off-diagonal elements of
the cubic crystals are zero ($10^{-8}$). The spontaneous polarization is the
difference of the formal polarizations of wurtzite and of zinc blende in a
six-atom hexagonal cell with the same in-plane lattice, so both have the same
quantum along $c$, $e/A$. A centrosymmetric crystal comes out on 0 or half a
quantum, and a rigid shift of all atoms changes $\mathbf P$ only by whole
quanta.

For AlN's clamped-ion tensor the basis gives $e_{33} = -0.44$ to $-0.48$
and $e_{31} = +0.25$ to $+0.27$ C/m² (table below), against about $-0.47$ for
$e_{33}$ in the literature; the components symmetry forbids come out at
$10^{-13}$.

The relaxed-ion tensor is the clamped one plus the ions' response,

$$e_{3\mu} = e^{clamped}_{3\mu} + \frac{4e}{\sqrt3\,a^2}\,Z^*_N\,
\frac{\partial u}{\partial\eta_\mu},\qquad
\frac{\partial u}{\partial\eta_\mu} = \frac{1}{k_u}\frac{\partial F_u}{\partial\eta_\mu},$$

(Bernardini, Fiorentini and Vanderbilt), with $u$ the internal parameter,
$F_u = -\partial E/\partial u$ and $k_u = -\partial F_u/\partial u$. Each
piece is its own set of central differences, so the table shows where a
number comes from. All rows are LDA, PAW-LCAO, grid_spacing = 0.25 Å, a
4×4×3 mesh, Fermi-Dirac 0.001 eV; $a$, $c$ in Å, $k_u$ in eV per unit of
$u$, $e$ in C/m²:

| basis | lattice ($a$, $c$, $u$) | $Z^*_N$ | $k_u$ | clamped $e_{33}$ / $e_{31}$ | ionic $e_{33}$ / $e_{31}$ | relaxed $e_{33}$ / $e_{31}$ |
|---|---|---:|---:|---:|---:|---:|
| SZP | 3.1600, 5.0639, 0.38171 (SZP equilibrium) | −2.692 | 623.8 | −0.475 / +0.268 | +2.000 / −0.934 | 1.525 / −0.666 |
| DZP | 3.1600, 5.0639, 0.38171 (SZP's) | −2.706 | 572.6 | −0.479 / +0.265 | +1.927 / −0.900 | 1.448 / −0.635 |
| TZP | 3.1600, 5.0639, 0.38171 (SZP's) | −2.706 | 576.5 | −0.457 / +0.264 | +1.907 / −0.895 | 1.450 / −0.631 |
| DZP | 3.1408, 5.0299, 0.38221 (DZP equilibrium) | −2.704 | 594.3 | −0.463 / +0.260 | +2.047 / −0.968 | 1.584 / −0.709 |
| SZP | 3.1100, 4.9800, 0.38200 (experimental) | −2.689 | 686.7 | −0.441 / +0.254 | +2.304 / −1.020 | 1.863 / −0.766 |

(The ionic column is the second term of the equation, from
$\partial u/\partial\eta_3$ / $\partial u/\partial\eta_1$ = −0.2005 /
+0.0936 (SZP), −0.1922 / +0.0898 (DZP), −0.1901 / +0.0892 (TZP), −0.2018 /
+0.0955 (DZP, own lattice) and −0.2240 / +0.0992 (SZP, experimental).)
A direct finite difference of the relaxed-ion tensor (every strain relaxed
with the crystal forces, SZP, first row's lattice) gives 1.480 / −0.682
with `fmax=0.002` against the pieces' 1.525 / −0.666, so the two routes
agree to 3 % in $e_{33}$ and 2 % in $e_{31}$. The default `fmax=0.01` stops
too early for $e_{31}$ (−0.594) and leaves $e_{33}$ unchanged: tighten
`fmax` when $e_{31}$ matters.

The literature's 1.46 and −0.60 (Bernardini, Fiorentini and Vanderbilt,
*Phys. Rev. B* **56**, R10024, 1997) are matched only by DZP and TZP on SZP's
lattice. That agreement is not a validation: it is 0.6 % in $a$ away from
their own equilibrium (stress +3.6 / +3.4 GPa in DZP, +4.2 / +4.0 GPa in
TZP, which wants a smaller cell), and at DZP's own equilibrium the tensor is
**1.58 / −0.71, 8 % (e33) and 18 % (e31) above the literature**. The
tensor is steep in the lattice: from 3.16 to 3.11 Å (SZP) $e_{33}$ rises
by 0.34, about 0.2 C/m² per 1 % of $a$. A calculation at the experimental
lattice is therefore not a fair test (SZP: 1.86 / −0.77, 28 % high in both);
neither is a comparison closer than the lattice is reproduced. The excess
sits in the ions' term, not in $Z^*$ (2.69–2.71, against 2.65–2.70) or the
clamped $e_{33}$ (−0.46 to −0.48, against −0.47). These models' cell is 1 %
too large (DZP: $a$ 3.141 Å against 3.11 measured; fcc Al comes out at
4.12 Å against about 4.0 Å in LDA), and a softer internal mode ($k_u$) raises
$\partial u/\partial\eta$. Untested causes: the basis, the grid
(the egg-box), the k-mesh and the frozen core of the dataset. Treat the
relaxed-ion tensor as good to about 20 % for the response and to about 3 %
for the numerical route.

## Wannier functions

`atoms.calc.wannier()` turns a group of a crystal's bands into maximally
localized Wannier functions (Marzari and Vanderbilt). Each function is a
unitary mixture of Bloch states that minimizes the total spread
$\Omega = \sum_n (\langle r^2\rangle_n - \langle\mathbf r\rangle_n^2)$.
By default the group is the occupied bands of an insulator:

```python
w = si.calc.wannier()                   # valence: trial orbitals on the bonds
w = si.calc.wannier(guess=[[...], ...], kpts=(6, 6, 6))
print(w.summary())                      # centers (A), spreads (A^2)
w.omega_invariant                       # Omega_I: depends on the subspace only
w.hamiltonian((0, 0, 1))                # <w_0|H|w_R>, eV
w.interpolate(kpoints_fractional)       # Wannier-interpolated bands, eV
w.write(0, "w0.cube")                   # function 0 on the grid (.cube, .xsf)
```

- **The mesh.** The full Γ-centered `kpts` mesh is diagonalized at the
  converged potential.
- **The overlaps.** The overlaps between neighboring k-points are those of
  the Berry phase, PAW on-site terms included. The finite-difference shells
  satisfy $\sum_b w_b b_ib_j = \delta_{ij}$, one shell for a cubic mesh and
  more for a low-symmetry one.
- **The starting gauge.** It comes from trial orbitals, one per function.
  `guess="bonds"`, the default, puts s Gaussians on the nearest-neighbor
  bond midpoints, which suits a covalent crystal's valence bands.
  `guess="sp3"` puts four sp³ hybrids on every atom, one toward each
  neighbor. A list of centers (Å) puts s Gaussians there. Orbitals of any
  angular momentum on atoms, such as a transition metal's d shell, are
  named by atom and orbital; see "Trial orbitals on atoms" below.
- **The minimization.** Steepest descent minimizes the gauge-dependent part
  of the spread.

Silicon's valence bands (LDA, PAW-LCAO SZP) give four equivalent functions
centered exactly on the bond midpoints. The spread per function converges
from below with the mesh: 1.70, 1.99 and 2.13 Å² on 4³, 6³ and 8³ meshes;
the literature reports about 1.9–2.0 Å² with LDA. $\Omega_I$ is the same
to eight digits from a poor starting guess. Minus the sum of the centers
equals the electrons' Berry-phase polarization, modulo the quantum.

### Trial orbitals on atoms

`guess` can name orbitals on atoms. The number of functions then defaults
to the number of trial orbitals:

```python
w = cu.calc.wannier(guess={"Cu": ["s", "p", "d"]}, windows=...)  # nine
w = svo.calc.wannier(guess={"V": "t2g"}, windows=...)          # three
w = calc.wannier(guess=[("V", "t2g", frame)], windows=...)     # rotated
w = calc.wannier(guess=[("Cu", "d"), ((0.9, 0.9, 0.9), "s")], ...)
w.labels                                # ("V1:dxy", "V1:dyz", "V1:dxz")
```

- **Sites.** A spec is a dict `{site: orbitals}` or a list of
  `(site, orbitals)` or `(site, orbitals, frame)`. A site is a chemical
  symbol (every atom of that element, in index order), an atom index, or,
  in a list, a Cartesian position in Å that need not hold an atom.
- **Orbitals.** One name or a list of names:
  - a whole shell, every m from −l to l: `"s"`, `"p"` (py, pz, px), `"d"`
    (dxy, dyz, dz2, dxz, dx2-y2) or `"f"`;
  - one real harmonic: `"px"`, `"py"`, `"pz"`, `"dxy"`, `"dyz"`, `"dz2"`,
    `"dxz"` or `"dx2-y2"`;
  - a d subset of an octahedral site: `"t2g"` (dxy, dyz, dxz, lobes
    between the axes) or `"eg"` (dz2, dx2-y2, lobes along them);
  - hybrids: `"sp"` (along ±z), `"sp2"` (in the xy plane, toward +x and
    120° either side) or `"sp3"` (toward (1, 1, 1), (1, −1, −1),
    (−1, 1, −1) and (−1, −1, 1)).
- **The frame.** The harmonics, the t2g/eg split and the hybrids'
  directions are defined in the local frame, three rows that are the local
  x, y and z axes in Cartesian coordinates. The default frame is the
  Cartesian axes; give an octahedron's own axes when it is rotated.
- **The harmonics.** They are the real spherical harmonics a molecule's
  basis uses (the real combinations of the Condon–Shortley ones), positive
  along their Cartesian lobes: $S_{1,1}\propto x$, $S_{1,-1}\propto y$,
  $S_{2,-2}\propto xy$, $S_{2,2}\propto x^2-y^2$.
- **The radial shape.** `trial_radial="gaussian"`, the default, uses the
  normalized $r^l e^{-r^2/2w^2}$ with width $w$ = 1 Bohr (the `width`
  option). `trial_radial="basis"` uses the radial function of the atom's
  own basis orbital of that l: its first zeta, or a polarization function
  where there is no other. Trials at a position without an atom keep
  Gaussians.
- **The count.** Without windows the bands are an isolated group, one
  function per band, so the trial orbitals must match the bands; a metal
  needs `bands=` or windows. Mismatches are refused with the counts.

The projection only has to overlap the target bands well, so the radial
shape does not matter to the result. Copper's nine s, p and d functions,
disentangled from the fifteen bands of a double-zeta basis (8³ mesh,
frozen up to 1 eV above the Fermi level), give the same $\Omega_I$
(2.98499 Å²) from Gaussians and from the basis' own radial functions, to
2e-11 Å². The centers sit on the atom and the d functions are the most
compact: 0.401 Å² (t2g) and 0.405 Å² (eg), against 1.01 Å² for p and
1.25 Å² for s.

Inside the frozen window, Wannier interpolation of copper's d bands and of
the s band that crosses them gives the following largest (root mean square)
errors over 60 random points, in meV:

| basis | bands | 6³ | 8³ | 10³ |
|---|---|---|---|---|
| single zeta (SZP) | 9 | 60 (10) | 15 (2.8) | 2.4 (0.4) |
| double zeta (DZP) | 15 | | 53 (23) | 35 (9) |

With the single-zeta basis the nine functions span every band, so the
disentanglement does nothing; the error is the functions' tails beyond
the Born-von Karman supercell, and the 10³ mesh is within 10 meV.
With the double-zeta basis the six extra
bands make the disentanglement real, and the interpolation needs a denser
mesh.

Two other copper sets fail. Six functions (s and d on the atom) move the
s function to the tetrahedral interstitial site. Seven functions (d on the
atom plus s at both interstitial sites, as Souza, Marzari and Vanderbilt
used) do not converge in 3000 iterations, and the s centers drift.

`max_iter=0` skips the spread minimization and keeps the trial orbitals'
Löwdin gauge, giving *projected* Wannier functions. They keep the trials'
symmetry. In bcc Fe the minimization mixes s, p and d into hybrids that
point off the atom toward its neighbors, while the projected d functions
stay on the atom as a t2g and an eg set.

### Entangled bands

Conduction bands and the bands of a metal cross bands you do not want.
Ask for fewer functions than bands and give energy windows (eV, on the
scale of `get_fermi_level()`); the subspace is then disentangled first
(Souza, Marzari and Vanderbilt):

```python
ef = si.calc.get_fermi_level()
w = si.calc.wannier(8, guess="sp3",
                    windows={"outer": (-20, ef + 20), "frozen": (-20, ef + 1)})
```

- **The outer window** selects, at each k-point, the bands the functions
  are made of (default: every band of `bands`, itself every band of the
  basis when windows are given).
- **The frozen window** is optional. Its states are kept exactly, so the
  interpolated bands reproduce the true ones inside it.
- **The subspace.** At each k-point it minimizes $\Omega_I$: the frozen
  states plus the leading eigenvectors of
  $Z(\mathbf k) = \sum_b w_b M(\mathbf k,\mathbf b) P(\mathbf k+\mathbf b)
  M(\mathbf k,\mathbf b)^\dagger$ over the window's other states, mixed
  between iterations. The gauge is then minimized as for an isolated group.
- **Symmetry.** The subspace keeps the crystal's space group (Sakuma's
  symmetry-adapted disentanglement). The trials' projector and every
  $Z(\mathbf k)$ are averaged over the operations, each acting on the
  Bloch states through the basis' atoms and harmonics, before their
  leading eigenvectors are taken. $\Omega_I$ alone need not be lowest on a
  symmetric subspace: a symmetric subspace can be a saddle point that the
  plain iteration slowly leaves for a symmetry-broken one, and where it
  ends then depends on the trial orbitals. An operation is used only if it
  maps the windows' states onto themselves, so a window that cuts a
  degenerate set drops the operations it breaks. `w.symmetry_operations`
  counts the operations kept. `symmetrize=False` restores the plain
  iteration.

Silicon's valence and four lowest conduction bands give eight sp³
hybrids, 0.41 Å from each atom along its bonds, all eight equivalent. The
lowest conduction state lies 1.05 eV above $E_F$, just outside the frozen
window. Without the symmetrization the iteration from the sp³ hybrids
reached the symmetric subspace and then, after a few hundred iterations,
drifted to a symmetry-broken one with the two atoms inequivalent
($\Omega_I$ 7.8199 against the symmetric 7.8262 Å²). Random s Gaussians
reached yet another (7.8173 Å²). Now both reach the symmetric subspace,
whose $\Omega_I$ agrees to 1e-11 Å², in about 40 iterations. Freezing to
1.2 eV above $E_F$ gives the same subspace. The total spread is not: the hybrids are a
local minimum of the gauge-dependent part, and another start can find a
lower one. Inside the frozen window, the interpolated bands equal the
non-self-consistent bands exactly on the mesh. Between mesh points they
are within 318, 134, 22 and 5.7 meV (root mean square 98, 40, 5.0 and
1.3 meV) on 3³, 4³, 6³ and 8³ meshes.

### Interpolation

`hamiltonian(R)` and `interpolate` place each pair's hopping on a lattice
vector of the Born-von Karman supercell; the bands are exact on the mesh
either way. `replicas="centers"`, the default, puts it on the replica
nearest in the distance between the two functions' centers.
`replicas="cells"` uses the Wigner-Seitz vectors between cell origins.
Neither is better everywhere. For silicon's sp³ hybrids, nearest centers
lower the root-mean-square error from 154 to 98 meV on 3³, from 6.7 to
5.0 meV on 6³ and from 2.1 to 1.3 meV on 8³. Silicon's valence bond
functions hop far along the [110] chains, and there cell origins do
better: 20 against 26 meV root mean square on 6³, 7.7 against 10.2 meV on
8³. Either way the error falls with the mesh, so interpolation needs a
dense mesh.

### The functions on the grid

`w.orbital(n)` returns function `n` on the supercell grid with the box
centered on it: its smooth part, in Bohr^-3/2. The global phase is chosen
to make the function as real as possible. `w.write(n, path, part="real")`
writes it as `.cube` or `.xsf`, with the supercell's atoms; `part` can
also be `"imaginary"` or `"modulus"`. These are one-particle orbitals. In
a centrosymmetric crystal they are real up to a phase:
`w.imaginary_ratio(n)` is about 1e-14 for silicon's sp³ hybrids.

### A downfolded many-body problem

`w.downfold(functions)` writes the many-body problem of a fragment of the
functions. Name each function by its index `n` (in the home cell) or as
`(n, R)` (in the cell at lattice vector `R`). The result can go to a
quantum method:

```python
ef = si.calc.get_fermi_level()
w = si.calc.wannier(8, guess="sp3",
                    windows={"outer": (-20, ef + 20), "frozen": (-20, ef + 1)})
bond = w.downfold([3, 5])               # the two hybrids of one bond
bond.kohn_sham, bond.two_body           # h^KS and <pq|rs>, Hartree
calc = Mandacaru(method="adapt-vqe", **w.as_quantum_problem([3, 5]))
calc.run()
```

- **One-body.** `kohn_sham` is the Kohn-Sham Hamiltonian between the
  fragment's functions, on the crystal's eigenvalue zero.
- **Two-body.** `two_body` is the Coulomb interaction
  $\langle pq|rs\rangle$, bare by default. The pair densities are the
  smooth products plus the PAW compensation charges. On the Born-von
  Karman supercell the interaction is cut at half its shortest lattice
  vector (Spencer and Alavi); see "Interactions with the fragment's own
  copies" below.
- **Double counting.** $h^{KS}$ already holds the Hartree and
  exchange-correlation potential of the fragment's own electrons. The
  model subtracts their Hartree and exchange potential at the Kohn-Sham
  density matrix $\gamma$:
  $V^{dc}_{pq} = \sum_{rs}\gamma_{rs}[\langle pr|qs\rangle -
  \tfrac12\langle pr|sq\rangle]$, with the same (bare or screened)
  interaction as the model. The one-body part is $h^{KS} - V^{dc}$, so
  the model's Fock matrix at $\gamma$ is $h^{KS}$ again.
- **The handoff.** `as_quantum_problem()` writes the model in the natural
  orbitals of $\gamma$, most occupied first, and within a set of equal
  occupations (a full or an empty channel) in the eigenvectors of
  $h^{KS}$, so the reference determinant is the one nearest the
  Kohn-Sham state. The phases are fixed so the integrals stay real. The
  electron count is the rounded trace of $\gamma$, or
  `num_particles=(up, down)`.

Two checks. First, the four valence functions of a silicon cell are fully
occupied ($\gamma = 2I$), and their one-body part traces to the band
energy per cell. Second, the two hybrids of one bond form a two-site,
two-electron model: on a 4³ mesh, hopping −4.66 eV, $U = 12.05$ eV and
inter-site $V = 8.32$ eV, all bare. ADAPT-VQE reaches the exact ground
state of the model to 1e-10 eV.

### A screened interaction

`downfold(..., screening="crpa")` screens the interaction with the static
constrained random-phase approximation (Aryasetiawan et al.):

```python
bond = w.downfold([3, 5], screening="crpa")
bond.two_body                           # W(omega = 0) in the functions
bond.dielectric_head                    # eps^-1_00 at q -> 0
```

- **The polarizability.** $\chi$ is built from the Kohn-Sham states of
  the full mesh, the lowest `screening_bands` of them (default: every
  band of the basis), on the vectors $|\mathbf q+\mathbf G|$ below
  `screening_cutoff` (default 3 Bohr⁻¹).
- **The constraint.** Transitions inside the target subspace do not
  screen. The target is the Bloch subspace of the fragment's functions,
  all their lattice translates. With entangled bands a state lies only
  partly in it, so each transition is weighted by $1 - p_ap_b$, with $p$
  each state's weight in the subspace (the projector-weighted form of
  Şaşıoğlu, Friedrich and Blügel). `screening="rpa"` excludes nothing.
- **The screened interaction.** $W = [1 - v\chi_r]^{-1}v$ with the
  periodic $v = 4\pi/|\mathbf q+\mathbf G|^2$, evaluated in the Wannier
  pair densities. The $\mathbf q \to 0$ head is computed from the same
  polarizability at a small $\mathbf q$ along each axis, averaged; the
  wings are dropped.
- **What is not included.** The screening is static ($\omega = 0$, no
  frequency dependence). It comes from the Kohn-Sham states, without
  self-consistency, and is the same for both spins.

Silicon's bond (LDA, PAW-LCAO SZP): the bare $U = 12.19$, $V = 8.37$ and
$J = 0.37$ eV become 4.35, 2.70 and 0.234 eV on 3³ and 3.88, 2.29 and
0.231 eV on 6³. $U$ still falls by about 0.12 eV per mesh step: the
long-wavelength dielectric constant converges slowly with the mesh (the
head gives $\epsilon_M$ of 24, 17, 14 and 12 on 3³ to 6³). $J$, an
exchange density without charge, is screened least and does not move with
the mesh. Full RPA lies 0.10–0.13 eV below cRPA. Excluding the whole
eight-band sp³ set instead of the bond's subspace leaves $U = 7.99$ eV
(3³): most of silicon's screening is valence-to-conduction, inside that
set.
The SZP basis has 18 bands: keeping 8 or 12 of them gives $U = 4.87$ or
4.48 eV against 4.35 eV, so the empty states are not converged. The
cutoff is converged (2, 3 and 4 Bohr⁻¹ agree to 15 meV).

### Interactions with the fragment's own copies

The functions are periodic on the Born-von Karman supercell of the mesh.
The interaction is cut at half the supercell's shortest lattice vector, so
the fragment meets its own copies once its charge reaches past that
radius. A downfold warns when the fragment's span exceeds it. The span is
the largest distance between two of its centers plus 1.5
root-mean-square radii past each.

`coulomb="isolated"` keeps each function only on the supercell's
Wigner-Seitz cell centered on the fragment, sets it in a supercell twice
as large, and cuts the interaction at the whole shortest vector. It costs
eight times the grid points.

On silicon's sp³ hybrids, the largest difference between the two
interactions:

| mesh | radius (Å) | one bond | two bonds | four bonds |
|---|---|---|---|---|
| 3³ | 5.76 | 7 meV | 31 meV | 15 meV |
| 4³ | 7.68 | 0.8 meV | 2.2 meV | 1.9 meV |
| 5³ | 9.60 | 0.2 meV | 0.4 meV | |
| 6³ | 11.52 | 0.1 meV | | |

A fragment within the radius stayed under 10 meV; one past it reached
31 meV. The bare values converge with the mesh: $U$ 12.19, 12.05, 12.12
and 12.09 eV; $t$ −4.82, −4.66, −4.67 and −4.65 eV on 3³ to 6³. In a
small box the copies matter far more. Two neighboring sites of a hydrogen
chain in a 6 Å box lose 1.6 eV of their inter-site $V$ (3.91 against
5.52 eV) unless the interaction is isolated.

Rule: the fragment's span, the distance between its outer centers plus
three root-mean-square radii, should stay below half the supercell's
shortest lattice vector. With `coulomb="isolated"` it should stay below
the whole vector.

### Spin-polarized crystals

A spin-polarized crystal gets one set of functions per spin channel. Each
set comes from the channel's own Bloch states and gauge, and the trial
orbitals are shared:

```python
w = chain.calc.wannier(guess=chain.positions, bands=[0])
w.centers, w.spreads, w.omega_invariant # per spin: (2, n, 3), (2, n), (2,)
w.interpolate(kpoints_fractional)       # (2, nk, n), eV
w.write(0, "w0_up.cube", spin=0)
problem = w.downfold([0, (0, (1, 0, 0))], coulomb="isolated")  # two sites
```

- **Per-spin options.** `bands`, `n_functions` and `windows` can each be
  one value for both spins or a pair `(up, down)`.
- **The model.** `downfold` gives a spin-resolved model.
  - `kohn_sham`, `density_matrix` and `double_counting` are `(2, n, n)`,
    one block per spin.
  - `two_body[s, t]` is $\langle p^sq^t|r^ss^t\rangle$, between each
    channel's own functions.
  - The double counting is
    $V^{dc,\sigma}_{pq} = \sum_{\sigma'}\sum_{rs}\gamma^{\sigma'}_{rs}
    \langle p^\sigma r^{\sigma'}|q^\sigma s^{\sigma'}\rangle -
    \sum_{rs}\gamma^\sigma_{rs}\langle p^\sigma r^\sigma|s^\sigma
    q^\sigma\rangle$.
- **The handoff.** `as_quantum_problem()` passes the spin-dependent
  one-body part to `Mandacaru(method="adapt-vqe", ...)` in the
  spin-orbital Hamiltonian.

A ferromagnetic hydrogen chain checks it: one atom per 2.6 Å cell, moment
1 μB, the up band full and the down band empty, 4.9 eV above it.

- The band of each channel interpolates exactly on the mesh and within
  4 meV between mesh points.
- On one site, $\sum_\sigma \operatorname{tr}(\gamma^\sigma h^\sigma)$
  equals the band energy.
- The Fock identity holds per spin.
- ADAPT-VQE reaches the exact (1, 1) ground state of the two-site
  spin-dependent model, −28.045 eV.

Each function's global phase makes it as real as it can be, and real where
a real gauge exists. The model's integrals are then real. Pools of real
excitations need this: with arbitrary phases the couplings came out
imaginary, and ADAPT-VQE did not leave its reference. Inside a set of
equal occupations the natural orbitals are fixed by $h^{KS}$.

Two equivalent atoms per cell are a poorer test. The two functions of the
empty down band have an almost flat spread ($\Omega_{OD}$ = 0.009 Å²), so
their mixture is set by numerical noise.

A spin-polarized run of silicon, which keeps no moment, reproduces the
restricted functions and model to a few µeV.

### A d shell

A transition metal's d shell, or a subset of it, downfolds like any other
fragment. In cubic SrVO3 the three t2g bands around the Fermi level carry
vanadium's one d electron; the standard three-band model takes them alone:

```python
ef = svo.calc.get_fermi_level()
w = svo.calc.wannier(guess={"V": "t2g"},
                     windows={"outer": (ef - 2, ef + 2),
                              "frozen": (ef - 2, ef + 0.5)})
t2g = w.downfold(screening="crpa")      # the cell's three t2g functions
g = t2g.two_body                        # <pq|rs>, Hartree
U, U_, J = g[0, 0, 0, 0], g[0, 1, 0, 1], g[0, 1, 1, 0]   # Hubbard-Kanamori
```

- **Why windows.** They are the general recipe when the t2g bands entangle
  with others, and the model keeps the states below $E_F$ + 0.5 eV and takes
  the smoothest t2g-like subspace from the rest of the window, as cRPA
  studies do (Vaugier, Jiang and Biermann used a window of ±1.8 eV). On the
  3³ and 4³ meshes below they are not needed: bands 20–22 (above the V 3s and 3p and
  Sr 4s and 4p semicore and O's twelve; 16–18 with the valence-only V) are an isolated group, $-1.34$ to $+1.71$ eV
  from $E_F$, with a smallest direct gap of 0.78 eV to the next band (at R
  the next triple lies at $+4.05$ eV), and the isolated-bands model and the
  windowed one agree to three decimals ($U$ = 4.625 eV on 4³ for both).
- **What it gives.** Three equivalent functions centered on V. In
  physicists' notation, $U = \langle aa|aa\rangle$,
  $U' = \langle ab|ab\rangle$ and $J = \langle ab|ba\rangle$, each the
  same on every orbital or pair. $J$ equals the pair-hopping integral
  $\langle aa|bb\rangle$, because the functions are real. The model
  holds one electron.

SrVO3 (LDA, PAW-LCAO SZP, grid_spacing = 0.2 Å, a 3.84 Å, the shipped
library, Fermi-Dirac 0.05 eV, windows $E_F$ −2 to +2 eV with the states up
to $E_F$ + 0.5 eV frozen, cRPA with the default 3 Bohr⁻¹ cutoff and all 53
bands):

| mesh | spread (Å²) | U bare | U′ bare | J bare | U cRPA | U′ cRPA | J cRPA |
|---|---|---|---|---|---|---|---|
| 3³ | 1.65 | 15.57 | 14.43 | 0.528 | 4.721 | 3.713 | 0.484 |
| 4³ | 1.86 | 15.49 | | | 4.298 | 3.297 | 0.480 |
| 5³ (before the on-site remainder, valence-only V) | 1.96 | 15.23 | 14.23 | 0.461 | 4.439 | 3.541 | 0.429 |
| 6³ (before the on-site remainder, valence-only V) | 2.03 | 15.23 | 14.23 | 0.463 | 4.348 | 3.449 | 0.429 |

(eV; the 6³ cRPA took 6 h and 4.9 GB, the 5³ 1 h 50 min and 3.2 GB.) The
three orbitals agree to 1e-4 eV in $U$ and $U'$, and to 0.5 %
in $J$, which the cell grid's discretization sets. $(U - U')/2$ is
0.50 eV bare and 0.45 eV screened, 5 % above $J$: the relation
$U' = U - 2J$ is exact only for a spherical atom. The interpolated bands
inside the frozen window are within 144, 48, 25 and 8.6 meV of the true
ones on 3³ to 6³ meshes (root mean square 68, 13, 9.4 and 3.3 meV).

The 3³ and 4³ rows use the semicore vanadium (3s² 3p⁶ 3d³ 4s², 13 electrons)
and include the PAW on-site remainder (with the valence-only vanadium they were
15.76 / 5.144 / 0.503 on 3³ and 15.69 / 4.726 / 0.499 on 4³); the 5³ and 6³ rows
(hours of runs) and the silicon bond tables above were measured before it and
have not been re-measured. The remainder is the part of the all-electron pair
density inside each augmentation sphere that the smooth density and its
multipole compensation charges miss. It enters the bare integrals (the
one-center Coulomb correction), the transition densities of the
polarizability and the Wannier pair densities contracted with $W - v$. For
the valence-only vanadium's 3d shell alone it moves the Slater integrals $F^0$, $F^2$ and
$F^4$ by 0.42, 0.93 and 0.75 eV and the form factor of the pair density by
9 % at 2 Bohr⁻¹ and 44 % at 3 Bohr⁻¹. In SrVO3 (3³, valence-only vanadium) it raises the bare $U$ by
0.44 eV and $J$ by 0.07 eV (1.11 eV and 0.13 eV to $U'$ with the semicore vanadium, whose larger 3d sphere holds more of the pair density), increases the screening by 0.34 eV and leaves the
screened $U$ 0.10 eV higher; the 3³ to 4³ step (−0.42 eV) is unchanged.

**The mesh.** $U$ keeps falling, but more slowly each time (−0.41, −0.19,
−0.09 eV for 3³→4³→5³→6³). Four points on $N_k = N^3$ fit best as
$U_\infty + A/N^2$ (residuals under 7 meV), which is $N_k^{-2/3}$; a free
exponent gives $N_k^{-0.74}$, and $1/N_k$ alone leaves 27 meV. A fit in
$1/N$ ($N_k^{-1/3}$) leaves 40 meV and is the one to discard. The
extrapolated values (fitted before the on-site remainder; with it the 3³ to
4³ step is unchanged, so $U$ is about 0.10 eV higher, about 4.25 eV, $U'$ about
3.2 eV and $J$ about 0.50 eV: an estimate, not a refit) are

| | $N_k^{-2/3}$ | free exponent | $1/N_k$ | adopted |
|---|---|---|---|---|
| $U$ | 4.11 | 4.16 | 4.27 | 4.15 ± 0.15 |
| $U'$ | 3.21 | 3.25 | 3.37 | 3.25 ± 0.15 |
| $J$ | 0.427 | 0.429 | 0.428 | 0.43 |

The uncertainty is the spread among the fits; four points cannot give a
statistical error. The 5³ run, predicted at 4.44 to 4.47 before it ran,
did not tell the forms apart (they differ by 0.03 eV there); only a larger
mesh would. $J$ is converged on every mesh.

**The literature.** Vaugier, Jiang and Biermann (*Phys. Rev. B* **86**,
165105, 2012, Table III) give, for the same three-band model, $U$ = 3.2 eV,
$J$ = 0.44 eV and $U' = U - 2J$ = 2.3 eV, with a bare $V$ = 16.1 eV and
$J_{bare}$ = 0.55 eV. Miyake and Aryasetiawan (*Phys. Rev. B* **77**, 085122,
2008) report $U$ = 3.0 eV and $J$ = 0.43 eV, and Sakakibara and Kotani
(arXiv:1812.03731, 2018) $U$ = 3.12, $U'$ = 2.17 and $J$ = 0.448 eV. With the
on-site remainder the bare $V$ (15.7 eV) and $J$ (0.54 eV) are within 3 % of
the literature's, so the difference is in the screening: the estimated
extrapolated $U$ is **about 1.0 eV (30 %) above the literature** and $U'$
about 1.0 eV, and the screened $J$ (0.50 eV against 0.43 to 0.45) is lowered
by 7 % from its bare value against 20 % there. On 3³ the excess was 1.9 eV,
so the mesh accounts for about half of it. The screening is also short-ranged:
the long-wavelength dielectric constant of the same machinery is right (the
RPA $\varepsilon_\infty$ of SrTiO3 extrapolates to about 6.5 against about 6.2).

Three settings were tried on 4³ before the on-site remainder (reference 4.625 eV, all 49 bands, a cutoff
of 3 Bohr⁻¹) and none closes it:

- **The empty states.** Keeping the lowest 37 of the 49 bands raises $U$
  by 0.013 eV; a double-zeta basis (73 bands)
  lowers it by 0.049 eV, to 4.576 eV ($U'$ 3.695, $J$ 0.421).
- **The cutoff.** 4 Bohr⁻¹ changes $U$ by 0.0001 eV, while 2 Bohr⁻¹ raises
  it by 0.32 eV (4.947 eV). 3 Bohr⁻¹ is converged.
- **The Sr dataset.** The 4s and 4p semicore are in valence (since
  2026-10-11); that lowered $U$ by only 0.2 eV on 3³ (from 5.25 eV with the
  valence-only dataset).

So neither the empty states nor the cutoff nor the mesh explains the
remaining 1 eV, and it is not shown to be an artifact. What would settle
it: the mesh at 8³ or a second family of extrapolation points (the Wannier
spread, 1.6 to 2.0 Å², still grows with the mesh); DZP or TZP on 5³ and 6³,
since the O 2p-to-V d transitions that do the screening depend on the
basis; the frozen V 3s and 3p semicore, which is not polarizable here; and
a basis with diffuse functions, to test whether the short-range response is
limited by the empty states.

bcc Fe is the spin-polarized check (DZP, h 0.2 Å, moment 2.01 μB on a
6³ mesh). The projected s, p and d functions of each channel
(`guess={"Fe": ["s", "p", "d"]}`, `max_iter=0`, frozen up to 1 eV above
the Fermi level) sit on the atom.

- **Electron count.** The occupied states lie in the frozen window, so the
  traces of the two density matrices count the mesh's electrons, 5.005
  up and 2.995 down. Their difference is the moment.
- **d occupations.** The d functions hold 4.55 (up) and 2.51 (down)
  electrons, against the crystal's Löwdin d populations of 4.34 and
  2.30. Each channel is 0.21 electrons higher, but the d moment agrees to
  0.003 μB (2.039 against 2.042).
- **The model.** The bare spin-resolved d model has $U$ = 20.9 (t2g) and
  20.6 eV (eg) and $J$ = 0.67–0.77 eV. The Fock identity holds per spin
  to 1e-13 Ha, and the integrals are real.

The minimized functions of bcc Fe are not atom-centered. Without
`max_iter=0` the spread minimization turns s, p and d into hybrids
0.5–0.7 Å off the atom, and does not converge in 2000 iterations.

## What is not implemented

The following are refused or unavailable:

- **The forces and stress of a two-component (spin-orbit or
  noncollinear) molecule or crystal**, and the spinor crystal analyses
  listed in *Spin-orbit coupling: two-component Kohn-Sham* (projectabilities,
  natural orbitals, volumetric fields, the downfolded problem of spinor
  Wannier functions, a finite field).
- **Crystals with an all-electron basis**: the periodic path is
  PAW-LCAO or UPAW-LCAO only.
- **D4 with LDA or with r2SCAN+rVV10**, as above.
- **The dipole of a crystal**: cell-dependent; its counterpart is the
  Berry-phase polarization, `get_polarization()`.

A molecule's polarizability is in {doc}`polarizability`; a crystal's
electronic dielectric tensor is above.


## Local-potential quadrature

Molecular and periodic PAW-LCAO calculations use the same native radial-spline
and spherical-harmonic evaluator. Shell components share interpolation, cubic
coefficients are cached by the spline object, and projectors on the same sphere
share sampled basis values. The fallback keeps the same basis evaluation
contract. This applies independently of functional, spin, atom or basis size,
including polarization and diffuse shells; no quadrature nodes or weights are
removed. Periodic sphere chunks also avoid repeated coordinate sorting.

## Live SCF output and timing

Each completed SCF iteration is written and flushed immediately to `txt` and,
when enabled, the terminal. This includes restricted, unrestricted, periodic
and two-component calculations. The table is available before convergence and
before the occupied-orbital reference integrals are built.

`[PERFORMANCE]` lists exclusive wall times for basis sampling, overlap and
projector integration, short-range local quadrature, PAW compensation matrices,
Poisson solves, density construction, XC evaluation and matrix projection,
SCF diagonalization and mixing, and occupied reference integrals. Nested stages
exclude their children's time, so their sum can be compared with total wall
time without double counting. Integration backend and OpenMP thread counts
describe the integral library; they do not measure BLAS thread utilization.


## References

- Hohenberg, P. and Kohn, W. (1964). Inhomogeneous electron gas. *Phys. Rev.*
  136, B864.
- Kohn, W. and Sham, L. J. (1965). Self-consistent equations including exchange
  and correlation effects. *Phys. Rev.* 140, A1133.
- Kübler, J., Höck, K.-H., Sticht, J. and Williams, A. R. (1988). Density
  functional theory of non-collinear magnetism. *J. Phys. F: Met. Phys.* 18,
  469.
- Perdew, J. P. and Zunger, A. (1981). Self-interaction correction to
  density-functional approximations for many-electron systems. *Phys. Rev. B*
  23, 5048.
- Perdew, J. P., Burke, K. and Ernzerhof, M. (1996). Generalized gradient
  approximation made simple. *Phys. Rev. Lett.* 77, 3865.
- Furness, J. W., Kaplan, A. D., Ning, J., Perdew, J. P. and Sun, J. (2020).
  Accurate and numerically efficient r$^2$SCAN meta-generalized gradient
  approximation. *J. Phys. Chem. Lett.* 11, 8208.
- Heyd, J., Scuseria, G. E. and Ernzerhof, M. (2003). Hybrid functionals
  based on a screened Coulomb potential. *J. Chem. Phys.* 118, 8207.
- Krukau, A. V., Vydrov, O. A., Izmaylov, A. F. and Scuseria, G. E. (2006).
  Influence of the exchange screening parameter on the performance of
  screened hybrid functionals. *J. Chem. Phys.* 125, 224106.
- Vydrov, O. A. and Van Voorhis, T. (2010). Nonlocal van der Waals density
  functional: The simpler the better. *J. Chem. Phys.* 133, 244103.
- Sabatini, R., Gorni, T. and de Gironcoli, S. (2013). Nonlocal van der Waals
  density functional made simple and efficient. *Phys. Rev. B* 87, 041108.
- Roman-Perez, G. and Soler, J. M. (2009). Efficient implementation of a van
  der Waals density functional: application to double-wall carbon nanotubes.
  *Phys. Rev. Lett.* 103, 096102.
- Ning, J., Kothakonda, M., Furness, J. W., Kaplan, A. D., Ehlert, S.,
  Brandenburg, J. G., Perdew, J. P. and Sun, J. (2022). Workhorse minimally
  empirical dispersion-corrected density functional with tests for weakly
  bound systems: r$^2$SCAN+rVV10. *Phys. Rev. B* 106, 075422.
- Paier, J., Hirschl, R., Marsman, M. and Kresse, G. (2005). The
  Perdew-Burke-Ernzerhof exchange-correlation functional applied to the
  G2-1 test set using a plane-wave basis set. *J. Chem. Phys.* 122, 234102.
- Vaugier, L., Jiang, H. and Biermann, S. (2012). Hubbard U and Hund
  exchange J in transition metal oxides: Screening versus localization
  trends from constrained random phase approximation. *Phys. Rev. B* 86,
  165105.
- Caldeweyher, E. *et al.* (2019). A generally applicable atomic-charge
  dependent London dispersion correction. *J. Chem. Phys.* 150, 154122.
- Jepsen, O. and Andersen, O. K. (1971). The electronic structure of h.c.p.
  ytterbium. *Solid State Commun.* 9, 1763.
- Lehmann, G. and Taut, M. (1972). On the numerical calculation of the
  density of states and related properties. *Phys. Status Solidi B* 54, 469.
- Bloechl, P. E., Jepsen, O. and Andersen, O. K. (1994). Improved tetrahedron
  method for Brillouin-zone integrations. *Phys. Rev. B* 49, 16223.
