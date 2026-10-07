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
                       dispersion="d4",           # None (default) | "d4"
                       basis={"name": "PAW-LCAO", "size": "DZP"},
                       h=0.2)
energy_ev = atoms.get_potential_energy()
```

The same calculation from the command line:

```bash
mandacaru H2O --method dft --xc pbe --dispersion d4 --cell 10
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
| `"r2scan-rvv10"` | r2SCAN plus rVV10 nonlocal correlation: dispersion from the density (molecules and crystals, forces and stress) | Vydrov and Van Voorhis 2010; Sabatini *et al.* 2013; Ning *et al.* 2022 |

PBE and r2SCAN use the Perdew-Wang 1992 parameterization of the uniform-gas
correlation energy. The exchange-correlation energy and its potential are
evaluated on the grid from the density, with spectral gradients; r2SCAN also
builds the kinetic-energy density
$\tau = \tfrac12\sum_{pq}D_{pq}\nabla\phi_p\cdot\nabla\phi_q^*$ from the
gradients of the orbitals and feeds its derivative back into the Kohn-Sham
operator.

- **LDA** is the default and the cheapest. It is also the functional the
  PAW-LCAO datasets were generated with, so it is the one with no mismatch
  between the molecule and its datasets.
- **PBE** adds the density gradient. Choose it when a gradient correction
  matters more than the dataset consistency discussed below.
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
- **Shifts against references** (DZP, h = 0.2 Angstrom, 3 Angstrom of
  vacuum; HSE06 - PBE): H$_2$O HOMO -1.74 eV (PAW-LCAO) against -1.45 eV for an
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

The density gradients PBE and r2SCAN need are spectral derivatives built
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
a real field never had, and nothing changes.

### r2SCAN+rVV10: dispersion from the density

`xc="r2scan-rvv10"` adds the rVV10 nonlocal correlation to r2SCAN, so the
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
argon dimer); $\beta = (3/b^2)^{3/4}/32$ makes the term vanish for the
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
- **Argon dimer** (TZP, unconfined basis, h = 0.2 Angstrom; reference
  CCSD(T) $-12.3$ meV at 3.76 Angstrom): $-11.6$ meV at 3.76, $-10.5$ at
  4.0, $-5.5$ at 4.5, $-3.3$ at 5.0 and $-0.6$ meV at 6.0 Angstrom;
  r2SCAN alone gives $-5.8$ meV at the minimum.
- **Use an unconfined basis for weak binding.** The default confinement
  (`energy_shift` 0.1 eV) makes the isolated fragments' orbitals too compact,
  and each fragment then borrows its neighbor's functions: argon dimer
  binding with PBE at 3.76 Angstrom is $-18.2$ meV confined and $-4.3$ meV
  unconfined, at any basis size. Pass `basis={"name": "PAW-LCAO",
  "energy_shift": None}` (or a much smaller shift) for interaction energies
  of this size, with any functional, and check them with the counterpoise
  correction (`interaction_energy(..., counterpoise=True)`, see
  {doc}`interaction_energy`).
- D4 is refused with it: the dispersion would be counted twice.

## D4 dispersion

`dispersion="d4"` adds the D4 correction (Caldeweyher *et al.* 2019), computed
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

D4 has parameters for PBE, r2SCAN and HSE06 and none for LDA, so `dispersion="d4"`
with `xc="lda"` is refused, as it is with `xc="r2scan-rvv10"`, which
carries its own dispersion. Without `dftd4` installed the calculation stops
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
unscreened. The shipped datasets are **LDA** datasets. `xc="pbe"` or
`xc="r2scan"` on them therefore mixes two functionals, the molecule's and the
dataset's, and the calculation emits a `RuntimeWarning` saying so. The numbers
are still useful for trends, but they are not the answer of a self-consistent
PBE or r2SCAN dataset. A PBE dataset is future work.

## The grid

Converge the grid before trusting a difference of functionals. A spacing of
`h` between 0.15 and 0.2 Å is a sensible range for H$_2$O and similar light
molecules. Three points to keep in mind:

- An absolute total energy on a real-space grid drifts with `h`: for H$_2$O it
  moves by about 0.3 Ha between `h = 0.25` and `0.12` Å, for Hartree-Fock and
  DFT alike, while the difference between the two stays constant to about
  8 mHa. Compare energies at one fixed `h`.
- The kinetic-energy operator is a finite-difference Laplacian by default.
  `kinetic="spectral"` evaluates it in reciprocal space instead and reduces the
  error for a hard smooth wave: for the He atom the missing energy at
  `h = 0.20` Å falls from about 108 mHa to about 9 mHa. It is shared by every
  method, not only DFT.
- The noble-gas atoms Ne and Ar in a PAW-LCAO basis need a spacing of about
  0.06 Å or smaller, well below the range above.

## The Hartree term

For a molecule, the Hartree matrix $J[D]$ is by default the potential of
the density from one Poisson solve per SCF iteration (`hartree="poisson"`),
with the same spectral isolated Coulomb kernel the two-body tensor is built
with. On a PAW-LCAO basis the compensation charges, every multipole the
datasets carry, enter as their low-rank matrices, exactly as they enter the
tensor. A semilocal Kohn-Sham run therefore never forms the $M^4$ two-body
tensor; it is built only when something reads it: the forces, a hybrid's
exact exchange (its short-range tensor), or the export to a quantum method.
The reference energy of the exported Hamiltonian needs only the occupied
orbitals' integrals, which come from their own pair solves.

`hartree="tensor"` contracts the two-body tensor instead. The two routes
give the same Fock matrix, energy, reference energy and forces to round-off
(about $10^{-13}$ Ha), so the option exists to cross-check, not to choose an
accuracy. The Poisson route is faster where the tensor dominates: on
ethylene in PAW-LCAO DZP (46 functions, h = 0.2 Angstrom) the energy took
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
| `dispersion_energy` | The D4 energy, in Hartree (zero without `dispersion`) |

The HOMO-LUMO gap is a difference of Kohn-Sham eigenvalues, an orbital-energy
gap, not an excitation energy or a quasiparticle gap.

`result.reference_energy` is **not** the Kohn-Sham energy. It is the energy of
the Kohn-Sham determinant inside the exported many-body Hamiltonian, which has
neither the dispersion term nor the core-correction offset.
`result.optimal_energy` is the Kohn-Sham total.

On H$_2$O in a PAW-LCAO DZP basis at $h = 0.2$ Å with 3 Å of vacuum, the three
functionals of the first example give these totals and gaps (the setup of
[the functionals example](../../../examples/new/03_DFT_H2O_functionals.py)):

| Functional | Total energy (Ha) | HOMO-LUMO gap (eV) |
| :--- | ---: | ---: |
| LDA | -17.560555 | 7.82 |
| PBE + D4 | -17.647979 | 7.80 |
| r2SCAN + D4 | -17.638390 | 9.18 |

The D4 term is $-1.97\times10^{-4}$ Ha for PBE. The PBE and r2SCAN rows carry
the dataset warning described above.

The Kohn-Sham eigenvalues also give the molecular density of states: the
levels, each broadened by a Gaussian, of one isolated molecule. This is a
picture of its orbital energies at the Gamma point, not a band structure; see
the "Bands and densities of states" section below and
[the density-of-states example](../../../examples/new/04_DFT_H2O_dos.py).

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
Li +0.24 (Hirshfeld), +0.66 (Voronoi) and +0.84 (Bader). Two notes:

- **Bader converges quickly with the grid**: the basins are traced by a
  continuous ascent on an interpolation of the density (see
  [Charges](charges.md)), so rocksalt LiH's Li is +0.843 / +0.847 / +0.847
  at 10 / 12 / 16 nodes per lattice vector and moves by ~1e-3 e when the
  atoms are shifted against the grid; NaCl's Na is +0.86 and MgO's Mg +1.59
  (SZ, LDA).
- **Use a real k-mesh.** A Gamma-only crystal's density is not converged:
  LiH's Bader charge is +0.63 at Gamma, +0.79 at 3x3x3 and 6x6x6.

Equivalent atoms come
out equal only on a grid that keeps the operations relating them: on a grid
whose nodes miss diamond's quarter translations, the two silicon atoms of the
discretized crystal are not equivalent, and a Voronoi split shows it. Cube
files, dipoles and natural orbitals of a crystal are not available yet.

## Reading the run log

A run writes the shared `[SYSTEM]`, `[BASIS]` and `[ELECTRONS]` blocks and then
three blocks of its own, read back by `parse_output` as `"setup"`,
`"scf_iterations"` and `"summary"` (see [Reading a run](run_output.md)):

- `[SCF SETUP]` names the functional (`xc_functional`), the dispersion
  correction (`dispersion`, `NONE` when there is none), how the Hartree term
  was built (`hartree`, see *The Hartree term* above) and the
  convergence
  criteria: at most 200 iterations, converged when both the energy change and
  the largest density-matrix change are below $10^{-8}$ Hartree, with DIIS and
  a level shift that is on only while the energy rises by more than
  $10^{-6}$ Hartree.
- `[SCF ITERATIONS]` has one row per iteration: the time, the energy (the
  free energy of a crystal), its change, the residual the convergence test
  measures and, for a spin-polarized crystal, the moment per cell.
- `[SCF SUMMARY]` gives the energy and its decomposition: the total, the
  reference energy of the exported Hamiltonian, the Hartree and
  exchange-correlation energies, the core-correction energy when the basis has
  a core correction and the dispersion energy when D4 ran.

The run also writes a `references.bib`: the Hohenberg-Kohn and Kohn-Sham
papers, the functional that ran and, with `dispersion="d4"`, the D4 paper.

## From Kohn-Sham to ADAPT-VQE

The converged orbitals are exported as the Hartree-Fock ones are: the result
carries the many-body Hamiltonian, with the bare Coulomb interaction, written
in the Kohn-Sham orbitals, and its reference state is the Kohn-Sham
determinant.

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

atoms.calc = Mandacaru(method="dft", xc="pbe", dispersion="d4", h=0.2,
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
h = 0.25 Angstrom).

r2SCAN forces are analytic too, but its energy surface on a finite real-space
grid is rough at the micro-Hartree level -- the kinetic-energy density enters
through a ratio that is ill-conditioned in the density tails -- so its
finite-difference check is limited to about 0.02 eV/Angstrom, and a relaxation
with `fmax` below that is not meaningful yet.

## Crystals: Bloch states, k-points and smearing

A periodic geometry -- any `atoms.pbc` set -- is solved as a crystal: Bloch
states on a Monkhorst-Pack mesh, with fractional occupations from a smearing
function, so metals and insulators are handled alike.

```python
from ase.build import bulk
from mandacaru import Mandacaru

al = bulk("Al", "fcc", a=4.05)
al.calc = Mandacaru(method="dft", xc="lda", h=0.25,
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
  [The aluminum smearing example](../../../examples/new/09_DFT_Al_smearing.py)
  sweeps the three methods over four widths and plots both quantities.
- **Electrostatics without finite-grid effects.** The smooth density, the
  compensation charges and the ions are combined into one neutral charge whose
  Coulomb energy is summed in reciprocal space; the compact charges are
  Fourier-filtered where they meet the grid and summed exactly among
  themselves. The energy is then nearly independent of the grid: H2 in an
  8 Angstrom cell changes by 0.08 mHa between h = 0.25 and 0.18 Angstrom and
  reproduces the converged molecular limit, which the molecular solver itself
  reaches only by extrapolating from h <= 0.12 Angstrom.
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
  (it refuses `kinetic="fd"`), so compare it with a molecule run with
  `kinetic="spectral"`: Ne with a 150 eV basis filter agrees to 1 uHa, and
  water, CO and HF in a 9 Angstrom box (SZP, h = 0.3 Angstrom) agree on the
  level spacings to 0.4 mHa, on the energy to 0.7 mHa and on the Hirshfeld
  charges to 0.001 e. The molecular default, the finite-difference stencil,
  differs from it by far more than that on these grids: 0.80 Ha for water,
  with a dipole 1.6 times larger.

Not yet available for crystals: D4, cube files and dipoles.

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
  k-mesh. For silicon (PAW-LCAO DZP, h = 0.3 Angstrom) the HSE06 - PBE gap
  shift is +0.66 and +0.58 eV on 3x3x3 and 4x4x4 meshes (a 2x2x2 mesh
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
o2.calc = Mandacaru(method="dft", xc="pbe", h=0.2,
                    basis={"name": "PAW-LCAO", "size": "DZP"})
o2.get_potential_energy()                 # the triplet: moment 2.000

fe = bulk("Fe", "bcc", a=2.87)
fe.set_initial_magnetic_moments([2.5])
fe.calc = Mandacaru(method="dft", xc="lda", h=0.2,
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
- **Checked**: O2's triplet is 1.25 eV (LDA) / 1.34 eV (PBE) below the
  closed-shell singlet (SZ); O2 in a periodic box reproduces the molecular
  unrestricted run (moment 2.000, 1.000 per atom, pi* exchange splitting
  1.555 vs 1.558 eV).

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
si.calc = Mandacaru(method="dft", xc="lda", h=0.25,
                    basis={"name": "PAW-LCAO", "size": "DZP", "filter": 300},
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
  1e-2 for the gradient functionals at h = 0.25 Angstrom).
- **The stress** (eV/Angstrom^3, ASE's sign: negative for a compressed cell)
  strains the cell, the atoms and the grid together, with the converged
  states carried along -- no new SCF. Only the strains the space group
  leaves unchanged are applied, since the stress lies in their span: two
  rebuilds of a cubic crystal, twelve without symmetry. The
  basis filter is held at the cutoff the run used, so the stress never
  includes the basis changing with the grid spacing (with the default
  grid-tied filter, a relaxation's *next* geometry will rebuild the basis
  at its own spacing: give `filter` in eV to keep one basis throughout).
- **Fixed node counts.** The grid follows the cell during a relaxation with
  the node counts set by `h`; when a count changes, the energy steps. Choose
  `h` so the counts stay put over the range the cell explores (the check
  below kept 15 nodes from a = 5.50 to 5.80 Angstrom with h = 0.265).
- **Checked**: Si (SZ, 2x2x2) relaxed from a = 5.50 Angstrom with a distorted
  basis to a = 5.7179 Angstrom in 9 BFGS steps, where an E(a) scan on the
  same settings has its minimum at 5.7180 (an SZ number, not silicon's
  lattice constant).
- **Cost**: both run on the SCF's irreducible k-points and are symmetrized
  (the force by the space group with its atom map); they equal the full
  mesh's to round-off. Diamond Si, DZP, 4x4x4, h = 0.2 Angstrom: forces
  29 s, stress 7.5 s on 8 cores.

## Bands and densities of states

After `atoms.get_potential_energy()`, the converged potential can be reused
without another SCF. Every method below is reached as `atoms.calc.<name>`.

```python
from ase.build import bulk
from mandacaru import Mandacaru

si = bulk("Si", "diamond", a=5.43)
si.calc = Mandacaru(method="dft", xc="lda", h=0.25,
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
how much. See [the silicon bands example](../../../examples/new/08_DFT_Si_bands.py)
and, for a molecule, [the density-of-states
example](../../../examples/new/04_DFT_H2O_dos.py).

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
  net part (0.12 eV/Å for AlN at $h$ = 0.25 Å). Left in, that net force
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

Checks (LDA, PAW-LCAO SZP, h = 0.25 Å, a 4³ or 4×4×3 mesh):

| | Mandacaru | literature |
| :--- | ---: | ---: |
| $Z^*$, rocksalt LiH | Li +1.055, H −1.055 | about 1.0–1.1 |
| $Z^*$, rocksalt NaCl | Na +1.165, Cl −1.165 | about 1.1 |
| $Z^*_{zz}$, wurtzite AlN | Al +2.662, N −2.672 | 2.65–2.70 |
| spontaneous $P$ of AlN (relative to zinc blende) | −0.076 C/m² | −0.08 to −0.09 |

The acoustic sum rule holds to 0.01 e, and the off-diagonal elements of
the cubic crystals are zero ($10^{-8}$). The spontaneous polarization is the
difference of the formal polarizations of wurtzite and of zinc blende in a
six-atom hexagonal cell with the same in-plane lattice, so both have the same
quantum along $c$, $e/A$. A centrosymmetric crystal comes out on 0 or half a
quantum, and a rigid shift of all atoms changes $\mathbf P$ only by whole
quanta.

For AlN's clamped-ion tensor this basis gives $e_{33} = -0.56$ and
$e_{31} = +0.27$ C/m², against about $-0.47$ for $e_{33}$ in the
literature; the components symmetry forbids come out at $10^{-8}$. The
relaxed-ion tensor is **not yet validated**, and the reason is the lattice,
not the method. On the experimental lattice this basis gives
$e_{33} = +2.63$ and $e_{31} = -1.29$ C/m², against the literature's
1.46 and −0.60. The internal strain there, $du/d\varepsilon_3 = -0.31$,
is the true minimum of this model's energy. The relaxation lands on it, and
$e_{33} - e_{33}^{(0)}$ equals $4eZ^*/(\sqrt3 a^2)\,du/d\varepsilon_3$
with the computed $Z^*$. But in this model the experimental lattice is
about 100 GPa from equilibrium: the energy minimum lies 11 % higher in $a$
and $c$, because bonds to aluminum come out far too long (fcc Al at
4.68 Å, against about 4.0 Å in LDA). That compression stiffens the
internal mode and inflates the coupling. On the model's own lattice,
$du/d\varepsilon_3 = -0.20$ (literature −0.18), but the 40 % larger
volume dilutes $e_{33}$ to 0.99. Compare relaxed-ion tensors only for
crystals whose lattice this basis and dataset reproduce.

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
from below with the mesh: 1.69, 1.98 and 2.12 Å² on 4³, 6³ and 8³ meshes;
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
- **The harmonics.** They are the real combinations of the complex
  spherical harmonics of the basis (Condon–Shortley phase), positive along
  their Cartesian lobes: $S_{1,1}\propto x$, $S_{1,-1}\propto y$,
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
| single zeta (SZP) | 9 | 35 (11) | 7.5 (3.6) | |
| double zeta (DZP) | 15 | | 53 (23) | 35 (9) |

With the single-zeta basis the nine functions span every band, so the 8³
mesh is already within 10 meV. With the double-zeta basis the six extra
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

Silicon's valence and four lowest conduction bands give eight sp³
hybrids, 0.41 Å from each atom along its bonds. $\Omega_I$ is the same to
1e-9 Å² from random s Gaussians. The total spread is not: the hybrids are a
local minimum of the gauge-dependent part, and another start can find a
lower one. Inside the frozen window, the interpolated bands equal the
non-self-consistent bands exactly on the mesh. Between mesh points they
are within 317, 126, 19 and 4.1 meV (root mean square 100, 38, 3.9 and
0.8 meV) on 3³, 4³, 6³ and 8³ meshes.

### Interpolation

`hamiltonian(R)` and `interpolate` place each pair's hopping on a lattice
vector of the Born-von Karman supercell; the bands are exact on the mesh
either way. `replicas="centers"`, the default, puts it on the replica
nearest in the distance between the two functions' centers.
`replicas="cells"` uses the Wigner-Seitz vectors between cell origins.
Neither is better everywhere. For silicon's sp³ hybrids, nearest centers
lower the root-mean-square error from 157 to 100 meV on 3³, from 5.8 to
3.9 meV on 6³ and from 1.4 to 0.8 meV on 8³. Silicon's valence bond
functions hop far along the [110] chains, and there cell origins do
better: 20 against 26 meV root mean square on 6³, 7.6 against 10.0 meV on
8³. Either way the error falls with the mesh, so interpolation needs a
dense mesh.

### The functions on the grid

`w.orbital(n)` returns function `n` on the supercell grid with the box
centered on it: its smooth part, in Bohr^-3/2. The global phase is chosen
to make the function as real as possible. `w.write(n, path, part="real")`
writes it as `.cube` or `.xsf`, with the supercell's atoms; `part` can
also be `"imaginary"` or `"modulus"`. These are one-particle orbitals. In
a centrosymmetric crystal they are real up to a phase:
`w.imaginary_ratio(n)` is about 3e-9 for silicon's functions.

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
two-electron model: on a 4³ mesh, hopping −4.73 eV, $U = 12.19$ eV and
inter-site $V = 8.34$ eV, all bare. ADAPT-VQE reaches the exact ground
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

Silicon's bond (LDA, PAW-LCAO SZP): the bare $U = 12.27$, $V = 8.37$ and
$J = 0.35$ eV become 4.45, 2.72 and 0.225 eV on 3³ and 3.97, 2.33 and
0.224 eV on 6³. $U$ still falls by about 0.13 eV per mesh step: the
long-wavelength dielectric constant converges slowly with the mesh (the
head gives $\epsilon_M$ of 22, 16, 13 and 11.5 on 3³ to 6³). $J$, an
exchange density without charge, is screened least and does not move with
the mesh. Full RPA lies 0.13–0.15 eV below cRPA. Excluding the whole
eight-band sp³ set instead of the bond's subspace leaves $U = 8.31$ eV
(3³): most of silicon's screening is valence-to-conduction, inside that
set.
The SZP basis has 18 bands: keeping 8 or 12 of them gives $U = 4.95$ or
4.56 eV against 4.45 eV, so the empty states are not converged. The
cutoff is converged (2, 3 and 4 Bohr⁻¹ agree to 13 meV).

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
| 3³ | 5.76 | 7 meV | 31 meV | 31 meV |
| 4³ | 7.68 | 0.8 meV | 2.0 meV | 1.9 meV |
| 5³ | 9.60 | 0.2 meV | 0.4 meV | |
| 6³ | 11.52 | 0.1 meV | | |

A fragment within the radius stayed under 10 meV; one past it reached
31 meV. The bare values converge with the mesh: $U$ 12.27, 12.19, 12.21
and 12.15 eV; $t$ −4.87, −4.73, −4.72 and −4.69 eV on 3³ to 6³. In a
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

- **Why windows.** The single-zeta basis puts a V p band inside the t2g
  energy range: at R it lies 0.96 eV above the Fermi level, below the t2g
  states at 1.71 eV. On a mesh containing R, bands 12–14 are not the t2g
  set and their projection on the t2g trials is singular there. The
  disentanglement keeps the states below $E_F$ + 0.5 eV and takes the
  smoothest t2g-like subspace from the rest of the window, as cRPA studies
  do (Vaugier, Jiang and Biermann used a window of ±1.8 eV).
- **What it gives.** Three equivalent functions centered on V. In
  physicists' notation, $U = \langle aa|aa\rangle$,
  $U' = \langle ab|ab\rangle$ and $J = \langle ab|ba\rangle$, each the
  same on every orbital or pair. $J$ equals the pair-hopping integral
  $\langle aa|bb\rangle$, because the functions are real. The model
  holds one electron.

SrVO3 (LDA, PAW-LCAO SZP, h 0.2 Å, the shipped library):

| mesh | spread (Å²) | U bare | U′ bare | J bare | U cRPA | U′ cRPA | J cRPA |
|---|---|---|---|---|---|---|---|
| 3³ | 1.60 | 15.70 | 14.65 | 0.493 | 5.25 | 4.30 | 0.447 |
| 4³ | 1.90 | 15.50 | 14.46 | 0.484 | 5.23 | 4.29 | 0.440 |
| 5³ | 2.67 | 14.69 | 13.71 | 0.453 | | | |

(eV.) The three orbitals agree to 1e-4 eV in $U$ and $U'$, and to 0.4 %
in $J$, which the cell grid's discretization sets. $(U - U')/2$ is
0.53 eV bare and 0.48 eV screened, 7 % above $J$: the relation
$U' = U - 2J$ is exact only for a spherical atom. The interpolated bands
inside the frozen window are within 184, 19, 12 and 24 meV of the true
ones on 3³ to 6³ meshes (root mean square 77, 8.7, 4.6 and 3.6 meV).

Vaugier, Jiang and Biermann's cRPA for the same three-band model gives
$U$ = 3.2 eV, $J$ = 0.44 eV and $U' = U - 2J$ = 2.3 eV, with a bare
$V$ = 16.1 eV and $J_{bare}$ = 0.55 eV. The bare values here are close,
and so is $J$. The screened $U$ is about 2 eV too large: here cRPA keeps a
third of the bare $U$, against a fifth in their calculation. The cause is
the screening, not the mesh, since 3³ and 4³ differ by 0.03 eV. Other
settings move it too little to close the gap:

- **The empty states.** Keeping 28 of the 40 bands raises $U$ by
  0.03 eV; a double-zeta basis (59 bands) lowers it by 0.13 eV.
- **The cutoff.** The default 3 Bohr⁻¹ is converged: 4 Bohr⁻¹ lowers $U$
  by 6 meV, while 2 Bohr⁻¹ raises it by 0.29 eV.

The shipped Sr dataset has no semicore 4s and 4p, whose screening is
missing. These numbers will move when the rebuilt datasets, with Sr
semicore, are installed.

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

- **Spin-orbit coupling** in the Kohn-Sham operator.
- **Crystals with an all-electron basis**: the periodic path is
  PAW-LCAO or UPAW-LCAO only.
- **D4 with LDA or with r2SCAN+rVV10**, and **D4 for a crystal**, as above.

A molecule's polarizability is in {doc}`polarizability`; a crystal's
electronic dielectric tensor is above.

## References

- Hohenberg, P. and Kohn, W. (1964). Inhomogeneous electron gas. *Phys. Rev.*
  136, B864.
- Kohn, W. and Sham, L. J. (1965). Self-consistent equations including exchange
  and correlation effects. *Phys. Rev.* 140, A1133.
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
