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
                       xc="pbe",                  # "lda" (default) | "pbe" | "r2scan"
                       dispersion="d4",           # None (default) | "d4"
                       basis={"name": "PAW-LCAO", "size": "DZP"},
                       h=0.2)
energy_ev = atoms.get_potential_energy()
```

The same calculation from the command line:

```bash
mandacaru H2O --method dft --xc pbe --dispersion d4 --cell 10
```

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

The density gradients PBE and r2SCAN need are spectral derivatives built
from the reciprocal vectors of the cell, so a skewed cell -- hexagonal,
monoclinic, triclinic -- is handled as exactly as an orthogonal one.

## D4 dispersion

`dispersion="d4"` adds the D4 correction (Caldeweyher *et al.* 2019), computed
for the functional that ran, with its gradient in the forces. It is taken
from the external `dftd4` package, an optional dependency:

```bash
pip install 'mandacaru[dispersion]'
```

D4 has parameters for PBE and r2SCAN and none for LDA, so `dispersion="d4"`
with `xc="lda"` is refused. Without `dftd4` installed the calculation stops
with an error naming the extra.

## Pseudopotential bases and their functional

The all-electron bases (`"HAO"`, `"NAO"`, the Gaussian families) and the
pseudopotential families (`"ONCVPSP"`, `"PAW-LCAO"`) all work, with every size
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
centers), so the basins are found on the valence density **plus the free
atoms' frozen cores**, and only the valence density is integrated. Every
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
  correction (`dispersion`, `NONE` when there is none) and the convergence
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
  forgiving choice: for Al on 4x4x4, it stays within 3 meV of a 0.05 eV
  Fermi-Dirac result at 0.2 eV, where Fermi-Dirac itself drifts by 28 meV.
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

Not yet available for crystals: D4, cube files and dipoles, and the forces
and stress of a spin-polarized crystal.

## Spin polarization

A molecule with `n_alpha != n_beta` -- an odd electron count, or initial
magnetic moments on the atoms (`atoms.set_initial_magnetic_moments`) -- is
solved **spin-unrestricted**: one determinant per spin, with the
spin-polarized LDA, PBE or r2SCAN. A crystal whose atoms carry nonzero
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

## What is not implemented

The following are refused or unavailable:

- **Spin-orbit coupling** in the Kohn-Sham operator.
- **Crystals with an all-electron or ONCVPSP basis**: the periodic path is
  PAW-LCAO only.
- **D4 with LDA**, and **D4 for a crystal**, as above.

Not implemented yet: the Fermi surface, the dielectric constant and hybrid
functionals.

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
- Caldeweyher, E. *et al.* (2019). A generally applicable atomic-charge
  dependent London dispersion correction. *J. Chem. Phys.* 150, 154122.
