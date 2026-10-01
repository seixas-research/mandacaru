# Kohn-Sham density functional theory

`method="dft"` solves the closed-shell Kohn-Sham equations (Kohn and Sham 1965)
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
[the density-of-states example](../../../examples/new/04_DFT_H2O_dos.py).

## Reading the run log

A run writes the shared `[SYSTEM]`, `[BASIS]` and `[ELECTRONS]` blocks and then
two blocks of its own, read back by `parse_output` as `"setup"` and
`"summary"` (see [Reading a run](run_output.md)):

- `[SCF SETUP]` names the functional (`xc_functional`), the dispersion
  correction (`dispersion`, `NONE` when there is none) and the convergence
  criteria: at most 200 iterations, converged when both the energy change and
  the largest density-matrix change are below $10^{-8}$ Hartree, with DIIS and
  a level shift that is on only while the energy rises by more than
  $10^{-6}$ Hartree.
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
  `{"size": ..., "gamma": True}`). The mesh is reduced by time reversal
  ($\mathbf k$ and $-\mathbf k$ give the same density).
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
- **Checks you can rely on.** A k-point mesh reproduces the corresponding
  supercell at Gamma exactly (diamond Si, 2x2x2 mesh against the 16-atom
  supercell: equal to 1e-8 Ha per cell).

Not yet available for crystals: forces and stress, D4, symmetry (IBZ)
reduction beyond time reversal -- every k-point of the time-reversed mesh is
solved -- spin polarization, and band structures or densities of states.

## What is not implemented

The following are refused or unavailable:

- **Open shells and spin polarization**: `n_alpha != n_beta` is refused.
- **Spin-orbit coupling** in the Kohn-Sham operator.
- **Crystals with an all-electron or ONCVPSP basis**: the periodic path is
  PAW-LCAO only.
- **D4 with LDA**, and **D4 or forces for a crystal**, as above.

Not implemented yet: symmetry-reduced k-point meshes, band structures along a
path, projected densities of states and fat bands, the Fermi surface, the
dielectric constant and hybrid functionals.

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
