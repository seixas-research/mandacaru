# Electric fields and the polarizability

A molecule can be put in a uniform static electric field, and its dipole
polarizability follows from a few such calculations.

## A molecule in a field

`electric_field=(Fx, Fy, Fz)` adds the field to any molecular calculation,
in atomic units (Hartree per e Bohr; 1 a.u. is 51.42 V/Å):

```python
from ase.build import molecule
from mandacaru import Mandacaru

water = molecule("H2O")
water.center(vacuum=4.0)
water.calc = Mandacaru(method="dft", xc="pbe",
                       basis={"name": "PAW-LCAO", "size": "TZP"},
                       electric_field=(0.0, 0.0, 0.002))
water.get_potential_energy()
water.calc.get_dipole_moment()
```

The potential is $V = -\mathbf F\cdot\boldsymbol\mu$: the electrons feel
$+\mathbf F\cdot\mathbf r$ in the one-body Hamiltonian, and the nuclei (the
valence ions, with a pseudopotential basis) add the constant
$-\mathbf F\cdot\sum_A Z_A\mathbf R_A$. With a PAW-LCAO basis the
electrons' position operator includes the augmentation, so the field couples
to exactly the dipole that `get_dipole_moment()` reports, and the
Hellmann–Feynman relation $\partial E/\partial\mathbf F = -\boldsymbol\mu$
holds for Hartree–Fock and Kohn–Sham states (to $10^{-6}$ a.u. in the test
suite). Every method sees the field, the variational ones included: the
field is part of the integrals the Hamiltonian is built from.

What is refused:

- **Crystals.** A uniform field breaks the periodicity, and the position
  operator is not defined in a periodic cell. A crystal's response needs the
  Berry-phase polarization.
- **Forces in a field.** The field pulls on the nuclei and moves the dipole
  matrices with the basis, and the gradient does not include those terms
  yet.
- **A loaded or user-built Hamiltonian** (`load_hamiltonian=`,
  `hamiltonian_builder=`): neither is built from the integrals the field
  acts on.

The run log names the field once, in the `[SYSTEM]` block.

## The polarizability

{func}`~mandacaru.algorithms.polarizability.polarizability` runs seven
calculations: no field, and $\pm F$ along each axis. All seven share one
grid. It then differentiates the dipole by central differences:

$$
\alpha_{ij} = \frac{\mu_i(F\hat e_j) - \mu_i(-F\hat e_j)}{2F}.
$$

```python
from mandacaru.algorithms import polarizability

result = polarizability(water, method="dft", xc="pbe",
                        basis={"name": "PAW-LCAO", "size": "TZP",
                               "energy_shift": None},
                        h=0.2)
result.tensor              # 3x3, Bohr^3 (atomic units)
result.isotropic           # Tr(alpha)/3
result.anisotropy
result.in_units("angstrom^3")
result.energy_diagonal     # -d2E/dF2 per axis: the independent check
```

`atoms.calc.polarizability(atoms)` does the same with a calculator's method,
basis and options. The default field, `2e-3` a.u., keeps the
finite-difference error, which is set by the second hyperpolarizability,
well below a percent for a small molecule. `energy_diagonal` is
$-[E(F)+E(-F)-2E_0]/F^2$. For a variational method it agrees with the
diagonal of `tensor`, because the field couples through the same matrices
the dipole is computed from. A correlated method's dipole is an expectation
value that lacks the orbital response, so the two can differ there.

## The basis decides the answer

A polarizability measures how the electron density stretches, and the
stretch lives in the outer, diffuse part of the basis. A compact basis
cannot hold it. The confined default PAW-LCAO basis (`energy_shift` 0.1 eV)
and small atom-centered sets underestimate $\alpha$ for that reason. Use an
unconfined basis (`"energy_shift": None`) and check convergence with its
size.

Water (PAW-LCAO, h = 0.2 Å, 5 Å of vacuum), isotropic $\alpha$ in a.u.
($\alpha_{xx}, \alpha_{yy}, \alpha_{zz}$ in parentheses):

| basis | PBE | Hartree–Fock |
| :--- | ---: | ---: |
| DZP, confined (the default) | 5.59 (3.76, 7.43, 5.59) | |
| TZP, confined | 5.78 (3.96, 7.59, 5.79) | |
| DZP, unconfined | 8.41 (7.63, 9.51, 8.09) | |
| TZP, unconfined | 8.73 (8.16, 9.65, 8.39) | 6.62 (5.76, 7.74, 6.35) |

The confinement costs a third of the polarizability, and the
out-of-plane component $\alpha_{xx}$ half of it. Even unconfined, an
atom-centered basis without diffuse functions stays below the basis-set
limit. In every run the dipole derivative and the energy curvature agree
to $10^{-3}$ a.u., and the off-diagonal elements vanish ($10^{-10}$), as
water's symmetry requires.

For comparison: Hartree–Fock at the basis-set limit gives about 8.5 a.u.,
PBE about 10.4 a.u., and experiment 9.64 a.u. (1.43 Å³).
