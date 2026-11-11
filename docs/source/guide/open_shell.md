(open-shell)=
# Open-shell systems: radicals, ions and high-spin states

Any system with an odd number of electrons (a radical, a cation such as H₂⁺, a
single hydrogen atom) and any even-electron system asked for a high-spin state
runs through the same calculator as a closed-shell molecule. Two things decide
the state: the electron count and the spin multiplicity.

## How the spin state is chosen

The number of electrons follows from the nuclei, `charge=` (an integer) and,
when you need it directly, `n_electrons=`. The spin follows the ASE convention:
the **initial magnetic moments** of the atoms, whose total is the number of
unpaired electrons, give the occupations `(n_alpha, n_beta)`.

```python
from ase.build import molecule
from mandacaru import Mandacaru

oh = molecule("OH")
oh.center(vacuum=3.0)
oh.set_initial_magnetic_moments([1.0, 0.0])      # 9 electrons: a doublet
oh.calc = Mandacaru(method="uhf", basis="HAO", grid_spacing=0.3)
oh.get_potential_energy()
print(oh.calc.result.num_particles)              # (n_alpha, n_beta)
```

For a triplet, set the total to two, for example
`o2.set_initial_magnetic_moments([1, 1])`; `examples/adapt_vqe/03_adapt_vqe_o2_triplet.py`
runs the O₂ triplet with ADAPT-VQE.

(noncollinear-moments)=
## Noncollinear initial moments

Moments given as one 3-vector per atom (Bohr magneton) start a noncollinear
calculation of a molecule with `method="dft"`. The steps are:

1. Collinear Kohn–Sham runs first, with the vectors' lengths as the moments.
2. Each atom's share of its magnetization is turned to that atom's vector.
3. Spinor (two-component) Kohn–Sham starts from that density. It includes the
   spin-orbit term of a spin-orbit dataset if there is one.

```python
from ase import Atoms
from mandacaru import Mandacaru

h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.06]], cell=[6, 6, 7])
h2.center()
h2.set_initial_magnetic_moments([[0.5, 0, 0], [0.5, 0, 0]])     # along x
h2.calc = Mandacaru(method="dft", charge=1, grid_spacing=0.4,
                    basis={"name": "PAW-LCAO", "size": "SZ"})
h2.get_potential_energy()
```

Without a spin-orbit term the collinear state is a stable fixed point of the
spinor SCF, so a noncollinear state is reached only from such a start. A common
direction gives the collinear energy, with the spin moment along it (H2+: the
same energy to 1e-12 Ha along z, x and a tilted direction). The analytic force
is the derivative of the spinor energy.

Spinor Kohn–Sham fills integer occupations. A nearly degenerate frontier,
which collinear Kohn–Sham shares fractionally
([Occupations](#dft-degenerate-shell)), is therefore not the same state.
Other methods raise `NotImplementedError` for moment vectors, and so does a
crystal. A spin-orbit crystal takes collinear moments as the start of its
noncollinear magnetization.

## Which method

- **`method="uhf"`** solves the unrestricted problem with separate spatial
  orbitals for the two spins. Use it for any open shell at the Hartree–Fock level.
- **`method="dft"`** switches to the unrestricted solver by itself whenever
  `n_alpha != n_beta`; `spin=True` (or `unrestricted=True`) asks for it on an
  even-electron closed shell too, for example to let a triplet start from
  different spin orbitals. For a Kohn–Sham run `atoms.get_magnetic_moment()`
  returns $N_\alpha - N_\beta$.
- **`rhf`** is the restricted closed-shell method.
- **Quantum methods** (`vqe`, `adapt-vqe`, …) first solve the mean field, restricted
  when the counts are equal and unrestricted otherwise, and build the
  second-quantized problem from it. For an unrestricted reference the starting
  determinant is the UHF determinant expressed in the shared set of model
  orbitals. That determinant need not be a stationary point of the many-electron
  energy in that basis, so single excitations can carry a non-zero gradient.
  After the run `calc.num_particles` reports the occupations of the quantum
  problem.

Open-shell MP2 active-space selection is not yet available
([open items](#not-yet-available)).

## Reading the result

`result.num_particles` is `(n_alpha, n_beta)`. `calc.natural_orbitals()` returns
the natural occupations and orbitals of the spin-summed density, and the
[visualization](visualization.md) page draws them. A cube file of the
`difference_density` shows what correlation does to the Hartree–Fock reference
of an open shell.
