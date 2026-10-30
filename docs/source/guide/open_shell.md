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
