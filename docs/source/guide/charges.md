(charges)=
# Charges and magnetic moments

A converged run knows the electron density; turning it into *per-atom* numbers
needs a boundary between atoms that the physics does not supply. This page lists
what the calculator reports today.

## Totals

The totals are integrals of the whole density, so no partition is involved:

```python
from ase import Atoms
from mandacaru import Mandacaru

cation = Atoms("H2", positions=[[2.5, 2.5, 2.2], [2.5, 2.5, 2.9]], cell=[5, 5, 5])
cation.set_initial_magnetic_moments([0.5, 0.5])    # one unpaired electron
cation.calc = Mandacaru(method="dft", xc="pbe", basis="HAO",
                        grid_spacing=0.35, charge=1)
cation.get_potential_energy()

print(cation.get_magnetic_moment())                # N_alpha - N_beta
print(cation.calc.result.num_particles)            # (n_alpha, n_beta)
```

- The total charge is the integer `charge=` of the calculator, and the electron
  count follows from it and from the nuclei (`n_electrons=` sets it directly).
- `result.num_particles` is `(n_alpha, n_beta)` for every method, and
  `result.n_spatial_orbitals` the size of the orbital space.
- For `method="dft"`, `atoms.get_magnetic_moment()` returns $N_\alpha - N_\beta$.
  The other methods report the spin through `num_particles`.

The spin state is chosen as described in [Open-shell systems](open_shell.md).

(charges-dipole)=
## The dipole moment of a molecule

The electric dipole moment is another total, so it needs no partition. After a
converged run on a molecule, `calc.get_dipole_moment()` returns it as a vector of three
components about the origin of the coordinates, in e Angstrom (e Bohr with
`atomic_units=True`). If no calculation has run yet, it runs it first.

```python
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("H2O")
atoms.center(vacuum=4.0)
atoms.calc = Mandacaru(method="dft", xc="pbe", basis={"name": "PAW-LCAO", "size": "DZP"},
                       grid_spacing=0.25)
print(atoms.calc.get_dipole_moment())      # e Angstrom, shape (3,)
atoms.calc.close()
```

- **What it is.** The charges of the ions at their positions (the PAW valence
  charges on a PAW-LCAO basis, the nuclear numbers on an all-electron one) minus the
  electrons' position expectation, which on a PAW-LCAO basis includes the one-center
  dipole of the spheres. The moment of a neutral molecule does not depend on the
  origin; a charged one's does, so read it about a stated point.
- **Which states.** The mean-field state of `method="dft"`, `"rhf"` or `"uhf"`,
  closed shell or open shell. With `excited_state=` the call still returns the
  moment of the ground-state reference, not that of the root
  ([one excited state as the calculation](#excited-states-surface)).
- **What raises.** A crystal raises `NotImplementedError` naming
  `get_polarization`, the Berry-phase polarization of an insulator
  ([the Berry guide](berry.md)). A correlated method (`vqe`, `adapt-vqe`, `vasqa`,
  `valsqa` and the subspace methods) raises `ValueError`: the call reads a
  mean-field state, not the quantum one. Each refusal was read from the code, not
  requested on this build.

The transition dipoles of `get_excitations()` are a different quantity: the
matrix elements of the position between the ground state and each root
([Excited states](excited_states.md)).

## Densities on the grid

`calc.volumetric_field("density")` returns the electron density on the
calculation's grid together with the nuclei, their charges (the valence charges
for a PAW-LCAO basis) and the electron count; `field.integral` is the number of
electrons on the grid. With a PAW-LCAO basis the grid holds the smooth valence
density and `field.augmentation_charge` is the charge inside the augmentation
spheres that the grid does not see ([Visualization](visualization.md)). The
field can be integrated over any region of your own choosing.

(charges-not-available)=
## Per-atom charges and moments

Atomic partial charges (Hirshfeld, Voronoi, Bader), per-atom magnetic moments,
and the `charges` and `magmoms` columns of an extended-XYZ file are not yet
available; see [not yet available](#not-yet-available). There is no
`atoms.get_charges()`.
