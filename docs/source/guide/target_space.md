# Choosing the target space

A target space is the set of states a downstream calculation works in: the
Bloch states a Wannier function is built from, or the orbitals an active space
keeps. This page used to describe choosing a crystal target space by atomic
projectability and by RPA natural orbitals. **None of that is available now.**
The calculator has no Wannier, projectability or RPA natural-orbital
interface, and the [native coverage table](../development/native_cutover.md)
lists Wannier construction, constrained RPA and downfolding among the work that
remains. The entry in [what is not yet available](#not-yet-available) will move
when it returns.

## What exists

* **Molecules.** An active space selects the orbitals by energy or by MP2
  natural-orbital occupation, with a count or a threshold; see
  [the active-space guide](active_space.md). The selector's spectrum is on the
  result as `result.hamiltonian.selection_occupations`.
* **Natural orbitals of a computed state.** After a molecular quantum run,
  `calc.natural_orbitals()` returns the occupations and orbital coefficients of
  the state's one-particle density, diagonalized in Rust. `state=` picks a level
  of a subspace search.
* **Crystals.** Periodic DFT gives bands, density of states and spectra; see
  [Kohn-Sham DFT](dft.md). The bands are the input a target-space choice would
  start from.

```python
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("H2")
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", grid_spacing=0.35)
atoms.get_potential_energy()

orbitals = atoms.calc.natural_orbitals()
print(orbitals)
```
