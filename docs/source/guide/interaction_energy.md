(interaction-energy)=
# Interaction energies on a shared grid

The binding energy of a complex is a difference of three total energies,

```{math}
E_\text{int} = E(\text{AB}) - E(\text{A}) - E(\text{B}),
```

and on a real-space grid that difference is only meaningful if all three are
evaluated on **the same grid, with every atom at the same position relative to
the grid nodes**. A calculator builds its grid from the geometry it is given, so a
fragment computed on its own is sampled differently than it is inside the
complex. For a soft hydrogen that costs a few meV; for an atom with a sharp core
it costs electronvolts, and the "interaction energy" then measures the grid and
not the chemistry. The same reasoning is why a force calculation freezes its grid
([Forces](forces_relaxation.md)).

A helper that builds the grid once and evaluates the complex and each fragment on
it, and the counterpoise correction for the basis-set superposition error (ghost
atoms carrying basis functions but no nucleus, pseudopotential or electrons), are
not yet available; see [not yet available](#not-yet-available).

Until they are, two choices keep a hand-made difference honest:

- use a PAW-LCAO basis, whose valence orbitals are resolved on the grid (an
  all-electron core is not), with a mean-field method as a cheap check before any
  variational run ([PAW-LCAO bases](pseudopotentials.md));
- an atom-centered basis lets each fragment in the complex borrow its partner's
  functions, which the isolated fragment does not have, so a compact or confined
  basis exaggerates the binding. Use an unconfined basis (`"energy_shift": None`
  for a PAW-LCAO basis) and converge the result with the basis size and with
  `grid_spacing`.
