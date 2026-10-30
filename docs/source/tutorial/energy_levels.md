# Molecular Energy Levels (Excited States)

Ground-state VQE finds the lowest eigenvalue of the molecular Hamiltonian. To get
the **energy levels**, the ground state *and* excited states, a `method="vqe"`
calculator has an `energy_levels` method built on **variational quantum
deflation** (VQD). Other methods refuse it: sequential deflation needs a fixed
ansatz.

After the $m$ lowest states $\{|\psi_j\rangle\}_{j<m}$ are found, the next one is
obtained by minimizing the *deflated* cost

```{math}
L_m(\vec\theta) = \langle\psi(\vec\theta)|H|\psi(\vec\theta)\rangle
    + \beta\sum_{j<m} |\langle\psi_j|\psi(\vec\theta)\rangle|^2 .
```

If the earlier states are exact, the ansatz is sufficiently expressive and
$\beta$ exceeds the relevant energy gaps, the global minimum targets the next
eigenstate. In practice the states and the optimization are approximate. The
reported energy excludes the penalty: it is an energy expectation value, not a
guarantee of an exact eigenvalue.

## Levels from a VQE calculator

Attach the calculator with `method="vqe"`, evaluate the ground state once (this
builds the Hamiltonian and configures the solver), then ask for the levels:

```python
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("H2")
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="vqe",
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       grid_spacing=0.25)
atoms.get_potential_energy()

levels = atoms.calc.energy_levels(num_states=2, restarts=4)
for i, energy in enumerate(levels.energies):
    print(f"Level {i}: {energy:.6f} eV")
print(f"Excitation energy: {levels.excitation_energies[1]:.4f} eV")
print(levels.in_units("Ha"))          # the same levels in Hartree
print(levels.success)                 # one flag per level
```

This is `examples/excited_states/03_deflation_h2.py`. `energy_levels(num_states=2,
restarts=4, beta=None, seed=0)` returns an `EnergyLevels` with ascending
`energies` (eV, like every Mandacaru result; Hartree with `atomic_units=True`), the
optimal `states`, and the views `ground_state_energy`, `excitation_energies`,
`gaps`, `in_units(...)`, together with the per-level `success` flags and
`messages` of the inner optimizations and the total `num_evaluations`.

`beta=None` leaves the penalty weight to the solver; pass a value to tune it.
`restarts` runs the optimizer several times per level, from seeded random starts
(`seed`), and keeps the best, which helps the excited-state searches escape local
minima. The optimizer is the calculator's own `optimizer=`; see
[choosing an optimizer](../guide/optimizers.md).

## What the levels are

The returned levels are variational estimates within the states the ansatz can
reach. Approximate deflation can leave residual overlap with earlier states, and
a local optimizer can miss the intended excited state. Check convergence, the
state overlaps and, for small problems, exact diagonalization in the same
particle-number sector.

A repeated level can indicate a restricted ansatz or a failed optimization; it
does not by itself prove that all accessible states have been found. More
restarts or a different ansatz can help diagnose the limitation.

To find all the levels in one optimization instead of one at a time, use
[subspace-search VQE](subspace_vqe.md).
