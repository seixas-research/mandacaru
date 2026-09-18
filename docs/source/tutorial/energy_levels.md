# Molecular Energy Levels (Excited States)

Ground-state VQE finds the lowest eigenvalue of the molecular Hamiltonian. To get
the **energy levels** — the ground state *and* excited states —
{class}`~carcara.algorithms.Carcara` exposes an `energy_levels` method
built on **variational quantum deflation** (VQD), available for every `method=`
(in particular `"vqe"` and `"adapt-vqe"`).

After the $m$ lowest states $\{|\psi_j\rangle\}_{j<m}$ are found, the next one is
obtained by minimising the *deflated* cost

```{math}
L_m(\vec\theta) = \langle\psi(\vec\theta)|H|\psi(\vec\theta)\rangle
    + \beta\sum_{j<m} |\langle\psi_j|\psi(\vec\theta)\rangle|^2 ,
```

If the earlier states are exact, the ansatz is sufficiently expressive and
$\beta$ exceeds the relevant energy gaps, the global minimum targets the next
eigenstate. In practice the states and optimisation are approximate. The
reported energy excludes the penalty; it is an energy expectation value, not
a guarantee of an exact eigenvalue.

---

## VQE energy levels

Attach the calculator with `method="vqe"`, evaluate the ground-state energy once
(this builds the Hamiltonian and configures the solver), then ask for the levels:

```python
import numpy as np
from ase import Atoms
from carcara.algorithms import Carcara

atoms = Atoms("H2", positions=[[4.0, 4.0, 3.63], [4.0, 4.0, 4.37]],
              cell=[[8.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]], pbc=True)
atoms.calc = Carcara(method="vqe",
                     basis="FAO",
                     h=0.20)
atoms.get_potential_energy()                 # configures the solver

levels = atoms.calc.energy_levels(num_states=2, restarts=4)
print(levels.energies)                       # ground + first excited (eV)
print(levels.excitation_energies)            # [0, E1 - E0, ...] (eV)
print(levels.in_units("Ha"))                 # the same levels in Hartree
```

`energy_levels` returns an {class}`~carcara.algorithms.EnergyLevels` with ascending
energies (eV, like every Carcará result; Hartree only with `atomic_units=True`),
the optimal state vectors, and convenience views: `ground_state_energy`,
`excitation_energies`, `gaps`, and `in_units("Ha")`.

`beta` defaults to a robust value derived from the Hamiltonian's coefficient
1-norm; pass it explicitly to tune. `restarts` runs the optimiser several times
per level (seeded random starts) and keeps the best, which helps the
excited-state searches escape local minima.

---

## ADAPT-VQE energy levels

The same call works with `method="adapt-vqe"`, which **grows a fresh deflated
ansatz for each level** — both the pool-screening gradient and the inner
re-optimisation carry the penalty term, so the adaptive ansatz builds itself
towards the next excited state:

```python
atoms.calc = Carcara(method="adapt-vqe",
                     pool="fermionic",
                     basis="FAO",
                     h=0.20)
atoms.get_potential_energy()

levels = atoms.calc.energy_levels(num_states=2)
print(levels.energies)                       # eV
print(levels.num_operators)                  # operators grown per level
```

---

## What the levels are

The returned levels are variational estimates within the states reachable by
the ansatz. Approximate deflation can leave residual overlap with earlier states,
and a local optimiser can miss the intended excited state. Check convergence,
state overlaps and, for small problems, exact diagonalisation in the same
particle-number sector.

A repeated level can indicate a restricted ansatz or failed optimisation; it
does not by itself prove that all accessible states have been found. Increasing
the number of starts or changing the ansatz can help diagnose the limitation.

A complete, runnable script is `examples/08_energy_levels_H2.py`.
