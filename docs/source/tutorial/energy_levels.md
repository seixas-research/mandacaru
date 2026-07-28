# Molecular Energy Levels (Excited States)

Ground-state VQE finds the lowest eigenvalue of the molecular Hamiltonian. To get
the **energy levels** — the ground state *and* excited states —
{class}`~carcara.algorithms.QuantumCalculator` exposes an `energy_levels` method
built on **variational quantum deflation** (VQD), available for every `method=`
(in particular `"vqe"` and `"adapt-vqe"`).

After the $m$ lowest states $\{|\psi_j\rangle\}_{j<m}$ are found, the next one is
obtained by minimizing the *deflated* cost

$$L_m(\vec\theta) = \langle\psi(\vec\theta)|H|\psi(\vec\theta)\rangle
    + \beta\sum_{j<m} |\langle\psi_j|\psi(\vec\theta)\rangle|^2 ,$$

whose global minimum is the $m$-th eigenstate: the penalty $\beta$ pushes the
search orthogonal to every lower state. The reported energy of each level is the
*bare* expectation value (the penalty vanishes at the eigenstate), so it is
unbiased.

---

## VQE energy levels

Attach the calculator with `method="vqe"`, evaluate the ground-state energy once
(this builds the Hamiltonian and configures the solver), then ask for the levels:

```python
import numpy as np
from ase import Atoms
from carcara.algorithms import QuantumCalculator

atoms = Atoms("H2", positions=[[4.0, 4.0, 3.63], [4.0, 4.0, 4.37]],
              cell=[[8.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]], pbc=True)
atoms.calc = QuantumCalculator(method="vqe", basis="FAO", h=0.20, verbose=False)
atoms.get_potential_energy()                 # configures the solver

levels = atoms.calc.energy_levels(num_states=2, restarts=4)
print(levels.in_units("eV"))                 # ground + first excited (eV)
print(levels.excitation_energies_in_units("eV"))   # [0, E1 - E0, ...]
```

`energy_levels` returns an {class}`~carcara.algorithms.EnergyLevels` with ascending
energies (Hartree), the optimal state vectors, and convenience views:
`ground_state_energy`, `excitation_energies`, `gaps`, and `in_units("eV")`.

`beta` defaults to a robust value derived from the Hamiltonian's coefficient
1-norm; pass it explicitly to tune. `restarts` runs the optimizer several times
per level (seeded random starts) and keeps the best, which helps the
excited-state searches escape local minima.

---

## ADAPT-VQE energy levels

The same call works with `method="adapt-vqe"`, which **grows a fresh deflated
ansatz for each level** — both the pool-screening gradient and the inner
re-optimization carry the penalty term, so the adaptive ansatz builds itself
toward the next excited state:

```python
atoms.calc = QuantumCalculator(method="adapt-vqe", pool="fermionic",
                               basis="FAO", h=0.20, verbose=False)
atoms.get_potential_energy()

levels = atoms.calc.energy_levels(num_states=2)
print(levels.in_units("eV"))
print(levels.num_operators)                  # operators grown per level
```

---

## What the levels are

Every energy returned is a **true eigenvalue** of the qubit Hamiltonian, but the
levels that can be reached are those the ansatz spans from its Hartree-Fock
reference. This is a physical feature of variational excited states, not an
approximation error:

- **UCCSD-VQE** optimizes all excitation amplitudes jointly, so it can reach some
  singly-excited levels.
- **ADAPT-VQE** grows greedily; from a closed-shell HF reference, single
  excitations have zero gradient (Brillouin's theorem), so on H\ :sub:`2` it
  reaches the seniority-zero excited partner (the doubly-excited
  $^1\Sigma_g^+$ level).

Requesting more states than the ansatz sector supports can therefore return a
level more than once — that signals the reachable spectrum is exhausted.

A complete, runnable script is `examples/08_energy_levels_H2.py`.
