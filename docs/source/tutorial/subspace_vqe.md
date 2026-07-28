# Simultaneous States with Subspace-Search VQE

The {doc}`energy_levels` tutorial found excited states **one at a time** with
deflation. **Subspace-search VQE** (SSVQE) instead finds the ground state and the
first few excited states **all at once**, in a single optimization — select it on
{class}`~carcara.algorithms.QuantumCalculator` with `method="subspace-vqe"`
(fixed ansatz) or `method="subspace-adapt-vqe"` (adaptively grown ansatz).

## The idea

Pick $k$ mutually orthogonal reference determinants $\{|\varphi_j\rangle\}$, send
them all through the **same** parameterized unitary $U(\vec\theta)$, and minimize
the *weighted* energy sum

$$L(\vec\theta) = \sum_{j=0}^{k-1} w_j\,
    \langle\varphi_j|U^\dagger(\vec\theta)\,H\,U(\vec\theta)|\varphi_j\rangle ,
    \qquad w_0 > w_1 > \dots > w_{k-1} > 0 .$$

Because $U$ is unitary, the images $U|\varphi_j\rangle$ stay orthonormal; the
descending weights force the largest weight onto the lowest energy, so at the
optimum $U|\varphi_0\rangle$ is the ground state, $U|\varphi_1\rangle$ the first
excited state, and so on. Each level's reported energy is the *bare* expectation
value $\langle\varphi_j|U^\dagger H U|\varphi_j\rangle$.

The reference determinants are chosen automatically as the $k$ lowest-lying
determinants of the Hartree-Fock particle-number sector (all orthonormal and with
the correct electron count).

---

## Subspace-VQE

Both methods run through the same ASE calculator. Attach it, evaluate the energy
(which builds the Hamiltonian and runs the subspace search), and read the full
spectrum off `result`:

```python
import numpy as np
from ase import Atoms
from carcara.algorithms import QuantumCalculator

atoms = Atoms("H2", positions=[[4.0, 4.0, 3.63], [4.0, 4.0, 4.37]],
              cell=[[8.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]], pbc=True)

atoms.calc = QuantumCalculator(method="subspace-vqe", basis="FAO", h=0.20,
                               num_states=2, weights=[2.0, 1.0], verbose=False)
atoms.get_potential_energy()               # ASE energy = ground state (eV)

result = atoms.calc.result
print(result.in_units("eV"))               # [E0, E1, ...] all levels
print(result.excitation_energies)          # [0, E1 - E0, ...] (Hartree)
print(result.levels)                        # an EnergyLevels view
```

`num_states` sets how many levels to compute; `weights` are the (strictly
decreasing, positive) SSVQE weights, defaulting to $(k, k-1, \dots, 1)$. The
result exposes `energies` (ascending, Hartree), `optimal_energy` (the ground
state), the orthonormal `states`, and `in_units("eV")`.

---

## Subspace-ADAPT-VQE

`method="subspace-adapt-vqe"` grows **one shared adaptive ansatz**
for all the states: its pool-screening gradient is the weighted sum of the
per-reference gradients $\sum_j w_j\,\langle\psi_j|[H, A_i]|\psi_j\rangle$, and the
inner re-optimization minimizes the weighted energy. It records how many operators
were grown.

```python
atoms.calc = QuantumCalculator(method="subspace-adapt-vqe", basis="FAO", h=0.20,
                               pool="fermionic", num_states=2, verbose=False,
                               gradient_tolerance=1e-4, max_iterations=20)
atoms.get_potential_energy()

result = atoms.calc.result
print(result.in_units("eV"))
print(result.num_operators)                # operators in the shared ansatz
```

---

## What the levels mean

The returned energies are variational **upper bounds** on the exact spectrum. For
orthonormal trial states the Hylleraas-Undheim-MacDonald theorem guarantees

$$E_i^{\text{SSVQE}} \ge \lambda_i \quad\text{for every } i,$$

where $\lambda_i$ is the $i$-th exact eigenvalue. So the levels are ordered,
orthonormal, and never below the true values.

Two practical points:

- **Ground state.** From the default start (all-zero parameters, i.e. the
  Hartree-Fock reference) the optimization settles near the Hartree-Fock basin and
  recovers the ground state essentially exactly.
- **Excited-state tightness depends on the ansatz.** SSVQE is exact only when the
  shared unitary is expressive enough to map *every* reference to *its* eigenstate
  simultaneously. A small ansatz (e.g. UCCSD on minimal-basis H\ :sub:`2`) gives
  looser excited-state bounds; a richer ansatz — or `method="subspace-adapt-vqe"`
  with a larger pool — tightens them. Increasing the ground-state weight $w_0$ pins the
  ground state more firmly at the cost of the excited estimates.

A complete, runnable script (both methods, compared to exact diagonalization) is
`examples/09_SubspaceVQE_H2.py`.
