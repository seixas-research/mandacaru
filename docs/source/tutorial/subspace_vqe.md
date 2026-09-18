# Simultaneous States with Subspace-Search VQE

The {doc}`energy_levels` tutorial found excited states **one at a time** with
deflation. **Subspace-search VQE** (SSVQE) instead finds the ground state and the
first few excited states **all at once**, in a single optimisation — select it on
{class}`~carcara.algorithms.Carcara` with `method="subspace-vqe"`
(fixed ansatz) or `method="subspace-adapt-vqe"` (adaptively grown ansatz).

## The idea

Pick $k$ mutually orthogonal reference determinants $\{|\varphi_j\rangle\}$, send
them all through the **same** parameterised unitary $U(\vec\theta)$, and minimise
the *weighted* energy sum

```{math}
L(\vec\theta) = \sum_{j=0}^{k-1} w_j\,
    \langle\varphi_j|U^\dagger(\vec\theta)\,H\,U(\vec\theta)|\varphi_j\rangle ,
    \qquad w_0 > w_1 > \dots > w_{k-1} > 0 .
```

Because $U$ is unitary, the images $U|\varphi_j\rangle$ stay orthonormal; the
descending weights force the largest weight onto the lowest energy, so at an exact, fully expressive global
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
from carcara.algorithms import Carcara

atoms = Atoms("H2", positions=[[4.0, 4.0, 3.63], [4.0, 4.0, 4.37]],
              cell=[[8.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]], pbc=True)

atoms.calc = Carcara(method="subspace-vqe",
                     basis="FAO",
                     h=0.20,
                     num_states=2,
                     weights=[2.0, 1.0])
atoms.get_potential_energy()               # ASE energy = ground state (eV)

result = atoms.calc.result
print(result.energies)                     # [E0, E1, ...] all levels (eV)
print(result.excitation_energies)          # [0, E1 - E0, ...] (eV)
print(result.in_units("Ha"))               # the same levels in Hartree
print(result.levels)                        # an EnergyLevels view
```

`num_states` sets how many levels to compute; `weights` are the (strictly
decreasing, positive) SSVQE weights, defaulting to $(k, k-1, \dots, 1)$. The
result exposes `energies` (ascending, eV -- like every Carcará result),
`optimal_energy` (the ground state), the orthonormal `states`, and
`in_units("Ha")` for the atomic-unit view.

---

## Subspace-ADAPT-VQE

`method="subspace-adapt-vqe"` grows **one shared adaptive ansatz**
for all the states: its pool-screening gradient is the weighted sum of the
per-reference gradients $\sum_j w_j\,\langle\psi_j|[H, A_i]|\psi_j\rangle$, and the
inner re-optimisation minimises the weighted energy. It records how many operators
were grown.

```python
atoms.calc = Carcara(method="subspace-adapt-vqe",
                     basis="FAO",
                     h=0.20,
                     pool="fermionic",
                     num_states=2,
                     gradient_tolerance=1e-4,
                     max_iterations=20)
atoms.get_potential_energy()

result = atoms.calc.result
print(result.energies)                     # eV
print(result.num_operators)                # operators in the shared ansatz
```

---

## What the levels mean

Each returned energy is an expectation value for a normalised trial state,
and therefore lies above the Hamiltonian's ground-state energy. Orthogonality
alone does **not** make the sorted expectation values separate upper bounds to
the corresponding excited-state eigenvalues.

For example, equal mixtures of two exact eigenstates are orthogonal but both
have their mean energy. The higher sorted expectation is then below the exact
excited-state energy. To obtain the usual Rayleigh–Ritz bounds, diagonalise the
Hamiltonian projected onto the trial subspace:

```{math}
H^{\mathrm{sub}}_{ij}=\langle\psi_i|\hat H|\psi_j\rangle,
\qquad E_i^{\mathrm{Ritz}} \geq \lambda_i.
```

Here $E_i^{\mathrm{Ritz}}$ are the ordered projected eigenvalues and
$\lambda_i$ are the ordered exact eigenvalues in the chosen sector. The
reported SSVQE expectations coincide with these eigenvalues only when the
trial states diagonalise the projected Hamiltonian.

A larger ansatz and successful joint optimisation can improve the estimates.
Neither a ground-state weight nor a small optimisation tolerance guarantees an
exact spectrum. Compare the resulting states and energies with a small exact
calculation where possible.

A complete, runnable script (both methods, compared to exact diagonalisation) is
`examples/09_SubspaceVQE_H2.py`.
