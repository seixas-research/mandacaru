# Simultaneous States with Subspace-Search VQE

The [energy levels](energy_levels.md) tutorial found excited states **one at a
time** with deflation. **Subspace-search VQE** (SSVQE) finds the ground state and
the first few excited states **all at once**, in a single optimization. Select it
with `method="subspace-vqe"` (a fixed ansatz) or `method="subspace-adapt-vqe"` (an
adaptively grown ansatz).

## The idea

Pick $k$ mutually orthogonal reference determinants $\{|\varphi_j\rangle\}$, send
them all through the **same** parameterized unitary $U(\vec\theta)$, and minimize
the *weighted* energy sum

```{math}
L(\vec\theta) = \sum_{j=0}^{k-1} w_j\,
    \langle\varphi_j|U^\dagger(\vec\theta)\,H\,U(\vec\theta)|\varphi_j\rangle ,
    \qquad w_0 > w_1 > \dots > w_{k-1} > 0 .
```

Because $U$ is unitary, the images $U|\varphi_j\rangle$ stay orthonormal; the
descending weights force the largest weight onto the lowest energy, so at an
exact, fully expressive global optimum $U|\varphi_0\rangle$ is the ground state,
$U|\varphi_1\rangle$ the first excited state, and so on. Each level's reported
energy is the *bare* expectation value $\langle\varphi_j|U^\dagger H U|\varphi_j\rangle$.
The reference determinants are chosen automatically, all with the correct
electron count.

## Subspace-VQE

Attach the calculator, evaluate the energy (which builds the Hamiltonian and runs
the subspace search), and read the spectrum off `result`:

```python
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("H2")
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="subspace-vqe",
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       grid_spacing=0.25,
                       num_states=2)
atoms.get_potential_energy()               # ASE energy = ground state (eV)

result = atoms.calc.result
for i, energy in enumerate(result.energies):
    print(f"State {i}: {energy:.6f} eV")
print(result.excitation_energies)          # [0, E1 - E0, ...] (eV)
print(result.in_units("Ha"))               # the ground state in Hartree
```

This is `examples/excited_states/01_ssvqe_h2.py`. `num_states` sets the number of
levels. `weights` takes one positive finite value per state, defaulting to
$(k, k-1, \dots, 1)$. The result has `energies` (ascending, eV, like every
Mandacaru result), `optimal_energy` (the ground state), `excitation_energies`
and the orthonormal `states`.

## Subspace-ADAPT-VQE

`method="subspace-adapt-vqe"` grows **one shared adaptive ansatz** for all the
states: the pool-screening gradient is the weighted sum of the per-reference
gradients $\sum_j w_j\,\langle\psi_j|[H, A_i]|\psi_j\rangle$, and the inner
re-optimization minimizes the weighted energy.

```python
atoms.calc = Mandacaru(method="subspace-adapt-vqe",
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       grid_spacing=0.25,
                       pool="fermionic",
                       num_states=2,
                       convergence={"gradient": 1e-4})
atoms.get_potential_energy()

result = atoms.calc.result
print(result.energies)                     # eV
print(f"{result.num_operators} operators in the shared ansatz")
```

This is `examples/excited_states/02_subspace_adapt_vqe_h2.py`. The iteration
energy is the lowest level, while the gradient screens the weighted objective.

## Limits

* Subspace methods do not execute circuits, write a checkpoint or resume from
  one: circuit execution (`shots`, a non-default `device`), `checkpoint=` and
  `resume=` raise `NotImplementedError`. They have no forces, and a force request
  raises. Subspace checkpoint continuation is listed in the
  [native coverage](../development/native_cutover.md).
* A spin-adapted subspace search with the Hamiltonian variational ansatz, with
  a named spin multiplicity, is not available. See
  [what is not yet available](#not-yet-available).

## What the levels mean

Each returned energy is an expectation value for a normalized trial state, and
therefore lies above the Hamiltonian's ground-state energy. Orthogonality alone
does **not** make the sorted expectation values separate upper bounds to the
corresponding excited-state eigenvalues. For example, equal mixtures of two exact
eigenstates are orthogonal but both have their mean energy, so the higher sorted
expectation is below the exact excited-state energy. To obtain the usual
Rayleigh-Ritz bounds, diagonalize the Hamiltonian projected onto the trial
subspace,

```{math}
H^{\mathrm{sub}}_{ij}=\langle\psi_i|\hat H|\psi_j\rangle,
\qquad E_i^{\mathrm{Ritz}} \geq \lambda_i ,
```

where $E_i^{\mathrm{Ritz}}$ are the ordered projected eigenvalues and
$\lambda_i$ the ordered exact eigenvalues in the chosen sector. A larger ansatz
and a successful joint optimization improve the estimates, but neither a
ground-state weight nor a small optimization tolerance guarantees an exact
spectrum. Compare with a small exact calculation where possible.
