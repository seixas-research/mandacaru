# Stochastic Adaptive Eigensolving with VASQE

**VASQE** — the **Variational Adaptive Stochastic Quantum Eigensolver**, selected
with `method="vasqe"` on {class}`~carcara.algorithms.QuantumCalculator` — is
ADAPT-VQE with a **stochastic operator-selection** rule.
ADAPT-VQE greedily appends the pool operator with the largest gradient magnitude;
VASQE instead **samples** the operator from a Boltzmann-like softmax of the
gradients at a *selection temperature* $\tau$:

$$P(i, \tau) = \frac{\exp\!\big(|g_i|/\tau\big)}{\sum_j \exp\!\big(|g_j|/\tau\big)} .$$

- As $\tau \to 0$ the probability concentrates on the largest-gradient operator, so
  **VASQE reduces exactly to ADAPT-VQE**.
- As $\tau \to \infty$ the distribution flattens toward uniform, letting the ansatz
  explore operators a greedy rule would never pick.

Convergence is unchanged from ADAPT-VQE — the loop still stops when
$\max_i |g_i| < \texttt{gradient\_tolerance}$. Only *which* operator is appended
changes, so VASQE reuses all of the ADAPT machinery (pools, gradients, the
growable ansatz, circuit profiling, ASE-calculator mode, frozen core, …).

---

## Running VASQE

VASQE runs through the same ASE calculator as every other method:

```python
from ase import Atoms
from carcara.algorithms import QuantumCalculator

atoms = Atoms("H2", positions=[[4.0, 4.0, 3.63], [4.0, 4.0, 4.37]],
              cell=[[8.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]], pbc=True)

atoms.calc = QuantumCalculator(method="vasqe", basis="FAO", pool="fermionic",
                               temperature=1.0, h=0.20, max_iterations=12,
                               gradient_tolerance=1e-5)
energy_ev = atoms.get_total_energy()

result = atoms.calc.result            # a VASQEResult (subclass of ADAPTVQEResult)
print(result.optimal_energy)          # Hartree
print(result.operators)               # the (stochastically) selected sequence
print(result.temperatures)            # the tau used at each growth step
```

Selection is reproducible: the RNG is reseeded from `seed` at the start of every
`run()`, so a fixed `seed` gives a fixed operator sequence.

---

## Temperature and annealing

The selection temperature can be held **constant** or **annealed** from a high
initial value (exploration) to a low final one (exploitation) over the growth
iterations. Four schedules are available via `schedule`:

| `schedule` | $\tau$ at step $k$ of $K$ | Behavior |
| :--- | :--- | :--- |
| `"constant"` | $\tau_0$ | fixed temperature (`final_temperature` ignored) |
| `"linear"` | $\tau_0 + (\tau_f-\tau_0)\,k/(K-1)$ | uniform cooling |
| `"exponential"` | $\tau_0\,(\tau_f/\tau_0)^{k/(K-1)}$ | fast early cooling |
| `"logarithmic"` | $\tau_0 + (\tau_f-\tau_0)\,\log(1{+}k)/\log K$ | slow early cooling |

```python
# Anneal from a hot, exploratory tau=2.0 down to a greedy tau=0.01.
atoms.calc = QuantumCalculator(method="vasqe", basis="FAO", temperature=2.0,
                               final_temperature=0.01, schedule="exponential",
                               annealing_steps=12, h=0.20,
                               max_iterations=12, gradient_tolerance=1e-5)
atoms.get_total_energy()
```

`annealing_steps` sets the horizon $K$ (default: `max_iterations`). A higher
initial temperature facilitates broader exploration of the ansatz space before the
schedule cools toward the greedy ADAPT-VQE choice.

The softmax and schedule helpers are also exposed directly:

```python
import numpy as np
from carcara.algorithms import softmax_selection_probabilities, annealed_temperature

softmax_selection_probabilities(np.array([0.1, 0.9, 0.4]), tau=0.05)  # ~[0,1,0]
annealed_temperature("exponential", 2.0, 0.01, step=3, horizon=12)
```

---

## Excited states

Because selection is the single {meth}`~carcara.algorithms.ADAPTVQE._select_operator`
hook, VASQE inherits both excited-state strategies from the driver framework, each
growing its ansatz **stochastically**:

```python
# Deflation: ground + excited states, one after another.
levels = atoms.calc.energy_levels(num_states=2)
print(levels.in_units("eV"))

# Subspace search: ground + excited states simultaneously.
atoms.calc = QuantumCalculator(method="subspace-vasqe", basis="FAO",
                               pool="fermionic", num_states=2,
                               temperature=0.5, h=0.20, gradient_tolerance=1e-5)
atoms.get_total_energy()
print(atoms.calc.result.in_units("eV"))
```

`method="subspace-vasqe"` combines the subspace-search machinery
(one shared ansatz over several orthogonal references, weighted-gradient screening)
with VASQE's stochastic selection.

A complete, runnable script comparing the temperature schedules on H\ :sub:`2` is
`examples/10_VASQE_H2.py`.
