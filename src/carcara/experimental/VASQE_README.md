
### Example 5: VASQE (experimental — stochastic ADAPT with temperature annealing)
Grow the ansatz by sampling operators from a softmax of the gradients, annealing
the selection temperature from exploratory to greedy. VASQE is an experimental
method (`carcara.experimental`); see `docs/experimental/vasqe.md`:
```python
from ase import Atoms
from carcara.algorithms import Carcara

atoms = Atoms("H2", positions=[[4.0, 4.0, 3.63], [4.0, 4.0, 4.37]],
              cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]], pbc=True)

atoms.calc = Carcara(method="vasqe", basis="FAO", pool="fermionic",
                               h=0.20, optimizer="L-BFGS-B",
                               temperature=2.0, final_temperature=0.01,
                               schedule="exponential",
                               max_iterations=12, gradient_tolerance=1e-5)
atoms.get_total_energy()
result = atoms.calc.result

print(f"Energy: {result.optimal_energy:.8f} Ha")
print(f"Operators: {result.operators}")
print(f"Selection temperatures: {result.temperatures}")
```