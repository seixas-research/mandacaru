# Adaptive Eigensolving with ADAPT-VQE

This tutorial demonstrates how to use **ADAPT-VQE** in Carcará to dynamically build compact quantum circuits for $H_2$ and $LiH$. We compare the hardware cost (CNOT gate counts and circuit depth) across four operator pools.

---

## Running ADAPT-VQE on LiH

Unlike the fixed UCCSD ansatz, ADAPT-VQE selects operators from a pool one by one based on their energy gradient:

$$g_i = \langle\psi^{(n)}| [H, A_i] |\psi^{(n)}\rangle$$

and grows the ansatz dynamically.

Here, we attach `ADAPTVQE` as an ASE calculator to solve for the ground state of $LiH$ using the hardware-efficient Coupled-Exchange Operator (`"ceo"`) pool:

```python
from ase import Atoms
from carcara.algorithms import ADAPTVQE

# Setup LiH molecule in a cell
atoms = Atoms("LiH", positions=[[4.0, 4.0, 3.20], [4.0, 4.0, 4.80]],
              cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]], pbc=True)

# Attach ADAPTVQE calculator
atoms.calc = ADAPTVQE(
    pool="ceo",
    basis="FAO",
    mapping="jordan_wigner",
    optimizer="COBYLA",
    gradient="parameter-shift",
    h=0.18,
    max_iterations=15,
    gradient_tolerance=1e-5
)

# Run calculation (energy returned in eV)
energy_ev = atoms.get_total_energy()
result = atoms.calc.adapt_result

print(f"ADAPT-VQE Converged: {result.converged}")
print(f"Optimal Energy: {result.optimal_energy:.8f} Ha")
print(f"Number of Operators Growth: {result.num_operators}")
```

---

## Comparing Operator Pools

Carcará provides four pools that trade off parameter freedom and circuit compilation complexity:

| Pool Name | Description | Gate Count Trade-off |
| :--- | :--- | :--- |
| `"fermionic"` | Spin-adapted excitations | Deep circuits due to Jordan-Wigner $Z$-string chains |
| `"qubit"` | Individual mapped Pauli strings | Shallow circuits per step, but higher total parameter count |
| `"qeb"` | Qubit Excitation Basis | $Z$-strings are omitted, yielding distance-independent entangling gates |
| `"ceo"` | Coupled-Exchange Operators | Shares entangling blocks, yielding maximum accuracy per CNOT |

We can run the comparative analysis across these pools on $H_2$:

```python
from ase import Atoms
from carcara.algorithms import ADAPTVQE
from carcara.optimizers import Optimizer

# H2 molecule
atoms = Atoms("H2", positions=[[4.0, 4.0, 3.63], [4.0, 4.0, 4.37]],
              cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]], pbc=True)

pools = ["fermionic", "qubit", "qeb", "ceo"]
results = {}

for name in pools:
    atoms.calc = ADAPTVQE(
        pool=name,
        basis="FAO",
        h=0.20,
        optimizer=Optimizer("L-BFGS-B", maxiter=2000),
        max_iterations=10,
        gradient_tolerance=1e-5,
        verbose=False
    )
    atoms.get_total_energy()
    results[name] = atoms.calc.adapt_result

# Print comparison
print(f"{'Pool':12s} | {'Energy (Ha)':>12s} | {'CNOT Count':>10s} | {'Depth':>8s}")
print("-" * 52)
for name in pools:
    res = results[name]
    print(f"{name:12s} | {res.optimal_energy:12.6f} | {res.metrics.cnot_count:10d} | {res.metrics.depth:8d}")
```

On $H_2$ all pools converge to the exact ground state, but the hardware-optimized qubit-based pools achieve it with significantly fewer CNOTs (e.g., `"qubit"` requires only 6 CNOTs, whereas `"fermionic"` requires 48).
