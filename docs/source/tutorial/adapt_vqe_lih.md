# LiH with ADAPT-VQE: choosing an operator pool

In [the VQE tutorial](vqe_lih.md), the circuit structure was fixed before the
optimisation began. ADAPT-VQE builds its circuit gradually, selecting generators
from an **operator pool**. We keep the LiH geometry, frozen core and basis the
same so that the change in algorithm is easy to follow.

## 1. Run an adaptive calculation

Save this complete example as `lih_adapt.py` and run `python lih_adapt.py`.

```python
from ase import Atoms
from carcara.algorithms import Carcara
from carcara.integrals import Grid
from carcara.optimizers import Optimizer

atoms = Atoms(
    "LiH",
    positions=[[0.0, 0.0, -0.8], [0.0, 0.0, 0.8]],
    pbc=False,
)
grid = Grid(center=[0.0, 0.0, 0.0], box_size=4.8, h=0.12)

atoms.calc = Carcara(
    method="adapt-vqe",
    pool="fermionic",
    basis="STO-3G",
    frozen_core=True,
    grid=grid,
    mapping="jordan_wigner",
    optimizer=Optimizer("L-BFGS-B", maxiter=1000),
    gradient="analytic",
    gradient_tolerance=1e-5,
    max_iterations=40,
    execute_circuits=False,
    profile=False,
    save_hamiltonian="lih_sto3g_r1p6.json",
    hamiltonian_format="json")

energy_ev = atoms.get_potential_energy()
result = atoms.calc.result
print(f"Total energy: {energy_ev:.6f} eV")
print(f"Pool-gradient criterion satisfied: {result.converged}")
print(f"Selected operators: {result.num_operators}")
print(f"Unsuccessful inner optimisations: {result.optimizer_failures}")
```

This calculation starts with the Hartree–Fock reference. At each growth step,
Carcará evaluates the energy derivative for each candidate generator $A_i$:

```{math}
g_i = \left.\frac{\mathrm{d}}{\mathrm{d}\theta}
\langle\psi|e^{-\theta A_i}\hat H e^{\theta A_i}|\psi\rangle
\right|_{\theta=0}
= \langle\psi|[\hat H,A_i]|\psi\rangle,
\qquad A_i^\dagger=-A_i.
```

The largest gradient magnitude identifies the next generator. Carcará adds it
and reoptimises all circuit parameters. The loop stops when the largest pool
gradient is below `gradient_tolerance`, or when it reaches `max_iterations`.
The screening threshold uses the internal Hamiltonian's **Hartree** units,
although the returned energies use eV.

There are two optimisation loops: `max_iterations` limits circuit growth;
`Optimizer(..., maxiter=1000)` limits each classical parameter optimisation.
Check both the outer convergence flag and `optimizer_failures`.

## 2. Understand the available pools

| Pool | Construction | What to compare |
| :--- | :--- | :--- |
| `fermionic` | Mapped single and double fermionic excitations | A chemistry-based starting point; parity strings can increase circuit depth. |
| `qubit` | Individual Pauli terms from mapped excitations | Fewer gates per generator may require more growth steps; individual terms need not preserve particle number. |
| `qeb` | Excitation generators with parity-only Z strings removed | Often shorter circuits; check the resulting accuracy and conserved quantities. |
| `ceo` | QEB generators grouped by qubit support | In Carcará's Jordan–Wigner implementation the groups are singletons, so this pool matches `qeb`. |

These are implementation descriptions, not universal rankings. Gate counts
depend on the mapping, selected operators, compiler and device connectivity.
Carcará does not implement the specialised CEO synthesis required for the gate
savings reported for that method in the literature.

## 3. Compare pools using one Hamiltonian

The first calculation saved the complete qubit Hamiltonian and its electronic
metadata. Append the following block to `lih_adapt.py` to reuse it. This avoids
rebuilding integrals and ensures each pool receives exactly the same problem.

```python
for pool in ("fermionic", "qubit", "qeb", "ceo"):
    calc = Carcara(
        method="adapt-vqe",
        pool=pool,
        load_hamiltonian="lih_sto3g_r1p6.json",
        optimizer=Optimizer("L-BFGS-B", maxiter=1000),
        gradient_tolerance=1e-5,
        max_iterations=40,
        execute_circuits=False,
        profile=False)
    result = calc.run()
    print(
        f"{pool:10s}  {result.optimal_energy:.6f} eV  "
        f"operators={result.num_operators:2d}  converged={result.converged}"
    )
```

Energy agreement is an observation to check, not an assumption. Limited circuit
growth, local minima and different reachable state spaces can cause differences.
A small pool gradient alone does not prove that the state is the exact ground
state. In particular, a pool that does not conserve particle number needs a
separate check of the final electron number.

`profile=False` skips circuit compilation for gate-count reporting. To study
hardware resources, set `profile=True` and inspect `result.metrics`; compilation
adds work even when energies are evaluated locally.

Continue with [the LiH distance scan](pes_scan.md) to compare these pools across
two basis sets and generate a PNG figure.
