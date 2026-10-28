# Native VQE for LiH

A neutral LiH molecule has four electrons. This example builds its molecular
Hamiltonian and optimizes a UCCSD state in Rust. Python provides ASE geometry
and result access. No quantum-service credentials are needed.

```python
from ase import Atoms
from mandacaru import Mandacaru

atoms = Atoms("LiH", positions=[[3., 3., 2.2], [3., 3., 3.8]], cell=[6.]*3)
atoms.calc = Mandacaru(
    method="vqe", engine="rust", basis="STO-3G", grid_spacing=.4,
    active_space={"frozen": "auto"}, mapping="jordan_wigner",
    optimizer={"method": "L-BFGS", "maxiter": 500, "tol": 1e-9},
    trace=False,
    save_hamiltonian="lih.json", hamiltonian_format="json",
)
energy = atoms.get_potential_energy()
result = atoms.calc.result
print("Total energy (eV):", energy)
print("Optimizer succeeded:", result.success)
print("Parameters:", result.num_parameters)
```

The grid spacing is deliberately coarse for a first run. Repeat with a finer
grid and a larger cell before making a physical comparison. The named Gaussian
family uses generated exponents. It does not reproduce a published basis table.

`active_space` freezes the selected doubly occupied core orbitals. Its constant
energy contribution remains in the total energy. Changing this reduction
changes the correlated problem, so keep it fixed when comparing solvers.

The native engine builds orbitals and integrals, solves the reference mean
field, maps the active Hamiltonian, evaluates states and derivatives, and runs
the optimizer. It does not compile at runtime or select a numerical fallback.
See [native execution](../guide/native_engine.md) for memory and method limits.

## Reuse the Hamiltonian

```python
calculator = Mandacaru(
    method="vqe", engine="rust", load_hamiltonian="lih.json",
    optimizer={"method": "L-BFGS", "maxiter": 500}, trace=False,
)
result = calculator.run()
print(result.optimal_energy)
```

Rebuild the Hamiltonian after changing the geometry, basis, grid, occupations
or active space. Reusing a cache is appropriate for a solver comparison on the
same electronic problem. The [ADAPT tutorial](adapt_vqe_lih.md) uses this cache.
