# Native ADAPT-VQE for LiH

First generate `lih.json` with the [VQE tutorial](vqe_lih.md). The native ADAPT
solver selects fermionic generators from their exact pool gradients, adds an
operator, and reoptimizes the growing circuit. The Hamiltonian is fixed across
this comparison.

```python
from mandacaru import Mandacaru

calculator = Mandacaru(
    method="adapt-vqe", engine="rust", load_hamiltonian="lih.json",
    pool="fermionic", gradient="analytic", profile=False,
    optimizer={"method": "L-BFGS", "maxiter": 500},
    convergence={"gradient": 1e-5}, max_iterations=30, trace=False,
)
result = calculator.run()
print("Energy (eV):", result.optimal_energy)
print("Converged:", result.converged)
print("Selected operators:", result.num_operators)
```

The outer gradient criterion and the inner optimizer describe different parts
of convergence. A small pool gradient does not establish exactness, grid
convergence or basis completeness. Compare the reference energy, final energy,
selected operators and both convergence outcomes.

The Rust solver currently supports the fermionic pool. Qubit, QEB and CEO pool
construction, architecture search and learned proposals remain outside its
coverage. They are not alternate spellings of this calculation.

A completed native checkpoint can warm-start a compatible Hamiltonian. Its
operator sequence and parameters are checked before use. It does not restore
an interrupted optimizer iteration; see [native execution](../guide/native_engine.md).

Continue with a [geometry scan](pes_scan.md) to compare energies at fixed
numerical settings. Changing a geometry requires a new Hamiltonian.
