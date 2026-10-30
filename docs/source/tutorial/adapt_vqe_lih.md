# Native ADAPT-VQE for LiH

First generate `lih.json` with the [VQE tutorial](vqe_lih.md). The native ADAPT
solver selects generators from their exact pool gradients, adds an operator, and
reoptimizes the growing circuit. The Hamiltonian is fixed across this comparison.

```python
from mandacaru import Mandacaru

calculator = Mandacaru(
    method="adapt-vqe", load_hamiltonian="lih.json",
    pool="fermionic", gradient="analytic", profile=False,
    optimizer={"method": "L-BFGS", "maxiter": 500},
    convergence={"gradient": 1e-5}, max_iterations=30, trace=False,
)
result = calculator.run()
print("Energy (eV):", result.optimal_energy)
print("Converged:", result.converged)
print("Selected operators:", result.num_operators)
```

The outer gradient criterion (`convergence`) and the inner optimizer describe
different parts of convergence. A small pool gradient does not establish
exactness, grid convergence or basis completeness. Compare the reference energy,
the final energy, the selected operators and both convergence outcomes: the outer
`result.converged`, and `result.optimizer_failures` for the inner minimizations.
`max_iterations` caps the number of operators the run may add.

## Pools

`pool=` chooses the operator pool: `"fermionic"` (the default), `"qubit"`, `"qeb"`,
`"ceo"`, `"ceo-ovp"` or `"spin-orbit"`. The pools differ in circuit depth and in
how many gradients each step measures, and they are not alternate spellings of
one another. Because the Hamiltonian is cached, comparing pools costs only the
ADAPT runs:

```python
for pool in ("fermionic", "qubit", "qeb", "ceo"):
    result = Mandacaru(method="adapt-vqe", load_hamiltonian="lih.json",
                       pool=pool, optimizer={"method": "L-BFGS", "maxiter": 500},
                       trace=False).run()
    print(f"{pool:10s} {result.optimal_energy:.6f} eV  "
          f"{result.num_operators} operators")
```

The `"qubit"` and `"spin-orbit"` pools do not conserve the particle number, so
they run without the particle-number sector. Architecture search over the same
pools is [MCAS-VQE](mcas_vqe.md), and its learned-proposal variant is
[VALQA](valqa.md).

## Checkpoints

`checkpoint=` writes the final operator sequence and parameters when the run ends,
and `resume=` warm-starts a compatible Hamiltonian from it; the stored operators
and parameters are checked before use. A checkpoint does not restore an
interrupted optimizer iteration. See [checkpoints and QPE](../guide/checkpoints_qpe.md).

Continue with a [geometry scan](pes_scan.md) to compare energies at fixed
numerical settings. Changing a geometry requires a new Hamiltonian.
