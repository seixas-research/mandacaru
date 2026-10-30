(dry-run)=
# Dry run: how many qubits will this need?

Before submitting a calculation to a quantum device, or to a state-vector
simulator whose memory doubles with every qubit, you want to know the size of the
register. `estimate_qubits` answers that by *counting* the basis functions of a
geometry; it computes no integral, builds no Hamiltonian and runs no circuit.

(dry-run-estimate)=
## The estimate

```python
from ase.build import molecule
from mandacaru import estimate_qubits

water = molecule("H2O")
water.center(vacuum=3.0)

estimate = estimate_qubits(water, basis="HAO")
print(estimate.n_qubits, estimate.n_spatial_orbitals, estimate.mapping)

reduced = estimate_qubits(water, basis={"name": "NAO", "size": "DZP"},
                          mapping="parity_reduced")
```

The arguments are the geometry (an ASE `Atoms` or a `Geometry`), `basis`
(any all-electron specification of [Basis sets](basis_sets.md), including a
per-element mapping), `mapping` (default `"jordan_wigner"`) and `resources` (the
dictionary the calculator takes, for the thread count and the memory limit). It
returns a `QubitEstimate` with the fields `n_qubits`, `n_spatial_orbitals` and
`mapping`.

## How the count is made

```{math}
N_\text{qubits} = 2\,M_\text{basis},
```

one qubit per **spin orbital**, where $M_\text{basis}$ is the number of basis
functions the engine generates for the geometry, counted rather than integrated.
The Jordan–Wigner, parity and Bravyi–Kitaev mappings all use this many qubits;
`parity_reduced` uses two fewer.

The estimate is the size of the **full** basis. It does not apply a frozen core
or any other [active space](active_space.md), so a run that truncates its active
space needs a smaller register; after the run, `calc.n_qubits` reports the size
of the register that was actually used. The pseudopotential families
(`PAW-LCAO`, `UPAW-LCAO`) are built from datasets and are not counted by this
function.

A shot-based run does not simulate a state vector, so memory is not the limit
there; the limits are the number of measurement circuits and the device
capacity, covered in [Measurement cost](measurement_cost.md).

The command line has no dry-run flag. Constructing a `Mandacaru` calculator
validates its options at once without building the solver, so a mistyped option
fails before anything expensive runs.
