# Wavefunction checkpoints and quantum phase estimation

Every state Mandacaru prepares has one shape: a reference determinant followed
by an ordered product of exponentials,

$$
|\Psi\rangle = \prod_k e^{\theta_k A_k}\,|\text{ref}\rangle ,
$$

whether ADAPT-VQE grew the generators $A_k$ one at a time or UCCSD fixed them
up front.  A **wavefunction checkpoint** is that description written to disk —
the register, the reference, the generators as Pauli sums, the angles, the
qubit Hamiltonian and the solver's progress — in a JSON file that needs no
Mandacaru object to be read back.  It serves two purposes: **resuming** a run
that was interrupted, failed or ran out of iterations, and **handing the
state to another algorithm**, here quantum phase estimation.

## Writing checkpoints during a run

```python
from ase import Atoms
from mandacaru import Mandacaru

atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[6.0] * 3)
atoms.center()
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis="HAO",
                       h=0.25,
                       checkpoint="examples/old/data/h2_wavefunction.json",
                       checkpoint_every=1)
atoms.get_potential_energy()
```

ADAPT-VQE writes the file after every `checkpoint_every` accepted operators;
VQE after every `checkpoint_every` cost evaluations, always with the **best
point seen so far** rather than the optimizer's current trial step.  Both
write it once more when the run ends, with `status.complete = True`.  Writes
are atomic (a temporary file renamed into place), so an interruption in the
middle of a write can never leave a torn file behind.

The record is also kept on the driver as `calc.solver.checkpoint` (or
`driver.checkpoint` on a bare `ADAPTVQE` / `VQE`) even when no path is given.

## Resuming

```python
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis="HAO",
                       h=0.25,
                       resume="examples/old/data/h2_wavefunction.json",
                       checkpoint="examples/old/data/h2_wavefunction.json",
                       max_iterations=40)
```

ADAPT-VQE rebuilds the grown ansatz from the file — each stored generator is
matched to the pool's own operator by content, and one the pool does not
contain is applied exactly as stored — restores the iteration history and the
evaluation count, and keeps growing.  `max_iterations` counts the *total*
number of operators, so a run that stopped on that limit continues only when
it is raised.  VQE checks that the stored generators are its own ansatz's and
starts the optimization from the stored angles.

A checkpoint resumes only into the register it was written for: the qubit
count, the mapping and the reference determinant must
match, or the run refuses with a message saying which does not.

A checkpoint written for **another Hamiltonian** -- the previous geometry of
a relaxation, a changed charge -- is a warm start, not a continuation. Its
angles are optimal there, not here, so ADAPT-VQE first re-optimizes all of
them under this run's Hamiltonian, and only then reads the pool gradients.
At angles left over from the other problem, the gradients would mostly
measure the stale angles, and the operators already in the ansatz would be
selected again. After the re-optimization the gradient alone decides: below
the convergence threshold the stored ansatz stands as it is, otherwise
operators are appended to it. The result's `start` and the run log's
`[OPTIMIZATION SETUP]` say which happened.

## Carrying the ansatz along a relaxation

Within one geometry optimization, `transfer=True` does the same without a
file, and follows the molecular orbitals between the geometries:

```python
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis={"name": "PAW-LCAO", "size": "DZP"},
                       active_space={"orbitals": 4, "method": "mp2",
                                     "symmetry": True},
                       transfer=True)
BFGS(atoms).run(fmax=0.05)
```

Each geometry starts from the operator sequence and angles the previous one
ended with. The orbitals are matched between the two geometries
({mod}`~mandacaru.algorithms.orbital_tracking`): an operator is renamed where
two orbitals swapped places, and its angle changes sign where an orbital
did. The angles are re-optimized, and the pool gradient then decides whether
the carried ansatz stands as it is or grows. The carried start is kept only
if it lies below the reference energy. It is refused, with the reason in the
log's `start` line, when the previous geometry was another system or pool,
when an orbital's best match falls below `transfer_threshold` (0.9), or when
an operator cannot be followed through the matching.

A degenerate set (LiH's π pair) has no orbitals of its own, only a span, and
the eigensolver would pick a different basis of it at each geometry -- by
round-off, 40 degrees apart between 1.595 and 1.580 Angstrom. The orbitals of
every degenerate set of the selector's natural orbitals are therefore fixed
as the eigenvectors of one fixed operator (a generic quadratic in the
position), which follow the geometry smoothly and change no physics. Keep the
set whole with `"symmetry": True`: a count that keeps one member of a pair is
an arbitrary choice, and the run warns.

On a LiH relaxation (PAW-LCAO-DZP, four MP2 natural orbitals, `qeb`), the
carried steps took 48 and 36 energy evaluations, against 150 from the empty
ansatz, and reached the same energy to 1e-8 eV. Two earlier steps were
refused: an MP2 natural orbital rotates strongly when the bond moves
0.19 Angstrom, and its best match fell to 0.72 and then to 0. Small steps,
or the canonical (`"energy"`) orbitals, keep the matching above the
threshold more often.

`checkpoint` and `resume` on the same path also carry the ansatz between
runs or processes, but without orbital tracking, because the file holds no
basis. The file must already exist: on the first geometry of a relaxation
there is nothing to resume, so use `transfer=True` there, or resume from a
file that a previous run wrote.

## The file as a standalone object

```python
from mandacaru.core import load_checkpoint

ck = load_checkpoint("examples/old/data/h2_wavefunction.json")
print(ck.summary())
psi = ck.state_vector()            # the amplitudes, prepared from the file
E = ck.expectation()               # <H> in Hartree, from the stored Hamiltonian
qc = ck.circuit()                  # the Qiskit state-preparation circuit
n, reference, generators, thetas, H = ck.problem()
```

`problem()` is the same tuple every circuit provider and the QPE driver take,
which is what makes the format algorithm-agnostic: nothing in it refers to a
pool, an ansatz class or a driver.

One thing the generators and angles do not determine is **how they are
combined**, so the file records it as `preparation`:

| `preparation` | state | written by |
| :--- | :--- | :--- |
| `"product"` | $\prod_k e^{\theta_k A_k}\,\vert\mathrm{ref}\rangle$ | ADAPT-VQE, UCCSD with `trotter=True`, and `ansatz="hva"` |
| `"sum"` | $e^{\sum_k \theta_k A_k}\,\vert\mathrm{ref}\rangle$ | the default, exact UCCSD of `method="vqe"` |

The two coincide only when the generators commute (on a three-angle H₂ example
their fidelity is 0.94), so a reader must not guess. `state_vector()`,
`expectation()` and QPE honor either form. A circuit is an ordered product, so
`circuit()` and `problem()` refuse a `"sum"` checkpoint, and so does resuming it
into a run that prepares a product; run the ansatz with `trotter=True` when the
state is meant for hardware. Files written before this field existed held
products only and load as such.

For `method="vqe"` with `ansatz="hva"` (or an `ansatz={"name": "hva", ...}`
dictionary), the checkpoint records `method: "vqe"` plus an `"hva"` metadata
block: the ordered Hamiltonian groups, layer count, evolution policy,
product-formula order and steps, taper sector, and any fixed UHF preparation.
Exact-group HVA checkpoints reproduce
the local state through `state_vector()` and can initialize QPE or Quantum
Echoes, but their `problem()` and `circuit()` methods refuse generic circuit
export: a noncommuting group exponential is not a list of Pauli rotations.
`"evolution": "trotter"` checkpoints store the compiled rotation stream and
can be exported as circuits. Both modes resume only with the same Hamiltonian,
groups, reference, mapping, and evolution policy.

A checkpoint also needs an ansatz that can be *described* — generators and a
reference determinant (`SerializableAnsatz`). A custom ansatz implementing only
the state-vector `Ansatz` protocol runs through `method="vqe"` unchanged; asking
for `checkpoint=` or `resume=` with it is refused before the optimization starts.

## Quantum phase estimation

QPE reads eigenvalues of $H$ off the phases of $U = e^{2\pi i (H - E_{\rm lo})/W}$:
an eigenstate $|E_j\rangle$ has phase $\varphi_j = (E_j - E_{\rm lo})/W$, and
$t$ evaluation qubits resolve it to $2^{-t}$, i.e. the energy to $W/2^t$.  Fed
$|\Psi\rangle = \sum_j c_j |E_j\rangle$, the evaluation register comes out
peaked at every $\varphi_j$ with weight $|c_j|^2$ — so the variational state is
the natural input: it overlaps the ground state almost perfectly, and QPE
turns the *variational* energy into a reading of the *exact* eigenvalue with a
known resolution.

```python
from mandacaru.algorithms import QuantumPhaseEstimation

qpe = QuantumPhaseEstimation(n_evaluation_qubits=10)
result = qpe.run("examples/old/data/h2_wavefunction.json")
print(result.summary())
result.energy                  # the most probable reading (eV)
result.resolution              # W / 2^t
result.peaks                   # [(energy, probability), ...] by probability
result.collapsed_state()       # the system register after that reading
result.exact_energies          # the spectrum, on the dense path
result.ground_state_overlap    # |<E0|Psi>|^2, the success probability
```

The default energy window is the rigorous bound from the Pauli coefficients,
$c_I \mp \sum_{P \neq I}|c_P|$ — safe but wide.  Passing a tight
`energy_window=(E_lo, E_hi)` (Hartree) around the states of interest buys
resolution at no cost in qubits; an eigenvalue that carries weight but lies
outside the window wraps around, so keep the window honest.

### Simulation and memory

The simulation is exact — the full $2^{n+t}$ state vector, built as
$U^k|\Psi\rangle$ for every evaluation index $k$ followed by the inverse QFT as
an FFT; no Trotter step, no truncation.  That vector *is* the cost: 16 bytes
per amplitude, about three of them in flight.  Before anything is allocated
the run sizes it and compares with the memory actually available:

```python
from mandacaru.algorithms import qpe_memory_estimate

print(qpe_memory_estimate(n_system=24, n_evaluation=16).summary())
# QPE statevector: 24 system + 16 evaluation qubits -> 2^40 amplitudes
# (16.00 TiB); working set 48.00 TiB = 48.00 TiB = 5.2e+05% of the ... available
```

`memory_policy="abort"` (default) raises `MemoryError` above
`memory_fraction` (50 %) of the available memory; `"prompt"` asks on the
terminal and aborts when there is none; `"ignore"` proceeds with a warning.
The estimate is printed on every run.

### The circuit

`qpe.circuit(checkpoint)` exports the experiment as a Qiskit circuit — the
checkpoint's state preparation on the system wires and
`qiskit.circuit.library.PhaseEstimation` around an exact
`PauliEvolutionGate` — whose state-vector simulation reproduces the native
distribution to machine precision (`QuantumPhaseEstimation.evaluation_distribution`).

This is the **exact reference** synthesis, not a hardware one:
`qiskit.synthesis.MatrixExponential` exponentiates the dense `2^n x 2^n`
Hamiltonian, so the controlled powers of `U` are arbitrary unitaries whose
gate count grows exponentially with the register — useful for checking a
small system against `run()`, not for sizing a real submission. Molecular QPE
on a processor needs a product-formula (or other) synthesis with an explicit
approximation budget, which this method does not provide.

See `examples/old/31_QPE_H2_from_checkpoint.py`.

### Citing it

Phase estimation runs from a checkpoint, outside any driver, so nothing writes
a bibliography for it the way a variational run does (see
[`references.bib`](#references-bib)). Ask
for the keys and write them yourself:

```python
from mandacaru.utils.citations import citation_keys, write_references

write_references("references.bib",
                 citation_keys(method="qpe", mapping="jordan_wigner"))
```
