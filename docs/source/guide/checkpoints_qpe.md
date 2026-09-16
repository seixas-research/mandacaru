# Wavefunction checkpoints and quantum phase estimation

Every state Carcará prepares has one shape: a reference determinant followed
by an ordered product of exponentials,

$$
|\Psi\rangle = \prod_k e^{\theta_k A_k}\,|\text{ref}\rangle ,
$$

whether ADAPT-VQE grew the generators $A_k$ one at a time or UCCSD fixed them
up front.  A **wavefunction checkpoint** is that description written to disk —
the register, the reference, the generators as Pauli sums, the angles, the
qubit Hamiltonian and the solver's progress — in a JSON file that needs no
Carcará object to be read back.  It serves two purposes: **resuming** a run
that was interrupted, failed or ran out of iterations, and **handing the
state to another algorithm**, here quantum phase estimation.

## Writing checkpoints during a run

```python
from ase import Atoms
from carcara import Carcara

atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[6.0] * 3)
atoms.center()
atoms.calc = Carcara(method="adapt-vqe",
                     basis="FAO",
                     h=0.25,
                     checkpoint="examples/data/h2_wavefunction.json",
                     checkpoint_every=1,
                     verbose=False)
atoms.get_potential_energy()
```

ADAPT-VQE writes the file after every `checkpoint_every` accepted operators;
VQE after every `checkpoint_every` cost evaluations, always with the **best
point seen so far** rather than the optimiser's current trial step.  Both
write it once more when the run ends, with `status.complete = True`.  Writes
are atomic (a temporary file renamed into place), so an interruption in the
middle of a write can never leave a torn file behind.

The record is also kept on the driver as `calc.solver.checkpoint` (or
`driver.checkpoint` on a bare `ADAPTVQE` / `VQE`) even when no path is given.

## Resuming

```python
atoms.calc = Carcara(method="adapt-vqe",
                     basis="FAO",
                     h=0.25,
                     resume="examples/data/h2_wavefunction.json",
                     checkpoint="examples/data/h2_wavefunction.json",
                     max_iterations=40,
                     verbose=False)
```

ADAPT-VQE rebuilds the grown ansatz from the file — each stored generator is
matched to the pool's own operator by content, and one the pool does not
contain is applied exactly as stored — restores the iteration history and the
evaluation count, and keeps growing.  `max_iterations` counts the *total*
number of operators, so a run that stopped on that limit continues only when
it is raised.  VQE checks that the stored generators are its own ansatz's and
starts the optimisation from the stored angles.

A checkpoint resumes only into the register it was written for: the qubit
count, the mapping, the two-qubit reduction and the reference determinant must
match, or the run refuses with a message saying which does not.  With
`checkpoint` and `resume` set to the same path, each geometry of a relaxation
warm-starts from the previous one.

## The file as a standalone object

```python
from carcara.core import load_checkpoint

ck = load_checkpoint("examples/data/h2_wavefunction.json")
print(ck.summary())
psi = ck.state_vector()            # the amplitudes, prepared from the file
E = ck.expectation()               # <H> in Hartree, from the stored Hamiltonian
qc = ck.circuit()                  # the Qiskit state-preparation circuit
n, reference, generators, thetas, H = ck.problem()
```

`problem()` is the same tuple every circuit provider and the QPE driver take,
which is what makes the format algorithm-agnostic: nothing in it refers to a
pool, an ansatz class or a driver.

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
from carcara.algorithms import QuantumPhaseEstimation

qpe = QuantumPhaseEstimation(n_evaluation_qubits=10)
result = qpe.run("examples/data/h2_wavefunction.json")
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
from carcara.algorithms import qpe_memory_estimate

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
Its gate count is what a hardware run would transpile.

See `examples/31_QPE_H2_from_checkpoint.py`.
