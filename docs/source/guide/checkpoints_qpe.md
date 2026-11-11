# Wavefunction checkpoints and quantum phase estimation

Every state Mandacaru prepares has one shape: a reference determinant followed
by an ordered product of exponentials,

$$
|\Psi\rangle = \prod_k e^{\theta_k A_k}\,|\text{ref}\rangle ,
$$

whether ADAPT-VQE grew the generators $A_k$ one at a time or UCCSD fixed them up
front. A **wavefunction checkpoint** is that description written to disk: the
register, the reference, the generators as Pauli sums, the angles, the qubit
Hamiltonian and the solver's progress, in a JSON file that needs no Mandacaru
object to be read back. It serves two purposes: **resuming** a run, and
**handing the state to another algorithm**, here quantum phase estimation.

## Writing a checkpoint

`checkpoint=` names the file. It is written when the run ends, with the
solver's final status (`complete`, `converged` and the termination reason).
`checkpoint_every=N` also writes it during the run (`vqe` and `adapt-vqe`):

- `adapt-vqe` writes after every `N` accepted growth steps, with what that step
  reached.
- `vqe` writes the best point so far after every `N` cost evaluations.

A mid-run file says `"complete": false` (with `iteration` or `num_evaluations`).
`resume=` continues from it like from a final one: LiH resumed from the file of
its second growth step grows the same six operators to the same energy as the
uninterrupted run. Every write is atomic, so an interrupted run leaves the last
whole checkpoint.

An `adapt-vqe` run with SLSQP (the default) or BFGS also stores the optimizer's
curvature estimate of the angles (`status["hessian"]`, checkpoint schema version
3: `optimizer`, `dimension` and the row-major `values`; SLSQP's is a Hessian
approximation, BFGS's an inverse one), and a run resumed with the same optimizer
continues from it: the growth that follows is the one the uninterrupted run would
have taken. The two are not interchangeable and are not converted: a file written
by SLSQP resumed with BFGS (or the reverse) starts from the identity, relaxes the
stored sequence and raises a `RuntimeWarning` naming both optimizers. A checkpoint written
by an earlier build (version 2) records none; resuming it works from the identity
and raises a `RuntimeWarning`, since the growth need not then reproduce the
uninterrupted run. A file written by another optimizer, or by a fixed ansatz,
has no Hessian and no warning.

The `subspace-*` methods write a checkpoint of their shared ansatz, with the state
weights and the sorted state energies (the file's `subspace` entry). `resume=`
continues from it with the same method and weights; other weights raise
`ValueError`. `subspace-vqe` restarts its optimization from the stored
parameters. `subspace-adapt-vqe` keeps growing from the stored sequence, its stored
steps counting toward `max_iterations`: LiH stopped after two growth steps and
resumed reaches the same state energies as the uninterrupted run. With SLSQP its
checkpoint stores the Hessian approximation like the `adapt-vqe` one (above), and
the resumed growth is the uninterrupted run's, bit for bit; a file without it
(another optimizer, or version 2) relaxes the stored sequence first.

```python
import os
from ase.build import molecule
from mandacaru import Mandacaru
from mandacaru.core import load_checkpoint

os.makedirs("data", exist_ok=True)
path = os.path.join("data", "h2_wavefunction.json")

atoms = molecule("H2")
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis="HAO",
                       grid_spacing=0.25,
                       pool="fermionic",
                       checkpoint=path)
atoms.get_potential_energy()

checkpoint = load_checkpoint(path)
print(checkpoint.energy, checkpoint.labels, checkpoint.parameters)
```

The record is also on the calculator as `calc.solver.checkpoint` when no path is
given. `checkpoint.energy` is in Hartree. The state, the circuit and the tuple
every circuit provider takes are all derived from the file:

```python
psi = checkpoint.state_vector()     # the amplitudes, prepared from the file
qc = checkpoint.circuit()           # the Qiskit state-preparation circuit
n, reference, generators, thetas, hamiltonian = checkpoint.problem()
```

Nothing in a checkpoint refers to a pool, an ansatz class or a driver, which is
what makes it algorithm-agnostic.

### Product or sum

The generators and angles do not say how they are combined, so the file records
`preparation`:

| `preparation` | state | written by |
| :--- | :--- | :--- |
| `"product"` | $\prod_k e^{\theta_k A_k}\,\vert\mathrm{ref}\rangle$ | ADAPT-VQE, and any run with `trotter=True` (forced by `ansatz="hva"`, circuit execution, `vasqa` and `valsqa`) |
| `"sum"` | $e^{\sum_k \theta_k A_k}\,\vert\mathrm{ref}\rangle$ | the default exact UCCSD of `method="vqe"` |

The two coincide only when the generators commute, so a reader must not guess.
`state_vector()` honors either form. A circuit is an ordered product, so
`circuit()` and `problem()` refuse a `"sum"` checkpoint; run with `trotter=True`
when the state is meant for hardware. An exact-group HVA checkpoint
(`"evolution": "exact"`) is refused for circuit export for the same reason, a
noncommuting group exponential not being a list of Pauli rotations, while its
`state_vector()` works; a `"trotter"` HVA checkpoint exports.

## Resuming

`resume=` reads a checkpoint into a new run, which starts from the stored
operators and angles instead of the empty ansatz.

```python
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis="HAO",
                       grid_spacing=0.25,
                       pool="fermionic",
                       resume=path,
                       checkpoint=path)
atoms.get_potential_energy()
```

A checkpoint written for another Hamiltonian (the previous geometry of a
relaxation, a changed charge) is a warm start, not a continuation: its angles
are optimal there, not here. The solver re-optimizes them under the new
Hamiltonian before reading any pool gradient. Resuming a completed native
checkpoint does not restore an interrupted optimizer iteration.

Subspace methods (`subspace-vqe`, `subspace-adapt-vqe`) refuse `checkpoint=`
and `resume=`, and the architecture searches (`vasqa`, `valsqa`) refuse
`resume=`.

## Carrying the ansatz along a relaxation

Within one geometry optimization, `transfer=True` does the same without a file
and follows the molecular orbitals between geometries:

```python
from ase.optimize import BFGS

atoms = molecule("H2")
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       grid_spacing=0.25,
                       transfer=True)
BFGS(atoms).run(fmax=0.05)
```

Each geometry starts from the operator sequence and angles the previous one
ended with. The orbitals of the two geometries are matched, generator signs
are carried over, and the angles are re-optimized against the new Hamiltonian
before the solver decides whether to keep the carried start or grow from the
reference. With SLSQP or BFGS the previous geometry's curvature estimate of the
angles travels too (conjugated by the generator signs) and the re-optimization
starts from it; without one it starts from the identity. The start is refused when an orbital's best match falls below
`transfer_threshold` (default `0.9`). `transfer_steps` sets the shorter chain
of the architecture searches that a carried start allows
([VASQA tutorial](../tutorial/vasqa.md)). The transfer is described in the
[native coverage](../development/native_cutover.md).

## Quantum phase estimation

QPE reads eigenvalues of $H$ off the phases of
$U = e^{2\pi i (H - E_{\rm lo})/W}$: an eigenstate $|E_j\rangle$ has phase
$\varphi_j = (E_j - E_{\rm lo})/W$, and $t$ evaluation qubits resolve it to
$2^{-t}$, so the energy to $W/2^t$. Fed $|\Psi\rangle = \sum_j c_j |E_j\rangle$,
the evaluation register peaks at every $\varphi_j$ with weight $|c_j|^2$, so the
variational state is the natural input: QPE turns the variational energy into a
reading of the exact eigenvalue with a known resolution.

```python
from mandacaru.core import load_checkpoint
from mandacaru.workflows.qpe import QuantumPhaseEstimation

checkpoint = load_checkpoint(path)
window = (checkpoint.energy - 0.02, checkpoint.energy + 0.02)      # Hartree
result = QuantumPhaseEstimation(n_evaluation_qubits=10,
                                energy_window=window).run(checkpoint)
print(f"QPE: {result.energy:.6f} eV  (resolution {result.resolution:.1e} eV, "
      f"probability {result.probability:.3f})")
```

`run` also takes the path of a checkpoint. `examples/qpe/01_qpe_h2_from_checkpoint.py`
is this workflow as a script. Energies are reported in eV, and the window is in
Hartree. Without `energy_window` the window is the rigorous bound from the
Pauli coefficients, safe but wide, so resolution suffers; a tight window around
the states of interest buys resolution at no cost in qubits. An eigenvalue that
carries weight outside the window wraps around, so keep the window honest.

The result also has `peaks` (a list of `(energy, probability)` by probability),
`exact_energies` and `ground_state_overlap` where the dense path is used, and
`collapsed_state()`, the system register after the most probable reading.

### Simulation and memory

The simulation is exact: the full $2^{n+t}$ state vector, with no Trotter step
and no truncation. That vector is the cost, 16 bytes per amplitude and about
three of them in flight. Before anything is allocated the run sizes it:

```python
from mandacaru.workflows.qpe import qpe_memory_estimate

print(qpe_memory_estimate(n_system=24, n_evaluation=16).summary())
```

`memory_policy="abort"` (the default) raises `MemoryError` above
`memory_fraction` (0.5) of the available memory; `"prompt"` asks on the
terminal and aborts when there is none; `"ignore"` proceeds with a warning.

### The circuit

`QuantumPhaseEstimation.circuit(checkpoint)` exports the experiment as a Qiskit
circuit: the checkpoint's state preparation on the system wires and a
`PhaseEstimation` around an exact `PauliEvolutionGate`. This is the **exact
reference** synthesis, not a hardware one: the controlled powers of $U$ are
arbitrary unitaries whose gate count grows exponentially with the register. It
is useful to check a small system against `run()`, not to size a submission.
