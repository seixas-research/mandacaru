# What a hardware measurement costs

A state-vector run reads the energy off the amplitudes. A processor cannot: it
returns bit-strings, so every Pauli string of the Hamiltonian has to be measured
in its own basis. This guide is about what that costs, which limits the
calculator enforces before a job is queued, and where the wall is. The devices
and providers themselves are in [backends](backends.md).

It exists because of one job. LiH in PAW-LCAO-TZP, 24 qubits, was submitted to
an IBM processor as 97,980 observables in about 21,000 measurement bases. It
waited 37 minutes in the queue and then failed with
`Program runtime ran out of memory` (error code 1336), discarding a converged
optimization. The default `measurement_budget` below is sized to refuse that job.

## The three walls

1. **Memory.** A single submission carrying a very large number of observables
   can exceed what the Runtime program can hold.
2. **Shots.** The number of qubit-wise commuting (QWC) bases grows roughly as
   $M^3$ in the number of orbitals $M$, and each basis needs enough shots for
   the coefficient 1-norm. The standard error of a measured energy is bounded
   by $\left(\sum_{P\neq I}|c_P|\right)/\sqrt{\text{shots}}$ per basis, so
   chemical accuracy (1.6 mHa) can need millions of shots.
3. **Fidelity.** Hardware noise, not shot noise, limits the useful circuit: a
   circuit of hundreds of two-qubit gates retains almost no signal, and no
   number of shots recovers it.

Fixing the first wall does not make a 24-qubit run meaningful. It makes a run
of a few qubits possible at all, which is where the useful demonstrations are.

## Size a measurement before paying for it

The qubit Hamiltonian of a finished run is on the calculator, and the grouping
and the noise bound need nothing else. Both are local computations:

```python
from ase.build import molecule
from mandacaru import Mandacaru
from mandacaru.backends import qubit_wise_commuting_groups, shot_noise_estimate
from mandacaru.core import PauliSum

atoms = molecule("H2")
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="vqe",
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       grid_spacing=0.25)
atoms.get_potential_energy()

problem = atoms.calc.problem
hamiltonian = PauliSum({label: complex(re, im)
                        for label, re, im in problem.hamiltonian_terms},
                       num_qubits=problem.num_qubits)
groups, identity = qubit_wise_commuting_groups(hamiltonian)
print(f"{len(hamiltonian.terms)} Pauli strings in {len(groups)} bases")
print(f"standard error <= {shot_noise_estimate(hamiltonian, 4096):.4f} Ha "
      "at 4096 shots per basis")
```

An energy is one weighted observable, so the Estimator groups the terms itself
and returns one number. The identity term needs no measurement.

## The measurement budget

`measurement_budget=` refuses an oversized submission before it queues. It takes
`observables`, `bases`, `circuit_instances` and `total_shots`; a key set to
`None` switches that check off, `False` switches all of them off, and a dict
overrides only the keys it names (an unknown key raises). The defaults are
50,000 observables, 10,000 bases, 2,000,000 circuit instances and no shot
ceiling.

The calculator enforces `observables` and `bases` on every submission, and on
the whole pool-gradient step of a shot-based ADAPT-VQE run. A submission over
either limit raises `ValueError` naming the count and the limit, before any job
is queued. Keep the budget on for any real submission.

(run-budget)=
## The run budget

An optimization on a processor submits a job per objective evaluation, so a
per-job limit cannot keep a run inside a quota (the IBM open plan is ten minutes
per month). A processor therefore also carries a **run-wide** budget, on by
default: an `ibm_*` device, a Braket QPU or a Braket managed simulator.

* **Default:** 250 jobs and no shot ceiling. A local or fake device has none
  unless you ask for one.
* **Setting it:** `backend_options={"run_budget": {"jobs": 500}}`; keys are
  `jobs` and `total_shots`, keys left out keep their defaults, `False` switches
  it off, and an unknown key raises. A rehearsal on a fake device that passes the
  same explicit dict is refused the same way.
* **Before the first job:** for `method="vqe"`, the optimizer's bound on its
  energy evaluations times the jobs one energy needs is checked against the
  budget. A run that cannot fit is refused with `RunBudgetExceeded`, naming the
  evaluations and the limit. SPSA and COBYLA have such a bound; Nelder-Mead does
  not, so the ledger stops it where the budget ends; see
  [the optimizer guide](#shots-optimizer).
* **During the run:** every submission is charged before it is queued. A
  submission that would exceed the budget is not sent, and the optimizer ends at
  its best point so far. The calculator issues a `RuntimeWarning` and keeps the
  reason in `calc.run_budget_exhausted`. If the budget stops the run before any
  energy exists, `RunBudgetExceeded` is raised instead.
* **Reading it back:** `calc.provider.run_usage()` returns the jobs and shots
  spent so far.

```python
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("H2")
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="vqe",
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       grid_spacing=0.25,
                       mapping="parity_reduced",
                       device="fake_kingston",
                       shots=4096,
                       optimizer={"method": "COBYLA", "maxiter": 30, "tol": 1e-3},
                       measurement_budget={"observables": 50_000, "bases": 10_000},
                       backend_options={"physical_qubits": [0, 1],
                                        "run_budget": {"jobs": 120},
                                        "estimator_options": {"max_execution_time": 60}})
energy = atoms.get_potential_energy()
print(f"E = {energy:.4f} eV, spent {atoms.calc.provider.run_usage()}")
```

This is `examples/ibm_quantum/01_vqe_h2_ibm.py` with the budgets written out. The
device is a fake backend: the real transpilation and noise model, locally, with
no queue and no bill. `max_execution_time` in `estimator_options` makes Runtime
cancel an overrun on a real device.

## Bounded jobs

`backend_options={"max_bases_per_job": 250}` splits a submission into several
jobs of at most that many QWC bases, never cutting a basis in half. It applies to
an energy too: the Hamiltonian is split into weighted observables and the number
of jobs one energy costs is what the run-budget pre-flight multiplies by. The
ansatz is transpiled once and shared by every job. Completed chunks of an
unfinished `expectation_values` request are kept, keyed by what was asked.

(adapt-on-a-device)=
## ADAPT-VQE on a device

With `shots > 0` nothing is simulated. Every quantity a shot-based ADAPT-VQE
uses comes from a measurement or from polynomial classical algebra: the
reference energy from the determinant's `I` and `Z` strings, every energy from
the Estimator, and every pool gradient as the expectation value of a commutator,

$$g_k = \langle\psi|[H, A_k]|\psi\rangle = \sum_R c_{kR}\,\langle\psi|R|\psi\rangle .$$

No $2^n$ array, no sector-sized vector and no local screening is built on this
path, on any device. The `shots=0` local simulation keeps its exact state-vector
route.

* **The first step is classical and exact.** At the Hartree-Fock determinant
  $\langle b|R|b\rangle$ is $\pm1$ for an `I`/`Z` string and 0 otherwise, so no
  measurement is needed.
* **After it, the commutators of the operators are merged:** a string shared by
  several operators is measured once, and the strings are grouped into QWC bases.
* **The plan is made before anything is queued.** It does not depend on the
  state. The run log reports it in its `[GRADIENT MEASUREMENT]` block (operators
  measured, Pauli strings, bases, jobs and shots per growth step), and
  `examples/ibm_quantum/05_adapt_h2_measured_gradients.py` reads it back from the
  log with `parse_output`. The plan is checked against the measurement budget.

A fake device needs `qiskit-aer`, which simulates the circuit under the
processor's noise model. A circuit transpiled to a 156-qubit processor
(`fake_kingston`, `fake_fez`, `fake_marrakesh`) spans all 156 wires, so the
five-qubit `fake_manila` is the cheap rehearsal device for the pool-gradient
example.

## What is not available

The calculator measures the **energy** (and, for ADAPT-VQE, the pool gradients)
on a circuit device. Measured reduced density matrices and forces, the
double-factorized measurement scheme, re-measuring an optimized state with
`remeasure()`, and a separate `measurement_provider` are not implemented; see
[what is not yet available](#not-yet-available). On a circuit device
`get_forces()` raises: forces are not an implemented property there.

## What to reach for

* **Too big to submit:** lower `resilience_level` in `estimator_options`, set
  `max_bases_per_job`, or shrink the register.
* **Too expensive in shots:** a smaller basis, `mapping="parity_reduced"`, or an
  [active space](active_space.md).
* **Too deep to trust:** `pool="ceo-ovp"`, and pin the layout with
  `backend_options={"physical_qubits": [...]}`. Prefer a live readout check over
  the calibration record.
* **Too many jobs for the run budget:** fewer evaluations
  (`optimizer={"maxiter": ...}`), or raise the budget only if the run needs it.
* **On the open plan,** the realistic target is two to four qubits.
  `examples/ibm_quantum/01_vqe_h2_ibm.py` is the shape that works: two qubits and
  a few CNOTs.
