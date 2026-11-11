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
   chemical accuracy (1.6 mHa) can need millions of shots. A run with `shots > 0`
   reports the measured one, `result.energy_error` (in the energy unit), from the
   counts with the covariances of the strings each basis reads on the same shots. Each
   basis is measured rotated to `Z` together with the distinct products of its strings,
   read on the same shots: they cost no shot or basis, and no `measurement_budget`
   ceiling counts them. A basis needing more than 20,000 products (a classical limit)
   leaves the run without an error: `result.energy_error` is `None` and
   `result.energy_error_reason` says why.
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

The defaults are sized for small molecules and stay as they are. **A large run passes its
own budget explicitly**: a 50-qubit ADAPT-VQE step at one alpha and one beta electron measures
about 13,000 gradient bases, over the default 10,000, so it is refused with the size to pass:

```text
measurement bases budget exceeded: 13056 > 10000; ... pass measurement_budget={"bases": N}
sized for this run (here N >= 13056)
```

Pass `measurement_budget={"bases": 13_056, "observables": 500_000}` (the numbers the refusal
reports, with room for the observables) and bound the job size with `max_bases_per_job`. The
run budget refuses the same way, naming `backend_options={"run_budget": {...}}` with the
jobs or shots this run needs.

**An oversized pool is refused before it is planned.** The default `"ceo"` pool is generalized
over every spin-orbital quadruple, so its size grows as the fourth power of the register: at 50
qubits it has 588,300 generators and one gradient step needs about 7.8 million Pauli
observables, against the default ceiling of 50,000. The native planner forms the pool's
commutators in bounded batches and stops at the first batch that carries the distinct Pauli
strings past the `observables` ceiling, so the refusal comes in seconds with bounded memory
(measured 2026-11-10 on the 50-qubit one-electron run: 360 s and 1.9 GB before, 2.1 s and
0.29 GB after; the figures are in `dev_plan/PAW_MIGRATION_LOG.md`) and no job or gradient plan is built:

```text
the 'ceo' pool has 588,300 generators and measuring its gradients needs more than 50,000 Pauli
observables (the first N commutators already hold M distinct strings): ... choose a smaller
pool, pool="fermionic" or "qeb", or pass measurement_budget={"observables": N} sized for this run
```

The count in the message is a lower bound (the walk stopped when the ceiling was crossed), never
an estimate: a pool that fits the ceiling is never refused. Name `pool="fermionic"` or `"qeb"`
for a register this wide, or raise `observables` above what the full pool needs.

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
unfinished `expectation_values` or `expectation_batch` request are kept, keyed by what was
asked.

**A measurement that needs several circuits is still one job.** The Qiskit provider's
`expectation_batch` sends one Estimator PUB per circuit in a single job, and the limit packs
whole circuits and chunks into jobs of at most that many bases, so the jobs of one measurement
are $\lceil \text{bases} / \text{limit} \rceil$, as `measurement_cost` counts. Either scheme
costs one job per energy: the commuting groups are one circuit with many observables, the
double-factorized energy is $L + 1$ circuits. Braket submits a task per basis, so there a
batch costs its bases in tasks.

(double-factorized-measurement)=
## Fewer bases: the double-factorized measurement

The commuting-group scheme needs $O(M^3)$ bases. `measurement_scheme="double-factorized"`
measures the energy in $L + 1$ bases instead, with $L$ the number of leaves of the
double factorization of the molecular integrals (Motta *et al.*, npj Quantum Inf.
**7**, 83, 2021; Huggins *et al.*, npj Quantum Inf. **7**, 23, 2021). The two-body
operator is a sum of squares of one-body operators, and each is diagonal in an orbital
basis of its own. A short network of Givens rotations on neighboring qubits changes the
basis, after which one computational-basis measurement of the whole register answers for
a whole leaf:

$$H = E_0 + \sum_k \epsilon_k \tilde n_k + \tfrac12\sum_\ell \lambda_\ell
\Bigl(\sum_k g_{\ell k}\, \tilde n^{(\ell)}_k\Bigr)^2 .$$

```python
calc = Mandacaru(method="vqe", basis="HAO", shots=20000, backend_provider="qiskit",
                 measurement_scheme="double-factorized")
```

Everything numerical is Rust: the factorization, the Givens networks, the diagonal
observables and the assembly of the energy. The calculator measures the energy of the
molecular Hamiltonian that the calculation built, so the energy equals the
commuting-group energy up to the leaf truncation (leaves below `1e-8` of the largest weight
are dropped; the energy differs from the exact one by about $4\times10^{-13}$ Ha for LiH and
$3\times10^{-14}$ Ha for H$_2$ in the `HAO` basis). `problem.factorized_measurement` reports
the number of bases, the longest Givens network and the discarded tensor norm.

* **The cost is priced before the first job.** The $L + 1$ bases are one batch of circuits, so
  on a Qiskit device an energy costs one job (or $\lceil (L + 1) / \text{limit} \rceil$ under
  `max_bases_per_job`) and $(L + 1)\times$ `shots` shots (times the resilience multipliers):
  `measurement_cost(L + 1, ...)`; on Braket, $L + 1$ tasks (`job_per_basis=True`). An energy
  that reuses the previous point's values costs less, never more. The run-budget pre-flight
  multiplies that by the optimizer's evaluation bound, and `measurement_budget["bases"]` is
  checked against $L + 1$.
* **The Givens network is circuit depth.** Fewer bases do not make the circuit shorter: each
  basis appends its network to the ansatz, and a rotation of two neighboring modes is two
  commuting two-qubit Pauli rotations, four two-qubit gates before transpilation. The longest
  network is `problem.factorized_measurement["rotations"]` (6 for LiH in the three-orbital
  `HAO` basis, so at most 24 more entanglers). On a processor this is the third wall, the fidelity wall,
  not the second.
* **The pool gradients of `adapt-vqe` stay on the commuting groups.** The commutator
  $[H, A_k]$ is not diagonal in any leaf basis; only the energies are factorized.
* **What it refuses.** A circuit device is required (the state-vector solver measures
  nothing). The molecular integrals must be available: a supplied or loaded Hamiltonian
  (`hamiltonian=`, `load_hamiltonian=`), a crystal and a spinor or complex model raise
  `ValueError`, as do mappings other than Jordan-Wigner (the scheme reads occupations off
  the computational basis), `taper=True` and `ansatz="hva"`. Nothing falls back to another
  scheme.
* **The shot path stays polynomial.** The factorization works on the $M^4$ integrals and an
  $M^2\times M^2$ pair matrix, never on a $2^n$ object.

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

The shot-based problem is built **register-free**: the Hamiltonian, the pool
generators and the measured commutators are Pauli operators, and there is no
state-vector basis, matrix or state. Particle-number conservation (`sector=True`)
and Z2 tapering (`taper=True`) are checked and applied on the Pauli terms. The
result carries no state, so everything that reads one refuses with a message
saying the run is register-free: density matrices, natural orbitals and
molecular fields, variational forces, `sparse=True`, `energy_levels`, and the
subspace and architecture-search methods. A fixed-ansatz VQE with a
derivative-free optimizer (SPSA, COBYLA, Nelder-Mead) reports no final gradient
(`final_max_gradient` is NaN), since measuring it costs two evaluations per
rotation. A test builds a 50-qubit problem on this path and asserts that no
exponential object appears (`test/architecture/test_shot_path_memory.py`).

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
on a circuit device. Measured reduced density matrices and forces,
re-measuring an optimized state with `remeasure()`, and a separate
`measurement_provider` are not implemented; see
[what is not yet available](#not-yet-available). On a circuit device
`get_forces()` raises: forces are not an implemented property there.

## What to reach for

* **Too big to submit:** lower `resilience_level` in `estimator_options`, set
  `max_bases_per_job`, or shrink the register. A refusal by `measurement_budget` names the
  value to pass; large runs pass it explicitly.
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
