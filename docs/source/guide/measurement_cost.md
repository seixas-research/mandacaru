# What a hardware measurement costs

A state-vector run reads the energy off the amplitudes. A processor cannot: it
returns bit-strings, so every Pauli string of the Hamiltonian has to be measured
in its own basis, and the number of those grows fast. This guide is about that
growth — what it costs, what Mandacaru does about it, and where the wall is.

It exists because of one job. LiH at 1.60 Å in PAW-LCAO-TZP, 24 qubits, submitted to
`ibm_fez` at `resilience_level=2`: 37 minutes in the queue, then

```text
RuntimeJobFailureError: Error code 1336; Program runtime ran out of memory
```

and a converged 75-operator optimization thrown away with it.

## The three walls, in the order you hit them

Measured on LiH/PAW-LCAO at `grid_spacing = 0.25`, `pool="ceo-ovp"`, Jordan-Wigner
(with `kinetic="fd"`):

| basis | qubits | H Pauli strings | submitted with RDMs | QWC bases | ADAPT ops | 2q gates | fidelity |
|---|---|---|---|---|---|---|---|
| SZ | 4 | 46 | 50 | 23 | 3 | 18 | 0.95 |
| DZ | 8 | 360 | 980 | 93 | 15 | 150 | 0.64 |
| DZP | 20 | 5,866 | 46,170 | 1,600 | 45 | 881 | 0.07 |
| TZP | 24 | 12,736 | 97,980 | 3,290 | 75 | 2,223 | 0.001 |

The last column is $(1-3\times10^{-3})^{n_{2q}}$, a nominal fidelity at a typical
two-qubit error.

1. **Memory.** One PUB carrying ~$10^5$ observables is what the Runtime program
   could not hold, and under zero-noise extrapolation it keeps a result array
   per observable. This is the wall the fixes below remove.
2. **Shots.** Measurement bases go as $\approx 0.2\,M^3$ — read the table. The
   1-norm bound for chemical accuracy (1.6 mHa) is $(\sum_j|c_j|/\varepsilon)^2$
   shots *per basis*: on LiH-DZ that is $1.2\times10^{7}$ shots across 79 bases,
   about $10^{9}$ in total. The IBM Open plan is ten minutes per 28 days.
3. **Fidelity.** At 881 two-qubit gates there is a twelfth of the signal left,
   and at 2,223 a thousandth. No amount of shots recovers that, and ZNE has
   nothing to extrapolate from.

**Fixing the first wall does not make a 24-qubit run meaningful.** It makes a
4-to-8-qubit run possible at all, which is where the useful demonstrations are.

## Energy runs do not measure the RDMs

`get_potential_energy()` measures only the Hamiltonian's Pauli strings, not
every spin-conserving 1- and 2-RDM operator as well: submitting the union of
both would mean 97,980 strings instead of 12,736 on the TZP case, plus 34 s of
local work building 117,792 operators, only to discard the RDM half of it. The
RDMs are what the **forces** need, so they are measured when forces are asked
for and not otherwise:

```python
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis={"name": "PAW-LCAO", "size": "DZ"},
                       grid_spacing=0.25,
                       pool="ceo-ovp",
                       measurement_provider=QiskitProvider(device="ibm_fez",
                                                           shots=4096))
atoms.get_potential_energy()   # 1 observable: <H> as one weighted operator
atoms.get_forces()             # the RDM operators, because these need them
```

An energy is **one weighted `SparsePauliOp`**, so the Estimator groups the
terms itself and returns one number instead of an array of 12,736 — which is
also what removes the per-observable ZNE arrays from the Runtime's memory.

## The plan, before the queue

Every submission is sized first, logged as `[MEASUREMENT PLAN]`, and refused if
it exceeds a budget. Everything in it is computed locally:

```text
[MEASUREMENT PLAN]
    device: statevector
    scheme: qwc
    measured: energy only
    qubits: 8
    observables: 360
    measurement_bases: 93
    jobs: 1
    shots_per_basis: 4096
    zne_noise_factors: 1
    twirling_randomizations: 1
    shots_per_randomization: 4096
    circuit_instances: 93
    total_shots: 380928
    hamiltonian_one_norm_Ha: 5.502732
    shot_noise_bound_eV: 2.339640
    double_factorized_bases: 11 (available: measurement_scheme='double-factorized')
    isa_two_qubit_gates: 96
    isa_depth: 169
    expected_fidelity: 7.494e-01 (at a nominal 0.003 two-qubit error)
```

This plan is local (`device: statevector`), so `resilience_level` counts as
`0` and neither ZNE nor twirling turns on. On an IBM processor an *unset*
`resilience_level` is Runtime's own default of `1`: no ZNE, but measurement
twirling is already on. `circuit_instances` is `bases × noise_factors ×
randomizations`, so at level `1` it jumps to `93 × 1 × 32 = 2,976` for this
plan — `shots=4096` on the `auto` rule takes `max(64, ceil(shots / 32))`
shots per randomization (`128` here), hence `32` of them.

`total_shots` does **not** carry that same factor: the randomizations
**share** the requested shots rather than each repeating them in full, so a
(basis, noise-factor) pair still costs `shots_per_basis` shots split across
its twirls, not `randomizations × shots_per_basis`. `total_shots` therefore
stays `93 × 1 × 4096 = 380,928`, unchanged from the no-mitigation case, and
the per-randomization share is the plan's `shots_per_randomization` line. At
`resilience_level=2` (ZNE on, 3 noise factors by default) both grow:
`circuit_instances = 93 × 3 × 32 = 8,928`, `total_shots = 93 × 3 × 4096 =
1,142,784` — the noise factors multiply the shots, the randomizations do not.

The plan is an **estimate**: Runtime submits a target *precision*, and its
own learning circuits (for twirling and ZNE) are extra, so the job's returned
metadata is the final word on what it billed.

```python
Mandacaru(..., measurement_budget={"observables": 20_000, "bases": 5_000})
```

`measurement_budget` takes `observables`, `bases`, `circuit_instances` and
`total_shots`; a key set to `None` switches that check off, `False` switches
them all off, and the shipped defaults refuse the job that motivated them. A
plan below `0.1` expected fidelity raises a `RuntimeWarning` saying so.

## A failed job keeps the optimization

The state is optimized locally and only then measured, so a measurement failure
costs no optimization time — the exception does not take the calculator down
with it:

```python
try:
    energy = atoms.get_potential_energy()
except MeasurementFailed as exc:
    print(exc)                       # states the observable / basis / shot counts
    atoms.calc.remeasure(rdms=False, shots=1024,
                         estimator_options={"resilience_level": 1})
```

`remeasure()` submits the **same** ansatz again — no re-optimization — with
whatever provider or options you hand it, and `calc.measurement_plan` still
holds the plan of the attempt that failed. Passed keyword overrides (`shots`,
`estimator_options`, `device`, ...) rebuild the provider through its own
`with_options(...)`, so nothing the failed one had cached survives into the
retry; an unknown option name is refused rather than silently dropped. The
forces of the earlier measurement are dropped along with it (they belonged to
other RDMs), so `atoms.get_forces()` recomputes them after a `remeasure()`.

A run-wide budget refuses a submission for a different reason — before it is
even queued, not after it fails — and is covered on its own [below](#run-budget).

## Bounded jobs instead of one unbounded PUB

When the RDMs really are needed, one job of $10^5$ observables will fail however
the energy is measured. `max_bases_per_job` splits the submission into several
jobs of at most that many qubit-wise commuting bases — never cutting a basis in
half, since the point of the grouping is that one circuit answers for its whole
group:

```python
QiskitProvider(device="ibm_fez", shots=4096, max_bases_per_job=250)
```

The same limit applies to an **energy**, not only the RDM operators:
`QiskitProvider.energies` splits the Hamiltonian into several weighted
observables of at most that many bases each (the identity riding with the
first piece), and `provider.jobs_per_energy(hamiltonian)` reports how many
submissions one energy costs — the number an on-hardware optimization's
run-budget preflight multiplies the optimizer's own
{meth}`~mandacaru.optimizers.optim.Optimizer.max_evaluations` by, before the
first job is queued. One `qwc_partition` is shared by the plan, this chunking
and a Braket submission, so the bases a plan reports are exactly the bases
that get measured. Completed chunks of an unfinished submission are **kept**,
keyed by what was asked for, so a `remeasure()` after a partial failure
submits only the rest.

The ansatz is transpiled once and shared, results accumulate as each job
returns, and `[PERFORMANCE]` sums the QPU accounting over every job rather than
reporting the last one. The numbers are identical to the unsplit submission.

(run-budget)=
## The run-wide budget refuses before it queues

A processor -- an `ibm_*` device, an AWS Braket QPU or managed simulator --
also carries a **run-wide** budget: 250 jobs by default, with no default
ceiling on total shots (each job's size is already the measurement budget's
to bound; a `total_shots` limit can be set), off on a local or fake device
unless asked for. It is passed as
`run_budget=` to a provider (`backend_options={"run_budget": {"jobs": 500}}`
for a driver); `False` switches it off, and a dict overrides only the keys it
names.

An optimization whose `optimizer.max_evaluations() * provider.jobs_per_energy
(hamiltonian)` would need more jobs than the budget (or, with a `total_shots`
limit, more shots than `provider.shots_per_energy(hamiltonian)` allows) is
refused up front,
before anything is queued, naming that count against the limit. A
shot-based ADAPT-VQE's gradient steps are counted the same way: one step's
jobs and shots are checked before the run starts and again before each step
([below](#adapt-on-a-device)). A submission
that would only exceed it partway through the run is refused at that job
instead (`RunBudgetExceeded`), and whatever the run had already reached -- the
optimizer's best point, the checkpoint -- is kept. `provider.run_usage()`
reports the jobs and shots spent so far in the run (`run_jobs` lists every
job handle), and it survives `with_options(...)` -- what `remeasure()`
rebuilds through -- so a retry belongs to the same run and the same budget.
`[PERFORMANCE]` reports the cumulative count as `qpu_submissions` for a run
whose whole optimization was on hardware.

## Odd-`Y` strings are exactly zero

A Pauli string with an odd number of `Y` is $i^{\text{odd}}$ times a real
*antisymmetric* matrix, so for a real state $\psi$

$$\langle\psi|P|\psi\rangle = i^{\text{odd}}\,\psi^{T} R\, \psi = 0$$

**exactly** — an identity, not an approximation. Mandacaru checks the premise
rather than assuming it (the Hamiltonian must be a real matrix, and so must
every ansatz generator; a molecule's real harmonics guarantee the first,
and a spin-orbit term would break it) and then sets those
expectations to zero without measuring them. On LiH/PAW-LCAO-DZ that is 472 of 981
labels: the RDM job halves, and the energy, the RDMs and the forces come out
bit-for-bit identical.

(double-factorization)=
## Double factorization: `O(M)` bases instead of `O(M³)`

Everything above is a constant factor. Removing the RDMs divides the observables
by eight, which buys about one basis-size step against a cost growing as $M^3$.
The exponent itself only moves by changing *how* the energy is measured.

Following Motta *et al.* (npj Quantum Inf. **7**, 83, 2021) and Huggins *et al.*
(npj Quantum Inf. **7**, 23, 2021), the two-body operator can be written as a sum
of squares of one-body operators, each diagonal in an orbital basis of its own:

$$H = E_0 + \sum_{pq}\tilde h_{pq}a^\dagger_p a_q
      + \tfrac12\sum_\ell \lambda_\ell
        \Bigl(\sum_k g_{\ell k}\,\tilde n^{(\ell)}_k\Bigr)^2 .$$

Every occupation within one leaf commutes with every other, so **one** circuit
per leaf — the basis rotation, then a computational-basis measurement of the
whole register — yields all of that leaf's correlators at once.

Turn it on with `measurement_scheme`:

```python
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis={"name": "PAW-LCAO", "size": "DZ"},
                       grid_spacing=0.25,
                       pool="ceo-ovp",
                       measurement_scheme="double-factorized",
                       measurement_provider=QiskitProvider(device="ibm_fez",
                                                           shots=4096))
```

Each rotation $U_\ell$ is realized by a **Givens network**: any real orthogonal
$M \times M$ matrix is a product of at most $M(M-1)/2$ rotations on *adjacent*
modes, and under Jordan-Wigner each of those is a two-qubit number-conserving
gate on neighboring wires (Clements *et al.*, Optica **3**, 1460, 2016;
Kivlichan *et al.*, PRL **120**, 110501, 2018). After the rotation every
occupation is diagonal, so each leaf is one **diagonal** observable in `I` and
`Z` — `L + 1` PUBs, one job, whatever the register size.

It is exact: on H₂ and LiH the measured energy reproduces the state vector's to
**4e-13 eV**, and the factorization itself maps back to the original Hamiltonian
to machine zero.

### What it actually trades

Fewer bases, but a **larger 1-norm** — the squares expand — and a Givens network
on top of the ansatz. On the LiH/PAW-LCAO series, with the naive shot model
(total shots $\propto$ bases $\times \lambda^2$):

| spin orbitals | QWC bases | DF bases | λ(Pauli) → λ(DF) | total shots |
|---|---|---|---|---|
| 4 | 21 | 4 (5.2× fewer) | 1.09 → 6.34 Ha (5.8×) | ×6.5 **worse** |
| 8 | 93 | 11 (8.5×) | 5.50 → 15.93 Ha (2.9×) | ×1.0 break-even |
| 20 | 1,600 | 55 (29.1×) | 61.41 → 106.32 Ha (1.7×) | **×0.10** |
| 24 | 3,290 | 73 (45.1×) | 95.58 → 143.48 Ha (1.5×) | **×0.05** |

**The crossover is around eight qubits.** Below it the qubit-wise path is
cheaper and the factorized one is a needless extra circuit; above it the basis
count wins and keeps winning, because bases grow as $M^3$ while the 1-norm
penalty *shrinks* with size (5.8× at four spin orbitals, 1.5× at twenty-four).
At the 24-qubit case that started all this, the factorized measurement is a
**twentyfold** reduction in total shots.

The plan reports the factorized 1-norm under its own name so the shot cost is
never understated, and the Givens gates show up in `isa_two_qubit_gates`
(LiH/HAO: 192 → 244, a 27 % longer circuit for 4.3× fewer bases).

```{note}
Jordan-Wigner only — the scheme reads occupations off the computational basis,
and a `parity_reduced` register has no such picture. It also needs the molecular
integrals, so a direct-mode problem or a loaded Hamiltonian is refused by name
rather than silently falling back.
```

(adapt-on-a-device)=
## ADAPT-VQE on a device: the pool gradients are measured

**The device decides, and with `shots > 0` nothing is simulated.** Every
quantity a shot-based ADAPT-VQE uses comes from a measurement or from
polynomial classical algebra: the reference energy from the determinant's
`I`/`Z` strings, every energy from the Estimator, and every pool gradient as
the expectation value of a commutator,

$$g_k = \langle\psi|[H, A_k]|\psi\rangle = \sum_R c_{kR}\,\langle\psi|R|\psi\rangle ,$$

whose strings $R = PQ$ and real coefficients come from the Pauli algebra on
bit tables ({mod}`~mandacaru.backends.gradient_measurement`). No `2^n`
array, no sector, no pool matrix, no dense `eigh` and no `expm_multiply` is
built anywhere on the path, on any device (`ibm_*`, `fake_*`, Braket, or the
local shot-based estimator). The `shots=0` local simulation keeps its exact
state-vector route unchanged.

* **The first step is classical and exact.** At the Hartree-Fock determinant
  $\langle b|R|b\rangle$ is $\pm1$ for an `I`/`Z` string and 0 otherwise, so
  only the products $PQ$ that share their flip pattern contribute: no
  measurement at all.
* **After it, the screened operators' commutators are merged.** They are
  formed in batches (at most a million strings before merging), a string
  shared by several operators -- or by two batches -- is measured once, and
  the strings are grouped into qubit-wise-commuting bases with the same
  greedy rule as an energy. One basis is one circuit.
* **The bases are packed into jobs** within `measurement_budget=` (10,000
  bases and 50,000 strings per job by default) and the provider's
  `max_bases_per_job`; every job goes through the provider's run ledger.
* **The cost is planned before anything is queued.** It does not depend on
  the state, so it is computed once, logged as `[GRADIENT MEASUREMENT]`
  ({doc}`run_output`), and checked before the run starts and before every
  step against the run budget (jobs and shots) and the measurement budget:
  an oversized step is refused with its numbers, and a step the run budget
  stops ends the growth with the ansatz grown so far.
* **Shot noise is reported, not hidden.** Each gradient comes with its
  standard error, and `gradient_noise_bound_Ha` bounds it at
  $\max_k\|c_k\|_2/\sqrt{\text{shots}}$. The largest of many noisy
  gradients sits above that bound, so a gradient criterion below it cannot be
  met: use the energy criterion, or more shots.

Validated against the state vector: with an exact device the measured run
takes the local run's every operator and every energy (H₂, LiH and H₂O in
HAO, fermionic, QEB and CEO pools, energies to 1e-9 eV); measured gradients
equal `2 Re<H psi|A psi>` to 1e-10 and, sampled, lie within their standard
errors; at 50 qubits a stabilizer simulation of the circuit at Clifford
angles gives the merged, batched gradients to 1e-10 of each commutator
measured alone.

### What a gradient step costs

One growth step's gradient measurement, merged, against one energy (1,000
shots per basis, the default measurement budget):

| system (PAW-LCAO, `grid_spacing=0.3`) | qubits | pool | H strings | energy bases | step strings | step bases | energies | jobs | shots |
|---|---|---|---|---|---|---|---|---|---|
| water DZ, CAS(8, 7) by MP2 | 14 | `fermionic`, 204 | 902 | 202 | 93,641 | 15,911 | 79 | 3 | 1.6e7 |
| | | `qeb`, 204 | | | 345,200 | 31,185 | 154 | 8 | 3.1e7 |
| | | `ceo-ovp`, 1,344 | | | 892,845 | 76,851 | 380 | 19 | 7.7e7 |
| methanol DZ, CAS(10, 10) by MP2 | 20 | `fermionic`, 875 | 7,279 | 1,676 | 924,530 | 229,641 | 137 | 34 | 2.3e8 |
| dense random integrals, CAS(10, 25) | 50 | `fermionic`, 14,000 | 571,876 | 116,419 | one operator alone: 602,208 | 187,306 | 1.6 | 21 | 1.9e8 |
| neighbor-only integrals, CAS(2, 25) (the test's) | 50 | `fermionic`, 624 | 548 | 14 | 177,570 | 60,525 | 4,300 | 7 | 6.1e7 |

The first step costs nothing (it is classical). An *energy*, for comparison,
is the "energy bases" column times the shots. On dense synthetic integrals at
8-20 qubits the merged step is 8 to 66 energies (0.10-0.39 energies per pool
operator); `qeb` costs about twice `fermionic` (its strings carry no `Z`
chains, so fewer are shared), and the qubit pool measures exactly the
fermionic pool's strings (191,300 against 191,240 at 16 qubits), so its whole
pool costs the same. The classical side stays small: the methanol plan peaks
at 73 MB of batch memory, the 50-qubit one-operator plan at 1.7 GB of RSS, most
of it the Hamiltonian itself.

The merging is what makes this polynomial *and* about a few tenths of an
energy per pool operator: measured operator by operator, the same steps need
10-25 times more bases. **The full fermionic pool at 50 qubits is
unaffordable even merged**: one operator's gradient alone is 602,208 strings in 187,306
bases (21 jobs, 1.6 energies), and the 14,000-operator step extrapolates to
2.6-6.4e8 bases -- 2,200 to 5,500 energies, 2.6-6.4e11 shots and 26,000 to
64,000 jobs at 1,000 shots per basis -- while a single energy of that
Hamiltonian is already 1.2e8 shots. So `screening="auto"` refuses it (not
one operator fits a job) rather than truncating it silently, and the
measurement wall, not memory, is what bounds a 50-qubit ADAPT-VQE: shrink
the active space, use a ranked subset (`screening={"operators": K}`) knowing
it is one, or wait for the cheaper groupings below.

Each step is O(N) to O(N²) energies here; the grouping of Anastasiou *et
al.* (arXiv:2306.03227), which pivots on a Hamiltonian term and measures the
resulting commuting (not qubit-wise) sets with at most $N-3$ extra CNOTs,
reaches about $4N$ energies for the qubit pool and is the next step; it is
not implemented.

### Screening: which operators a step measures

`screening=` decides it:

* `"full"`: every operator, in as many jobs as the measurement budget makes of
  it;
* `{"operators": K}`: the `K` operators with the largest gradients at the
  reference determinant (exact, classical), a fixed candidate set;
* `"auto"` (the default): every operator on a local simulation; with
  `shots > 0` every operator whose step fits **one job** of the measurement
  budget, else the longest prefix of that ranking that does -- with a
  `RuntimeWarning` and the reason in `[GRADIENT MEASUREMENT]`'s
  `screened_subset`. If not even one operator fits a job (a single 50-qubit
  commutator of a dense molecular Hamiltonian is millions of strings), the
  run is refused with the numbers, before anything is queued.

The ranking is a truncation of the pool and is reported as one: an operator
outside the candidate set reads a zero gradient and is never selected. A
coupled-exchange (`ceo`) growth that expands into its members measures their
gradients in one more small job.

### Rehearsing above 24 qubits

A fake backend without `qiskit-aer` falls back to Qiskit's `BasicSimulator`,
which simulates at most 24 qubits -- and a circuit transpiled to a 156-qubit
processor (`fake_kingston`, `fake_fez`, `fake_marrakesh`) spans all 156 wires,
so without `qiskit-aer` those three cannot run even H₂; the five-qubit
`fake_manila` can (`examples/ibm_quantum/05_adapt_h2_measured_gradients.py`).
`qiskit-aer` (not a dependency) removes the first limit, and its
matrix-product-state method could reach 50 qubits for low-entanglement
circuits. What is checked at 50
qubits instead, all polynomial: a device double that answers every request
by arithmetic (the test suite's 50-qubit ADAPT-VQE and VQE, with every
exponential constructor made to raise; the ADAPT run peaks at 24 MiB of
traced memory), a stabilizer simulation of the actual circuit at Clifford
angles (`qiskit.quantum_info.StabilizerState`), and the reference
determinant, where every gradient is exact. Rehearse with the same options
on a fake backend before any submission.

## What to reach for

* **Too big to submit** → drop the forces (an energy is one observable), lower
  `resilience_level`, set `max_bases_per_job`.
* **Too expensive in shots** → a smaller `size`, `mapping="parity_reduced"`,
  `active_space={"frozen": "auto"}`. Above about eight qubits,
  `measurement_scheme="double-factorized"` is the structural answer.
* **Too deep to trust** → `pool="ceo-ovp"` (about half the CNOTs of `qeb`),
  `tetris=True`, `prune=True`, and pin the layout with `physical_qubits=`.
* **A shot-based ADAPT-VQE step too large** → `screening={"operators": K}`,
  a smaller pool (`qeb` costs about twice `fermionic`, the generalized `ceo`
  pools more), a smaller active space; the dry run estimates the step and
  the run logs its exact plan.
* **Too many jobs for the run budget** → fewer evaluations
  (`optimizer={"maxiter": ...}`), `measurement_provider=` instead of
  `device="ibm_*"` (optimize locally, measure once), or raise `run_budget=`
  in `backend_options=` if the run really needs it — see
  [above](#run-budget).
* **On the Open plan**, the realistic target stays 4–8 qubits. `examples/ibm_quantum/01_vqe_h2_ibm.py`
  is the shape that works: two qubits, five Pauli terms, four CNOTs.
