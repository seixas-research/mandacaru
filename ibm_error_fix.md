# IBM Runtime error 1336 (out of memory) — analysis and proposed fix

Analysis of `LiH/scanning/ibm.log` (failed `run_ibm_test.py`, LiH d = 1.60 Å,
PAW-TZP, 24 qubits, `ibm_fez`, `resilience_level=2`, 4096 shots). Nothing in
the repository was changed; every modification below is a proposal.

> Note: `run_ibm_test.py` now says `"size": "DZ"`, but it was edited at 00:55,
> after the failing job was submitted (00:52). The job in `ibm.log` is the TZP
> one — its local log is `output_ibm_test_PAW-TZP-0.1_1.60.txt` (finished 00:51).

## 1. What failed

```
mandacaru/algorithms/calculator.py  _measure            -> provider.expectation_values(...)
mandacaru/backends/providers.py     expectation_values  -> self._run_pubs([(qc, observables)])
mandacaru/backends/providers.py     _run_pubs           -> self.last_job.result()
RuntimeJobFailureError: Error code 1336; Program runtime ran out of memory
```

The local ADAPT-VQE converged normally (75 operators, −21.7019 eV). The memory
was exhausted **on IBM's side**, inside the Estimator program that expands the
PUB into circuits and post-processes the results — not on `rubi`. Local memory
is not the issue, so the fix is about *what Mandacaru submits*.

## 2. Root cause: the job is ~8× larger than the energy needs

`Mandacaru._measure` (`calculator.py:850`) always measures the union of the
Hamiltonian's Pauli strings **and every spin-conserving 1- and 2-RDM
operator**, as one PUB holding one single-Pauli `SparsePauliOp` per label
(`providers.py:606`). It does so even when only the energy was requested —
`run_ibm_test.py` calls `get_potential_energy()` only, so the RDMs are never
used.

Measured for this exact system (reproduced locally, scripts in the session
scratchpad):

| quantity | 8 qubits (DZ) | 24 qubits (TZP) |
|---|---:|---:|
| Hamiltonian Pauli terms | 361 | 12,737 |
| RDM operators built (`ones` + `twos`) | 32 + 1,312 | 288 + 117,792 |
| **labels actually submitted (H ∪ RDM)** | **980** | **97,980** |
| QWC measurement bases, H only (greedy) | 80 | 3,067 |
| QWC measurement bases, H ∪ RDM (greedy) | — | 21,153 |
| time to build the RDM operators locally | 0.2 s | 49 s |

So the failed job asked for **97,980 observables ≈ 21,000 measurement bases**
of a circuit that transpiles (fake_fez, `optimization_level=3`) to **2,223 CZ,
2q-depth 1,418, ~8,800 1q gates**. `resilience_level=2` multiplies that again:
ZNE (default noise factors 1, 3, 5 → the folded circuits carry ~3× and ~5× the
gates) and gate twirling (default 32 randomizations). Order of magnitude:

```
21,153 bases x 3 noise factors x 32 twirls  ~ 2 x 10^6 circuit instances,
each 10^4 - 5 x 10^4 gates, plus per-observable ZNE result arrays for 97,980 observables
```

That is what ran out of memory. I cannot see which internal stage of the
Runtime program hit the limit, but every factor above is set by Mandacaru's
submission, and the label count is the one that is both largest and avoidable.

`MAX_PAULI_RDM_MODES` (12 upstream, raised to 32 in site-packages, 25 in the
working tree) was the guard against exactly this; the `O(n^4)` comment next to
it is the scaling seen in the table. Raising it removed the only check.

## 3. Proposed modifications (ordered by memory saved per line changed)

### 3.1 Do not measure RDMs unless something needs them — `calculator.py`

The single biggest saving: 97,980 → 12,736 observables (7.7×) and 21,153 →
3,067 bases (6.9×), and it skips the 49 s RDM-operator build. It also makes the
`MAX_PAULI_RDM_MODES` limit irrelevant for energy-only runs, so the
site-packages patch is no longer needed.

In `calculate()` (`calculator.py:739-743`):

```python
if self.measurement_provider is not None and not solver.dry_run:
    t0 = _perf()
    measured = self._measure(solver, rdms=want_forces)
    ...
```

In `_measure`:

```python
def _measure(self, solver, rdms: bool = True):
    ...
    hamiltonian = solver.hamiltonian
    provider = self.measurement_provider
    if not rdms:
        # Energy only: <H> as ONE observable (see 3.2); no RDM operators built.
        energy = provider.energy(*solver.ansatz_problem()[:4], hamiltonian)
        job = getattr(provider, "last_job", None)
        self.measurement = {
            "energy_hartree": energy, "energy_eV": energy * HARTREE_TO_EV,
            "rdms": None, "expectation_values": None, "stds": None,
            "job_id": job.job_id() if job is not None else None}
        return self.measurement
    ...  # existing H ∪ RDM path, unchanged
```

`_log_measurement` already tolerates missing `stds` / `expectation_values`
(`measured.get(...) or {}`). `_forces(..., rdms=measured["rdms"])` is only
reached when `want_forces` is true, i.e. when `rdms=True` was used. Population
analysis reads the local state vector (`_state_rdms`), not `self.measurement`,
so it is unaffected. Update the `measurement_provider` docstring
(`calculator.py:289-297`): "RDM operators are measured only when forces are
requested".

### 3.2 Energy as one `SparsePauliOp`, not N single-Pauli observables — `providers.py`

`QiskitProvider.energies()` / `pub()` already do this correctly
(`observable = hamiltonian.to_sparse_pauli_op()`); only `expectation_values`
uses the one-observable-per-label form. With an observables array of length N
the Runtime returns (and, under ZNE, extrapolates) `evs`, `stds`,
`evs_noise_factors`, `evs_extrapolated`, … **per array element**; with a single
weighted observable it groups the terms itself and returns one scalar. 3.1 as
written routes the energy-only path through `provider.energy`, so this comes
for free.

Verified locally on the DZ case: `provider.energies([...])` reproduces the VQE
energy to 2e-12 eV. (The same check on 24 qubits with the local
`StatevectorEstimator` was started but is slow — 12,736 expectations on 2^24
amplitudes — and had not finished when this was written.)

**Workaround available today, without touching the package** — drop
`measurement_provider=` from the calculator and measure explicitly:

```python
atoms.calc = Mandacaru(method="adapt-vqe", basis=basis, ...)   # no measurement_provider
atoms.get_potential_energy()                                    # local optimization
s = atoms.calc.solver
e_ha = measurement_provider.energy(*s.ansatz_problem()[:4], s.hamiltonian)
print(e_ha * HARTREE_TO_EV)        # from mandacaru.units import HARTREE_TO_EV
```

### 3.3 Bound the size of any one job: chunk by measurement basis — `providers.py`

For the forces path (RDMs really needed) one PUB of ~10^5 observables will
still fail. Split by QWC group with the partition Mandacaru already has
(`backends.measurement.qubit_wise_commuting_groups`), so each job has a known,
bounded number of bases:

```python
def expectation_values(self, n_qubits, occupied, generators, thetas, labels,
                       max_bases_per_job: int | None = None):
    labels = list(labels)
    qc, layout = self._transpiled(self.build(...), n_qubits)   # transpile ONCE
    limit = max_bases_per_job or self.max_bases_per_job        # new ctor arg, e.g. 250
    values, stds, self.jobs = {}, {}, []
    for chunk in _chunks_by_basis(labels, limit):              # QWC groups -> chunks
        obs = [SparsePauliOp(l) for l in chunk]
        if layout is not None:
            obs = [o.apply_layout(layout) for o in obs]
        result = self._run_pubs([(qc, obs)])
        self.jobs.append(self.last_job)
        ...  # fill values / stds
    return values, stds
```

Notes:
- `qubit_wise_commuting_groups` is a pure-Python O(N·G) loop; for ~10^5 labels
  vectorize it (labels as a `uint8` array, one merged-basis row per group — the
  NumPy version used for the table above groups 97,980 labels in 42 s).
- `qpu_usage` (`providers.py:900`) reads only `last_job`; it should sum over
  `provider.jobs`.
- Persist partial results per chunk (job ids + values) so a late failure does
  not discard the QPU time already spent — same reasoning as the per-point CSV
  write in `run_mandacaru_ibm.py`.

### 3.4 Fewer labels when RDMs *are* needed — `rdm.py`

- **Skip the odd-Y strings when the state is real.** This Hamiltonian is real
  (max |Im c| = 0, no odd-Y term), and the QEB/HF ansatz then gives real
  amplitudes, for which every Pauli string with an odd number of `Y` has zero
  expectation. That halves the RDM job exactly: 97,980 → 49,140 labels,
  21,153 → 10,634 bases. Set those expectations to 0 instead of measuring them
  (`rdms_from_expectations` can use `expectations.get(label, 0.0)`). Gate it on
  "Hamiltonian has no odd-Y term and the pool generators are real"; complex
  orbitals (l > 0 shells at lower symmetry) would break the assumption.
- **Build only the independent RDM elements.** `rdm_qubit_operators` maps all
  117,792 ordered `(p,q,r,s)`; `Γ_pqrs = −Γ_qprs = −Γ_pqsr = Γ*_rspq` leaves
  ~1/8 of them. This does not change the label set (the permutations share
  their strings) but cuts the 49 s build and the dict of 117k `PauliSum`s.
- Longer term: forces only need `Σ Γ_pqrs ∂<pq|rs>` — i.e. the expectation of
  `∂H/∂R`, whose Pauli support is that of `H` (12.7k strings here), not the
  full 2-RDM. Measuring only strings with non-zero weight in `H` and `∂H/∂R_i`
  would bring the forces job down to roughly the size of the energy job.

Coefficient truncation of `H` is **not** a useful lever here: dropping
|c| < 1e-5 Ha removes only 440 of 12,736 terms (bound 0.06 eV), and 1e-4
already risks 2.6 eV.

### 3.5 Pre-flight check instead of a mode cap — `calculator.py` / `providers.py`

Replace the `MAX_PAULI_RDM_MODES` hard limit for the *hardware* path by an
estimate computed before `estimator.run`, logged in a `[MEASUREMENT PLAN]`
block and enforced with a clear error (overridable):

```
observables, measurement bases, jobs (after chunking)
ISA circuit: 2q gates, 2q depth
circuit instances ~ bases x len(zne.noise_factors) x twirling.num_randomizations
estimated QPU seconds vs. a user budget (qpu_budget_s=...)
```

Everything needed is local (`_transpiled`, the QWC grouping, the
`estimator_options`). It would have stopped this job before it queued for 37
minutes, and `Mandacaru.dry_run()` is the natural place to expose it.

### 3.6 Fail softly — `providers.py::_run_pubs` / `calculator.py::calculate`

The `RuntimeJobFailureError` discarded an 86 s converged optimization. Catch it
around `_measure`, keep `self.solver` and the local energy available, write the
job id and the measurement plan to the log, and re-raise a `RuntimeError` that
states the observable/basis/circuit counts. A `calc.remeasure()` that reuses the
stored ansatz would let the measurement be retried with other options without
re-optimizing.

## 4. Fixing the memory error does not make this particular job viable

Worth knowing before spending queue time on the TZP circuit:

- **Signal.** 2,223 CZ at ibm_fez's typical ~3e-3 CZ error gives a circuit
  fidelity of about e^(−6.7) ≈ 1e-3, before decoherence over a 2q-depth of
  1,418. ZNE folds that to 3× and 5×, where nothing is left to extrapolate. The
  TZP energy from hardware would be noise, with or without the OOM. The DZ
  circuit (456 CNOT before transpilation, 80 bases) is the realistic target;
  for deep circuits use `resilience_level=1` (TREX only), which also removes
  the 3× ZNE factor from the memory and QPU-time budget.
- **QPU time.** Even energy-only, 3,067 bases × 4,096 shots × 3 noise factors
  ≈ 3.8e7 shots — hours of QPU time against the Open plan's 10 minutes per 28
  days. The pre-flight estimate of 3.5 is what should catch this.

## 5. Summary

| change | file | effect on this job |
|---|---|---|
| measure RDMs only when forces are requested | `algorithms/calculator.py` | 97,980 → 12,736 observables; 21,153 → 3,067 bases; no 49 s build; no `MAX_PAULI_RDM_MODES` patch |
| energy as one `SparsePauliOp` (`provider.energy`) | `algorithms/calculator.py` | 1 result element instead of N; usable today as a workaround |
| chunk observables by QWC basis, several bounded jobs | `backends/providers.py`, `backends/measurement.py` | memory per job bounded for the forces path |
| drop odd-Y strings for real states; build only independent RDM elements | `algorithms/rdm.py` | RDM job halved (49,140 labels, 10,634 bases); build ~8× cheaper |
| pre-flight measurement plan + budget | `calculator.py`, `providers.py`, `dry_run` | refuses oversized jobs before queueing |
| catch job failure, keep the converged state, `remeasure()` | `providers.py`, `calculator.py` | no lost optimization |
