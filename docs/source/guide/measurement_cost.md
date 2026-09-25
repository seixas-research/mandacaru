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

Measured on LiH/PAW-LCAO at `h = 0.25`, `pool="ceo-ovp"`, Jordan-Wigner:

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
                       h=0.25,
                       pool="ceo-ovp",
                       measurement_provider=QiskitProvider(device="ibm_fez",
                                                           shots=4096))
atoms.get_potential_energy()   # 1 observable: <H> as one weighted operator
atoms.get_forces()             # the RDM operators, because these need them
```

An energy is now **one weighted `SparsePauliOp`**, so the Estimator groups the
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
    circuit_instances: 93
    total_shots: 380928
    hamiltonian_one_norm_Ha: 5.502732
    shot_noise_bound_eV: 2.339640
    double_factorized_bases: 11 (available: measurement_scheme='double-factorized')
    isa_two_qubit_gates: 96
    isa_depth: 169
    expected_fidelity: 7.494e-01 (at a nominal 0.003 two-qubit error)
```

`circuit_instances` is `bases × noise_factors × twirls`: `resilience_level=2`
multiplies the job by 96 before a single shot is counted, which is the factor
that is easiest to forget.

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
holds the plan of the attempt that failed.

## Bounded jobs instead of one unbounded PUB

When the RDMs really are needed, one job of $10^5$ observables will fail however
the energy is measured. `max_bases_per_job` splits the submission into several
jobs of at most that many qubit-wise commuting bases — never cutting a basis in
half, since the point of the grouping is that one circuit answers for its whole
group:

```python
QiskitProvider(device="ibm_fez", shots=4096, max_bases_per_job=250)
```

The ansatz is transpiled once and shared, results accumulate as each job
returns, and `[PERFORMANCE]` sums the QPU accounting over every job rather than
reporting the last one. The numbers are identical to the unsplit submission.

## Odd-`Y` strings are exactly zero

A Pauli string with an odd number of `Y` is $i^{\text{odd}}$ times a real
*antisymmetric* matrix, so for a real state $\psi$

$$\langle\psi|P|\psi\rangle = i^{\text{odd}}\,\psi^{T} R\, \psi = 0$$

**exactly** — an identity, not an approximation. Mandacaru checks the premise
rather than assuming it (the Hamiltonian must be a real matrix, and so must
every ansatz generator; a complex basis would break it) and then sets those
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
                       h=0.25,
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

## What to reach for

* **Too big to submit** → drop the forces (an energy is one observable), lower
  `resilience_level`, set `max_bases_per_job`.
* **Too expensive in shots** → a smaller `size`, `mapping="parity_reduced"`,
  `frozen_core`. Above about eight qubits,
  `measurement_scheme="double-factorized"` is the structural answer.
* **Too deep to trust** → `pool="ceo-ovp"` (about half the CNOTs of `qeb`),
  `tetris=True`, `prune=True`, and pin the layout with `physical_qubits=`.
* **On the Open plan**, the realistic target stays 4–8 qubits. `examples/25_ADAPTVQE_H2_IBM.py`
  is the shape that works: two qubits, five Pauli terms, four CNOTs.
