# Running on Amazon Braket (and real QPUs)

The `"braket"` provider can target the **AWS Braket service**: the managed
simulators (SV1, DM1, TN1) and the real QPUs (IonQ, IQM, Rigetti), not just the
local simulator.

:::{important}
**A QPU never returns a state vector.** Braket rejects the `StateVector` result
type whenever `shots > 0`, and *every* QPU requires `shots > 0`. The exact
state-vector path that makes simulation fast therefore **cannot run on
hardware**. The energy has to be *measured* instead,

```{math}
\langle H\rangle = \sum_j c_j \langle P_j\rangle ,
```

with each $\langle P_j\rangle$ estimated from shots in that Pauli's own
eigenbasis.
:::

Pass `shots > 0` and the calculator switches from amplitudes to measurements.

## Quick start

The local simulator runs the same protocol as a QPU, with no account and no
charge. `examples/aws_braket/01_vqe_h2_braket_local.py` is this script:

```python
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("H2")
atoms.center(vacuum=3.0)
for shots in (0, 1000, 10000):
    atoms.calc = Mandacaru(method="vqe",
                           basis={"name": "PAW-LCAO", "size": "SZ"},
                           grid_spacing=0.25,
                           mapping="parity_reduced",
                           device="braket-local",
                           shots=shots,
                           optimizer={"method": "COBYLA", "maxiter": 60,
                                      "tol": 1e-4})
    energy = atoms.get_potential_energy()
    print(f"shots = {shots:6d}  E = {energy:.4f} eV")
```

`shots=0` reads the exact state vector, which only a simulator offers. To submit
to the managed simulator or a QPU, change only the device: `"braket-sv1"`,
`"braket-ionq-aria"`, and so on. Naming a Braket device selects the `braket`
provider, so `backend_provider` is optional, and the rest of the calculator is
unchanged.

:::{warning}
AWS devices need configured credentials (`aws configure`) and **bill your
account** per quantum task. Size the run first (see *Cost* below).
:::

## Registered devices

| device | kind | shots | ARN |
|---|---|---|---|
| `braket-local` | simulator | optional | *(local)* |
| `braket-sv1` | simulator | optional | `arn:aws:braket:::device/quantum-simulator/amazon/sv1` |
| `braket-dm1` | simulator | optional | `arn:aws:braket:::device/quantum-simulator/amazon/dm1` |
| `braket-tn1` | simulator | optional | `arn:aws:braket:::device/quantum-simulator/amazon/tn1` |
| `braket-ionq-aria` | QPU | **required** | `arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1` |
| `braket-ionq-forte` | QPU | **required** | `arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1` |
| `braket-iqm-garnet` | QPU | **required** | `arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet` |
| `braket-rigetti-ankaa` | QPU | **required** | `arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3` |

A **raw ARN** is accepted too, so a device released after this version can still
be named, and is treated as a QPU (shot-based and billed). Naming a QPU without
`shots` is refused when the calculator is constructed, with the message
`this device requires shots > 0`, rather than at submission time.

## How the energy is measured

Measuring one Pauli term per circuit is correct but ruinous: a modest active
space has $10^2$ to $10^4$ terms. Mandacaru partitions the Hamiltonian into
**qubit-wise commuting (QWC)** groups. Two Pauli strings are QWC when, on every
qubit where both act non-trivially, they carry the same Pauli, so a group is
measurable by one circuit: rotate each qubit once into the basis the group
prescribes, measure everything, and read every term's expectation value out of
the same bit-strings. The identity term needs no measurement.

One energy evaluation costs **one quantum task per group**, all queued before
any is awaited. For `method="adapt-vqe"` the same provider measures the pool
gradients as commutator expectation values, in the same grouped form; see
[ADAPT-VQE on a device](#adapt-on-a-device). The estimate converges as
$1/\sqrt{\text{shots}}$, and its standard error is bounded by
$\left(\sum_{P\neq I}|c_P|\right)/\sqrt{\text{shots}}$ per group. That bound is
why hardware VQE needs a small circuit and a noise-tolerant optimizer, not a
defect of the implementation.

Mandacaru emits only `X`, `H`, `S`, `S†`, `CNOT` and `Rz`, all Braket-native;
the device's own compiler maps them to its native basis.

## Cost

A hardware run is billed per quantum task. Size the grouping and the noise
bound locally, before submitting, as in
[What a hardware measurement costs](measurement_cost.md), whose grouping
function (`mandacaru.backends.qubit_wise_commuting_groups`) is the one the
provider uses. Then bound the run:

* A Braket QPU or managed simulator carries a **run-wide budget** by default:
  250 tasks, no shot ceiling. An optimization whose evaluation bound cannot fit
  it is refused before the first task is queued ([the run budget](#run-budget)).
  Change it with `backend_options={"run_budget": {...}}`.
* Use a noise-tolerant optimizer: SPSA (the default with `shots > 0`) or COBYLA.
  The gradient methods are refused ([choosing an optimizer](#shots-optimizer)).
* Keep the circuit small: `mapping="parity_reduced"` removes two qubits for a
  closed-shell molecule at no cost in physics.

Forces are not an implemented property on a circuit device, so a Braket run
gives energies; see [what is not yet available](#not-yet-available).
