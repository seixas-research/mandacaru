# Running on Amazon Braket (and real QPUs)

The `"braket"` provider can target the **AWS Braket service** — the managed
simulators (SV1, DM1, TN1) and the real QPUs (IonQ, IQM, Rigetti) — not just the
local simulator.

Doing so requires one change of protocol, because of a hard constraint:

:::{important}
**A QPU never returns a state vector.** Braket rejects the `StateVector` result
type whenever `shots > 0`, and *every* QPU requires `shots > 0`. The exact
state-vector path that makes simulation fast therefore **cannot run on
hardware**.

The energy has to be *measured* instead:

```{math}
\langle H\rangle = \sum_j c_j \langle P_j\rangle,
```

with each $\langle P_j\rangle$ estimated from shots in that Pauli's own
eigenbasis.
:::

Carcará implements that path. Pass `shots > 0` and the calculator switches from
amplitudes to measurements automatically.

---

## Quick start

```python
from ase import Atoms
from carcara.algorithms import Carcara

atoms = Atoms("H2", positions=[[3, 3, 2.63], [3, 3, 3.37]],
              cell=[[6, 0, 0], [0, 6, 0], [0, 0, 6]], pbc=True)

# Local Braket simulator, shot-based (identical protocol to a QPU).
atoms.calc = Carcara(method="vqe",
                     basis="FAO",
                     h=0.35,
                     device="braket-local",
                     shots=8192)
atoms.get_total_energy()

# The AWS managed simulator.
atoms.calc = Carcara(method="vqe",
                     basis="FAO",
                     h=0.35,
                     device="braket-sv1",
                     shots=8192)

# A real trapped-ion QPU.
atoms.calc = Carcara(method="vqe",
                     basis="FAO",
                     h=0.35,
                     device="braket-ionq-aria",
                     shots=8192)
```

Naming a Braket device selects the `braket` provider automatically, so
`backend_provider` is optional. Anything beyond `device=` and `shots=` is
unchanged — the whole calculator API is identical.

:::{warning}
AWS devices need configured credentials (`aws configure`) and **bill your
account** per quantum task. Estimate the cost first (see *Cost* below).
:::

---

## How the energy is measured

Measuring one Pauli term per circuit is correct but ruinous — a modest active
space has $10^2$–$10^4$ terms. Carcará instead partitions the Hamiltonian into
**qubit-wise commuting (QWC)** groups: two Pauli strings are QWC when, on every
qubit where both act non-trivially, they carry the *same* Pauli. A QWC set is
measurable by a single circuit — rotate each qubit once into the basis its group
prescribes, measure everything, and read every term's expectation value out of
the same bit-strings.

```python
from carcara.backends.measurement import qubit_wise_commuting_groups

groups, identity = qubit_wise_commuting_groups(hamiltonian)
len(groups)      # circuits per energy evaluation
```

On LiH this collapses **118 Pauli terms into 29 measurement circuits**; on H₂,
14 terms into 5. The identity term needs no measurement and is added as a
constant.

The estimate converges as $1/\sqrt{\text{shots}}$:

```text
   shots          E (eV)   error (eV)   1-sigma (eV)
     500    -25.530589    -1.08e-02      1.80e+00
    5000    -25.585892    -6.61e-02      5.69e-01
   50000    -25.498090    +2.17e-02      1.80e-01
   exact    -25.519749
```

---

## IBM Quantum hardware (Qiskit Runtime)

On IBM processors the energy is the expectation value returned by the Qiskit
Runtime **Estimator**: the Hamiltonian becomes a `SparsePauliOp` observable,
the ansatz circuit is transpiled to the processor and the observable mapped to
its layout, and one job returns $\langle H\rangle$ with its standard error.
Three device spellings cover the workflow, from rehearsal to the real machine:

```python
from carcara.backends.providers import QiskitProvider

QiskitProvider(shots=4096)                          # local estimator, sampled
QiskitProvider(device="fake_kingston", shots=4096)  # that processor's fake backend, locally
QiskitProvider(device="ibm_kingston,ibm_fez,ibm_marrakesh", shots=4096)  # least busy of these
```

Credentials come from `QiskitRuntimeService.save_account(...)` run once, or
from `QiskitProvider(instance=..., token=..., channel=...)`.

**Optimise locally, measure once.** QPU time is scarce (the open plan gives
ten minutes a month), and a variational optimisation needs hundreds of energy
evaluations. So run the optimisation on the local state vector and measure
only the optimised states on hardware:

```python
from carcara.algorithms.base import measure_energies

atoms.calc = Carcara(method="adapt-vqe",
                     pool="ceo",
                     basis="GTO")
atoms.get_total_energy()                                  # local
provider = QiskitProvider(device="ibm_kingston", shots=4096)
e_hw = atoms.calc.solver.measured_energy(provider)        # one job
e_curve = measure_energies(solvers, provider)             # many geometries, one job
```

**Keep the circuit small.** Hardware noise, not shot noise, limited the first
LiH and H₂ runs (14 eV off for 92 CZ gates; a few tenths of an eV for 17). For
molecules with a closed-shell reference the parity mapping's two-qubit
reduction removes two qubits and their gates at no cost in physics:

```python
Carcara(method="adapt-vqe",
        pool="fermionic",
        mapping="parity",
        two_qubit_reduction=True,
        basis="FAO")     # H2 on 2 qubits
```

A driver with `shots > 0` and an IBM device (`Carcara(..., device="ibm_kingston",
shots=4096)`) runs the whole optimisation through the Estimator instead, one
job per energy evaluation; do that on a fake backend, not on a budget.
Example `24_ADAPTVQE_LiH_IBM.py` follows the optimise-locally pattern.

## Registered devices

```python
from carcara.backends.hardware import describe_devices, device_arn

for device in describe_devices():
    print(device.name, device.simulator, device.arn)
```

| device | kind | shots | ARN |
|---|---|---|---|
| `braket-local` | simulator | optional | *(local)* |
| `braket-sv1` | simulator | optional | `arn:aws:braket:::device/quantum-simulator/amazon/sv1` |
| `braket-dm1` | simulator | optional | `…/quantum-simulator/amazon/dm1` |
| `braket-tn1` | simulator | optional | `…/quantum-simulator/amazon/tn1` |
| `braket-ionq-aria` | QPU | **required** | `arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1` |
| `braket-ionq-forte` | QPU | **required** | `…/qpu/ionq/Forte-1` |
| `braket-iqm-garnet` | QPU | **required** | `arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet` |
| `braket-rigetti-ankaa` | QPU | **required** | `arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3` |

A **raw ARN** is accepted too, so a device released after this version can still
be named:

```python
Carcara(method="vqe",
        basis="FAO",
        device="arn:aws:braket:eu-west-2::device/qpu/vendor/New-1",
        shots=4096)
```

Naming a QPU **without** `shots` is refused up front, when the solver is built,
rather than at submission time:

```pycon
>>> Carcara(method="adapt-vqe",
...         device="braket-ionq-aria").run()
ValueError: device 'braket-ionq-aria' is real quantum hardware, which cannot
return a state vector: pass shots > 0 (e.g. shots=8192) so the energy is
estimated from measurements.
```

---

## Gate set

Carcará emits only `X`, `H`, `S`, `Si`, `CNot` and `Rz` — all Braket-native and
available on every Braket QPU (the device's own compiler maps them to its native
basis). `examples/13_braket_aws_compatibility.py` verifies this on every run.

---

## Cost

A hardware run is billed per quantum task, and one energy evaluation costs one
task per QWC group. Plan before submitting:

```python
from carcara.backends.measurement import shot_noise_estimate
from carcara.backends.providers import build_provider
from carcara.units import from_hartree

provider = build_provider("braket", shots=8192)
groups = provider.measurement_groups(hamiltonian)

print(f"{len(groups)} tasks per energy evaluation")
# The estimate is in the Hamiltonian's own unit (Hartree); report it in eV.
print(f"+/- {from_hartree(shot_noise_estimate(hamiltonian, 8192), 'eV'):.2e} eV "
      "at 8192 shots")
```

The standard error is bounded by $\left(\sum_j |c_j|\right)/\sqrt{\text{shots}}$
— the coefficient 1-norm estimate. For H₂ that 1-norm is 40 eV (1.48 Ha), so
chemical accuracy (0.043 eV = 1.6 mHa) needs $\sim\!8.5\times10^5$ shots per
group in the worst case.
That bound is *why* hardware VQE needs error mitigation and smarter estimators;
it is not a defect of the implementation.

---

## Current limitation

:::{admonition} Only the energy evaluation is hardware-native
:class: caution

The energy runs on the device, but **ADAPT-VQE's pool-gradient screening is
still computed classically** from the state vector. A fully hardware-native
adaptive loop would have to measure each pool gradient as well; that is not
implemented yet.

For a fixed ansatz, `method="vqe"` is fully hardware-native
today — every cost evaluation in the optimisation is measured on the device.
:::

---

## Verifying compatibility

`examples/13_braket_aws_compatibility.py` runs the full check locally — no AWS
account, no charges — and prints a report covering the gate set, the
shots-versus-state-vector constraint, QWC grouping, shot-noise convergence,
per-evaluation task count, and the registered devices:

```bash
python examples/13_braket_aws_compatibility.py
```
