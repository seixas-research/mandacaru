# Devices and circuit providers: Qiskit, Braket and Cirq

Mandacaru's ansätze are products of exponentials of anti-Hermitian generators,

```{math}
|\psi(\vec\theta)\rangle = \prod_k e^{\theta_k A_k}\,|\mathrm{HF}\rangle .
```

Two options decide how those circuits are used. `backend_provider` chooses the
quantum SDK that **builds** them and, when they are executed, **runs** them.
`device` chooses **where** they run. `shots` chooses between an exact state
vector (`0`) and a measured energy (`> 0`). The options reference lists them all;
this page says how they combine.

## Providers

| `backend_provider` | package | aliases | executes on |
|---|---|---|---|
| `"qiskit"` | `qiskit` | `"ibm"`, `"ibmq"` | the local state-vector estimator, a fake backend, or IBM Runtime |
| `"braket"` | `amazon-braket-sdk` | `"aws"`, `"amazon-braket"` | the local simulator or the AWS Braket service |
| `"cirq"` | `cirq` | `"google"` | `cirq.Simulator`, exact only |

Naming a provider never fails at import time: the SDK is imported on first use.
`mandacaru.backends.provider_available("braket")` reports whether it is
importable, and `mandacaru.backends.BACKEND_PROVIDERS` lists the canonical names.
A device that belongs to one provider selects it, so `device="braket-local"` or
`device="fake_kingston"` needs no `backend_provider`; naming a provider that
disagrees with the device raises.

## Building versus executing

By default the energy is computed natively from the state vector, and circuits
are not run. `execute_circuits` decides whether the SDK actually prepares each
state:

* `execute_circuits=None` (the default) is **True for `"braket"` and `"cirq"`**,
  where naming the provider is a request to use it, and False for `"qiskit"`,
  which keeps the fast native numerics unless execution is asked for.
* Execution is also on whenever `shots > 0` or `device` is not the default
  `"AER_simulator"`.
* Execution forces the ordered-product form of the ansatz (`trotter=True`), since
  a circuit is an ordered product.

Circuit execution is slower and gives the same state to machine precision, so for
a simulator it is a verification path, and for a processor the only one.

```python
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("H2")
atoms.center(vacuum=3.0)
for execute in (False, True):
    atoms.calc = Mandacaru(method="vqe",
                           basis={"name": "PAW-LCAO", "size": "SZ"},
                           grid_spacing=0.25,
                           mapping="parity_reduced",
                           backend_provider="qiskit",
                           execute_circuits=execute)
    print(execute, f"{atoms.get_potential_energy():.6f} eV")
```

Every generator is a sum of **commuting** Pauli strings with real coefficients,
so `exp(theta A)` factorizes exactly into Pauli rotations, with no Trotter error.
This is checked rather than assumed: a generator with two anticommuting terms is
refused when its circuit is built. The three providers emit the same gate set
(`X`, `H`, `S`, `S†`, `CNOT`, `Rz`) and agree on the state to machine precision.

:::{note}
**Endianness.** Mandacaru puts qubit 0 in the most significant position of the
amplitude index, as Braket and Cirq do. Qiskit is little-endian, so the Qiskit
provider lays Mandacaru qubit `k` on Qiskit wire `n-1-k`. Gate counts are
unaffected.
:::

## Devices

`device` is one of the registered names, any `ibm_*` or `fake_*` name, or a raw
Braket ARN. `mandacaru.backends.available_devices()` lists the registered ones.

| device | kind | `shots > 0` |
|---|---|---|
| `"AER_simulator"` (default) | ideal, noiseless | optional |
| `fake_*` (for example `fake_kingston`, `fake_manila`) | a Qiskit Runtime fake backend: the processor's transpilation and noise model, run locally | **required** |
| `ibm_*` (for example `ibm_kingston`), or `"ibm-quantum"` for the least busy | IBM processor via Qiskit Runtime | **required** |
| `braket-local` | local Braket simulator | optional |
| `braket-sv1`, `braket-dm1`, `braket-tn1` | AWS managed simulators | optional |
| `braket-ionq-aria`, `braket-ionq-forte`, `braket-iqm-garnet`, `braket-rigetti-ankaa` | AWS QPUs | **required** |
| `arn:aws:braket...` | any other AWS device, assumed to be a QPU | **required** |

A device that requires shots, named with `shots=0`, is refused when the
calculator is constructed, not at submission. A comma-separated list of `ibm_*`
names picks the least busy of them.

**The device decides where the optimization runs.** A local device (the exact
state vector, `braket-local`, a `fake_*` backend) simulates locally. An `ibm_*`
device with `shots > 0` runs the *whole* optimization on the processor, one job
per energy evaluation, and an AWS device likewise one task per measurement basis.
Optimize locally and then re-measure is not offered. Rehearse the identical call
on a fake backend first; see [what a measurement costs](measurement_cost.md) for
the budgets that protect a real submission.

`backend_options` is a dictionary of provider constructor options.
`shots` and `device` are set by the calculator and need not be repeated.

* Qiskit: `instance`, `token`, `channel` (otherwise the account saved by
  `QiskitRuntimeService.save_account` is used), `optimization_level`,
  `estimator_options` (Runtime options), `physical_qubits`, `seed`,
  `max_bases_per_job`, `enable_fractional_gates` and `run_budget`.
* Braket: `s3_folder`, `poll_timeout_seconds` and `run_budget`.
* Cirq runs exactly and refuses `shots > 0`.

An option the provider does not take raises. Pin the layout with `physical_qubits` on a processor: a
qubit whose readout has drifted since the last calibration ruins an energy while
its recorded error still looks fine.

## Limits of circuit execution

* Subspace methods and the architecture searches (`mcas-vqe`, `valqa`) do not
  execute circuits and raise `NotImplementedError`.
* Forces are not an implemented property on a circuit device, and measured
  reduced density matrices are not available; see
  [what is not yet available](#not-yet-available).
* With `shots > 0`, the state retained on the result is the **ideal circuit
  state**, not a measured device state.
* A circuit can only be initialized in a computational-basis state, so the
  reference is a Slater determinant.

Running on Amazon Braket, including QPUs, is in [the Braket guide](aws_braket.md).
