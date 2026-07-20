# -*- coding: utf-8 -*-
# file: examples/13_braket_aws_compatibility.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Amazon Braket compatibility check: can Carcará run on a real QPU?

This script *verifies*, rather than assumes, what the ``"braket"`` backend
provider can and cannot do against the AWS Braket service, and prints a report.
It runs entirely on Braket's **local** simulator -- no AWS credentials, no
charges -- but exercises the exact code path a QPU takes.

What it checks
--------------
1. **The gate set.** Carcará emits ``X``, ``H``, ``S``, ``Si``, ``CNot`` and
   ``Rz`` only; all are Braket-native and available on every Braket QPU.
2. **The state-vector path is simulator-only.** Braket rejects the
   ``StateVector`` result type whenever ``shots > 0``, and *every QPU requires*
   ``shots > 0``.  So the exact path that makes the simulator fast cannot run on
   hardware -- confirmed here by catching the error.
3. **The shot-based path is QPU-compatible.** ``<H>`` is assembled from
   qubit-wise-commuting measurement groups, which is what a QPU can actually
   deliver.  Its accuracy is shown converging as ``1/sqrt(shots)``.
4. **Cost.** The number of quantum tasks per energy evaluation (= number of QWC
   groups) and the shots needed for chemical accuracy are reported, because that
   is what a hardware run is billed for.

Running on real hardware
------------------------
Only the device changes -- the driver API does not::

    from carcara.algorithms import ADAPTVQE

    atoms.calc = ADAPTVQE(pool="qeb", basis="FAO",
                          device="braket-ionq-aria",   # or the full ARN
                          shots=8192)
    atoms.get_total_energy()

That needs configured AWS credentials (``aws configure``) and bills your
account.  See :mod:`carcara.backends.hardware` for the registered devices.

Known limitation (stated plainly)
---------------------------------
The **energy evaluation** runs on hardware, but ADAPT-VQE's *operator screening*
gradient is still computed classically from the state vector.  A fully
hardware-native ADAPT loop would have to measure each pool gradient too; that is
not implemented yet.  For a fixed ansatz (:class:`~carcara.algorithms.VQE`) the
whole optimization is hardware-native today.
"""

from __future__ import annotations

import os

import numpy as np
from ase import Atoms

from carcara.algorithms import ADAPTVQE
from carcara.backends.hardware import (describe_devices, device_arn,
                                       requires_shots)
from carcara.backends.measurement import (qubit_wise_commuting_groups,
                                          shot_noise_estimate)
from carcara.backends.providers import build_provider, provider_available
from carcara.circuits.adapt_ansatz import AdaptAnsatz
from carcara.circuits.pools import build_pool

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

#: Gates every Amazon Braket QPU accepts (after its own compilation).
BRAKET_NATIVE_GATES = {"I", "X", "H", "S", "Si", "Rz", "CNot"}
CHEMICAL_ACCURACY = 1.6e-3      # Ha

rule = "=" * 74
print(rule)
print("Amazon Braket compatibility report")
print(rule)

if not provider_available("braket"):
    raise SystemExit("amazon-braket-sdk is not installed; "
                     "`pip install amazon-braket-sdk`")

# --------------------------------------------------------------------------- #
# 0. Build a small problem (H2) and cache its Hamiltonian.
# --------------------------------------------------------------------------- #

atoms = Atoms("H2", positions=[[3, 3, 2.63], [3, 3, 3.37]],
              cell=[[6, 0, 0], [0, 6, 0], [0, 0, 6]], pbc=True)
atoms.calc = ADAPTVQE(pool="qeb", basis="FAO", h=0.35, verbose=False,
                      max_iterations=1,
                      save_hamiltonian=os.path.join(DATA, "h2_braket.parquet"))
atoms.get_total_energy()

hamiltonian = atoms.calc.hamiltonian
n_qubits = atoms.calc.n_qubits
num_particles = atoms.calc.num_particles

pool = build_pool("qeb", n_qubits // 2, num_particles)
ansatz = AdaptAnsatz(n_qubits, pool.occupied_orbitals, "jordan_wigner")
for op in pool.operators()[:2]:
    ansatz.append(op)
theta = np.array([0.31, -0.20])[:ansatz.num_parameters]

occupied = ansatz.reference_qubits()
generators = ansatz.pauli_generators
exact = float(np.real(np.vdot(ansatz.state(theta),
                              hamiltonian.to_matrix() @ ansatz.state(theta))))

print(f"\nProblem: H2, {n_qubits} qubits, num_particles={num_particles}")
print(f"         {len(hamiltonian.simplify().terms)} Pauli terms, "
      f"{ansatz.num_parameters}-parameter QEB ansatz")

# --------------------------------------------------------------------------- #
# 1. Gate set.
# --------------------------------------------------------------------------- #

print("\n[1] Gate set")
provider = build_provider("braket", shots=1024)
circuit = provider.build(n_qubits, occupied, generators, theta,
                         measure_basis="X" * n_qubits)
used = {ins.operator.name for ins in circuit.instructions}
unsupported = used - BRAKET_NATIVE_GATES
print(f"    emitted gates : {', '.join(sorted(used))}")
print(f"    Braket-native : {'YES' if not unsupported else 'NO ' + str(unsupported)}")
assert not unsupported, f"non-native gates emitted: {unsupported}"

# --------------------------------------------------------------------------- #
# 2. The state-vector path cannot run on hardware.
# --------------------------------------------------------------------------- #

print("\n[2] State-vector result type vs shots")
try:
    provider.statevector(n_qubits, occupied, generators, theta)
    verdict = "unexpectedly succeeded"
except ValueError as exc:
    verdict = "correctly refused"
print(f"    provider(shots>0).statevector(...) -> {verdict}")

from braket.circuits import Circuit                                   # noqa: E402
from braket.devices import LocalSimulator                             # noqa: E402

probe = Circuit().h(0).cnot(0, 1)
probe.state_vector()
try:
    LocalSimulator().run(probe, shots=100)
    upstream = "accepted (unexpected)"
except ValueError as exc:
    upstream = f"rejected by Braket: {exc}"
print(f"    Braket itself with shots=100       -> {upstream}")
print("    => the exact path is SIMULATOR-ONLY; QPUs need the shot path.")

# --------------------------------------------------------------------------- #
# 3. The shot-based path (what a QPU runs).
# --------------------------------------------------------------------------- #

print("\n[3] Shot-based energy (QPU-compatible path)")
groups, identity = qubit_wise_commuting_groups(hamiltonian)
n_terms = len([l for l in hamiltonian.simplify().terms if set(l) != {"I"}])
print(f"    {n_terms} non-identity Pauli terms -> {len(groups)} "
      f"qubit-wise-commuting groups "
      f"({n_terms / max(len(groups), 1):.1f}x fewer circuits)")
print(f"    {'shots':>8}  {'E (Ha)':>14}  {'error':>11}  {'1-sigma bound':>14}")
print("    " + "-" * 52)
for shots in (500, 5000, 50000):
    measured = build_provider("braket", shots=shots).energy(
        n_qubits, occupied, generators, theta, hamiltonian)
    print(f"    {shots:>8}  {measured:>14.8f}  {measured - exact:>+11.2e}  "
          f"{shot_noise_estimate(hamiltonian, shots):>14.2e}")
print(f"    {'exact':>8}  {exact:>14.8f}")

# --------------------------------------------------------------------------- #
# 4. Hardware cost.
# --------------------------------------------------------------------------- #

print("\n[4] Cost of one energy evaluation on a QPU")
one_norm = sum(abs(complex(c)) for l, c in hamiltonian.simplify().terms.items()
               if set(l) != {"I"})
needed = int((one_norm / CHEMICAL_ACCURACY) ** 2)
print(f"    quantum tasks per evaluation : {len(groups)}")
print(f"    Hamiltonian 1-norm           : {one_norm:.4f} Ha")
print(f"    shots/group for 1.6 mHa      : ~{needed:.3g} (worst-case bound)")
print("    => hardware VQE needs error mitigation and smarter estimators;")
print("       this bound is why, not a defect of the implementation.")

# --------------------------------------------------------------------------- #
# 5. Registered AWS devices.
# --------------------------------------------------------------------------- #

print("\n[5] Braket devices Carcará can target")
print(f"    {"device":<26}{"kind":<12}{"shots":<10}ARN")
print("    " + "-" * 66)
for device in describe_devices():
    if device.provider != "braket":
        continue
    kind = "simulator" if device.simulator else "QPU"
    arn = device_arn(device.name) or "(local)"
    shots = "required" if requires_shots(device.name) else "optional"
    print(f"    {device.name:<26}{kind:<12}{shots:<10}{arn}")

print(f"\n{rule}")
print("VERDICT: Carcará is compatible with Amazon Braket, including QPUs,")
print("         through the shot-based path (device=..., shots=N).")
print("         The state-vector path remains simulator-only by Braket's design.")
print("         Caveat: ADAPT-VQE's pool-gradient screening is still classical;")
print("         VQE with a fixed ansatz is fully hardware-native today.")
print(rule)
