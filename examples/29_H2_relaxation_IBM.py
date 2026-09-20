# -*- coding: utf-8 -*-
# file: examples/29_H2_relaxation_IBM.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""H2 geometry relaxation on IBM Quantum hardware (ADAPT-VQE, fermionic pool).

PAW pseudopotentials with the parity mapping and the two-qubit reduction put H2
on 2 qubits.  At each geometry the ansatz is optimized locally; the optimized
state is then measured on the processor -- one Estimator job per geometry --
and both the energy and the forces (from the measured reduced density
matrices) come from that measurement.  BFGS takes only STEPS steps, i.e.
STEPS + 1 jobs, to keep the QPU time small:

    HARDWARE = "ibm_kingston,ibm_fez,ibm_marrakesh"   # least busy of these
    HARDWARE = "fake_kingston"                        # local rehearsal
    HARDWARE = None                                   # local state vector only

Writes the ADAPT-VQE log of the last geometry, the trajectory and the final
geometry to examples/data/.
"""

import os

from ase import Atoms
from ase.io import write
from ase.optimize import BFGS

from mandacaru import Mandacaru
from mandacaru.backends.providers import QiskitProvider

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)

HARDWARE = "ibm_kingston,ibm_fez,ibm_marrakesh"
SHOTS = 4096
STEPS = 2
#: Quantum-time cap per job (Qiskit Runtime cancels a job past it), so the run
#: can never use more than (STEPS + 1) * MAX_SECONDS_PER_JOB QPU-seconds.
MAX_SECONDS_PER_JOB = 25

atoms = Atoms("H2",
              positions=[[0.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0]],
              cell=[10.0, 10.0, 10.0])
atoms.center()

provider = None
if HARDWARE:
    options = {}
    if HARDWARE.startswith("ibm"):
        from qiskit_ibm_runtime import QiskitRuntimeService

        remaining = QiskitRuntimeService().usage()["usage_remaining_seconds"]
        budget = (STEPS + 1) * MAX_SECONDS_PER_JOB
        if remaining < budget:
            raise SystemExit(f"{remaining} QPU-seconds left this month; this "
                             f"run may use up to {budget}")
        options = {"max_execution_time": MAX_SECONDS_PER_JOB}
    provider = QiskitProvider(device=HARDWARE, shots=SHOTS,
                              estimator_options=options)

atoms.calc = Mandacaru(method="adapt-vqe",
                       basis="PAW",
                       h=0.20,
                       pool="fermionic",
                       mapping="parity_reduced",
                       optimizer="L-BFGS-B",
                       max_iterations=10,
                       gradient_tolerance=1e-5,
                       measurement_provider=provider,
                       output=os.path.join(DATA, "h2_relax_ibm_output.txt"))


def report():
    job = atoms.calc.measurement["job_id"] if atoms.calc.measurement else None
    print(f"    H-H distance {atoms.get_distance(0, 1):.4f} A"
          + (f"  (job {job})" if job else ""))


opt = BFGS(atoms, trajectory=os.path.join(DATA, "h2_relax_ibm.traj"))
opt.attach(report)
opt.run(fmax=0.05, steps=STEPS)
atoms.calc.write_optimization_summary(optimizer=opt)

write(os.path.join(DATA, "h2_relax_ibm.xyz"), atoms)
print(f"Final H-H distance: {atoms.get_distance(0, 1):.4f} A")
print(f"Final energy: {atoms.get_potential_energy():.6f} eV")
