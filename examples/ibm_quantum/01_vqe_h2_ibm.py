"""VQE of H2 on an IBM Quantum processor.

The fake backend rehearses the run locally with the processor's noise model.
Set DEVICE to "ibm_kingston" to submit it to the real processor.
"""
from ase.build import molecule
from mandacaru import Mandacaru

DEVICE = "fake_kingston"

atoms = molecule("H2")
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="vqe",
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       h=0.25,
                       mapping="parity_reduced",
                       device=DEVICE,
                       shots=4096,
                       optimizer={"method": "COBYLA", "maxiter": 30, "tol": 1e-3},
                       backend_options={"physical_qubits": [0, 1],
                                        "estimator_options": {"max_execution_time": 60}})

energy = atoms.get_potential_energy()
print(f"{DEVICE}: E = {energy:.4f} eV on {atoms.calc.n_qubits} qubits")
