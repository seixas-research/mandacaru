"""ADAPT-VQE of H2 with every quantity measured on the device.

With shots > 0 nothing is simulated: the reference energy is the
determinant's, the first pool gradients are exact at it, and every later
gradient is the measured expectation value of its commutator [H, A_k], merged
with the other operators' into qubit-wise-commuting bases.  The run log's
[GRADIENT MEASUREMENT] block says what one growth step measures and costs.

The fake backend rehearses the run locally.  fake_manila is five qubits:
without qiskit-aer a fake backend runs on Qiskit's BasicSimulator, which takes
at most 24 qubits, and a circuit transpiled to a 156-qubit processor
(fake_kingston, fake_fez, fake_marrakesh) spans all 156 wires.  With qiskit-aer
installed, rehearse on the processor you mean to use.  Set DEVICE to an ibm_*
name to submit to a real processor -- only after reading the plan in the log,
and with service.usage() checked against the quota.
"""
import os

from ase.build import molecule
from mandacaru import Mandacaru
from mandacaru.utils.logging import parse_output

DEVICE = "fake_manila"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LOG = os.path.join(DATA, f"adapt_h2_measured_gradients_{DEVICE}.txt")


def main():
    # The fake backend's simulator starts worker processes, which import this
    # file again: the run belongs under the main guard.
    os.makedirs(DATA, exist_ok=True)
    atoms = molecule("H2")
    atoms.center(vacuum=3.0)
    atoms.calc = Mandacaru(method="adapt-vqe",
                           basis={"name": "PAW-LCAO", "size": "SZ"},
                           grid_spacing=0.25,
                           pool="qeb",
                           max_iterations=2,
                           device=DEVICE,
                           shots=4096,
                           optimizer={"method": "COBYLA", "maxiter": 30,
                                      "tol": 1e-3},
                           measurement_budget={"observables": 50_000,
                                               "bases": 10_000},
                           backend_options={"physical_qubits": [0, 1, 2, 3],
                                            "run_budget": {"jobs": 120},
                                            "estimator_options": {
                                                "max_execution_time": 60}},
                           txt=LOG)

    energy = atoms.get_potential_energy()
    plan = parse_output(LOG)["gradient_measurement"]
    print(f"{DEVICE}: E = {energy:.4f} eV on {atoms.calc.n_qubits} qubits")
    print(f"one gradient step: {plan['measured_operators']} operators, "
          f"{plan['strings_per_step']} Pauli strings in {plan['bases_per_step']} "
          f"bases, {plan['jobs_per_step']} job(s), {plan['shots_per_step']} shots")


if __name__ == "__main__":
    main()
