"""Ethanol with ADAPT-VQE in PAW-LCAO-TZP: an MP2 active space on 20 qubits.

TZP gives ethanol 87 spatial orbitals (174 spin orbitals). Five occupied and
five virtual MP2 natural orbitals reach the register, CAS(10,10), which is 20
qubits under the Jordan-Wigner mapping. The two-electron integrals are
Cholesky factors generated on the grid, so the 87^4 tensor is never formed:
the SCF, the MP2 selection and the active-space integrals are built from the
factors. The run log (orbital table, ADAPT
iterations, summary) and the iteration history are written to ``data/``.
"""
import csv
import os
from ase import Atoms
from mandacaru import Mandacaru
from mandacaru.utils import parse_output

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)
LOG = os.path.join(DATA, "ethanol_adapt_vqe.txt")

atoms = Atoms("CCOHHHHHH",
              positions=[[8.114804, 5.636852, 5.886933],
                         [6.946623, 6.596696, 5.886933],
                         [5.756540, 5.809565, 5.886933],
                         [5.000000, 6.418759, 5.886933],
                         [6.989180, 7.244742, 6.773866],
                         [6.989180, 7.244742, 5.000000],
                         [9.062514, 6.182034, 5.886933],
                         [8.075222, 5.000000, 6.772814],
                         [8.075222, 5.000000, 5.001052]],
              cell=[14.062514, 12.244742, 11.773866])
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis={"name": "PAW-LCAO", "size": "TZP"},
                       grid_spacing=0.25,
                       two_body={"method": "cholesky", "tolerance": 1e-7},
                       active_space={"orbitals": {"occupied": 5, "virtual": 5},
                                     "method": "mp2"},
                       pool="qeb",
                       max_iterations=60,
                       convergence={"gradient": 3e-2},
                       txt=LOG)

energy = atoms.get_potential_energy()
result = atoms.calc.result

with open(os.path.join(DATA, "ethanol_adapt_vqe.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["iteration", "operator", "max_gradient", "energy_eV",
                     "parameters", "cnot_count"])
    for n, it in enumerate(result.iterations, start=1):
        writer.writerow([n, it.operator_label, f"{it.max_gradient:.8f}",
                         f"{it.energy:.8f}", it.num_parameters, it.cnot_count])

blocks = parse_output(LOG)
print(f"Active space: {blocks['electrons']['spatial orbitals']} of "
      f"{blocks['basis']['basis_functions']} orbitals "
      f"({blocks['active_space']['frozen_orbitals']} frozen)")
print(f"Hartree-Fock energy: {result.reference_energy:.6f} eV")
print(f"ADAPT-VQE energy:    {energy:.6f} eV on {atoms.calc.n_qubits} qubits")
print(f"Correlation energy:  {result.correlation_energy:.4f} eV "
      f"({result.num_operators} operators, converged: {result.converged})")
