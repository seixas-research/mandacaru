"""LiH with VQE and UCCSD in an analytic (HAO) and a Gaussian (STO-3G) basis."""
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("LiH")
atoms.center(vacuum=3.0)

for basis in ("HAO", "STO-3G"):
    atoms.calc = Mandacaru(method="vqe",
                           basis=basis,
                           h=0.10,
                           optimizer={"method": "SLSQP", "maxiter": 1000,
                                      "tol": 1e-10})
    energy = atoms.get_potential_energy()
    result = atoms.calc.result
    print(f"{basis:6s} HF {result.reference_energy:.6f} eV  "
          f"VQE {energy:.6f} eV  ({atoms.calc.n_qubits} qubits, "
          f"{result.num_parameters} amplitudes)")
