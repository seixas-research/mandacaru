"""The same VQE on H2 with different classical optimizers."""
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("H2")
atoms.center(vacuum=3.0)

for method in ("SLSQP", "BFGS", "L-BFGS", "CG", "COBYLA", "Nelder-Mead", "SPSA"):
    atoms.calc = Mandacaru(method="vqe",
                           basis={"name": "PAW-LCAO", "size": "SZ"},
                           h=0.25,
                           optimizer={"method": method, "maxiter": 500,
                                      "tol": 1e-8})
    energy = atoms.get_potential_energy()
    print(f"{method:12s} {energy:.6f} eV  "
          f"({atoms.calc.result.num_evaluations} evaluations)")
