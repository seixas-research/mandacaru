"""Geometry optimization of H2O with ADAPT-VQE forces."""
from ase.build import molecule
from ase.optimize import BFGS
from mandacaru import Mandacaru

atoms = molecule("H2O")
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       h=0.25,
                       transfer=True)

BFGS(atoms).run(fmax=0.05)
print(f"O-H: {atoms.get_distance(0, 1):.4f} Angstrom, "
      f"H-O-H: {atoms.get_angle(1, 0, 2):.2f} degrees")
