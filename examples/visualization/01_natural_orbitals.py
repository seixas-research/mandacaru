"""Draw the natural orbitals of H2O from its ADAPT-VQE ground state as PNG images."""
import os
from ase.build import molecule
from mandacaru import Mandacaru, Viewer3D

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA, exist_ok=True)

atoms = molecule("H2O")
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       h=0.25)
atoms.get_potential_energy()

for index in range(6):
    path = os.path.join(DATA, f"h2o_natural_orbital_{index}.png")
    Viewer3D(atoms.calc, quantity="natural_orbital", index=index, h=0.12,
             mode="scatter").save(path)
    print(f"Wrote {path}")
