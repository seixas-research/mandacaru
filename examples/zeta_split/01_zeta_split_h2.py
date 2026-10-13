"""H2 in a double-zeta basis built with the two split-valence schemes.

Each scheme carries its own parameter: ``first_zeta`` splits every extra zeta
from the first by the tail's norm, ``last_zeta`` each from the previous one by
the tail's squared norm.
"""
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("H2")
atoms.center(vacuum=3.0)

for zeta_split in ("first_zeta", "last_zeta"):
    atoms.calc = Mandacaru(method="adapt-vqe",
                           basis={"name": "PAW-LCAO", "size": "DZ",
                                  "zeta_split": zeta_split},
                           grid_spacing=0.25)
    energy = atoms.get_potential_energy()
    print(f"zeta_split={zeta_split!r}  E = {energy:.6f} eV")
