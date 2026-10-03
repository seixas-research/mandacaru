"""H2 in a double-zeta basis built with the two split-valence schemes."""
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("H2")
atoms.center(vacuum=3.0)

for options in ({"zeta_split": "first_zeta"},
                {"zeta_split": "first_zeta", "tail_norm": 0.3},
                {"zeta_split": "last_zeta"},
                {"zeta_split": "last_zeta", "split_norm": 0.3}):
    atoms.calc = Mandacaru(method="adapt-vqe",
                           basis={"name": "PAW-LCAO", "size": "DZ", **options},
                           h=0.25)
    energy = atoms.get_potential_energy()
    print(f"{options}  E = {energy:.6f} eV")
