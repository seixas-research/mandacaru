"""H2O with the PAW-LCAO basis: minimal (SZ) and double-zeta polarized (DZP)."""
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("H2O")
atoms.center(vacuum=3.0)

for size in ("SZ", "DZP"):
    atoms.calc = Mandacaru(method="adapt-vqe",
                           basis={"name": "PAW-LCAO", "size": size,
                                  "energy_shift": 0.1},
                           h=0.25,
                           active_space={"orbitals": 6, "method": "mp2"})
    energy = atoms.get_potential_energy()
    print(f"PAW-LCAO {size}: {energy:.6f} eV")
