"""H2 at Hartree-Fock in numerical atomic orbitals of increasing size (SZ to DZP)."""
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("H2")
atoms.center(vacuum=3.0)

for size in ("SZ", "DZ", "TZ", "DZP"):
    atoms.calc = Mandacaru(method="rhf",
                           basis={"name": "NAO", "size": size},
                           h=0.12)
    energy = atoms.get_potential_energy()
    print(f"NAO {size:4s} E_RHF = {energy:.4f} eV")
