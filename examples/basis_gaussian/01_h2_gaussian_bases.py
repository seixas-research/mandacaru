"""H2 with all-electron Gaussian basis sets of increasing size."""
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("H2")
atoms.center(vacuum=3.0)

for basis in ("STO-3G", "6-31G", "6-31G(d)"):
    atoms.calc = Mandacaru(method="adapt-vqe", basis=basis, h=0.2)
    energy = atoms.get_potential_energy()
    print(f"{basis:9s} {energy:.6f} eV on {atoms.calc.n_qubits} qubits")
