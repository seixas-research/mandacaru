"""Build the LiH qubit Hamiltonian once, save it, and run every pool from the file."""
import os
from ase.build import molecule
from mandacaru import Mandacaru
from mandacaru.core import detect_format

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)

atoms = molecule("LiH")
atoms.center(vacuum=3.0)
for fmt in ("parquet", "json"):
    path = os.path.join(DATA, f"lih_hamiltonian.{fmt}")
    atoms.calc = Mandacaru(method="adapt-vqe",
                           basis={"name": "PAW-LCAO", "size": "SZ"},
                           h=0.25,
                           save_hamiltonian=path,
                           hamiltonian_format=fmt)
    print(f"{detect_format(path):8s} E = {atoms.get_potential_energy():.6f} eV  "
          f"({os.path.getsize(path) / 1024:.1f} KiB)")

# No geometry, no integrals, no mapping: the file is the whole problem.
for pool in ("fermionic", "qubit", "qeb", "ceo"):
    result = Mandacaru(method="adapt-vqe", pool=pool,
                       load_hamiltonian=os.path.join(DATA, "lih_hamiltonian.parquet")).run()
    print(f"{pool:10s} E = {result.optimal_energy:.6f} eV  ({result.num_operators} operators)")
