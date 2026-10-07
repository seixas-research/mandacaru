"""VQE of H2 through Amazon Braket's local simulator, exact and with shots.

shots=0 reads the exact state vector, which only a simulator offers; every
Braket QPU requires shots > 0, so the shot-based runs follow the path a QPU
takes.  Set DEVICE to a QPU name (e.g. "braket-ionq-aria") to submit it.
"""
from ase.build import molecule
from mandacaru import Mandacaru

DEVICE = "braket-local"

atoms = molecule("H2")
atoms.center(vacuum=3.0)

for shots in (0, 1000, 10000):
    atoms.calc = Mandacaru(method="vqe",
                           basis={"name": "PAW-LCAO", "size": "SZ"},
                           h=0.25,
                           mapping="parity_reduced",
                           device=DEVICE,
                           shots=shots,
                           optimizer={"method": "COBYLA", "maxiter": 60, "tol": 1e-4})
    energy = atoms.get_potential_energy()
    print(f"shots = {shots:6d}  E = {energy:.4f} eV")
