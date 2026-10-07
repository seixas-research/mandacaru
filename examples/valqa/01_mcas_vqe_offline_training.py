"""Record MCAS-VQE chains on small molecules and train VALQA's proposal model."""
import os
import numpy as np
from ase import Atoms
from mandacaru import Mandacaru
from mandacaru.algorithms import fit

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STORE = os.path.join(DATA, "offline_store")
os.makedirs(STORE, exist_ok=True)


def linear(symbols, spacings):
    """A linear molecule along z, centered in a 10 Angstrom box."""
    z = np.concatenate([[0.0], np.cumsum(spacings)])
    return Atoms(symbols, positions=[[5.0, 5.0, 5.0 + zi - z.mean()] for zi in z],
                 cell=[10.0, 10.0, 10.0])


for symbols, distances, bonds in (("H2", (0.65, 0.75, 0.85), 1),
                                  ("LiH", (1.45, 1.55, 1.65), 1),
                                  ("HBeH", (1.25, 1.35, 1.45), 2),
                                  ("H4", (0.85, 0.95, 1.05), 3)):
    for r in distances:
        atoms = linear(symbols, [r] * bonds)
        atoms.calc = Mandacaru(method="mcas-vqe",
                               basis={"name": "PAW-LCAO", "size": "SZ"},
                               h=0.20,
                               pool="qeb",
                               max_steps=40,
                               max_length=12,
                               warm_start=False,
                               screen_insertions=4,
                               length_penalty=1e-3,
                               record=STORE,
                               seed=1)
        print(f"{symbols:5s} r = {r:.2f} A  E = {atoms.get_potential_energy():.6f} eV")

model = fit(STORE, fractions=(1.0,))
print(f"ready: {model.ready} -- {model.report['reason']}")
