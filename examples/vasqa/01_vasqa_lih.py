"""LiH with VASQA: a Markov-chain search over operator sequences."""
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("LiH")
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="vasqa",
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       grid_spacing=0.25,
                       pool="qeb",
                       max_steps=60,
                       seed=7)

energy = atoms.get_potential_energy()
result = atoms.calc.result
print(f"VASQA energy: {energy:.6f} eV")
print(f"Best sequence ({len(result.operators)} operators): {result.operators}")
