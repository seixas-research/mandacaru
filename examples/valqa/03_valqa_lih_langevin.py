"""Langevin dynamics of LiH with VALQA, carrying the ansatz from step to step."""
import os
from ase import Atoms, units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from mandacaru import Mandacaru

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)

atoms = Atoms("LiH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]])
atoms.center(vacuum=4.0)
atoms.calc = Mandacaru(method="valqa",
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       h=0.20,
                       pool="qeb",
                       max_steps=40,
                       transfer=True,
                       transfer_steps=10,
                       screen_insertions=2,
                       update_between_geometries=True,
                       record=os.path.join(DATA, "langevin_edits"),
                       seed=7)

MaxwellBoltzmannDistribution(atoms, temperature_K=300.0)
dynamics = Langevin(atoms, timestep=0.5 * units.fs, temperature_K=300.0,
                    friction=0.01 / units.fs)
for step in range(20):
    dynamics.run(1)
    print(f"t = {0.5 * (step + 1):4.1f} fs  d = {atoms.get_distance(0, 1):.4f} A  "
          f"{atoms.calc.result.num_evaluations:4d} evaluations")
