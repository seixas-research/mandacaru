"""Out-of-time-order correlator of the H2 ground state prepared by ADAPT-VQE."""
from ase.build import molecule
from mandacaru import Mandacaru
from mandacaru.algorithms import NestedOTOC
from mandacaru.core import PauliSum

atoms = molecule("H2")
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       h=0.25)
atoms.get_potential_energy()

prepared = atoms.calc.solver.checkpoint
otoc = NestedOTOC(prepared.hamiltonian,
                  butterfly=PauliSum({"IIIZ": 1}),
                  measurement=PauliSum({"ZIII": 1}))
for time in (0.5, 1.0, 2.0):
    result = otoc.run(prepared, time=time, otoc_order=2, steps=40)
    print(f"t = {time:.1f}  C = {result.correlator:.6f}")
