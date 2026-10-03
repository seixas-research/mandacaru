"""A periodic hydrogen chain: energy per cell and quasiparticle bands."""
from ase import Atoms
from mandacaru import Mandacaru

chain = Atoms("H", positions=[[0.0, 0.0, 0.0]],
              cell=[1.0, 10.0, 10.0], pbc=[True, False, False])
chain.calc = Mandacaru(method="bloch-adapt-vqe",
                       basis="HAO",
                       h=0.25,
                       kpts={"size": (4, 1, 1), "gamma": True})

energy = chain.get_potential_energy()
bands = chain.calc.bands()
print(f"Energy per cell: {energy:.6f} eV")
for k, band in zip(chain.calc.kpoints[:, 0], bands[:, 0]):
    print(f"k = {k:+.3f}  E = {band:+.3f} eV")
