"""Molecular dynamics of H2 with ADAPT-VQE forces (velocity Verlet)."""
from ase import Atoms, units
from ase.md.verlet import VelocityVerlet
from mandacaru import Mandacaru

atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.80]])
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis={"name": "PAW-LCAO", "size": "DZP"},
                       h=0.25,
                       # Five MP2 natural orbitals: the bonding and
                       # antibonding pair, the next sigma and the whole pi
                       # pair.  Four would split the degenerate pi pair, and
                       # the selection would jump when the sigma level crosses
                       # it along the trajectory.
                       active_space={"orbitals": 5, "method": "mp2"},
                       transfer=True)

dynamics = VelocityVerlet(atoms, timestep=0.5 * units.fs)
for step in range(10):
    dynamics.run(1)
    print(f"t = {0.5 * (step + 1):4.1f} fs  d = {atoms.get_distance(0, 1):.4f} A  "
          f"E_total = {atoms.get_total_energy():.5f} eV")
