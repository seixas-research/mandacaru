"""Occupation smearing in fcc aluminum: the free energy and its sigma -> 0 estimate."""
from ase.build import bulk
from mandacaru import Mandacaru

atoms = bulk("Al", "fcc", a=4.05)

for method in ("fermi-dirac", "gaussian", "methfessel-paxton"):
    for width in (0.05, 0.1, 0.2, 0.4):
        atoms.calc = Mandacaru(method="dft",
                               xc="lda",
                               basis={"name": "PAW-LCAO", "size": "DZP"},
                               h=0.25,
                               kpts={"size": (8, 8, 8), "gamma": True},
                               smearing={"method": method, "width": width})
        estimate = atoms.get_potential_energy()
        free = atoms.get_potential_energy(force_consistent=True)
        print(f"{method:18s} sigma = {width:.2f} eV  "
              f"E(sigma -> 0) = {estimate:.6f} eV  F = {free:.6f} eV")
