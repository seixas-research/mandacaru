"""Silicon downfolded to a two-site bond model with cRPA, then solved with ADAPT-VQE."""
import numpy as np
from ase.build import bulk
from mandacaru import Mandacaru
from mandacaru.units import HARTREE_TO_EV

atoms = bulk("Si", "diamond", a=5.43)
atoms.calc = Mandacaru(method="dft",
                       xc="lda",
                       basis={"name": "PAW-LCAO", "size": "DZP"},
                       h=0.25,
                       kpts={"size": (4, 4, 4), "gamma": True})
atoms.get_potential_energy()
fermi = atoms.calc.get_fermi_level()

wannier = atoms.calc.wannier(8, guess="sp3",
                             windows={"outer": (fermi - 20.0, fermi + 20.0),
                                      "frozen": (fermi - 20.0, fermi + 1.0)})

# The two sp3 hybrids that face each other across the first Si-Si bond
start, end = atoms.positions
bond = [int(np.argmin(np.linalg.norm(wannier.centers - (a + 0.2 * (b - a)), axis=1)))
        for a, b in ((start, end), (end, start))]

for screening in (None, "rpa", "crpa"):
    model = wannier.downfold(bond, screening=screening)
    U = model.two_body[0, 0, 0, 0].real * HARTREE_TO_EV
    J = model.two_body[0, 1, 1, 0].real * HARTREE_TO_EV
    print(f"{screening or 'bare':5s} U = {U:6.3f} eV  J = {J:6.3f} eV")

result = Mandacaru(method="adapt-vqe", **model.as_quantum_problem()).run()
print(f"ADAPT-VQE ground state of the cRPA model: {result.optimal_energy:.6f} eV")
