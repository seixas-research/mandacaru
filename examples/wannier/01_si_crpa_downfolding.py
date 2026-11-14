"""Silicon downfolded to a two-site model (the most strongly coupled pair of maximally localized
hybrids) with cRPA, then solved with ADAPT-VQE."""
import numpy as np
from ase.build import bulk
from mandacaru import Mandacaru
from mandacaru.units import HARTREE_TO_EV

atoms = bulk("Si", "diamond", a=5.43)
atoms.calc = Mandacaru(method="dft",
                       xc="lda",
                       basis={"name": "PAW-LCAO", "size": "DZP"},
                       grid_spacing=0.25,
                       kpts={"size": (4, 4, 4), "gamma": True},
                       # a small smearing: a gapped crystal's screening has no thermal carriers
                       smearing={"method": "fermi-dirac", "width": 0.01})
atoms.get_potential_energy()
fermi = atoms.calc.get_fermi_level()

wannier = atoms.calc.wannier(8, guess="sp3",
                             windows={"outer": (fermi - 20.0, fermi + 20.0),
                                      "frozen": (fermi - 20.0, fermi + 1.0)})

# The maximally localized functions of this space are atom-centered hybrids that point into the
# interstitial, away from the neighbors (the bond-pointing sp3 set is a saddle of the spread). The
# two-site model takes the pair on different atoms with the largest hopping |t|; the quantum
# problem rotates it to its bonding and antibonding combinations. Each function belongs to the
# atom of its largest trial weight, which does not depend on the gauge.
atom = [wannier.labels[int(np.argmax(wannier.trial_weights[n]))].split(":")[0]
        for n in range(wannier.n_functions)]
hamiltonian, vectors = wannier.hamiltonian_R, wannier.lattice_vectors   # <w_i,0|H|w_j,R>, eV
t, i, j, R = max((abs(hamiltonian[r, i, j]), i, j, tuple(int(x) for x in vectors[r]))
                 for r in range(len(vectors)) for i in range(wannier.n_functions)
                 for j in range(wannier.n_functions) if atom[i] != atom[j])
pair = [(i, (0, 0, 0)), (j, R)]
print(f"largest-|t| pair: function {i} ({atom[i]}) and {j} ({atom[j]}) in cell {R}, |t| = {t:.3f} eV")

for screening in (None, "rpa", "crpa"):
    model = wannier.downfold(pair, screening=screening)
    U = model.two_body[0, 0, 0, 0].real * HARTREE_TO_EV   # on-site U of one interstitial-pointing hybrid
    J = model.two_body[0, 1, 1, 0].real * HARTREE_TO_EV
    print(f"{screening or 'bare':5s} U = {U:6.3f} eV  J = {J:6.3f} eV")

result = Mandacaru(method="adapt-vqe", **model.as_quantum_problem()).run()
print(f"ADAPT-VQE ground state of the cRPA model: {result.optimal_energy:.6f} eV")
