"""DFT+U on antiferromagnetic NiO: the Kohn-Sham gap at U = 0 and at U = 6 eV.

Rocksalt NiO (a = 4.17 angstrom) in the two-formula-unit type-II antiferromagnetic cell,
LDA with a PAW-LCAO SZ basis on a 2x2x2 mesh. Plain LDA leaves a gap of about 0.6 eV;
`hubbard_u` adds the on-site correction to the Ni 3d shell. The correction's energy, the
occupation matrix of each Ni site are printed; the run logs, with the `[SCF SUMMARY]`
lines of the correction, go to `data/`. Each calculation takes seconds on two threads.
"""
import os

import numpy as np
from ase.build import bulk, make_supercell
from mandacaru import Mandacaru

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)

U = 6.0   # eV; J defaults to 0, so U_eff = U


def nio_afm2(a=4.17):
    """The type-II cell: ferromagnetic (111) sheets of Ni with alternating moments."""
    atoms = make_supercell(bulk("NiO", "rocksalt", a=a), [[1, 0, 0], [0, 1, 0], [-1, -1, 2]])
    atoms.set_initial_magnetic_moments([2.0, 0.0, -2.0, 0.0])   # Ni, O, Ni, O
    return atoms


def kohn_sham_gap(calc):
    """Lowest empty level minus highest occupied level over the k mesh and both spins, in eV."""
    fermi = calc.get_fermi_level()
    levels = np.concatenate([calc.get_eigenvalues(kpt=k, spin=s)
                             for s in range(calc.get_number_of_spins())
                             for k in range(len(calc.get_ibz_k_points()))])
    return levels[levels > fermi].min() - levels[levels <= fermi].max()


def calculate(hubbard_u, log):
    atoms = nio_afm2()
    atoms.calc = Mandacaru(method="dft",
                           xc="lda",
                           basis={"name": "PAW-LCAO", "size": "SZ"},
                           grid_spacing=0.25,
                           kpts={"size": (2, 2, 2), "gamma": True},
                           smearing={"method": "fermi-dirac", "width": 0.05},
                           spin=True,
                           hubbard_u=hubbard_u,
                           resources={"threads": 2},
                           txt=os.path.join(DATA, log))
    energy = atoms.get_potential_energy()
    return atoms.calc, energy


calc, plain = calculate(None, "nio_afm_u0.txt")
print(f"U = 0 eV: E = {plain:.4f} eV, gap = {kohn_sham_gap(calc):.3f} eV")
calc.close()

calc, corrected = calculate({"Ni": {"l": "d", "U": U}}, "nio_afm_u6.txt")
result = calc.result
print(f"U = {U:g} eV: E = {corrected:.4f} eV, gap = {kohn_sham_gap(calc):.3f} eV")
print(f"  E_U = {result.hubbard_energy:.4f} eV (already inside E)")
for shell in result.hubbard:
    up, down = shell["n"][0], shell["n"][1]
    eigenvalues = np.concatenate([np.linalg.eigvalsh(up), np.linalg.eigvalsh(down)])
    print(f"  Ni {shell['l']} shell of atom {shell['atom']}: "
          f"trace up {shell['trace'][0]:.3f}, down {shell['trace'][1]:.3f}, "
          f"occupation eigenvalues {eigenvalues.min():.3f} to {eigenvalues.max():.3f}, "
          f"E_U = {shell['energy']:.4f} eV")
calc.close()
