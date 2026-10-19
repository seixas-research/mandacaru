"""The electron affinity of hydrogen with and without diffuse functions.

H- is bound only by correlation: Hartree-Fock leaves it unbound, and so does a
basis that cannot hold the extra electron.  The confined PAW-LCAO-DZP basis
is such a basis; `"diffuse": True` adds one wide s and p shell, and ADAPT-VQE
in an MP2-ranked active space then binds the anion.  The H atom (one
electron) is exact at UHF.  The experimental affinity is 0.754 eV.

The diffuse shells need the box to hold them: hydrogen's s shell reaches
7.2 Angstrom (the `diffuse_box` table of each log says so).
"""
import csv
import os

from ase import Atoms
from mandacaru import Mandacaru

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)


def hydrogen():
    atoms = Atoms("H", positions=[[0.013, -0.021, 0.007]])
    atoms.center(vacuum=7.3)
    return atoms


rows = []
for diffuse in (False, True):
    basis = {"name": "PAW-LCAO", "size": "DZP", "diffuse": diffuse}
    tag = "diffuse" if diffuse else "plain"
    atom = hydrogen()
    atom.calc = Mandacaru(method="uhf", basis=basis, grid_spacing=0.25,
                          txt=os.path.join(DATA, f"h_atom_{tag}.txt"))
    e_atom = atom.get_potential_energy()

    anion = hydrogen()
    # With diffuse shells an active space must rank its virtual orbitals by
    # occupation ("mp2"): the "energy" ranking would keep the diffuse ones
    # first and is refused.  "symmetry" keeps the degenerate p sets whole.
    active_space = ({"method": "mp2", "orbitals": 8, "symmetry": True}
                    if diffuse else None)
    anion.calc = Mandacaru(method="adapt-vqe", pool="fermionic", basis=basis,
                           grid_spacing=0.25, charge=-1,
                           active_space=active_space,
                           optimizer={"method": "L-BFGS", "maxiter": 2000,
                                      "tol": 1e-10},
                           convergence={"gradient": 1e-6},
                           txt=os.path.join(DATA, f"h_anion_{tag}.txt"))
    e_anion = anion.get_potential_energy()
    affinity = e_atom - e_anion
    rows.append({"basis": f"DZP {tag}", "qubits": anion.calc.n_qubits,
                 "E_H_eV": e_atom, "E_H-_eV": e_anion, "EA_eV": affinity})
    print(f"DZP {tag:8s} {anion.calc.n_qubits:2d} qubits  "
          f"EA = {affinity:+.3f} eV")

with open(os.path.join(DATA, "h_anion_affinity.csv"), "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
