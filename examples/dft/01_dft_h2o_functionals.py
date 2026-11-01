"""Kohn-Sham DFT of H2O with three exchange-correlation functionals."""
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("H2O")
atoms.center(vacuum=3.0)

for xc, vdw in (("lda", None), ("pbe", "rvv10"), ("r2scan", "rvv10")):
    atoms.calc = Mandacaru(method="dft",
                           xc=xc,
                           vdw=vdw,
                           basis={"name": "PAW-LCAO", "size": "DZP"},
                           grid_spacing=0.2)
    energy = atoms.get_potential_energy()
    scf = atoms.calc.result.scf
    print(f"{atoms.calc.xc:14s} E = {energy:.4f} eV  ({scf.n_iterations} SCF iterations)")
