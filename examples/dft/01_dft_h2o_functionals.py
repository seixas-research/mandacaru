"""Kohn-Sham DFT of H2O with three exchange-correlation functionals."""
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("H2O")
atoms.center(vacuum=3.0)

for xc, dispersion in (("lda", None), ("pbe", "d4"), ("r2scan", "d4")):
    atoms.calc = Mandacaru(method="dft",
                           xc=xc,
                           dispersion=dispersion,
                           basis={"name": "PAW-LCAO", "size": "DZP"},
                           h=0.2)
    energy = atoms.get_potential_energy()
    scf = atoms.calc.result.scf
    print(f"{xc:7s} E = {energy:.4f} eV  ({scf.n_iterations} SCF iterations)")
