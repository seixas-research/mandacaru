"""Band structure and density of states of diamond silicon, saved as a PNG."""
import os
import matplotlib.pyplot as plt
from ase.build import bulk
from mandacaru import Mandacaru

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)

atoms = bulk("Si", "diamond", a=5.43)
atoms.calc = Mandacaru(method="dft",
                       xc="lda",
                       basis={"name": "PAW-LCAO", "size": "DZP"},
                       h=0.25,
                       kpts={"size": (4, 4, 4), "gamma": True})
atoms.get_potential_energy()
fermi = atoms.calc.get_fermi_level()

bands = atoms.calc.band_structure(path="LGXWKG", npoints=80)
levels = bands.energies[0]
gap = levels[levels > fermi].min() - levels[levels < fermi].max()
print(f"Kohn-Sham gap along LGXWKG: {gap:.3f} eV")

energy, dos = atoms.calc.dos(width=0.15, kpts=(8, 8, 8))

fig, (ax_bands, ax_dos) = plt.subplots(1, 2, sharey=True,
                                       gridspec_kw={"width_ratios": [3, 1]})
bands.subtract_reference().plot(ax=ax_bands, emin=-14.0, emax=8.0)
ax_dos.plot(dos, energy - fermi)
ax_dos.set_xlabel("DOS (states/eV)")
fig.savefig(os.path.join(DATA, "si_bands.png"), dpi=150)
