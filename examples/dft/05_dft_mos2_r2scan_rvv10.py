"""MoS2 monolayer with r2SCAN+rVV10 in PAW-LCAO, band structure as a PNG.

The 2H monolayer is periodic in the plane (``pbc=[True, True, False]``) with
vacuum along z, so the k-point mesh, the band path and the density of
states are two-dimensional. ``xc="r2scan+rvv10"`` adds the rVV10 nonlocal
correlation to the meta-GGA, the dispersion that binds MoS2 layers to each
other. The datasets are the scalar-relativistic PBE set, ``pbe-sr/`` (no
r2SCAN datasets are generated), written out here although it is already
this functional's default. The energy, the Fermi level and the Kohn-Sham
gap are printed; the bands along Gamma-M-K-Gamma and the DOS are written to
``data/``.
"""
import os
import matplotlib.pyplot as plt
from ase.build import mx2
from mandacaru import Mandacaru

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)

atoms = mx2("MoS2", kind="2H", a=3.16, thickness=3.17, vacuum=5.0)
atoms.pbc = [True, True, False]

atoms.calc = Mandacaru(method="dft",
                       xc="r2scan+rvv10",
                       basis={"name": "PAW-LCAO", "size": "TZP", "energy_shift": 0.01},
                       paw_dataset="pbe-sr",
                       grid_spacing=0.15,
                       kpts={"size": (8, 8, 1), "gamma": True})
energy = atoms.get_potential_energy()
fermi = atoms.calc.get_fermi_level()
print(f"Total energy: {energy:.6f} eV")
print(f"Fermi level:  {fermi:.4f} eV")

bands = atoms.calc.band_structure(path="GMKG", npoints=60)
levels = bands.energies[0]
gap = levels[levels > fermi].min() - levels[levels < fermi].max()
print(f"Kohn-Sham gap along GMKG: {gap:.3f} eV")

energies, dos = atoms.calc.dos(width=0.10, kpts=(16, 16, 1))

fig, (ax_bands, ax_dos) = plt.subplots(1, 2, sharey=True,
                                       gridspec_kw={"width_ratios": [3, 1]})
bands.subtract_reference().plot(ax=ax_bands, emin=-6.0, emax=4.0)
ax_dos.plot(dos, energies - fermi)
ax_dos.set_xlabel("DOS (states/eV)")
fig.savefig(os.path.join(DATA, "mos2_bands.png"), dpi=150)
