"""Simulation of bulk Bi2Te3 with and without Spin-Orbit Coupling (SOC).

Demonstrates:
- Setting up the rhombohedral primitive cell of bulk Bi2Te3 (space group R-3m, #166).
- Running scalar-relativistic DFT (without SOC) using the 'pbe-sr' PAW dataset.
- Running fully relativistic two-component spinor DFT (with SOC) using the 'pbe-fr' PAW dataset.
- Plotting and comparing continuous band structures along the high-symmetry path Z-Gamma-F-Z-L.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.spacegroup import crystal
from mandacaru import Mandacaru
from mandacaru.units import HARTREE_TO_EV

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)

# 1. Rhombohedral primitive cell of Bi2Te3 (Space group R-3m, #166)
a_hex = 4.383
c_hex = 30.487
atoms = crystal(
    symbols=["Te", "Bi", "Te"],
    basis=[(0, 0, 0.2117), (0, 0, 0.3990), (0, 0, 0.0)],
    spacegroup=166,
    cellpar=[a_hex, a_hex, c_hex, 90, 90, 120],
    primitive_cell=True,
)

basis_opts = {"name": "PAW-LCAO", "size": "SZ", "filter_cutoff": False}
kpts_scf = {"size": (4, 4, 4), "gamma": True}
path = "ZGFZL"
npoints = 100

# 2. Run WITHOUT Spin-Orbit Coupling ('pbe-sr')
print("Running PBE WITHOUT Spin-Orbit Coupling (pbe-sr)...")
calc_nosoc = Mandacaru(
    atoms=atoms,
    method="dft",
    xc="pbe",
    paw_dataset="pbe-sr",
    basis=basis_opts,
    kpts=kpts_scf,
    grid_spacing=0.32,
    trace=False,
    max_iter=40,
)
atoms.calc = calc_nosoc
atoms.get_potential_energy()
ef_nosoc = calc_nosoc.get_fermi_level()
bs_nosoc = calc_nosoc.band_structure(path=path, npoints=npoints)
energies_nosoc = bs_nosoc.energies[0] - ef_nosoc

# 3. Run WITH Spin-Orbit Coupling ('pbe-fr')
print("Running PBE WITH Spin-Orbit Coupling (pbe-fr)...")
calc_soc = Mandacaru(
    atoms=atoms,
    method="dft",
    xc="pbe",
    paw_dataset="pbe-fr",
    basis=basis_opts,
    kpts=kpts_scf,
    grid_spacing=0.32,
    trace=False,
    max_iter=40,
)
atoms.calc = calc_soc
atoms.get_potential_energy()
ef_soc = calc_soc.get_fermi_level()
scf_soc = calc_soc.result.scf
eso_ev = (
    scf_soc.spin_orbit_energy * HARTREE_TO_EV
    if scf_soc.spin_orbit_energy is not None
    else 0.0
)
bs_soc = calc_soc.band_structure(path=path, npoints=npoints)
energies_soc = bs_soc.energies[0] - ef_soc

print(f"Spin-Orbit Energy: {eso_ev:.4f} eV")

# 4. Plot Comparison
kpath = bs_soc.path
xcoords, special_x, special_labels = kpath.get_linear_kpoint_axis()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)

# Without SOC
ax1.plot(xcoords, energies_nosoc, color="tab:blue", lw=1.2, alpha=0.85)
ax1.plot(xcoords, energies_nosoc[:, 38], color="navy", lw=2.0)
ax1.plot(xcoords, energies_nosoc[:, 39], color="darkred", lw=2.0)
ax1.axhline(0.0, color="gray", linestyle="--", lw=1.0)
for x in special_x:
    ax1.axvline(x, color="black", lw=0.6, linestyle=":")
ax1.set_xticks(special_x)
ax1.set_xticklabels([r"$\Gamma$" if l == "G" else l for l in special_labels], fontsize=11)
ax1.set_xlim(0, xcoords[-1])
ax1.set_ylim(-2.0, 2.5)
ax1.set_ylabel(r"$E - E_F\ (\mathrm{eV})$", fontsize=12)
ax1.set_title(r"Without SOC ($\mathtt{pbe\text{-}sr}$)", fontsize=13, fontweight="bold")
ax1.grid(True, alpha=0.25, linestyle="--")

# With SOC
ax2.plot(xcoords, energies_soc, color="tab:red", lw=1.2, alpha=0.85)
ax2.plot(xcoords, energies_soc[:, 77], color="navy", lw=2.0)
ax2.plot(xcoords, energies_soc[:, 78], color="darkred", lw=2.0)
ax2.axhline(0.0, color="gray", linestyle="--", lw=1.0)
for x in special_x:
    ax2.axvline(x, color="black", lw=0.6, linestyle=":")
ax2.set_xticks(special_x)
ax2.set_xticklabels([r"$\Gamma$" if l == "G" else l for l in special_labels], fontsize=11)
ax2.set_xlim(0, xcoords[-1])
ax2.set_ylim(-2.0, 2.5)
ax2.set_title(r"With Spin-Orbit Coupling ($\mathtt{pbe\text{-}fr}$)", fontsize=13, fontweight="bold")
ax2.grid(True, alpha=0.25, linestyle="--")

fig.suptitle(r"Bulk $\mathrm{Bi}_2\mathrm{Te}_3$ Band Structure ($R\bar{3}m$)", fontsize=14, fontweight="bold")
plt.tight_layout()

out_png = os.path.join(DATA, "bi2te3_bands_soc.png")
try:
    fig.savefig(out_png, dpi=150)
    print(f"Saved figure to {out_png}")
except Exception as ex:
    print(f"Could not save to {out_png}: {ex}")
plt.close(fig)
