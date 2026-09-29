# -*- coding: utf-8 -*-
# file: examples/36_spectral_function_square_lattice.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""A 2-D square lattice: band paths, crystal symmetry and ``A(E, k)``.

In one dimension the Brillouin-zone sampling and the band path are the same
object -- a chain's mesh *is* a line of k-points, so
``examples/35_spectral_function_H_chain.py`` can plot the spectral function
straight against the mesh.  In two dimensions they part company.  The sampling
fills the zone while a band plot is a one-dimensional walk along segments
joining high-symmetry points, here ``G-X-M-G``.

**A path selects k-points; it never invents them.**  The Bloch operators exist
only at the commensurate k-points of the Born-von Karman supercell, so
``calc.band_path("GXMG")`` returns the mesh points that lie *on* those segments
and lists any high-symmetry point the mesh does not carry.  A ``2x2x1`` mesh is
the happy case: its four points are exactly ``G``, ``X``, ``X'`` and ``M``, so
the whole path is resolved at the cost of eight qubits.  A ``3x3x1`` mesh has
thirds and carries neither ``X`` nor ``M``, and the script shows it saying so.

**Symmetry does not come for free.**  ``calc.irreducible_zone()`` reduces the
mesh with the crystal's own space group (found with spglib), and
``get_spectral_function(irreducible=True)`` evaluates only the wedge.  That is
exact **when the state carries the lattice symmetry** -- and this system is a
deliberate counter-example.  Four electrons in four orbitals leave a degenerate
manifold half filled, the mean-field reference picks one member of it, and C4
breaks.  The reduction is therefore *audited*: one extra k-point is evaluated
and compared with the representative it was meant to stand for, and the run
warns rather than quietly returning a symmetrized fiction.  A closed-shell
system -- the 6-k-point chain of example 35 -- reduces exactly instead.

One more thing this lattice is used to show: the periodic grid is made
**commensurate with the primitive cell**, so a primitive lattice translation is
a whole number of grid steps.  Without it Bloch's theorem does not hold on the
grid and k-points the point group says are identical are not.

Writes ``data/square_lattice_spectral_function.csv`` and
``data/square_lattice_spectral_function.png``.
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms

from mandacaru.algorithms import Mandacaru

# All generated files go to examples/data/.
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)

LATTICE = 2.6          # square lattice constant (Angstrom)
VACUUM = 9.0           # vacuum gap along the open direction (Angstrom)
MESH = (2, 2, 1)       # k-point mesh = cells in the supercell
GRID = 0.35            # real-space grid spacing (Angstrom)
PATH = "GXMG"          # the high-symmetry walk to plot along
ETA = 0.30             # Lorentzian broadening (eV)
WINDOW = 20.0          # energy window around E0 (eV)
# At least ~5 samples across a peak's full width (2 eta), or the Lorentzians
# alias and the map shows the sampling rather than the structure.
POINTS = int(np.ceil(5 * 2 * WINDOW / ETA))
OPTIMIZER = {"method": "SLSQP", "maxiter": 1000, "tol": 1e-12}


def square_lattice(mesh=MESH):
    """One H per primitive cell, periodic in x and y."""
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]],
                  cell=[[LATTICE, 0.0, 0.0],
                        [0.0, LATTICE, 0.0],
                        [0.0, 0.0, VACUUM]],
                  pbc=[True, True, False])
    atoms.calc = Mandacaru(method="bloch-adapt-vqe",
                           kpts={"size": mesh, "gamma": True},
                           basis="HAO",
                           mapping="jordan_wigner",
                           h=GRID,
                           optimizer=OPTIMIZER,
                           trace=False)
    return atoms


# --------------------------------------------------------------------------- #
# The lattice, its symmetry and its path -- none of this needs a calculation.
# --------------------------------------------------------------------------- #
atoms = square_lattice()
calc = atoms.calc

print(f"Square lattice of H, a = {LATTICE} A, mesh {MESH[0]}x{MESH[1]}x{MESH[2]}")
print(f"  space group     : {calc.symmetry().summary()}")
print(f"  irreducible zone: {calc.irreducible_zone().summary()}")
print(f"  default path    : {calc.band_path().path}")

kpath = calc.band_path(PATH)
print(f"  {kpath.summary()}")
for label, distance in zip(kpath.labels, kpath.label_distances):
    print(f"      {label:<2} at {distance:7.4f} 1/A")

# The same question on a mesh that cannot answer it: 3x3x1 has thirds, and
# X = (0, 1/2, 0) and M = (1/2, 1/2, 0) are at halves.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    coarse = square_lattice(mesh=(3, 3, 1)).calc.band_path(PATH)
print(f"\n  on a 3x3x1 mesh the same path is incomplete: "
      f"missing {', '.join(coarse.missing)}")
# ASE emits unrelated NumPy deprecations from the same block, so pick ours out
# by category and content rather than taking whatever arrived first.
ours = [record for record in caught
        if issubclass(record.category, RuntimeWarning)
        and "not on this" in str(record.message)]
print(f"      (and warns: ...{str(ours[0].message).split('so the')[-1].strip()})")

# --------------------------------------------------------------------------- #
# Solve, then read the spectral function along the path.
# --------------------------------------------------------------------------- #
energy_per_cell = atoms.get_potential_energy()
print(f"\n  energy          : {energy_per_cell:+.6f} eV per cell "
      f"({calc.n_qubits} qubits, {calc.result.num_operators} operators)")

energies = np.linspace(-WINDOW, WINDOW, POINTS)
spectral = calc.get_spectral_function(energies=energies, eta=ETA, path=PATH)

print(f"  chemical potential: {spectral.chemical_potential:+.4f} eV")
print(f"  sum rule deviation: {spectral.sum_rule:.2e}  (exact identity: 1)")
assert spectral.sum_rule < 1e-9, "the spectral function violates the sum rule"

bands = calc.bands(spectral=spectral)
weights = calc.band_weights(spectral=spectral)
print("\n  quasiparticle peak along the path:")
print(f"  {'label':>6} {'k':>22} {'x (1/A)':>9} {'E - E0 (eV)':>12} {'Z':>7}")
labels_at = dict(zip(np.round(kpath.label_distances, 10), kpath.labels))
for index, distance in enumerate(spectral.kpath.distances):
    label = labels_at.get(round(float(distance), 10), "")
    fractional = np.round(spectral.kpoints[index], 3).tolist()
    print(f"  {label:>6} {str(fractional):>22} {distance:>9.4f} "
          f"{bands[index, 0]:>12.4f} {weights[index, 0]:>7.4f}")

# --------------------------------------------------------------------------- #
# The irreducible zone, audited.
# --------------------------------------------------------------------------- #
# A(E, k) has the symmetry of the STATE, not of the lattice.  Half filling puts
# a degenerate manifold at the Fermi level, the reference picks one member, and
# C4 breaks -- so the wedge is not enough here, and the audit says so.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    reduced = calc.get_spectral_function(energies=energies, eta=ETA,
                                         irreducible=True)
full = calc.get_spectral_function(energies=energies, eta=ETA)
residual = reduced.irreducible.symmetry_residual
print(f"\n  irreducible=True on {len(full.kpoints)} k-points -> "
      f"{len(reduced.irreducible.points)} evaluated "
      f"({reduced.irreducible.reduction:.2f}x)")
print(f"  symmetry residual : {residual:.3e}  "
      f"({'broken' if residual > 1e-6 else 'clean'})")
print(f"  |A_full - A_ibz|  : {np.abs(full.weights - reduced.weights).max():.3e}")
if caught:
    print("  -> the run warned, which is the point: the lattice has C4 and "
          "this half-filled state does not.")

target = spectral.write(os.path.join(DATA,
                                     "square_lattice_spectral_function.csv"))
print(f"\nwrote {target}")

# --------------------------------------------------------------------------- #
# Plot: A(E, k) along the path, and the mesh inside the Brillouin zone.
# --------------------------------------------------------------------------- #
figure, (band, zone) = plt.subplots(
    1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [2, 1]})

# Left: the spectral weight along G-X-M-G.  The path points are NOT evenly
# spaced (the segments have different lengths), so the cell edges are built
# from midpoints rather than from a constant step.
x = spectral.kpath.distances
edges = np.concatenate([[x[0] - 0.5 * (x[1] - x[0])],
                        0.5 * (x[1:] + x[:-1]),
                        [x[-1] + 0.5 * (x[-1] - x[-2])]])
# The energy grid is uniform, so its edges are a half step either side.
half = 0.5 * (energies[1] - energies[0])
energy_edges = np.concatenate([energies - half, [energies[-1] + half]])
mesh_plot = band.pcolormesh(edges, energy_edges, spectral.weights.T,
                            cmap="Blues", vmin=0.0)
figure.colorbar(mesh_plot, ax=band, label=r"$A(E,\mathbf{k})$  (eV$^{-1}$)")

# The poles themselves, sized by weight: the map is a broadened picture of them.
for index, (poles, amplitudes) in enumerate(spectral.poles):
    keep = amplitudes > 0.02
    band.scatter(np.full(keep.sum(), x[index]), poles[keep],
                 s=140 * amplitudes[keep], facecolors="none",
                 edgecolors="0.15", linewidths=0.9, zorder=3)

band.axhline(spectral.chemical_potential, color="tab:red", lw=1.2, ls="--",
             label=rf"$\mu$ = {spectral.chemical_potential:.2f} eV")
for distance in kpath.label_distances:
    band.axvline(distance, color="0.35", lw=0.8, alpha=0.6)
band.set_xticks(kpath.label_distances)
band.set_xticklabels([r"$\Gamma$" if label == "G" else label
                      for label in kpath.labels])
band.set_xlim(edges[0], edges[-1])
band.set_ylim(-WINDOW, WINDOW)
band.set_ylabel(r"$E - E_0$  (eV)")
band.set_title(rf"$A(E,\mathbf{{k}})$ along {PATH.replace('G', 'Γ')}"
               f"  ({calc.n_qubits} qubits)")
band.legend(loc="upper right", framealpha=0.9)

# Right: where those k-points sit in the zone, and which ones the symmetry
# reduction would have evaluated.
full_mesh = calc.get_bz_k_points()
wedge = calc.get_ibz_k_points()
zone.add_patch(plt.Rectangle((-0.5, -0.5), 1.0, 1.0, fill=False,
                             edgecolor="0.6", lw=1.0, ls=":"))
zone.plot(np.append(kpath.points[:, 0], kpath.points[0, 0]),
          np.append(kpath.points[:, 1], kpath.points[0, 1]),
          color="tab:red", lw=1.4, zorder=2, label=PATH)
zone.scatter(full_mesh[:, 0], full_mesh[:, 1], s=110, facecolors="none",
             edgecolors="0.35", zorder=3, label="mesh")
# Drawn last and larger, or the path line sits on top of the very points the
# panel exists to distinguish.
zone.scatter(wedge[:, 0], wedge[:, 1], s=46, color="tab:blue", zorder=4,
             label="irreducible")
zone.margins(0.18)
for label, point in kpath.special_points.items():
    if label in kpath.labels:
        zone.annotate(r"$\Gamma$" if label == "G" else label,
                      (point[0], point[1]), textcoords="offset points",
                      xytext=(6, 5), color="tab:red")
zone.set_xlabel(r"$k_1$")
zone.set_ylabel(r"$k_2$")
zone.set_title("Brillouin-zone sampling")
zone.set_aspect("equal")
zone.legend(loc="lower left", fontsize=8, framealpha=0.9)

figure.tight_layout()
image = os.path.join(DATA, "square_lattice_spectral_function.png")
figure.savefig(image, dpi=150)
print(f"wrote {image}")
