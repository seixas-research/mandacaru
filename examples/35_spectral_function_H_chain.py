# -*- coding: utf-8 -*-
# file: examples/35_spectral_function_H_chain.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The interacting band structure of a 1-D hydrogen chain: ``A(E, k)``.

A band is a sharp line only when the electrons do not interact.  Switch the
two-body term on and a one-particle level stops being an eigenvalue of
anything: adding or removing an electron at crystal momentum ``k`` reaches a
*set* of ``N+-1`` eigenstates, each with some weight.  What replaces the band is
the **spectral function**

.. math::

    A(k, E) = \\sum_m |\\langle m|c_k|\\Psi_0\\rangle|^2\\,\\delta(E - (E_0 - E_m))
            + \\sum_n |\\langle n|c^\\dagger_k|\\Psi_0\\rangle|^2\\,\\delta(E - (E_n - E_0)),

the removal branch plus the addition branch.  Its weight is the answer to "how
much of a real one-particle excitation is this?"; a non-interacting band is the
special case where one pole per ``k`` carries all of it.

``get_spectral_function`` builds it from the correlated ground state the run
already produced, by diagonalizing the Hamiltonian in the ``N-1`` and ``N+1``
particle-number sectors.  Two things are worth knowing before reading the plot:

* **The k-resolution is the mesh, and nothing finer.** The Bloch combinations
  ``c_k = N^-1/2 sum_R e^{-ikR} c_R`` exist only at the commensurate k-points of
  the Born-von Karman supercell.  A smoother band needs a larger mesh, hence a
  larger supercell and more qubits -- here 4 k-points cost 8.
* **Energies are measured from the N-electron ground state**, so removal poles
  are negative and addition poles positive, and the chemical potential falls
  between the branches rather than at zero.

The exact check is the **sum rule**: for each ``k``, spin and orbital the
removal and addition weights must add to exactly 1, since
``{c, c^dagger} = 1``.  The script asserts it.

Writes ``data/h_chain_spectral_function.csv`` and
``data/h_chain_spectral_function.png``.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms

from mandacaru.algorithms import Mandacaru

# All generated files go to examples/data/.
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)

SPACING = 1.0          # H-H distance = lattice constant (Angstrom)
VACUUM = 10.0          # y/z vacuum gap (Angstrom)
N_K = 8                # k-points = cells in the supercell = 2 x qubits
SPACING_GRID = 0.35    # real-space grid spacing (Angstrom)
ETA = 0.15             # Lorentzian broadening (eV)
WINDOW = 35.0          # energy window around E0 to plot (eV)
# At least ~5 grid points across a peak's full width (2 eta), or the
# Lorentzians alias and the map shows sampling rather than structure.
POINTS = int(np.ceil(5 * 2 * WINDOW / ETA))
OPTIMIZER = {"method": "SLSQP", "maxiter": 1000, "tol": 1e-12}

# One H per primitive cell, periodic along x only.
atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]],
              cell=[[SPACING, 0.0, 0.0], [0.0, VACUUM, 0.0], [0.0, 0.0, VACUUM]],
              pbc=[True, False, False])
atoms.calc = Mandacaru(method="bloch-adapt-vqe",
                       kpts={"size": (N_K, 1, 1), "gamma": True},
                       basis="HAO",
                       mapping="jordan_wigner",
                       h=SPACING_GRID,
                       optimizer=OPTIMIZER,
                       max_iterations=12,
                       gradient_tolerance=1e-3,
                       trace=False)

energy_per_cell = atoms.get_potential_energy()
print(f"H chain, {SPACING:.2f} A spacing, {N_K} k-points "
      f"({atoms.calc.n_qubits} qubits), eta = {ETA} eV, {POINTS} energy points")
print(f"  correlated energy : {energy_per_cell:+.6f} eV per cell "
      f"({atoms.calc.result.num_operators} operators)")

# --------------------------------------------------------------------------- #
# The spectral function.
# --------------------------------------------------------------------------- #
# An explicit window: the default spans every pole, including satellites
# carrying ~1e-4 of the weight, which squashes the part worth looking at.
spectral = atoms.calc.get_spectral_function(
    energies=np.linspace(-WINDOW, WINDOW, POINTS), eta=ETA)

print(f"  chemical potential: {spectral.chemical_potential:+.4f} eV")
print(f"  sum rule deviation: {spectral.sum_rule:.2e}  (exact identity: 1)")
assert spectral.sum_rule < 1e-9, "the spectral function violates the sum rule"

# The poles come one set per (orbital, spin); the two spins are degenerate
# here, so collapse them and report the weight per spin channel.
print("\n  poles carrying more than 1 % of the weight (per spin channel):")
print(f"  {'k':>6} {'E - E0 (eV)':>13} {'weight':>8}  branch")
order = np.argsort(spectral.kpoints[:, 0])
for index in order:
    poles, weights = spectral.poles[index]
    merged: dict[float, float] = {}
    for pole, weight in zip(poles, weights):
        merged[round(float(pole), 6)] = merged.get(round(float(pole), 6), 0.0) + weight
    channels = 2 * (len(poles) // len(merged)) // 2 or 1
    for pole in sorted(merged):
        weight = merged[pole] / channels
        if weight > 0.01:
            branch = "removal" if pole <= 0.0 else "addition"
            print(f"  {spectral.kpoints[index][0]:>6.3f} {pole:>13.4f} "
                  f"{weight:>8.4f}  {branch}")

path = spectral.write(os.path.join(DATA, "h_chain_spectral_function.csv"))
print(f"\nwrote {path}")

# --------------------------------------------------------------------------- #
# Plot: A(k, E) as a map, with the poles on top.
# --------------------------------------------------------------------------- #
# The mesh comes in Monkhorst-Pack order (0, 1/4, 1/2, -1/4), so it has to be
# sorted before it can be drawn as an axis.
order = np.argsort(spectral.kpoints[:, 0])
fraction = spectral.kpoints[order, 0]
weights = spectral.weights[order]
poles_by_k = [spectral.poles[index] for index in order]

figure, (band, cut) = plt.subplots(
    1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [2, 1]})

# Left: the spectral weight. The k-points are evenly spaced by construction,
# so a half-step either side gives imshow the exact cell edges.
step = (fraction[1] - fraction[0]) / 2 if fraction.size > 1 else 0.5
mesh = band.imshow(weights.T, origin="lower", aspect="auto", cmap="Blues",
                   vmin=0.0,
                   extent=[fraction[0] - step, fraction[-1] + step,
                           spectral.energies[0], spectral.energies[-1]])
# Mark only the quasiparticle peaks.  At a small eta many satellites clear a
# 1e-3 cutoff and the panel fills with circles that say nothing; 2 % of the
# weight is the level at which a pole is worth pointing at.
MARKER_WEIGHT = 0.02
for index, (poles, amplitude) in enumerate(poles_by_k):
    keep = amplitude > MARKER_WEIGHT
    band.scatter(np.full(keep.sum(), fraction[index]), poles[keep],
                 s=140 * amplitude[keep], facecolors="none",
                 edgecolors="0.15", linewidths=1.2, zorder=3)
band.axhline(spectral.chemical_potential, color="tab:red", lw=1.2, ls="--",
             label=rf"$\mu$ = {spectral.chemical_potential:.2f} eV")
band.set_ylim(spectral.energies[0], spectral.energies[-1])
band.set_xlabel("k (fractional, $\\Gamma \\rightarrow$ X)")
band.set_ylabel("$E - E_0$ (eV)")
band.set_title(f"$A(k, E)$ — H chain, {N_K} k-points, $\\eta$ = {ETA} eV")
band.legend(loc="upper right", framealpha=0.85)
figure.colorbar(mesh, ax=band, label="spectral weight (eV$^{-1}$)")

# Right: the k-resolved cuts, so the weights are readable as numbers.
for index in range(len(fraction)):
    cut.plot(spectral.energies, weights[index], lw=1.4,
             ls="--" if fraction[index] < 0 else "-",
             label=f"k = {fraction[index]:.2f}")
cut.axvline(spectral.chemical_potential, color="tab:red", lw=1.2, ls="--")
cut.set_xlabel("$E - E_0$ (eV)")
cut.set_ylabel("$A(k, E)$ (eV$^{-1}$)")
cut.set_title("cuts at each k-point")
cut.legend(fontsize=8)

figure.tight_layout()
plot_path = os.path.join(DATA, "h_chain_spectral_function.png")
figure.savefig(plot_path, dpi=150)
print(f"wrote {plot_path}")
