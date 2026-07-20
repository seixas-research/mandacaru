# -*- coding: utf-8 -*-
# file: examples/19_pseudopotential_generation.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Norm-conserving pseudopotentials: Troullier-Martins + Kleinman-Bylander.

Generates pseudopotentials from scratch -- an all-electron LDA atom, the
Troullier-Martins pseudization of each valence channel, unscreening, and the
Kleinman-Bylander separable transformation -- and then measures the thing they
exist to fix.

Why they exist here
-------------------
Carcará integrates on a uniform real-space grid.  The :math:`1s` shell of a
heavy atom has length scale :math:`a_0/Z` -- 0.066 Angstrom for oxygen -- while a
practical grid spacing is 0.15-0.30 Angstrom.  That mismatch produced, in
``examples/18_H2O_geometry_relaxation.py``, a spurious force of ~3400 eV/Angstrom
on an isolated oxygen atom (whose exact force is zero) and a 178 eV egg-box error
for water, neither of which converged under grid refinement.  Removing the core
removes the cause.

Output: ``examples/data/pseudopotential_O.png`` and a validation table.
"""

from __future__ import annotations

import os

import numpy as np

from carcara.basis import generate_pseudopotential, solve_atom
from carcara.basis.pseudopotential import check_channel, report
from carcara.units import BOHR_TO_ANGSTROM

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)
PNG_PATH = os.path.join(DATA, "pseudopotential_O.png")
RULE = "=" * 76

# --------------------------------------------------------------------------- #
# 1. Generate across the first rows.
# --------------------------------------------------------------------------- #

print(RULE)
print("1. Generating norm-conserving pseudopotentials from scratch")
print(RULE)
print(f"{'atom':>5}{'Z_ion':>7}{'channels':>11}{'local':>7}"
      f"{'norm err':>12}{'tail err':>11}{'eps err':>12}{'V(0) Ha':>11}")

potentials = {}
for symbol in ("H", "Li", "Be", "C", "N", "O", "F"):
    pp = generate_pseudopotential(symbol)
    potentials[symbol] = pp
    worst_norm = max(check_channel(pp, l)["norm_error"] for l in pp.channels)
    worst_tail = max(check_channel(pp, l)["tail_error"] for l in pp.channels)
    worst_eps = max(abs(check_channel(pp, l)["eigenvalue_error"])
                    for l in pp.channels)
    channels = ",".join(f"l{l}" for l in sorted(pp.channels))
    print(f"{symbol:>5}{pp.valence_charge:>7.0f}{channels:>11}"
          f"{'l' + str(pp.local_l):>7}{worst_norm:>12.1e}{worst_tail:>11.1e}"
          f"{worst_eps:>12.1e}{pp.v_local[0]:>11.2f}")

print("\nnorm err : |1 - Q_ps/Q_ae| inside r_c -- norm conservation is what makes")
print("           the pseudopotential transferable between environments.")
print("tail err : |R_ps - R_ae| beyond r_c -- must be exactly zero.")
print("eps err  : eigenvalue recovered by re-solving in the pseudopotential.")
print("V(0)     : finite, where the all-electron potential diverges as -Z/r.")

# --------------------------------------------------------------------------- #
# 2. Oxygen in detail.
# --------------------------------------------------------------------------- #

oxygen = potentials["O"]
print(f"\n{RULE}")
print("2. Oxygen in detail")
print(RULE)
print(report(oxygen))
print("\nThe Kleinman-Bylander form replaces the semilocal sum over channels")
print("with one rank-one projector each: for M basis functions the nonlocal")
print("matrix costs O(M) projections instead of O(M^2) radial integrals.")

# --------------------------------------------------------------------------- #
# 3. The payoff: grid sensitivity.
# --------------------------------------------------------------------------- #

print(f"\n{RULE}")
print("3. Egg-box error: energy change when the atom is translated across the")
print("   grid.  The exact answer is zero -- nothing physical depends on where")
print("   a grid happens to be placed.")
print(RULE)


def egg_box(pp, spacing_angstrom, all_electron=False):
    """Range of ``int rho V`` over sub-grid offsets (Hartree)."""
    spacing = spacing_angstrom / BOHR_TO_ANGSTROM
    axis = (np.arange(31) - 15) * spacing
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    values = []
    for fraction in np.linspace(0.0, 1.0, 5)[:-1]:
        shift = spacing * fraction
        radius = np.sqrt((X - shift) ** 2 + (Y - shift) ** 2 + (Z - shift) ** 2)
        if all_electron:
            density = np.interp(radius, pp.atom.r, pp.atom.density,
                                left=pp.atom.density[0], right=0.0)
            # The engine's own regularization: -Z/max(r, half a grid step).
            potential = -pp.atomic_number / np.maximum(radius, 0.5 * spacing)
        else:
            density = np.interp(radius, pp.r, pp.valence_density,
                                left=pp.valence_density[0], right=0.0)
            potential = pp.local_potential(radius)
        values.append(float(np.sum(density * potential) * spacing ** 3))
    return max(values) - min(values)


print(f"{'h (A)':>8}{'all-electron':>18}{'pseudopotential':>19}{'improvement':>14}")
for spacing in (0.30, 0.20, 0.15):
    bare = egg_box(oxygen, spacing, all_electron=True)
    pseudo = egg_box(oxygen, spacing)
    print(f"{spacing:>8.2f}{bare:>18.4f}{pseudo:>19.4f}{bare / pseudo:>13.0f}x")

print("\nThe all-electron column barely improves with refinement -- that is the")
print("pathology.  The pseudopotential column falls steeply, which is what makes")
print("geometry optimization tractable.")

# --------------------------------------------------------------------------- #
# 4. Plot.
# --------------------------------------------------------------------------- #

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:                                     # pragma: no cover
    raise SystemExit("matplotlib is not installed; the tables were still printed")

atom = oxygen.atom
r = oxygen.r
fig, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.8))

# (a) Orbitals: pseudo vs all-electron.
colors = {0: "#0072B2", 1: "#D55E00"}
for l, channel in sorted(oxygen.channels.items()):
    all_electron = atom.radial(channel.n, l)
    left.plot(r, all_electron * r, color=colors[l], lw=1.4, ls="--",
              label=f"all-electron {channel.n}{'sp'[l]}")
    left.plot(r, channel.pseudo_radial * r, color=colors[l], lw=2.2,
              label=f"pseudo {channel.n}{'sp'[l]}")
    left.axvline(channel.r_cut, color=colors[l], lw=0.9, alpha=0.5)
left.set_xlim(0, 5)
left.set_xlabel("r (Bohr)")
left.set_ylabel(r"$u(r) = r\,R(r)$")
left.set_title("(a) Pseudo-orbitals: nodeless inside $r_c$, exact outside")
left.grid(True, color="0.92", lw=0.8)
left.legend(frameon=False, fontsize=9)

# (b) Potentials.
right.plot(r, -oxygen.atomic_number / r, color="0.4", lw=1.6, ls=":",
           label=r"all-electron $-Z/r$")
right.plot(r, oxygen.v_local, color="#009E73", lw=2.4,
           label=f"pseudopotential $V_{{loc}}$ (l={oxygen.local_l})")
right.plot(r, -oxygen.valence_charge / np.maximum(r, 1e-9), color="0.4",
           lw=1.2, ls="--", label=r"ionic tail $-Z_{ion}/r$")
right.set_xlim(0, 5)
right.set_ylim(-40, 5)
right.set_xlabel("r (Bohr)")
right.set_ylabel("V (Hartree)")
right.set_title("(b) The singularity is gone: $V(0)$ is finite and flat")
right.grid(True, color="0.92", lw=0.8)
right.legend(frameon=False, fontsize=9)

for axis in (left, right):
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)

fig.suptitle("Troullier-Martins norm-conserving pseudopotential for oxygen "
             f"($Z_{{ion}} = {oxygen.valence_charge:g}$, generated from scratch)",
             fontsize=12)
fig.tight_layout()
fig.savefig(PNG_PATH, dpi=150)
print(f"\nwrote {PNG_PATH}")

print(f"\n{RULE}")
print("USING THEM")
print(RULE)
print("These are wired all the way through the integral engine: the smooth")
print("local potential replaces -Z/r, and the Kleinman-Bylander term enters the")
print("one-body Hamiltonian as a rank-one update per channel,")
print("  H_nl[mu,nu] = sum_l <phi_mu|chi_lm> E_KB,l <chi_lm|phi_nu>,")
print("with the derivative of those projector overlaps carried into the force.")
print()
print("Any driver takes them as one argument:")
print("    atoms.calc = ADAPTVQE(basis='FAO', pseudopotentials=True, h=0.15)")
print()
print("The bundled library covers every element with Z < 90 and is loaded by")
print("symbol.  See example 20 for what it buys.")
print(RULE)
