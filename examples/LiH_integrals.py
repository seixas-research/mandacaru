# -*- coding: utf-8 -*-
# file: examples/lih_integrals.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""One- and two-body integrals for the LiH molecule.

Builds a small minimal basis -- the Li 1s, 2s and 2p_z orbitals plus the H 1s
orbital -- and uses :class:`~carcara.integrals.IntegralEngine` to compute the
real-space one-body (kinetic ``T`` and nuclear-attraction ``V``) matrices and
the two-body electron-repulsion tensor ``<ab|cd>`` in the physicists' convention.

The nuclear potential uses the *true* nuclear charges (Z_Li = 3, Z_H = 1),
while the hydrogenic basis orbitals use *effective* charges (Slater's rules) so
that the contracted Li core is representable on a modest real-space grid.

Run with::

    python examples/lih_integrals.py
"""

from __future__ import annotations

import numpy as np

from carcara.basis import FullAtomicOrbital
from carcara.integrals import Grid, IntegralEngine, Potentials

# Nuclear charges and geometry.  Lengths are in the user-facing unit (Angstrom):
# the equilibrium bond length of LiH is ~1.60 A; Li and H sit along z about the
# origin.
Z_LI = 3.0
Z_H = 1.0
R = 1.595
li_pos = np.array([0.0, 0.0, -R / 2])
h_pos = np.array([0.0, 0.0, +R / 2])

# External potential with the *true* nuclear charges (Z_Li = 3, Z_H = 1),
# independent of the effective basis charges chosen below.
potentials = Potentials([(Z_LI, li_pos), (Z_H, h_pos)])

# A cubic real-space grid large enough to contain the diffuse Li 2s/2p tails.
# A fine spacing (h = 0.10 Angstrom) is used to resolve the contracted Li 1s core.
grid = Grid(center=[0.0, 0.0, 0.0], box_size=4.8, h=0.10)

# Minimal basis with effective charges from Slater's rules: Li {1s, 2s, 2p_z}
# + H {1s}.  ``from_slater`` derives Z_eff from the atomic number of the center.
labels = ["Li 1s", "Li 2s", "Li 2pz", "H 1s"]
basis = [FullAtomicOrbital.from_slater(1, 0, 0, atomic_number=3, center=li_pos),
         FullAtomicOrbital.from_slater(2, 0, 0, atomic_number=3, center=li_pos),
         FullAtomicOrbital.from_slater(2, 1, 0, atomic_number=3, center=li_pos),
         FullAtomicOrbital.from_slater(1, 0, 0, atomic_number=1, center=h_pos)]

engine = IntegralEngine(basis, grid)
print(f"C backend in use: {engine.uses_c_backend}")
print("Basis order:", ", ".join(f"{i}={l}" for i, l in enumerate(labels)))
print("Slater Z_eff:", ", ".join(f"{l}={o.Z:.2f}" for l, o in zip(labels, basis)))

# One-body integrals: kinetic T[a,b] and nuclear attraction V[a,b].  Energies
# are returned in the user-facing unit (eV) by default.
T, V = engine.one_body(potentials.nuclear_potential)
h_core = T + V  # the one-body core Hamiltonian

# Two-body electron-repulsion tensor <ab|cd>, physicists' notation.
eri = engine.two_body(method="fft")

np.set_printoptions(precision=3, suppress=True)
print("\nKinetic energy matrix T (eV):")
print(T.real)
print("\nNuclear attraction matrix V (eV):")
print(V.real)
print("\nCore Hamiltonian h = T + V (eV):")
print(h_core.real)
print("\nSelected two-body integrals (eV):")
print(f"  <00|00> = {eri[0, 0, 0, 0].real:.3f}   Li 1s on-site repulsion")
print(f"  <33|33> = {eri[3, 3, 3, 3].real:.3f}   H 1s on-site repulsion")
print(f"  <13|13> = {eri[1, 3, 1, 3].real:.3f}   Li 2s - H 1s Coulomb")
print(f"  <11|33> = {eri[1, 1, 3, 3].real:.3f}   Li 2s - H 1s exchange")
