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
the two-body electron-repulsion tensor ``(ab|cd)`` in the chemists' convention.

The nuclear potential uses the *true* nuclear charges (Z_Li = 3, Z_H = 1),
while the hydrogenic basis orbitals use *effective* charges (Slater's rules) so
that the contracted Li core is representable on a modest real-space grid.

Run with::

    python examples/lih_integrals.py
"""

from __future__ import annotations

import numpy as np

from carcara.basis import HydrogenicOrbital
from carcara.integrals import Grid, IntegralEngine, Potentials

# Nuclear charges and geometry (atomic units, Bohr).  The equilibrium bond
# length of LiH is ~3.015 a0; Li and H sit along z about the origin.
Z_LI = 3.0
Z_H = 1.0
R = 3.015
li_pos = np.array([0.0, 0.0, -R / 2])
h_pos = np.array([0.0, 0.0, +R / 2])

# External potential with the *true* nuclear charges (Z_Li = 3, Z_H = 1),
# independent of the effective basis charges chosen below.
potentials = Potentials([(Z_LI, li_pos), (Z_H, h_pos)])

# A cubic real-space grid large enough to contain the diffuse Li 2s/2p tails.
grid = Grid(center=[0.0, 0.0, 0.0], box_size=9.0, points=72)

# Minimal basis with effective (Slater) charges: Li {1s, 2s, 2p_z} + H {1s}.
labels = ["Li 1s", "Li 2s", "Li 2pz", "H 1s"]
basis = [HydrogenicOrbital(1, 0, 0, Z=2.69, center=li_pos),   # Li 1s core
         HydrogenicOrbital(2, 0, 0, Z=1.28, center=li_pos),   # Li 2s valence
         HydrogenicOrbital(2, 1, 0, Z=1.28, center=li_pos),   # Li 2pz valence
         HydrogenicOrbital(1, 0, 0, Z=1.00, center=h_pos)]    # H 1s

engine = IntegralEngine(basis, grid)
print(f"C backend in use: {engine.uses_c_backend}")
print("Basis order:", ", ".join(f"{i}={l}" for i, l in enumerate(labels)))

# One-body integrals: kinetic T[a,b] and nuclear attraction V[a,b].
T, V = engine.one_body(potentials.nuclear_potential)
h_core = T + V  # the one-body core Hamiltonian

# Two-body electron-repulsion tensor (ab|cd), chemists' notation.
eri = engine.two_body(method="fft")

np.set_printoptions(precision=4, suppress=True)
print("\nKinetic energy matrix T (Ha):")
print(T.real)
print("\nNuclear attraction matrix V (Ha):")
print(V.real)
print("\nCore Hamiltonian h = T + V (Ha):")
print(h_core.real)
print("\nSelected two-body integrals (Ha):")
print(f"  (00|00) = {eri[0, 0, 0, 0].real:.4f}   Li 1s on-site repulsion")
print(f"  (33|33) = {eri[3, 3, 3, 3].real:.4f}   H 1s on-site repulsion")
print(f"  (11|33) = {eri[1, 1, 3, 3].real:.4f}   Li 2s - H 1s Coulomb")
print(f"  (13|13) = {eri[1, 3, 1, 3].real:.4f}   Li 2s - H 1s exchange")
