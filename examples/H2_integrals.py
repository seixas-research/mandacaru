# -*- coding: utf-8 -*-
# file: examples/h2_integrals.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""One- and two-body integrals for the H2 molecule.

Builds a minimal basis of one hydrogen 1s orbital on each nucleus, then uses
:class:`~carcara.integrals.IntegralEngine` to compute the real-space one-body
(kinetic ``T`` and nuclear-attraction ``V``) matrices and the two-body
electron-repulsion tensor ``(ab|cd)`` in the chemists' convention.

Run with::

    python examples/h2_integrals.py
"""

from __future__ import annotations

import numpy as np

from carcara.basis import HydrogenicOrbital
from carcara.integrals import Grid, IntegralEngine, Potentials

# Nuclear charges and geometry.  Lengths are in the user-facing unit (Angstrom):
# the equilibrium bond length of H2 is ~0.74 A, the two protons sitting
# symmetrically about the origin.
Z = 1.0
R = 0.74
proton_a = np.array([0.0, 0.0, -R / 2])
proton_b = np.array([0.0, 0.0, +R / 2])

# External electron-nuclear potential V(r) = -sum_A Z / |r - R_A|.
potentials = Potentials([(Z, proton_a), (Z, proton_b)])

# A cubic real-space grid large enough to contain both 1s tails, sampled at a
# spacing of h = 0.10 Angstrom.
grid = Grid(center=[0.0, 0.0, 0.0], box_size=5.0, h=0.10)

# Minimal basis: one 1s orbital centered on each proton.
basis = [HydrogenicOrbital(1, 0, 0, Z=Z, center=proton_a),
            HydrogenicOrbital(1, 0, 0, Z=Z, center=proton_b)]

engine = IntegralEngine(basis, grid)
print(f"C backend in use: {engine.uses_c_backend}")

# One-body integrals: kinetic T[a,b] and nuclear attraction V[a,b].  Energies
# are returned in the user-facing unit (eV) by default.
T, V = engine.one_body(potentials.nuclear_potential)
h_core = T + V  # the one-body core Hamiltonian

# Two-body electron-repulsion tensor (ab|cd), chemists' notation.
eri = engine.two_body(method="fft")

np.set_printoptions(precision=3, suppress=True)
print("\nKinetic energy matrix T (eV):")
print(T.real)
print("\nNuclear attraction matrix V (eV):")
print(V.real)
print("\nCore Hamiltonian h = T + V (eV):")
print(h_core.real)
print("\nSelected two-body integrals (eV):")
print(f"  (00|00) = {eri[0, 0, 0, 0].real:.3f}   on-site repulsion")
print(f"  (00|11) = {eri[0, 0, 1, 1].real:.3f}   inter-site Coulomb")
print(f"  (01|01) = {eri[0, 1, 0, 1].real:.3f}   exchange")

