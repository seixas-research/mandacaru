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
from carcara.integrals import Grid, IntegralEngine

# Nuclear charges and geometry (atomic units, Bohr).  The equilibrium bond
# length of H2 is ~1.4 a0; the two protons sit symmetrically about the origin.
Z = 1.0
R = 1.4
nuclei = np.array([[0.0, 0.0, -R / 2],
                   [0.0, 0.0, +R / 2]])


def nuclear_potential(x, y, z):
    """External potential of the two protons: ``V(r) = -sum_A Z / |r - R_A|``."""
    v = np.zeros_like(x, dtype=float)
    for Rx, Ry, Rz in nuclei:
        r = np.sqrt((x - Rx) ** 2 + (y - Ry) ** 2 + (z - Rz) ** 2)
        v -= Z / np.maximum(r, 1e-12)
    return v


def main():
    # A cubic real-space grid large enough to contain both 1s tails.
    grid = Grid(center=[0.0, 0.0, 0.0], box_size=10.0, points=64)

    # Minimal basis: one 1s orbital centered on each proton.
    basis = [HydrogenicOrbital(1, 0, 0, Z=Z, center=nuclei[0]),
             HydrogenicOrbital(1, 0, 0, Z=Z, center=nuclei[1])]

    engine = IntegralEngine(basis, grid)
    print(f"C backend in use: {engine.uses_c_backend}")

    # One-body integrals: kinetic T[a,b] and nuclear attraction V[a,b].
    T, V = engine.one_body(nuclear_potential)
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
    print(f"  (00|00) = {eri[0, 0, 0, 0].real:.4f}   on-site repulsion")
    print(f"  (00|11) = {eri[0, 0, 1, 1].real:.4f}   inter-site Coulomb")
    print(f"  (01|01) = {eri[0, 1, 0, 1].real:.4f}   exchange")


if __name__ == "__main__":
    main()
