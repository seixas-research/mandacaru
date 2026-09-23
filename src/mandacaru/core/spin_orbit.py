# -*- coding: utf-8 -*-
# file: core/spin_orbit.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Spin-orbit coupling as a one-body term of the many-body Hamiltonian.

A pseudopotential generated with ``relativity="dirac"`` carries a spin-orbit
term alongside its ordinary channels
(:func:`mandacaru.pseudopotentials.oncv.generate_oncv`): each :math:`l` has a
separable radial part and the angular operator
:math:`\mathbf{L}\cdot\mathbf{S}`,

.. math::

    V_{SO} = \sum_{a}\sum_{l}\sum_{ij}\sum_{mm'}
        |\chi^{a,l}_i Y_{lm}\rangle\, D^{SO,a,l}_{ij}\,
        \langle lm\sigma|\mathbf{L}\cdot\mathbf{S}|lm'\sigma'\rangle\,
        \langle \chi^{a,l}_j Y_{lm'}| .

This module turns that into the one-body matrix
:math:`h^{SO}_{P Q}` over spin-orbitals, which
:meth:`~mandacaru.core.hamiltonian.MolecularIntegrals.molecular_hamiltonian`
adds to :math:`h`.

What spin-orbit coupling costs
------------------------------

Everything else in Mandacaru's one-body matrix is **block-diagonal in spin and
diagonal in** :math:`m`: :meth:`~.hamiltonian.MolecularIntegrals.kb_nonlocal`
returns an :math:`(M, M)` spatial matrix, and
:func:`~.hamiltonian.spin_block_integrals` copies it into the
:math:`\alpha\alpha` and :math:`\beta\beta` blocks.  :math:`\mathbf{L}\cdot
\mathbf{S}` breaks both:

- :math:`\tfrac12(L_+S_- + L_-S_+)` connects :math:`|m,\alpha\rangle` to
  :math:`|m{+}1,\beta\rangle`, so :math:`h` acquires an
  :math:`\alpha\beta` block and **is no longer block-diagonal in spin**;
- that block makes :math:`S_z` no longer a good quantum number, only
  :math:`J_z`.  The Hamiltonian still conserves particle number, so the
  Jordan-Wigner register and the variational machinery still apply, but
  anything that assumed a fixed :math:`(n_\alpha, n_\beta)` -- the
  particle-number **sector** reduction, the parity mapping's two-qubit taper,
  and the excitation pools -- has to be told not to.
  :meth:`~.hamiltonian.MolecularIntegrals.molecular_hamiltonian` refuses the
  combinations that would silently return the wrong answer rather than
  quietly dropping the term.

:math:`h^{SO}` is complex and Hermitian.  That is not a problem in itself: the
fermionic operator it builds is Hermitian, so its Pauli expansion has real
coefficients -- it simply contains :math:`XY` and :math:`YX` strings that a
real :math:`h` never produces.
"""

from __future__ import annotations

import numpy as np

#: Spin index of the alpha (``S_z = +1/2``) and beta blocks, matching the
#: spin-orbital ordering ``P = p + sigma * M`` used throughout.
ALPHA, BETA = 0, 1


def ls_matrix(l: int) -> np.ndarray:
    r""":math:`\mathbf{L}\cdot\mathbf{S}` in the :math:`|lm\sigma\rangle` basis.

    Ordered spin-major -- index ``sigma * (2l + 1) + (m + l)`` -- so the
    :math:`\alpha\alpha`, :math:`\alpha\beta`, :math:`\beta\alpha` and
    :math:`\beta\beta` blocks are the four quadrants.  With complex spherical
    harmonics (which is what :mod:`mandacaru.basis._angular` builds)
    :math:`L_z` is diagonal and :math:`L_\pm` are the textbook ladder
    operators:

    .. math::

        \mathbf{L}\cdot\mathbf{S} = L_zS_z
            + \tfrac12\big(L_+S_- + L_-S_+\big),

    giving :math:`\pm m/2` on the diagonal and
    :math:`\tfrac12\sqrt{l(l+1)-m(m\pm1)}` off it.

    The eigenvalues are :math:`l/2` with multiplicity :math:`2l+2` and
    :math:`-(l+1)/2` with multiplicity :math:`2l` -- the two :math:`j` levels,
    which is what ``test/core/test_spin_orbit.py`` checks.
    """
    l = int(l)
    size = 2 * l + 1
    out = np.zeros((2 * size, 2 * size), dtype=complex)

    def at(sigma, m):
        return sigma * size + (m + l)

    for m in range(-l, l + 1):
        out[at(ALPHA, m), at(ALPHA, m)] = 0.5 * m
        out[at(BETA, m), at(BETA, m)] = -0.5 * m
        if m + 1 <= l:
            # L_+ S_-: |m, alpha> -> |m+1, beta>
            element = 0.5 * np.sqrt(l * (l + 1) - m * (m + 1))
            out[at(BETA, m + 1), at(ALPHA, m)] = element
            out[at(ALPHA, m), at(BETA, m + 1)] = element
    return out


def _projector_index(projectors):
    """``{(atom, l): {(m, i): position}}`` over the flat projector list."""
    index: dict = {}
    for position, projector in enumerate(projectors):
        key = (projector.atom_index, projector.l)
        index.setdefault(key, {})[(projector.m, projector.index)] = position
    return index


def spin_orbit_one_body(projections, projectors, blocks) -> np.ndarray:
    r"""The ``(2M, 2M)`` spin-orbital matrix of :math:`V_{SO}`.

    Parameters
    ----------
    projections : ndarray
        ``C[mu, p] = <phi_mu|chi_p>``, the same ``(M, P)`` array
        :meth:`~.hamiltonian.MolecularIntegrals.projections` returns.  The
        spin-orbit projectors must be *these* projectors: the radial parts of
        the two ``j`` branches, in the order
        :func:`~mandacaru.pseudopotentials.oncv.oncv_projectors` emits them.
    projectors : list
        The :class:`~mandacaru.pseudopotentials.orbitals.KBProjector` list, for
        the ``(atom, l, m, index)`` labels.
    blocks : dict
        ``{(atom, l): D_SO}`` -- the spin-orbit coupling of that channel,
        ``(n, n)`` for its ``n`` radial projectors.  A channel absent from
        ``blocks`` contributes nothing, which is how ``l = 0`` (no spin-orbit
        term at all) and a scalar-relativistic atom in the same molecule are
        handled.

    Returns
    -------
    ndarray
        Complex Hermitian ``(2M, 2M)``, in the spin-blocked ordering
        ``P = p + sigma * M``.
    """
    C = np.asarray(projections)
    M = C.shape[0]
    out = np.zeros((2 * M, 2 * M), dtype=complex)
    if not blocks:
        return out

    index = _projector_index(projectors)
    for (atom, l), positions in index.items():
        D = blocks.get((atom, l))
        if D is None or int(l) == 0:
            continue                      # no spin-orbit term for an s channel
        D = np.asarray(D, dtype=complex)
        ms = sorted({m for m, _i in positions})
        radial = sorted({i for _m, i in positions})
        if D.shape != (len(radial), len(radial)):
            raise ValueError(
                f"spin-orbit block for (atom {atom}, l={l}) must be "
                f"({len(radial)}, {len(radial)}) for its {len(radial)} radial "
                f"projectors, got {D.shape}")

        # C_al[mu, m, i]
        C_al = np.zeros((M, len(ms), len(radial)), dtype=complex)
        for a, m in enumerate(ms):
            for b, i in enumerate(radial):
                C_al[:, a, b] = C[:, positions[(m, i)]]

        LS = ls_matrix(l)
        size = 2 * l + 1
        if len(ms) != size:
            raise ValueError(
                f"channel (atom {atom}, l={l}) has {len(ms)} m values, "
                f"expected {size}")
        # T[mu, m, j] = sum_i C_al[mu, m, i] D_ij
        T = np.einsum("umi,ij->umj", C_al, D)
        for sigma in (ALPHA, BETA):
            for tau in (ALPHA, BETA):
                quadrant = LS[sigma * size:(sigma + 1) * size,
                              tau * size:(tau + 1) * size]
                block = np.einsum("umj,mn,vnj->uv", T, quadrant,
                                  C_al.conj())
                out[sigma * M:(sigma + 1) * M,
                    tau * M:(tau + 1) * M] += block
    return 0.5 * (out + out.conj().T)


def breaks_spin_symmetry(h_spin_orbit, tolerance: float = 1e-12) -> bool:
    """Whether the term actually couples the two spin blocks.

    Zero for a molecule of scalar-relativistic atoms, and zero for s-only
    atoms (hydrogen), where the ``L . S`` operator vanishes identically -- so a
    calculation that asks for spin-orbit coupling and gets nothing is told so
    rather than paying for a broken :math:`S_z`.
    """
    h = np.asarray(h_spin_orbit)
    M = h.shape[0] // 2
    return bool(np.max(np.abs(h[:M, M:])) > tolerance)
