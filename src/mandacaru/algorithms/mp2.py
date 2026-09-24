# -*- coding: utf-8 -*-
# file: algorithms/mp2.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Second-order Moller-Plesset theory, and the natural orbitals it defines.

This module exists for one purpose: to decide **which orbitals are worth
putting on a qubit register**.  A large basis -- PAW-LCAO-TZP, say -- buys its
accuracy with virtual orbitals, and every virtual orbital costs two qubits
under Jordan-Wigner.  Water in PAW-LCAO-TZP has 4 occupied and 25 virtual
spatial orbitals: carrying all of them is 58 spin-orbitals, and almost all of
that width is spent on virtuals that the correlated wavefunction barely
touches.  Perturbation theory is the cheapest honest way to find out which
ones those are.

The quantities
--------------

In a Hartree-Fock reference with spatial orbital energies
:math:`\varepsilon_p`, the first-order doubles amplitudes are

.. math::

    t_{ij}^{ab} = \frac{\langle ij|ab\rangle}
                       {\varepsilon_i + \varepsilon_j
                        - \varepsilon_a - \varepsilon_b} ,

with :math:`i, j` occupied, :math:`a, b` virtual, and :math:`\langle ij|ab
\rangle` the physicists'-notation two-electron integral Mandacaru uses
throughout (:meth:`~mandacaru.core.mapping.Fermion.from_integrals`).  Writing
:math:`\tilde t_{ij}^{ab} = 2t_{ij}^{ab} - t_{ij}^{ba}`, the closed-shell
correlation energy is

.. math::

    E^{(2)} = \sum_{ijab} \tilde t_{ij}^{ab}\,\langle ij|ab\rangle
            = \sum_{ijab} t_{ij}^{ab}
              \bigl(2\langle ij|ab\rangle - \langle ij|ba\rangle\bigr) ,

and the second-order (unrelaxed) one-particle density matrix has no
occupied-virtual block, only

.. math::

    D^{(2)}_{ab} &= 2 \sum_{ijc} \tilde t_{ij}^{ac}\, t_{ij}^{bc} , \\
    D^{(2)}_{ij} &= -2 \sum_{abk} \tilde t_{ik}^{ab}\, t_{jk}^{ab} .

Both are derived here by spin-summing the spin-orbital expressions
:math:`D_{AB} = \tfrac12 \sum_{IJC} t_{IJ}^{AC} t_{IJ}^{BC}` and
:math:`D_{IJ} = -\tfrac12 \sum_{ABK} t_{IK}^{AB} t_{JK}^{AB}` over the four
spin cases of a closed shell; ``test/algorithms/test_mp2.py`` keeps an
independent spin-orbital implementation and pins the two against each other,
because the prefactors are exactly the kind of thing that is invisible in a
selection (which only uses the *ordering* of the eigenvalues) and wrong in a
reported occupation number.

The two blocks share one sum with relabelled dummies, so
:math:`\operatorname{Tr} D^{(2)}_{vv} + \operatorname{Tr} D^{(2)}_{oo} = 0`
**identically**: the second-order density moves charge from the occupied space
into the virtual space without changing the electron count.

Why the eigenvalues and not the energies
----------------------------------------

Ordering virtuals by :math:`\varepsilon_a` asks "which orbital is cheapest to
excite into"; ordering them by the eigenvalues of :math:`D^{(2)}_{vv}` -- the
**frozen natural orbitals** (FNO) of Sosa *et al.* and Taube & Bartlett -- asks
"which orbital does the correlated wavefunction actually occupy".  The second
question is the one an active space is asking, and the difference is not
cosmetic: the virtual natural orbitals are a *rotation* of the canonical
virtuals, so a handful of them can carry correlation that is spread thinly over
many canonical ones.

The rotation is confined to the virtual-virtual block, which is what makes it
safe here.  The occupied block is untouched, so the reference determinant, the
Hartree-Fock energy and the reference occupation the ansatz prepares are all
exactly what they were; only the labels on the orbitals nobody is occupying
change.

A caution about the reference
-----------------------------

For a closed-shell RHF reference the **reference** natural orbitals carry no
information at all: the RHF density is idempotent, so its eigenvalues are
exactly 2 and 0 and every ordering of the virtuals is as good as every other.
Selecting an active space from them is not a cheaper approximation to this
module, it is a no-op wearing a physical name, and
:func:`~mandacaru.algorithms.active_space.resolve_active_space` refuses it by
name for a closed shell rather than returning the energy ordering under a
label that claims otherwise.  Reference natural orbitals do carry information
for an **open-shell (UHF)** reference, which is why that combination is
allowed.

Semicanonicalization
--------------------

The formulas above assume a diagonal Fock matrix.  Mandacaru's MO basis is
canonical up to one thing: :func:`
~mandacaru.core.hamiltonian.conjugation_real_orbitals` recombines degenerate
partners to make the Hamiltonian real, which keeps :math:`F` diagonal only to
the extent that the partners are exactly degenerate.  :func:`fock_matrix`
therefore builds :math:`F` explicitly and :func:`semicanonical_rotation`
block-diagonalizes it inside the occupied and virtual blocks before the
amplitudes are formed.  That rotation changes no physics -- it is a unitary
mixing inside each block, so the determinant and the mean-field energy are
invariant -- and it makes the perturbation series well defined.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Largest off-diagonal Fock element (Hartree) left alone as already canonical.
SEMICANONICAL_TOLERANCE = 1e-8

#: Smallest ``|eps_i + eps_j - eps_a - eps_b|`` (Hartree) the amplitudes accept.
#: Below it the perturbation series has no small parameter and the amplitudes
#: are meaningless rather than merely inaccurate, so they are refused.
MIN_DENOMINATOR = 1e-6


@dataclass(frozen=True)
class MP2Result:
    r"""MP2 correlation energy and the natural orbitals of its density.

    Attributes
    ----------
    correlation_energy : float
        :math:`E^{(2)}` in Hartree.  Always negative.
    occupied_occupations : ndarray
        ``2 + diag(D_oo)`` per **canonical** occupied orbital, in canonical
        order -- how much charge second-order correlation takes out of each one.
        Reported, not acted on: the occupied block is deliberately left alone
        (see :func:`mp2_natural_orbitals`), so these are diagonal elements in
        the semicanonical basis rather than eigenvalues.
    virtual_occupations : ndarray
        Occupation numbers of the virtual natural orbitals, the eigenvalues of
        ``D_vv``, in **descending** order -- the most populated orbital first,
        which is the order an active space should keep them in.
    rotation : ndarray
        ``(M, M)`` orthogonal matrix taking the input MO basis to the frozen
        natural orbitals: the identity on the occupied block (up to
        semicanonicalization) and the virtual natural orbitals, ordered to match
        ``virtual_occupations``, on the virtual block.  Applying it to the MO
        coefficients gives those orbitals in the underlying basis.
    """

    correlation_energy: float
    occupied_occupations: np.ndarray
    virtual_occupations: np.ndarray
    rotation: np.ndarray

    @property
    def occupations(self) -> np.ndarray:
        """Every occupation number, occupied block then virtual block."""
        return np.concatenate([self.occupied_occupations,
                               self.virtual_occupations])


def fock_matrix(h_mo: np.ndarray, eri_mo: np.ndarray, n_occ: int) -> np.ndarray:
    r"""Closed-shell Fock matrix in the MO basis the integrals are given in.

    .. math::

        F_{pq} = h_{pq} + \sum_{i}^{\rm occ}
                 \bigl(2\langle pi|qi\rangle - \langle pi|iq\rangle\bigr) ,

    which is the same contraction
    :func:`~mandacaru.core.hamiltonian.freeze_core_integrals` performs for a
    frozen core, with the sum running over *all* occupied orbitals.  For
    canonical orbitals the result is diagonal and its diagonal holds the
    orbital energies.
    """
    h_mo = np.asarray(h_mo)
    eri_mo = np.asarray(eri_mo)
    n_occ = int(n_occ)
    occupied = slice(0, n_occ)
    # 2 <pi|qi> - <pi|iq>, summed over i.
    coulomb = np.einsum("piqi->pq", eri_mo[:, occupied, :, occupied])
    exchange = np.einsum("piiq->pq", eri_mo[:, occupied, occupied, :])
    fock = h_mo + 2.0 * coulomb - exchange
    return 0.5 * (fock + fock.conj().T)


def semicanonical_rotation(fock: np.ndarray, n_occ: int,
                           tolerance: float = SEMICANONICAL_TOLERANCE):
    r"""``(rotation, energies)`` block-diagonalizing ``fock`` in occ / virt.

    Diagonalizes :math:`F_{oo}` and :math:`F_{vv}` separately, so the returned
    ``rotation`` never mixes an occupied orbital with a virtual one: the
    reference determinant it defines is the same one, and the mean-field energy
    is unchanged.  ``energies`` holds the eigenvalues, ascending within each
    block.

    When ``fock`` is already diagonal to ``tolerance`` the identity is returned
    (as the identity matrix, not as ``None``) together with its diagonal, so
    callers need no special case.
    """
    fock = np.asarray(fock)
    n = fock.shape[0]
    n_occ = int(n_occ)
    blocks = [slice(0, n_occ), slice(n_occ, n)]
    off = 0.0
    for block in blocks:
        sub = np.asarray(fock[block, block])
        if sub.shape[0] > 1:
            off = max(off, float(np.max(np.abs(sub - np.diag(np.diag(sub))))))
    rotation = np.eye(n, dtype=float)
    energies = np.real(np.diag(fock)).astype(float).copy()
    if off <= float(tolerance):
        return rotation, energies
    for block in blocks:
        sub = np.asarray(fock[block, block])
        if sub.size == 0:
            continue
        eigenvalues, vectors = np.linalg.eigh(sub)
        rotation[block, block] = np.real(vectors)
        energies[block] = eigenvalues
    return rotation, energies


def rotate_integrals(h_mo: np.ndarray, eri_mo: np.ndarray,
                     rotation: np.ndarray):
    """``(h, eri)`` in the basis ``rotation``'s columns define.

    ``rotation`` maps old orbitals to new ones (new orbital ``j`` is
    ``sum_p rotation[p, j] * old_p``), the convention MO coefficient matrices
    use everywhere else here.
    """
    C = np.asarray(rotation)
    h = np.einsum("pi,qj,pq->ij", C.conj(), C, np.asarray(h_mo),
                  optimize=True)
    eri = np.einsum("pi,qj,rk,sl,pqrs->ijkl", C.conj(), C.conj(), C, C,
                    np.asarray(eri_mo), optimize=True)
    return h, eri


def _denominators(energies: np.ndarray, n_occ: int) -> np.ndarray:
    r"""``(o, o, v, v)`` array of ``eps_i + eps_j - eps_a - eps_b``."""
    eps = np.asarray(energies, dtype=float)
    occupied, virtual = eps[:n_occ], eps[n_occ:]
    d = (occupied[:, None, None, None] + occupied[None, :, None, None]
         - virtual[None, None, :, None] - virtual[None, None, None, :])
    smallest = float(np.min(np.abs(d))) if d.size else np.inf
    if smallest < MIN_DENOMINATOR:
        raise ValueError(
            f"the smallest MP2 denominator is {smallest:.3e} Ha: the "
            f"Hartree-Fock reference has an (almost) vanishing gap, so the "
            f"perturbation series has no small parameter and its amplitudes "
            f"are not an approximation to anything.  Such a system needs a "
            f"multireference active space chosen by hand "
            f"(active_orbitals=[...]) rather than by perturbation theory.")
    return d


def mp2_amplitudes(eri_mo: np.ndarray, energies: np.ndarray,
                   n_occ: int) -> np.ndarray:
    r"""First-order doubles amplitudes ``t[i, j, a, b]``, closed shell.

    Requires a **semicanonical** basis: ``energies`` are read as the diagonal
    of the Fock matrix and the off-diagonal part is assumed zero (see
    :func:`semicanonical_rotation`).
    """
    eri_mo = np.asarray(eri_mo)
    n_occ = int(n_occ)
    o, v = slice(0, n_occ), slice(n_occ, eri_mo.shape[0])
    # <ij|ab>, physicists' notation.
    integrals = np.real(eri_mo[o, o, v, v])
    return integrals / _denominators(energies, n_occ)


def _tilde(t: np.ndarray) -> np.ndarray:
    r"""``2 t_{ij}^{ab} - t_{ij}^{ba}``, the spin-summed combination."""
    return 2.0 * t - np.swapaxes(t, 2, 3)


def mp2_energy(eri_mo: np.ndarray, energies: np.ndarray, n_occ: int) -> float:
    r"""Closed-shell MP2 correlation energy in Hartree (negative).

    ``eri_mo`` must be in the same semicanonical basis as ``energies``.
    """
    n_occ = int(n_occ)
    eri_mo = np.asarray(eri_mo)
    t = mp2_amplitudes(eri_mo, energies, n_occ)
    o, v = slice(0, n_occ), slice(n_occ, eri_mo.shape[0])
    integrals = np.real(eri_mo[o, o, v, v])
    return float(np.einsum("ijab,ijab->", _tilde(t), integrals, optimize=True))


def mp2_density(eri_mo: np.ndarray, energies: np.ndarray, n_occ: int):
    r"""``(D_oo, D_vv)``: the second-order unrelaxed one-particle density.

    Spin-summed (so a doubly occupied orbital counts 2), symmetric, and
    traceless as a pair: ``Tr(D_oo) + Tr(D_vv) == 0`` identically.  ``D_oo`` is
    negative semi-definite (holes), ``D_vv`` positive semi-definite
    (promoted charge).
    """
    n_occ = int(n_occ)
    t = mp2_amplitudes(eri_mo, energies, n_occ)
    tt = _tilde(t)
    d_vv = 2.0 * np.einsum("ijac,ijbc->ab", tt, t, optimize=True)
    d_oo = -2.0 * np.einsum("ikab,jkab->ij", tt, t, optimize=True)
    return 0.5 * (d_oo + d_oo.T), 0.5 * (d_vv + d_vv.T)


def mp2_natural_orbitals(h_mo: np.ndarray, eri_mo: np.ndarray,
                         n_occ: int) -> MP2Result:
    r"""MP2 correlation energy and the **frozen** natural orbitals of its density.

    Semicanonicalizes the input basis, forms the second-order density and
    diagonalizes its **virtual-virtual block only**, ordering the result by
    descending occupation.  The returned ``rotation`` composes both steps, so it
    takes the input MO basis straight to those orbitals.

    The occupied block is left in canonical order on purpose, and this is a
    choice worth stating rather than an omission.  Two things depend on it.
    The reference determinant and the occupation the ansatz prepares are
    defined by *which* orbitals are occupied, and a rotation inside the
    occupied block leaves the determinant invariant but renames its orbitals --
    so anything that identifies a core orbital by its index, ``frozen_core=
    "auto"`` above all (the noble-gas core is "the lowest so many MOs", which
    is only the chemical core while the orbitals are energy-ordered), would
    silently freeze the wrong ones.  And truncating the occupied space is a far
    coarser approximation than truncating the virtual space, so it belongs to
    the explicit frozen-core controls and not to an automatic selector.  This
    is the classic frozen-natural-orbital scheme, and it is deliberately the
    conservative half of it.
    """
    h_mo = np.asarray(h_mo)
    eri_mo = np.asarray(eri_mo)
    n_occ = int(n_occ)
    M = h_mo.shape[0]
    if not 0 < n_occ < M:
        raise ValueError(
            f"MP2 needs at least one occupied and one virtual orbital, got "
            f"{n_occ} occupied of {M}")

    canonical, energies = semicanonical_rotation(
        fock_matrix(h_mo, eri_mo, n_occ), n_occ)
    if not np.allclose(canonical, np.eye(M)):
        h_mo, eri_mo = rotate_integrals(h_mo, eri_mo, canonical)

    correlation = mp2_energy(eri_mo, energies, n_occ)
    d_oo, d_vv = mp2_density(eri_mo, energies, n_occ)

    # Descending, so the virtuals an active space should keep come first.
    vir_values, vir_vectors = np.linalg.eigh(np.real(d_vv))
    vir_values, vir_vectors = vir_values[::-1], vir_vectors[:, ::-1]

    natural = np.eye(M, dtype=float)
    natural[n_occ:, n_occ:] = vir_vectors
    return MP2Result(correlation_energy=float(correlation),
                     occupied_occupations=2.0 + np.real(np.diag(d_oo)),
                     virtual_occupations=vir_values,
                     rotation=canonical @ natural)
