# -*- coding: utf-8 -*-
# file: algorithms/hubbard.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""The on-site Hubbard correction of a PAW-LCAO Kohn-Sham calculation.

The simplified (rotationally invariant) functional of Dudarev *et al.*,

.. math::

    E_U = \sum_{A,\sigma} \frac{U^A}{2}\,\mathrm{Tr}\big[n^{A\sigma}
          (1 - n^{A\sigma})\big],
    \qquad
    V^{A\sigma} = \frac{\partial E_U}{\partial n^{A\sigma}}
                = U^A\big(\tfrac12 - n^{A\sigma}\big),

acts on the occupation matrix :math:`n^{A\sigma}_{mm'}` of one correlated
shell (here the :math:`l = 3` shell of a lanthanide, or a :math:`d` shell)
of atom :math:`A`.  It is zero for a shell that is full or empty and
positive in between, so it pushes the occupations of the shell towards 0 and
1: the second-order, on-site part of the Coulomb repulsion that the linearized
one-center Hartree term of the datasets leaves out
(:mod:`mandacaru.pseudopotentials.onecenter`).  ``U`` here is
:math:`U_{\rm eff} = U - J`, in eV.

This module holds what does not depend on the Hamiltonian it is added to: the
validation of the ``hubbard_u`` option and the energy and potential of one
shell.  The occupation matrix itself is built where the projections live
(:class:`~mandacaru.pseudopotentials.paw.PAWIntegrals`).

Reference: S. L. Dudarev, G. A. Botton, S. Y. Savrasov, C. J. Humphreys and
A. P. Sutton, Phys. Rev. B 57, 1505 (1998).
"""

from __future__ import annotations

import math

import numpy as np
from ase.data import atomic_numbers

from ..units import EV_TO_HARTREE

#: The correlated shells the option accepts, by letter.
HUBBARD_SHELLS = {"d": 2, "f": 3}


def resolve_hubbard_u(spec) -> dict:
    """Validate ``hubbard_u=`` and return ``{symbol: {l: U in Hartree}}``.

    ``spec`` is ``None`` (no correction) or a mapping from an element symbol to
    a mapping from the shell letter (``"d"`` or ``"f"``) to ``U_eff`` in eV,
    for example ``{"Gd": {"f": 6.0}}``.  The shell is never guessed: a bare
    number is refused.
    """
    if spec is None:
        return {}
    if not isinstance(spec, dict):
        raise ValueError(
            f"hubbard_u must be None or a mapping like {{'Gd': {{'f': 6.0}}}} "
            f"(U_eff in eV per element and shell), got {spec!r}")
    resolved: dict = {}
    for symbol, shells in spec.items():
        if not isinstance(symbol, str) or symbol not in atomic_numbers:
            raise ValueError(f"hubbard_u: {symbol!r} is not an element symbol")
        if not isinstance(shells, dict) or not shells:
            raise ValueError(
                f"hubbard_u[{symbol!r}] must map a shell ('d' or 'f') to U_eff "
                f"in eV, got {shells!r}")
        resolved[symbol] = {}
        for shell, value in shells.items():
            if shell not in HUBBARD_SHELLS:
                raise ValueError(
                    f"hubbard_u[{symbol!r}]: shell {shell!r} is not one of "
                    f"{sorted(HUBBARD_SHELLS)}")
            if (isinstance(value, bool) or not isinstance(value, (int, float,
                                                                  np.floating))
                    or not math.isfinite(value) or value < 0.0):
                raise ValueError(
                    f"hubbard_u[{symbol!r}][{shell!r}] must be a finite U_eff "
                    f">= 0 in eV, got {value!r}")
            if value > 0.0:
                resolved[symbol][HUBBARD_SHELLS[shell]] = float(value) * EV_TO_HARTREE
        if not resolved[symbol]:
            del resolved[symbol]
    return resolved


def onsite_energy(occupation, u: float) -> float:
    """``(u / 2) Tr[n (1 - n)]`` for one spin channel of one shell (Hartree)."""
    n = np.asarray(occupation)
    return 0.5 * u * float(np.real(np.trace(n) - np.trace(n @ n)))


def onsite_potential(occupation, u: float) -> np.ndarray:
    """``u (1/2 - n)`` of one spin channel of one shell: ``dE/dn`` (Hartree)."""
    n = np.asarray(occupation)
    return u * (0.5 * np.eye(n.shape[0]) - n)


# --------------------------------------------------------------------------- #
# The occupation matrix of a shell, from the projections.
# --------------------------------------------------------------------------- #

def shell_weights(dataset, l: int, basis: str) -> np.ndarray:
    r"""``w_i = <chi|phi_i>`` over the augmentation sphere, one weight per
    projector ``i`` of channel ``l`` in the projectors' ``basis``.

    :math:`\chi` is the bound all-electron partial wave of the channel (the
    first occupied reference: the 4f of a lanthanide, the 3d of an iron-group
    atom) normalized to one inside the channel's cutoff radius.  A state
    :math:`\psi` has the one-center expansion
    :math:`\sum_i \phi_i\langle\tilde p_i|\psi\rangle` inside the sphere, so
    its amplitude on :math:`\chi` is :math:`\sum_i w_i \langle\tilde
    p_i|\psi\rangle` and the occupation matrix of the shell is
    :math:`n = W^T R W` with :math:`R = \langle\tilde p|\psi\rangle
    \langle\psi|\tilde p\rangle` (:meth:`OnsiteHubbard.evaluate`).

    The projector basis is not orthonormal -- even for a unitary dataset the
    dual partial waves of one channel overlap by 0.7 -- so the bare diagonal of
    ``R`` is not an occupation.  The weights are taken in the basis the
    projections were built in (``"raw"``: the partial waves
    :math:`\sum_i (B^{-1})_{ki}\phi_i`, see
    :func:`~mandacaru.pseudopotentials.onecenter._basis_transform`).
    An isolated shell that lies entirely inside the sphere has an occupation
    equal to its electron count; the share of the true orbital outside the
    sphere is not counted, so a very small sphere under-counts.
    """
    from scipy.integrate import simpson

    from ..pseudopotentials.multipoles import partial_waves
    from ..pseudopotentials.onecenter import _basis_transform

    channel = dataset.channels[int(l)]
    r = np.asarray(dataset.r, dtype=float)
    inside = r <= float(channel.r_cut)
    ae, _smooth = partial_waves(dataset, l)
    occupied = [i for i, f in enumerate(getattr(channel, "occupations", ()) or ())
                if float(f) > 0.0]
    chi = ae[occupied[0] if occupied else 0]
    transform = _basis_transform(dataset, l, basis)
    waves = list(ae) if transform is None else [
        sum(transform[i, k] * ae[k] for k in range(len(ae)))
        for i in range(len(ae))]
    weight = r[inside] ** 2
    norm = float(simpson(chi[inside] ** 2 * weight, x=r[inside]))
    if not norm > 0.0:
        raise ValueError(f"the l = {l} channel of {dataset.symbol} has no "
                         "bound partial wave to project on")
    return np.array([float(simpson(chi[inside] * wave[inside] * weight,
                                   x=r[inside])) for wave in waves]) / math.sqrt(norm)


class OnsiteHubbard:
    r"""The on-site correction of a set of shells, as the Hamiltonians use it.

    ``shells`` is a list of ``(u, positions, w)``: the :math:`U_{\rm eff}` in
    Hartree, the global positions of the shell's projectors ordered by
    :math:`m = -l..l`, then radial index, and the weights ``w`` of the radial
    indices (:func:`shell_weights`).  ``size`` is the total projector count.
    """

    def __init__(self, shells, size: int):
        self.size = int(size)
        self.shells = []
        for u, positions, weights in shells:
            positions = np.asarray(positions, dtype=int)
            weights = np.asarray(weights, dtype=float)
            n_radial = weights.size
            n_m = positions.size // n_radial
            selection = np.zeros((positions.size, n_m))
            for m in range(n_m):
                for i in range(n_radial):
                    selection[m * n_radial + i, m] = weights[i]
            self.shells.append((float(u), positions, selection))

    def occupations(self, projected) -> list:
        """``[n]`` of every shell for one spin channel's projected density
        matrix ``R`` (``size x size``)."""
        R = np.asarray(projected)
        return [selection.T @ R[np.ix_(positions, positions)] @ selection
                for _u, positions, selection in self.shells]

    def evaluate(self, projected, restricted: bool):
        """``(energy, [operator per spin channel])`` in Hartree.

        ``projected`` is ``(R,)`` with both spins summed when ``restricted``
        (then each spin holds half of ``R`` and the energy counts both), else
        ``(R_up, R_down)``.  The operators are ``size x size`` and enter the
        Fock matrix as ``C O C^dagger``.
        """
        spins = 1 if restricted else 2
        if len(projected) != spins:
            raise ValueError(f"{'restricted' if restricted else 'unrestricted'} "
                             f"needs {spins} projected density matrix(es)")
        energy = 0.0
        operators = []
        for R in projected:
            R = np.asarray(R)
            if restricted:
                R = 0.5 * R
            operator = np.zeros((self.size, self.size), dtype=R.dtype)
            for (u, positions, selection), n in zip(self.shells,
                                                    self.occupations(R)):
                energy += onsite_energy(n, u)
                operator[np.ix_(positions, positions)] += (
                    selection @ onsite_potential(n, u) @ selection.T)
            operators.append(operator)
        if restricted:
            energy *= 2.0
        return energy, operators


def build_onsite(projectors, datasets, resolved) -> "OnsiteHubbard | None":
    """The :class:`OnsiteHubbard` of the atoms in ``projectors`` that
    ``resolved`` (:func:`resolve_hubbard_u`) names, or ``None``.

    ``datasets`` maps a symbol to its PAW dataset.  Raises ``ValueError`` for
    an element that is absent from the system or whose dataset has no channel
    in the requested shell.
    """
    present = {p.symbol for p in projectors}
    missing = sorted(set(resolved) - present)
    if missing:
        raise ValueError(f"hubbard_u names {', '.join(missing)}, which "
                         "is not in the system")
    groups: dict = {}
    for position, p in enumerate(projectors):
        groups.setdefault((p.atom_index, p.symbol, p.l), []).append(
            (int(p.m), int(p.index), position))
    shells = []
    for (atom, symbol, l), entries in sorted(groups.items()):
        u = resolved.get(symbol, {}).get(l)
        if not u:
            continue
        basis = getattr(projectors[entries[0][2]], "projector_basis", "raw")
        weights = shell_weights(datasets[symbol], l, basis)
        entries.sort()
        if len(entries) != (2 * l + 1) * weights.size:
            raise ValueError(f"{symbol} l={l}: {len(entries)} projectors, "
                             f"expected {(2 * l + 1) * weights.size}")
        shells.append((u, [position for _m, _i, position in entries], weights))
    for symbol, wanted in resolved.items():
        for l in wanted:
            if not any(g[1] == symbol and g[2] == l for g in groups):
                raise ValueError(
                    f"hubbard_u: the {symbol} dataset has no l = {l} channel")
    return OnsiteHubbard(shells, len(projectors)) if shells else None
