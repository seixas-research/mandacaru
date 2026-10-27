# -*- coding: utf-8 -*-
# file: algorithms/onsite.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""The on-site corrections of a PAW-LCAO Kohn-Sham calculation, as the
Hamiltonians use them.

An on-site correction is a functional of the one-center density matrices
:math:`R^\sigma_{ij} = \langle\tilde p_i|\psi\rangle\langle\psi|\tilde p_j
\rangle` of the augmentation spheres, summed over the occupied states.  It
adds an energy and, in projector space, an operator
:math:`O = \partial E/\partial R^T` to the Fock matrix as
:math:`C\,O\,C^\dagger` (:math:`C` the projections), and its geometry
dependence is the projections' alone.  So every correction is one object with
one interface, ``evaluate(projected, restricted) -> (energy, [operators])``,
and the solvers, the crystal loops, the forces and the stress treat a
:class:`CombinedOnsite` of them like a single one:

* :class:`~mandacaru.algorithms.hubbard.OnsiteHubbard`, the Dudarev
  correction of a d or f shell;
* :class:`QuadraticHartree`, below;
* :class:`NonlinearXC`, the nonlinear, spin-polarized one-center
  exchange-correlation of the spheres
  (:mod:`mandacaru.pseudopotentials.onecenter_xc`).

The one-center Hartree term, quadratic
--------------------------------------
The datasets linearize the Hartree energy of each sphere about the isolated
reference atom, whose projected density matrix is :math:`D^0`.  With
:math:`n^1 = \sum D_{ij}\phi_i\phi_j^*` the all-electron one-center density,
the exact energy is :math:`\tfrac12\sum D D\,\Delta W` and the linearization
:math:`\tfrac12 D^0D^0\Delta W + (D - D^0)\,\Delta W D^0`, so what is omitted
is exactly

.. math::

    E_2 = \tfrac12\sum_A (D_A - D^0)\,\Delta W_A\,(D_A - D^0),

:math:`\Delta W` being :func:`~mandacaru.pseudopotentials.onecenter.
one_center_coulomb` of the sphere.  It is positive, quadratic in how far the
sphere's density has moved from the atom's, and zero for the atom itself.  For
a compact shell whose occupation or shape changes with the environment it is
not small: 15 to 20 eV per atom for the 4f of gadolinium in the metal, and
4 eV more at 0.88 of the lattice than at 1.12, against 0.07 eV (and no
dependence on the volume) for lanthanum, whose 4f is empty.  Adding it is the
"exact" one-center Hartree term of the wave-function Hamiltonians
(:class:`~mandacaru.pseudopotentials.onecenter.OneCenterExact`) in a
Kohn-Sham calculation; the dataset's frozen exchange-correlation stays
frozen.

The density enters through its total, so the same operator
:math:`O = \Delta W\,(D - D^0)` acts on both spins.
"""

from __future__ import annotations

import numpy as np

from ..pseudopotentials.onecenter import (_projector_basis,
                                          exact_one_center_terms,
                                          one_center_exchange_tensor)

#: The values the ``one_center_hartree`` option takes.
ONE_CENTER_HARTREE = ("linearized", "quadratic")

#: The values the ``one_center_xc`` option takes.
ONE_CENTER_XC = ("linearized", "nonlinear")


def resolve_one_center_xc(value) -> str:
    """Validate ``one_center_xc=``: ``"linearized"`` (the datasets' own, the
    default) or ``"nonlinear"``."""
    if value is None:
        return "linearized"
    if not isinstance(value, str) or value not in ONE_CENTER_XC:
        raise ValueError(
            f"one_center_xc must be one of {ONE_CENTER_XC}, got {value!r}")
    return value


def resolve_one_center_hartree(value) -> str:
    """Validate ``one_center_hartree=``: ``"linearized"`` (the datasets' own,
    the default) or ``"quadratic"``."""
    if value is None:
        return "linearized"
    if not isinstance(value, str) or value not in ONE_CENTER_HARTREE:
        raise ValueError(
            f"one_center_hartree must be one of {ONE_CENTER_HARTREE}, "
            f"got {value!r}")
    return value


class QuadraticHartree:
    r""":math:`E_2 = \tfrac12 \delta D\,\Delta W\,\delta D` of every sphere.

    ``spheres`` is a list of ``(positions, W, D0)``: the global positions of
    the sphere's projectors, its :math:`\Delta W` ``(n, n, n, n)`` and the
    reference atom's :math:`D^0` ``(n, n)`` in the projectors' own basis.
    ``size`` is the total projector count.
    """

    def __init__(self, spheres, size: int):
        self.size = int(size)
        self.spheres = [(np.asarray(positions, dtype=int), np.asarray(W),
                         np.asarray(D0)) for positions, W, D0 in spheres]

    def evaluate(self, projected, restricted: bool):
        """``(energy, [operator per spin channel])`` in Hartree.

        ``projected`` is ``(R,)`` with both spins summed when ``restricted``,
        else ``(R_up, R_down)``; the density the energy sees is their sum, and
        both channels get the same ``(P, P)`` operator."""
        spins = 1 if restricted else 2
        if len(projected) != spins:
            raise ValueError(f"{'restricted' if restricted else 'unrestricted'}"
                             f" needs {spins} projected density matrix(es)")
        total = np.asarray(projected[0] if restricted
                           else projected[0] + projected[1])
        operator = np.zeros((self.size, self.size), dtype=total.dtype)
        energy = 0.0
        for positions, W, D0 in self.spheres:
            # n(r) = sum R_ij phi_i phi_j^*, so D_ji = R_ij.
            delta = total[np.ix_(positions, positions)].T - D0
            G = np.einsum("jilk,lk->ji", W, delta, optimize=True)
            energy += 0.5 * float(np.real(np.sum(G * delta)))
            operator[np.ix_(positions, positions)] += G
        return energy, [operator.copy() for _ in range(spins)]


def build_quadratic(projectors, datasets) -> "QuadraticHartree | None":
    """The :class:`QuadraticHartree` of the atoms in ``projectors``, or
    ``None`` without a PAW-LCAO sphere.  ``datasets`` maps a symbol to its
    PAW dataset."""
    groups: dict = {}
    for position, projector in enumerate(projectors):
        groups.setdefault(projector.atom_index, []).append(position)
    spheres = []
    for _atom, members in sorted(groups.items()):
        own = [projectors[p] for p in members]
        dataset = datasets[own[0].symbol]
        basis = _projector_basis(own, None)
        W = one_center_exchange_tensor(dataset, own, 0.0, basis)
        terms = exact_one_center_terms(dataset, basis)
        D0 = np.zeros((len(own), len(own)))
        for i, a in enumerate(own):
            for j, b in enumerate(own):
                if (a.l, a.m) == (b.l, b.m):
                    D0[i, j] = terms[a.l][2][a.index, b.index]
        spheres.append((members, W, D0))
    return QuadraticHartree(spheres, len(projectors)) if spheres else None


class NonlinearXC:
    r"""The remainder :math:`E_{xc}^{(2)}` of the spheres' linearized
    exchange-correlation (:class:`~mandacaru.pseudopotentials.onecenter_xc.
    XCSphere`), summed over the atoms.

    ``spheres`` is a list of ``(positions, XCSphere)``; ``size`` the total
    projector count.  Spin-resolved: a restricted calculation is the two
    channels at half the total each, and its one operator is the channel's.
    """

    def __init__(self, spheres, size: int):
        self.size = int(size)
        self.spheres = [(np.asarray(positions, dtype=int), sphere)
                        for positions, sphere in spheres]

    def evaluate(self, projected, restricted: bool):
        spins = 1 if restricted else 2
        if len(projected) != spins:
            raise ValueError(f"{'restricted' if restricted else 'unrestricted'}"
                             f" needs {spins} projected density matrix(es)")
        channels = ((0.5 * np.asarray(projected[0]),) * 2 if restricted
                    else tuple(np.asarray(R) for R in projected))
        dtype = np.result_type(channels[0], float)
        operators = [np.zeros((self.size, self.size), dtype=dtype)
                     for _ in range(2)]
        energy = 0.0
        for positions, sphere in self.spheres:
            block = np.ix_(positions, positions)
            e, o_up, o_dn = sphere.remainder(channels[0][block],
                                             channels[1][block])
            energy += e
            operators[0][block] += o_up
            operators[1][block] += o_dn
        return energy, operators[:spins]


def build_nonlinear_xc(projectors, datasets) -> "NonlinearXC | None":
    """The :class:`NonlinearXC` of the atoms in ``projectors``; ``datasets``
    maps a symbol to its PAW dataset.  One quadrature table per species and
    projector layout."""
    from ..pseudopotentials.onecenter_xc import XCSphere

    groups: dict = {}
    for position, projector in enumerate(projectors):
        groups.setdefault(projector.atom_index, []).append(position)
    tables: dict = {}
    spheres = []
    for _atom, members in sorted(groups.items()):
        own = [projectors[p] for p in members]
        layout = (own[0].symbol,
                  tuple((p.l, p.m, p.index, bool(p.real)) for p in own))
        basis = _projector_basis(own, None)
        if layout not in tables:
            tables[layout] = XCSphere(datasets[own[0].symbol], own, basis)
        spheres.append((members, tables[layout]))
    return NonlinearXC(spheres, len(projectors)) if spheres else None


class CombinedOnsite:
    """Several on-site corrections evaluated together.

    ``components`` maps a name (``"hubbard"``, ``"one_center_hartree"``) to an
    object with ``evaluate(projected, restricted)``.  :attr:`energies` holds
    each component's energy of the last evaluation, so a result can report
    them separately; the energy and the operators it returns are the sums.
    """

    def __init__(self, **components):
        if not components:
            raise ValueError("a combination needs at least one correction")
        self.components = components
        self.size = next(iter(components.values())).size
        self.energies = {name: 0.0 for name in components}

    def evaluate(self, projected, restricted: bool):
        total, operators = 0.0, None
        for name, component in self.components.items():
            energy, parts = component.evaluate(projected, restricted)
            self.energies[name] = energy
            total += energy
            operators = (parts if operators is None
                         else [a + b for a, b in zip(operators, parts)])
        return total, operators
