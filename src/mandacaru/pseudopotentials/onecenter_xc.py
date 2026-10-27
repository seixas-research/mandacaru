# -*- coding: utf-8 -*-
# file: pseudopotentials/onecenter_xc.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""The nonlinear one-center exchange-correlation energy of a PAW-LCAO sphere.

The exchange-correlation energy of the all-electron density, split as the
Hartree energy is (:mod:`mandacaru.pseudopotentials.onecenter`), is the smooth
part on the grid plus, per augmentation sphere,

.. math::

    F[D^\uparrow, D^\downarrow] = \int_{\rm sphere}
        \Big( f[n^1_\uparrow + \tfrac12 n_c,\; n^1_\downarrow + \tfrac12 n_c]
            - f[\tilde n^1_\uparrow + \tfrac12\tilde n_c,\;
                \tilde n^1_\downarrow + \tfrac12\tilde n_c] \Big),

with :math:`n^1_\sigma(\mathbf r) = \sum_{ij} R^\sigma_{ij}\,\phi_i(\mathbf r)
\phi_j^*(\mathbf r)` the one-center density of a spin channel, :math:`n_c` the
spherical, unpolarized all-electron core and :math:`\tilde n_c` the smooth
(partial) core.  A dataset replaces :math:`F` by its linearization about the
isolated atom: a per-species constant and a frozen one-body :math:`V` taken
from the *spherical, spin-unpolarized* reference density.  The linearization
leaves out two things at once:

* the nonlinearity in how the density moves (charge transfer, rehybridization,
  the occupation of a compact shell), and
* the spin polarization of the all-electron density inside the sphere.  The
  reference is unpolarized, so the spin-polarization energy of a localized 4f
  shell (its Hund's-rule exchange) is not in a spin-polarized Kohn-Sham
  energy; the grid's spin functional sees only the smooth 4f.

Both are in the remainder

.. math::

    E_{xc}^{(2)} = F[R] - F[R^0] - \sum_\sigma
        \mathrm{Tr}\big[O^\sigma(R^0)\,(R^\sigma - R^0)\big],
    \qquad
    O^\sigma_{ab}(R) = \partial F/\partial R^\sigma_{ba},

with the reference :math:`R^0 = D^0/2` in each channel.  It is defined against
this module's own derivative at the reference, so the energy and the operator
vanish there by construction; whether that derivative equals the dataset's
:math:`V` is a diagnostic (:func:`reference_potential`), not an assumption.

Quadrature: Gauss-Legendre nodes in the radius between 0 and the sphere
radius, and the product Gauss-Legendre by uniform-azimuth rule of
:func:`~mandacaru.pseudopotentials.multipoles._angular_grid`.  The functional
is not a polynomial in the angles, so the rule is larger than the Gaunt one;
``tests`` double it.  Local density approximation only.
"""

from __future__ import annotations

import numpy as np

from ..basis._angular import spherical_harmonic
from ..basis.xc_spin import spin_partials
from .multipoles import _angular_grid, partial_waves
from .onecenter import _basis_transform, _smooth_xc_core

#: Radial Gauss-Legendre nodes, and the polar / azimuthal angular points.
RADIAL_NODES = 96
POLAR_POINTS = 20
AZIMUTHAL_POINTS = 40


class XCSphere:
    """The nonlinear one-center xc of one species' sphere, over its projector
    layout.

    ``projectors`` are the sphere's own :class:`KBProjector` objects (any one
    atom of the species: only ``l``, ``m``, ``index`` and ``real`` are used),
    ``basis`` the projector basis the projections were built in.
    """

    def __init__(self, dataset, projectors, basis: str,
                 radial: int = RADIAL_NODES, polar: int = POLAR_POINTS,
                 azimuthal: int = AZIMUTHAL_POINTS):
        if "lda" != _functional_key(dataset.xc):
            raise NotImplementedError(
                f"the nonlinear one-center exchange-correlation is for the "
                f"local density approximation, not {dataset.xc!r}")
        self.relativistic = bool(getattr(dataset, "relativistic_exchange",
                                         False))
        channels = sorted(dataset.channels)
        sphere = max(float(dataset.projector_radius(l)) for l in channels)
        nodes, weights = np.polynomial.legendre.leggauss(int(radial))
        r = 0.5 * sphere * (nodes + 1.0)
        w_r = 0.5 * sphere * weights * r * r
        theta, phi, w_omega = _angular_grid(int(polar), int(azimuthal))
        source = np.asarray(dataset.r, dtype=float)
        waves = {}
        for l in channels:
            ae, ps = partial_waves(dataset, l)
            ae_r = [np.interp(r, source, w) for w in ae]
            ps_r = [np.interp(r, source, w) for w in ps]
            transform = _basis_transform(dataset, l, basis)
            if transform is not None:
                ae_r = [sum(transform[i, k] * ae_r[k]
                            for k in range(len(ae_r)))
                        for i in range(len(ae_r))]
                ps_r = [sum(transform[i, k] * ps_r[k]
                            for k in range(len(ps_r)))
                        for i in range(len(ps_r))]
            waves[l] = (ae_r, ps_r)
        n = len(projectors)
        points = r.size * theta.size
        ae = np.empty((n, points), dtype=complex)
        ps = np.empty((n, points), dtype=complex)
        harmonic_cache: dict = {}
        for a, p in enumerate(projectors):
            key = (p.l, p.m, bool(p.real))
            if key not in harmonic_cache:
                harmonic_cache[key] = spherical_harmonic(
                    p.l, p.m, theta, phi, real=bool(p.real))
            Y = harmonic_cache[key]
            ae[a] = np.outer(waves[p.l][0][p.index], Y).ravel()
            ps[a] = np.outer(waves[p.l][1][p.index], Y).ravel()
        if all(p.real for p in projectors):
            ae, ps = ae.real, ps.real
        self.n = n
        self.ae, self.ps = ae, ps
        self.weight = np.outer(w_r, w_omega).ravel()
        core = np.interp(r, source, np.asarray(dataset.core_density,
                                               dtype=float))
        smooth = np.interp(r, source, _smooth_xc_core(dataset))
        self.core = np.repeat(core, theta.size)
        self.smooth_core = np.repeat(smooth, theta.size)
        # The reference atom's projected density, per m, in the basis.
        self.reference = self._reference(dataset, projectors, basis)
        self._reference_operators = None

    @staticmethod
    def _reference(dataset, projectors, basis):
        from .onecenter import exact_one_center_terms
        terms = exact_one_center_terms(dataset, basis)
        D0 = np.zeros((len(projectors), len(projectors)))
        for i, a in enumerate(projectors):
            for j, b in enumerate(projectors):
                if (a.l, a.m) == (b.l, b.m):
                    D0[i, j] = terms[a.l][2][a.index, b.index]
        return D0

    # -- the functional ---------------------------------------------------- #

    def _density(self, waves, R):
        """``n(r) = sum R_ij phi_i phi_j^*`` at every quadrature point."""
        return np.einsum("in,in->n", waves, (R @ waves.conj())).real

    def functional(self, R_up, R_dn):
        """``(F, O_up, O_dn)``: the energy of the sphere (Hartree) and the
        operators ``O_ab = dF/dR_ba`` of the two spin channels."""
        total = 0.0
        operators = []
        up_ae = self._density(self.ae, R_up) + 0.5 * self.core
        dn_ae = self._density(self.ae, R_dn) + 0.5 * self.core
        up_ps = self._density(self.ps, R_up) + 0.5 * self.smooth_core
        dn_ps = self._density(self.ps, R_dn) + 0.5 * self.smooth_core
        f_ae, vu_ae, vd_ae = self._partials(up_ae, dn_ae)
        f_ps, vu_ps, vd_ps = self._partials(up_ps, dn_ps)
        total = float(np.sum(self.weight * (f_ae - f_ps)))
        for v_ae, v_ps in ((vu_ae, vu_ps), (vd_ae, vd_ps)):
            operators.append(self._operator(self.ae, self.weight * v_ae)
                             - self._operator(self.ps, self.weight * v_ps))
        return total, operators[0], operators[1]

    def _partials(self, up, dn):
        f, v_up, v_dn, *_ = spin_partials(
            "lda", np.maximum(up, 0.0), np.maximum(dn, 0.0),
            relativistic=self.relativistic)
        # Where a channel's density is clipped the functional is constant in
        # it: its derivative is zero, as the grid code's `_clipped` has it.
        return f, np.where(up > 0.0, v_up, 0.0), np.where(dn > 0.0, v_dn, 0.0)

    @staticmethod
    def _operator(waves, weighted_potential):
        """``O_ab = sum_n w v conj(phi_a) phi_b`` -- ``dF/dR_ba`` for
        ``n = sum R_ij phi_i phi_j^*``."""
        return (waves.conj() * weighted_potential) @ waves.T

    # -- the remainder ------------------------------------------------------ #

    def reference_operators(self):
        """``(F0, O0_up, O0_dn)`` at the unpolarized reference atom."""
        if self._reference_operators is None:
            half = 0.5 * self.reference
            self._reference_operators = self.functional(half, half)
        return self._reference_operators

    def remainder(self, R_up, R_dn):
        """``(E, O_up, O_dn)``: the nonlinear remainder
        :math:`E_{xc}^{(2)}` and its operators, zero at the reference."""
        F0, O0_up, O0_dn = self.reference_operators()
        F, O_up, O_dn = self.functional(R_up, R_dn)
        half = 0.5 * self.reference
        energy = F - F0 - float(np.real(
            np.trace(O0_up @ (R_up - half)) + np.trace(O0_dn @ (R_dn - half))))
        return energy, O_up - O0_up, O_dn - O0_dn

    def reference_potential(self):
        """The dataset-independent diagnostic: ``V_ab`` of the spherical
        reference, ``O0`` of either channel, in the projector basis."""
        return self.reference_operators()[1]


def _functional_key(name) -> str:
    key = str(name).strip().lower().replace("-", "").replace("_", "")
    return "lda" if key in ("lda", "pz", "pz81", "ldapz") else key
