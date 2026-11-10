# -*- coding: utf-8 -*-
# file: pseudopotentials/diagnostics.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Checks of a PAW-LCAO / UPAW-LCAO dataset: does its smooth Hamiltonian have the atom's
spectrum and scatter like the atom?

Every check is native (``mandacaru-core`` ``paw::diagnose``).  A dataset is handed over in
the layout of its file (:func:`~.record.record_of`), the all-electron atom it is compared
with is solved again from the record, and a check that cannot be evaluated (a level or phase
that is not finite) raises :class:`~.defects.DiagnosticError`, never a pass.
"""

from __future__ import annotations

import numpy as np

from .. import _native
from .defects import DiagnosticError
from .native_context import context
from .record import record_of


def diagnose(pp, *, levels: bool = False) -> dict:
    """Every check of ``pp`` against the all-electron atom it was built on.

    The result has ``ghosts`` (``{l: eps_0 - eps_ref}`` of the channels that hold one),
    ``phases`` (``{l: (near, far)}`` scattering phase errors, rad), ``unprojected``
    (``{l: rad}`` of the angular momenta without projectors), ``s_miss`` (the intruding-1s
    miss of the s channel, ``None`` without one), ``missing`` (``{l: level - reference}`` of
    the channels that do not bind a physical reference), ``branch_ghosts`` (``{"l:kappa":
    depth}`` of the ``j`` branches of a fully relativistic dataset), ``branch_phases``,
    ``wrong_phases`` (the phases outside their tolerances), ``unprojected_missing``
    (``{l: offset}``, Hartree, of the bound all-electron levels a channel without projectors
    does not reproduce) and ``defects`` (what a ``ghosts="flag"`` generation records).  With
    ``levels``, ``levels`` holds the lowest levels the ghost test reads for each channel.
    """
    answer = _native.paw_diagnose(context(), *record_of(pp), levels=bool(levels))
    if "unevaluated" in answer:
        raise DiagnosticError(answer["unevaluated"])
    answer["phases"] = {l: tuple(p) for l, p in answer["phases"].items()}
    return answer


def ghost_levels(pp) -> dict:
    """``{l: [eps_0, eps_1, ...]}``: the levels the ghost test reads for each channel (two,
    three when a channel holds two occupied references), without the all-electron atom."""
    return {int(l): [float(e) for e in levels] for l, levels in
            _native.paw_ghost_levels(context(), *record_of(pp)).items()}


def intruding_s_miss(pp) -> float | None:
    """How much of a hydrogen 1s entering the sphere the s projectors miss, as a fraction of
    its norm (``None`` without an s channel).  Above 1 the s channel cannot represent a
    neighbor's orbital and bonds to the element come out too long or collapse."""
    return _native.paw_intruding_s_miss(context(), *record_of(pp))


def paw_spectrum(pp, l: int, n_states: int = 3, r_max: float | None = None,
                 stride: int = 2) -> np.ndarray:
    r"""Lowest eigenvalues of the generalized problem :math:`(T_l + \tilde v^{scr} +
    \sum|\tilde p_i\rangle D^{scr}_{ij}\langle\tilde p_j|)c = \varepsilon(1 +
    \sum|\tilde p_i\rangle q_{ij}\langle\tilde p_j|)c` on the radial grid (3-point Laplacian at
    two spacings, Richardson-extrapolated).  A ghost state shows up as an eigenvalue below
    the bound reference energy.  ``r_max`` is the radius of the box (default: where the
    outermost occupied smooth wave has decayed).  The pair of spacings is fixed
    (``stride=2``); any other value raises.
    """
    if int(stride) != 2:
        raise NotImplementedError(
            "the spectrum uses one pair of grid spacings (stride 2); another stride "
            "is not available")
    return np.asarray(_native.paw_levels(context(), *record_of(pp), int(l), int(n_states),
                                         r_max=r_max))


def log_derivative_paw(pp, l: int, energy: float, r_cut: float | None = None) -> float:
    r"""Smooth logarithmic derivative :math:`\tilde R'/\tilde R` at ``r_cut`` of the
    generalized problem at ``energy`` (the projector radius by default).  A channel without
    projectors sees the screened local potential alone, and then ``r_cut`` is required."""
    l = int(l)
    if l not in pp.projectors and r_cut is None:
        raise ValueError(f"channel l={l} has no projectors: pass r_cut")
    values = _native.paw_log_derivatives(context(), *record_of(pp), l, [float(energy)],
                                         r_cut=None if r_cut is None else float(r_cut))
    return float(values[0])


def reconstruct_ae(pp, l: int, r, u) -> np.ndarray:
    r"""Bloechl's transformation of a smooth radial function ``u = r R`` back to the
    all-electron one: :math:`u = \tilde u + \sum_i (u_i - \tilde u_i)\langle\tilde
    p_i|\tilde u\rangle`."""
    return np.asarray(_native.paw_reconstruct(
        context(), *record_of(pp), int(l),
        np.ascontiguousarray(r, dtype=np.float64),
        np.ascontiguousarray(u, dtype=np.float64)))


def log_derivative_ae(r, potential, l: int, energy: float, r_cut: float, z_eff: float,
                      treatment: str = "none", kappa: int | None = None) -> float:
    r"""All-electron logarithmic derivative :math:`R'/R` at ``r_cut``."""
    return float(_native.paw_log_derivatives_ae(
        context(), np.ascontiguousarray(r, dtype=np.float64),
        np.ascontiguousarray(potential, dtype=np.float64), int(l), [float(energy)],
        float(r_cut), float(z_eff), str(treatment), kappa)[0])


def report_paw(pp) -> str:
    """Human-readable validation summary of a dataset: what it stores and what
    :func:`diagnose` finds, channel by channel."""
    from .defects import defect_message
    from .paw import _deficit_label

    found = diagnose(pp, levels=True)
    lines = [f"{pp!r}",
             f"  valence charge  : {pp.valence_charge:g}",
             f"  norm deficit    : {_deficit_label(pp.norm_deficit)}",
             f"  local potential : polynomial inside rcl = {pp.r_cut_local:.3f} "
             f"Bohr, V_loc(0) = {pp.v_local_screened[0]:+.4f} Ha (screened)",
             f"  compensation    : Q^ = {pp.compensation_charge:+.5f} e in "
             f"rg = {pp.compensation_radius:.3f} Bohr, int v_H g = "
             f"{pp.hartree_screening:+.5f} Ha",
             f"  one-center      : E_1c = {pp.one_center_energy:+.6f} Ha "
             f"(reference valence {pp.energies.get('reference_valence', 0.0):+.6f}, "
             f"pseudo atom {pp.energies.get('pseudo_atom', 0.0):+.6f})"]
    if pp.defects:
        lines.append("  WARNING         : "
                     + defect_message(pp.symbol, pp.family, pp.defects))
    for l in sorted(pp.channels):
        channel = pp.channels[l]
        near, far = found["phases"][l]
        lines.append(f"  l={l}: rc={channel.r_cut:.3f}  eps="
                     + ", ".join(f"{e:+.4f}" for e in channel.reference_energies)
                     + f"  q={[[round(float(v), 5) for v in row] for row in channel.overlap_correction]}"
                     f"  duality={channel.duality_error:.1e}"
                     f"  eps err={found['levels'][l][0] - channel.reference_energies[0]:+.1e}"
                     f"  phase={near:.3f}/{far:.3f} rad")
        lines.append("        D^ion = " + str([[round(float(v), 5) for v in row]
                                             for row in channel.coupling]))
    if found["s_miss"] is not None:
        lines.append(f"  intruding 1s    : the s projectors miss {found['s_miss']:.3f} of its norm")
    return "\n".join(lines)
