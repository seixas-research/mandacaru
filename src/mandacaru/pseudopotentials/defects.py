# -*- coding: utf-8 -*-
# file: pseudopotentials/defects.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""What a generated dataset can still get wrong, and how Mandacaru says so.

The search that repairs a ghost state or a wrong scattering, the checks it applies and the
record of what it could not remove are native (``mandacaru-core``: ``paw::generate::search``,
``paw::diagnose``, ``paw::record``).  This module owns the words: the three exceptions of the
``ghosts=`` option, the modes it accepts, the reader of the ``defects`` block of a record
(:func:`read_defects`) and the warning text for a dataset that carries any
(:func:`defect_message`, :func:`warn_defects`).

A record's ``defects`` block keys are ``ghosts`` (``{l: eps_0 - eps_ref}``, Hartree),
``phases`` (``{l: (near, far)}``, rad), ``residuals``, ``s_miss`` (the intruding-1s miss),
``unprojected`` (``{l: rad}``), and the two keys of the native generator, ``missing``
(``{l: nearest level - reference}``, Hartree: a physical state the dataset lacks) and
``branch_ghosts`` (``{"l:kappa": depth}``: a ghost of a ``j`` branch of a fully relativistic
dataset).  A file written before the last two existed lacks them.
"""

from __future__ import annotations

import warnings

#: The values of the ``ghosts=`` option, in the order the error text lists them.
GHOST_MODES = ("repair", "refuse", "keep", "flag")


class GhostStateError(RuntimeError):
    """A generated channel binds a state below its reference energy."""


class GhostStateWarning(UserWarning):
    """A dataset in use carries a ghost state (or wrong scattering) that its
    generator could not remove."""


class DiagnosticError(RuntimeError):
    """A check of a dataset could not be evaluated: a level, a logarithmic
    derivative or a potential is not finite.  Never a pass: ``max(0.0, nan)`` is
    ``0.0`` and every comparison with NaN is false, so an unevaluated check
    would otherwise read as a perfect one."""


def check_mode(mode) -> str:
    """``mode`` when it is one of :data:`GHOST_MODES`, else ``ValueError``."""
    if mode not in GHOST_MODES:
        raise ValueError(f"ghosts must be one of {GHOST_MODES}, not {mode!r}")
    return mode


def read_defects(record) -> dict:
    """The ``defects`` of a stored record; ``{}`` when there are none."""
    record = record or {}
    ghosts = {int(l): float(e) for l, e in (record.get("ghosts") or {}).items()}
    phases = {int(l): (float(a), float(b))
              for l, (a, b) in (record.get("phases") or {}).items()}
    residuals = {int(l): float(e) for l, e in
                 (record.get("residuals") or {}).items()}
    s_miss = record.get("s_miss")
    unprojected = {int(l): float(e) for l, e in
                   (record.get("unprojected") or {}).items()}
    missing = {int(l): float(e) for l, e in (record.get("missing") or {}).items()}
    branch_ghosts = {str(k): float(e) for k, e in
                     (record.get("branch_ghosts") or {}).items()}
    if not (ghosts or phases or residuals or unprojected or missing
            or branch_ghosts) and s_miss is None:
        return {}
    result = {"ghosts": ghosts, "phases": phases}
    if residuals:
        result["residuals"] = residuals
    if s_miss is not None:
        result["s_miss"] = float(s_miss)
    if unprojected:
        result["unprojected"] = unprojected
    if missing:
        result["missing"] = missing
    if branch_ghosts:
        result["branch_ghosts"] = branch_ghosts
    return result


def warn_defects(dataset, family: str) -> None:
    """Raise the :class:`GhostStateWarning` of a dataset that has defects,
    on every load -- the library, a path or a user's own folder."""
    if getattr(dataset, "defects", None):
        warnings.warn(defect_message(dataset.symbol, family, dataset.defects),
                      GhostStateWarning, stacklevel=3)


def defect_message(symbol: str, family: str, defects: dict) -> str:
    """The warning text for a dataset's recorded ``defects``."""
    ghosts = defects.get("ghosts") or {}
    branch_ghosts = defects.get("branch_ghosts") or {}
    parts = [f"l={l} ghost {float(e):+.3g} Ha" for l, e in sorted(ghosts.items())]
    parts += [f"l={key.split(':')[0]} kappa={key.split(':')[1]} ghost {float(e):+.3g} Ha"
              for key, e in sorted(branch_ghosts.items())]
    parts += [f"l={l} phase error {float(near):.2f}/{float(far):.2f} rad"
              for l, (near, far) in sorted(defects.get("phases", {}).items())]
    parts += [f"l={l} residual kinetic energy {float(e):.3g} Ha"
              for l, e in sorted(defects.get("residuals", {}).items())]
    if defects.get("s_miss") is not None:
        parts.append(f"s projectors miss an intruding hydrogen 1s by "
                     f"{float(defects['s_miss']):.2f} of its norm")
    parts += [f"l={l} (no projectors) phase error {float(e):.2f} rad"
              for l, e in sorted(defects.get("unprojected", {}).items())]
    parts += [f"l={l} does not bind its reference (nearest level "
              f"{float(e):+.3g} Ha from it)"
              for l, e in sorted((defects.get("missing") or {}).items())]
    if ghosts or branch_ghosts:
        what = "ghost states"
        consequence = (f"A variational calculation containing {symbol} can "
                       f"collapse into a spurious state, so its energies and "
                       f"forces are not reliable.")
    elif defects.get("s_miss") is not None and not defects.get("phases"):
        what = "an incomplete s channel"
        consequence = (f"A neighbor's orbital entering the {symbol} sphere is "
                       f"not represented, so bonds to {symbol} can be too "
                       f"long or collapse.")
    elif defects.get("unprojected") and not defects.get("phases"):
        what = "a local potential that scatters wrongly"
        consequence = (f"Angular momenta without projectors see the local "
                       f"potential alone, so bonds and lattice constants of "
                       f"{symbol} compounds can be several percent off.")
    elif defects.get("residuals"):
        what = "divergent projector residuals"
        consequence = (f"Its projectors are numerically unreliable, so "
                       f"energies and forces of systems containing "
                       f"{symbol} carry an error of unknown size.")
    elif defects.get("missing") and not defects.get("phases"):
        what = "a missing bound state"
        consequence = (f"Its smooth Hamiltonian does not bind a state the atom "
                       f"has, so energies and forces of systems containing "
                       f"{symbol} carry an error of unknown size.")
    else:
        what = "scattering errors"
        consequence = (f"Its channels do not scatter like the all-electron "
                       f"atom, so energies and forces of systems containing "
                       f"{symbol} carry an error of unknown size.")
    return (f"the {family} dataset for {symbol} has {what} "
            f"({'; '.join(parts)}).  {consequence}")
