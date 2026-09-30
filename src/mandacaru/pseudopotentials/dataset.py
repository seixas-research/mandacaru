# -*- coding: utf-8 -*-
# file: pseudopotentials/dataset.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""The dataset layout every pseudopotential family shares.

Why Mandacaru needs pseudopotentials
------------------------------------
The real-space integral engine cannot resolve a heavy-atom core.  The :math:`1s`
shell has length scale :math:`a_0/Z` -- 0.066 Angstrom for oxygen -- while a
practical grid spacing is 0.15--0.30 Angstrom.  The consequences were measured in
:mod:`mandacaru.algorithms.forces`: a spurious force of :math:`\sim\!10^3`
eV/Angstrom on an isolated oxygen atom (whose exact force is zero by symmetry),
growing rather than shrinking under grid refinement, and an egg-box energy error
of ~178 eV for water.  Neither is a bug in the gradient; both come from
integrating :math:`-Z/r` and a cusped core density on a grid too coarse for them.

A pseudopotential removes the cause.  The core electrons are taken out of the
calculation entirely and the singular :math:`-Z/r` is replaced by a *smooth*
potential that reproduces the valence scattering properties.  The length scale
the grid must resolve then becomes the valence one (~1 Bohr), which existing
grids already handle comfortably.

This module holds what the families have in common: :class:`Channel` (one
angular-momentum channel), :class:`PseudoPotential` (the per-element record
the valence basis, the local potential sampler and the multiple-zeta hierarchy
read), and two helpers of their generators.  The families themselves --
ONCVPSP (:mod:`.oncv`) and PAW-LCAO (:mod:`.paw`) -- subclass both.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..basis._config import ground_state_config, valence_subshells
from ..basis.atomic_solver import AtomicResult


def _local_derivatives(r: np.ndarray, values: np.ndarray, r0: float,
                       order: int = 4, window: int = 25, degree: int = 8):
    """Derivatives of ``values(r)`` at ``r0``, up to ``order``.

    Fits a polynomial in ``(r - r0)`` over a window of grid points centered on
    ``r0`` and reads the derivatives off its coefficients.  A local polynomial is
    used rather than repeated finite differencing because matching a smooth
    function at a cutoff needs derivatives up to the *fourth*, which naive
    differencing renders useless.
    """
    index = int(np.argmin(np.abs(r - r0)))
    low = max(index - window, 0)
    high = min(index + window + 1, r.size)
    shifted = r[low:high] - r0
    coefficients = np.polyfit(shifted, values[low:high], degree)
    # np.polyfit returns highest power first; derivative k at 0 is k! * a_k.
    ascending = coefficients[::-1]
    factorial = 1.0
    out = []
    for k in range(order + 1):
        out.append(ascending[k] * factorial if k < ascending.size else 0.0)
        factorial *= (k + 1)
    return np.array(out)


@dataclass
class Channel:
    """One angular-momentum channel of a pseudopotential.

    ``pseudo_radial``/``eigenvalue``/``coefficients`` describe the channel's
    first (bound) smooth partial wave, which is also its first-zeta basis
    function; a family's subclass adds the rest of its partial-wave set.
    """

    l: int
    n: int                              # principal quantum number of the valence shell
    eigenvalue: float                   # Hartree, reproduced by construction
    r_cut: float                        # Bohr
    coefficients: np.ndarray            # expansion of the first smooth wave
    pseudo_radial: np.ndarray           # R_ps(r) on the atom's grid
    v_screened: np.ndarray              # screened channel potential
    v_ionic: np.ndarray = None          # after unscreening
    occupation: float = 0.0
    norm_error: float = 0.0

    def __repr__(self) -> str:
        return (f"Channel(l={self.l}, n={self.n}, "
                f"eps={self.eigenvalue:+.6f} Ha, rc={self.r_cut:.3f} Bohr)")


@dataclass
class PseudoPotential:
    r"""The per-element record every pseudopotential family builds on.

    The potential acting on the valence electrons is a smooth local part
    :attr:`v_local` (no :math:`-Z/r` singularity) plus a separable nonlocal
    part built from :attr:`projectors`, whose coupling each family defines.
    """

    symbol: str
    atomic_number: int
    valence_charge: float               # Z_ion = Z - n_core electrons
    r: np.ndarray
    channels: dict                      # l -> Channel
    v_local: np.ndarray                 # ionic local potential
    #: Pseudopotential **family** this object belongs to -- the key of
    #: :data:`mandacaru.pseudopotentials.families.PSEUDO_FAMILIES`.
    #: Written to and read from the library files.
    family: str
    projectors: dict = field(default_factory=dict)   # l -> [chi(r), ...]
    valence_density: np.ndarray = None
    atom: AtomicResult = None
    #: How the reference atom was solved.
    xc: str = "lda"
    relativity: str = "none"
    #: Whether the functional carried the relativistic exchange correction
    #: (:func:`~mandacaru.basis.xc.relativistic_exchange_factors`) -- what a
    #: relativistic reference atom uses.  Stored, not derived from
    #: ``relativity``: a file written before the correction existed is
    #: relativistic without it, and rescreening it must match how it was
    #: unscreened.
    relativistic_exchange: bool = False
    #: Partial core density the local potential was unscreened with (zero
    #: without a core correction), and the record of how it was built
    #: (:mod:`.core_correction`).
    core_density: np.ndarray = None
    nlcc: dict = field(default_factory=dict)
    #: What its generator could not remove (``ghosts="flag"``):
    #: ``{"ghosts": {l: depth}, "phases": {l: (near, far)}}``.
    defects: dict = field(default_factory=dict)

    def local_potential(self, radius) -> np.ndarray:
        """Interpolate ``V_loc`` onto arbitrary radii (Bohr).

        Beyond the grid the potential is the bare ionic tail
        :math:`-Z_{\\text{ion}}/r`, which is what it decays to by construction.
        """
        radius = np.asarray(radius, dtype=float)
        inside = radius <= self.r[-1]
        out = np.where(inside,
                       np.interp(np.clip(radius, self.r[0], self.r[-1]),
                                 self.r, self.v_local),
                       -self.valence_charge / np.maximum(radius, 1e-12))
        # At r below the first grid point the potential is flat (finite).
        return np.where(radius < self.r[0], self.v_local[0], out)


def _valence_configuration(atomic_number: int, configuration=None):
    """``(valence subshells, core subshells)`` of the reference atom.

    ``configuration`` defaults to the aufbau filling; a generator that has
    already solved its reference atom passes that atom's ``occupations``, so
    the split describes the atom whose orbitals are about to be pseudized
    rather than a different one.
    """
    configuration = (ground_state_config(atomic_number)
                     if configuration is None
                     else {k: v for k, v in configuration.items() if v > 0})
    valence = set(valence_subshells(atomic_number,
                                    configuration=configuration))
    valence_config = {k: v for k, v in configuration.items() if k in valence}
    core_config = {k: v for k, v in configuration.items() if k not in valence}
    return valence_config, core_config
