# -*- coding: utf-8 -*-
# file: algorithms/convergence.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""When an adaptive solver stops growing its ansatz.

An adaptive driver (ADAPT-VQE, subspace ADAPT-VQE, the deflated excited
states) appends one operator per growth step.  It stops on the
``convergence`` option, a dictionary of up to two criteria:

``"gradient"``
    the largest pool gradient at the current state, :math:`\max_i |g_i|`
    (Hartree), must fall below this value.  A small screening gradient says
    no pool operator can lower the energy to first order.
``"energy"``
    the energy change of the last growth step, :math:`|\Delta E|`
    (**eV**, whatever unit the run reports energies in), must fall below this
    value -- the ``dE`` column of the run
    log's ``[ITERATIONS]`` table, the first step measured from the reference
    state.  It says growth has stopped paying.

A criterion set to ``None``, or left out of the dictionary, is not used.
With both set, **both** must hold, as in the usual quantum-chemistry
convergence tests: a small energy change alone can be a plateau the next
operator would leave, and a small gradient alone can sit on a flat saddle.
The energy criterion cannot hold before the first growth step, since there is
no energy change yet.

A **vanished gradient** stops the growth whatever the criteria say: below
:data:`VANISHED_GRADIENT` no pool operator can change the energy.  Without it,
a system one operator makes exact (H2) would still grow a redundant second
one, because that first step's energy change is large.

``convergence=None`` is :data:`DEFAULT_CONVERGENCE`: both criteria, at
:data:`DEFAULT_GRADIENT_CONVERGENCE` and :data:`DEFAULT_ENERGY_CONVERGENCE`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from ..units import EV_TO_HARTREE

#: Largest pool gradient (Hartree) below which growth stops by default.
DEFAULT_GRADIENT_CONVERGENCE = 1e-3
#: Energy change of the last growth step (eV) below which growth stops by
#: default.
DEFAULT_ENERGY_CONVERGENCE = 1e-3
#: Largest pool gradient (Hartree) at which growth stops under any criteria.
#: Appending an operator with screening gradient g lowers the energy by about
#: g^2 / (2 kappa), kappa ~ 0.1-1 Ha the curvature along it: ~1e-10 Ha here,
#: far below any energy threshold.  Well above the residual gradient the
#: inner optimizer leaves at an exact state (H2: 1.5e-6 after its one
#: operator), which a floor at round-off level would not catch.
VANISHED_GRADIENT = 1e-5
#: The criteria ``convergence`` accepts, in the order they are reported.
CRITERIA = ("gradient", "energy")


@dataclass(frozen=True)
class Convergence:
    """The resolved ``convergence`` option: a threshold, or ``None``, per
    criterion (``gradient`` in Hartree, ``energy`` in eV)."""

    gradient: float | None = DEFAULT_GRADIENT_CONVERGENCE
    energy: float | None = DEFAULT_ENERGY_CONVERGENCE

    def __post_init__(self) -> None:
        for name in CRITERIA:
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"convergence[{name!r}] must be a positive "
                                 f"number or None, not {value!r}")
            object.__setattr__(self, name, float(value))
        if self.gradient is None and self.energy is None:
            raise ValueError(
                "convergence needs at least one criterion: give 'gradient' "
                "and/or 'energy' a threshold (Hartree and eV); with neither, only "
                "max_iterations would stop the growth")

    @classmethod
    def resolve(cls, spec: Convergence | Mapping[str, float | None] | None
                ) -> Convergence:
        """``None`` (the default criteria), a :class:`Convergence`, or a
        dictionary with keys from :data:`CRITERIA`; a key left out is a
        criterion not used."""
        if spec is None:
            return DEFAULT_CONVERGENCE
        if isinstance(spec, Convergence):
            return spec
        if not isinstance(spec, Mapping):
            raise TypeError(
                "convergence must be a dict such as {'gradient': 1e-3, "
                f"'energy': 1e-6}}, not {type(spec).__name__}")
        unknown = sorted(set(spec) - set(CRITERIA))
        if unknown:
            raise ValueError(f"unknown convergence criteria {unknown}; "
                             f"available: {list(CRITERIA)}")
        return cls(gradient=spec.get("gradient"), energy=spec.get("energy"))

    @property
    def energy_hartree(self) -> float | None:
        """The energy threshold in Hartree, the unit the drivers work in."""
        return None if self.energy is None else self.energy * EV_TO_HARTREE

    def reached(self, max_gradient: float | None,
                delta_energy: float | None) -> bool:
        """Whether every criterion that is set holds, or the gradient has
        vanished (:data:`VANISHED_GRADIENT`).

        ``delta_energy`` is in Hartree, as the drivers compute it, and is
        ``None`` before the first growth step, when the energy criterion does
        not hold.
        """
        if max_gradient is not None and max_gradient < VANISHED_GRADIENT:
            return True
        if self.gradient is not None and not (
                max_gradient is not None and max_gradient < self.gradient):
            return False
        if self.energy is not None and not (
                delta_energy is not None
                and abs(delta_energy) < self.energy_hartree):
            return False
        return True

    def as_dict(self) -> dict[str, float | None]:
        """``{"gradient": ..., "energy": ...}``, the option as it was resolved."""
        return {name: getattr(self, name) for name in CRITERIA}

    def describe(self, energy_scale: float = 1.0,
                 energy_unit: str = "Ha") -> str:
        """One line, e.g. ``max|grad| < 0.001 Ha and |dE| < 0.001 eV``;
        ``energy_scale`` converts Hartree into ``energy_unit`` (1 for
        Hartree), so the energy threshold reads against the ``dE`` column."""
        parts = []
        if self.gradient is not None:
            parts.append(f"max|grad| < {self.gradient:g} Ha")
        if self.energy is not None:
            parts.append(f"|dE| < {self.energy_hartree * energy_scale:g} "
                         f"{energy_unit}")
        return " and ".join(parts)


#: What ``convergence=None`` means.
DEFAULT_CONVERGENCE = Convergence()
