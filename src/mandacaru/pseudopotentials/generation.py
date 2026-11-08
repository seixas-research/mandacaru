# -*- coding: utf-8 -*-
# file: pseudopotentials/generation.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Generating a PAW-LCAO or UPAW-LCAO dataset: the options, and the call into the engine.

The construction (reference atom, smooth partial waves, projectors, one-center terms, the
Dirac spin-orbit chain) and the search that repairs a ghost state or a wrong scattering are
native: ``mandacaru-core`` ``paw::generate``.  Which radial exchange-correlation scheme the
reference atom and the unscreening use is not an option: production uses the variational
scheme of the molecular PAW path.  This module validates and words the options, hands them
across, maps a refusal of the search to :class:`~.defects.GhostStateError`, and wraps the
record in the :class:`~.paw.PAWDataset` view.

``ghosts`` is what happens to a construction that binds a ghost state or scatters wrongly:
``"repair"`` (the default) rebuilds it, ``"refuse"`` raises, ``"keep"`` returns the first
construction unexamined, ``"flag"`` returns the least defective one with its defects
recorded (loading it warns).

An option the engine cannot express raises: free waves (``norm_deficit=None``) have no place
in the dataset record, which stores the deficit, and a reference atom solved elsewhere
(``atom=``) cannot cross the boundary because the search shares one atom across its attempts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real

from .. import _native
from ..basis.relativity import _resolve as _resolve_relativity
from .defects import GhostStateError, check_mode
from .native_context import context
from .record import dataset_from_record


PAW_FAMILY = "paw-lcao"
UPAW_FAMILY = "upaw-lcao"


@dataclass(frozen=True)
class SearchReport:
    """What the search noted about the dataset it returned (``dataset.search``).

    ``index`` is the position of the selected construction (0 is the first one) and
    ``label`` its remedy; ``tried`` lists the attempts that were not accepted, in order;
    ``flagged`` says the dataset still has a defect (its ``defects`` say which);
    ``built`` counts the constructions actually generated; ``missing`` is
    ``{l: nearest level - reference}`` of the channels that do not bind a physical
    reference; ``notes`` are observations of the generator that are not numerics.
    """

    index: int
    label: str
    tried: tuple = ()
    flagged: bool = False
    built: int = 1
    missing: dict = field(default_factory=dict)
    notes: tuple = ()


def _pairs(name: str, value) -> list:
    """``value`` as a list of ``(n, l)`` integer pairs, or the error the option always gave."""
    if value is None:
        return []
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{name} must be a list of (n, l) pairs")
    pairs = []
    for orbital in value:
        if (not isinstance(orbital, tuple) or len(orbital) != 2
                or any(isinstance(x, bool) or not isinstance(x, Integral)
                       for x in orbital)):
            raise ValueError(f"{name} must contain (n, l) integer pairs")
        pairs.append((int(orbital[0]), int(orbital[1])))
    return pairs


def _options(family: str, xc, relativity, ghosts, atom, r_cut, rc_factor, r_cut_local,
             local_factor, local_shift, q_cut, energy_offset, second_reference,
             n_bessel, norm_deficit, points, r_max, nlcc, extra_l, frozen_subshells,
             semicore_subshells, reference_configuration) -> dict:
    """The options the caller named, validated, for ``_native.paw_generate``."""
    if atom is not None:
        raise NotImplementedError(
            "a reference atom solved elsewhere cannot be handed to the generator: the "
            "ghost search shares one atom across its attempts and solves it itself")
    if norm_deficit is None:
        raise NotImplementedError(
            "norm_deficit=None (free smooth waves) is no longer available: the dataset "
            "record stores the norm deficit, so a free construction has no place in it")
    if xc not in ("lda", "pbe"): raise ValueError(f"expected canonical XC name in ('lda', 'pbe'), got {xc!r}")  # noqa: E701
    options = {"family": family, "xc": xc,
               "relativity": _resolve_relativity(relativity),
               "ghosts": check_mode(ghosts)}
    named = {"rc_factor": rc_factor, "r_cut_local": r_cut_local,
             "local_factor": local_factor, "local_shift": local_shift,
             "q_cut": q_cut, "energy_offset": energy_offset, "n_bessel": n_bessel,
             "points": points, "r_max": r_max, "extra_l": extra_l}
    for name, value in named.items():
        if value is not None:
            options[name] = value
    if r_cut is not None:
        options["r_cut"] = ({int(l): float(r) for l, r in r_cut.items()}
                            if isinstance(r_cut, dict) else float(r_cut))
    if norm_deficit != "default":
        options["norm_deficit"] = float(norm_deficit)
    if second_reference is not None:
        options["second_reference"] = {int(l): float(e)
                                       for l, e in second_reference.items()}
    if nlcc is not None:
        options["nlcc"] = nlcc if isinstance(nlcc, bool) else float(nlcc)
    if frozen_subshells is not None:
        options["frozen_subshells"] = _pairs("frozen_subshells", frozen_subshells)
    if semicore_subshells is not None:
        options["semicore_subshells"] = _pairs("semicore_subshells", semicore_subshells)
    if reference_configuration is not None:
        options["reference_configuration"] = {
            (int(n), int(l)): float(q) for (n, l), q in reference_configuration.items()}
    for name in ("rc_factor", "r_cut_local", "local_factor", "local_shift", "q_cut",
                 "energy_offset", "r_max"):
        if name in options and not isinstance(options[name], Real):
            raise TypeError(f"{name} must be a number")
    return options


def _generate(symbol: str, family: str, **options):
    answer = _native.paw_generate(context(), symbol, _options(family, **options))
    if "refusal" in answer:
        raise GhostStateError(answer["refusal"])
    dataset = dataset_from_record(answer["payload"], answer["tables"], warn=False)
    dataset.search = SearchReport(
        index=answer["index"], label=answer["label"], tried=tuple(answer["tried"]),
        flagged=answer["flagged"], built=answer["built"],
        missing=dict(answer["missing"]), notes=tuple(answer["notes"]))
    return dataset


def generate_paw(symbol: str, *, r_cut=None, rc_factor=None, r_cut_local=None,
                 local_factor=None, local_shift=None, q_cut=None, energy_offset=None,
                 second_reference=None, n_bessel=None, norm_deficit="default",
                 points=None, r_max=None, atom=None, xc: str = "lda",
                 relativity: str = "scalar", nlcc=None, extra_l=None,
                 frozen_subshells=None, semicore_subshells=None,
                 reference_configuration=None, ghosts: str = "repair"):
    r"""Generate a PAW-LCAO dataset for ``symbol``.

    Every option left at ``None`` takes the value the policy tables give the element and
    functional (``mandacaru-core``: ``paw::generate::tables_*``), and a value the caller
    names is the caller's: the search never changes ``r_cut``, ``r_cut_local`` or
    ``local_shift`` (a ghost there is refused rather than repaired).

    Parameters
    ----------
    r_cut : float or dict, optional
        Augmentation radii (Bohr), one value or ``{l: r_c}``.
    rc_factor : float, optional
        Times the outermost maximum of the bound partial wave, when no table gives the
        radius (default 1.3).
    r_cut_local : float, optional
        Radius of the polynomial local potential (default ``local_factor`` times the
        largest augmentation radius).
    local_factor : float, optional
        Default 0.9.
    local_shift : float, optional
        Raise of the local potential at the origin above its polynomial continuation
        (Hartree), the knob against ghost states of a deep local potential.
    q_cut : float, optional
        Wave-vector cutoff of the residual kinetic energy (Bohr^-1; default 5).
    energy_offset : float, optional
        Second reference energy above the bound state (Hartree).
    second_reference : dict, optional
        ``{l: energy}``: absolute second reference energies of single channels.
    n_bessel : int, optional
        Spherical Bessel functions per smooth partial wave.
    norm_deficit : float, optional
        Fraction of the all-electron inner norm the smooth waves give up; ``"default"``
        takes the table.  ``None`` (free waves) raises ``NotImplementedError``.
    points, r_max : optional
        Radial grid points of the all-electron atom and its extent (Bohr; default 30).
    xc : {"lda", "pbe"}
        The exchange-correlation functional of the reference atom.
    relativity : {"none", "scalar", "dirac"}
        ``"dirac"`` is scalar-relativistic plus the j-resolved spin-orbit term.
    nlcc : bool or float, optional
        The nonlinear core correction: on (the default), off, or the radius at which the
        true core is re-pseudized.
    extra_l : int, optional
        Empty channels above the highest valence one.
    frozen_subshells, semicore_subshells : sequence of tuple, optional
        ``(n, l)`` pairs: occupied valence subshells generated into the frozen core, and
        filled core subshells generated into the valence.
    reference_configuration : dict, optional
        ``{(n, l): occupancy}`` laid over the relaxed ground-state configuration.
    atom : None
        A pre-solved atom cannot cross the boundary; anything else raises.
    ghosts : {"repair", "refuse", "keep", "flag"}
        See the module docstring.

    The returned dataset carries ``search``, the :class:`SearchReport` of the selection.
    """
    return _generate(symbol, PAW_FAMILY, xc=xc, relativity=relativity, ghosts=ghosts,
                     atom=atom, r_cut=r_cut, rc_factor=rc_factor, r_cut_local=r_cut_local,
                     local_factor=local_factor, local_shift=local_shift, q_cut=q_cut,
                     energy_offset=energy_offset, second_reference=second_reference,
                     n_bessel=n_bessel, norm_deficit=norm_deficit, points=points,
                     r_max=r_max, nlcc=nlcc, extra_l=extra_l,
                     frozen_subshells=frozen_subshells,
                     semicore_subshells=semicore_subshells,
                     reference_configuration=reference_configuration)


def generate_upaw(symbol: str, *, r_cut=None, rc_factor=None, r_cut_local=None,
                  local_factor=None, local_shift=None, q_cut=None, energy_offset=None,
                  second_reference=None, n_bessel=None, points=None, r_max=None,
                  atom=None, xc: str = "lda", relativity: str = "scalar", nlcc=None,
                  extra_l=None, frozen_subshells=None, semicore_subshells=None,
                  reference_configuration=None, ghosts: str = "repair", **unknown):
    r"""Generate a **unitary** PAW-LCAO (UPAW-LCAO) dataset: the ``norm_deficit = 0`` PAW-LCAO.

    The smooth partial waves carry the full all-electron inner norm, so :math:`q_{ij} = 0`,
    the transformation satisfies :math:`T^\dagger T = I` and the overlap operator is the
    identity -- the construction of Ivanov *et al.* (arXiv:2408.03159).  The per-element
    recipes of the family (empty channels, semicore shells, radii, raises) are policy
    tables of the engine and lie under the options named here.  Everything else is
    :func:`generate_paw`.

    ``norm_deficit`` is not accepted: it *is* what distinguishes the two families, and a
    UPAW-LCAO dataset with a nonzero deficit would be a PAW-LCAO dataset wearing the wrong
    name.
    """
    if "norm_deficit" in unknown:
        raise TypeError(
            "generate_upaw does not take `norm_deficit`: UPAW-LCAO is defined by "
            "norm_deficit = 0 (use generate_paw for any other value)")
    if unknown:
        raise TypeError(f"generate_upaw got unexpected keyword argument(s) "
                        f"{sorted(unknown)}")
    return _generate(symbol, UPAW_FAMILY, xc=xc, relativity=relativity, ghosts=ghosts,
                     atom=atom, r_cut=r_cut, rc_factor=rc_factor, r_cut_local=r_cut_local,
                     local_factor=local_factor, local_shift=local_shift, q_cut=q_cut,
                     energy_offset=energy_offset, second_reference=second_reference,
                     n_bessel=n_bessel, norm_deficit="default", points=points,
                     r_max=r_max, nlcc=nlcc, extra_l=extra_l,
                     frozen_subshells=frozen_subshells,
                     semicore_subshells=semicore_subshells,
                     reference_configuration=reference_configuration)
