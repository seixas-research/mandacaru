"""The PAW-LCAO basis options of a calculator, as the Rust recipe (``paw::basis_recipe``) reads them.

One translation for molecules and crystals: each element's size, ``energy_shift``, confining wall,
split-valence scheme, polarization shell and diffuse shells, resolved with the option rules of
:mod:`~mandacaru.pseudopotentials.confinement`, :mod:`~mandacaru.models.radial` and
:mod:`~mandacaru.pseudopotentials.diffuse`. The shells themselves are built in Rust; the molecular
route receives them here (:func:`basis_tables`), the crystal setup builds them from the same
recipe inside ``periodic::setup``.
"""

import json
from types import SimpleNamespace

import numpy as np

from .. import _native
from ..units import to_bohr


def _per_element(spec, symbol):
    if isinstance(spec, dict):
        table = {(k if k == "*" else str(k).capitalize()): v for k, v in spec.items()}
        if symbol.capitalize() not in table and "*" not in table:
            raise ValueError(f"no pseudo-basis size for element {symbol!r}; add it or a '*' "
                             "default to the size mapping")
        return table.get(symbol.capitalize(), table.get("*"))
    return spec


def _diffuse(recipe):
    """The Rust form of one element's ``DiffuseRecipe`` (``rms_radius`` in Bohr)."""
    if recipe is None:
        return None
    return dict(shells=None if recipe.shells == "present" else list(recipe.shells),
                count=int(recipe.count), ratio=float(recipe.ratio),
                cutoff_norm=float(recipe.cutoff_norm), rms_ratio=float(recipe.rms_ratio),
                rms_radius=(None if recipe.rms_radius is None
                            else to_bohr(float(recipe.rms_radius))))


def element_recipe(family, options, symbol, cutoff):
    """The recipe of ``symbol``'s shells under the basis ``options`` of ``family``, filtered at
    ``cutoff`` (Bohr^-1; ``None`` unfiltered)."""
    from ..models.radial import DEFAULT_SPLIT_NORM, resolve_split_scheme, resolve_zeta
    from ..pseudopotentials.confinement import (DEFAULT_ENERGY_SHIFT, energy_shift_of,
                                                resolve_polarization_method,
                                                validate_confinement_potential)
    from ..pseudopotentials.diffuse import diffuse_of

    confinable = "energy_shift" in family.options
    # An absent key is the default shift; an explicit None is the unconfined orbital.
    spec = options.get("energy_shift", DEFAULT_ENERGY_SHIFT) if confinable else None
    shift = energy_shift_of(spec, symbol) if confinable else None
    zetas, polarizations = resolve_zeta(_per_element(options.get("size", "SZ"), symbol))
    method = (resolve_polarization_method(options.get("polarization_method"), spec, symbol)
              if confinable else "orbital")
    if shift is None and method == "gaussian" and polarizations > 0:
        raise ValueError(
            f"polarization_method='gaussian' needs an energy_shift for {symbol!r}: the Gaussian "
            "shell takes its cutoff from the confined orbital, and an unconfined orbital has none")
    split, tails = resolve_split_scheme(options.get("zeta_split"))
    diffuse = options.get("diffuse") if "diffuse" in family.options else None
    return dict(
        zetas=zetas, polarizations=polarizations, cutoff=cutoff, energy_shift=shift,
        wall=list(validate_confinement_potential(options.get("confinement_potential"))),
        split_norm=DEFAULT_SPLIT_NORM if split is None else split,
        tail_norms=None if tails is None else list(tails), polarization=method,
        diffuse=_diffuse(diffuse_of(diffuse, symbol) if diffuse else None))


def basis_tables(context, family, options, species, cutoff, report):
    """``{symbol: [shell]}``: each element's radial shells from the Rust recipe (``n``, ``l``,
    ``zeta``, ``polarization``, ``diffuse``, ``r``, ``values``), with the confined orbitals and the
    Gaussian shell left in ``report`` for the run log."""
    from ..pseudopotentials.record import record_of

    out = {}
    for symbol, pp in species.items():
        built = _native.paw_basis(context, *record_of(pp),
                                  json.dumps(element_recipe(family, options, symbol, cutoff)))
        out[symbol] = [SimpleNamespace(n=n, l=l, zeta=zeta, polarization=polarization,
                                       diffuse=diffuse, r=np.asarray(r), values=np.asarray(values))
                       for n, l, zeta, polarization, diffuse, r, values in built["shells"]]
        for c in built["confined"]:
            report["confinement"].setdefault(symbol, {})[(c["l"], c["state"])] = SimpleNamespace(**c)
        if built["polarization"] is not None:
            report["polarization"][symbol] = SimpleNamespace(**built["polarization"])
        if built["diffuse"] and "diffuse" in report:
            report["diffuse"][symbol] = [SimpleNamespace(symbol=symbol, **row) for row in built["diffuse"]]
    return out


def basis_entries(symbols, tables_of, hold):
    """The molecular input's ``basis`` list: one entry per ``m`` of every shell of every atom, each
    radial table held once per element (``hold`` registers an array and returns its key)."""
    held, entries = {}, []
    for atom, symbol in enumerate(symbols):
        for index, shell in enumerate(tables_of[symbol]):
            if (symbol, index) not in held:
                held[(symbol, index)] = hold(shell.r), hold(shell.values)
            r_key, values_key = held[(symbol, index)]
            entries.extend(dict(atom=atom, l=int(shell.l), m=int(m), r=r_key, values=values_key,
                                diffuse=bool(shell.diffuse))
                           for m in range(-shell.l, shell.l + 1))
    return entries
