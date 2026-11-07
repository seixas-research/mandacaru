"""The ``hubbard_u`` option of a periodic DFT calculation: the hand-off to the native engine.

Rust reads and decides everything (``mandacaru_core::models::hubbard_request``: the mapping,
its units, its form, every refusal; ``mandacaru_core::periodic::hubbard_table``: which
projectors form each atom's shell, the weights and the seed, and the comparison with a loaded
crystal file). Python keeps the option as the JSON text of the user's mapping, hands it over
together with the crystal's labels and reference-atom arrays, and returns the ``hubbard``
object of the crystal snapshot that Rust writes.
"""

from __future__ import annotations

import json

import numpy as np

from . import _native


def _plain(value):
    """JSON form of a value the encoder does not know: NumPy scalars become Python ones, any
    other object its ``repr`` (a string, which Rust then refuses as a value)."""
    return value.item() if isinstance(value, np.generic) else repr(value)


def validate(spec, *, xc, precision):
    """Check ``hubbard_u=`` and return its JSON text, or ``None`` for no option.

    ``spec`` maps an element symbol to ``{"l": "d" or "f", "U": ..., "J": ...}`` with the
    energies in eV, ``J`` defaulting to 0 and an optional ``"form"``. Rust refuses a malformed
    mapping, a screened hybrid functional and a single-precision calculation, with the reason.
    """
    if spec is None:
        return None
    try:
        text = json.dumps(spec, default=_plain)
    except (TypeError, ValueError):
        text = json.dumps(repr(spec))
    _native.hubbard_validate(text, xc, precision)
    return text


def require_crystal(spec, pbc, method="dft"):
    """Check that system and method support hubbard_u."""
    if spec is None:
        return
    if not any(pbc) and method != "dft":
        raise ValueError(
            "hubbard_u for molecules is implemented for method='dft' only"
        )


def paw_basis(spec, basis):
    """The ``(family, options)`` of the PAW basis that ``hubbard_u`` needs, or ``None`` for no
    option.

    The Hubbard shell is the projection on a PAW partial wave of the dataset, so a basis with
    no dataset (``HAO``, Gaussian families, per-element bases) cannot carry the option.
    """
    if spec is None:
        return None
    from .models.setup import pseudopotential_family, resolve_basis

    name, options = resolve_basis(basis)
    family = pseudopotential_family(name)
    if family is None or family.name not in ("paw-lcao", "upaw-lcao"):
        raise ValueError(
            "hubbard_u needs a PAW-LCAO or UPAW-LCAO basis: the correlated shell is the "
            f"projection on a PAW partial wave of the dataset, and basis {name!r} has none; "
            "give basis='PAW-LCAO' (or 'UPAW-LCAO')"
        )
    return family.name, dict(options)


def _channels(dataset):
    """The reference occupations of each channel of ``dataset``, one number per partial wave."""
    from .pseudopotentials.onecenter import _occupations

    return {
        int(l): np.asarray(_occupations(dataset, l), dtype=float) for l in dataset.channels
    }


def check_atoms(spec, context, atoms, paw, xc):
    """Refuse at construction what the elements and their datasets already rule out.

    Loads the dataset of each distinct element (cached, and read again by the calculation) from
    the set of the functional ``xc`` unless the basis names a folder, and
    hands its file and reference occupations to Rust, which refuses a spin-orbit dataset, an
    element the option names that ``atoms`` lacks, a dataset without the shell and a shell
    with no occupied reference wave. What needs the assembled crystal (a k mesh reduced by
    symmetry) is refused at the first calculation, before any SCF.
    """
    if spec is None or paw is None or not any(atoms.pbc):
        return
    from .pseudopotentials.environment import with_functional_set
    from .pseudopotentials.families import resolve_family
    from .pseudopotentials.paw import get_paw, get_upaw

    family, options = paw
    options = with_functional_set(family, resolve_family(family).resolved_options(options), xc)
    folder = options.get("paw_dataset")
    load = get_upaw if family == "upaw-lcao" else get_paw
    symbols = sorted(set(atoms.get_chemical_symbols()))
    datasets = [load(symbol, folder) for symbol in symbols]
    _native.hubbard_check_atoms(
        context,
        spec,
        any(getattr(dataset, "has_spin_orbit", False) for dataset in datasets),
        symbols,
        [str(getattr(dataset, "source", "")) for dataset in datasets],
        [_channels(dataset) for dataset in datasets],
    )


def table(spec, context, crystal, moments):
    """The ``hubbard`` object of the snapshot of ``crystal`` for the option ``spec``.

    ``None`` for no option. The crystal's projector labels (atom, ``l``, ``m``, radial index,
    symbol, projector basis), one reference-atom dataset per distinct dataset object and the
    atoms' initial moments go to ``_native.hubbard_table``, which builds the table.
    """
    if spec is None:
        return None
    projectors = crystal.projectors
    slot, species, of_atom = {}, [], []
    for dataset in crystal.datasets:
        if id(dataset) not in slot:
            slot[id(dataset)] = len(species)
            occupations = _channels(dataset)
            species.append(
                (
                    np.ascontiguousarray(dataset.r, dtype=float),
                    {
                        int(l): (
                            np.ascontiguousarray(channel.ae_waves, dtype=float),
                            np.ascontiguousarray(channel.vanderbilt, dtype=float),
                            float(channel.r_cut),
                            occupations[int(l)],
                        )
                        for l, channel in dataset.channels.items()
                    },
                    str(getattr(dataset, "source", "")),
                )
            )
        of_atom.append(slot[id(dataset)])
    labels = np.array(
        [(p.atom_index, p.l, p.m, p.index) for p in projectors], dtype=np.int64
    ).reshape(-1, 4)
    text = _native.hubbard_table(
        context,
        spec,
        bool(crystal.has_spin_orbit),
        [p.symbol for p in projectors],
        [str(getattr(p, "projector_basis", "raw")) for p in projectors],
        labels,
        of_atom,
        species,
        np.ascontiguousarray(moments, dtype=float),
    )
    return json.loads(text)


def check_loaded(spec, problem):
    """Compare the option with the shells a loaded crystal file carries.

    The file owns the correction: with no option it applies as written; an option that names
    other shells, other energies or another form raises ``ValueError`` (Rust compares).
    """
    if spec is not None:
        problem.check_hubbard_u(spec)
