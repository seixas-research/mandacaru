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


def require_crystal(spec, pbc):
    """Raise when DFT+U is asked for a geometry with no periodic direction."""
    if spec is not None and not any(pbc):
        raise ValueError(
            "hubbard_u needs a periodic cell: DFT+U is implemented for crystals only (the "
            "molecular PAW route has no +U term and refuses d and f cores); give the system "
            "a periodic direction"
        )


def table(spec, context, crystal, moments):
    """The ``hubbard`` object of the snapshot of ``crystal`` for the option ``spec``.

    ``None`` for no option. The crystal's projector labels (atom, ``l``, ``m``, radial index,
    symbol, projector basis), one reference-atom dataset per distinct dataset object and the
    atoms' initial moments go to ``_native.hubbard_table``, which builds the table.
    """
    if spec is None:
        return None
    from .pseudopotentials.onecenter import _occupations

    projectors = crystal.projectors
    slot, species, of_atom = {}, [], []
    for dataset in crystal.datasets:
        if id(dataset) not in slot:
            slot[id(dataset)] = len(species)
            species.append(
                (
                    np.ascontiguousarray(dataset.r, dtype=float),
                    {
                        int(l): (
                            np.ascontiguousarray(channel.ae_waves, dtype=float),
                            np.ascontiguousarray(channel.vanderbilt, dtype=float),
                            float(channel.r_cut),
                            np.asarray(_occupations(dataset, l), dtype=float),
                        )
                        for l, channel in dataset.channels.items()
                    },
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
