# -*- coding: utf-8 -*-
# file: test/migration/freeze_paw_recipe.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Freeze the policy tables of the Python PAW generator before they are deleted (A3).

``generate_paw`` resolves, before it looks at the solved atom, the norm deficit, the Bessel
count, the second reference energies, the local shift, the augmentation radii of the tables,
the frozen and semicore subshells, the empty channels and the reference configuration of a
dataset from the module-level tables of ``pseudopotentials/paw.py``; ``generate_upaw`` lays
``DEFAULT_UPAW_OPTIONS`` over them.  This script writes what that resolution gives for every
element of the library (H-U) in LDA and PBE, plus the generic constants, to
``reference/paw_recipe/policy.json``, which ``crates/mandacaru-core/tests/paw_recipe.rs``
compares with ``paw::generate::recipe::Recipe`` entry by entry.  The resolution below is the
code of ``generate_paw`` (lines "dataset = ..." to "cutoff_table = ...") written out once;
``reference/paw_generate`` (the A0 freeze) is the independent check that it is the one the
generator used.

The script needs the Python generator, so it dies with it at the cutover (A7); the JSON stays.

Usage: ``python test/migration/freeze_paw_recipe.py``
"""

from __future__ import annotations

import json
from pathlib import Path

from ase.data import chemical_symbols

from mandacaru.pseudopotentials import paw

OUT = Path(__file__).resolve().parent / "reference" / "paw_recipe" / "policy.json"

#: The elements of the shipped library.
ELEMENTS = chemical_symbols[1:93]


def _radii(table: dict) -> dict:
    return {str(l): float(r) for l, r in sorted(table.items())}


def resolve(symbol: str, xc: str) -> dict:
    """What ``generate_paw`` fixes for ``(symbol, xc)`` without the atom."""
    dataset = (symbol, xc)
    return {
        "norm_deficit": float(paw.DEFAULT_NORM_DEFICITS_BY_DATASET.get(
            dataset, paw.DEFAULT_NORM_DEFICITS.get(symbol, paw.DEFAULT_NORM_DEFICIT))),
        "n_bessel": int(paw.DEFAULT_N_BESSEL_BY_DATASET.get(
            dataset, paw._S_CHANNEL_REPAIRS.get(dataset, (paw.DEFAULT_N_BESSEL,))[0])),
        "energy_offset": float(paw.DEFAULT_ENERGY_OFFSETS_BY_DATASET.get(
            dataset, paw.DEFAULT_ENERGY_OFFSETS.get(symbol, paw.DEFAULT_ENERGY_OFFSET))),
        "second_reference": {str(l): float(e) for l, e in sorted(
            paw.DEFAULT_SECOND_REFERENCES.get(dataset, {}).items())},
        "local_shift": float(paw.DEFAULT_LOCAL_SHIFTS_BY_DATASET.get(
            dataset, paw.DEFAULT_LOCAL_SHIFTS.get(symbol, 0.0))),
        "cutoff_table": _radii({**paw.DEFAULT_CUTOFFS.get(symbol, {}),
                                **paw.DEFAULT_CUTOFFS_BY_DATASET.get(dataset, {})}),
        "frozen_subshells": [list(s) for s in paw.DEFAULT_FROZEN_SUBSHELLS.get(symbol, ())],
        "semicore_subshells": [list(s) for s in paw.DEFAULT_SEMICORE_SUBSHELLS.get(symbol, ())],
        "highest_channel": paw.DEFAULT_HIGHEST_CHANNEL.get(symbol),
        "reference_configuration": [[n, l, float(q)] for (n, l), q in sorted(
            paw.DEFAULT_REFERENCE_CONFIGURATIONS.get(symbol, {}).items())],
        "core_radius": paw.DEFAULT_CORE_RADII.get(symbol),
    }


def upaw_options(symbol: str, xc: str) -> dict:
    """The options ``generate_upaw`` lays over the defaults (``generate_paw`` keywords)."""
    recipe = paw.DEFAULT_UPAW_OPTIONS.get((symbol, xc), {})
    out = {}
    for key, value in sorted(recipe.items()):
        if key == "r_cut":
            out["r_cut"] = _radii(value)
        elif key == "semicore_subshells":
            out[key] = [list(s) for s in value]
        elif key == "reference_configuration":
            out[key] = [[n, l, float(q)] for (n, l), q in sorted(value.items())]
        else:
            out[key] = value
    return out


def main() -> None:
    document = {
        "note": "policy tables of paw.py resolved per dataset; see freeze_paw_recipe.py",
        "constants": {
            "n_bessel": paw.DEFAULT_N_BESSEL, "q_cut": paw.DEFAULT_Q_CUT,
            "energy_offset": paw.DEFAULT_ENERGY_OFFSET, "norm_deficit": paw.DEFAULT_NORM_DEFICIT,
            "rc_factor": paw.DEFAULT_RC_FACTOR, "local_factor": paw.DEFAULT_LOCAL_FACTOR,
            "semicore_cutoff_fraction": paw.SEMICORE_CUTOFF_FRACTION,
            "frozen_scattering_energy": paw.FROZEN_SCATTERING_ENERGY,
            "gga_local_fit_fraction": paw.GGA_LOCAL_FIT_FRACTION,
            "extra_anchor_energies": list(paw.EXTRA_ANCHOR_ENERGIES),
        },
        "datasets": {f"{symbol}/{xc}": resolve(symbol, xc)
                     for symbol in ELEMENTS for xc in ("lda", "pbe")},
        "upaw": {f"{symbol}/{xc}": upaw_options(symbol, xc)
                 for symbol in ELEMENTS for xc in ("lda", "pbe")
                 if upaw_options(symbol, xc)},
        "generation_points": {symbol: paw.generation_points(i + 1)
                              for i, symbol in enumerate(ELEMENTS)}
        if hasattr(paw, "generation_points") else {},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
