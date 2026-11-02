"""Freeze the Python split-valence construction (``basis/multizeta.py``) before its numerics are deleted.

``zeta_tables`` (with ``split_radius`` and ``split_valence_tail``) becomes a one-call forwarder to the Rust
``radial_zetas``.  Run on the tree that still has the scipy implementation, this script writes the numbers the Rust
route must reproduce to ``reference/basis_zetas/``:

* ``synthetic``: the grids of ``test_native_basis_zetas.py`` (700 nodes of step 0.02, from 0 and from h), l = 0..2,
  split norms 0.15 and 0.3 with 1..4 zetas (last zeta), tail norms (0.15, 0.31, 0.54) with 2..4 zetas (first zeta);
* ``paw``: the O and Fe PAW-LCAO grids (they start at h, uniform to 1e-8), every channel's stored pseudo wave, 3 zetas,
  both schemes (the extra zetas only; the first one is the input);
* ``hydrogen_1s``: the confined hydrogen 1s of ``test/basis/test_multizeta.py`` (four zetas), plus ``split_radius`` for
  four split norms and the raw ``split_valence_tail`` at the default split.

Usage (repository root, the ``mandacaru`` environment, ``MANDACARU_PAW_PATH`` set for the ``paw`` cases):

    python test/migration/freeze_basis_zetas.py

It ran once, on the tree that still had the scipy implementation (2026-11-01, stamp in ``zeta_tables.json``). It cannot
run again: ``split_radius``, ``split_valence_tail`` and the scipy ``zeta_tables`` were deleted in the same change, and the
files it wrote are the oracle of ``test_native_basis_zetas.py`` and ``test/basis/test_multizeta.py``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OUT = HERE / "reference" / "basis_zetas"
STEP = 0.02
TAILS = (0.15, 0.31, 0.54)
SPLITS = (0.15, 0.3)
HYDROGEN_SPLITS = (0.05, 0.15, 0.30)


def grid(start):
    return start + STEP * np.arange(700)


def radial(r, l):
    """A bound-state shape that vanishes smoothly at the last node."""
    return r**l * np.exp(-1.3 * r) * np.clip(1 - (r / r[-1]) ** 2, 0, None) ** 2


def synthetic(zeta_tables) -> tuple[dict, list]:
    arrays, cases = {}, []
    for start in (0.0, STEP):
        r = grid(start)
        for l in (0, 1, 2):
            for count in (1, 2, 3, 4):
                for split in SPLITS:
                    cases.append(dict(start=start, l=l, count=count, split=split, tails=None))
                if count >= 2:
                    cases.append(dict(start=start, l=l, count=count, split=0.15, tails=list(TAILS)))
    for i, case in enumerate(cases):
        r = grid(case["start"])
        tables = zeta_tables(r, radial(r, case["l"]), case["l"] + 1, case["l"], case["count"], case["split"],
                             tail_norms=None if case["tails"] is None else tuple(case["tails"]))
        case["count_out"] = len(tables)
        for k, table in enumerate(tables):
            arrays[f"syn{i}_{k}"] = np.asarray(table.values)
    return arrays, cases


def paw(zeta_tables) -> tuple[dict, list]:
    from mandacaru.pseudopotentials import get_paw
    arrays, cases = {}, []
    for symbol in ("O", "Fe"):
        pp = get_paw(symbol)
        r = np.asarray(pp.r)
        for l, channel in sorted(pp.channels.items()):
            values = np.asarray(channel.pseudo_waves[0], dtype=float)
            for scheme, tails in (("last_zeta", None), ("first_zeta", TAILS)):
                tables = zeta_tables(r, values, l + 1, l, 3, 0.15,
                                     tail_norms=None if tails is None else tuple(tails))
                key = f"paw_{symbol}_{l}_{scheme}"
                cases.append(dict(symbol=symbol, l=l, scheme=scheme, count_out=len(tables), key=key))
                for k, table in enumerate(tables[1:], start=1):
                    arrays[f"{key}_{k}"] = np.asarray(table.values)
    return arrays, cases


def hydrogen(zeta_tables) -> tuple[dict, dict]:
    from mandacaru.basis.multizeta import (DEFAULT_SPLIT_NORM, split_radius,
                                           split_valence_tail)
    from mandacaru.basis.nao import solve_confined_radial
    r, values, _energy = solve_confined_radial(1, 0, 1.0, 4.0, 800)
    arrays = {"r": r, "values": values}
    tables = zeta_tables(r, values, 1, 0, 4)
    for k, table in enumerate(tables):
        arrays[f"zeta{k + 1}"] = np.asarray(table.values)
    norms = list(HYDROGEN_SPLITS) + [DEFAULT_SPLIT_NORM]
    radii = [split_radius(r, values, norm) for norm in norms]
    arrays["tail_default"] = split_valence_tail(r, values, 0, radii[-1])
    return arrays, dict(split_norms=norms, split_radii=radii, count_out=len(tables))


def main() -> None:
    from freeze_stamp import stamp
    from mandacaru.basis.multizeta import zeta_tables
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {}
    arrays = {}
    got, meta["synthetic"] = synthetic(zeta_tables)
    arrays.update(got)
    got, meta["hydrogen_1s"] = hydrogen(zeta_tables)
    arrays.update({f"h1s_{k}": v for k, v in got.items()})
    paw_cases = []
    try:
        got, paw_cases = paw(zeta_tables)
        arrays.update(got)
    except (FileNotFoundError, ValueError) as error:
        sys.exit(f"the paw cases need MANDACARU_PAW_PATH: {error}")
    meta["paw"] = paw_cases
    library = Path(os.environ["MANDACARU_PAW_PATH"]) / "lda-sr"
    meta["stamp"] = stamp(extra=dict(script="freeze_basis_zetas.py", step=STEP, tails=list(TAILS)),
                          datasets=[library / f"{symbol}.parquet" for symbol in ("O", "Fe")])
    np.savez_compressed(OUT / "zeta_tables.npz", **arrays)
    (OUT / "zeta_tables.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(f"froze {len(arrays)} arrays: {len(meta['synthetic'])} synthetic cases, {len(paw_cases)} paw cases")


if __name__ == "__main__":
    main()
