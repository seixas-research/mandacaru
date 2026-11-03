# -*- coding: utf-8 -*-
"""SYM-8c gate: Space group number and symbol identification parity tests across all settings."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import warnings

from ase.spacegroup import Spacegroup, crystal
import numpy as np
import pytest

from mandacaru.core.symmetry import crystal_symmetry

try:
    from .symmetry.corpus import SEED, cellpar, general_site, settings, too_close
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from symmetry.corpus import SEED, cellpar, general_site, settings, too_close


@pytest.fixture(scope="module")
def corpus_tier1():
    ref_path = Path(__file__).parent / "reference" / "space_group_corpus.json.gz"
    with gzip.open(ref_path, "rt", encoding="utf-8") as f:
        return json.load(f)["tier1"]


def _build_general_pair(number: int, setting: int, seed: int = SEED):
    rng0 = np.random.default_rng([seed, number, setting])
    sg = Spacegroup(number, setting)
    assert sg.no == number
    cp = cellpar(number, setting, rng0)
    base = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _attempt in range(30):
            g1, g2 = general_site(rng0), general_site(rng0)
            try:
                a = crystal(
                    ["Na", "Cl"],
                    [g1, g2],
                    spacegroup=sg,
                    cellpar=cp,
                    onduplicates="replace",
                    symprec=1e-6,
                )
            except Exception:
                continue
            if not too_close(a):
                base = a
                break
        if base is None:
            return None
        prim = crystal(
            ["Na", "Cl"],
            [g1, g2],
            spacegroup=sg,
            cellpar=cp,
            primitive_cell=True,
            onduplicates="replace",
            symprec=1e-6,
        )
    return base, prim


@pytest.mark.parametrize("number,setting", settings())
def test_space_group_number_and_symbol(number: int, setting: int, corpus_tier1: dict):
    pair = _build_general_pair(number, setting)
    if pair is None:
        pytest.skip(f"No collision-free general site in corpus for sg{number}.{setting}")

    base, prim = pair
    cases = [("conv", base), ("prim", prim)]

    for variant, at in cases:
        cid = f"sg{number}.{setting}/gen/{variant}"
        assert cid in corpus_tier1, f"Missing reference record for {cid}"
        rec = corpus_tier1[cid]

        info = crystal_symmetry(at)

        assert info.number == number, (
            f"{cid}: expected number {number}, got {info.number}"
        )
        assert info.international == rec["s"], (
            f"{cid}: expected symbol {rec['s']}, got {info.international}"
        )
        assert info.point_group == rec["pg"], (
            f"{cid}: expected point group {rec['pg']}, got {info.point_group}"
        )
        assert info.n_operations == rec["nops"], (
            f"{cid}: expected {rec['nops']} ops, got {info.n_operations}"
        )
