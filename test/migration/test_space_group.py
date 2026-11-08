# -*- coding: utf-8 -*-
"""SYM-6 gate: Parquet and symmetry migration test suite against frozen reference records."""

from __future__ import annotations

import ast
import gzip
import json
from pathlib import Path
import time

from ase import Atoms
import numpy as np
import pytest

from .space_group import crystal_symmetry

try:
    from .symmetry.compare import EQUIV, MUTANTS, compare
    from .symmetry.corpus_real import big_supercell
    from .symmetry.validity import (
        input_hash,
        keeping_mesh,
        mp_mesh,
        ops_equal,
        validate_ops,
        w_digest,
    )
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from symmetry.compare import EQUIV, MUTANTS, compare
    from symmetry.corpus_real import big_supercell
    from symmetry.validity import (
        input_hash,
        keeping_mesh,
        mp_mesh,
        ops_equal,
        validate_ops,
        w_digest,
    )


@pytest.fixture(scope="module")
def space_group_data():
    json_path = Path(__file__).parent / "reference" / "space_group.json"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def corpus_tier1():
    gz_path = Path(__file__).parent / "reference" / "space_group_corpus.json.gz"
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        return json.load(f)["tier1"]


def test_reference_files_exist_and_counts():
    """Verify that reference files exist and contain the required record counts."""
    ref_dir = Path(__file__).parent / "reference"
    sg_json = ref_dir / "space_group.json"
    sg_gz = ref_dir / "space_group_corpus.json.gz"

    assert sg_json.is_file(), "Missing test/migration/reference/space_group.json"
    assert sg_gz.is_file(), "Missing test/migration/reference/space_group_corpus.json.gz"

    with open(sg_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["tier2"]) == 124, f"Expected 124 tier2 records, got {len(data['tier2'])}"
    assert len(data["tier3_scale"]) == 3, f"Expected 3 tier3_scale records, got {len(data['tier3_scale'])}"
    assert len(data["kpts"]) == 255, f"Expected 255 kpts records, got {len(data['kpts'])}"

    with gzip.open(sg_gz, "rt", encoding="utf-8") as f:
        corpus = json.load(f)
    assert len(corpus["tier1"]) == 4211, f"Expected 4211 tier1 records, got {len(corpus['tier1'])}"


def _tier2_ids():
    json_path = Path(__file__).parent / "reference" / "space_group.json"
    if not json_path.is_file():
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        return sorted(json.load(f).get("tier2", {}).keys())


@pytest.mark.parametrize("cid", _tier2_ids())
def test_tier2_record(cid: str, space_group_data: dict):
    """Test all 124 tier2 records against crystal_symmetry."""
    rec = space_group_data["tier2"][cid]

    if "error" in rec:
        # Singular / degenerate cell case
        at = Atoms(rec["labels"], positions=rec["frac"], cell=rec["cell"], pbc=True)
        with pytest.raises(Exception) as excinfo:
            crystal_symmetry(at)
        assert "failed" in str(excinfo.value).lower(), (
            f"Expected 'failed' in exception message, got: {excinfo.value}"
        )
        return

    at = Atoms(rec["labels"], cell=rec["cell"], scaled_positions=rec["frac"], pbc=True)
    info = crystal_symmetry(at, symprec=rec["symprec"])

    cand = {
        "rotations": info.rotations,
        "translations": info.translations,
        "number": info.number,
        "international": info.international,
        "pointgroup": info.point_group,
    }
    ref_cand = {
        "rotations": np.array(rec["rotations"], int),
        "translations": np.array(rec["translations"], float),
        "number": rec["number"],
        "international": rec["international"],
        "pointgroup": rec["pointgroup"],
    }

    ok, msg = ops_equal(ref_cand, cand, rec["cell"], tol=rec["symprec"])
    assert ok, f"{cid} ops_equal failed: {msg}"

    bad = validate_ops(rec["cell"], rec["frac"], rec["labels"], cand, tol=rec["symprec"])
    assert not bad, f"{cid} validate_ops failed: {bad}"

    assert info.number == rec["number"], (
        f"{cid} number {info.number} != {rec['number']}"
    )
    assert info.international == rec["international"], (
        f"{cid} symbol {info.international} != {rec['international']}"
    )
    assert info.point_group == rec["pointgroup"], (
        f"{cid} point group {info.point_group} != {rec['pointgroup']}"
    )


def test_kpts_mesh(space_group_data: dict):
    """Test the 255 kpts combinations against native Context.kmesh."""
    from mandacaru._native import Context

    ctx = Context(threads=1)
    kpts = space_group_data["kpts"]
    tier2 = space_group_data["tier2"]

    for k, ref in kpts.items():
        cid, size_str, gamma_str = k.split("|")
        size = ast.literal_eval(size_str)
        gamma = bool(int(gamma_str))

        rots = np.array(tier2[cid]["rotations"], int)
        mesh = mp_mesh(size, gamma)
        keep = keeping_mesh(rots, mesh)

        assert len(keep) == ref["kept"], (
            f"{k}: kept ops count {len(keep)} != {ref['kept']}"
        )

        kept_rots = rots[keep]
        res = ctx.kmesh(
            size,
            gamma=gamma,
            rotations=[list(map(int, W.ravel())) for W in kept_rots],
            time_reversal=True,
        )

        n_irr = len(res["points"])
        assert n_irr == ref["n_irr"], f"{k}: n_irr {n_irr} != {ref['n_irr']}"

        N = int(np.prod(size))
        int_weights = sorted(int(round(w * N)) for w in res["weights"])
        assert int_weights == ref["weights"], f"{k}: weights {int_weights} != {ref['weights']}"


@pytest.mark.parametrize("case", big_supercell(), ids=lambda c: c["id"])
def test_tier3_scale_supercell(case: dict, space_group_data: dict):
    """Test 512-atom supercell scale cases (< 1.5 s, matching w_digest and operations count)."""
    cid = case["id"]
    ref = space_group_data["tier3_scale"][cid]
    at = Atoms(case["labels"], cell=case["cell"], scaled_positions=case["frac"], pbc=True)

    t0 = time.time()
    info = crystal_symmetry(at)
    dt = time.time() - t0

    assert dt < 1.5, f"{cid} took {dt:.3f} s (budget < 1.5 s)"

    cand = {"rotations": info.rotations}
    wd = w_digest(cand)

    assert info.n_operations == ref["nops"], (
        f"{cid}: expected {ref['nops']} ops, got {info.n_operations}"
    )
    assert wd == ref["w"], f"{cid}: expected w_digest {ref['w']}, got {wd}"
    assert info.number == ref["n"], f"{cid}: expected number {ref['n']}, got {info.number}"
    assert info.international == ref["s"], (
        f"{cid}: expected symbol {ref['s']}, got {info.international}"
    )


def test_compare_harness_mutants_and_equivalences(space_group_data: dict):
    """Verify that compare.py accepts EQUIV forms and catches MUTANTS on all tier2 cases."""
    tier2 = space_group_data["tier2"]

    for cid, rec in tier2.items():
        if "error" in rec:
            continue
        case = {"cell": rec["cell"], "frac": rec["frac"], "labels": rec["labels"]}
        ref = {
            "rotations": np.array(rec["rotations"], int),
            "translations": np.array(rec["translations"], float),
            "number": rec["number"],
            "international": rec["international"],
            "pointgroup": rec["pointgroup"],
            "equivalent_atoms": np.array(rec["equivalent_atoms"], int),
        }

        # 1. Identity must pass
        f = compare(case, ref, ref, rec["symprec"])
        assert not f, f"Identity failed on {cid}: {f}"

        # 2. Equivalences must pass
        for name, e in EQUIV.items():
            f = compare(case, ref, e(ref), rec["symprec"])
            assert not f, f"Equivalence {name} failed on {cid}: {f}"

        # 3. Mutants must be caught
        for name, m in MUTANTS.items():
            cand = m(ref)
            # Skip degenerate non-mutations
            if name == "dup_op" and len(ref["rotations"]) <= 1:
                continue
            if name == "bad_equiv" and len(set(ref["equivalent_atoms"].tolist())) <= 1:
                continue
            if name == "bad_rotation" and not ref["rotations"][len(ref["rotations"]) // 2].any():
                continue

            f = compare(case, ref, cand, rec["symprec"])
            assert f, f"Mutant {name} was missed on {cid}!"


@pytest.mark.slow
def test_tier1_full_corpus_sweep(corpus_tier1: dict):
    """Full 4,211-structure tier1 sweep test."""
    try:
        from .symmetry.corpus import space_group_cases
        from .symmetry.corpus_real import pinned, real_systems
    except ImportError:
        from symmetry.corpus import space_group_cases
        from symmetry.corpus_real import pinned, real_systems

    checked = 0
    for c in space_group_cases(extra_cells=True):
        cid = c["id"]
        assert cid in corpus_tier1, f"Missing tier1 reference record for {cid}"
        rec = corpus_tier1[cid]

        h = input_hash(c["cell"], c["frac"], c["labels"])
        assert h == rec["h"], f"{cid}: input_hash {h} != {rec['h']}"

        at = Atoms(c["labels"], cell=c["cell"], scaled_positions=c["frac"], pbc=True)
        info = crystal_symmetry(at, symprec=1e-5)

        cand = {
            "rotations": info.rotations,
            "translations": info.translations,
        }
        wd = w_digest(cand)
        assert wd == rec["w"], f"{cid}: w_digest {wd} != {rec['w']}"
        assert info.n_operations == rec["nops"], f"{cid}: n_ops {info.n_operations} != {rec['nops']}"

        bad = validate_ops(c["cell"], c["frac"], c["labels"], cand, tol=1e-5)
        assert not bad, f"{cid}: validate_ops failed: {bad}"

        assert info.number == rec["n"], f"{cid}: number {info.number} != {rec['n']}"
        assert info.international == rec["s"], f"{cid}: symbol {info.international} != {rec['s']}"
        checked += 1

    for c in pinned() + real_systems():
        if len(c["frac"]) > 16:
            cid = c["id"]
            assert cid in corpus_tier1, f"Missing tier1 reference record for {cid}"
            rec = corpus_tier1[cid]

            h = input_hash(c["cell"], c["frac"], c["labels"])
            assert h == rec["h"], f"{cid}: input_hash {h} != {rec['h']}"

            at = Atoms(c["labels"], cell=c["cell"], scaled_positions=c["frac"], pbc=True)
            info = crystal_symmetry(at, symprec=1e-5)
            cand = {
                "rotations": info.rotations,
                "translations": info.translations,
            }
            wd = w_digest(cand)
            assert wd == rec["w"], f"{cid}: w_digest {wd} != {rec['w']}"
            assert info.n_operations == rec["nops"], f"{cid}: n_ops {info.n_operations} != {rec['nops']}"
            bad = validate_ops(c["cell"], c["frac"], c["labels"], cand, tol=1e-5)
            assert not bad, f"{cid}: validate_ops failed: {bad}"

            assert info.number == rec["n"], f"{cid}: number {info.number} != {rec['n']}"
            assert info.international == rec["s"], f"{cid}: symbol {info.international} != {rec['s']}"
            checked += 1

    assert checked == 4211, f"Expected 4211 checked structures, got {checked}"
