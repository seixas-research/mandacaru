# -*- coding: utf-8 -*-
"""compare(case, ref, cand) -> list of failure strings. `cand`/`ref` are dicts as returned by reference oracle
(keys rotations, translations, number, international, pointgroup, equivalent_atoms, optional).
"""

from __future__ import annotations

import numpy as np

from .validity import ops_equal, partition, validate_ops


def compare(case, ref, cand, symprec, exact=True, check_meta=True, check_equiv=True):
    fails = []
    if "error" in ref or "error" in cand:
        if ("error" in ref) != ("error" in cand):
            fails.append(f"error mismatch ref={ref.get('error')} cand={cand.get('error')}")
        return fails
    tol = 1e-6 if exact else symprec  # Angstrom, between translation parts
    ok, msg = ops_equal(ref, cand, case["cell"], tol)
    if not ok:
        fails.append("ops: " + msg)
    if check_meta:
        for k in ("number", "international", "pointgroup"):
            if k in cand and ref[k] != cand[k]:
                fails.append(f"{k}: {ref[k]} vs {cand[k]}")
    if check_equiv and "equivalent_atoms" in cand and partition(ref["equivalent_atoms"]) != partition(cand["equivalent_atoms"]):
        fails.append("equivalent_atoms partition differs")
    bad = validate_ops(case["cell"], case["frac"], case["labels"], cand, symprec)  # oracle-free
    fails += ["invalid:" + b for b in bad]
    return fails


# ----- mutants (must be CAUGHT) and equivalences (must PASS)
def _copy(r):
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in r.items()}


def m_drop_op(r):
    c = _copy(r)
    c["rotations"] = c["rotations"][:-1]
    c["translations"] = c["translations"][:-1]
    return c


def m_shift_t(r):
    c = _copy(r)
    c["translations"][len(c["translations"]) // 2, 0] += 0.01
    return c


def m_wrong_number(r):
    c = _copy(r)
    c["number"] += 1
    return c


def m_flip_sign(r):
    c = _copy(r)
    i = len(c["rotations"]) // 2
    if c["rotations"][i].any():
        c["rotations"][i][0, 1] += 1
    return c


def m_dup_op(r):
    c = _copy(r)
    c["rotations"] = np.concatenate([c["rotations"][:-1], c["rotations"][:1]])
    c["translations"] = np.concatenate([c["translations"][:-1], c["translations"][:1]])
    return c


def m_bad_equiv(r):
    c = _copy(r)
    eq = c["equivalent_atoms"]
    if len(eq) > 1 and len(set(eq.tolist())) > 1:
        c["equivalent_atoms"] = np.zeros_like(eq)
    return c


MUTANTS = dict(
    drop_op=m_drop_op,
    shift_t=m_shift_t,
    wrong_number=m_wrong_number,
    bad_rotation=m_flip_sign,
    dup_op=m_dup_op,
    bad_equiv=m_bad_equiv,
)


def e_reorder(r, seed=0):
    c = _copy(r)
    p = np.random.default_rng(seed).permutation(len(c["rotations"]))
    c["rotations"] = c["rotations"][p]
    c["translations"] = c["translations"][p]
    return c


def e_rewrap(r):
    c = _copy(r)
    c["translations"] = c["translations"] + np.random.default_rng(1).integers(
        -2, 3, c["translations"].shape
    )
    return c


def e_relabel_equiv(r):  # a different representative convention (highest index) is the same partition
    c = _copy(r)
    eq = c["equivalent_atoms"]
    out = eq.copy()
    for rep in set(eq.tolist()):
        idx = np.nonzero(eq == rep)[0]
        out[idx] = idx.max()
    c["equivalent_atoms"] = out
    return c


EQUIV = dict(reorder=e_reorder, rewrap_translations=e_rewrap, relabel_equivalent_atoms=e_relabel_equiv)
