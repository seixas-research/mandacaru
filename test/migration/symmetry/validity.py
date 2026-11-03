# -*- coding: utf-8 -*-
"""Oracle-free canonicalization, comparison, and validation routines for symmetry operations."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import numpy as np


def _fold(x):
    return x - np.round(x)


def ops_equal(A, B, cell, tol):
    """Operation sets modulo input-cell lattice translations, ignoring order.

    Returns (ok, message). Multisets: each (W,t) of A is matched to exactly one of B.
    tol is a Cartesian distance (Angstrom) between translation parts.
    """
    Ra, ta, Rb, tb = A["rotations"], A["translations"], B["rotations"], B["translations"]
    if len(Ra) != len(Rb):
        return False, f"n_ops {len(Ra)} vs {len(Rb)}"
    cell = np.asarray(cell, float)
    used = np.zeros(len(Rb), bool)
    bykey = {}
    for j, W in enumerate(Rb):
        bykey.setdefault(W.tobytes(), []).append(j)
    for i, W in enumerate(Ra):
        hit = None
        for j in bykey.get(W.tobytes(), []):
            if used[j]:
                continue
            d = _fold(ta[i] - tb[j]) @ cell
            if np.linalg.norm(d) <= tol:
                hit = j
                break
        if hit is None:
            return False, f"op {i} W={W.tolist()} t={ta[i].round(6).tolist()} unmatched"
        used[hit] = True
    return True, ""


def partition(eq):
    d = {}
    for i, r in enumerate(eq):
        d.setdefault(int(r), []).append(i)
    return sorted(tuple(v) for v in d.values())


def canon_digest(res):
    """Order-independent digest of the op set, t rounded to 1e-6 and folded to [0,1)."""
    items = sorted(
        (tuple(W.ravel().tolist()), tuple(np.round(np.mod(np.round(t, 6), 1.0), 6).tolist()))
        for W, t in zip(res["rotations"], res["translations"])
    )
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()[:16]


def validate_ops(cell, frac, labels, res, tol):
    """V1 integer det +-1, V2 metric preserved, V3 maps structure onto itself (same label) within tol,
    V4 closure mod lattice, V5 coset structure n_ops = |{W}| * n_pure_translations. Returns list of failures.
    """
    cell = np.asarray(cell, float)
    frac = np.asarray(frac, float)
    labels = np.asarray(labels)
    Rs, ts = res["rotations"], res["translations"]
    bad = []
    for W in Rs:
        if abs(round(np.linalg.det(W))) != 1:
            bad.append("det")
            break
    for W in Rs:
        Wc = cell.T @ W @ np.linalg.inv(cell.T)  # Cartesian map R = A W A^-1 (A = cell^T columns)
        if np.abs(Wc.T @ Wc - np.eye(3)).max() > 1e-6 + 10 * tol / np.linalg.norm(cell, axis=1).min():
            bad.append("metric")
            break
    n = len(frac)
    for W, t in zip(Rs, ts):
        img = frac @ W.T + t
        ok = True
        for lo in range(0, n, 64):  # chunked: memory bounded by 64 x N, not N x N
            d = _fold(img[lo : lo + 64, None, :] - frac[None, :, :]) @ cell
            dist = np.linalg.norm(d, axis=2)
            j = dist.argmin(axis=1)
            ok &= bool(
                (dist.min(axis=1) <= 2 * tol + 1e-9).all()
                and (labels[j] == labels[lo : lo + 64]).all()
            )
            if not ok:
                break
        if not ok:
            bad.append("not-a-symmetry")
            break
    # closure on a sample (all pairs if small)
    keys = {W.tobytes(): [] for W in Rs}
    for W, t in zip(Rs, ts):
        keys[W.tobytes()].append(t)
    m = len(Rs)
    pairs = (
        [(i, j) for i in range(m) for j in range(m)]
        if m <= 60
        else [(int(a), int(b)) for a, b in np.random.default_rng(0).integers(0, m, (3000, 2))]
    )
    for i, j in pairs:
        W = Rs[i] @ Rs[j]
        t = Rs[i] @ ts[j] + ts[i]
        ok = any(
            np.linalg.norm(_fold(t - u) @ cell) <= 4 * tol + 1e-6
            for u in keys.get(W.tobytes(), [])
        )
        if not ok:
            bad.append("closure")
            break
    # coset structure
    counts = [len(v) for v in keys.values()]
    if len(set(counts)) != 1:
        bad.append("cosets")
    return bad


def brute_ops(cell, frac, labels, tol=1e-5, max_atoms=96):
    """Enumerate lattice point-group ops from the Minkowski-reduced basis, then lattice translations from atom
    matching. Independent reference search. Returns dict(rotations, translations) in the INPUT basis.
    """
    from ase.geometry import minkowski_reduce

    cell = np.asarray(cell, float)
    frac = np.asarray(frac, float)
    labels = np.asarray(labels)
    if len(frac) > max_atoms:
        return None
    red, U = minkowski_reduce(cell)  # red = U @ cell, U unimodular
    U = np.rint(U).astype(int)
    rng_ = range(-3, 4)
    idx = np.array([(i, j, k) for i in rng_ for j in rng_ for k in rng_])
    vecs = idx @ red
    norms = np.linalg.norm(vecs, axis=1)
    cand = []
    for a in range(3):
        la = np.linalg.norm(red[a])
        cand.append(idx[np.abs(norms - la) < 1e-6 * max(1, la) + 1e-7])
    G = red @ red.T
    Ws = []
    for r0 in cand[0]:
        for r1 in cand[1]:
            for r2 in cand[2]:
                M = np.array([r0, r1, r2])  # rows: images of reduced basis vectors in reduced coords
                if abs(round(np.linalg.det(M))) != 1:
                    continue
                Mi = M @ red
                if np.abs(Mi @ Mi.T - G).max() < 1e-6 * G.max():
                    Ws.append(M)
    rots, trs = [], []
    # reference atom: rarest label
    lab, cnt = np.unique(labels, return_counts=True)
    ref = int(np.nonzero(labels == lab[cnt.argmin()])[0][0])
    for M in Ws:
        # input-basis fractional op: x_in' = W_in x_in ; x_red = U^-T x_in (columns) => W_in = U^T W_red U^-T
        Wred = M.T
        Win = np.rint(U.T @ Wred @ np.linalg.inv(U.T)).astype(int)
        if abs(round(np.linalg.det(Win))) != 1:
            continue
        img0 = Win @ frac[ref]
        for j in np.nonzero(labels == labels[ref])[0]:
            t = frac[j] - img0
            img = frac @ Win.T + t
            d = np.linalg.norm(_fold(img[:, None, :] - frac[None, :, :]) @ cell, axis=2)
            if (
                (d.min(axis=1) <= tol).all()
                and (labels[d.argmin(axis=1)] == labels).all()
                and len(set(d.argmin(axis=1).tolist())) == len(frac)
            ):
                rots.append(Win)
                trs.append(np.mod(t, 1.0))
    if not rots:
        return {"rotations": np.zeros((0, 3, 3), int), "translations": np.zeros((0, 3))}
    return {"rotations": np.array(rots, int), "translations": np.array(trs, float)}


def mp_mesh(size, gamma):
    """Mirror of spectrum::mesh (crates/mandacaru-core/src/periodic/spectrum.rs:166-176): Monkhorst-Pack,
    Gamma-centered for even n when gamma=True. Row-major (i,j,k).
    """
    g = np.meshgrid(*[np.arange(n) for n in size], indexing="ij")
    out = []
    for a, n in enumerate(size):
        x = (2.0 * g[a].ravel() + 1 - n) / (2.0 * n)
        if gamma and n % 2 == 0:
            x = x + 0.5 / n
        out.append(x)
    return np.stack(out, axis=1)


def keeping_mesh(rots, mesh):
    """Keep W with mesh@W a subset of mesh mod lattice."""
    keep = []
    for s, W in enumerate(rots):
        pts = mesh @ W.astype(float)
        d = pts[:, None, :] - mesh[None, :, :]
        d -= np.round(d)
        if np.all(np.any(np.all(np.abs(d) < 1e-8, axis=2), axis=1)):
            keep.append(s)
    return keep


def ibz_summary(mesh, rots, time_reversal=True):
    """(n_points, sorted weights) of the orbit decomposition of mesh under {W^T k} (+ -k)."""
    n = len(mesh)
    key = lambda k: tuple(np.mod(np.round(k * 1e8).astype(np.int64), 100000000).tolist())
    look = {key(p): i for i, p in enumerate(mesh)}
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, p in enumerate(mesh):
        imgs = [p @ W.astype(float) for W in rots]
        if time_reversal:
            imgs.append(-p)
        for q in imgs:
            j = look.get(key(q))
            if j is None:
                raise ValueError("mesh not closed")
            a, b = find(i), find(j)
            parent[max(a, b)] = min(a, b)
    c = Counter(find(i) for i in range(n))
    return len(c), sorted(c.values()), [find(i) for i in range(n)]


def w_digest(res):
    """Digest of the multiset of rotation parts (exact integers). With 'every returned (W,t) is a valid symmetry
    of the input' (validate_ops) and 'all (W,t) distinct mod lattice', this fixes the op set completely: for each
    W the valid translations are one coset of the pure-translation group, whose size is the multiplicity.
    """
    c = Counter(tuple(W.ravel().tolist()) for W in res["rotations"])
    return hashlib.sha256(json.dumps(sorted(c.items())).encode()).hexdigest()[:16]


def input_hash(cell, frac, labels):
    f = np.mod(np.round(np.mod(np.asarray(frac, float), 1.0), 9), 1.0)
    return hashlib.sha256(
        json.dumps(
            [
                np.round(np.asarray(cell, float), 9).tolist(),
                f.tolist(),
                np.asarray(labels).tolist(),
            ]
        ).encode()
    ).hexdigest()[:16]
