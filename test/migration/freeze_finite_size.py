#!/usr/bin/env python
"""Freeze the OLD finite-size references (KZK, Wigner-Seitz kernel, MPC, CCMH) before the Rust port (plan FS-0).

What is frozen, under ``test/migration/reference/finite_size/`` (units: Hartree, Bohr, per supercell unless a
key says otherwise; ``*.npz`` are git-ignored, every number that decides pass or fail is in the tracked JSON):

* ``stamps.json``       sha256 of the old sources and of the extension, git revisions, versions.
* ``test_counts.json``  collected items, functions and outcomes of ``test/core/test_{mpc,kzk,ccmh}.py`` with only
                        the two dangling imports stubbed.
* ``kzk_table.json``    Table I constants, A0, branch boundaries, ``fs_exchange``/``fs_correlation``/xc on a grid of
                        (r_s, L) incl. the continuity points; own re-derivation (4x4 solve for the cubic, own
                        Perdew-Zunger) beside it.
* ``kzk_density.json``  ``kzk_correction`` on synthetic densities (uniform, Gaussian blob, two-site, negative/NaN/inf
                        silent-drop cases, non-cubic cells, ``n_cells``).
* ``kzk_heg.json``      Gamma-point HF exchange of the unpolarized gas vs ``fs_exchange`` (independent diagnostic).
* ``voxel.json``        ``voxel_self_potential`` (both code paths) vs the analytic cube value and an independent
                        face-quadrature of the divergence-theorem formula.
* ``madelung.json``     v_M for sc/fcc/bcc: own Ewald at two splitting parameters and the native ``Context.ewald``.
* ``ws_kernel.json/.npz`` the Wigner-Seitz minimum-image kernel: sums vs the analytic/independent reference, h-error
                        and Richardson order, d = 0 node, samples, digests.
* ``mpc_n1.json``       N = 1 Gaussian: the OLD ``exchange_correlation_hole_energy`` beside the CORRECTED expected
                        value (carries the 1/2 N v_M double count; NOT a Rust gate), contraction parity case.
* ``ccmh_synth.json``   synthetic Gaussian-hole structure factor (OLD per-electron number beside the corrected
                        N_e-times value) and the N = 1 full-vs-connected structure factor.
* ``ccmh_index.json``   the Gamma_prsq vs Gamma_prqs question settled by a 2-electron operator-algebra example.
* ``runs.json``         wall time and peak memory of every child.

``run`` executes every stage in its own child process (one at a time, guarded by ``scripts/memguard.py``);
``verify`` replays the cheap cases and compares them bitwise to the stored JSON (``--full`` also replays the
n = 96 kernels).  The OLD modules are not imported from the working tree: the script pulls them with
``git show`` from pinned revisions into a temporary package, so ``verify`` keeps working after
``core/{kzk,mpc,ccmh}.py`` are deleted from the tree (it needs the git history and, for ``lda_correlation``, the
extension).  Once the Rust port exists the Rust tests read the JSON and this script is only the provenance.

    python test/migration/freeze_finite_size.py run
    python test/migration/freeze_finite_size.py verify [--full]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("FS_FREEZE_ROOT", Path(__file__).resolve().parents[2]))
REF = Path(os.environ.get("FS_FREEZE_REF", ROOT / "test" / "migration" / "reference" / "finite_size"))
MEMGUARD = Path(os.environ.get("FS_FREEZE_MEMGUARD", ROOT / "scripts" / "memguard.py"))
SCRIPT = "test/migration/freeze_finite_size.py"

#: Revision that holds core/{kzk,mpc,ccmh}.py, units.py and basis/atomic_solver.py (HEAD when frozen, v26.10.44).
SRC_REV = "89052afdd0e193a0f96dd01025e1f5c09a511593"
#: Last revision that holds integrals/{poisson,fft}.py (deleted by 241d688); the parent of that commit.
DEP_REV = "a0fd3b5c4c8ecb46ff61efe00ba7a04a8a621690"
SRC_FILES = {"core/kzk.py": SRC_REV, "core/mpc.py": SRC_REV, "core/ccmh.py": SRC_REV, "units.py": SRC_REV,
             "basis/atomic_solver.py": SRC_REV, "integrals/poisson.py": DEP_REV, "integrals/fft.py": DEP_REV}
#: Old test files whose collection is recorded.
OLD_TESTS = ("test/core/test_mpc.py", "test/core/test_kzk.py", "test/core/test_ccmh.py")

#: 3 ln(2 + sqrt 3) - pi/2: the integral of 1/r over the unit cube centered on the origin.
CUBE_CONSTANT = 3.0 * math.log(2.0 + math.sqrt(3.0)) - 0.5 * math.pi
#: The simple-cubic Madelung constant L*v_M of the Wigner lattice (literature, 10 digits).
SC_LVM_LITERATURE = -2.837297479


# ----------------------------------------------------------------------------------------- generic helpers
def enc(x):
    """JSON-safe copy: numpy to python, non-finite floats to strings (so the file is strict JSON)."""
    if isinstance(x, dict):
        return {str(k): enc(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [enc(v) for v in x]
    if isinstance(x, np.ndarray):
        return enc(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return x
    if isinstance(x, complex) or isinstance(x, np.complexfloating):
        return {"re": enc(x.real), "im": enc(x.imag)}
    return x


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def sampled(a, k=16):
    """``k`` evenly spread flat entries of an array: index and value (fingerprint beside the digest)."""
    flat = np.asarray(a).ravel()
    idx = np.unique(np.linspace(0, flat.size - 1, k).astype(int))
    return {"index": idx.tolist(), "value": enc(flat[idx])}


def fp(a, k=16):
    a = np.asarray(a, dtype=float)
    return {"sha256": sha(a), "shape": list(a.shape), "sum": float(a.sum()), "max": float(np.max(a)),
            "min": float(np.min(a)), "samples": sampled(a, k)}


def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout


def extension_path() -> Path:
    found = sorted((ROOT / "src" / "mandacaru").glob("_native.cpython-*-darwin.so")) + \
        sorted((ROOT / "src" / "mandacaru").glob("_native*.so"))
    return found[0]


_OLD = None


def old():
    """The OLD modules, materialized from git into a temporary package ``mandacaru_old`` (once per process)."""
    global _OLD
    if _OLD is not None:
        return _OLD
    tmp = Path(tempfile.mkdtemp(prefix="fs0_old_"))
    pkg = tmp / "mandacaru_old"
    for rel, rev in SRC_FILES.items():
        text = git("show", f"{rev}:src/mandacaru/{rel}")
        path = pkg / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    for d in ("", "core", "integrals", "basis"):
        (pkg / d / "__init__.py").write_text("")
    sys.path.insert(0, str(tmp))
    import mandacaru._native as native      # the only compiled dependency: lda_terms, units constants, ewald

    top = importlib.import_module("mandacaru_old")
    top._native = native
    sys.modules["mandacaru_old._native"] = native
    mods = {n: importlib.import_module(f"mandacaru_old.{m}") for n, m in
            (("kzk", "core.kzk"), ("mpc", "core.mpc"), ("ccmh", "core.ccmh"), ("poisson", "integrals.poisson"),
                       ("atomic", "basis.atomic_solver"))}
    _OLD = type("Old", (), mods)
    return _OLD


# ------------------------------------------------------------------------- independent re-derivations
PZ = dict(A=0.0311, B=-0.048, C=0.0020, D=-0.0116, g=-0.1423, b1=1.0529, b2=0.3334)


def own_pz(rs, modified=False):
    """Perdew-Zunger (1981) correlation per electron in Hartree and its r_s derivative.

    ``modified=False``: the literature constants (A, B, C, D) below r_s = 1 (value jumps by 3.4e-5 Ha at r_s = 1).
    ``modified=True``: A and B kept, C and D re-derived so that value and slope are continuous at r_s = 1 -- what the
    extension's ``lda_terms`` evaluates (read in ``crates/mandacaru-core/src/models/xc.rs::pz``)."""
    rs = np.asarray(rs, dtype=float)
    lo = rs < 1.0
    C, D = PZ["C"], PZ["D"]
    if modified:
        den1 = 1.0 + PZ["b1"] + PZ["b2"]
        D = PZ["g"] / den1 - PZ["B"]
        C = -PZ["g"] * (PZ["b1"] / 2.0 + PZ["b2"]) / den1 ** 2 - PZ["A"] - D
    ec = np.where(lo, PZ["A"] * np.log(rs) + PZ["B"] + C * rs * np.log(rs) + D * rs,
                  PZ["g"] / (1.0 + PZ["b1"] * np.sqrt(rs) + PZ["b2"] * rs))
    den = 1.0 + PZ["b1"] * np.sqrt(rs) + PZ["b2"] * rs
    dec = np.where(lo, PZ["A"] / rs + C * (np.log(rs) + 1.0) + D,
                   -PZ["g"] * (0.5 * PZ["b1"] / np.sqrt(rs) + PZ["b2"]) / den ** 2)
    return ec, dec


def own_kzk(rs, L, consts):
    """Eq. (3) and (6) coded from the docstring equations; the cubic from a 4x4 linear solve (not the closed form).

    Returns (e_x, e_c, e_xc_inf) in Hartree for an array of r_s.  ``consts`` = (A1, A2, A3, G1, G2, G3, G4)."""
    A1, A2, A3, G1, G2, G3, G4 = consts
    rs = np.asarray(rs, dtype=float)
    a0 = -0.75 * (9.0 / (4.0 * np.pi ** 2)) ** (1.0 / 3.0) * 2.0           # Rydberg
    rs_of = lambda n: L * (3.0 / (4.0 * np.pi * n)) ** (1.0 / 3.0)
    gam, gh, gl = rs_of(2), rs_of(12), rs_of(0.5)

    def high(r):
        ec, dec = own_pz(r, modified=True)
        g = G1 * r * np.log(r) + G2 * r + G3 * r ** 1.5 + G4 * r ** 2
        dg = G1 * (np.log(r) + 1) + G2 + 1.5 * G3 * np.sqrt(r) + 2 * G4 * r
        return 2 * ec - A1 / L ** 2 * r + g / L ** 3, 2 * dec - A1 / L ** 2 + dg / L ** 3

    vh, sh = high(np.array(gh))
    M = np.array([[1, gh, gh ** 2, gh ** 3], [0, 1, 2 * gh, 3 * gh ** 2],
                  [1, gl, gl ** 2, gl ** 3], [0, 1, 2 * gl, 3 * gl ** 2]], dtype=float)
    coef = np.linalg.solve(M, np.array([float(vh), float(sh), 0.0, 0.0]))
    ex = np.where(rs <= gam, a0 / rs + A1 / L ** 2 * rs + A2 / L ** 3 * rs ** 2, A3 * L ** 5 / rs ** 6) * 0.5
    mid = coef[0] + coef[1] * rs + coef[2] * rs ** 2 + coef[3] * rs ** 3
    with np.errstate(all="ignore"):
        hv, _ = high(rs)
    ec = np.where(rs <= gh, hv, np.where(rs <= gl, mid, 0.0)) * 0.5
    return ex, ec, (a0 / rs + 2 * own_pz(rs, modified=True)[0]) * 0.5


def lattices(L):
    """Primitive vectors as rows (Bohr), conventional/cubic side ``L`` where it means something."""
    s3 = math.sqrt(3.0)
    return {
        "sc": L * np.eye(3),
        "fcc": 0.5 * L * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float),
        "bcc": 0.5 * L * np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=float),
        "hex": L * np.array([[1, 0, 0], [-0.5, 0.5 * s3, 0], [0, 0, 1.6]], dtype=float),
        "triclinic": L * np.array([[1, 0, 0], [0.3, 1.1, 0], [0.2, 0.25, 0.9]], dtype=float),
    }


def ws_cell_integral(cell):
    """``int_WS d^3r/r`` of the Wigner-Seitz cell of the lattice with rows ``cell`` (independent of the kernel code).

    Voronoi cell of the origin from scipy; divergence theorem ``1/2 sum_f h_f int_f dS/r`` with smooth Gauss-Legendre
    quadrature on every triangle of every face."""
    from scipy.spatial import Voronoi

    span = range(-3, 4)
    pts = np.array([i * cell[0] + j * cell[1] + k * cell[2] for i in span for j in span for k in span])
    zero = int(np.argmin(np.linalg.norm(pts, axis=1)))
    vor = Voronoi(pts)
    xg, wg = np.polynomial.legendre.leggauss(24)
    xg, wg = 0.5 * (xg + 1), 0.5 * wg
    total = 0.0
    for (a, b), ridge in zip(vor.ridge_points, vor.ridge_vertices):
        if zero not in (a, b) or -1 in ridge:
            continue
        v = vor.vertices[ridge]
        c = v.mean(axis=0)
        normal = pts[b if a == zero else a]
        normal = normal / np.linalg.norm(normal)
        height = abs(float(normal @ c))
        e1 = v[0] - c
        e1 = e1 - (e1 @ normal) * normal
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(normal, e1)
        order = np.argsort(np.arctan2((v - c) @ e2, (v - c) @ e1))
        v = v[order]
        for i in range(1, len(v) - 1):
            p0, p1, p2 = v[0], v[i], v[i + 1]
            area2 = np.linalg.norm(np.cross(p1 - p0, p2 - p0))
            # collapsed (Duffy) coordinates on the triangle: x = p0 + u (p1 - p0) + u v (p2 - p1)
            u, w = np.meshgrid(xg, xg, indexing="ij")
            wuv = (wg[:, None] * wg[None, :]) * u
            x = p0 + u[..., None] * (p1 - p0) + (u * w)[..., None] * (p2 - p1)
            total += 0.5 * height * area2 * float(np.sum(wuv / np.linalg.norm(x, axis=-1)))
    return float(total)


# ------------------------------------------------------------------------------------------------ stages
def stage_stamps():
    ext = extension_path()
    files = {}
    for rel, rev in SRC_FILES.items():
        text = git("show", f"{rev}:src/mandacaru/{rel}")
        files[rel] = {"rev": rev, "sha256": hashlib.sha256(text.encode()).hexdigest(), "lines": text.count("\n")}
    tree = {}
    for rel in ("core/kzk.py", "core/mpc.py", "core/ccmh.py", "basis/atomic_solver.py"):
        p = ROOT / "src" / "mandacaru" / rel
        tree[rel] = sha_file(p) if p.exists() else None
    tests = {}
    for rel in OLD_TESTS:
        p = ROOT / rel
        tests[rel] = sha_file(p) if p.exists() else None
    newest = 0.0
    for dp, dn, fn in os.walk(ROOT / "crates"):
        dn[:] = [d for d in dn if d != "target"]
        newest = max([newest] + [os.stat(os.path.join(dp, f)).st_mtime for f in fn])
    import scipy
    return {
        "note": "old sources are the git blobs below (read-only git show); working-tree copies listed for reference",
        "git_head": git("rev-parse", "HEAD").strip(), "src_rev": SRC_REV, "dep_rev": DEP_REV,
        "old_sources": files, "working_tree_sha256": tree, "old_tests_sha256": tests,
        "extension": {"path": ext.name, "sha256": sha_file(ext), "bytes": ext.stat().st_size,
                      "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ext.stat().st_mtime)),
                      "newer_than_every_crate_file": bool(ext.stat().st_mtime >= newest)},
        "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(), "script": SCRIPT,
    }


def _stub_and_run(args):
    """Run pytest in-process with ONLY the two dangling imports stubbed; return (items, outcomes)."""
    import types
    import mandacaru.integrals as integrals_pkg
    from importlib.util import module_from_spec, spec_from_file_location

    O = old()
    tmp = Path(O.poisson.__file__).parent
    fft = module_from_spec(spec_from_file_location("mandacaru.integrals.fft", tmp / "fft.py"))
    sys.modules["mandacaru.integrals.fft"] = fft
    fft.__spec__.loader.exec_module(fft)
    sys.modules["mandacaru.integrals.poisson"] = O.poisson
    integrals_pkg.poisson = O.poisson
    alg = types.ModuleType("mandacaru.algorithms")
    alg.__path__ = []
    pf = types.ModuleType("mandacaru.algorithms.pseudo_forces")

    def spatial_rdms(gamma, gamma2, n_orbitals):                         # verbatim from 241d688^
        M = int(n_orbitals)
        gamma = np.asarray(gamma, dtype=complex)
        gamma2 = np.asarray(gamma2, dtype=complex)
        if gamma.shape != (2 * M, 2 * M):
            raise ValueError("expected a 2Mx2M one-body RDM")
        a, b = slice(0, M), slice(M, 2 * M)
        return (gamma[a, a] + gamma[b, b],
                gamma2[a, a, a, a] + gamma2[a, b, a, b] + gamma2[b, a, b, a] + gamma2[b, b, b, b])

    pf.spatial_rdms = spatial_rdms
    alg.pseudo_forces = pf
    sys.modules["mandacaru.algorithms"] = alg
    sys.modules["mandacaru.algorithms.pseudo_forces"] = pf
    import pytest

    outcomes, items = {}, []

    class Plugin:
        def pytest_collection_finish(self, session):
            items.extend(i.nodeid for i in session.items)

        def pytest_runtest_logreport(self, report):
            if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
                key = report.nodeid
                kind = report.outcome if report.when == "call" else ("error" if report.outcome == "failed" else "skipped")
                outcomes[key] = {"outcome": kind, "reason": (str(report.longrepr).strip().splitlines() or [""])[-1][:160]}

    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        pytest.main(args + ["-p", "no:cacheprovider", "-q", "--no-header", "-o", "addopts=", "--tb=line"],
                    plugins=[Plugin()])
    return items, outcomes


def stage_counts():
    out = {"note": "pytest with ONLY mandacaru.algorithms.pseudo_forces.spatial_rdms and mandacaru.integrals.{poisson,fft} "
                   "stubbed (restored 241d688^ text); every other failure is natural", "files": {}}
    for rel in OLD_TESTS:
        path = ROOT / rel
        if not path.exists():
            out["files"][rel] = "absent"
            continue
        import ast
        tree = ast.parse(path.read_text())
        funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
        items, outcomes = _stub_and_run([str(path)])
        # a test function that errors in setup and passes nothing is still one item
        tally = {}
        for o in outcomes.values():
            tally[o["outcome"]] = tally.get(o["outcome"], 0) + 1
        out["files"][rel] = {
            "functions": len(funcs), "collected_items": len(items), "outcomes": tally,
            "not_passed": {k.split("::", 1)[1]: v for k, v in outcomes.items() if v["outcome"] != "passed"}}
    return out


def stage_kzk_table(heavy=False):
    O = old()
    kzk = O.kzk
    res = {"units": "Hartree per electron except where 'Ry' is said; r_s and L in Bohr"}
    res["constants"] = {
        "A1_Ry": kzk.A1, "A2_Ry": kzk.A2, "A3_Ry": kzk.A3, "G1_Ry": kzk.G1, "G2_Ry": kzk.G2, "G3_Ry": kzk.G3,
        "G4_Ry": kzk.G4, "A0_Ry": kzk.A0, "HARTREE_PER_RYDBERG": kzk.HARTREE_PER_RYDBERG,
        "DENSITY_FLOOR": kzk.DENSITY_FLOOR,
        "A0_independent_Ry": -2.0 * 0.75 * (9.0 / (4.0 * math.pi ** 2)) ** (1.0 / 3.0),
        "transcription": "TO VERIFY against Kwee, Zhang, Krakauer, PRL 100, 126404 (2008), Table I (a1, a2, a3, g1..g4); "
                         "not re-checked in this session. vera memory finite-size-forces-stress-2026-09-23.md records a "
                         "match to the PDF on that date, which was not reproduced here (no PDF in the repository). "
                         "A0 is NOT a table entry: it is the analytic Slater constant, computed and checked here."}
    consts = (kzk.A1, kzk.A2, kzk.A3, kzk.G1, kzk.G2, kzk.G3, kzk.G4)
    # --- native Perdew-Zunger against the literature form
    lda_correlation = O.atomic.lda_correlation          # the OLD kzk's own dependency (native lda_terms), pinned rev
    rs_grid = np.concatenate([np.geomspace(0.05, 0.999, 12), [1.0], np.geomspace(1.001, 200, 25)])
    dens = 3.0 / (4.0 * np.pi * rs_grid ** 3)
    ec_n, vc_n = lda_correlation(dens)
    ec_o, dec_o = own_pz(rs_grid)
    vc_o = ec_o - rs_grid / 3.0 * dec_o
    ec_m, dec_m = own_pz(rs_grid, modified=True)
    vc_m = ec_m - rs_grid / 3.0 * dec_m
    res["perdew_zunger"] = {
        "note": "native lda_terms = Perdew-Zunger with A, B kept and C, D re-derived for value+slope continuity at "
                "r_s = 1; it differs from the literature constants below r_s = 1 (see max_abs_diff_*_literature)",
        "rs": rs_grid, "ec_native": ec_n, "vc_native": vc_n, "ec_literature": ec_o, "vc_literature": vc_o,
        "ec_continuous": ec_m, "vc_continuous": vc_m,
        "max_abs_diff_ec_literature": float(np.max(np.abs(ec_n - ec_o))),
        "max_abs_diff_vc_literature": float(np.max(np.abs(vc_n - vc_o))),
        "max_rel_diff_ec_literature": float(np.max(np.abs(ec_n - ec_o) / np.abs(ec_o))),
        "rs_of_max_diff_literature": float(rs_grid[int(np.argmax(np.abs(ec_n - ec_o)))]),
        "max_abs_diff_ec_continuous": float(np.max(np.abs(ec_n - ec_m))),
        "max_abs_diff_vc_continuous": float(np.max(np.abs(vc_n - vc_m))),
        "literature_jump_at_rs1": float(own_pz(np.array(1.0 - 1e-15))[0] - own_pz(np.array(1.0))[0])}
    res["rydberg_pz"] = {"rs": rs_grid, "ec_Ry": kzk._pz_correlation_rydberg(rs_grid)[0],
                         "dec_Ry": kzk._pz_correlation_rydberg(rs_grid)[1]}
    res["g_function"] = {"rs": rs_grid, "g": kzk._g(rs_grid), "dg": kzk._dg(rs_grid)}
    # --- boundaries and the (r_s, L) grid
    rs_dense = np.concatenate([np.geomspace(0.3, 25.0, 40)])
    per_L = {}
    for L in (7.5, 8.0, 10.0, 12.0, 20.0, 40.0):
        alpha, beta, gh, gl = kzk._intermediate_cubic(L)
        gamma = kzk._rs_at(2, L)
        marks = [f * b for b in (gamma, gh, gl) for f in (0.5, 0.9, 1 - 1e-6, 1.0, 1 + 1e-6, 1.1, 2.0)]
        rs = np.concatenate([rs_dense, marks])
        ex, ec, xc = kzk.fs_exchange(rs, L), kzk.fs_correlation(rs, L), kzk.fs_exchange_correlation(rs, L)
        inf = kzk.infinite_exchange_correlation(rs)
        oex, oec, oinf = own_kzk(rs, L, consts)
        cont = {}
        for name, b in (("gamma", gamma), ("gamma_h", gh), ("gamma_l", gl)):
            d = 1e-9 * b
            below, at, above = (np.array(b - d), np.array(b), np.array(b + d))
            cont[name] = {
                "rs": b, "ex": [float(kzk.fs_exchange(x, L)) for x in (below, at, above)],
                "ec": [float(kzk.fs_correlation(x, L)) for x in (below, at, above)],
                "ec_slope_left": float((kzk.fs_correlation(at, L) - kzk.fs_correlation(below, L)) / d),
                "ec_slope_right": float((kzk.fs_correlation(above, L) - kzk.fs_correlation(at, L)) / d)}
        lo = rs_dense[rs_dense <= min(gamma, gh)]
        per_L[str(L)] = {
            "L": L, "alpha": alpha, "beta": beta, "gamma": gamma, "gamma_h": gh, "gamma_l": gl,
            "electrons_at_boundaries": [2, 12, 0.5],
            "rs": rs, "ex": ex, "ec": ec, "xc_fs": xc, "xc_inf": inf,
            "diff_to_own": {"ex": float(np.max(np.abs(ex - oex))), "ec": float(np.max(np.abs(ec - oec))),
                            "xc_inf": float(np.max(np.abs(inf - oinf)))},
            "continuity": cont,
            "leading_term_cancellation": kzk.leading_term_cancellation(lo, L) if lo.size else None,
            "effective_length_of_volume": kzk.effective_length(L ** 3)}
    res["by_L"] = per_L
    # xc scaling 1/L^3 after the a1/L^2 cancellation (independent identity)
    sl = {}
    for rs in (1.0, 2.0):
        Ls = np.array([20.0, 40.0, 80.0, 160.0, 320.0])
        d = np.array([abs(float(kzk.fs_exchange_correlation(rs, L) - kzk.infinite_exchange_correlation(rs))) for L in Ls])
        sl[str(rs)] = {"L": Ls, "abs_diff": d, "slopes": np.diff(np.log(d)) / np.diff(np.log(Ls))}
    res["xc_scaling"] = sl
    return res


def _blob(n, L, centers, sigma, charge):
    h = L / n
    x = np.arange(n) * h
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    rho = np.zeros((n, n, n))
    for c in centers:
        # minimum-image Gaussian (periodic wrap): nearest image only is enough for sigma << L
        dx = (X - c[0] + 0.5 * L) % L - 0.5 * L
        dy = (Y - c[1] + 0.5 * L) % L - 0.5 * L
        dz = (Z - c[2] + 0.5 * L) % L - 0.5 * L
        rho += np.exp(-(dx ** 2 + dy ** 2 + dz ** 2) / (2 * sigma ** 2))
    rho *= charge / (rho.sum() * h ** 3)
    return rho.ravel(), h ** 3


def _record(c):
    return {"correction": c.correction, "xc_infinite": c.xc_infinite, "xc_finite": c.xc_finite, "length": c.length,
            "volume": c.volume, "n_cells": c.n_cells, "n_electrons": c.n_electrons,
            "cubic_deviation": c.cubic_deviation, "rs_range": list(c.rs_range), "energy_per_cell": c.energy_per_cell,
            "correction_per_cell": c.correction_per_cell, "corrected_energy_per_cell": c.corrected_energy_per_cell,
            "summary": c.summary()}


def stage_kzk_density(heavy=False):
    O = old()
    kzk = O.kzk
    res = {"units": "Hartree per supercell; densities in Bohr^-3; lengths Bohr"}
    consts = (kzk.A1, kzk.A2, kzk.A3, kzk.G1, kzk.G2, kzk.G3, kzk.G4)

    def own(density, dV, L):
        d = np.asarray(density, float).ravel()
        d = d[d > kzk.DENSITY_FLOOR]
        rs = (3.0 / (4.0 * np.pi * d)) ** (1.0 / 3.0)
        ex, ec, inf = own_kzk(rs, L, consts)
        return float((d * (inf - ex - ec)).sum() * dV)

    cases = {}
    # 1. uniform: the exact identity N (eps_inf - eps_FS)
    for rs in (1.0, 2.0, 3.0, 4.0, 6.0):
        for L, n in ((12.0, 24), (8.0, 16), (20.0, 32)):
            n0 = 3.0 / (4.0 * np.pi * rs ** 3)
            dV = (L / n) ** 3
            rho = np.full(n ** 3, n0)
            c = kzk.kzk_correction(rho, dV, np.eye(3) * L)
            ident = c.n_electrons * float(kzk.infinite_exchange_correlation(rs) - kzk.fs_exchange_correlation(rs, L))
            r = _record(c)
            r.update(identity_N_times_diff=ident, identity_residual_rel=abs(c.correction - ident) / max(abs(ident), 1e-300),
                     own_correction=own(rho, dV, L), kind="uniform", rs=rs, L=L, n=n)
            cases[f"uniform_rs{rs}_L{L}_n{n}"] = r
    # 2. Gaussian blob in a box (N = 4), 3 box sizes and a spacing pair for the quadrature error
    for L, n in ((8.0, 16), (12.0, 24), (12.0, 48), (20.0, 40)):
        rho, dV = _blob(n, L, [(L / 2,) * 3], 1.5, 4.0)
        c = kzk.kzk_correction(rho, dV, np.eye(3) * L)
        r = _record(c)
        below = rho <= kzk.DENSITY_FLOOR
        r.update(own_correction=own(rho, dV, L), kind="gaussian", sigma=1.5, charge=4.0, L=L, n=n,
                 nodes_below_floor=int(below.sum()), electrons_below_floor=float(rho[below].sum() * dV),
                 density_digest=sha(rho))
        cases[f"gauss_L{L}_n{n}"] = r
    # 3. two-site density (N = 2)
    for L, n in ((12.0, 24),):
        rho, dV = _blob(n, L, [(L / 2 - 1.0, L / 2, L / 2), (L / 2 + 1.0, L / 2, L / 2)], 0.8, 2.0)
        c = kzk.kzk_correction(rho, dV, np.eye(3) * L)
        r = _record(c)
        r.update(own_correction=own(rho, dV, L), kind="two_site", L=L, n=n, density_digest=sha(rho))
        cases[f"two_site_L{L}_n{n}"] = r
    # 4. the silent-drop and refusal behavior (uniform rs = 2, L = 12, n = 24)
    L, n, rs = 12.0, 24, 2.0
    n0, dV = 3.0 / (4.0 * np.pi * rs ** 3), (12.0 / 24) ** 3
    base = np.full(n ** 3, n0)
    edge = {}

    def attempt(name, rho, **kw):
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                c = kzk.kzk_correction(rho, dV, np.eye(3) * L, **kw)
            rec = {"raised": None, **_record(c), "warnings": [str(x.message)[:80] for x in w]}
        except Exception as e:  # noqa: BLE001  - the behavior is the datum
            rec = {"raised": type(e).__name__, "message": str(e)[:120]}
        edge[name] = rec

    neg = base.copy()
    neg[:100] = -1e-3
    attempt("negative_100_points", neg)
    nan = base.copy()
    nan[5] = np.nan
    attempt("nan_1_point", nan)
    inf = base.copy()
    inf[7] = np.inf
    attempt("inf_1_point", inf)
    zero = base.copy()
    zero[:50] = 0.0
    attempt("zero_50_points", zero)
    tiny = base.copy()
    tiny[:50] = 5e-13
    attempt("below_floor_50_points", tiny)
    attempt("exactly_at_floor_50_points", np.where(np.arange(base.size) < 50, kzk.DENSITY_FLOOR, base))
    attempt("all_vacuum", np.full(n ** 3, 1e-14))
    attempt("all_negative", np.full(n ** 3, -1e-3))
    attempt("floor_raised_to_1e-3_blob", _blob(24, 12.0, [(6.0,) * 3], 1.5, 4.0)[0], floor=1e-3)
    attempt("vacuum_blob_rs_far_beyond_gamma_l", _blob(24, 12.0, [(6.0,) * 3], 1.0, 0.3)[0])
    res["edge"] = edge
    # reference numbers for the plan's claims: electrons summed vs electrons that contribute
    res["edge_note"] = ("negative and NaN densities are dropped without a message (keep = density > floor) while "
                        "n_electrons still sums them (NaN propagates into n_electrons); the Rust port raises")
    # 5. non-cubic cells and n_cells / energy_per_cell
    for name, cell in (("orthorhombic_8_10_15", np.diag([8.0, 10.0, 15.0])),
                       ("triclinic", lattices(10.0)["triclinic"]),
                       ("fcc_primitive_x2", 2.0 * lattices(8.0)["fcc"])):
        vol = abs(np.linalg.det(cell))
        n = 24
        dV = vol / n ** 3
        Leff = float(np.cbrt(vol))
        rho = np.full(n ** 3, 3.0 / (4.0 * np.pi * 2.0 ** 3))
        c = kzk.kzk_correction(rho, dV, cell, n_cells=4, energy_per_cell=-1.25)
        r = _record(c)
        r.update(own_correction=own(rho, dV, Leff), kind="uniform_rs2_noncubic", cell=cell, n=n)
        cases[f"noncubic_{name}"] = r
    res["cases"] = cases
    return res


def stage_kzk_heg(heavy=False):
    O = old()
    kzk = O.kzk
    rs = 2.0
    rows = []

    def occ(R2):
        r = int(math.sqrt(R2)) + 1
        return np.array([m for m in itertools.product(range(-r, r + 1), repeat=3) if sum(x * x for x in m) <= R2])

    for R2 in (1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 29, 30, 32, 33,
               34, 35, 36):
        o = occ(R2)
        Ns = len(o)
        N = 2 * Ns
        L = rs * (4 * np.pi * N / 3) ** (1 / 3)
        n = N / L ** 3
        d2 = ((o[:, None, :] - o[None, :, :]) ** 2).sum(-1)
        with np.errstate(divide="ignore"):
            vE = np.where(d2 > 0, 4 * np.pi / (d2 * (2 * np.pi / L) ** 2), 0.0)
        A = -(vE.sum()) / (L ** 3) / N + 0.5 * SC_LVM_LITERATURE / L
        e_inf = -0.75 * (3 * n / np.pi) ** (1 / 3)
        rows.append((N, L, A, e_inf, float(kzk.fs_exchange(rs, L))))
    rows = np.array(rows)
    dA = rows[:, 2] - rows[:, 3]
    dK = rows[:, 4] - rows[:, 3]
    return {
        "note": "Gamma-point HF exchange per electron of the unpolarized gas at r_s = 2 (Ewald kernel + 1/2 v_M, "
                "v_M = -2.837297479/L) against the OLD fs_exchange; diagnostic of a1/L^2 and the branch, not a Rust gate",
        "rs": rs, "N": rows[:, 0], "L": rows[:, 1], "hf_exchange": rows[:, 2], "e_x_inf": rows[:, 3],
        "kzk_fs_exchange": rows[:, 4], "hf_minus_inf": dA, "kzk_minus_inf": dK,
        "mean_hf_minus_inf": float(dA.mean()), "mean_kzk_minus_inf": float(dK.mean()),
        "rms_hf_minus_kzk": float(np.sqrt(((dA - dK) ** 2).mean())),
        "loglog_slope_hf": float(np.polyfit(np.log(rows[:, 1]), np.log(np.abs(dA)), 1)[0]),
        "loglog_slope_kzk": float(np.polyfit(np.log(rows[:, 1]), np.log(np.abs(dK)), 1)[0])}


def stage_voxel(heavy=False):
    O = old()
    P = O.poisson
    res = {"units": "Bohr^2 (the integral of 1/r over the voxel)", "cube_constant": CUBE_CONSTANT,
           "cube_constant_literature": 2.3800774}
    xg, wg = np.polynomial.legendre.leggauss(64)
    xg, wg = 0.5 * xg, 0.5 * wg                 # nodes on [-1/2, 1/2], weights summing to 1

    def faces_quadrature(step):
        """int_V d^3r/r = 1/2 sum_f h_f int_f dS/r, smooth 2-D Gauss-Legendre on each parallelogram."""
        step = np.asarray(step, float)
        total = 0.0
        for axis in range(3):
            e = [step[:, i] for i in range(3) if i != axis]
            normal = np.cross(*e)
            area = np.linalg.norm(normal)
            normal /= area
            for side in (-0.5, 0.5):
                c = side * step[:, axis]
                height = abs(float(normal @ c))
                u, v = np.meshgrid(xg, xg, indexing="ij")
                x = c + u[..., None] * e[0] + v[..., None] * e[1]
                total += 0.5 * height * area * float(np.sum(wg[:, None] * wg[None, :] / np.linalg.norm(x, axis=-1)))
        return total

    rng = np.random.default_rng(20261030)
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    cases = {
        "cube_h0.5": 0.5 * np.eye(3),
        "cube_h1.3": 1.3 * np.eye(3),
        "orthorhombic": np.diag([0.4, 0.7, 1.1]),
        "rotated_cube_h0.5": 0.5 * Q,
        "rotated_cube_h1.3": 1.3 * Q,
        "sheared": np.array([[0.5, 0.15, 0.1], [0.0, 0.55, 0.12], [0.0, 0.0, 0.45]]),
        "triclinic_voxel": lattices(1.0)["triclinic"].T / 12.0,
        "fcc_voxel": lattices(8.0)["fcc"].T / 12.0,
        "hex_voxel": lattices(8.0)["hex"].T / 12.0,
    }
    out = {}
    for name, step in cases.items():
        value = P.voxel_self_potential(step)
        vol = abs(np.linalg.det(step))
        ref = faces_quadrature(step)
        rec = {"step": step, "volume": vol, "value": value, "value_over_volume": value / vol, "independent_faces": ref,
               "rel_diff_to_independent": abs(value - ref) / ref,
               "code_path": "rectangular closed form"
               if np.all(np.abs(step - np.diag(np.diag(step))) <= 1e-12 * max(np.abs(np.diag(step)).max(), 1.0))
               else "general triangle path"}
        if name.startswith("cube") or name.startswith("rotated_cube"):
            h = float(np.cbrt(vol))
            rec["analytic_cube"] = h * h * CUBE_CONSTANT
            rec["rel_diff_to_analytic"] = abs(value - h * h * CUBE_CONSTANT) / (h * h * CUBE_CONSTANT)
        out[name] = rec
    res["cases"] = out
    res["cell_self_potential_cube"] = {"h": 0.5, "value": P.cell_self_potential(0.5, 0.5, 0.5)}
    return res


def _vm(cell, eta, nr=6, ng=10):
    from scipy.special import erfc
    cell = np.asarray(cell, float)
    Om = abs(np.linalg.det(cell))
    B = 2 * np.pi * np.linalg.inv(cell).T
    sr = 0.0
    for n in itertools.product(range(-nr, nr + 1), repeat=3):
        if n == (0, 0, 0):
            continue
        r = np.linalg.norm(np.array(n) @ cell)
        sr += erfc(eta * r) / r
    lr = 0.0
    for m in itertools.product(range(-ng, ng + 1), repeat=3):
        if m == (0, 0, 0):
            continue
        G = np.array(m) @ B
        g2 = G @ G
        lr += 4 * np.pi / g2 * np.exp(-g2 / (4 * eta ** 2))
    return float(sr + lr / Om - 2 * eta / np.sqrt(np.pi) - np.pi / (eta ** 2 * Om))


def stage_madelung(heavy=False):
    from mandacaru import _native
    ctx = _native.Context()
    res = {"units": "Hartree*Bohr per unit charge; v_M = Madelung potential of one point charge with its images in the "
                    "neutralizing background; E_Ewald(one charge) = v_M/2"}
    for name, cell in (("sc", lattices(1.0)["sc"]), ("fcc", lattices(1.0)["fcc"]), ("bcc", lattices(1.0)["bcc"]),
                       ("hex", lattices(1.0)["hex"])):
        om = abs(np.linalg.det(cell))
        rws = (3 * om / (4 * np.pi)) ** (1 / 3)
        v1, v2 = _vm(cell, 2.0), _vm(cell, 3.0)
        e = ctx.ewald(np.zeros((1, 3)), np.array([1.0]), cell)["energy"]
        # same lattice at two sizes: v_M scales as 1/L exactly
        e10 = ctx.ewald(np.zeros((1, 3)), np.array([1.0]), 10.0 * cell)["energy"]
        res[name] = {"cell": cell, "volume": om, "r_ws": rws,
                     "vM_eta2": v1, "vM_eta3": v2, "eta_spread": abs(v1 - v2),
                     "vM_times_rws_eta2": v1 * rws, "vM_times_rws_eta3": v2 * rws,
                     "native_energy": e, "native_2E": 2 * e, "native_2E_times_rws": 2 * e * rws,
                     "native_minus_own": 2 * e - 0.5 * (v1 + v2),
                     "scaling_residual_L10": 2 * e10 * 10.0 - 2 * e}
    res["sc"]["L_vM_own"] = res["sc"]["vM_eta2"]
    res["sc"]["L_vM_literature"] = SC_LVM_LITERATURE
    res["sc"]["L_vM_minus_literature"] = res["sc"]["vM_eta2"] - SC_LVM_LITERATURE
    res["literature_vM_rws"] = {"note": "Wigner-lattice energies per particle -0.895929 (bcc) and -0.895874 (fcc) "
                                        "e^2/r_s (literature values from memory, to verify); v_M r_ws = 2x those",
                                "bcc": -2 * 0.895929, "fcc": -2 * 0.895874}
    return res


def _kernel_record(O, name, cell, n, ref, keep=None):
    step = cell.T / n
    t0 = time.time()
    K = O.mpc.wigner_seitz_kernel((n, n, n), step)
    dt = time.time() - t0
    dV = abs(np.linalg.det(step))
    s = float(K.sum() * dV)
    refl = np.roll(K[::-1, ::-1, ::-1], 1, axis=(0, 1, 2))          # K(-d) in FFT order
    pos = K > 0
    idx = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1), (n // 4, 0, 0), (n // 4, n // 4, 0),
           (n // 4, n // 4, n // 4), (n // 2, 0, 0), (n // 2, n // 2, 0), (n // 2, n // 2, n // 2),
           (n - 1, 0, 0), (n - 1, n - 1, n - 1), (n // 3, n // 5, n // 7)]
    rec = {"lattice": name, "n": n, "step": step, "cell": cell, "dV": dV, "volume": abs(np.linalg.det(cell)),
           "sum_times_dV": s, "reference": ref, "error_abs": s - ref, "error_rel": (s - ref) / ref,
           "K000": float(K[0, 0, 0]), "K000_times_dV": float(K[0, 0, 0] * dV),
           "min_image_radius": float(np.max(1.0 / K[pos])), "nonzero": int(pos.sum()), "total_nodes": int(K.size),
           "enclosed_volume": float(np.count_nonzero(K) * dV),
           "is_even": bool(np.array_equal(K, refl)), "asymmetry_max": float(np.abs(K - refl).max()),
           "asymmetric_nodes": int((K != refl).sum()),
           "node_index": [list(i) for i in idx], "node_value": [float(K[i]) for i in idx],
           "digest": fp(K), "seconds": dt}
    if keep is not None:
        keep[f"{name}_n{n}"] = K
        rec["npz_key"] = f"{name}_n{n}"          # in ws_kernel.npz; its digest above is the sha256 of these bytes
    return rec


def stage_ws_kernel(heavy=False):
    O = old()
    res = {"units": "kernel in Bohr^-1, sums in Bohr^2; node d is index offsets in FFT order", "cube_constant": CUBE_CONSTANT,
           "kernels": {}}
    keep = {}
    refs = {}
    sizes_cubic = (12, 24, 48) + ((96,) if heavy else ())
    plan = []
    for L in (8.0, 12.0, 20.0):
        plan += [("sc", L, n) for n in sizes_cubic]
    for name in ("fcc", "bcc", "hex", "triclinic"):
        plan += [(name, 12.0, n) for n in (12, 24, 48)]
    for name, L, n in plan:
        cell = lattices(L)[name]
        key = (name, L)
        if key not in refs:
            refs[key] = (L * L * CUBE_CONSTANT) if name == "sc" else ws_cell_integral(cell)
        rec = _kernel_record(O, name, cell, n, refs[key], keep if (name, L, n) in (("sc", 12.0, 48), ("sc", 12.0, 96), ("hex", 12.0, 12)) else None)
        rec["L"] = L
        res["kernels"][f"{name}_L{L}_n{n}"] = rec
    for name, L in sorted({(p[0], p[1]) for p in plan}):
      for tag, nmax in (("richardson", 48), ("richardson_with_96", 96)):
        ns = [p[2] for p in plan if p[0] == name and p[1] == L and p[2] <= nmax]
        if tag == "richardson_with_96" and 96 not in ns:
            continue
        errs = [res["kernels"][f"{name}_L{L}_n{n}"]["error_abs"] for n in ns]
        sums = [res["kernels"][f"{name}_L{L}_n{n}"]["sum_times_dV"] for n in ns]
        orders = [float(math.log(abs(errs[i] / errs[i + 1])) / math.log(ns[i + 1] / ns[i])) for i in range(len(ns) - 1)
                  if errs[i] != 0 and errs[i + 1] != 0]
        res.setdefault(tag, {})[f"{name}_L{L}"] = {
            "n": ns, "sum_times_dV": sums, "error_abs": errs, "reference": refs[(name, L)],
            "observed_order_between_consecutive_n": orders,
            "extrapolated_h2": [float(sums[i + 1] + (sums[i + 1] - sums[i]) / 3.0) for i in range(len(ns) - 1)],
            "extrapolated_error": [float(sums[i + 1] + (sums[i + 1] - sums[i]) / 3.0 - refs[(name, L)]) for i in range(len(ns) - 1)]}
    res["_npz"] = keep
    # the independent polyhedron integral validated where an analytic value exists (the cube) and by the h^2 limit above
    cube = ws_cell_integral(lattices(12.0)["sc"])
    res["reference_check"] = {"sc_L12_polyhedron": cube, "sc_L12_analytic": 144.0 * CUBE_CONSTANT,
                              "rel_diff": (cube - 144.0 * CUBE_CONSTANT) / (144.0 * CUBE_CONSTANT)}
    # the earlier-analysis 12^3 set (kept as it was frozen: same step matrices, same shape)
    old12 = {"cubic": np.eye(3) * 0.5, "hex": np.array([[1, 0, 0], [-.5, np.sqrt(3) / 2, 0], [0, 0, 1.6]]).T * 0.5,
             "fcc": 0.5 * np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]]).T * 2,
             "triclinic": np.array([[1, 0, 0], [.3, 1.1, 0], [.2, .25, 0.9]]).T * 0.5}
    first = {}
    for k, step in old12.items():
        K = O.mpc.wigner_seitz_kernel((12, 12, 12), step)
        first[k] = {"sum": float(K.sum()), "K000": float(K[0, 0, 0]), "K111": float(K[1, 1, 1]), "K654": float(K[6, 5, 4]),
                    "min_r": float(1 / K[K > 0].min()), "nonzero": int((K > 0).sum())}
    res["analysis_12cubed_set"] = first
    return res


def _jgauss(O, L, n, s):
    """One electron, Gaussian density: periodic (G-sum) and Wigner-Seitz pair integrals and the OLD contraction."""
    h = L / n
    step = np.eye(3) * h
    x = (np.arange(n) - n // 2) * h
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    rho = np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / (2 * s * s))
    rho /= rho.sum() * h ** 3
    rho = np.fft.ifftshift(rho)
    dV, Om = h ** 3, L ** 3
    g = 2 * np.pi * np.fft.fftfreq(n, d=h)
    G2 = g[:, None, None] ** 2 + g[None, :, None] ** 2 + g[None, None, :] ** 2
    with np.errstate(divide="ignore"):
        vE = np.where(G2 > 0, 4 * np.pi / G2, 0.0)
    rk = np.fft.fftn(rho) * dV
    JE = float(np.real(np.sum(vE * np.abs(rk) ** 2)) / Om)
    K = O.mpc.wigner_seitz_kernel((n, n, n), step)
    Jf = float(np.real(np.sum(np.fft.fftn(K) * dV * np.abs(rk) ** 2)) / Om)
    fs = O.mpc.exchange_correlation_hole_energy(np.array([[[[JE]]]]), np.array([[[[Jf]]]]), np.array([[1.0]]),
                                                np.zeros((1, 1, 1, 1)), volume=Om)
    return JE, Jf, fs


def stage_mpc_n1(heavy=False):
    O = old()
    res = {"units": "Hartree per supercell; s, L in Bohr",
           "warning": "the 'old_*' numbers carry the 1/2 N v_M double count; the Rust gate is 'corrected_expected', not "
                      "'old_correction'",
           "cases": {}}
    sizes = (32, 48) + ((96,) if heavy else ())
    for L, s in ((10.0, 1.0), (14.0, 1.0), (18.0, 1.0), (14.0, 1.5)):
        vM = SC_LVM_LITERATURE / L
        for n in sizes:
            if n == 96 and s != 1.0:
                continue
            JE, Jf, fs = _jgauss(O, L, n, s)
            iso = 1.0 / (s * math.sqrt(math.pi))
            Om = L ** 3
            q = 4 * math.pi * s * s / Om
            need = 0.5 * q                      # 2 pi s^2 / Omega = 2 pi <r^2>/(3 Omega): the corrected expected value
            old_c = fs.correction
            res["cases"][f"L{L}_s{s}_n{n}"] = {
                "L": L, "s": s, "n": n, "h": L / n, "JE": JE, "Jf": Jf, "iso": iso,
                "kernel_identity_lhs_JE_minus_iso": JE - iso, "kernel_identity_rhs_vM_plus_4pi_s2_over_Om": vM + q,
                "kernel_identity_residual": JE - iso - (vM + q), "Jf_minus_iso": Jf - iso,
                "old_correction": old_c, "old_hole_periodic": fs.hole_energy_periodic,
                "old_hole_truncated": fs.hole_energy_truncated, "old_correction_per_cell": fs.correction_per_cell,
                "vM": vM, "vM_half_plus_2pi_s2_over_Om": 0.5 * vM + need,
                "corrected_expected": need, "old_minus_corrected": old_c - need,
                "excess_over_half_vM": (old_c - need) / (0.5 * vM), "corrected_by_formula": old_c - 0.5 * vM}
    # contraction parity: random tensors, complex part present (the old code takes the real part)
    rng = np.random.default_rng(20261030)
    M = 3
    sym = lambda t: 0.5 * (t + np.einsum("pqrs->qpsr", t))
    ep, et = sym(rng.standard_normal((M,) * 4)), sym(rng.standard_normal((M,) * 4))
    D = rng.standard_normal((M, M))
    D = 0.5 * (D + D.T)
    G = rng.standard_normal((M,) * 4) + 1j * rng.standard_normal((M,) * 4)
    fs = O.mpc.exchange_correlation_hole_energy(ep, et, D, G, volume=1000.0, n_cells=4, energy_per_cell=-3.0)
    res["contraction"] = {"M": M, "seed": 20261030, "eri_periodic_sha": sha(ep), "eri_truncated_sha": sha(et),
                          "D_sha": sha(D), "Gamma_re_sha": sha(G.real), "Gamma_im_sha": sha(G.imag),
                          "hole_energy_periodic": fs.hole_energy_periodic, "hole_energy_truncated": fs.hole_energy_truncated,
                          "correction": fs.correction, "correction_per_cell": fs.correction_per_cell,
                          "n_electrons": fs.n_electrons, "mpc_energy_per_cell": fs.mpc_energy_per_cell,
                          "per_electron": fs.per_electron, "summary": fs.summary(),
                          "convention": "hole = Gamma - D_pr D_qs; E = 1/2 sum eri_pqrs hole_pqrs; real part taken"}
    # null case: identical kernels give exactly zero
    z = O.mpc.exchange_correlation_hole_energy(ep, ep, D, G, volume=1.0)
    res["null_case_correction"] = z.correction
    return res


def _ccmh_pair_sum(L, a, rng_m):
    m = np.array([t for t in itertools.product(range(-rng_m, rng_m + 1), repeat=3) if t != (0, 0, 0)])
    k = np.linalg.norm(m, axis=1) * 2 * np.pi / L
    S = 1 - np.exp(-k ** 2 * a * a / 4)
    return float((1 / (2 * L ** 3)) * np.sum(4 * np.pi / k ** 2 * (S - 1)))


def _fake_integrals(psi, h, n, L, nb):
    import types
    grid = types.SimpleNamespace(shape=(n, n, n), dV=h ** 3)
    return types.SimpleNamespace(grid=grid, _engine=types.SimpleNamespace(_psi=psi[:nb]), orthogonalize=False,
                                 mo_coefficients=None, cell=np.eye(3) * L, _lowdin_x=lambda: np.eye(nb))


def stage_ccmh_synth(heavy=False):
    O = old()
    cc = O.ccmh
    res = {"units": "Hartree; L in Bohr; 'old_correction' is the OLD per-electron number labelled 'per supercell'",
           "warning": "old totals carry defects 2 (missing N_e) and 3 (full not connected S): not a Rust gate",
           "triples": {}, "synthetic": {}, "gaussian_hole": {}, "n1": {}}
    for name, cell in (("cubic_14", np.eye(3) * 14.0), ("triclinic_10", lattices(10.0)["triclinic"])):
        trip, kv = cc._reciprocal_vectors(cell, 3)
        res["triples"][name] = {"count": len(trip), "triples_digest": fp(trip.astype(float)), "kvectors_digest": fp(kv),
                                "shell_magnitudes": np.unique(np.round(np.linalg.norm(kv, axis=1), 10))[:6]}
    # (1) the earlier frozen synthetic: L = 14, N = 64, S = 1 - exp(-k^2)
    trip, kv = cc._reciprocal_vectors(np.eye(3) * 14.0, 3)
    kk = np.linalg.norm(kv, axis=1)
    sf = cc.StructureFactor(kv, 1 - np.exp(-kk ** 2), kk, 64.0, 0.0)
    c = cc.ccmh_correction(sf, volume=14.0 ** 3)
    sh_k, sh_s = sf.shells(3)
    res["synthetic"] = {"L": 14.0, "N": 64.0, "S": "1-exp(-k^2)", "correction": c.correction, "limit": c.limit,
                        "rpa_limit": c.rpa_limit, "rpa_correction": c.rpa_correction, "plasma_frequency": c.plasma_frequency,
                        "quadratic_spread": c.quadratic_spread, "reliable": c.reliable, "density": c.density,
                        "shell_k": sh_k, "shell_S": sh_s, "summary": c.summary()}
    # (2) unit Gaussian xc hole S = 1 - exp(-k^2 a^2/4): per-electron closure
    a = 2.0
    eps_inf = -1.0 / (math.sqrt(math.pi) * a)
    for L in (10.0, 14.0, 20.0):
        Om = L ** 3
        pair = _ccmh_pair_sum(L, a, 20)
        vM = SC_LVM_LITERATURE / L
        exact_missing = math.pi * a * a / (2 * Om)
        trip, kv = cc._reciprocal_vectors(np.eye(3) * L, 3)
        kk = np.linalg.norm(kv, axis=1)
        sf = cc.StructureFactor(kv, 1 - np.exp(-kk ** 2 * a * a / 4), kk, 64.0, 0.0)
        for shells in (1, 2, 3, 5):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                c = cc.ccmh_correction(sf, volume=Om, shells=shells)
            res["gaussian_hole"][f"L{L}_shells{shells}"] = {
                "L": L, "a": a, "shells": shells, "N_e": 64.0,
                "old_correction": c.correction, "old_correction_per_cell": c.correction_per_cell, "limit": c.limit,
                "exact_limit_a2_over_4": a * a / 4, "quadratic_spread": c.quadratic_spread, "reliable": c.reliable,
                "warned": len(w) > 0, "exact_missing_per_electron": exact_missing,
                "corrected_total_N_e_times_exact_missing": 64.0 * exact_missing,
                "corrected_total_N_e_times_old_per_electron": 64.0 * c.correction,
                "pair_sum": pair, "vM_half": 0.5 * vM, "eps_inf": eps_inf,
                "closure_before": pair + 0.5 * vM - eps_inf, "closure_after_exact_limit": pair + 0.5 * vM + exact_missing - eps_inf,
                "closure_after_old_extrapolated": pair + 0.5 * vM + c.correction - eps_inf}
    # (3) one electron, Gaussian density, through structure_factor(): full S vs connected S_c
    for L, n in ((12.0, 36), (16.0, 48)):
        h = L / n
        s = 1.0
        x = (np.arange(n) - n // 2) * h
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        g = np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / (4 * s * s))
        psi = []
        for f in (g, X * g, Y * g, Z * g):
            f = np.fft.ifftshift(f).ravel()
            psi.append(f / np.sqrt((f ** 2).sum() * h ** 3))
        psi = np.array(psi)
        for nb, label in ((1, "minimal"), (4, "s_p")):
            D = np.zeros((nb, nb))
            D[0, 0] = 1.0
            Gm = np.zeros((nb,) * 4)
            sfac = cc.structure_factor(_fake_integrals(psi, h, n, L, nb), D, Gm, limit=3)
            sp = np.fft.fftn((psi[0] * psi[0]).reshape(n, n, n)) * h ** 3
            trip, kv = cc._reciprocal_vectors(np.eye(3) * L, 3)
            f0 = np.array([sp[tuple(int(m) % n for m in t)] for t in trip])
            Sc = sfac.values - np.abs(f0) ** 2
            sfc = cc.StructureFactor(sfac.kvectors, Sc, sfac.magnitudes, 1.0, 0.0)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                cf, cn = cc.ccmh_correction(sfac, volume=L ** 3), cc.ccmh_correction(sfc, volume=L ** 3)
            first = int(np.argmin(sfac.magnitudes))
            res["n1"][f"L{L}_n{n}_{label}"] = {
                "L": L, "n": n, "s": s, "basis": label, "k_min": float(sfac.magnitudes.min()),
                "overlap_error": sfac.overlap_error, "S_full_first_shell": float(sfac.values[first]),
                "S_connected_first_shell": float(Sc[first]), "k2_s2_exact_small_k": float(sfac.magnitudes.min() ** 2 * s * s),
                "old_correction_full_S": cf.correction, "reliable_full": cf.reliable, "spread_full": cf.quadratic_spread,
                "connected_correction": cn.correction, "reliable_connected": cn.reliable, "spread_connected": cn.quadratic_spread,
                "exact_2pi_s2_over_Om": 2 * math.pi * s * s / L ** 3,
                "connected_over_exact": cn.correction / (2 * math.pi * s * s / L ** 3),
                "full_over_exact": cf.correction / (2 * math.pi * s * s / L ** 3),
                "S_full_digest": fp(sfac.values), "S_connected_digest": fp(Sc)}
    return res


def _fock_ops(n_modes):
    """Jordan-Wigner annihilation operators of ``n_modes`` modes as dense matrices (mode 0 leftmost)."""
    I2, Z, a = np.eye(2), np.diag([1.0, -1.0]), np.array([[0.0, 1.0], [0.0, 0.0]])
    ops = []
    for j in range(n_modes):
        mats = [Z] * j + [a] + [I2] * (n_modes - j - 1)
        m = mats[0]
        for t in mats[1:]:
            m = np.kron(m, t)
        ops.append(m)
    return ops


def stage_ccmh_index(heavy=False):
    """Which Gamma ordering (and which D index) makes N S(k) = <rho_k^dag rho_k> true?

    Spin-summed spatial RDMs of states built in a 3-orbital, 6-spin-orbital Fock space; the right-hand side is the
    operator expectation computed directly on the 64-dimensional state vector."""
    M_orb = 3
    n = 2 * M_orb                                  # spin orbital P = p + sigma*M (alpha first), as spatial_rdms expects
    ann = _fock_ops(n)
    cre = [a.T.copy() for a in ann]
    dim = 2 ** n
    vac = np.zeros(dim)
    vac[0] = 1.0           # a = [[0,1],[0,0]] sends the second basis vector to the first: empty = first, so |0> = index 0
    assert all(np.allclose(a @ vac, 0) for a in ann)
    rng = np.random.default_rng(20261030)

    def alpha_beta(C, A=None):
        """sum_pq C_pq a+_p,alpha a+_q,beta |0>  (+ sum_{p<q} A_pq a+_p,alpha a+_q,alpha |0>)."""
        psi = np.zeros(dim, dtype=complex)
        for p in range(M_orb):
            for q in range(M_orb):
                psi += C[p, q] * (cre[p] @ cre[M_orb + q] @ vac)
        if A is not None:
            for p in range(M_orb):
                for q in range(p + 1, M_orb):
                    psi += A[p, q] * (cre[p] @ cre[q] @ vac)
        return psi / np.linalg.norm(psi)

    def determinant(U, occ):
        """prod over occupied spatial j (both spins) of b+_j,sigma |0>, b+_j = sum_p U_pj a+_p."""
        psi = vac.astype(complex)
        for j in occ:
            for sg in (0, 1):
                b = sum(U[p, j] * cre[p + sg * M_orb] for p in range(M_orb))
                psi = b @ psi
        return psi / np.linalg.norm(psi)

    def rdms(psi):
        D = np.zeros((M_orb, M_orb), complex)
        G = np.zeros((M_orb,) * 4, complex)
        for p, q in itertools.product(range(M_orb), repeat=2):
            for sg in (0, 1):
                D[p, q] += np.vdot(ann[p + sg * M_orb] @ psi, ann[q + sg * M_orb] @ psi)
        for p, q, r, s_ in itertools.product(range(M_orb), repeat=4):
            for sa in (0, 1):
                for sb in (0, 1):
                    # Gamma_pqrs = <a+_p a+_q a_s a_r>  (spin of p,r = sa; spin of q,s = sb)
                    bra = ann[q + sb * M_orb] @ (ann[p + sa * M_orb] @ psi)
                    ket = ann[s_ + sb * M_orb] @ (ann[r + sa * M_orb] @ psi)
                    G[p, q, r, s_] += np.vdot(bra, ket)
        return D, G

    def rho_op(M):
        op = np.zeros((dim, dim), complex)
        for p, q in itertools.product(range(M_orb), repeat=2):
            for sg in (0, 1):
                op += M[p, q] * (cre[p + sg * M_orb] @ ann[q + sg * M_orb])
        return op

    def candidates(M, D, G):
        Mc = np.conj(M)
        pair = {"prsq": np.einsum("pq,rs,prsq->", M, Mc, G, optimize=True),        # ccmh.py:328
                "prqs": np.einsum("pq,rs,prqs->", M, Mc, G, optimize=True)}
        exch = {"D_ps": np.einsum("pq,qs,ps->", M, Mc, D, optimize=True),          # ccmh.py:329 as written
                "D_sp": np.einsum("pq,qs,sp->", M, Mc, D, optimize=True)}
        # exact algebra for ANY M:  <rho^+ rho> = sum conj(M_rs) M_pq [ Gamma_{s p r q} + delta_rp D_{s q} ]
        exact = (np.einsum("pq,rs,sprq->", M, Mc, G, optimize=True) + np.einsum("pq,ps,sq->", M, Mc, D, optimize=True))
        out = {f"pair_{a}__exch_{b}": pair[a] + exch[b] for a in pair for b in exch}
        out["exact_algebra"] = exact
        return out

    def rmat(complex_=False):
        z = rng.standard_normal((M_orb, M_orb))
        return z + 1j * rng.standard_normal((M_orb, M_orb)) if complex_ else z

    Cs = rmat(True)
    Cr = rmat()
    Ar = rmat()
    Q, _ = np.linalg.qr(rng.standard_normal((M_orb, M_orb)))
    Uc, _ = np.linalg.qr(rmat(True))
    states = {
        "2e_singlet_complex": (alpha_beta(0.5 * (Cs + Cs.T)), "spin-pure, complex"),
        "2e_singlet_real": (alpha_beta(0.5 * (Cr + Cr.T)), "spin-pure, real"),
        "2e_spin_mixed_real": (alpha_beta(Cr), "singlet + triplet(Sz=0), real"),
        "2e_triplet_Sz0_real": (alpha_beta(0.5 * (Cr - Cr.T)), "triplet Sz=0, real"),
        "2e_triplet_Sz1_real": (alpha_beta(np.zeros((M_orb, M_orb)) + 1e-30, Ar), "triplet Sz=1, real"),
        "2e_general_complex": (alpha_beta(Cs, rmat(True)), "no symmetry"),
        "4e_closed_shell_det_real": (determinant(Q, [0, 1]), "spin-pure N=4 singlet, real"),
        "4e_closed_shell_det_complex": (determinant(Uc, [0, 1]), "spin-pure N=4 singlet, complex"),
    }
    Mz = rmat(True)
    Mkinds = {"complex_symmetric(physical: real orbitals)": 0.5 * (Mz + Mz.T), "real_symmetric": (lambda R: 0.5 * (R + R.T))(rmat()),
              "general_complex": rmat(True)}
    res = {"units": "dimensionless (N S(k) of the state)", "convention": {
        "native_two_rdm": "quantum/properties.rs rdms: two[p,q,r,s] = <a+_p a+_q a_s a_r>; spatial_rdms sums the spin blocks "
                          "with p,r of one spin and q,s of the other, one[p,r] = <a+_p a_r> (read in this session)",
        "old_spatial_rdms": "algorithms/pseudo_forces.py at 241d688^: same blocks; hole trace sum_q(Gamma-DD)_pqrq = -D_pr",
        "old_contraction": "ccmh.py:328-329: M_pq conj(M_rs) Gamma_prsq + M_pq conj(M_qs) D_ps",
        "derived": "<rho^+ rho> = sum conj(M_rs) M_pq [Gamma_sprq + delta_rp D_sq]; for symmetric M the pair term is "
                   "Gamma_prqs and the exchange term uses D_qp (D transposed relative to the old line)"},
        "seed": 20261030, "cases": {}}
    phys = "complex_symmetric(physical: real orbitals)"
    for sname, (psi, label) in states.items():
        D, G = rdms(psi)
        N = float(np.real(np.trace(D)))
        row = {"label": label, "N": N, "Sz_like_trace_rule_residual": float(np.max(np.abs(np.einsum("pqrq->pr", G) - (N - 1.0) * D))),
               "Gamma_prsq_minus_prqs_max": float(np.max(np.abs(G - np.einsum("prsq->prqs", G)))),
               "D_minus_Dtranspose_max": float(np.max(np.abs(D - D.T))), "imag_D_max": float(np.max(np.abs(D.imag))),
               "state_digest": fp(np.abs(psi)), "M": {}}
        for mname, Mmat in Mkinds.items():
            op = rho_op(Mmat)
            direct = float(np.real(np.vdot(op @ psi, op @ psi)))
            cand = candidates(Mmat, D, G)
            row["M"][mname] = {"direct_operator_norm2": direct,
                               **{f"err_{k}": float(abs(v - direct)) for k, v in cand.items()}}
        res["cases"][sname] = row
    tol = 1e-10
    ok = {}
    for a in ("prsq", "prqs"):
        for b in ("D_ps", "D_sp"):
            key = f"err_pair_{a}__exch_{b}"
            ok[f"pair_{a}__exch_{b}"] = {
                "exact_for_physical_M_in_every_state": all(res["cases"][s]["M"][phys][key] < tol for s in states),
                "exact_for_physical_M_in_every_REAL_state": all(res["cases"][s]["M"][phys][key] < tol
                                                                for s in states if res["cases"][s]["imag_D_max"] < 1e-12),
                "exact_for_physical_M_in_spin_pure_singlets": all(res["cases"][s]["M"][phys][key] < tol
                                                                  for s in states if "singlet" in s or "closed_shell" in s),
                "worst_error_over_states": max(res["cases"][s]["M"][phys][key] for s in states)}
    res["verdict_for_physical_M"] = ok
    res["exact_algebra_worst_error_all_M_all_states"] = max(res["cases"][s]["M"][m]["err_exact_algebra"]
                                                            for s in states for m in Mkinds)
    return res


STAGES = {
    "stamps": (stage_stamps, False), "counts": (stage_counts, False), "kzk_table": (stage_kzk_table, True),
    "kzk_density": (stage_kzk_density, True), "kzk_heg": (stage_kzk_heg, True), "voxel": (stage_voxel, True),
    "madelung": (stage_madelung, True), "ws_kernel": (stage_ws_kernel, True), "mpc_n1": (stage_mpc_n1, True),
    "ccmh_synth": (stage_ccmh_synth, True), "ccmh_index": (stage_ccmh_index, True),
}
ORDER = ["stamps", "counts", "kzk_table", "kzk_density", "kzk_heg", "voxel", "madelung", "ws_kernel", "mpc_n1",
         "ccmh_synth", "ccmh_index"]


# ----------------------------------------------------------------------------------------------- driver
def diff(a, b, path="", out=None):
    """Bitwise comparison of two encoded structures; collects mismatches."""
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k in b:
                diff(a[k], b[k], f"{path}/{k}", out)
            else:
                out.append((f"{path}/{k}", "not in the stored file", None))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, f"length {len(a)} vs {len(b)}", None))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diff(x, y, f"{path}[{i}]", out)
    elif isinstance(a, float) and isinstance(b, float):
        if a != b:
            out.append((path, f"{a!r} vs {b!r}", abs(a - b) / max(abs(a), abs(b), 1e-300)))
    elif a != b:
        out.append((path, f"{a!r} vs {b!r}", None))
    return out


#: stored keys that depend on the clock, the install or the machine and are not replayed
VOLATILE = ("seconds", "mtime", "elapsed", "path", "working_tree_sha256", "old_tests_sha256", "newer_than_every_crate_file",
            "reason", "_stamp")


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in VOLATILE}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def stamp_compact():
    """Written into every stage file: what produced it (full table in stamps.json)."""
    import scipy
    return {"old_sources_sha256": {rel: hashlib.sha256(git("show", f"{rev}:src/mandacaru/{rel}").encode()).hexdigest()
                                   for rel, rev in SRC_FILES.items()},
            "src_rev": SRC_REV, "dep_rev": DEP_REV, "extension_sha256": sha_file(extension_path()),
            "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__, "script": SCRIPT}


def child(stage, heavy, verify):
    fn, takes = STAGES[stage]
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = fn(heavy=heavy) if takes else fn()
    npz = data.pop("_npz", None)
    data = enc(data)
    data["_stamp"] = stamp_compact()
    path = REF / f"{stage}.json"
    if verify:
        stored = json.loads(path.read_text())
        if stage == "stamps":
            # the pinned blobs must be identical; the rest is information that legitimately moves
            for k in ("git_head", "extension", "python", "numpy", "scipy", "platform"):
                if data.get(k) != stored.get(k):
                    print(f"   note: stamps[{k}] differs from the freeze (information, not a failure)")
            data = {"old_sources": data["old_sources"]}
            stored = {"old_sources": stored["old_sources"]}
        if stage == "counts" and any(v == "absent" for v in data["files"].values()):
            print("VERIFY counts: SKIPPED (the old test files are no longer in the tree)")
            sys.exit(0)
        mism = diff(strip(data), strip(stored))
        mism = [m for m in mism if "(heavy)" not in m[0]]
        worst = max([m[2] for m in mism if m[2] is not None], default=0.0)
        print(f"VERIFY {stage}: {'OK bitwise' if not mism else str(len(mism)) + ' mismatches'}"
              f" (largest relative {worst:.2e}) [{time.time() - t0:.1f} s]")
        for m in mism[:8]:
            print("   ", m)
        sys.exit(0 if not mism else 1)
    REF.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=0, separators=(",", ":")) + "\n")
    if npz:
        np.savez_compressed(REF / f"{stage}.npz", **npz)
    print(f"WROTE {stage}: {path.stat().st_size} B [{time.time() - t0:.1f} s]")


def run_child(stage, heavy, verify):
    cmd = [sys.executable, "-I", "-W", "ignore", str(Path(__file__).resolve()), "--child", stage]
    if heavy:
        cmd.append("--heavy")
    if verify:
        cmd.append("--verify")
    if not MEMGUARD.exists():
        sys.exit(f"memguard missing: {MEMGUARD}")
    guard = subprocess.run([sys.executable, str(MEMGUARD), "--need-gb", "1", "--wait", "120"])
    if guard.returncode != 0:
        sys.exit("memguard refused to start the child")
    env = dict(os.environ, OMP_NUM_THREADS="1", FS_FREEZE_ROOT=str(ROOT), FS_FREEZE_REF=str(REF),
               PYTHONPATH=str(ROOT / "src") if False else os.environ.get("PYTHONPATH", ""))
    t0 = time.time()
    p = subprocess.run(["/usr/bin/time", "-l"] + cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    wall = time.time() - t0
    sys.stdout.write(p.stdout)
    m = re.search(r"(\d+)\s+maximum resident set size", p.stderr)
    f = re.search(r"(\d+)\s+peak memory footprint", p.stderr)
    lines = p.stderr.splitlines()
    cut = next((i for i, l in enumerate(lines) if re.match(r"\s*[\d.,]+\s+real", l)), len(lines))
    if "\n".join(lines[:cut]).strip():
        sys.stderr.write("\n".join(lines[:cut]) + "\n")
    return {"stage": stage, "returncode": p.returncode, "wall_s": round(wall, 2),
            "max_rss_MB": round(int(m.group(1)) / 2 ** 20, 1) if m else None,
            "peak_footprint_MB": round(int(f.group(1)) / 2 ** 20, 1) if f else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", nargs="?", choices=["run", "verify"])
    ap.add_argument("--stage", action="append", choices=ORDER)
    ap.add_argument("--full", action="store_true", help="include the n = 96 kernels (heavy)")
    ap.add_argument("--child", choices=ORDER)
    ap.add_argument("--heavy", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.child:
        child(a.child, a.heavy, a.verify)
        return
    if not a.mode:
        ap.error("give run or verify")
    stages = a.stage or ORDER
    runs, bad = [], 0
    for s in stages:
        r = run_child(s, a.full or a.mode == "run", a.mode == "verify")
        runs.append(r)
        bad += r["returncode"] != 0
    if a.mode == "run":
        REF.mkdir(parents=True, exist_ok=True)
        prev = json.loads((REF / "runs.json").read_text()) if (REF / "runs.json").exists() else {}
        prev.update({r["stage"]: r for r in runs})
        (REF / "runs.json").write_text(json.dumps(prev, indent=1) + "\n")
    print(f"{len(runs) - bad} of {len(runs)} stages ok; peak footprint {max((r['peak_footprint_MB'] or 0) for r in runs)} MB")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
