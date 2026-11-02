"""Freeze the eight WP-X functions of ``periodic_paw.py`` and the ported dead-block functions (WP-F item 7).

Run from the repository root with the conda python and ``MANDACARU_PAW_PATH`` set::

    python test/migration/freeze_wpx_functions.py all                # every case, four processes each
    python test/migration/freeze_wpx_functions.py case si            # one case (driver: memguard, 4 workers, merge)
    python test/migration/freeze_wpx_functions.py list

The output directory (``--output``) defaults to ``test/migration/reference/periodic/wpx``; it gets ``<case>.json``.
Every case runs in its own process; the driver starts ``scripts/memguard.py`` before each of four workers (two at
4 threads, one at 1, one with every atom coordinate nudged one ulp) and stores, per quantity, the largest difference
to the first as the measured floor.  A worker builds the
crystal, evaluates the frozen quantities (the eight functions in ``freeze_wpx_eight.py``, the dead-block functions
in ``freeze_wpx_dead_block.py``), the independent checks (central differences of the same sums, sum rules, hermiticity)
and the input perturbations that prove each array is not vacuous.  Arrays are ``{"shape", "dtype", "re"[, "im"]}``.
Units: Bohr, Bohr^-1, Hartree, Hartree/Bohr unless a name says otherwise.  Quantities are keyed by the function name.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from freeze_stamp import extension_sha, repo_root, stamp  # noqa: E402
from freeze_wpx_eight import Recorder, collect_eight, dec, enc, moved  # noqa: E402

BASIS = {"size": "SZ", "filter_cutoff": 200}
MOVE = [0.05, 0.02, -0.03]      # Angstrom: the displacement of the existing lattice-sum tests
CASES = {
    "si": dict(symbol="Si", structure="diamond", a=5.43, grid_spacing="tenth", kpts={"size": (2, 2, 1), "gamma": True},
               move=MOVE, groups=("eight",)),
    "al2": dict(symbol="Al", structure="fcc_ortho", a=4.05, grid_spacing=0.3, kpts=None, move=MOVE, groups=("eight",)),
    "al": dict(symbol="Al", structure="fcc", a=4.05, grid_spacing=0.3, kpts={"size": (2, 2, 2), "gamma": False}, move=None,
               groups=("eight", "dead")),
    "si_sym": dict(symbol="Si", structure="diamond", a=5.43, grid_spacing="tenth", kpts={"size": (2, 2, 2), "gamma": False},
                   move=None, groups=("dead",)),
}


def atoms_of(spec):
    from ase.build import bulk
    if spec["structure"] == "fcc_ortho":
        atoms = bulk(spec["symbol"], "fcc", a=spec["a"], orthorhombic=True)
    else:
        atoms = bulk(spec["symbol"], spec["structure"], a=spec["a"])
    if spec["move"] is not None:
        atoms.positions[1] += spec["move"]
    if os.environ.get("WPX_ULP"):   # the floor run: every coordinate one ulp up (zeros by 1e-16 Angstrom)
        atoms.positions = np.where(atoms.positions == 0, 1e-16, np.nextafter(atoms.positions, np.inf))
    return atoms


def build(spec, **options):
    from mandacaru.pseudopotentials import periodic_paw as pp
    atoms = atoms_of(spec)
    grid_spacing = float(np.linalg.norm(atoms.cell[0])) / 10 if spec["grid_spacing"] == "tenth" else spec["grid_spacing"]
    crystal, _ = pp.build_crystal(atoms, grid_spacing, BASIS, kpts=spec["kpts"], **options)
    return atoms, grid_spacing, crystal


def worker(case, raw):
    from mandacaru.pseudopotentials import periodic_paw as pp
    spec = CASES[case]
    atoms, grid_spacing, crystal = build(spec)
    rec, rng = Recorder(), np.random.default_rng(20261101)
    rec.inputs.update(case=case, symbols=atoms.get_chemical_symbols(), cell_angstrom=enc(atoms.cell[:]),
                      positions_angstrom=enc(atoms.positions), grid_spacing_A=grid_spacing, basis_options=BASIS,
                      kpts=spec["kpts"], grid_shape=list(crystal.grid.shape), M=crystal.M,
                      centers_bohr=enc(crystal.centers), lattice_bohr=enc(crystal.lattice), seed=20261101)
    if "eight" in spec["groups"]:
        collect_eight(rec, pp, crystal, rng)
    if "dead" in spec["groups"]:
        import freeze_wpx_dead_block as dead
        dead.collect(rec, pp, spec, atoms, grid_spacing, crystal, rng, enc, moved, build)
    root = repo_root()
    path = os.environ["MANDACARU_PAW_PATH"]
    datasets = [str(Path(path) / "lda-sr" / f"{s}.parquet") for s in sorted(set(atoms.get_chemical_symbols()))]
    out = {"stamp": stamp(datasets=datasets, extra={"case": case, "groups": list(spec["groups"]),
                                                      "threads": os.environ.get("OMP_NUM_THREADS")}),
           "quantities": {k: enc(v) for k, v in rec.q.items()}, "perturbed": {k: enc(v) for k, v in rec.perturbed.items()},
           "notes": rec.notes, "checks": rec.checks, "inputs": rec.inputs,
           "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}
    Path(raw).write_text(json.dumps(out))
    print(f"worker {case}: {len(rec.q)} quantities, {len(rec.checks)} checks, root {root}")


def merge(case, runs, output):
    a, b, c, u = (json.loads(Path(p).read_text()) for p in runs)
    shas = {r["stamp"]["extension"]["sha256"] for r in (a, b, c, u)} | {extension_sha()}
    sources = {json.dumps(r["stamp"]["sources_sha256"], sort_keys=True) for r in (a, b, c, u)}
    if len(shas) != 1 or len(sources) != 1:
        raise SystemExit(f"{case}: the extension or a source changed during the runs; redo every case")
    floors, vacuity = {}, {}
    for name, ea in a["quantities"].items():
        x, y, z, w = (dec(r["quantities"][name]) for r in (a, b, c, u))
        scale = float(np.abs(x).max()) if x.size else 0.0
        entry = {"scale": scale, "dtype": ea["dtype"]}
        if ea["dtype"] == "int64":
            entry.update(repeat_exact=bool(np.array_equal(x, y)), threads1_exact=bool(np.array_equal(x, z)),
                         ulp_exact=bool(np.array_equal(x, w)))
        else:
            d1, d2, d3 = (float(np.abs(x - o).max()) for o in (y, z, w))
            entry.update(repeat_abs=d1, repeat_rel=d1 / scale if scale else 0.0, threads1_abs=d2,
                         threads1_rel=d2 / scale if scale else 0.0, ulp_abs=d3, ulp_rel=d3 / scale if scale else 0.0)
        floors[name] = entry
        if name in a["perturbed"]:
            change = float(np.abs(x - dec(a["perturbed"][name])).max())
            floor = max(entry.get("repeat_abs", 0.0), entry.get("threads1_abs", 0.0), entry.get("ulp_abs", 0.0), 1e-16 * scale)
            vacuity[name] = {"max_abs_change": change, "change_rel": change / scale if scale else 0.0,
                             "floor_abs": floor, "ratio_to_floor": change / floor if floor else None,
                             "pass": bool(change > 1e3 * floor), "perturbation": a["notes"][name]}
        else:
            vacuity[name] = {"pass": None, "perturbation": "none: " + a["notes"][name]}
    out = dict(a, floors=floors, non_vacuity=vacuity)
    out.pop("perturbed")
    out["stamp"]["extra"]["floor_runs"] = ("repeat: a second process at 4 threads; threads1: one process at 1 thread; ulp: "
                                           "4 threads with every atom coordinate nudged one ulp up (WPX_ULP=1)")
    out["stamp"]["extra"]["crates_newer_than_extension"] = newer_crates()
    out["tolerance_note"] = ("floors are measured, not tolerances: the completion plan sets tol = 100 x floor and a "
                             "default of 1e-12 for function cases on frozen inputs; checks[*].gate are the gates of the "
                             "existing tests")
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / f"{case}.json").write_text(json.dumps(out))
    bad = [k for k, v in vacuity.items() if v["pass"] is False] + [k for k, v in a["checks"].items() if v["pass"] is False]
    print(f"{case}: written; failed non-vacuity or gates: {bad or 'none'}")
    return bad


def newer_crates():
    ext = next(Path(repo_root(), "src", "mandacaru").glob("_native.cpython-314-darwin.so"))
    return sorted(str(p.relative_to(repo_root())) for p in Path(repo_root(), "crates").rglob("*.rs")
                  if p.stat().st_mtime > ext.stat().st_mtime and "target" not in p.parts)


def drive(case, output):
    root = repo_root()
    work, runs = Path(tempfile.mkdtemp(prefix=f"wpx_{case}_")), []
    for tag, threads in (("a", "4"), ("b", "4"), ("c", "1"), ("u", "4")):
        subprocess.run([sys.executable, "scripts/memguard.py", "--need-gb", "2", "--wait", "120"], cwd=root, check=True)
        env = dict(os.environ, WPX_ULP="1" if tag == "u" else "", **{v: threads for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                                        "VECLIB_MAXIMUM_THREADS", "MANDACARU_NUM_THREADS")})
        raw = work / f"{tag}.json"
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", case, str(raw)], cwd=root, env=env,
                       check=True)
        runs.append(raw)
    bad = merge(case, runs, output)
    for p in runs:
        p.unlink()
    work.rmdir()
    return bad


def main():
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("all", "case", "list", "worker"))
    parser.add_argument("args", nargs="*")
    parser.add_argument("--output", type=Path, default=root / "test" / "migration" / "reference" / "periodic" / "wpx")
    ns = parser.parse_args()
    if ns.action == "list":
        print("\n".join(f"{k}: {v['groups']}" for k, v in CASES.items()))
    elif ns.action == "worker":
        worker(ns.args[0], ns.args[1])
    elif ns.action == "case":
        raise SystemExit(1 if drive(ns.args[0], ns.output) else 0)
    else:
        bad = [b for case in CASES for b in drive(case, ns.output)]
        raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
