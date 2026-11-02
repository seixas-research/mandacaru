"""Freeze item 5: the Python memory baselines of the periodic route (M1, M2, M6, and the M7 spread).

Run from the repository root with the conda python and ``MANDACARU_PAW_PATH`` set::

    python test/migration/freeze_periodic_memory.py all [--output DIR]       # every run below, then the table
    python test/migration/freeze_periodic_memory.py run --cases m1,m2        # only some groups
    python test/migration/freeze_periodic_memory.py assemble

Cases (the protocol of ``reference/precision/memory_table.json``: LDA, SZ, 2 threads, a 1800 MiB Context limit, the default
SCF options, ``tracemalloc`` on): ``m1`` Al cubic 2x1x1 (8 atoms) Gamma grid_spacing=0.25; ``m2`` Si primitive, ``kpts=(3, 3, 3)``,
grid_spacing=0.15; ``m6`` Al cubic (4 atoms) DZP Gamma grid_spacing=0.15 with the real k count and the psi size printed.  M7 is the float64 M1 case
run twice more for the noise spread.  The multi-k variant of M6 is not run unless ``--multi-k`` is given.  M4 (32 atoms) is
NEVER run: no Python periodic run of that size is made.

Each run is a fresh process after ``scripts/memguard.py``.  ``rss_baseline`` is ``ru_maxrss`` after ``import mandacaru``,
the atoms and the calculator construction, before ``run()`` (the post-import baseline of the plan); ``rss_delta`` is the peak
``ru_maxrss`` minus it (MiB, 2^20 bytes).  The traced peak is a REPORT (tracemalloc cannot see Rust).  A run that would pass
1 GB of RSS (polled with ``ps``) or 180 s is killed and recorded as skipped with what was measured.  Every file is stamped.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MIB = 1024 ** 2
CASES = {  # id: (symbol, cubic, repeat, size, kpts, grid_spacing)
    "m1": ("Al", True, (2, 1, 1), "SZ", None, 0.25),
    "m2": ("Si", False, (1, 1, 1), "SZ", (3, 3, 3), 0.15),
    "m6": ("Al", True, (1, 1, 1), "DZP", None, 0.15),
    "m6k": ("Al", True, (1, 1, 1), "DZP", {"size": (2, 2, 2), "gamma": True}, 0.15)}
#: ``(case, precision, tracemalloc, tag)``; ``r*`` are the M7 repeats.
PLAN = [("m1", "float64", True, "a"), ("m1", "float64", True, "r1"), ("m1", "float64", True, "r2"), ("m1", "float32", True, "a"),
        ("m1", "float64", False, "untraced"), ("m2", "float64", True, "a"), ("m2", "float32", True, "a"),
        ("m2", "float64", False, "untraced"), ("m6", "float64", True, "a"), ("m6", "float32", True, "a"),
        ("m6", "float64", False, "untraced")]
#: Frozen Python numbers of ``memory_table.json`` (extension 39cf62c7, 2026-10-29), MiB.
OLD = {("m1", "float64"): dict(rss_delta=319.265625, traced=227.2001895904541, ctx_peak=34.84938621520996, ctx_used=26.687918663024902),
       ("m1", "float32"): dict(rss_delta=326.578125, traced=225.19199657440186, ctx_peak=35.2961950302124, ctx_used=24.687918663024902),
       ("m2", "float64"): dict(rss_delta=198.96875, traced=182.67385005950928, ctx_peak=27.4107723236084, ctx_used=18.6533260345459),
       ("m2", "float32"): dict(rss_delta=208.046875, traced=182.6746768951416, ctx_peak=23.611364364624023, ctx_used=14.362310409545898)}


def rss_mib() -> float:
    scale = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale / MIB


def worker(case: str, precision: str, traced: bool) -> dict:
    import tracemalloc
    from ase.build import bulk
    from freeze_stamp import stamp
    from mandacaru import Mandacaru
    symbol, cubic, repeat, size, kpts, grid_spacing = CASES[case]
    atoms = bulk(symbol, "fcc" if symbol == "Al" else "diamond", a=4.05 if symbol == "Al" else 5.43, cubic=cubic).repeat(repeat)
    options = dict(method="dft", xc="lda", basis={"name": "PAW-LCAO", "size": size}, grid_spacing=grid_spacing, precision=precision,
                   resources={"threads": 2, "memory_limit_bytes": 1800 * MIB})
    if kpts:
        options["kpts"] = kpts
    out = dict(case=case, precision=precision, tracemalloc=traced, natoms=len(atoms), options={k: repr(v) for k, v in options.items()},
               stamp=stamp(datasets=[str(Path(os.environ["MANDACARU_PAW_PATH"]) / "lda-sr" / f"{symbol}.parquet")],
                           extra={"case": case, "precision": precision}))
    try:
        calculator = Mandacaru(**options)
        atoms.calc = calculator
        out["rss_baseline_mib"] = rss_mib()
        if traced:
            tracemalloc.start()
        start = time.time()
        energy = float(atoms.get_potential_energy())
        wall = time.time() - start
        peak = rss_mib()
        sn = calculator.result.scf.snapshot()
        weights, eig = sn["weights"], sn["eigenvalues"]
        grid_points = int(sn["density"].shape[-1])
        out.update(
            ok=True, E_eV=energy, n_iter=int(sn["n_iterations"]), converged=bool(sn["converged"]), wall_s=round(wall, 1),
            rss_peak_mib=peak, rss_delta_mib=peak - out["rss_baseline_mib"],
            traced_peak_mib=(tracemalloc.get_traced_memory()[1] / MIB if traced else None),
            context_peak_mib=calculator.context.memory_peak_bytes / MIB, context_used_mib=calculator.context.memory_used_bytes / MIB,
            irreducible_k=len(weights), bands=int(eig.shape[-1]), grid_points=grid_points,
            psi_bytes_complex128=len(weights) * int(eig.shape[-1]) * grid_points * 16,
            psi_mib_complex128=len(weights) * int(eig.shape[-1]) * grid_points * 16 / MIB, kmesh=sn["fractional_kpoints"].tolist())
    except Exception as exc:
        out.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:300]}", rss_peak_mib=rss_mib())
    return out


def run(plan, work: Path) -> None:
    from freeze_periodic_results import guarded
    work.mkdir(parents=True, exist_ok=True)
    for case, precision, traced, tag in plan:
        got = guarded([sys.executable, __file__, "worker", case, precision, "1" if traced else "0"], timeout=180.0)
        (work / f"{case}__{precision}__{tag}.json").write_text(json.dumps(got, sort_keys=True))
        r = got.get("result")
        print(f"{case} {precision} {tag}: " + (f"rss_delta {r['rss_delta_mib']:.1f} ctx_peak {r['context_peak_mib']:.1f} traced "
              f"{r['traced_peak_mib']} wall {r['wall_s']} k {r['irreducible_k']} polled {got['polled_rss_peak_mb']:.0f}" if r and r["ok"]
              else (r or got).get("error", got.get("skipped"))), flush=True)


def assemble(work: Path) -> dict:
    from freeze_periodic_results import extension_currency
    from freeze_stamp import stamp
    rows, stamps = {}, set()
    for path in sorted(work.glob("*.json")):
        got = json.loads(path.read_text())
        entry = dict(got.get("result") or {})
        if "skipped" in got:
            entry = dict(skipped=got["skipped"])
        entry["polled_rss_peak_mb_total_process"] = got.get("polled_rss_peak_mb")
        rows[path.stem] = entry
        if "stamp" in entry:
            stamps.add(entry["stamp"]["extension"]["sha256"])
    spread = {}
    for case in ("m1",):
        runs = [rows[k] for k in (f"{case}__float64__a", f"{case}__float64__r1", f"{case}__float64__r2") if k in rows and rows[k].get("ok")]
        if len(runs) >= 2:
            spread[case] = {key: dict(values=[r[key] for r in runs], range=max(r[key] for r in runs) - min(r[key] for r in runs))
                            for key in ("rss_delta_mib", "traced_peak_mib", "context_peak_mib", "context_used_mib", "E_eV")}
    old = {f"{c}__{p}": v for (c, p), v in OLD.items()}
    return dict(stamp=stamp(extra={"script": "freeze_periodic_memory.py", "extension_currency": extension_currency(), "extension_shas_in_parts": sorted(stamps)}), units="MiB (2^20 bytes)",
                runs=rows, m7_noise_spread=spread, python_old_baseline_memory_table_json=old,
                notes=["M4 (32 atoms) is not baselined: no Python periodic run of that size is made",
                       "RSS is ru_maxrss; a macOS process can also hold compressed pages that RSS hides (polled_rss_peak_mb_total_process "
                       "is the RSS polled with ps by the driver, baseline included)"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("all", "run", "assemble", "worker"))
    parser.add_argument("args", nargs="*")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--work", type=Path, default=Path(tempfile.gettempdir()) / "mandacaru_periodic_memory")
    parser.add_argument("--cases", default="")
    parser.add_argument("--multi-k", action="store_true")
    args = parser.parse_args()
    if args.command == "worker":
        print("RESULT " + json.dumps(worker(args.args[0], args.args[1], args.args[2] == "1"), allow_nan=False))
        return
    from freeze_stamp import repo_root
    output = args.output or repo_root() / "test" / "migration" / "reference" / "periodic"
    wanted = set(args.cases.split(",")) if args.cases else {"m1", "m2", "m6"}
    plan = [p for p in PLAN if p[0] in wanted]
    if args.multi_k:
        plan += [("m6k", "float64", True, "a"), ("m6k", "float32", True, "a")]
    if args.command in ("all", "run"):
        run(plan, args.work)
    if args.command in ("all", "assemble"):
        output.mkdir(parents=True, exist_ok=True)
        (output / "periodic_memory.json").write_text(json.dumps(assemble(args.work), indent=1, sort_keys=True) + "\n")
        print(f"wrote {output / 'periodic_memory.json'}")


if __name__ == "__main__":
    main()
