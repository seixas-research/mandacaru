"""Freeze the periodic end results (WP-F items 3 and 4) from the live Python tree.

Run from the repository root with the conda python and ``MANDACARU_PAW_PATH`` set::

    python test/migration/freeze_periodic_results.py all                    # every case, floors, constants, assemble
    python test/migration/freeze_periodic_results.py run [--cases a,b]      # SCF runs only (parts in --work)
    python test/migration/freeze_periodic_results.py assemble               # parts -> periodic_results.json
    python test/migration/freeze_periodic_results.py constants              # periodic_constants.json (no SCF)
    python test/migration/freeze_periodic_results.py list

Writes ``<output>/reference/periodic/periodic_results.json`` and ``periodic_constants.json`` (``--output`` defaults to
``test/migration``).  Every SCF runs in a process of its own after ``scripts/memguard.py``; the driver polls the child's RSS
and kills it above 1 GB or the time cap, and records the case as skipped with the measurement.  A float64 case runs twice at
2 threads and, when it takes under 10 s, once at 1 thread: the run-to-run differences are the measured floor.  Quantities are
in Hartree unless the key says ``_eV``.  SCF tolerance 1e-12, density 1e-10.  The tolerance classes of the plan:
T3 energy 1e-10 Ha, eigenvalues 1e-8 Ha; T4 float32 minus float64 energy 1e-6 eV.

The scratchpad numbers of PBE, r2SCAN and PBE+rVV10 were computed while a crystal read the ``lda-sr`` set for PBE: they stay
as ``prior_wrong_dataset`` (context, never a gate).  Periodic HSE06 is incomplete (no one-center screened term): its entries
are ``provisional_not_a_gate``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from freeze_periodic_cases import PRIOR_STILL_VALID, PRIOR_WRONG, case_ids  # noqa: E402

RSS_CAP_MB = 1000.0
T3_ENERGY_HA, T3_EIG_HA, T4_EV, OCC_TOL = 1e-10, 1e-8, 1e-6, 1e-10
HA_EV = 27.211386245988


def repo_root() -> Path:
    from freeze_stamp import repo_root as root
    return root()


def extension_currency() -> dict:
    """Crate sources newer than the extension: the frozen numbers are those of the BUILT extension, not of the working tree."""
    from datetime import datetime, timezone
    from freeze_stamp import extension_path
    root = repo_root()
    built = extension_path(root).stat().st_mtime
    newer = sorted((path.stat().st_mtime, str(path.relative_to(root))) for path in (root / "crates").rglob("*.rs")
                   if "target" not in path.parts and path.stat().st_mtime > built)
    stamp_time = lambda t: datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")
    return dict(extension_built=stamp_time(built), n_crate_files_newer=len(newer),
                newest=[stamp_time(newer[-1][0]), newer[-1][1]] if newer else None, files=[name for _, name in newer])


def guarded(command: list[str], timeout: float) -> dict:
    """Run ``command`` after memguard; poll its RSS (``ps``) and kill it above the cap or the timeout."""
    root = repo_root()
    guard = subprocess.run([sys.executable, str(root / "scripts" / "memguard.py"), "--need-gb", "2", "--wait", "120"],
                           capture_output=True, text=True, check=False)
    if guard.returncode != 0:
        return dict(skipped="memguard refused: " + guard.stdout.strip()[-200:])
    child = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    box = {"peak": 0.0, "killed": None}
    done = threading.Event()

    def watch():
        start = time.time()
        while not done.wait(1.0):
            listing = subprocess.run(["ps", "-o", "rss=", "-p", str(child.pid)], capture_output=True, text=True).stdout.strip()
            if listing:
                box["peak"] = max(box["peak"], int(listing) / 1024)
            if box["peak"] > RSS_CAP_MB:
                box["killed"] = f"RSS {box['peak']:.0f} MB above the {RSS_CAP_MB:.0f} MB cap"
            elif time.time() - start > timeout:
                box["killed"] = f"over the {timeout:.0f} s cap"
            if box["killed"]:
                child.kill()
                return

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    out, err = child.communicate()
    done.set()
    thread.join()
    line = [row for row in out.splitlines() if row.startswith("RESULT ")]
    if box["killed"] or not line:
        return dict(skipped=box["killed"] or f"no result, exit {child.returncode}: {err[-300:]}", polled_rss_peak_mb=box["peak"])
    return dict(result=json.loads(line[-1][7:]), polled_rss_peak_mb=box["peak"])


def run_cases(cases: list[str], work: Path, timeout: float) -> None:
    work.mkdir(parents=True, exist_ok=True)
    script = str(HERE / "freeze_periodic_cases.py")

    def one(case, precision, tag, threads):
        path = work / f"{case}__{precision}__{tag}.json"
        got = guarded([sys.executable, script, case, precision, str(threads)], timeout)
        path.write_text(json.dumps(got, sort_keys=True))
        result = got.get("result")
        print(f"{case:34s} {precision} {tag}: " + (f"{result['wall_s']} s, E={result.get('energy')}" if result and result["ok"]
              else (result or got).get("error", got.get("skipped"))), flush=True)
        return got.get("result")

    for case in cases:
        first = one(case, "float64", "a", 2)
        if first and first["ok"]:
            one(case, "float64", "b", 2)
            if first["wall_s"] < 10:
                one(case, "float64", "c", 1)
        one(case, "float32", "a", 2)


def part(work: Path, case: str, precision: str, tag: str):
    path = work / f"{case}__{precision}__{tag}.json"
    return json.loads(path.read_text()) if path.is_file() else None


def numbers(run) -> dict | None:
    if not run or "result" not in run or not run["result"]["ok"]:
        return None
    return run["result"]


def spread(a: dict, b: dict) -> dict:
    import numpy as np
    return dict(energy=abs(a["energy"] - b["energy"]), free_energy=abs(a["free_energy"] - b["free_energy"]),
                eigenvalues=float(np.abs(np.asarray(a["eigenvalues"]) - np.asarray(b["eigenvalues"])).max()),
                moment=abs(a["moment"] - b["moment"]), fermi_level=abs(a["fermi_level_reported"] - b["fermi_level_reported"]), n_iter=[a["n_iter"], b["n_iter"]])


def assemble_case(case: str, work: Path, constants: dict, frozen: list) -> dict:
    import numpy as np
    from freeze_periodic_cases import definition
    atoms, _, meta = definition(case)
    record = dict(meta={k: v for k, v in meta.items() if k != "species"}, symbols=atoms.get_chemical_symbols())
    runs = {name: part(work, case, *name.split("/")) for name in ("float64/a", "float64/b", "float64/c", "float32/a")}
    for precision in ("float64", "float32"):
        run = runs[f"{precision}/a"]
        got = numbers(run)
        if got is None:
            record[precision] = dict(unavailable=(run or {}).get("skipped") or ((run or {}).get("result") or {}).get("error", "not run"))
            continue
        keep = ("E_eV", "energy", "spin_orbit_energy", "free_energy", "extrapolated_energy", "eigenvalues", "eigenvalues_shape", "occupation_sum",
                "expected_electrons", "moment", "n_iter", "converged", "n_spins", "spinor", "kpoints_fractional", "weights",
                "terms", "last_history", "context_peak_bytes", "wall_s", "rss_peak_mb")
        entry = {k: got.get(k) for k in keep}
        entry["fermi_level"] = got["fermi_level_reported"] if meta.get("metal") else None
        entry["fermi_level_reported_nonunique"] = None if meta.get("metal") else got["fermi_level_reported"]
        symbols = record["symbols"]
        constant = sum(constants[f"{meta['dataset']}/{s}"]["constant"] for s in symbols)
        entry["checks"] = dict(
            converged=got["converged"], occupation_sum_error=abs(got["occupation_sum"] - got["expected_electrons"]),
            occupation_sum_pass=abs(got["occupation_sum"] - got["expected_electrons"]) < OCC_TOL,
            terms_constant_minus_species_sum=got["terms"]["constant"] - constant,
            weights_sum_error=abs(sum(got["weights"]) - 1.0))
        record[precision] = entry
        if precision == "float64":
            record["stamp"] = got["stamp"]
    f64, f32 = numbers(runs["float64/a"]), numbers(runs["float32/a"])
    floors = {tag: spread(f64, numbers(runs[f"float64/{tag}"])) for tag in ("b", "c") if f64 and numbers(runs[f"float64/{tag}"])}
    record["floor"] = dict(same_threads_repeat=floors.get("b"), one_thread=floors.get("c"),
                           note="run-to-run differences of the float64 SCF (b: repeat at 2 threads, c: 1 thread, only if the case took < 10 s)")
    if f64 and f32:
        d_ev = f32["E_eV"] - f64["E_eV"]
        record["float32_minus_float64"] = dict(
            dE_eV=d_ev, T4_pass=abs(d_ev) <= T4_EV, max_eigenvalue_Ha=float(np.abs(np.asarray(f32["eigenvalues"]) - np.asarray(f64["eigenvalues"])).max()),
            electron_count_error=abs(f32["occupation_sum"] - f32["expected_electrons"]), n_iter=[f64["n_iter"], f32["n_iter"]])
    if "frozen_index" in meta and f64:
        old = frozen[meta["frozen_index"]]
        e_ref, eig_ref = old["float64"], np.asarray(old["float64"]["eigenvalues"])
        deig = float(np.abs(np.asarray(f64["eigenvalues"]).ravel() - eig_ref).max())
        record["vs_frozen_periodic_energies_json"] = dict(
            old_extension_sha256_prefix="39cf62c7", dE_Ha=f64["energy"] - e_ref["energy"], dF_Ha=f64["free_energy"] - e_ref["free_energy"],
            dE_eV=f64["E_eV"] - e_ref["E_eV"], max_eigenvalue_Ha=deig,
            dFermi_level_Ha=(abs(f64["fermi_level_reported"] - e_ref["fermi_level"]) if meta.get("metal") else None), n_iter=[f64["n_iter"], e_ref["n_iter"]],
            T3_pass=abs(f64["energy"] - e_ref["energy"]) <= T3_ENERGY_HA and abs(f64["free_energy"] - e_ref["free_energy"]) <= T3_ENERGY_HA
            and deig <= T3_EIG_HA)
    if case in PRIOR_WRONG or case in PRIOR_STILL_VALID:
        prior = dict(PRIOR_WRONG.get(case) or PRIOR_STILL_VALID[case])
        key = "prior_wrong_dataset" if case in PRIOR_WRONG else "prior_still_valid"
        if f64:
            prior["E_eV_new_minus_prior"] = f64["E_eV"] - prior["E_eV"]
        prior["note"] = ("scratchpad number of extension 39cf62c7: crystals then read lda-sr for PBE; the change is the dataset fix, "
                         "not a regression" if key == "prior_wrong_dataset" else "set named explicitly: not affected, compare")
        record[key] = prior
    if meta.get("provisional_not_a_gate"):
        record["provisional_not_a_gate"] = "periodic HSE06 has no one-center screened term: do not gate a test on this entry"
    return record


def generic_k_count(fractional) -> int:
    """The k with k != -k modulo a reciprocal vector (a component outside {0, 1/2}): the sign of k.R matters there."""
    import numpy as np
    k = np.asarray(fractional, dtype=float)
    return int(sum(1 for row in k if np.any(np.abs(2 * row - np.round(2 * row)) > 1e-9)))


def cross_checks(cases: dict) -> dict:
    """Relations between cases that must hold for the frozen numbers to mean anything (non-vacuity of the set)."""
    import numpy as np
    from mandacaru import _native
    out = {}
    f64 = lambda name: (cases[name].get("float64") or {})
    out["generic_k"] = {name: generic_k_count(f64(name)["kpoints_fractional"]) for name in cases
                        if "kpoints_fractional" in f64(name)}
    required = [n for n in cases if not n.startswith(("skew_", "companion_al", "companion_si_so")) and "hse06_full" not in n
                and n != "extra_si_hse06"]
    out["generic_k_every_required_case_has_one"] = all(out["generic_k"].get(n, 0) > 0 for n in required)
    out["generic_k_note"] = "count of irreducible k with k != -k mod G; zero means the case is blind to the sign of k.R"
    tuple_mesh = _native.Context(threads=1).kmesh((3, 3, 3), gamma=False, time_reversal=False)["full"]
    dict_mesh = _native.Context(threads=1).kmesh((3, 3, 3), gamma=True, time_reversal=False)["full"]
    out["mesh_3x3x3_tuple_equals_gamma_dict"] = bool(np.allclose(np.sort(np.round(tuple_mesh, 12), axis=0), np.sort(np.round(dict_mesh, 12), axis=0)))
    k3 = f64("lda_si_sz_k3_spacing0.15")
    if k3:
        out["si_k3_spacing0.15_vs_memory_table_json"] = dict(dE_eV=k3["E_eV"] - cases["lda_si_sz_k3_spacing0.15"]["meta"]["memory_table_E_eV"],
                                                       note="memory_table.json (extension 39cf62c7) used kpts=(3, 3, 3), default SCF options")
    so, sr = f64("companion_si_so"), f64("companion_si_so_scalar")
    if so and sr:
        gamma = [i for i, k in enumerate(so["kpoints_fractional"]) if max(abs(x) for x in k) < 1e-9][0]
        spinor_bands = np.asarray(so["eigenvalues"])[0, gamma]
        scalar_bands = np.asarray(sr["eigenvalues"])[0, gamma]
        out["si_so_vs_scalar"] = dict(
            spinor=so["spinor"], scalar_spinor=sr["spinor"], dE_Ha=so["energy"] - sr["energy"],
            spin_orbit_energy_Ha=so["spin_orbit_energy"], spin_orbit_energy_floor_Ha=cases["companion_si_so"]["floor"]["same_threads_repeat"]["energy"],
            gamma_top_valence_spread_meV=dict(spinor=float(np.ptp(spinor_bands[2:8]) * HA_EV * 1e3), scalar=float(np.ptp(scalar_bands[1:4]) * HA_EV * 1e3)),
            spinor_bands=list(np.asarray(so["eigenvalues"]).shape), scalar_bands=list(np.asarray(sr["eigenvalues"]).shape),
            note="dE_Ha mixes the lda-fr and lda-sr datasets (different frozen cores): it is NOT the spin-orbit effect. The effect is "
                 "spin_orbit_energy_Ha (far above its floor). No Delta_so is claimed: the 10^3 grid breaks the cubic symmetry (the scalar "
                 "Gamma_25' triplet splits 2+1, spread above), so the levels are not those of Si; the (2,M,2,M) block is frozen with the snapshot")
    return out


def assemble(work: Path, output: Path) -> dict:
    from freeze_periodic_constants import constants as compute_constants
    from freeze_stamp import stamp
    root = repo_root()
    constants = json.loads((output / "periodic_constants.json").read_text())["species"] if (output / "periodic_constants.json").is_file() \
        else compute_constants()
    frozen = json.loads((root / "test" / "migration" / "reference" / "precision" / "periodic_energies.json").read_text())["cases"]
    cases = {case: assemble_case(case, work, constants, frozen) for case in case_ids()}
    shas = {r["stamp"]["extension"]["sha256"] for r in cases.values() if "stamp" in r}
    root_stamp = stamp(extra={"script": "freeze_periodic_results.py", "extension_currency": extension_currency(), "extension_shas_in_parts": sorted(shas),
                              "tolerances": dict(T3_energy_Ha=T3_ENERGY_HA, T3_eigenvalues_Ha=T3_EIG_HA, T4_eV=T4_EV)},
                       datasets=sorted({d for r in cases.values() for d in r.get("stamp", {}).get("datasets_sha256", {})}))
    return dict(stamp=root_stamp, units="Hartree unless the key ends in _eV", scf=dict(tolerance=1e-12, density_tolerance=1e-10),
                cross_checks=cross_checks(cases), cases=cases)


def write_constants(output: Path) -> dict:
    from freeze_periodic_constants import constants
    from freeze_stamp import stamp
    species = constants()
    payload = dict(stamp=stamp(datasets=sorted(r["file"] for r in species.values()), extra={"script": "freeze_periodic_constants.py"}),
                   units="Hartree", rust_comparison="Species::dft_constant (paw/molecular/data.rs) has no binding that reports it; "
                   "the Rust test reads this file and compares to 1e-9", species=species)
    (output / "periodic_constants.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    return payload


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("all", "run", "assemble", "constants", "list"))
    parser.add_argument("--output", type=Path, default=root / "test" / "migration" / "reference" / "periodic")
    parser.add_argument("--work", type=Path, default=Path(tempfile.gettempdir()) / "mandacaru_periodic_results")
    parser.add_argument("--cases", default="")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    if "MANDACARU_PAW_PATH" not in os.environ:
        raise SystemExit("set MANDACARU_PAW_PATH on the command line")
    if args.command == "list":
        print("\n".join(case_ids()))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command in ("all", "constants"):
        write_constants(args.output)
    if args.command in ("all", "run"):
        run_cases(args.cases.split(",") if args.cases else case_ids(), args.work, args.timeout)
    if args.command in ("all", "assemble"):
        path = args.output / "periodic_results.json"
        path.write_text(json.dumps(assemble(args.work, args.output), indent=1, sort_keys=True) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
