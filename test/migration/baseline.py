"""Run isolated migration references and compare their scientific outputs.

Commands: export-h2 --output PATH; run --output PATH; compare REFERENCE CANDIDATE.
Each sample has its own interpreter and working directory. No cloud jobs run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import signal
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if __package__:
    from .inventory import digest_manifest, git_revision, source_manifest
else:
    from inventory import digest_manifest, git_revision, source_manifest

CASES_PATH = HERE / "cases.json"
FIXTURE = HERE / "reference" / "h2_hamiltonian.json"
THREAD_ENV = ("MANDACARU_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "RAYON_NUM_THREADS")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def fingerprint(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def cases() -> list[dict]:
    return read_json(CASES_PATH)["cases"]


def controlled_env(backend: str, threads: int, work: Path) -> dict[str, str]:
    env = dict(os.environ)
    # Do not let training records or alternate native libraries silently change
    # what a named workload measures.
    for key in ("MANDACARU_PROPOSAL_DATA", "MANDACARU_INTEGRALS_LIB"):
        env.pop(key, None)
    env.update({key: str(threads) for key in THREAD_ENV})
    env.update(MANDACARU_BACKEND=backend, MPLBACKEND="Agg",
               MPLCONFIGDIR=str(work / "matplotlib"), PYTHONHASHSEED="0",
               JAX_ENABLE_X64="true")
    return env


def dataset_inputs(case: dict) -> tuple[list[dict], list[str]]:
    inputs, missing = [], []
    location = os.environ.get("MANDACARU_PAW_PATH")
    for relative in case.get("datasets", []):
        path = Path(location) / relative if location else None
        sha = fingerprint(path) if path else None
        inputs.append({"family": "paw-lcao", "file": relative, "sha256": sha})
        if sha is None:
            missing.append("MANDACARU_PAW_PATH/" + relative)
    return inputs, missing


def environment() -> dict:
    import psutil
    from threadpoolctl import threadpool_info
    versions = {}
    for distribution in sorted(importlib.metadata.distributions(),
                               key=lambda d: d.metadata["Name"].lower()):
        versions[distribution.metadata["Name"]] = distribution.version
    manifest = source_manifest()
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "physical_cpu_count": psutil.cpu_count(logical=False),
            "physical_memory_bytes": psutil.virtual_memory().total, "packages": versions,
            "git_revision": git_revision(), "source_sha256": digest_manifest(manifest),
            "thread_environment": {key: os.environ.get(key) for key in THREAD_ENV},
            "blas": [{k: v for k, v in item.items() if k != "filepath"}
                     for item in threadpool_info()]}


def numerical(value):
    import numpy as np
    array = np.asarray(value)
    if np.iscomplexobj(array):
        return {"real": array.real.tolist(), "imag": array.imag.tolist()}
    return array.tolist()


def molecular_calculator(case: dict):
    from ase import Atoms
    from mandacaru import Mandacaru
    atoms = Atoms(case["symbols"], positions=case["positions"], cell=case["cell"])
    atoms.calc = Mandacaru(**case["options"], trace=False,
                           references=False, dump_buffers=False)
    return atoms


def export_h2(path: Path) -> None:
    from mandacaru.core.serialization import save_hamiltonian
    case = next(c for c in cases() if c["id"] == "h2_rhf")
    atoms = molecular_calculator(case)
    atoms.get_potential_energy()
    result = atoms.calc.result
    if not result.success:
        raise RuntimeError("Cannot export an unconverged H2 reference")
    save_hamiltonian(path, result.qubit_hamiltonian(), format="json",
                     mapping="jordan_wigner", num_particles=result.num_particles,
                     n_spatial_orbitals=result.n_spatial_orbitals,
                     metadata={"migration_case": case, "source_sha256":
                               digest_manifest(source_manifest())})


def algebra(case: dict) -> dict:
    import numpy as np
    from mandacaru.core.mapping import Fermion
    from mandacaru.core.matrix_free import PauliOperator
    from mandacaru.core.sector import ParticleSector
    rng = np.random.default_rng(case["seed"])
    h = np.array([[-1.0, 0.13 + 0.07j], [0.13 - 0.07j, -0.3]])
    fermion = Fermion.from_spatial_integrals(h)
    values, checks = {}, {}
    for mapping in ("jordan_wigner", "parity", "bravyi_kitaev", "parity_reduced"):
        operator = fermion.map_to_qubits(mapping, num_particles=(1, 1))
        matrix = operator.to_matrix()
        state = rng.normal(size=matrix.shape[0]) + 1j * rng.normal(size=matrix.shape[0])
        state /= np.linalg.norm(state)
        product = PauliOperator(operator, cache_bytes=0) @ state
        sector = ParticleSector(operator.num_qubits, (1, 1), mapping=mapping)
        restricted = sector.restrict(operator).toarray()
        values[mapping] = {
            "matrix": numerical(matrix), "matvec": numerical(product),
            "sector_eigenvalues": numerical(np.linalg.eigvalsh(restricted))}
        checks[mapping + ".hermitian"] = bool(np.allclose(matrix, matrix.conj().T, atol=1e-12))
        checks[mapping + ".matvec"] = bool(np.allclose(product, matrix @ state, atol=1e-12))
        checks[mapping + ".conserves"] = bool(sector.conserves(operator))
    return {"values": values, "checks": checks, "contract": {"seed": case["seed"],
            "qubit_order": "qubit 0 is most significant", "energy_unit": "Ha"}}


def calculate(case: dict) -> dict:
    import numpy as np
    from mandacaru import Mandacaru
    from mandacaru.core.serialization import load_hamiltonian
    if case["kind"] == "algebra":
        return algebra(case)
    diagnostics = {}
    common = {"trace": False, "references": False,
              "dump_buffers": False}
    if case["kind"] == "cached":
        calc = Mandacaru(**case["options"], **common, load_hamiltonian=str(FIXTURE),
                         optimizer={"method": "L-BFGS", "maxiter": 500, "tol": 1e-12})
        result = calc.run()
        record = load_hamiltonian(FIXTURE)
        converged = result.converged if case["options"]["method"] == "adapt-vqe" else result.success
        checks = {"converged": bool(converged)}
        values = {"energy_ev": float(result.optimal_energy),
                  "reference_energy_ev": float(result.reference_energy)}
        # These references are small enough for a separate dense exact check.
        from mandacaru.core.sector import ParticleSector
        sector = ParticleSector(record.num_qubits, record.num_particles, mapping=record.mapping)
        exact_ev = float(np.linalg.eigvalsh(sector.restrict(record.hamiltonian).toarray()).min()) * 27.211386245988
        values["exact_sector_energy_ev"] = exact_ev
        checks["reaches_exact_energy"] = abs(values["energy_ev"] - exact_ev) < 1e-6
        state = calc._converged_state(calc.solver)
        active_sector = getattr(calc.solver, "_sector", None)
        indices = active_sector.indices if active_sector is not None else np.arange(state.size)
        norm = float(np.vdot(state, state).real)
        count = float(np.abs(state) ** 2 @ np.bitwise_count(indices.astype(np.uint64)))
        values.update(state_norm=norm, particle_count=count,
                      state_projector=numerical(np.outer(state, state.conj())))
        checks.update(normalized=abs(norm - 1.0) < 1e-10,
                      particle_number=abs(count - sum(record.num_particles)) < 1e-9)
        contract = {"method": calc.method, "energy_unit": result.energy_unit,
                    "n_qubits": calc.n_qubits, "num_particles": list(calc.num_particles)}
    else:
        if case["kind"] == "crystal":
            from ase.build import bulk
            atoms = bulk(case["element"], case["structure"], a=case["a"])
            atoms.calc = Mandacaru(**case["options"], **common)
        else:
            atoms = molecular_calculator(case)
        calc = atoms.calc
        values = {"energy_ev": float(atoms.get_potential_energy())}
        result = calc.result
        checks = {"converged": bool(result.success)}
        checks["ase_energy_units"] = abs(float(result.in_units("eV")) - values["energy_ev"]) < 1e-7
        contract = {"method": calc.method, "energy_unit": result.energy_unit,
                    "num_particles": list(result.num_particles) if hasattr(result, "num_particles") else None}
        scf = result.scf
        for name in ("mo_energies", "orbital_energies", "eps_alpha", "eps_beta"):
            if hasattr(scf, name) and getattr(scf, name) is not None:
                values[name + "_ha"] = numerical(getattr(scf, name))
        if case.get("integrals"):
            integrals = calc.solver._gradient_context["integrals"]
            values["overlap"] = numerical(integrals.overlap())
            values["one_body_ha"] = numerical(integrals.one_body())
            values["two_body_ha"] = numerical(integrals.two_body())
        if case["kind"] == "crystal":
            values["fermi_level_ev"] = float(calc.get_fermi_level())
            values["kpoint_weights"] = numerical(calc.get_k_point_weights())
            values["eigenvalues_ev"] = numerical([calc.get_eigenvalues(kpt=k)
                                                  for k in range(len(calc.get_k_point_weights()))])
            checks["normalized_weights"] = bool(abs(sum(calc.get_k_point_weights()) - 1.0) < 1e-12)
        if case.get("forces"):
            values["forces_ev_per_angstrom"] = numerical(atoms.get_forces())
            diagnostics["result_available_after_forces"] = calc.result is not None
            diagnostics["force_breakdown_available_after_forces"] = calc.force_result is not None
            values["free_energy_ev"] = float(atoms.get_potential_energy(force_consistent=True))
        if fd := case.get("finite_difference"):
            atom, axis, step = fd["atom"], fd["axis"], fd["step"]
            position = float(atoms.positions[atom, axis])
            atoms.positions[atom, axis] = position + step
            plus = float(atoms.get_potential_energy(force_consistent=True))
            atoms.positions[atom, axis] = position - step
            minus = float(atoms.get_potential_energy(force_consistent=True))
            atoms.positions[atom, axis] = position
            force = -(plus - minus) / (2 * step)
            error = abs(force - values["forces_ev_per_angstrom"][atom][axis])
            values.update(finite_difference_force_ev_per_angstrom=force, force_fd_error=error)
            checks["force_is_energy_derivative"] = error <= fd["atol"]
        if case.get("repeat_geometry"):
            # A -> B -> A checks state invalidation and reuse on an ASE trajectory.
            atoms.positions[1, 2] += 0.02
            values["displaced_energy_ev"] = float(atoms.get_potential_energy())
            atoms.positions[1, 2] -= 0.02
            values["returned_energy_ev"] = float(atoms.get_potential_energy())
            checks["geometry_roundtrip"] = abs(values["returned_energy_ev"] - values["energy_ev"]) < 1e-7
    return {"values": values, "checks": checks, "contract": contract,
            "diagnostics": {"num_evaluations": int(result.num_evaluations), **diagnostics}}


def worker(case_id: str, output: Path) -> int:
    # Import before the compute timer; process_wall_s separately includes imports.
    import numpy
    from mandacaru import Mandacaru
    from mandacaru.integrals import ensure_backend
    case = next(c for c in cases() if c["id"] == case_id)
    start = time.perf_counter()
    sample = {}
    try:
        status = ensure_backend(build=False)
        start = time.perf_counter()
        sample = calculate(case)
        sample["status"] = "ok" if all(sample["checks"].values()) else "failed"
    except Exception as exc:
        sample.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    sample["compute_wall_s"] = time.perf_counter() - start
    usage = resource.getrusage(resource.RUSAGE_SELF)
    sample.update(peak_rss_bytes=int(usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)),
                  user_cpu_s=usage.ru_utime, system_cpu_s=usage.ru_stime,
                  environment=environment())
    from mandacaru.integrals import check_backend
    status = check_backend()
    native = Path(status.path) if status.path else None
    sample["native_backend"] = {"available": status.available, "message": status.message,
                                "is_rust": status.is_rust, "threads": status.n_threads,
                                "library_sha256": fingerprint(native) if native else None}
    try:
        write_json(output, sample)
    except ValueError:
        sample["status"] = "failed"
        write_json(output, {"status": "failed", "error": "Nonfinite scientific output or metadata"})
    return 0 if sample["status"] == "ok" else 1


def run(args) -> int:
    selected = [c for c in cases() if (c["id"] in args.case if args.case else
                                      c["tier"] in (["quick", "extended"] if args.tier == "all" else [args.tier]))]
    unknown = set(args.case or ()) - {c["id"] for c in cases()}
    if unknown:
        raise ValueError(f"Unknown cases: {sorted(unknown)}")
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "backend_policy": args.backend, "threads": args.threads,
              "cases_sha256": fingerprint(CASES_PATH), "fixture_sha256": fingerprint(FIXTURE),
              "runner_sha256": fingerprint(Path(__file__)),
              "inventory_tool_sha256": fingerprint(HERE / "inventory.py"),
              "cases": {}}
    output = args.output.resolve()
    report["runtime_artifact_root"] = (str(output.parent.relative_to(ROOT))
                                       if output.parent.is_relative_to(ROOT) else str(output.parent))
    work_root = output.parent / (output.stem + "-work")
    for case in selected:
        datasets, missing = dataset_inputs(case)
        entry = {"definition": case, "dataset_inputs": datasets, "samples": []}
        report["cases"][case["id"]] = entry
        if missing:
            entry.update(status="skipped", reason="Missing " + ", ".join(missing))
        else:
            for repeat in range(args.repeat):
                work = work_root / case["id"] / str(repeat)
                work.mkdir(parents=True, exist_ok=True)
                sample_path = work / "sample.json"
                # Remove a previous result so a crashed rerun cannot reuse it.
                sample_path.unlink(missing_ok=True)
                start = time.perf_counter()
                with (work / "stdout.log").open("w") as log:
                    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                                "_worker", case["id"], str(sample_path)],
                                               cwd=work, env=controlled_env(args.backend, args.threads, work),
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        code = process.wait(timeout=args.timeout)
                        sample = read_json(sample_path) if sample_path.exists() else {
                            "status": "failed", "error": "Worker produced no result"}
                        if code != 0:
                            sample["status"] = "failed"
                        sample["exit_code"] = code
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        sample = {"status": "timeout", "error": f"Exceeded {args.timeout} seconds"}
                sample["process_wall_s"] = time.perf_counter() - start
                sample["log"] = str((work / "stdout.log").relative_to(output.parent))
                entry["samples"].append(sample)
                print(f"{case['id']} [{repeat + 1}/{args.repeat}]: {sample['status']}", flush=True)
            entry["status"] = "ok" if all(s["status"] == "ok" for s in entry["samples"]) else "failed"
            entry["measurements"] = {}
            for key in ("compute_wall_s", "process_wall_s", "peak_rss_bytes", "user_cpu_s", "system_cpu_s"):
                measured = [s[key] for s in entry["samples"] if key in s]
                if measured:
                    entry["measurements"][key] = {"median": statistics.median(measured),
                                                  "min": min(measured), "max": max(measured)}
        write_json(output, report)
    return 0 if all(e["status"] == "ok" for e in report["cases"].values()) else 1


def differences(expected, actual, *, atol: float, rtol: float, path="values") -> list[str]:
    """Compare finite numbers recursively and require identical structure."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        errors = []
        if expected.keys() != actual.keys():
            errors.append(f"{path}: field set changed")
        for key in sorted(expected.keys() & actual.keys()):
            errors.extend(differences(expected[key], actual[key], atol=atol, rtol=rtol, path=f"{path}.{key}"))
        return errors
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: array length changed"]
        return [error for i, (left, right) in enumerate(zip(expected, actual))
                for error in differences(left, right, atol=atol, rtol=rtol, path=f"{path}[{i}]")]
    if type(expected) in (int, float) and type(actual) in (int, float):
        return [] if math.isfinite(expected) and math.isfinite(actual) and math.isclose(
            expected, actual, abs_tol=atol, rel_tol=rtol) else [f"{path}: {actual!r} != {expected!r}"]
    return [] if type(expected) is type(actual) and expected == actual else [f"{path}: value or type changed"]


def compare(reference: dict, candidate: dict) -> list[str]:
    errors = []
    if not reference["cases"]:
        errors.append("empty reference cannot certify compatibility")
    for key in ("schema_version", "cases_sha256", "fixture_sha256"):
        if reference.get(key) != candidate.get(key):
            errors.append(f"{key}: incompatible inputs/schema")
    if reference["cases"].keys() != candidate["cases"].keys():
        errors.append("case set changed; partial runs cannot certify the reference")
    for name in sorted(reference["cases"].keys() & candidate["cases"].keys()):
        left, right = reference["cases"][name], candidate["cases"][name]
        if left["definition"] != right["definition"] or left["dataset_inputs"] != right["dataset_inputs"]:
            errors.append(f"{name}: workload or dataset changed")
        if left["status"] != "ok" or right["status"] != "ok" or not left["samples"] or not right["samples"]:
            errors.append(f"{name}: failed, skipped or empty reference/candidate")
            continue
        expected = left["samples"][0]
        for sample in [*left["samples"], *right["samples"]]:
            if sample.get("status") != "ok" or not sample.get("checks") or not all(
                    value is True for value in sample["checks"].values()):
                errors.append(f"{name}: a sample failed its invariant checks")
                continue
            if expected["contract"] != sample["contract"] or expected["checks"].keys() != sample["checks"].keys():
                errors.append(f"{name}: contract or invariant set changed")
            errors.extend(f"{name}: {error}" for error in differences(
                expected["values"], sample["values"], atol=left["definition"]["atol"],
                rtol=left["definition"]["rtol"]))
    return errors


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export-h2")
    export.add_argument("--output", type=Path, required=True)
    capture = commands.add_parser("run")
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--backend", choices=("numpy", "rust", "c", "auto"), default="numpy")
    capture.add_argument("--threads", type=positive, default=1)
    capture.add_argument("--repeat", type=positive, default=3)
    capture.add_argument("--timeout", type=positive, default=180)
    capture.add_argument("--tier", choices=("quick", "extended", "all"), default="quick")
    capture.add_argument("--case", action="append")
    comparison = commands.add_parser("compare")
    comparison.add_argument("reference", type=Path)
    comparison.add_argument("candidate", type=Path)
    child = commands.add_parser("_worker", help=argparse.SUPPRESS)
    child.add_argument("case_id")
    child.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "_worker":
        return worker(args.case_id, args.output)
    if args.command == "export-h2":
        export_h2(args.output.resolve())
        return 0
    if args.command == "run":
        return run(args)
    errors = compare(read_json(args.reference), read_json(args.candidate))
    for error in errors[:50]:
        print(error)
    print(f"Comparison: {'FAIL' if errors else 'PASS'} ({len(errors)} differences)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
