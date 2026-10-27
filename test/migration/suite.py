"""Run the existing suite with a budget and retain failures as migration evidence.

python test/migration/suite.py --output .mandacaru/migration/suite.json
The baseline deliberately excludes tests of the migration tooling itself.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

if __package__:
    from .baseline import ROOT, controlled_env, environment, positive, write_json
else:
    from baseline import ROOT, controlled_env, environment, positive, write_json


def summarize(path: Path) -> dict:
    result = {"passed": 0, "failures": [], "errors": [], "skips": [], "executed": 0}
    if not path.is_file():
        return {**result, "missing_junit": True}
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        return {**result, "invalid_junit": str(exc)}
    for case in tree.getroot().iter("testcase"):
        result["executed"] += 1
        entry = {"classname": case.get("classname"), "name": case.get("name"),
                 "seconds": float(case.get("time", "0"))}
        for tag, key in (("failure", "failures"), ("error", "errors"), ("skipped", "skips")):
            finding = case.find(tag)
            if finding is not None:
                result[key].append({**entry, "message": finding.get("message", "")[:2000]})
                break
        else:
            result["passed"] += 1
    return result


def run(args) -> int:
    output = args.output.resolve()
    work = output.parent / (output.stem + "-work")
    work.mkdir(parents=True, exist_ok=True)
    junit = work / "pytest.xml"
    junit.unlink(missing_ok=True)
    command = [sys.executable, "-m", "pytest", "--ignore=test/migration",
               "-m", args.markers, "--durations=20", "--junitxml=" + str(junit),
               *args.path]
    env = controlled_env(args.backend, args.threads, work)
    start = time.perf_counter()
    timed_out = False
    with (work / "pytest.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            # Give pytest a chance to emit JUnit evidence for completed tests.
            os.killpg(process.pid, signal.SIGINT)
            try:
                code = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                code = 124
    results = summarize(junit)
    metadata = environment()
    metadata["thread_environment"] = {key: env[key] for key in env if key in (
        "MANDACARU_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS")}
    metadata["blas"] = None  # Not probed inside the pytest process.
    valid = not results.get("missing_junit") and not results.get("invalid_junit")
    write_json(output, {"schema_version": 1, "environment": metadata,
                        "backend_policy": args.backend, "threads": args.threads,
                        "markers": args.markers, "excluded": ["test/migration"],
                        "paths": args.path or ["test"],
                        "budget_s": args.timeout, "wall_s": time.perf_counter() - start,
                        "timed_out": timed_out, "exit_code": code,
                        "complete": bool(valid and not timed_out and code in (0, 1)),
                        "command": ["python", *["--junitxml=<work>/pytest.xml" if arg.startswith("--junitxml=") else arg
                                                   for arg in command[1:]]],
                        "results": results})
    print(f"Suite exit {code}; {results['passed']} passed, {len(results['failures'])} failed, "
          f"{len(results['errors'])} errors, {len(results['skips'])} skipped; timeout={timed_out}")
    return 124 if timed_out else (code or (0 if valid else 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("numpy", "rust", "c", "auto"), default="auto")
    parser.add_argument("--threads", type=positive, default=1)
    parser.add_argument("--timeout", type=positive, default=1200)
    parser.add_argument("--markers", default="not slow")
    parser.add_argument("--path", action="append", default=[],
                        help="Test file/directory to capture (repeatable); default: existing suite")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
