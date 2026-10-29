"""Run the existing examples in disposable copies and report every failure.

No example is excluded by default. Outbound network connections are blocked so
checking a hardware example cannot submit a paid remote job. Local simulators
can communicate over loopback. Scripts in one invocation share a disposable
workspace, preserving generated inputs such as the offline proposal store for
the subsequent inference example. Original examples and recorded data are retained.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
CHILD = r"""
import functools, ipaddress, json, os, pathlib, runpy, socket, sys, time, weakref
for name in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
    os.environ.pop(name,None)
original_connect=socket.socket.connect
def local_connect(sock,address):
    if isinstance(address,tuple):
        host=address[0]
        if host != 'localhost':
            try:
                local=ipaddress.ip_address(host).is_loopback
            except ValueError:
                local=False
            if not local:
                raise RuntimeError('example check blocks outbound network: '+str(host))
    return original_connect(sock,address)
socket.socket.connect=local_connect
path=sys.argv[1]
scientific_path=pathlib.Path(sys.argv[2])
from mandacaru import Mandacaru
original_run=Mandacaru.run
owners=weakref.WeakKeyDictionary()
observations=[]
@functools.wraps(original_run)
def observed_run(calculator,*args,**kwargs):
    started=time.monotonic()
    result=original_run(calculator,*args,**kwargs)
    if calculator not in owners:
        owners[calculator]=len(observations)
        observations.append(dict(method=calculator.method,runs=[]))
    reference=result.scf
    observations[owners[calculator]]["runs"].append(dict(
        energy_ha=float(result.energy_hartree),
        solver_converged=bool(result.success),
        completed=bool(result.completed),
        scf_converged=None if reference is None else bool(reference.converged),
        scf_iterations=None if reference is None else int(reference.n_iterations),
        inner_optimizer_failures=result.optimizer_failures,
        memory_peak_bytes=int(calculator.context.memory_peak_bytes),
        seconds=time.monotonic()-started,
    ))
    temporary=scientific_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(observations,indent=2,allow_nan=False))
    temporary.replace(scientific_path)
    return result
Mandacaru.run=observed_run
sys.argv=[path]
runpy.run_path(path,run_name='__main__')
"""


def run_example(source, timeout, *, plot_cache=None, workspace=None):
    if workspace is None:
        with tempfile.TemporaryDirectory(prefix="mandacaru-example-") as temporary:
            return run_example(source, timeout, plot_cache=plot_cache, workspace=Path(temporary))
    started = time.monotonic()
    target = workspace / "examples" / source.parent.name
    if not target.exists():
        shutil.copytree(
            source.parent, target, ignore=shutil.ignore_patterns("__pycache__")
        )
    env = dict(
        os.environ,
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(
            workspace / "matplotlib" if plot_cache is None else plot_cache
        ),
        PYTHONUNBUFFERED="1",
        AWS_EC2_METADATA_DISABLED="true",
    )
    logs = workspace / "logs" / source.parent.name
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / (source.stem + ".stdout.log")
    stderr_path = logs / (source.stem + ".stderr.log")
    pid_path = logs / (source.stem + ".pid")
    scientific_path = logs / (source.stem + ".scientific.json")
    scientific_path.unlink(missing_ok=True)
    with stdout_path.open("w") as stdout_file, stderr_path.open("w") as stderr_file:
        process = subprocess.Popen(
            [sys.executable, "-I", "-c", CHILD, str(target / source.name), str(scientific_path)],
            cwd=workspace,
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
        )
        pid_path.write_text(str(process.pid))
        try:
            process.wait(timeout=timeout)
            status = "passed" if process.returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            status = "timed_out"
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            pid_path.unlink(missing_ok=True)
    # The retained workspace supplies live logs for long calculations. Read
    # only a bounded tail into the report, regardless of the run's verbosity.
    def tail(path):
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 16000))
            return stream.read().decode(errors="replace")

    return dict(
        example=str(source.relative_to(ROOT)),
        status=status,
        seconds=time.monotonic() - started,
        stdout=tail(stdout_path),
        stderr=tail(stderr_path),
        scientific=json.loads(scientific_path.read_text()) if scientific_path.exists() else [],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("examples", nargs="*", type=Path)
    parser.add_argument("--timeout", type=float, default=None,
                        help="optional positive time limit in seconds for each example")
    parser.add_argument("--workspace", type=Path,
                        help="retain the example copy and generated outputs in this directory")
    parser.add_argument(
        "--report", type=Path, default=Path("/tmp/mandacaru-example-report.json")
    )
    options = parser.parse_args()
    if options.timeout is not None and (not math.isfinite(options.timeout) or options.timeout <= 0):
        parser.error("--timeout must be positive and finite")
    sources = (
        sorted((ROOT / "examples").rglob("*.py"))
        if not options.examples
        else [p.absolute() for p in options.examples]
    )
    results = []
    workspace_context = (
        tempfile.TemporaryDirectory(prefix="mandacaru-example-suite-")
        if options.workspace is None else nullcontext(options.workspace.absolute())
    )
    with workspace_context as temporary:
        workspace = Path(temporary)
        workspace.mkdir(parents=True, exist_ok=True)
        plot_cache = workspace / "matplotlib"
        for source in sources:
            result = run_example(source, options.timeout, plot_cache=plot_cache, workspace=workspace)
            results.append(result)
            print(
                f"{result['status']:10s} {result['seconds']:7.2f}s {result['example']}",
                flush=True,
            )
            options.report.parent.mkdir(parents=True, exist_ok=True)
            options.report.write_text(json.dumps(dict(workspace=str(workspace), results=results), indent=2))
    passed = sum(row["status"] == "passed" for row in results)
    print(f"{passed}/{len(results)} passed; report: {options.report}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
