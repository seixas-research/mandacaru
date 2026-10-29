"""Run the existing examples in disposable copies and report every failure.

No example is excluded by default. Outbound network connections are blocked so
checking a hardware example cannot submit a paid remote job. Local simulators
can communicate over loopback. Scripts in one invocation share a disposable
workspace, preserving generated inputs such as the offline proposal store for
the subsequent inference example. Original examples and recorded data are retained.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
CHILD = r"""
import ipaddress, os, runpy, socket, sys
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
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", CHILD, str(target / source.name)],
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        status = "passed" if result.returncode == 0 else "failed"
        stdout, stderr = result.stdout, result.stderr
    except subprocess.TimeoutExpired as exc:
        status = "timed_out"
        stdout = exc.stdout or b""
        stderr = exc.stderr or b""
        stdout = (
            stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
        )
        stderr = (
            stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
        )
    return dict(
        example=str(source.relative_to(ROOT)),
        status=status,
        seconds=time.monotonic() - started,
        stdout=stdout[-16000:],
        stderr=stderr[-16000:],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("examples", nargs="*", type=Path)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--report", type=Path, default=Path("/tmp/mandacaru-example-report.json")
    )
    options = parser.parse_args()
    sources = (
        sorted((ROOT / "examples").rglob("*.py"))
        if not options.examples
        else [p.absolute() for p in options.examples]
    )
    results = []
    with tempfile.TemporaryDirectory(prefix="mandacaru-example-suite-") as temporary:
        workspace = Path(temporary)
        plot_cache = workspace / "matplotlib"
        for source in sources:
            result = run_example(source, options.timeout, plot_cache=plot_cache, workspace=workspace)
            results.append(result)
            print(
                f"{result['status']:10s} {result['seconds']:7.2f}s {result['example']}",
                flush=True,
            )
            options.report.parent.mkdir(parents=True, exist_ok=True)
            options.report.write_text(json.dumps(dict(results=results), indent=2))
    passed = sum(row["status"] == "passed" for row in results)
    print(f"{passed}/{len(results)} passed; report: {options.report}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
