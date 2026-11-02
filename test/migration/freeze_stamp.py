"""Provenance stamp carried by every migration fixture and result JSON (freeze item 6).

    from freeze_stamp import stamp
    payload["stamp"] = stamp(datasets=[...], extra={"case": "si_so"})

The stamp pins what a frozen number was computed with: the two Python sources that still own the periodic setup,
the compiled extension, the dataset files that were read, the git revision with its dirty state (and the dirty
files under ``crates/mandacaru-kernels``, which change the extension without changing a Python file), the
interpreter and array-library versions, and the date.  Run from the repository root (or anywhere inside it):

    python test/migration/freeze_stamp.py [OUTPUT.json]
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SOURCES = ("src/mandacaru/pseudopotentials/periodic_paw.py", "src/mandacaru/integrations/paw.py")
KERNELS = "crates/mandacaru-kernels"


def repo_root() -> Path:
    """The repository holding ``src/mandacaru``: the working directory's, else this file's."""
    for start in (Path.cwd().resolve(), Path(__file__).resolve().parent):
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "mandacaru").is_dir():
                return candidate
    raise RuntimeError("freeze_stamp: not inside the mandacaru repository")


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extension_path(root: Path | None = None) -> Path:
    """The extension this interpreter would load: ``src/mandacaru/_native*.so`` for its ABI tag."""
    root = root or repo_root()
    found = sorted((root / "src" / "mandacaru").glob("_native*.so"))
    if not found:
        raise RuntimeError("freeze_stamp: no src/mandacaru/_native*.so; rebuild the extension")
    abi = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    tagged = [p for p in found if abi in p.name and platform.system().lower() in p.name]
    if len(tagged) != 1:
        raise RuntimeError(f"freeze_stamp: expected one {abi} extension for this platform, found "
                           f"{[p.name for p in found]}")
    return tagged[0]


def extension_sha(root: Path | None = None) -> str:
    return sha256_file(extension_path(root))


def _git(root: Path, *args: str) -> str:
    # GIT_OPTIONAL_LOCKS=0: reading the status must not refresh (write) the index.
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, env=env, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"freeze_stamp: git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout


def _dataset_entries(root: Path, paths) -> dict:
    out = {}
    for item in paths or ():
        path = Path(item)
        if not path.is_absolute():
            path = root / path
        out[str(item)] = sha256_file(path)
    return out


def stamp(extra: dict | None = None, datasets=None) -> dict:
    """Return the provenance dict.  ``datasets`` lists the dataset files the fixture read (path -> sha256)."""
    import numpy
    import scipy
    root = repo_root()
    ext = extension_path(root)
    porcelain = [line for line in _git(root, "status", "--porcelain").splitlines() if line.strip()]
    kernels = sorted(line[3:] for line in porcelain if line[3:].startswith(KERNELS))
    ext_stat = ext.stat()
    out = {
        "sources_sha256": {name: sha256_file(root / name) for name in SOURCES},
        "extension": {"file": ext.name, "sha256": sha256_file(ext), "size": ext_stat.st_size,
                      "mtime": datetime.fromtimestamp(ext_stat.st_mtime, timezone.utc).isoformat(timespec="seconds")},
        "datasets_sha256": _dataset_entries(root, datasets),
        "git": {"head": _git(root, "rev-parse", "HEAD").strip(), "dirty": bool(porcelain),
                "n_dirty_files": len(porcelain), "dirty_kernel_files": kernels},
        "versions": {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__},
        "platform": f"{platform.system()} {platform.machine()}",
        "date": datetime.now(timezone.utc).date().isoformat(),
    }
    if extra:
        out["extra"] = dict(extra)
    return out


def main() -> None:
    text = json.dumps(stamp(), indent=2, sort_keys=True) + "\n"
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
