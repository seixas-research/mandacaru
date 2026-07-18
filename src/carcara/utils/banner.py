# -*- coding: utf-8 -*-
# file: utils/banner.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Start-up banner: version, host and dependency information.

:func:`show` writes a banner plus the runtime environment (platform, Python and
key dependency versions) to **standard output**.  The variational drivers call it
at the start of a run -- before anything is written to the structured
``output.txt`` log -- so the console always opens with the provenance of the
calculation.

Output goes to ``sys.stdout`` through :func:`_write` (not the built-in ``print``),
and every dependency version is looked up defensively so a missing optional
package never breaks the banner.
"""

from __future__ import annotations

import getpass
import os
import platform
import sys
from socket import gethostname

from ..version import __version__


def _write(line: str = "") -> None:
    sys.stdout.write(f"{line}\n")


def _dep(name: str) -> tuple[str, str]:
    """``(version, directory)`` of an importable dependency, or ``("n/a", "")``."""
    try:
        module = __import__(name)
    except Exception:
        return "n/a", ""
    version = getattr(module, "__version__", "n/a")
    path = getattr(module, "__file__", "") or ""
    return str(version), os.path.dirname(path)


def _username() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return os.environ.get("USER", "?")


def show() -> None:
    """Write the Carcará banner and environment information to standard output."""
    _write("       _____                                  ")
    _write("      / ____|                                 ")
    _write("     | |     __ _ _ __ ___ __ _ _ __ __ _     ")
    _write("     | |    / _` | '__/ __/ _` | '__/ _` |    ")
    _write("     | |___| (_| | | | (_| (_| | | | (_| |    ")
    _write("      \\_____\\__,_|_|  \\___\\__,_|_|  \\__,_|    ")
    _write("")
    _write("-----------------------------------------------------------------")
    _write(f"    version:       {__version__}")
    _write("    developed by:  Leandro Seixas Rocha")
    _write("    homepage:      https://github.com/seixas-research/carcara")
    _write("    documentation: https://carcara.readthedocs.io/")
    _write("-----------------------------------------------------------------")
    _write("")
    _write("System:")
    _write(f"├── architecture: {platform.machine()}")
    _write(f"├── platform:     {platform.system()}")
    _write(f"├── user:         {_username()}")
    _write(f"├── hostname:     {gethostname()}")
    _write(f"├── cwd:          {os.getcwd()}")
    _write(f"└── PID:          {os.getpid()}")
    _write("")
    _write("Python:")
    _write(f"├── version:    {sys.version.splitlines()[0]}")
    _write(f"└── executable: {sys.executable}")
    _write("")
    _write("Dependencies:")
    deps = ["ase", "numpy", "scipy", "matplotlib", "qiskit"]
    for i, name in enumerate(deps):
        version, directory = _dep(name)
        branch = "└──" if i == len(deps) - 1 else "├──"
        _write(f"{branch} {name + ' version:':<20s} {version:<10s} [{directory}]")
    _write("")
