# -*- coding: utf-8 -*-
# file: utils/banner.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Start-up banner: version, host and dependency information.

:func:`lines` renders the banner plus the runtime environment (platform, Python
and key dependency versions) as a list of text lines, and :func:`show` writes
them to **standard output**.  The variational drivers call ``show`` at the start
of a run -- before anything is written to the structured ``output.txt`` log --
so the console always opens with the provenance of the calculation, and
:class:`~mandacaru.utils.logging.AdaptOutputLogger` writes the same lines once at
the top of the log file, so a log kept from a long run carries its own
provenance.

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


#: The "MANDACARU" wordmark, in Unicode block elements (U+2580 upper half,
#: U+2584 lower half, U+2588 full block): three rows, 57 columns.
WORDMARK = (
    "██▄  ▄██  ▄▄▄  ▄▄  ▄▄ ▄▄▄▄   ▄▄▄   ▄▄▄▄  ▄▄▄  ▄▄▄▄  ▄▄ ▄▄",
    "██ ▀▀ ██ ██▀██ ███▄██ ██▀██ ██▀██ ██▀▀▀ ██▀██ ██▄█▄ ██ ██",
    "██    ██ ██▀██ ██ ▀██ ████▀ ██▀██ ▀████ ██▀██ ██ ██ ▀███▀",
)

#: Left margin of the wordmark: it centers the 57 columns under the 65-column
#: rule and lines up with the indented block below it.
INDENT = "    "


def _write(line: str = "") -> None:
    """Write one line to standard output.

    The wordmark is not ASCII, and a console is not always UTF-8 (``LANG=C``,
    a redirected stream on an old locale).  A banner must never be the reason a
    calculation does not start, so a stream that cannot encode a line gets it
    with the unencodable characters replaced instead of a ``UnicodeEncodeError``.
    """
    text = f"{line}\n"
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        sys.stdout.write(text.encode(encoding, "replace").decode(encoding))


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


def lines() -> list[str]:
    """The banner and environment block, as a list of lines (no line endings).

    The single source of the banner's text: :func:`show` writes these lines to
    standard output and the ``output.txt`` logger writes the same ones into the
    log, so the console and the file cannot drift apart.
    """
    out = [""] + [INDENT + row for row in WORDMARK] + [
        "",
        "-----------------------------------------------------------------",
        f"    version:       {__version__}",
        "    developed by:  Leandro Seixas Rocha",
        "    homepage:      https://github.com/seixas-research/mandacaru",
        "    documentation: https://mandacaru.readthedocs.io/",
        "-----------------------------------------------------------------",
        "",
        "System:",
        f"├── architecture: {platform.machine()}",
        f"├── platform:     {platform.system()}",
        f"├── user:         {_username()}",
        f"├── hostname:     {gethostname()}",
        f"├── cwd:          {os.getcwd()}",
        f"└── PID:          {os.getpid()}",
        "",
        "Python:",
        f"├── version:    {sys.version.splitlines()[0]}",
        f"└── executable: {sys.executable}",
        "",
        "Dependencies:",
    ]
    deps = ["ase", "numpy", "scipy", "matplotlib", "qiskit"]
    for i, name in enumerate(deps):
        version, directory = _dep(name)
        branch = "└──" if i == len(deps) - 1 else "├──"
        out.append(f"{branch} {name + ' version:':<20s} {version:<10s} "
                   f"[{directory}]")
    out.append("")
    return out


def show() -> None:
    """Write the Mandacaru banner and environment information to standard output."""
    for line in lines():
        _write(line)
