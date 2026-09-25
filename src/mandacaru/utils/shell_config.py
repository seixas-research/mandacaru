# -*- coding: utf-8 -*-
# file: utils/shell_config.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Persist an environment variable in the user's shell configuration.

``mandacaru --set-paw DIR`` and its siblings end here: the variable goes into
``~/.zshrc`` or ``~/.bashrc`` -- whichever the login shell (``$SHELL``) reads
-- as one ``export NAME=value`` line.  A variable the file already sets is
replaced where it stands, after a ``[Y/n]`` confirmation when the value
changes, so the file never collects duplicates.  The write is atomic (a
temporary file renamed over the original, its permissions kept), and a
configuration file that is a symbolic link (a dotfiles checkout) is written
through the link rather than replaced by a regular file.

Only the file changes: a shell that is already open keeps its environment
until it reads the file again, and :func:`set_shell_variable` says so.
"""

from __future__ import annotations

import os
import re
import shlex
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

#: Login shell (basename of ``$SHELL``) -> its configuration file under $HOME.
SHELL_FILES = {"zsh": ".zshrc", "bash": ".bashrc"}
#: Written above every line this module adds, so it is recognizable later.
MARKER = "# Added by mandacaru"


class ShellConfigError(RuntimeError):
    """The shell configuration cannot be updated safely (unsupported shell,
    no terminal to confirm an overwrite on, unreadable file)."""


@dataclass(frozen=True)
class ShellUpdate:
    """What :func:`set_shell_variable` did.

    ``action`` is ``"added"``, ``"replaced"``, ``"unchanged"`` (the file
    already held this value) or ``"declined"`` (the user answered no).
    ``previous`` is the value the file held before, if any.
    """

    path: Path
    name: str
    value: str
    action: str
    previous: str | None = None

    @property
    def changed(self) -> bool:
        """Whether the file was written."""
        return self.action in ("added", "replaced")


def shell_config_file(shell: str | None = None, home: str | None = None) -> Path:
    """The configuration file of ``shell`` (default: the basename of
    ``$SHELL``) under ``home`` (default: the user's home directory).

    Raises
    ------
    ShellConfigError
        For a shell other than zsh or bash, naming the line to add by hand.
    """
    shell = shell if shell is not None else os.environ.get("SHELL", "")
    name = os.path.basename(shell.strip())
    if name not in SHELL_FILES:
        raise ShellConfigError(
            f"cannot tell which file configures the shell {shell or '(unset $SHELL)'!r}; "
            f"Mandacaru edits {' or '.join('~/' + f for f in SHELL_FILES.values())} "
            "only.  Add the export line to your shell's configuration yourself.")
    base = Path(home) if home is not None else Path.home()
    return base / SHELL_FILES[name]


def _pattern(name: str) -> re.Pattern:
    # `export NAME=...` or a bare `NAME=...`, not a commented-out line.
    return re.compile(rf"^\s*(?:export\s+)?{re.escape(name)}=(.*)$")


def _unquote(raw: str) -> str:
    try:
        parts = shlex.split(raw, comments=True)
    except ValueError:
        return raw.strip()
    return parts[0] if parts else ""


def read_shell_variable(path: Path, name: str) -> str | None:
    """The value the last assignment to ``name`` in ``path`` gives it, or
    ``None`` when the file does not assign it (or does not exist)."""
    if not path.exists():
        return None
    value = None
    pattern = _pattern(name)
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            value = _unquote(match.group(1))
    return value


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a sibling temporary file and a rename,
    keeping the original's permissions; a symbolic link is written through."""
    target = Path(os.path.realpath(path))
    mode = target.stat().st_mode & 0o7777 if target.exists() else 0o644
    handle, temporary = tempfile.mkstemp(dir=target.parent,
                                         prefix=f".{target.name}.",
                                         suffix=".mandacaru")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write(text)
        os.chmod(temporary, mode)
        os.replace(temporary, target)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


def _ask(question: str) -> bool:
    """``[Y/n]`` on the terminal; empty means yes."""
    answer = input(f"{question} [Y/n] ").strip().lower()
    return answer in ("", "y", "yes")


def set_shell_variable(name: str, value: str, *, path: Path | None = None,
                       confirm: Callable[[str], bool] | None = None,
                       interactive: bool | None = None) -> ShellUpdate:
    """Make ``path`` (default :func:`shell_config_file`) export ``name=value``.

    Parameters
    ----------
    name, value : str
        The variable and its value; the value is shell-quoted when written.
    path : Path, optional
        The configuration file; created when missing.
    confirm : callable, optional
        ``confirm(question) -> bool``, asked before replacing a different
        value (default: a ``[Y/n]`` prompt on the terminal).
    interactive : bool, optional
        Whether a prompt can be answered (default: whether stdin is a
        terminal).  Without one, replacing a different value is refused.

    Returns
    -------
    ShellUpdate

    Raises
    ------
    ShellConfigError
        When a different value would have to be replaced and nobody can be
        asked, or the file cannot be read.
    """
    import sys

    path = shell_config_file() if path is None else Path(path)
    line = f"export {name}={shlex.quote(value)}"
    try:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
    except (OSError, UnicodeDecodeError) as error:
        raise ShellConfigError(f"cannot read {path}: {error}") from error

    previous = read_shell_variable(path, name)
    if previous is None:
        separator = "" if not text or text.endswith("\n") else "\n"
        _atomic_write(path, f"{text}{separator}\n{MARKER}\n{line}\n")
        return ShellUpdate(path, name, value, "added")
    if previous == value:
        return ShellUpdate(path, name, value, "unchanged", previous)

    question = (f"{path} already sets {name}={previous}.  "
                f"Replace it with {value}?")
    if confirm is None:
        if interactive is None:
            interactive = sys.stdin.isatty()
        if not interactive:
            raise ShellConfigError(
                f"{path} already sets {name}={previous}; there is no terminal "
                f"to confirm replacing it.  Edit the file yourself:\n    {line}")
        confirm = _ask
    if not confirm(question):
        return ShellUpdate(path, name, value, "declined", previous)

    pattern = _pattern(name)
    lines = text.splitlines()
    # Replace the *last* assignment, the one that wins, and drop any earlier
    # ones so the file states the variable exactly once.
    hits = [i for i, old in enumerate(lines) if pattern.match(old)]
    lines[hits[-1]] = line
    for i in reversed(hits[:-1]):
        del lines[i]
    _atomic_write(path, "\n".join(lines) + "\n")
    return ShellUpdate(path, name, value, "replaced", previous)
