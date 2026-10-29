# -*- coding: utf-8 -*-
# file: pseudopotentials/library_cli.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The ``mandacaru-paw`` console script: point Mandacaru at the libraries.

``mandacaru-paw --set DIR`` records ``DIR``, a checkout of ``mandacaru-paw``,
as ``MANDACARU_PAW_PATH`` in the shell configuration
(:func:`~mandacaru.utils.shell_config.set_shell_variable`); ``--set-upaw DIR``
does the same for ``mandacaru-upaw`` and ``MANDACARU_UPAW_PATH``.
``mandacaru-paw --status`` draws one table row per dataset set (``lda-sr/``,
``pbe-fr/``, ...) of each library, colored by state.  None of them runs a
calculation.
"""

from __future__ import annotations

import argparse
import os
import sys

from .environment import (FAMILY_VARIABLES, LibraryStatus, SET_COMMANDS,
                          _folders, dataset_count, library_status)

#: ``--set`` / ``--set-upaw``: the family each one writes the variable of.
SET_FAMILIES = {"set": "paw-lcao", "set_upaw": "upaw-lcao"}

#: ANSI styles; plain text when color is off.
STYLES = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33",
          "cyan": "36"}
#: What each relativistic treatment is called in the table.
TREATMENTS = {"scalar": "scalar", "dirac": "spin-orbit"}
#: The calculations that read a PAW-LCAO set unless ``paw_dataset`` names
#: another (:func:`~.environment.default_library_folder`).
READ_BY = {"lda-sr": "LDA, HF, quantum solvers",
           "pbe-sr": "PBE, HSE06, r2SCAN"}
#: Elements the shipped library covers; a set with fewer is marked partial.
COMPLETE = 92


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mandacaru-paw",
        description="Set or check where Mandacaru reads the PAW-LCAO and "
                    "UPAW-LCAO datasets from.  Give at least one action; "
                    "they run in the order set, set-upaw, status.")
    parser.add_argument("--set", metavar="DIR", default=None,
                        help="record DIR, a checkout of mandacaru-paw, as "
                             "MANDACARU_PAW_PATH in ~/.zshrc or ~/.bashrc "
                             "(asking before replacing a different value).")
    parser.add_argument("--set-upaw", metavar="DIR", default=None,
                        help="the same for the UPAW-LCAO datasets "
                             "(mandacaru-upaw, MANDACARU_UPAW_PATH); "
                             "optional, since UPAW-LCAO is otherwise "
                             "generated on demand.")
    parser.add_argument("--status", action="store_true",
                        help="draw a table of every dataset set each library "
                             "variable reaches; the exit status is 1 while a "
                             "required library is unset or not a folder.")
    parser.add_argument("--color", choices=("auto", "always", "never"),
                        default="auto",
                        help="color the status table (default: when stdout is "
                             "a terminal and NO_COLOR is unset).")
    return parser


def use_color(mode: str, stream=None) -> bool:
    """Whether to color for ``mode`` (``auto`` follows ``NO_COLOR`` and the
    stream being a terminal; ``FORCE_COLOR`` turns ``auto`` on)."""
    if mode != "auto":
        return mode == "always"
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    stream = sys.stdout if stream is None else stream
    return stream.isatty()


def paint(text: str, *styles: str, color: bool) -> str:
    """``text`` in ``styles`` (names of :data:`STYLES`) when ``color``."""
    if not color or not styles:
        return text
    codes = ";".join(STYLES[name] for name in styles)
    return f"\033[{codes}m{text}\033[0m"


def _frame(stream=None) -> dict:
    """Box-drawing characters, or ASCII when the terminal cannot show them."""
    encoding = getattr(sys.stdout if stream is None else stream, "encoding",
                       None) or "ascii"
    try:
        "─│┌┐└┘├┤┬┴┼✔✘●○".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return dict(h="-", v="|", tl="+", tr="+", bl="+", br="+", ml="+",
                    mr="+", tm="+", bm="+", mm="+", ok="ok", bad="x", dot="*", off="o")
    return dict(h="─", v="│", tl="┌", tr="┐", bl="└", br="┘", ml="├", mr="┤",
                tm="┬", bm="┴", mm="┼", ok="✔", bad="✘", dot="●", off="○")


def render_table(headers, rows, *, color: bool, frame: dict) -> list[str]:
    """Draw ``rows`` of ``(text, style)`` cells under ``headers``.  Cells are
    padded by their text, so ANSI codes never disturb the alignment."""
    widths = [max(len(head), *(len(row[i][0]) for row in rows))
              for i, head in enumerate(headers)]
    h, v = frame["h"], frame["v"]

    def rule(left, mid, right):
        return left + mid.join(h * (w + 2) for w in widths) + right

    def line(cells):
        return v + v.join(f" {paint(text.ljust(w), *style, color=color)} "
                          for (text, style), w in zip(cells, widths)) + v

    out = [rule(frame["tl"], frame["tm"], frame["tr"]),
           line([(head, ("bold",)) for head in headers]),
           rule(frame["ml"], frame["mm"], frame["mr"])]
    out.extend(line(row) for row in rows)
    out.append(rule(frame["bl"], frame["bm"], frame["br"]))
    return out


def _rows(status: LibraryStatus, frame: dict) -> list[list]:
    """The table rows of one library: a row per set, or one saying why not."""
    family = (status.family, ("bold",))
    blank = [("-", ("dim",))] * 5
    if status.state != "ok" or not status.sets:
        if status.state == "unset":
            text, style = ((f"{frame['off']} optional", ("yellow",))
                           if status.optional
                           else (f"{frame['bad']} not set", ("red",)))
        elif status.state == "invalid":
            text, style = f"{frame['bad']} not a folder", ("red",)
        else:
            text, style = f"{frame['bad']} no dataset folder", ("red",)
        return [[family, *blank, (text, style)]]
    rows = []
    for index, item in enumerate(status.sets):
        complete = item.count >= COMPLETE
        read = READ_BY.get(item.name) if status.family == "paw-lcao" else None
        rows.append([
            family if index == 0 else ("", ()),
            (item.name + "/", ("cyan",)),
            (item.functional.upper(), ()),
            (TREATMENTS.get(item.relativity, item.relativity or "-"), ()),
            (str(item.count), ("green",) if complete else ("yellow",)),
            (read or "-", () if read else ("dim",)),
            (f"{frame['ok']} complete" if complete
             else f"{frame['dot']} partial", ("green",) if complete
             else ("yellow",))])
    return rows


def _where(status: LibraryStatus, frame: dict, color: bool) -> str:
    """The line above the table: the variable and where it points."""
    if status.state == "ok":
        mark, style, value = ((frame["ok"], "green", status.path) if status.sets
                              else (frame["bad"], "red",
                                    f"{status.path} (no dataset folder)"))
    elif status.state == "invalid":
        mark, style, value = frame["bad"], "red", f"{status.value} (not a folder)"
    else:
        mark, style = (frame["off"], "yellow") if status.optional \
            else (frame["bad"], "red")
        value = "not set" + ("; optional, generated on demand"
                             if status.optional else "")
    return (f"{paint(mark, style, color=color)} "
            f"{paint(status.variable, 'bold', color=color)}  "
            f"{paint(value, 'dim' if status.ok else style, color=color)}")


def status_report(report=None, *, color: bool = False, frame=None) -> list[str]:
    """The lines of ``--status``: where each variable points, the table of
    sets, and the command that fixes each library that needs it."""
    report = library_status() if report is None else report
    frame = _frame() if frame is None else frame
    lines = [_where(status, frame, color) for status in report] + [""]
    rows = [row for status in report for row in _rows(status, frame)]
    headers = ("Library", "Set", "Functional", "Treatment", "Datasets",
               "Read by default for", "State")
    lines += render_table(headers, rows, color=color, frame=frame)
    for status in report:
        if not status.ok:
            lines.append(paint(f"{frame['bad']} {SET_COMMANDS[status.family]}",
                               "red", color=color))
    return lines


def set_library(family: str, source: str) -> int:
    """``--set DIR`` / ``--set-upaw DIR`` for ``family``.

    ``DIR`` must exist.  It is recorded, absolute, as the family's variable in
    the shell configuration (which asks before replacing a different value)
    and set in this process too.  Returns 0 when the variable was recorded or
    already held that value, 1 otherwise.
    """
    from ..utils.shell_config import ShellConfigError, set_shell_variable

    variable = FAMILY_VARIABLES[family]
    path = os.path.abspath(os.path.expanduser(source))
    if not os.path.isdir(path):
        print(f"{variable}: {source!r} is not a folder; nothing was written.")
        return 1
    try:
        update = set_shell_variable(variable, path)
    except ShellConfigError as error:
        print(f"{variable}: {error}")
        return 1
    print({"added": f"{variable}={path} added to {update.path}",
           "replaced": (f"{variable}={path} replaces {update.previous} in "
                        f"{update.path}"),
           "unchanged": f"{update.path} already sets {variable}={path}",
           "declined": (f"{variable} left at {update.previous} in "
                        f"{update.path}")}[update.action])
    if update.action == "declined":
        return 0
    os.environ[variable] = path
    held = {name: dataset_count(os.path.join(path, name))
            for name in _folders(path)
            if dataset_count(os.path.join(path, name))}
    if held:
        print("    " + ", ".join(f"{name}/: {count} datasets"
                                 for name, count in held.items()))
    else:
        print(f"    warning: {path} has no dataset folder yet; "
              "calculations will not find datasets there")
    if update.changed:
        print("Open a new terminal, or `source` the file above, for the "
              "variable to be defined in your shell.")
    return 0


def report_status(color: bool = False) -> int:
    """``--status``: print the table; 1 while a required library is unset or
    names no folder, or an optional one is set to something that is not."""
    report = library_status()
    print("\n".join(status_report(report, color=color)))
    return 0 if all(status.ok for status in report) else 1


def main(argv=None) -> int:
    parser = build_parser()
    options = parser.parse_args(argv)
    if not (options.set or options.set_upaw or options.status):
        parser.error("give at least one of --set, --set-upaw and --status")
    code = 0
    for option, family in SET_FAMILIES.items():
        source = getattr(options, option)
        if source is not None:
            code = max(code, set_library(family, source))
    if options.status:
        code = max(code, report_status(use_color(options.color)))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
