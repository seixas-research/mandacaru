# -*- coding: utf-8 -*-
# file: test/conftest.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Resource monitoring for the test suite.

Every test under ``test/`` is timed (setup + call + teardown) and the process
peak RSS (``resource.getrusage(RUSAGE_SELF).ru_maxrss``) is read after it.  A
summary table -- the :data:`TERMINAL_ROWS` slowest tests -- is printed at the
end of the session (in the terminal summary, and with ``pytest -s``), and the
complete table is written to ``test/.resource_report.txt``.

Budget every run must stay within (the report flags any breach; shrink the
grid or the cell of the offending test rather than the limits):

* every single test  < :data:`TEST_LIMIT_S` (3 minutes),
* the whole session  < :data:`SESSION_LIMIT_S` (10 minutes),
* peak RSS           < :data:`RSS_LIMIT_GB` (8 GB).

The budget was set for the pseudopotential tests (``test_ncpp_family``,
``test_oncvpsp``, ``test_paw``, ``test_pseudopotential_engine``, ...), which
are the heaviest; it now covers the whole suite.
"""

from __future__ import annotations

import os
import resource
import sys
import time

import pytest

TEST_LIMIT_S = 180.0
SESSION_LIMIT_S = 600.0
RSS_LIMIT_GB = 8.0
#: Rows of the per-test table shown in the terminal (the file has them all).
TERMINAL_ROWS = 25

REPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           ".resource_report.txt")


def peak_rss_bytes() -> int:
    """Process peak resident set size in bytes (macOS reports bytes, Linux KiB)."""
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(maxrss if sys.platform == "darwin" else maxrss * 1024)


class ResourceLog:
    """Per-test wall time and the peak RSS observed after each test."""

    def __init__(self):
        self.records: list[tuple[str, float, int]] = []
        self.session_start = time.perf_counter()
        self.session_wall: float | None = None

    def add(self, nodeid: str, seconds: float):
        self.records.append((nodeid, seconds, peak_rss_bytes()))

    def finish(self):
        self.session_wall = time.perf_counter() - self.session_start

    @property
    def peak_rss(self) -> int:
        return max((rss for _n, _s, rss in self.records), default=peak_rss_bytes())

    def table(self, rows: int | None = None) -> str:
        """The report; ``rows`` limits the per-test table to the slowest ones."""
        total = self.session_wall if self.session_wall is not None else \
            time.perf_counter() - self.session_start
        width = max((len(n) for n, _s, _r in self.records), default=20)
        width = min(max(width, 20), 96)
        shown = sorted(self.records, key=lambda r: -r[1])
        if rows is not None:
            shown = shown[:rows]
        title = "Test-suite resource report" + (
            f" ({len(shown)} slowest of {len(self.records)} tests)"
            if rows is not None and len(shown) < len(self.records) else "")
        lines = ["", title,
                 "=" * (width + 30),
                 f"{'test':<{width}} {'wall (s)':>10} {'peak RSS (MB)':>15}",
                 "-" * (width + 30)]
        for nodeid, seconds, rss in shown:
            flag = "  OVER LIMIT" if seconds > TEST_LIMIT_S else ""
            name = nodeid if len(nodeid) <= width else "..." + nodeid[-(width - 3):]
            lines.append(f"{name:<{width}} {seconds:>10.2f} "
                         f"{rss / 1e6:>15.1f}{flag}")
        lines.append("-" * (width + 30))
        peak_gb = self.peak_rss / 1e9
        slowest = max((s for _n, s, _r in self.records), default=0.0)
        lines.append(f"tests: {len(self.records)}   slowest test: {slowest:.2f} s "
                     f"(limit {TEST_LIMIT_S:.0f} s)")
        lines.append(f"session wall time: {total:.2f} s (limit "
                     f"{SESSION_LIMIT_S:.0f} s)")
        lines.append(f"peak RSS: {peak_gb:.3f} GB (limit {RSS_LIMIT_GB:.0f} GB)")
        breaches = []
        if slowest > TEST_LIMIT_S:
            breaches.append("a test exceeded the per-test limit")
        if total > SESSION_LIMIT_S:
            breaches.append("the session exceeded its limit")
        if peak_gb > RSS_LIMIT_GB:
            breaches.append("peak RSS exceeded its limit")
        lines.append("limits: " + ("RESPECTED" if not breaches
                                   else "BREACHED -- " + "; ".join(breaches)))
        lines.append("")
        return "\n".join(lines)


_LOG = ResourceLog()


@pytest.fixture(scope="session", autouse=True)
def resource_report():
    """Session-wide resource log; prints and writes the summary at teardown."""
    _LOG.session_start = time.perf_counter()
    yield _LOG
    _LOG.finish()
    text = _LOG.table()
    try:
        with open(REPORT_PATH, "w", encoding="utf-8") as handle:
            handle.write(text)
    except OSError:
        pass
    # Shown with ``pytest -s``; the terminal summary below shows it always.
    print(_LOG.table(TERMINAL_ROWS))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    """Time the whole protocol (setup + call + teardown) of one test."""
    start = time.perf_counter()
    yield
    _LOG.add(item.nodeid, time.perf_counter() - start)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _LOG.records:
        return
    if _LOG.session_wall is None:
        _LOG.finish()
    for line in _LOG.table(TERMINAL_ROWS).splitlines():
        terminalreporter.write_line(line)
