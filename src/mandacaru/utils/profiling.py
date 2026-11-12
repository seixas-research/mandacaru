# -*- coding: utf-8 -*-
# file: utils/profiling.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Lightweight timing / memory / core-count profiling for the simulation stages.

The variational drivers (:class:`~mandacaru.algorithms.vqe.VQE`,
:class:`~mandacaru.algorithms.adapt_vqe.ADAPTVQE`) and the real-space integral
engine record how long each stage takes, the peak process memory, and how many
processors the calculation runs on, then surface them in the standard-output summary.

* **Time** -- wall-clock seconds via :func:`time.perf_counter`, accumulated per
  named stage (a stage can be timed repeatedly, e.g. once per ADAPT iteration).
* **Memory** -- the process peak resident set size (RSS) via
  :func:`resource.getrusage`; a whole-process high-water mark, so it includes the
  memory the native kernels allocate (grid fields, ERI tensor), not just NumPy arrays.
* **Processors** -- the calculation's processor layout
  (``kpts 4 x bands 1 x threads 2 (8 workers)``, ``None`` when not recorded).
"""

from __future__ import annotations

import os
import sys
import time
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps


_ACTIVE_TIMINGS = ContextVar("mandacaru_active_timings", default=None)


def profile_stage(name):
    """Measure exclusive wall time only inside an active profiling scope.

    Nested stages subtract their duration from their parent, so the printed
    sum never counts the same work twice. Other methods incur no timers.
    """
    def decorate(function):
        @wraps(function)
        def measured(*args, **kwargs):
            timings = _ACTIVE_TIMINGS.get()
            if timings is None:
                return function(*args, **kwargs)
            with timings.exclusive(name):
                return function(*args, **kwargs)
        return measured
    return decorate


def peak_memory_mb() -> float:
    """Process peak resident memory in MiB (a monotonic high-water mark)."""
    try:
        import resource
    except ImportError:                                   # non-Unix (e.g. Windows)
        return float("nan")
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS, kibibytes on Linux.
    scale = 1.0 if sys.platform == "darwin" else 1024.0
    return float(ru) * scale / (1024.0 ** 2)


def current_memory_mb() -> float:
    """Resident set size of this process *right now*, in MiB.

    The companion of :func:`peak_memory_mb`, which is a monotonic high-water
    mark: after a stage that allocated and freed a large tensor the peak stays
    up while this comes back down, so a per-step report needs both.
    """
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        scale = 1.0 if sys.platform == "darwin" else 1024.0
        return float(ru) * scale / (1024.0 ** 2)
    except Exception:
        return float("nan")


def cpu_count() -> int | None:
    """Logical CPUs visible to this process (``None`` if undetectable): what the machine
    offers, beside the processors the calculation's layout uses."""
    try:
        count = len(os.sched_getaffinity(0))     # Linux: the allowed set
    except AttributeError:
        count = os.cpu_count()
    return None if not count else int(count)


class Timings:
    """Ordered accumulator of named stage wall-times, with memory/core context.

    Use :meth:`time` as a context manager to time a stage (accumulating across
    repeated calls), or :meth:`add` to fold in an externally measured duration.
    :attr:`peak_memory_mb` and :attr:`parallelization` (the calculation's processor layout,
    as :func:`mandacaru.resources.layout_label` writes it) capture the resource context for
    the summary.
    """

    def __init__(self, parallelization: str | None = None):
        self.stages: "OrderedDict[str, float]" = OrderedDict()
        self.parallelization = parallelization
        self.wall_time: float | None = None      # end-to-end wall time (optional)
        self._peak_mb = peak_memory_mb()
        self._frames = []

    @contextmanager
    def activate(self):
        """Enable stage instrumentation for the current calculation."""
        token = _ACTIVE_TIMINGS.set(self)
        try:
            yield self
        finally:
            _ACTIVE_TIMINGS.reset(token)

    @contextmanager
    def exclusive(self, name: str):
        """Time this stage, excluding time attributed to nested stages."""
        frame = [time.perf_counter(), 0.0]
        self._frames.append(frame)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - frame[0]
            self._frames.pop()
            self.add(name, max(0.0, elapsed - frame[1]))
            if self._frames:
                self._frames[-1][1] += elapsed

    @contextmanager
    def time(self, name: str):
        """Time the wrapped block, accumulating into stage ``name``."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t0)

    def add(self, name: str, seconds: float) -> None:
        """Fold ``seconds`` into stage ``name`` and refresh the memory peak."""
        self.stages[name] = self.stages.get(name, 0.0) + float(seconds)
        self._peak_mb = max(self._peak_mb, peak_memory_mb())

    @property
    def peak_memory_mb(self) -> float:
        self._peak_mb = max(self._peak_mb, peak_memory_mb())
        return self._peak_mb

    @property
    def total(self) -> float:
        return float(sum(self.stages.values()))

    def as_dict(self) -> dict:
        """Serializable view: stage times plus the resource context."""
        return {
            "stages_s": dict(self.stages),
            "total_s": self.total,
            "wall_time_s": self.wall_time,
            "peak_memory_mb": self.peak_memory_mb,
            "parallelization": self.parallelization,
        }

    def resources(self) -> dict:
        """The machine context for a performance report.

        ``parallelization`` is the calculation's processor layout and ``cpu_count`` what the
        machine offers; ``mpi`` records that there is no distributed parallelism at all
        (Mandacaru is a single process, shared-memory threads inside the native kernels),
        which is worth stating rather than leaving to be inferred.
        """
        return {
            "integration_backend": "Rust",
            "parallelization": ("n/a (not recorded)" if self.parallelization is None
                                else self.parallelization),
            "cpu_count": cpu_count(),
            "mpi": "not used (single process, shared-memory scoped threads)",
            "peak_memory_MiB": self.peak_memory_mb,
            "resident_memory_MiB": current_memory_mb(),
        }

    def format_report(self, indent: str = "  ", width: int = 30) -> str:
        """Multi-line report of stage times, the processor layout and peak memory."""
        lines = ["Timings (wall-clock):"]
        for name, secs in self.stages.items():
            lines.append(f"{indent}{name:<{width}s} {secs:>10.4f} s")
        lines.append(f"{indent}{'sum of timed stages':<{width}s} "
                     f"{self.total:>10.4f} s")
        if self.wall_time is not None:
            lines.append(f"{indent}{'total wall time':<{width}s} "
                         f"{self.wall_time:>10.4f} s")
        layout = "n/a (not recorded)" if self.parallelization is None else self.parallelization
        lines.append("Resources:")
        lines.append(f"{indent}{'parallelization':<{width}s} {layout}")
        lines.append(f"{indent}{'peak memory':<{width}s} "
                     f"{self.peak_memory_mb:>10.1f} MiB")
        return "\n".join(lines)
