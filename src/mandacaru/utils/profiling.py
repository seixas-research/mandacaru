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
cores the C backend used, then surface them in the standard-output summary.

* **Time** -- wall-clock seconds via :func:`time.perf_counter`, accumulated per
  named stage (a stage can be timed repeatedly, e.g. once per ADAPT iteration).
* **Memory** -- the process peak resident set size (RSS) via
  :func:`resource.getrusage`; a whole-process high-water mark, so it includes the
  memory the C backend allocates (grid fields, ERI tensor), not just NumPy.
* **Cores** -- the number of OpenMP threads the C integral backend runs with
  (``None`` when the C library is not built and the NumPy fallback is used).
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
    up while this comes back down, so a per-step report needs both.  Needs
    ``psutil``; ``nan`` without it.
    """
    try:
        import psutil
    except ImportError:
        return float("nan")
    try:
        return float(psutil.Process().memory_info().rss) / (1024.0 ** 2)
    except Exception:
        return float("nan")


def cpu_count() -> int | None:
    """Logical CPUs visible to this process (``None`` if undetectable).

    What the machine offers, against which the C backend's OpenMP thread count
    (:func:`backend_cores`) says how much is actually used.
    """
    try:
        count = len(os.sched_getaffinity(0))     # Linux: the allowed set
    except AttributeError:
        count = os.cpu_count()
    return None if not count else int(count)


def backend_cores() -> int | None:
    """Thread count of the native integral backend, or ``None`` (NumPy path)."""
    try:
        from ..integrals import _backend
        return _backend.num_threads()
    except Exception:
        return None


def backend_label() -> str:
    """Label of the active native backend (Rust or C or NumPy)."""
    try:
        from ..integrals import _backend
        return _backend.check_backend().label
    except Exception:
        return "NumPy"


class Timings:
    """Ordered accumulator of named stage wall-times, with memory/core context.

    Use :meth:`time` as a context manager to time a stage (accumulating across
    repeated calls), or :meth:`add` to fold in an externally measured duration.
    :attr:`peak_memory_mb` and :attr:`n_cores` capture the resource context for
    the summary.
    """

    def __init__(self, n_cores: int | None = None, backend: str | None = None):
        self.stages: "OrderedDict[str, float]" = OrderedDict()
        self.n_cores = n_cores
        self.backend = backend
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
            "n_cores": self.n_cores,
            "backend": self.backend,
        }

    def resources(self) -> dict:
        """The machine context for a performance report.

        ``threads`` / ``openmp_threads`` is what the native integral backend
        actually used and ``cpu_count`` what the machine offers; ``mpi`` records
        that there is no distributed parallelism at all (Mandacaru is a single process --
        shared-memory threads inside the native kernels), which is worth stating
        rather than leaving to be inferred.
        """
        backend = self.backend or backend_label()
        cores = self.n_cores if self.n_cores is not None else backend_cores()
        is_rust = "Rust" in backend
        threads_key = "threads" if is_rust else "openmp_threads"
        mpi_desc = ("not used (single process, shared-memory scoped threads)"
                    if is_rust else "not used (single process, shared-memory OpenMP)")
        res = {
            "integration_backend": backend,
            threads_key: cores,
            "cpu_count": cpu_count(),
            "mpi": mpi_desc,
            "peak_memory_MiB": self.peak_memory_mb,
            "resident_memory_MiB": current_memory_mb(),
        }
        try:
            from ..integrals.buffers import is_dump_buffers_enabled, get_buffer_dir
            if is_dump_buffers_enabled():
                res["disk_buffer_dump"] = f"active (.bin files in {get_buffer_dir()})"
        except Exception:
            pass
        return res

    def format_report(self, indent: str = "  ", width: int = 30) -> str:
        """Multi-line report of stage times, cores and peak memory."""
        lines = ["Timings (wall-clock):"]
        for name, secs in self.stages.items():
            lines.append(f"{indent}{name:<{width}s} {secs:>10.4f} s")
        lines.append(f"{indent}{'sum of timed stages':<{width}s} "
                     f"{self.total:>10.4f} s")
        if self.wall_time is not None:
            lines.append(f"{indent}{'total wall time':<{width}s} "
                         f"{self.wall_time:>10.4f} s")
        cores = "n/a (NumPy fallback)" if self.n_cores is None else str(self.n_cores)
        backend = self.backend or ("C (OpenMP)" if self.n_cores else "NumPy")
        lines.append("Resources:")
        lines.append(f"{indent}{'integration backend':<{width}s} {backend}")
        lines.append(f"{indent}{'cores (OpenMP threads)':<{width}s} {cores}")
        lines.append(f"{indent}{'peak memory':<{width}s} "
                     f"{self.peak_memory_mb:>10.1f} MiB")
        return "\n".join(lines)
