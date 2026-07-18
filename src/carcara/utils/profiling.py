# -*- coding: utf-8 -*-
# file: utils/profiling.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Lightweight timing / memory / core-count profiling for the simulation stages.

The variational drivers (:class:`~carcara.algorithms.vqe.VQE`,
:class:`~carcara.algorithms.adapt_vqe.ADAPTVQE`) and the real-space integral
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

import sys
import time
from collections import OrderedDict
from contextlib import contextmanager


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


def backend_cores() -> int | None:
    """OpenMP thread count of the C integral backend, or ``None`` (NumPy path)."""
    try:
        from ..integrals import _backend
    except Exception:
        return None
    return _backend.num_threads()


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

    def merge(self, other: "Timings", prefix: str = "") -> None:
        """Merge another :class:`Timings`' stages (optionally name-prefixed)."""
        for name, secs in other.stages.items():
            self.add(f"{prefix}{name}", secs)
        if other.n_cores is not None and self.n_cores is None:
            self.n_cores = other.n_cores
        if other.backend is not None and self.backend is None:
            self.backend = other.backend

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
