"""Per-calculation allocation, execution and scratch ownership.

Two options configure a calculation's native context, each with one parser here:

* ``parallelization`` -- the processor layout ``{"kpts": k, "bands": b, "threads": t}``:
  ``k`` groups of k-point (and spin channel) tasks, each split into ``b`` groups of states,
  each running its kernels on ``t`` workers; the processor count is ``k * b * t``. A missing
  key is 1. Omitted, the context takes ``{"threads": default}``, the default being
  ``MANDACARU_NUM_THREADS``, then ``OMP_NUM_THREADS``, then the hardware.
* ``resources`` -- the memory budget (``memory_limit_bytes``), the size from which bulk arrays
  live on disk (``disk_offload_bytes``) and the scratch parent (``scratch_dir``).
"""

from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from . import _native

#: The levels of ``parallelization``, outermost first.
PARALLEL_LEVELS = ("kpts", "bands", "threads")
#: The keys of ``resources``.
RESOURCE_KEYS = ("memory_limit_bytes", "disk_offload_bytes", "scratch_dir")


def parse_parallelization(value, *, levels=PARALLEL_LEVELS, owner="parallelization"):
    """``{"kpts": k, "bands": b, "threads": t}`` from a ``parallelization`` option, or ``None``
    when it is omitted (the native default worker count). ``levels`` are the levels the caller
    has; a level it lacks may be given only as 1. A missing key is 1."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(
            f"{owner} must be a dict such as {{'kpts': 4, 'threads': 2}}, got {value!r}"
        )
    unknown = sorted(str(key) for key in value if key not in PARALLEL_LEVELS)
    if unknown:
        raise TypeError(
            f"{owner} accepts the keys 'kpts', 'bands' and 'threads', got {unknown}"
        )
    layout = {}
    for key in PARALLEL_LEVELS:
        count = value.get(key, 1)
        if type(count) is not int:
            raise TypeError(f"{owner}[{key!r}] must be a positive integer, got {count!r}")
        if count < 1:
            raise ValueError(f"{owner}[{key!r}] must be a positive integer, got {count}")
        if count > 1 and key not in levels:
            raise ValueError(
                f"{owner}[{key!r}]={count}: this calculation has no {key} level; it takes "
                f"{', '.join(repr(level) for level in levels)} only"
            )
        layout[key] = count
    return layout


def parse_resources(value, *, owner="resources", takes_layout=True):
    """The ``resources`` option as keyword arguments of the native context. The processor
    count is not a resource: ``threads`` is refused, with a pointer to ``parallelization``
    when the caller takes one (``takes_layout``), else to the default worker count a helper
    runs on."""
    settings = dict(value or {})
    if "threads" in settings:
        raise ValueError(
            f"{owner} does not take 'threads': the processors are set by "
            "parallelization={'kpts': ..., 'bands': ..., 'threads': ...}"
            if takes_layout
            else f"{owner} does not take 'threads': this helper runs on the default worker "
            "count (MANDACARU_NUM_THREADS, then OMP_NUM_THREADS, then the hardware); a "
            "calculation's processors are Mandacaru(parallelization=...)"
        )
    unknown = sorted(str(key) for key in settings if key not in RESOURCE_KEYS)
    if unknown:
        raise TypeError(
            f"{owner} accepts the keys {', '.join(repr(key) for key in RESOURCE_KEYS)}, "
            f"got {unknown}"
        )
    disk = settings.get("disk_offload_bytes")
    if disk is not None and (type(disk) is not int or disk < 0):
        raise ValueError(
            "disk_offload_bytes must be a nonnegative integer number of bytes "
            f"(0 writes every bulk array to disk), got {disk!r}"
        )
    if settings.get("scratch_dir") is not None:
        settings["scratch_dir"] = str(Path(settings["scratch_dir"]).absolute())
    return settings


def native_context(resources=None, parallelization=None, *, takes_layout=False):
    """A native context from the ``resources`` option of a helper (memory budget, offload
    threshold, scratch parent), and from its ``parallelization`` when it takes one
    (``takes_layout``: a helper has the ``threads`` level only)."""
    layout = parse_parallelization(parallelization, levels=("threads",))
    settings = parse_resources(resources, takes_layout=takes_layout)
    if layout is None:
        return _native.Context(**settings)
    return _native.Context(
        threads=layout["threads"], kpts=layout["kpts"], bands=layout["bands"], **settings
    )


def layout_label(context) -> str:
    """The layout of a native context as the run log writes it:
    ``kpts 4 x bands 1 x threads 2 (8 workers)``."""
    return (
        f"kpts {context.kpts} x bands {context.bands} x threads {context.threads} "
        f"({context.workers} workers)"
    )


class CalculationResources:
    def __init__(
        self,
        *,
        parallelization=None,
        resources=None,
        context=None,
    ):
        settings = parse_resources(resources)
        layout = parse_parallelization(parallelization)
        if context is not None:
            self.context = context
        elif layout is None:
            self.context = _native.Context(**settings)
        else:
            self.context = _native.Context(
                threads=layout["threads"], kpts=layout["kpts"], bands=layout["bands"],
                **settings,
            )
        self.lock = RLock()
        scratch_dir = settings.get("scratch_dir")
        self._scratch_parent = None if scratch_dir is None else Path(scratch_dir)
        self._scratch = None
        self.closed = False

    @property
    def scratch(self):
        with self.lock:
            if self.closed:
                raise RuntimeError("calculation resources are closed")
            if self._scratch is None:
                if self._scratch_parent is not None:
                    self._scratch_parent.mkdir(parents=True, exist_ok=True)
                self._scratch = TemporaryDirectory(
                    prefix="mandacaru-", dir=self._scratch_parent
                )
            return Path(self._scratch.name)

    def close(self):
        with self.lock:
            if self._scratch is not None:
                self._scratch.cleanup()
            self.closed = True
            # What the calculation freed goes back to the system now: a scan of geometries
            # would otherwise carry each point's retained pages into the next.
            self.context.release_freed_memory()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
