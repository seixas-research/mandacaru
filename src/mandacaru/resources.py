"""Per-calculation allocation, execution and scratch ownership."""

from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from . import _native


class CalculationResources:
    def __init__(
        self,
        *,
        threads=None,
        memory_limit_bytes=32 * 1024**3,
        disk_offload_bytes=None,
        scratch_dir=None,
        context=None,
    ):
        if disk_offload_bytes is not None and (
            type(disk_offload_bytes) is not int or disk_offload_bytes < 0
        ):
            raise ValueError(
                "disk_offload_bytes must be a nonnegative integer number of bytes "
                f"(0 writes every bulk array to disk), got {disk_offload_bytes!r}"
            )
        self.context = (
            context
            if context is not None
            else _native.Context(
                threads=threads,
                memory_limit_bytes=memory_limit_bytes,
                disk_offload_bytes=disk_offload_bytes,
            )
        )
        self.lock = RLock()
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
