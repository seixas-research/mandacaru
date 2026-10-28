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
        memory_limit_bytes=512 * 1024**2,
        scratch_dir=None,
        context=None,
    ):
        self.context = (
            context
            if context is not None
            else _native.Context(threads=threads, memory_limit_bytes=memory_limit_bytes)
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

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
