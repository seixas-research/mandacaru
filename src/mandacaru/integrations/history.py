"""Disk-backed trajectory and geometry optimization history.

Keeps RAM usage strictly O(1) by streaming step records directly to and
from a temporary JSON Lines file on disk instead of accumulating unbounded
dictionaries in memory across hundreds or thousands of steps.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence


class DiskOptimizationHistory(Sequence):
    """Sequence of geometry step records backed by a disk file."""

    def __init__(self, scratch_dir: str | None = None):
        self._dir = scratch_dir if (scratch_dir and os.path.isdir(scratch_dir)) else None
        self._temp = tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".jsonl", dir=self._dir
        )
        self._path = self._temp.name
        self._count = 0
        self._first_entry: dict | None = None
        self._last_entry: dict | None = None

    @property
    def path(self) -> str:
        return self._path

    def append(self, entry: dict) -> None:
        """Append one geometry step record to disk."""
        item = dict(entry)
        if self._first_entry is None:
            self._first_entry = item
        self._last_entry = item
        line = json.dumps(item, allow_nan=False) + "\n"
        self._temp.write(line)
        self._temp.flush()
        self._count += 1

    def replace_last(self, entry: dict) -> None:
        """Replace the last record (e.g. force updates on the same geometry)."""
        item = dict(entry)
        if self._count <= 1:
            self.clear()
            self.append(item)
            return

        self._last_entry = item
        # Rewriting the file with the updated last record
        self._temp.flush()
        lines = []
        with open(self._path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < self._count - 1:
                    lines.append(line)
        lines.append(json.dumps(item, allow_nan=False) + "\n")
        self._temp.seek(0)
        self._temp.truncate(0)
        self._temp.writelines(lines)
        self._temp.flush()

    def clear(self) -> None:
        """Clear all records from disk."""
        self._temp.seek(0)
        self._temp.truncate(0)
        self._temp.flush()
        self._count = 0
        self._first_entry = None
        self._last_entry = None

    def __len__(self) -> int:
        return self._count

    def __bool__(self) -> bool:
        return self._count > 0

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self._count))]
        if index < 0:
            index += self._count
        if index < 0 or index >= self._count:
            raise IndexError("history index out of range")
        if index == 0 and self._first_entry is not None:
            return dict(self._first_entry)
        if index == self._count - 1 and self._last_entry is not None:
            return dict(self._last_entry)

        self._temp.flush()
        with open(self._path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == index:
                    return json.loads(line.strip())
        raise IndexError("history index out of range")

    def __iter__(self):
        if self._count == 0:
            return
        self._temp.flush()
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def close(self) -> None:
        """Close and remove the backing disk file."""
        try:
            self._temp.close()
        except Exception:
            pass
        try:
            if os.path.exists(self._path):
                os.remove(self._path)
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()
