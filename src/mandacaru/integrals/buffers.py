# -*- coding: utf-8 -*-
# file: integrals/buffers.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Disk buffer management for real-space grids, orbital evaluations and workspaces.

When buffer dumping is enabled (``dump_buffers=True``, default for all DFT runs),
large multi-megabyte grid arrays (such as the basis evaluations ``_psi``,
periodic Bloch grids, FFT workspace buffers, and volumetric fields) are backed
by binary files (``.bin``) via memory mapping (:class:`numpy.memmap`).

This drastically lowers the physical RAM (Resident Set Size) footprint of the
process, eliminates memory leaks and heap fragmentation across geometry steps,
and keeps Mandacaru within strict RAM thresholds (such as the GPAW memory baseline).
"""

from __future__ import annotations

import atexit
import os
import uuid
from pathlib import Path
from typing import Sequence

import numpy as np

#: Global active buffers
_ACTIVE_BUFFERS: list[Path] = []
_BUFFER_DIR: Path | None = None
#: Default is True for DFT calculations
_DUMP_BUFFERS_ENABLED: bool = True


def resolve_default_buffer_dir() -> Path:
    """Find a writable directory for .bin buffers (.mandacaru/buffers or user home/tmp)."""
    candidates = [
        Path.cwd() / ".mandacaru" / "buffers",
        Path.home() / ".mandacaru" / "buffers",
        Path("/tmp/mandacaru/buffers"),
    ]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            test_file = candidate / f".write_test_{uuid.uuid4().hex[:6]}"
            test_file.touch()
            test_file.unlink()
            return candidate
        except (OSError, PermissionError):
            continue
    fallback = Path("/tmp/mandacaru_buffers")
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def set_buffer_dir(path: str | Path | None) -> Path:
    """Set the active directory where .bin buffer files are dumped."""
    global _BUFFER_DIR
    if path is None:
        _BUFFER_DIR = resolve_default_buffer_dir()
    else:
        p = Path(path).resolve()
        if p.name != "buffers":
            p = p / "buffers"
        p.mkdir(parents=True, exist_ok=True)
        _BUFFER_DIR = p
    return _BUFFER_DIR


def get_buffer_dir() -> Path:
    """Get the active directory where .bin buffer files are dumped."""
    global _BUFFER_DIR
    if _BUFFER_DIR is None:
        _BUFFER_DIR = resolve_default_buffer_dir()
    return _BUFFER_DIR


def is_dump_buffers_enabled() -> bool:
    """Return whether disk buffer dumping to .bin files is currently active."""
    global _DUMP_BUFFERS_ENABLED
    return _DUMP_BUFFERS_ENABLED


def set_dump_buffers(enabled: bool) -> None:
    """Enable or disable dumping grid/orbital buffers to disk .bin files."""
    global _DUMP_BUFFERS_ENABLED
    _DUMP_BUFFERS_ENABLED = bool(enabled)


def create_disk_buffer(name: str, shape: Sequence[int],
                       dtype=np.float32, prefix: str = "") -> np.ndarray:
    """Create a :class:`numpy.memmap` backed array written as a ``.bin`` file in the buffer directory."""
    buf_dir = get_buffer_dir()
    buf_dir.mkdir(parents=True, exist_ok=True)

    uid = uuid.uuid4().hex[:8]
    fname = f"{prefix}{name}_{uid}.bin" if prefix else f"{name}_{uid}.bin"
    file_path = buf_dir / fname

    mmap_arr = np.memmap(file_path, dtype=dtype, mode="w+", shape=tuple(shape))
    _ACTIVE_BUFFERS.append(file_path)
    return mmap_arr


def cleanup_disk_buffers(remove_files: bool = False) -> None:
    """Flush and optionally unlink active .bin buffer files."""
    global _ACTIVE_BUFFERS
    for p in list(_ACTIVE_BUFFERS):
        try:
            if remove_files and p.exists():
                p.unlink()
        except Exception:
            pass
    if remove_files:
        _ACTIVE_BUFFERS.clear()


def clear_all_disk_buffers() -> None:
    """Remove all .bin files in the active buffer directory."""
    buf_dir = get_buffer_dir()
    if buf_dir.exists():
        for p in buf_dir.glob("*.bin"):
            try:
                p.unlink()
            except Exception:
                pass
    cleanup_disk_buffers(remove_files=True)


# Clean up when python interpreter exits
atexit.register(cleanup_disk_buffers, remove_files=False)
