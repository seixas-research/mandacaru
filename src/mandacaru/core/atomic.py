# -*- coding: utf-8 -*-
# file: core/atomic.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Atomic file replacement, shared by every snapshot Mandacaru writes.

A Hamiltonian cache, a wavefunction checkpoint and an inspection dump are all
*snapshots*: opening the destination in write mode means a crash, a
serialization error or a full filesystem destroys the previous one and leaves a
truncated file in its place.  They are therefore written to a temporary file in
the **same directory** (so the final move never crosses a filesystem) and moved
onto the destination with :func:`os.replace`, which is atomic -- a reader sees
the old file or the new one, never half of either.

The schemas stay where they are; only this mechanism is shared.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager


@contextmanager
def atomic_path(path):
    """Yield a staging path; on a clean exit move it onto ``path``.

    The parent directory is created.  If the block raises, the staging file is
    removed and ``path`` is left exactly as it was.
    """
    path = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=parent, prefix=os.path.basename(path) + ".", suffix=".tmp",
        delete=False)
    handle.close()
    staging = handle.name
    try:
        yield staging
        os.replace(staging, path)
    except BaseException:
        try:
            os.unlink(staging)
        except OSError:
            pass
        raise
