# -*- coding: utf-8 -*-
# file: utils/memory.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Memory the process may hold and memory it can take now.

Reads the machine (``sysconf``, ``/proc/meminfo``) and the memory cgroup the
process runs in (a container, a batch job); no numerics.  The QPE memory
guard (:meth:`~mandacaru.workflows.qpe.QuantumPhaseEstimation.memory_estimate`) compares its
estimate against :func:`available_memory_bytes`.
"""

from __future__ import annotations

import os


def _installed_memory_bytes() -> int:
    """Installed memory of this machine (``sysconf``), or 16 GiB if unknown."""
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return 16 * 2 ** 30


def _read_bytes(path) -> int | None:
    """An integer file of the cgroup interface, ``None`` for ``max``/absent."""
    try:
        with open(path) as handle:
            text = handle.read().strip()
    except OSError:
        return None
    if not text or text == "max":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _membership(lines):
    """``(controllers, path)`` of the first line of ``/proc/self/cgroup`` that
    names the v2 hierarchy (no controllers) or the v1 memory controller."""
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) == 3 and (parts[1] == "" or "memory" in parts[1].split(",")):
            return parts[1], parts[2]
    return None


def _v2_limits(root: str, path: str) -> list[int]:
    """Every ``memory.max`` from the process's group up to the root."""
    limits = []
    current = os.path.join(root, path) if path else root
    while True:
        value = _read_bytes(os.path.join(current, "memory.max"))
        if value is not None:
            limits.append(value)
        if os.path.normpath(current) == os.path.normpath(root):
            return limits
        current = os.path.dirname(current)


def cgroup_memory(root: str = "/sys/fs/cgroup",
                  membership: str = "/proc/self/cgroup"):
    """``(limit, usage)`` of this process's memory cgroup, in bytes.

    cgroup v2 (``memory.max`` and ``memory.current``, the tightest limit of
    the process's group and its ancestors) or v1 (``memory.limit_in_bytes``,
    ``memory.usage_in_bytes``); ``(None, None)`` outside a limited group.  A
    v1 "no limit" (a page-rounded ``2^63``) is reported as ``None``.
    """
    try:
        with open(membership) as handle:
            found = _membership(handle.read().splitlines())
    except OSError:
        return None, None
    if found is None:
        return None, None
    controllers, path = found
    path = path.strip("/")
    if controllers == "":                                       # v2
        limits = _v2_limits(root, path)
        usage = _read_bytes(os.path.join(root, path, "memory.current"))
        return (min(limits) if limits else None), usage
    base = os.path.join(root, "memory", path)                   # v1
    limit = _read_bytes(os.path.join(base, "memory.limit_in_bytes"))
    if limit is not None and limit >= 2 ** 62:
        limit = None
    return limit, _read_bytes(os.path.join(base, "memory.usage_in_bytes"))


def memory_limit_bytes() -> int:
    """Memory this process may hold: the installed memory, capped by its
    cgroup's limit (a container, a batch job).

    A property of where the run is placed, not of the moment: what chooses a
    representation reads this, so the choice does not change with the
    machine's load.
    """
    limit, _usage = cgroup_memory()
    installed = _installed_memory_bytes()
    return installed if limit is None else min(installed, limit)


def available_memory_bytes() -> int:
    """Memory that can be allocated now, in bytes.

    ``MemAvailable`` of ``/proc/meminfo`` (free plus reclaimable cache: what
    the kernel can hand out without swapping), capped by what the process's
    cgroup has left (limit minus usage).  :func:`memory_limit_bytes` where
    neither can be read.  The memory guards compare against this, not the
    installed memory: other processes' memory is not there to take.
    """
    candidates = []
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    candidates.append(int(line.split()[1]) * 1024)
                    break
    except (OSError, ValueError, IndexError):
        pass
    limit, usage = cgroup_memory()
    if limit is not None:
        candidates.append(max(0, limit - (usage or 0)))
    return min(candidates) if candidates else memory_limit_bytes()
