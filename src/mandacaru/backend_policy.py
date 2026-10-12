# -*- coding: utf-8 -*-
# file: backend_policy.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The ``MANDACARU_BACKEND`` policy, read the same way by both native libraries.

Two compiled libraries sit behind Mandacaru: the integral kernels
(:mod:`mandacaru.integrals._backend`) and the radial kernels of atom and
pseudopotential generation (:mod:`mandacaru.basis.radial_backend`).  One
environment variable selects for both, so it has one parser: a typo such as
``MANDACARU_BACKEND=fortran`` is refused by either library rather than read
as ``auto`` by one of them.  The module imports nothing from the package,
because :mod:`mandacaru.integrals` already imports :mod:`mandacaru.basis` and
the parser cannot live in either without a cycle.
"""

from __future__ import annotations

import os

#: The values ``MANDACARU_BACKEND`` accepts.
BACKEND_POLICIES = ("auto", "c", "numpy")


def backend_preference() -> str:
    """The ``MANDACARU_BACKEND`` policy: ``"auto"`` (default), ``"c"`` or ``"numpy"``.

    ``"auto"`` prefers the C library and compiles it when it is missing;
    ``"c"`` does the same but raises when no C backend can be had;
    ``"numpy"`` forces the reference kernels (benchmarks, debugging).  Unset
    or blank means ``"auto"``; case and surrounding spaces are ignored.

    Raises
    ------
    ValueError
        For any other value.
    """
    value = os.environ.get("MANDACARU_BACKEND", "auto").strip().lower() or "auto"
    if value not in BACKEND_POLICIES:
        raise ValueError(
            f"MANDACARU_BACKEND={value!r} is not one of "
            + ", ".join(repr(policy) for policy in BACKEND_POLICIES))
    return value
