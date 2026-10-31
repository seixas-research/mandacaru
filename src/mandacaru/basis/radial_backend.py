"""Typed adapters to the packaged Rust radial kernels."""

from __future__ import annotations
import numpy as np
from .. import _native


def tridiagonal_eigenpair(diag, off, k, guess=None, *, context=None):
    """Return the ordered eigenpair; the native solver chooses its own bracket."""
    ctx = _native.Context(threads=1) if context is None else context
    return ctx.eigenpair(
        np.ascontiguousarray(diag, dtype=float),
        np.ascontiguousarray(off, dtype=float),
        k,
    )


def numerov_outward_kernel(f, s, h2, start, u, *, context=None):
    """Evaluate in Rust, copy the result into the caller's solution array."""
    ctx = _native.Context(threads=1) if context is None else context
    result = ctx.numerov(
        np.ascontiguousarray(f, dtype=float),
        np.ascontiguousarray(s, dtype=float),
        np.ascontiguousarray(u, dtype=float),
        h2,
        start,
    )
    np.copyto(u, result, casting="same_kind")
    return u


def numerov_inward_kernel(f, h2, stop, u, *, context=None):
    ctx = _native.Context(threads=1) if context is None else context
    result = ctx.numerov(
        np.ascontiguousarray(f, dtype=float),
        np.zeros_like(f, dtype=float),
        np.ascontiguousarray(u, dtype=float),
        h2,
        stop,
        inward=True,
    )
    np.copyto(u, result, casting="same_kind")
    return u
