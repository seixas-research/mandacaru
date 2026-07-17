# -*- coding: utf-8 -*-
# file: integrals/_backend.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Binding to the high-performance C integral backend (``ctypes``, zero-copy).

Design
------
* **Zero-copy pointer passing.** NumPy ``complex128`` is bit-identical to C99
  ``double _Complex`` and ``float64`` to ``double``; ``numpy.ctypeslib.ndpointer``
  passes the array's data pointer straight to C with no marshalling or copy.
* **ABI-independent build.** We load a plain OpenMP shared library
  (``libcarcara_integrals``), *not* a CPython extension, so a single compiled
  artifact serves every Python version/ABI.
* **Graceful fallback.** If the library is not built, ``HAS_C_BACKEND`` is
  ``False`` and vectorized NumPy reference implementations are used instead, so
  the package is always importable and testable.

The backend is deliberately *basis-agnostic*: it consumes **sampled values**
``psi[i, :]`` on the grid, never analytic orbital forms.  Injecting Wannier (or
any other) functions therefore requires no change here.
"""

from __future__ import annotations

import ctypes
import os
from ctypes.util import find_library
from pathlib import Path

import numpy as np
from numpy.ctypeslib import ndpointer

_PKG_DIR = Path(__file__).resolve().parent

# ndpointer specs reused across signatures (C-contiguous, correct dtype).
_C128 = ndpointer(dtype=np.complex128, flags="C_CONTIGUOUS")
_C128_W = ndpointer(dtype=np.complex128, flags="C_CONTIGUOUS,WRITEABLE")
_F64 = ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")


def _find_library() -> str | None:
    """Locate ``libcarcara_integrals`` across the usual build locations."""
    names = ["libcarcara_integrals.dylib", "libcarcara_integrals.so",
             "carcara_integrals.dll"]
    candidates = []
    env = os.environ.get("CARCARA_INTEGRALS_LIB")
    if env:
        candidates.append(Path(env))
    search_dirs = [_PKG_DIR, _PKG_DIR / "csrc", _PKG_DIR / "csrc" / "build"]
    for d in search_dirs:
        candidates += [d / n for n in names]
    for c in candidates:
        if c.is_file():
            return str(c)
    found = find_library("carcara_integrals")
    return found


def _load():
    path = _find_library()
    if path is None:
        return None
    try:
        lib = ctypes.CDLL(path)
    except OSError:
        return None

    lib.carcara_one_body.restype = None
    lib.carcara_one_body.argtypes = [
        _C128,                       # psi   (M * ngrid)
        _F64,                        # Vext  (ngrid)
        ctypes.c_int,                # M
        ctypes.c_int,                # npts (points per dimension)
        ctypes.c_double,             # dx
        _C128_W,                     # out_T (M * M)
        _C128_W,                     # out_V (M * M)
    ]

    lib.carcara_two_body.restype = None
    lib.carcara_two_body.argtypes = [
        _C128,                       # psi (M * ngrid)
        _F64, _F64, _F64,            # xg, yg, zg (ngrid)
        ctypes.c_int,                # M
        ctypes.c_int,                # ngrid
        ctypes.c_double,             # dV
        ctypes.c_double,             # softening
        _C128_W,                     # out_eri (M^4)
    ]
    return lib


_LIB = _load()
HAS_C_BACKEND = _LIB is not None


# --------------------------------------------------------------------------- #
# Public API (C-accelerated when available, NumPy fallback otherwise).
# --------------------------------------------------------------------------- #

def _as_shape(shape):
    """Normalize a node-count spec to a ``(nx, ny, nz)`` tuple of ints."""
    if np.isscalar(shape):
        n = int(shape)
        return (n, n, n)
    nx, ny, nz = (int(s) for s in shape)
    return (nx, ny, nz)


def one_body_matrices(psi_stack, Vext, dx, shape):
    """Kinetic ``T`` and potential ``V`` matrices for ``M`` sampled functions.

    ``T[a, b] = <psi_a | -1/2 nabla^2 | psi_b>`` (finite-difference Laplacian),
    ``V[a, b] = <psi_a | Vext | psi_b>``.

    Parameters
    ----------
    psi_stack : (M, ngrid) complex128
        Row ``a`` holds ``psi_a`` sampled on the flattened grid.
    Vext : (ngrid,) float64
        External (e.g. electron-nuclear) potential sampled on the grid.
    dx : float
        Grid spacing (uniform across axes).  ``dV = dx**3``.
    shape : int or (int, int, int)
        Nodes per Cartesian axis.  A scalar means a cubic grid
        (``ngrid == shape**3``); a triple ``(nx, ny, nz)`` a non-cubic one
        (``ngrid == nx*ny*nz``).

    Notes
    -----
    The C backend assumes an equal node count on every axis, so it is used only
    for cubic grids; non-cubic grids fall back to the vectorized NumPy kernel
    (which handles any ``(nx, ny, nz)`` and is exercised by the tests).
    """
    psi_stack = np.ascontiguousarray(psi_stack, dtype=np.complex128)
    Vext = np.ascontiguousarray(Vext, dtype=np.float64)
    nx, ny, nz = _as_shape(shape)
    M = psi_stack.shape[0]
    T = np.zeros((M, M), dtype=np.complex128)
    V = np.zeros((M, M), dtype=np.complex128)

    if HAS_C_BACKEND and nx == ny == nz:
        _LIB.carcara_one_body(psi_stack.reshape(-1), Vext, M, int(nx),
                              float(dx), T.reshape(-1), V.reshape(-1))
        return T, V
    return _one_body_numpy(psi_stack, Vext, dx, (nx, ny, nz))


def two_body_tensor(psi_stack, xg, yg, zg, dV, softening=0.0):
    """Electron-repulsion tensor ``<ab|cd>`` for ``M`` sampled functions.

    ``eri[a, b, c, d] = ∫∫ psi_a*(1) psi_c(1) (1/r12) psi_b*(2) psi_d(2) dV1 dV2``
    -- physicists' notation, electron 1 carrying the index pair ``(a, c)`` and
    electron 2 the pair ``(b, d)``.  ``softening`` regularizes the ``r12 -> 0``
    node.
    """
    psi_stack = np.ascontiguousarray(psi_stack, dtype=np.complex128)
    xg = np.ascontiguousarray(xg, dtype=np.float64)
    yg = np.ascontiguousarray(yg, dtype=np.float64)
    zg = np.ascontiguousarray(zg, dtype=np.float64)
    M = psi_stack.shape[0]
    ngrid = psi_stack.shape[1]
    eri = np.zeros((M, M, M, M), dtype=np.complex128)

    if HAS_C_BACKEND:
        _LIB.carcara_two_body(psi_stack.reshape(-1), xg, yg, zg, M, ngrid,
                              float(dV), float(softening), eri.reshape(-1))
        return eri
    return _two_body_numpy(psi_stack, xg, yg, zg, dV, softening)


# --------------------------------------------------------------------------- #
# NumPy reference fallbacks (mirror the C kernels exactly).
# --------------------------------------------------------------------------- #

def _laplacian_fd(field3d, dx):
    """7-point finite-difference Laplacian; zero (decayed) outside the box."""
    lap = -6.0 * field3d
    for axis in range(3):
        lap += (np.roll(field3d, 1, axis) + np.roll(field3d, -1, axis))
        # zero out the wrapped boundary contribution
        sl_lo = [slice(None)] * 3
        sl_hi = [slice(None)] * 3
        sl_lo[axis] = 0
        sl_hi[axis] = -1
        lap[tuple(sl_lo)] -= np.roll(field3d, 1, axis)[tuple(sl_lo)]
        lap[tuple(sl_hi)] -= np.roll(field3d, -1, axis)[tuple(sl_hi)]
    return lap / (dx * dx)


def _one_body_numpy(psi_stack, Vext, dx, shape):
    M = psi_stack.shape[0]
    dV = dx ** 3
    T = np.zeros((M, M), dtype=np.complex128)
    V = np.zeros((M, M), dtype=np.complex128)
    shape = _as_shape(shape)
    lap = [_laplacian_fd(psi_stack[b].reshape(shape), dx).reshape(-1)
           for b in range(M)]
    for a in range(M):
        conj_a = np.conj(psi_stack[a])
        for b in range(M):
            T[a, b] = np.sum(conj_a * (-0.5 * lap[b])) * dV
            V[a, b] = np.sum(conj_a * Vext * psi_stack[b]) * dV
    return T, V


def _two_body_numpy(psi_stack, xg, yg, zg, dV, softening):
    M = psi_stack.shape[0]
    ngrid = psi_stack.shape[1]
    eri = np.zeros((M, M, M, M), dtype=np.complex128)
    # Coulomb potential of each density pair rho_bd, then contract with rho_ac.
    for b in range(M):
        for d in range(M):
            rho2 = np.conj(psi_stack[b]) * psi_stack[d]          # (ngrid,)
            phi = np.empty(ngrid, dtype=np.complex128)
            for i in range(ngrid):
                dxr = xg[i] - xg
                dyr = yg[i] - yg
                dzr = zg[i] - zg
                r12 = np.sqrt(dxr * dxr + dyr * dyr + dzr * dzr
                              + softening * softening)
                r12 = np.where(r12 < 1e-15, 1e-15, r12)
                phi[i] = np.sum(rho2 / r12) * dV
            for a in range(M):
                for c in range(M):
                    rho1 = np.conj(psi_stack[a]) * psi_stack[c]
                    eri[a, b, c, d] = np.sum(rho1 * phi) * dV
    return eri
