# -*- coding: utf-8 -*-
# file: integrals/_backend.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Array adapters to the Rust extension's exported native ABI.

The packaged extension supplies these kernels for PAW setup. It must be
installed before use; compilation belongs to the package build. Importing this
module binds every kernel below or raises: a missing extension file, or an
extension that loads but lacks a kernel, is a ``RuntimeError`` naming the
reinstall, never a silent switch to another calculation.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from functools import wraps

import numpy as np
from numpy.ctypeslib import ndpointer

from .. import backend_policy

# ndpointer specs reused across signatures (C-contiguous, correct dtype).
_C128 = ndpointer(dtype=np.complex128, flags="C_CONTIGUOUS")
_C128_W = ndpointer(dtype=np.complex128, flags="C_CONTIGUOUS,WRITEABLE")
_F64 = ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_F64_W = ndpointer(dtype=np.float64, flags="C_CONTIGUOUS,WRITEABLE")
_F32_W = ndpointer(dtype=np.float32, flags="C_CONTIGUOUS,WRITEABLE")


def _find_library():
    """Use the packaged Rust extension, with no system-library discovery."""
    backend_policy.backend_preference()
    from .. import _native
    return _native.__file__


def _load():
    """Load and bind the packaged extension; raise ``RuntimeError`` if it cannot serve."""
    path = _find_library()
    try:
        lib = ctypes.CDLL(path)
    except OSError as error:
        raise RuntimeError(
            f"the native extension {path} could not be opened ({error}); "
            "reinstall Mandacaru") from error
    try:
        bound = _bind(lib)
    except AttributeError as error:
        # A stale extension that lacks a required kernel cannot serve this adapter.
        raise RuntimeError(
            f"the native extension {path} loaded but its kernels did not bind "
            f"({error}); reinstall Mandacaru") from error
    bound._mandacaru_origin = path
    return bound


def _real_shell_argtypes(out_type):
    """The argument types of ``mandacaru_real_shell`` (``f32`` differs by ``out_type`` alone)."""
    index_type = ctypes.c_int64
    return [
        _F64, _F64, _F64,            # x, y, z (npts)
        index_type,               # npts
        _F64,                        # center (3)
        ctypes.c_double,             # support
        _F64,                        # breaks (nb)
        _F64,                        # coeffs ((nb - 1) * 4)
        ctypes.c_int,                # nb
        ctypes.c_double,             # rc (zero at r >= rc)
        ctypes.c_int,                # l
        ctypes.c_int,                # derivative (-1, or the axis)
        ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),   # ms (nm)
        ndpointer(dtype=np.int64, flags="C_CONTIGUOUS"),   # rows (nm)
        ctypes.c_int,                # nm
        index_type,               # M
        out_type,                    # out (M * npts)
    ]


def _bind(lib):
    # The Rust ABI uses fixed-width i64 indices.
    index_type = ctypes.c_int64
    lib.mandacaru_backend_abi_version.restype = ctypes.c_int
    lib.mandacaru_backend_abi_version.argtypes = []
    if lib.mandacaru_backend_abi_version() != 1:
        raise AttributeError("unsupported Mandacaru native ABI version")

    lib.mandacaru_bloch_shell.restype = None
    lib.mandacaru_bloch_shell.argtypes = [
        _F64, _F64, _F64,            # x, y, z (npts)
        index_type,               # npts
        _F64,                        # center (3)
        _F64,                        # translations (n_images * 3)
        ctypes.c_int,                # n_images
        _C128,                       # phases (nk * n_images)
        ctypes.c_int,                # nk
        ctypes.c_double,             # support
        _F64,                        # breaks (nb)
        _F64,                        # coeffs ((nb - 1) * 4)
        ctypes.c_int,                # nb
        ctypes.c_double,             # rc (zero at r >= rc)
        ctypes.c_int,                # l
        ctypes.c_int,                # derivative (-1, or the axis)
        ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),   # ms (nm)
        ndpointer(dtype=np.int64, flags="C_CONTIGUOUS"),   # rows (nm)
        ctypes.c_int,                # nm
        index_type,               # M
        _C128_W,                     # out (nk * M * npts)
    ]

    for kernel, out_type in ((lib.mandacaru_real_shell, _F64_W),
                             (lib.mandacaru_real_shell_f32, _F32_W)):
        kernel.restype = None
        kernel.argtypes = _real_shell_argtypes(out_type)

    lib.mandacaru_bloch_project.restype = None
    lib.mandacaru_bloch_project.argtypes = [
        _F64, _F64, _F64,            # x, y, z (npts)
        index_type,               # npts
        _F64,                        # center (3)
        _F64,                        # translations (n_images * 3)
        ctypes.c_int,                # n_images
        ctypes.c_double,             # support
        _F64,                        # breaks (nb)
        _F64,                        # coeffs ((nb - 1) * 4)
        ctypes.c_int,                # nb
        ctypes.c_double,             # rc (zero at r >= rc)
        ctypes.c_int,                # l
        ctypes.c_int,                # derivative (-1, or the axis)
        ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),   # ms (nm)
        ctypes.c_int,                # nm
        _C128,                       # cols (P * npts)
        ctypes.c_int,                # P
        _C128_W,                     # out (n_blocks * n_images * nm * P)
    ]

    _I32_W = ndpointer(dtype=np.int32, flags="C_CONTIGUOUS,WRITEABLE")
    _I64 = ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
    lib.mandacaru_bader_ascent.restype = None
    lib.mandacaru_bader_ascent.argtypes = [
        _F64, index_type,         # starts (n * 3), n
        _F64, _F64, _F64,            # node gradient, index components
        ctypes.c_int, ctypes.c_int, ctypes.c_int,   # grid shape
        ctypes.c_int,                # periodic
        _F64, _F64, _F64,            # A, A_inv (3 x 3), origin (3)
        _F64, _F64,                  # lattice, its inverse (3 x 3)
        _F64, _F64, ctypes.c_int,    # positions, fractional, n_atoms
        _I64, _F64, _F64, _F64,      # core offsets, r, slope, support
        ctypes.c_double, ctypes.c_double,           # step, capture
        ctypes.c_int, ctypes.c_int, ctypes.c_double,  # steps, window, stall
        _I32_W, _F64_W,              # owner (n), final index (n * 3)
    ]

    lib.mandacaru_num_threads.restype = ctypes.c_int
    lib.mandacaru_num_threads.argtypes = []

    return lib


_LIB = _load()
# Constant: ``_load`` raised above unless every kernel bound. The flag stays
# only for its three callers (sampling.py, periodic_paw.py, test_profiling.py);
# W4-C removes it with the sampling fallback.
HAS_NATIVE_BACKEND = True


# --------------------------------------------------------------------------- #
# Backend check.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class BackendStatus:
    """Outcome of :func:`ensure_backend` / :func:`check_backend`.

    Attributes
    ----------
    available : bool
        ``True`` when the packaged Rust extension is loaded.
    path : str
        The extension file in use.
    n_threads : int
        Threads the Rust kernels run with.
    message : str
        One line saying how the backend was resolved.
    """
    available: bool
    path: str
    n_threads: int
    message: str

    @property
    def label(self) -> str:
        """Short backend label for summaries (``"Rust (8 threads)"``)."""
        if self.n_threads and self.n_threads > 1:
            return f"Rust ({self.n_threads} threads)"
        return "Rust (serial)"


def check_backend() -> BackendStatus:
    """Report the packaged Rust extension in use."""
    return BackendStatus(True, _LIB._mandacaru_origin, num_threads(),
                         "Rust backend loaded")


def ensure_backend():
    """Return the bound Rust extension's status; ``_load`` already raised if it could not bind.

    Building belongs to package installation."""
    backend_policy.backend_preference()
    return check_backend()


# --------------------------------------------------------------------------- #
# BLAS beside the Rust kernels.
# --------------------------------------------------------------------------- #

def single_threaded_blas(function):
    """Return ``function`` unchanged (identity wrapper).

    The wrapper once limited every loaded BLAS to one thread beside the native
    kernels. It limits nothing now; it stays only because
    ``pseudopotentials/periodic_paw.py`` still decorates two functions with it,
    and it is deleted with that file's numerics (WP-G).
    """

    @wraps(function)
    def wrapper(*args, **kwargs):
        return function(*args, **kwargs)

    return wrapper


def grid_blas_threads() -> int:
    """BLAS threads for a product over the grid, one per physical core.

    The products of the Kohn-Sham matrix and the density with the Bloch sums
    (``M x ngrid``) are large enough to split, and inside the SCF loop no
    native kernel waits for the cores.  Hardware threads beyond the cores lost
    (``M = 832``: 2.3 s on 12 threads, 2.9 s on 24, 14.4 s on one).
    """
    import psutil

    return max(1, psutil.cpu_count(logical=False) or os.cpu_count() or 1)


# --------------------------------------------------------------------------- #
# Public API.
# --------------------------------------------------------------------------- #

def num_threads() -> int:
    """Thread count of the Rust kernels."""
    return int(_LIB.mandacaru_num_threads())


def bloch_shell(points, center, translations, phases, support, spline, l,
                ms, rows, out, derivative: int = -1) -> bool:
    """Accumulate one shell's Bloch sums into ``out`` (``(nk, M, npts)``).

    ``spline`` is ``(breaks, coeffs, rc)``: the breakpoints and the
    ``(n - 1, 4)`` cubic coefficients of the radial function's spline, and
    the radius at which it is cut off.  ``derivative`` (0, 1, 2) sums the
    functions' Cartesian derivative along that axis instead.  Always returns
    ``True``: the kernel is bound at import, so there is no "not done" case.
    The flag stays for the sampling fallback that W4-C removes.
    """
    x, y, z = (np.ascontiguousarray(c, dtype=np.float64) for c in points)
    breaks, coeffs, rc = spline
    translations = np.ascontiguousarray(translations, dtype=np.float64)
    phases = np.ascontiguousarray(phases, dtype=np.complex128)
    _LIB.mandacaru_bloch_shell(
        x, y, z, x.size, np.ascontiguousarray(center, dtype=np.float64),
        translations, len(translations), phases, phases.shape[0],
        float(support), np.ascontiguousarray(breaks, dtype=np.float64),
        np.ascontiguousarray(coeffs, dtype=np.float64), len(breaks),
        float(rc), int(l), int(derivative),
        np.ascontiguousarray(ms, dtype=np.int32),
        np.ascontiguousarray(rows, dtype=np.int64), len(ms), out.shape[1],
        out)
    return True


def real_shell(points, center, support, spline, l, ms, rows, out,
               derivative: int = -1) -> bool:
    """Evaluate real spherical harmonics or Cartesian derivatives into ``out``.

    ``out`` is ``(M, npts)`` and ``float64`` or ``float32``; any other dtype
    raises ``TypeError``. Always returns ``True`` (the kernels are bound at
    import); the flag stays for the sampling fallback that W4-C removes.
    """
    if out.dtype == np.float32:
        kernel = _LIB.mandacaru_real_shell_f32
    elif out.dtype == np.float64:
        kernel = _LIB.mandacaru_real_shell
    else:
        raise TypeError(f"real_shell writes float64 or float32, not {out.dtype}")
    x, y, z = (np.ascontiguousarray(c, dtype=np.float64) for c in points)
    breaks, coeffs, rc = spline
    kernel(
        x, y, z, x.size, np.ascontiguousarray(center, dtype=np.float64),
        float(support), np.ascontiguousarray(breaks, dtype=np.float64),
        np.ascontiguousarray(coeffs, dtype=np.float64), len(breaks),
        float(rc), int(l), int(derivative),
        np.ascontiguousarray(ms, dtype=np.int32),
        np.ascontiguousarray(rows, dtype=np.int64), len(ms), out.shape[0],
        out)
    return True


def bloch_project(points, center, translations, support, spline, l, ms,
                  columns, derivative: int = -1):
    """``(n_blocks, n_images, nm, P)`` image-resolved projections of one
    shell on ``columns`` (``(P, npts)``): ``sum_j conj(phi_{R,i}(r_j))
    columns[p, j]`` per block of consecutive points and lattice image
    (``translations``); zero for an image that misses the block.  The
    caller adds the blocks and applies the k-point phases."""
    x, y, z = (np.ascontiguousarray(c, dtype=np.float64) for c in points)
    breaks, coeffs, rc = spline
    translations = np.ascontiguousarray(translations, dtype=np.float64)
    columns = np.ascontiguousarray(columns, dtype=np.complex128)
    ms = np.ascontiguousarray(ms, dtype=np.int32)
    out = np.zeros((len(translations), len(ms), len(columns)),
                   dtype=np.complex128)
    _LIB.mandacaru_bloch_project(
        x, y, z, x.size, np.ascontiguousarray(center, dtype=np.float64),
        translations, len(translations), float(support),
        np.ascontiguousarray(breaks, dtype=np.float64),
        np.ascontiguousarray(coeffs, dtype=np.float64), len(breaks),
        float(rc), int(l), int(derivative), ms, len(ms), columns,
        len(columns), out)
    return out


def bader_ascent(starts, node_gradient, periodic, A, A_inv, origin, lattice,
                 positions, cores, step_length, capture, max_steps,
                 stall_window, stall_distance):
    """Climb from every index in ``starts`` (``(n, 3)``); the kernel is
    ``mandacaru-kernels/src/bader.rs``.  ``cores`` is a per-atom list of
    ``None`` or ``(r, slope, support)``.  Returns ``(owner, final_index)`` --
    the atom each trajectory was captured by, or -1 with where it stopped."""
    f64 = lambda a: np.ascontiguousarray(a, dtype=np.float64)  # noqa: E731
    starts = f64(starts).reshape(-1, 3)
    shape = node_gradient[0].shape
    positions = f64(positions).reshape(-1, 3)
    if periodic:
        L = f64(lattice)
        L_inv = f64(np.linalg.inv(L))
        fractional = f64(positions @ L_inv.T)
    else:
        L = L_inv = f64(np.eye(3))
        fractional = positions
    offsets, rs, slopes, supports = [0], [], [], []
    for core in cores:
        if core is None:
            supports.append(0.0)
        else:
            r, slope, support = core
            rs.append(f64(r))
            slopes.append(f64(slope))
            supports.append(float(support))
        offsets.append(offsets[-1] + (0 if core is None else len(core[0])))
    core_r = f64(np.concatenate(rs)) if rs else np.zeros(1)
    core_slope = f64(np.concatenate(slopes)) if slopes else np.zeros(1)
    owner = np.empty(len(starts), dtype=np.int32)
    final = np.empty((len(starts), 3))
    _LIB.mandacaru_bader_ascent(
        starts, len(starts), *(f64(g) for g in node_gradient),
        shape[0], shape[1], shape[2], int(bool(periodic)), f64(A),
        f64(A_inv), f64(origin), L, L_inv, positions, fractional,
        len(positions), np.asarray(offsets, dtype=np.int64), core_r,
        core_slope, f64(supports), float(step_length), float(capture),
        int(max_steps), int(stall_window), float(stall_distance), owner,
        final)
    return owner, final
