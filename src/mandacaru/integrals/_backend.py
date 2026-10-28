# -*- coding: utf-8 -*-
# file: integrals/_backend.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Array adapters to the Rust extension's exported native ABI.

The packaged extension supplies these kernels for optional PAW setup. It must
be installed before use; compilation belongs to the package build.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import numpy as np
from numpy.ctypeslib import ndpointer

from .. import backend_policy

_PKG_DIR = Path(__file__).resolve().parent

# ndpointer specs reused across signatures (C-contiguous, correct dtype).
_C128 = ndpointer(dtype=np.complex128, flags="C_CONTIGUOUS")
_C128_W = ndpointer(dtype=np.complex128, flags="C_CONTIGUOUS,WRITEABLE")
_C64 = ndpointer(dtype=np.complex64, flags="C_CONTIGUOUS")
_C64_W = ndpointer(dtype=np.complex64, flags="C_CONTIGUOUS,WRITEABLE")
_F64 = ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_F64_W = ndpointer(dtype=np.float64, flags="C_CONTIGUOUS,WRITEABLE")
_F32 = ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
_F32_W = ndpointer(dtype=np.float32, flags="C_CONTIGUOUS,WRITEABLE")


def _find_library():
    """Use the packaged Rust extension, with no system-library discovery."""
    backend_policy.backend_preference()
    from .. import _native
    return _native.__file__


def _load(fresh: bool = False):
    """Load and bind the library, or ``None``.

    ``fresh`` loads a private copy of the file: a process that already
    loaded a stale build at the same path would otherwise get that handle
    back from the dynamic loader, and a library just compiled to replace it
    could not be used before the next process.
    """
    path = _find_library()
    if path is None:
        return None
    origin = path
    if fresh:
        try:
            handle, copy = tempfile.mkstemp(prefix="mandacaru_integrals_",
                                            suffix=Path(path).suffix)
            os.close(handle)
            shutil.copyfile(path, copy)
            path = copy
        except OSError:
            return None
    try:
        lib = ctypes.CDLL(path)
    except OSError:
        return None
    finally:
        if fresh:
            try:
                os.unlink(path)       # the mapping outlives the file (POSIX)
            except OSError:
                pass
    try:
        bound = _bind(lib)
        bound._mandacaru_origin = origin
        if backend_policy.backend_preference() == "c" and hasattr(bound, "mandacaru_real_shell_f32"):
            return None
        return bound
    except AttributeError:
        # A stale library missing a newer symbol (e.g. mandacaru_one_body_general):
        # fall back to the NumPy kernels rather than crashing on import.
        return None


def _bind(lib):
    # Rust uses fixed-width i64; the historical C interface uses C long.
    index_type = ctypes.c_int64 if hasattr(lib, "mandacaru_backend_abi_version") else ctypes.c_long
    if hasattr(lib, "mandacaru_backend_abi_version"):
        lib.mandacaru_backend_abi_version.restype = ctypes.c_int
        lib.mandacaru_backend_abi_version.argtypes = []
        if lib.mandacaru_backend_abi_version() != 1:
            raise AttributeError("unsupported Mandacaru native ABI version")
    lib.mandacaru_one_body.restype = None
    lib.mandacaru_one_body.argtypes = [
        _C128,                       # psi   (M * ngrid)
        _F64,                        # Vext  (ngrid)
        ctypes.c_int,                # M
        ctypes.c_int,                # npts (points per dimension)
        ctypes.c_double,             # dx
        _C128_W,                     # out_T (M * M)
        _C128_W,                     # out_V (M * M)
    ]

    lib.mandacaru_one_body_general.restype = None
    lib.mandacaru_one_body_general.argtypes = [
        _C128,                       # psi   (M * ngrid)
        _F64,                        # Vext  (ngrid)
        ctypes.c_int,                # M
        ctypes.c_int,                # nx
        ctypes.c_int,                # ny
        ctypes.c_int,                # nz
        _F64,                        # ginv  (9, row-major inverse metric)
        ctypes.c_double,             # dV
        _C128_W,                     # out_T (M * M)
        _C128_W,                     # out_V (M * M)
    ]

    lib.mandacaru_laplacian_real.restype = None
    lib.mandacaru_laplacian_real.argtypes = [
        _F64,                        # field (n * ngrid)
        ndpointer(dtype=np.float64, flags="C_CONTIGUOUS,WRITEABLE"),  # lap
        ctypes.c_int,                # n
        ctypes.c_int,                # nx
        ctypes.c_int,                # ny
        ctypes.c_int,                # nz
        _F64,                        # ginv  (9, row-major inverse metric)
    ]

    lib.mandacaru_kb_project.restype = None
    lib.mandacaru_kb_project.argtypes = [
        _C128,                       # psi (M * ngrid)
        _C128,                       # chi (P * ngrid)
        ctypes.c_int,                # M
        ctypes.c_int,                # P
        index_type,               # ngrid
        ctypes.c_double,             # dV
        _C128_W,                     # out_P (M * P)
    ]

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

    if hasattr(lib, "mandacaru_real_shell"):
        lib.mandacaru_real_shell.restype = None
        lib.mandacaru_real_shell.argtypes = [
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
            _F64_W,                      # out (M * npts)
        ]

    if hasattr(lib, "mandacaru_real_shell_f32"):
        lib.mandacaru_real_shell_f32.restype = None
        lib.mandacaru_real_shell_f32.argtypes = [
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
            _F32_W,                      # out (M * npts)
        ]

    if hasattr(lib, "mandacaru_laplacian_real_f32"):
        lib.mandacaru_laplacian_real_f32.restype = None
        lib.mandacaru_laplacian_real_f32.argtypes = [
            _F32,                        # field (n * ngrid)
            _F32_W,                      # lap
            ctypes.c_int,                # n
            ctypes.c_int,                # nx
            ctypes.c_int,                # ny
            ctypes.c_int,                # nz
            _F64,                        # ginv  (9, row-major inverse metric)
        ]

    if hasattr(lib, "mandacaru_kb_project_f32"):
        lib.mandacaru_kb_project_f32.restype = None
        lib.mandacaru_kb_project_f32.argtypes = [
            _C64,                        # psi (M * ngrid)
            _C64,                        # chi (P * ngrid)
            ctypes.c_int,                # M
            ctypes.c_int,                # P
            index_type,               # ngrid
            ctypes.c_double,             # dV
            _C128_W,                     # out_P (M * P)
        ]

    lib.mandacaru_bloch_block.restype = index_type
    lib.mandacaru_bloch_block.argtypes = []
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

    _I32 = ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")

    if hasattr(lib, "mandacaru_localized_density_real"):
        lib.mandacaru_localized_density_real.restype = None
        lib.mandacaru_localized_density_real.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int,
            _I32, _I32, _I32,
            ctypes.POINTER(ctypes.c_void_p),
            _F64, ctypes.c_int,
            _F64_W,
        ]

    if hasattr(lib, "mandacaru_localized_potential_matrix_real"):
        lib.mandacaru_localized_potential_matrix_real.restype = None
        lib.mandacaru_localized_potential_matrix_real.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_double, ctypes.c_int,
            _I32, _I32, _I32,
            ctypes.POINTER(ctypes.c_void_p),
            _F64, ctypes.c_int,
            _F64_W,
        ]

    if hasattr(lib, "mandacaru_localized_cross_real"):
        lib.mandacaru_localized_cross_real.restype = None
        lib.mandacaru_localized_cross_real.argtypes = [
            ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_double, ctypes.c_int,
            _I32, _I32, _I32,
            _F64,
            ctypes.POINTER(ctypes.c_void_p),
            _F64, ctypes.c_int,
            _F64_W,
        ]

    if hasattr(lib, "mandacaru_spherical_bessel_matrix"):
        lib.mandacaru_spherical_bessel_matrix.restype = None
        lib.mandacaru_spherical_bessel_matrix.argtypes = [
            ctypes.c_int,
            _F64, ctypes.c_int,
            _F64, ctypes.c_int,
            _F64_W,
        ]

    if hasattr(lib, "mandacaru_fourier_bessel_transform"):
        lib.mandacaru_fourier_bessel_transform.restype = None
        lib.mandacaru_fourier_bessel_transform.argtypes = [
            ctypes.c_int,
            _F64, _F64, ctypes.c_int,
            _F64, ctypes.c_int,
            ctypes.c_int,
            _F64_W,
        ]

    if hasattr(lib, "mandacaru_two_center_s_table"):
        lib.mandacaru_two_center_s_table.restype = None
        lib.mandacaru_two_center_s_table.argtypes = [
            ctypes.c_int, ctypes.c_double,
            _F64, _F64, ctypes.c_int,
            _F64, ctypes.c_int,
            _F64_W,
        ]

    if hasattr(lib, "mandacaru_radial_coulomb"):
        lib.mandacaru_radial_coulomb.restype = ctypes.c_double
        lib.mandacaru_radial_coulomb.argtypes = [
            ctypes.c_int, ctypes.c_int,
            _F64, _F64, _F64,
        ]

    if hasattr(lib, "mandacaru_radial_poisson_solve"):
        lib.mandacaru_radial_poisson_solve.restype = None
        lib.mandacaru_radial_poisson_solve.argtypes = [
            ctypes.c_int, ctypes.c_int,
            _F64, _F64, _F64,
            _F64_W,
        ]

    if hasattr(lib, "mandacaru_radial_slater_tensor"):
        lib.mandacaru_radial_slater_tensor.restype = None
        lib.mandacaru_radial_slater_tensor.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            _F64, _F64, _F64,
            _F64_W,
        ]

    if hasattr(lib, "mandacaru_numerov_outward"):
        lib.mandacaru_numerov_outward.restype = None
        lib.mandacaru_numerov_outward.argtypes = [
            ctypes.c_int, _F64, _F64, ctypes.c_double, ctypes.c_int, _F64_W,
        ]

    if hasattr(lib, "mandacaru_numerov_inward"):
        lib.mandacaru_numerov_inward.restype = None
        lib.mandacaru_numerov_inward.argtypes = [
            ctypes.c_int, _F64, ctypes.c_double, ctypes.c_int, _F64_W,
        ]

    if hasattr(lib, "mandacaru_radial_abi_version"):
        lib.mandacaru_radial_abi_version.restype = ctypes.c_int
        lib.mandacaru_radial_abi_version.argtypes = []

    if hasattr(lib, "mandacaru_tridiagonal_eigenpair"):
        lib.mandacaru_tridiagonal_eigenpair.restype = ctypes.c_int
        lib.mandacaru_tridiagonal_eigenpair.argtypes = [
            ctypes.c_int, _F64, _F64, ctypes.c_int, ctypes.c_double,
            ctypes.POINTER(ctypes.c_double), _F64_W,
        ]

    return lib


_LIB = _load()
if _LIB is None:
    raise ImportError("the packaged Rust ABI could not be loaded; reinstall Mandacaru")
HAS_NATIVE_BACKEND = _LIB is not None


# --------------------------------------------------------------------------- #
# Backend check and on-demand compilation.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class BackendStatus:
    """Outcome of :func:`ensure_backend` / :func:`check_backend`.

    Attributes
    ----------
    available : bool
        ``True`` when the C library is loaded and the integrals run in C.
    path : str or None
        The shared library in use (``None`` on the NumPy fallback).
    n_threads : int or None
        OpenMP threads the C kernels run with (``None`` on the fallback).
    compiled : bool
        ``True`` when the library was compiled during *this* check.
    message : str
        One line saying how the backend was resolved (or why it was not).
    """
    available: bool
    path: str | None
    n_threads: int | None
    compiled: bool
    message: str

    @property
    def is_rust(self) -> bool:
        """True if the active backend is the Rust crate."""
        if not self.available or _LIB is None:
            return False
        return hasattr(_LIB, "mandacaru_real_shell_f32")

    @property
    def label(self) -> str:
        """Short backend label for summaries (``"Rust (8 threads)"`` or ``"C (OpenMP, 8 threads)"``)."""
        if not self.available:
            return "NumPy (backend unavailable)"
        name = "Rust" if self.is_rust else "C (OpenMP)"
        if self.n_threads and self.n_threads > 1:
            return f"{name} ({self.n_threads} threads)"
        return f"{name} (serial)"


def _reload(fresh: bool = False) -> bool:
    """Re-run the library search and rebind the module globals (``fresh``:
    from a private copy of the file, :func:`_load`)."""
    global _LIB, HAS_NATIVE_BACKEND
    _LIB = _load(fresh)
    HAS_NATIVE_BACKEND = _LIB is not None
    return HAS_NATIVE_BACKEND


def check_backend() -> BackendStatus:
    """Report which integral backend is in use, without trying to build."""
    if HAS_NATIVE_BACKEND:
        is_rust = hasattr(_LIB, "mandacaru_real_shell_f32")
        desc = "Rust backend loaded" if is_rust else "C backend loaded"
        path = getattr(_LIB, "_mandacaru_origin", _LIB._name)
        return BackendStatus(True, path, num_threads(), False, desc)
    return BackendStatus(False, None, None, False,
                         "C backend not built; NumPy reference kernels in use")


def ensure_backend(*, verbose=False):
    """Require the installed Rust ABI. Building belongs to package installation."""
    backend_policy.backend_preference()
    if not HAS_NATIVE_BACKEND or not check_backend().is_rust:
        raise RuntimeError("the packaged Rust extension is unavailable; reinstall Mandacaru")
    return check_backend()

def _unload() -> None:
    global _LIB, HAS_NATIVE_BACKEND
    _LIB = None
    HAS_NATIVE_BACKEND = False


def warn_fallback(status: BackendStatus) -> None:
    """Emit (once per process) the warning that the integrals run in NumPy."""
    global _fallback_warned
    if status.available or _fallback_warned:
        return
    _fallback_warned = True
    warnings.warn(
        "the real-space integrals are running on the NumPy reference kernels: "
        + status.message + ".  They are slower and use more memory than the "
        "C backend.  Run `mandacaru --build-backend` to compile it and see "
        "exactly which tool chain is missing.",
        RuntimeWarning, stacklevel=3)


# --------------------------------------------------------------------------- #
# BLAS beside the OpenMP kernels.
# --------------------------------------------------------------------------- #

def single_threaded_blas(function):
    """Run ``function`` with every loaded BLAS limited to one thread.

    The crystal workflow alternates small dense linear algebra -- ``M x M``
    eigensolves and products per k-point, ``M`` in the tens to hundreds --
    with this library's OpenMP kernels (Bloch sums, Bader).  A threaded BLAS
    loses on both counts there.  Measured with OpenBLAS on 12 cores / 24
    threads: it starts a team for matrices too small to split (a 26 x 26
    generalized ``eigh`` took 13 ms on 24 threads, 0.1 ms on one), and the
    Bloch kernel ran 1.8x slower beside a threaded BLAS, even with its
    threads lowered again between the two.  Other BLAS builds (MKL,
    Accelerate) are limited the same way but were not measured.

    The caller's setting (``OPENBLAS_NUM_THREADS`` included) is overridden
    for the call and restored afterwards, so nested calls are harmless.  The
    limit is applied per call rather than once at import, because NumPy and
    SciPy each load their own BLAS and the second may not be loaded yet.
    """

    @wraps(function)
    def wrapper(*args, **kwargs):
        return function(*args, **kwargs)

    return wrapper


def grid_blas_threads() -> int:
    """BLAS threads for a product over the grid where no OpenMP kernel runs.

    One per physical core: the products of the Kohn-Sham matrix and the
    density with the Bloch sums (``M x ngrid``) are large enough to split,
    and inside the SCF loop no OpenMP region waits for the cores.  Hardware
    threads beyond the cores lost (``M = 832``: 2.3 s on 12 threads, 2.9 s on
    24, 14.4 s on one).
    """
    import psutil

    return max(1, psutil.cpu_count(logical=False) or os.cpu_count() or 1)


# --------------------------------------------------------------------------- #
# Public API (C-accelerated when available, NumPy fallback otherwise).
# --------------------------------------------------------------------------- #

def num_threads() -> int | None:
    """OpenMP thread count of the C backend, or ``None`` when it is not built.

    ``None`` signals the vectorized NumPy fallback path (no explicit OpenMP
    parallelism managed here); a positive integer is the number of cores the C
    integral kernels run on.
    """
    if HAS_NATIVE_BACKEND and hasattr(_LIB, "mandacaru_num_threads"):
        return int(_LIB.mandacaru_num_threads())
    return None


def _as_shape(shape):
    """Normalize a node-count spec to a ``(nx, ny, nz)`` tuple of ints."""
    if np.isscalar(shape):
        n = int(shape)
        return (n, n, n)
    nx, ny, nz = (int(s) for s in shape)
    return (nx, ny, nz)


def _check_samples(psi_stack, ngrid: int, what: str) -> None:
    """Reject a sample stack whose buffer does not match the grid.

    The C kernels take raw pointers and a node count; a short buffer would be
    read out of bounds.  dtype/contiguity alone do not establish the length.
    """
    if psi_stack.ndim != 2 or psi_stack.shape[1] != ngrid:
        raise ValueError(
            f"{what}: expected samples of shape (M, {ngrid}) for this grid, "
            f"got {psi_stack.shape}")
    if not np.isfinite(psi_stack).all():
        raise ValueError(f"{what}: the sampled functions contain "
                         f"non-finite values")


def _check_filled(array, psi_stack, what: str):
    """Catch a C kernel that returned without writing (a failed allocation).

    The kernels are ``void`` and bail out on a failed ``malloc``, leaving the
    caller's zero-initialized output untouched -- which would read as perfectly
    valid zero integrals.  Non-zero samples cannot give an exactly zero result.
    """
    if np.any(psi_stack) and not np.any(array):
        raise MemoryError(
            f"the C integral backend returned an all-zero {what}: it most "
            f"likely could not allocate its workspace. Re-run with "
            f"MANDACARU_BACKEND=numpy, or use a coarser grid / fewer basis "
            f"functions.")
    return array




def kb_projections(psi_stack, chi_stack, dV):
    r"""Overlaps ``<phi_a|chi_p>`` of a basis against Kleinman-Bylander projectors.

    Returns an ``(M, P)`` complex array,
    ``out[a, p] = dV * sum_g conj(psi_a[g]) chi_p[g]``.

    This is the only grid work the nonlocal pseudopotential needs: the nonlocal
    matrix itself is the small outer product ``P diag(E_KB) P^dagger``, which is
    why the separable form costs ``O(M P)`` instead of the ``O(M^2)`` radial
    integrals a semilocal form would need.

    Uses the C backend when available; the NumPy fallback is a single BLAS
    ``gemm`` and is already close to optimal, so the two differ mainly in
    threading.  Real samples on both sides (a molecule's real harmonics)
    are one real ``gemm``, and a real ``(M, P)`` result.
    """
    if not (np.iscomplexobj(psi_stack) or np.iscomplexobj(chi_stack)):
        is_f32 = (psi_stack.dtype == np.float32 or chi_stack.dtype == np.float32)
        target_dtype = np.float32 if is_f32 else np.float64
        psi_stack = np.asarray(psi_stack, dtype=target_dtype)
        chi_stack = np.asarray(chi_stack, dtype=target_dtype)
        if chi_stack.shape[1] != psi_stack.shape[1]:
            raise ValueError(
                f"projectors are sampled on {chi_stack.shape[1]} points but "
                f"the basis on {psi_stack.shape[1]}")
        return (psi_stack @ chi_stack.T) * float(dV)
    is_c64 = (psi_stack.dtype == np.complex64 or chi_stack.dtype == np.complex64)
    if is_c64 and HAS_NATIVE_BACKEND and hasattr(_LIB, "mandacaru_kb_project_f32"):
        psi_stack = np.ascontiguousarray(psi_stack, dtype=np.complex64)
        chi_stack = np.ascontiguousarray(chi_stack, dtype=np.complex64)
        M, ngrid = psi_stack.shape
        P = chi_stack.shape[0]
        if chi_stack.shape[1] != ngrid:
            raise ValueError(
                f"projectors are sampled on {chi_stack.shape[1]} points but the "
                f"basis on {ngrid}")
        if P == 0:
            return np.zeros((M, 0), dtype=np.complex128)
        out = np.empty((M, P), dtype=np.complex128)
        _LIB.mandacaru_kb_project_f32(psi_stack, chi_stack, M, P, ngrid,
                                      float(dV), out)
        return out
    psi_stack = np.ascontiguousarray(psi_stack, dtype=np.complex128)
    chi_stack = np.ascontiguousarray(chi_stack, dtype=np.complex128)
    M, ngrid = psi_stack.shape
    P = chi_stack.shape[0]
    if chi_stack.shape[1] != ngrid:
        raise ValueError(
            f"projectors are sampled on {chi_stack.shape[1]} points but the "
            f"basis on {ngrid}")
    if P == 0:
        return np.zeros((M, 0), dtype=np.complex128)

    if HAS_NATIVE_BACKEND:
        out = np.empty((M, P), dtype=np.complex128)
        _LIB.mandacaru_kb_project(psi_stack, chi_stack, M, P, ngrid,
                                  float(dV), out)
        return out
    return (np.conj(psi_stack) @ chi_stack.T) * dV


def bloch_shell(points, center, translations, phases, support, spline, l,
                ms, rows, out, derivative: int = -1) -> bool:
    """Accumulate one shell's Bloch sums into ``out`` (``(nk, M, npts)``).

    ``spline`` is ``(breaks, coeffs, rc)``: the breakpoints and the
    ``(n - 1, 4)`` cubic coefficients of the radial function's spline, and
    the radius at which it is cut off.  ``derivative`` (0, 1, 2) sums the
    functions' Cartesian derivative along that axis instead.  Returns
    ``False`` -- having done nothing -- without the C backend; the caller
    then evaluates the shell itself.
    """
    if not HAS_NATIVE_BACKEND or not hasattr(_LIB, "mandacaru_bloch_shell"):
        return False
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
    """Evaluate real-valued spherical harmonics or Cartesian derivatives directly into float64 or float32 out array."""
    if not HAS_NATIVE_BACKEND:
        return False
    if out.dtype == np.float32 and hasattr(_LIB, "mandacaru_real_shell_f32"):
        x, y, z = (np.ascontiguousarray(c, dtype=np.float64) for c in points)
        breaks, coeffs, rc = spline
        _LIB.mandacaru_real_shell_f32(
            x, y, z, x.size, np.ascontiguousarray(center, dtype=np.float64),
            float(support), np.ascontiguousarray(breaks, dtype=np.float64),
            np.ascontiguousarray(coeffs, dtype=np.float64), len(breaks),
            float(rc), int(l), int(derivative),
            np.ascontiguousarray(ms, dtype=np.int32),
            np.ascontiguousarray(rows, dtype=np.int64), len(ms), out.shape[0],
            out)
        return True
    if hasattr(_LIB, "mandacaru_real_shell"):
        x, y, z = (np.ascontiguousarray(c, dtype=np.float64) for c in points)
        breaks, coeffs, rc = spline
        _LIB.mandacaru_real_shell(
            x, y, z, x.size, np.ascontiguousarray(center, dtype=np.float64),
            float(support), np.ascontiguousarray(breaks, dtype=np.float64),
            np.ascontiguousarray(coeffs, dtype=np.float64), len(breaks),
            float(rc), int(l), int(derivative),
            np.ascontiguousarray(ms, dtype=np.int32),
            np.ascontiguousarray(rows, dtype=np.int64), len(ms), out.shape[0],
            out)
        return True
    return False


def bloch_project(points, center, translations, support, spline, l, ms,
                  columns, derivative: int = -1):
    """``(n_blocks, n_images, nm, P)`` image-resolved projections of one
    shell on ``columns`` (``(P, npts)``): ``sum_j conj(phi_{R,i}(r_j))
    columns[p, j]`` per block of :func:`bloch_block` consecutive points and
    lattice image (``translations``); zero for an image that misses the
    block.  The caller adds the blocks and applies the k-point phases.
    ``None`` without the C backend."""
    if not HAS_NATIVE_BACKEND or not hasattr(_LIB, "mandacaru_bloch_project"):
        return None
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


def bloch_block() -> int:
    """Points per block of the Bloch kernels (``bloch_project`` slices)."""
    return int(_LIB.mandacaru_bloch_block())


def bader_ascent(starts, node_gradient, periodic, A, A_inv, origin, lattice,
                 positions, cores, step_length, capture, max_steps,
                 stall_window, stall_distance):
    """Climb from every index in ``starts`` (``(n, 3)``); see
    ``csrc/mandacaru_bader.c``.  ``cores`` is a per-atom list of ``None`` or
    ``(r, slope, support)``.  Returns ``(owner, final_index)`` -- the atom
    each trajectory was captured by, or -1 with where it stopped -- or
    ``None`` without the C backend."""
    if not HAS_NATIVE_BACKEND or not hasattr(_LIB, "mandacaru_bader_ascent"):
        return None
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


# --------------------------------------------------------------------------- #
# NumPy reference fallbacks (mirror the C kernels exactly).
# --------------------------------------------------------------------------- #





#: Functions whose finite-difference Laplacians are formed together by the
#: real one-body path (bounds its working set to this many grids).
REAL_LAPLACIAN_BLOCK = 32
