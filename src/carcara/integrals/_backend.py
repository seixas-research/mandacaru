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
import platform
import shutil
import subprocess
import time
import warnings
from ctypes.util import find_library
from dataclasses import dataclass
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
    try:
        return _bind(lib)
    except AttributeError:
        # A stale library missing a newer symbol (e.g. carcara_one_body_general):
        # fall back to the NumPy kernels rather than crashing on import.
        return None


def _bind(lib):
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

    lib.carcara_one_body_general.restype = None
    lib.carcara_one_body_general.argtypes = [
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

    lib.carcara_kb_project.restype = None
    lib.carcara_kb_project.argtypes = [
        _C128,                       # psi (M * ngrid)
        _C128,                       # chi (P * ngrid)
        ctypes.c_int,                # M
        ctypes.c_int,                # P
        ctypes.c_long,               # ngrid
        ctypes.c_double,             # dV
        _C128_W,                     # out_P (M * P)
    ]

    lib.carcara_num_threads.restype = ctypes.c_int
    lib.carcara_num_threads.argtypes = []
    return lib


_LIB = _load()
HAS_C_BACKEND = _LIB is not None


# --------------------------------------------------------------------------- #
# Backend check and on-demand compilation.
# --------------------------------------------------------------------------- #

_SRC_DIR = _PKG_DIR / "csrc"
_BUILD_DIR = _SRC_DIR / "build"
_LIB_NAMES = {"Darwin": "libcarcara_integrals.dylib",
              "Windows": "carcara_integrals.dll"}
_BUILD_TIMEOUT = 600.0        # seconds; a compile that runs longer is abandoned
_build_attempted = False       # one compile attempt per process
_fallback_warned = False       # one NumPy-fallback warning per process


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
    def label(self) -> str:
        """Short backend label for summaries (``"C (OpenMP, 8 threads)"``)."""
        if not self.available:
            return "NumPy (C backend unavailable)"
        if self.n_threads and self.n_threads > 1:
            return f"C (OpenMP, {self.n_threads} threads)"
        return "C (serial)"


def backend_preference() -> str:
    """The ``CARCARA_BACKEND`` policy: ``"auto"`` (default), ``"c"`` or ``"numpy"``.

    ``"auto"`` prefers the C library and compiles it when it is missing;
    ``"c"`` does the same but raises when no C backend can be had;
    ``"numpy"`` forces the reference NumPy kernels (benchmarks, debugging).
    """
    value = os.environ.get("CARCARA_BACKEND", "auto").strip().lower() or "auto"
    if value not in ("auto", "c", "numpy"):
        raise ValueError(
            f"CARCARA_BACKEND={value!r} is not one of 'auto', 'c', 'numpy'")
    return value


def _reload() -> bool:
    """Re-run the library search and rebind the module globals."""
    global _LIB, HAS_C_BACKEND
    _LIB = _load()
    HAS_C_BACKEND = _LIB is not None
    return HAS_C_BACKEND


def _c_compiler() -> str | None:
    for name in (os.environ.get("CC"), "cc", "clang", "gcc"):
        if name and shutil.which(name):
            return shutil.which(name)
    return None


def _homebrew_libomp() -> Path | None:
    """Homebrew's ``libomp`` prefix on macOS, or ``None``."""
    brew = shutil.which("brew")
    if brew is None:
        return None
    try:
        out = subprocess.run([brew, "--prefix", "libomp"], capture_output=True,
                             text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    prefix = Path(out.stdout.strip())
    if out.returncode == 0 and (prefix / "lib").is_dir():
        return prefix
    return None


def _compile_attempts(system: str, build_dir: Path) -> list[list[list[str]]]:
    """The ways to build the library on this system, best first.

    Each attempt is a list of commands run in order; :func:`build_backend`
    moves to the next attempt when one fails.  CMake is preferred (it carries
    the OpenMP discovery for every platform).  Without CMake the bare
    compiler is used: first with its native ``-fopenmp`` (GCC, LLVM clang),
    then -- on macOS, where Apple's clang ships no OpenMP runtime -- through
    Homebrew's ``libomp``, and finally serial.  Returns an empty list when no
    tool chain is found.
    """
    cmake = shutil.which("cmake")
    if cmake is not None:
        configure = [cmake, "-S", str(_SRC_DIR), "-B", str(build_dir),
                     "-DCMAKE_BUILD_TYPE=Release"]
        if system == "Darwin":
            libomp = _homebrew_libomp()
            if libomp is not None:
                configure.append(f"-DOpenMP_ROOT={libomp}")
        return [[configure,
                 [cmake, "--build", str(build_dir), "--config", "Release"]]]

    cc = _c_compiler()
    if cc is None or system == "Windows":
        return []
    source = str(_SRC_DIR / "carcara_integrals.c")
    common = [cc, "-std=c11", "-O3", "-ffast-math", "-funroll-loops", "-fPIC",
              f"-I{_SRC_DIR}", source]
    if system == "Darwin":
        link = ["-dynamiclib", "-o", str(build_dir / _LIB_NAMES["Darwin"])]
        attempts = [[common + ["-fopenmp"] + link]]
        libomp = _homebrew_libomp()
        if libomp is not None:
            attempts.append([common + [
                "-Xpreprocessor", "-fopenmp", f"-I{libomp / 'include'}",
                f"-L{libomp / 'lib'}", "-lomp",
                "-Wl,-rpath," + str(libomp / "lib")] + link])
        attempts.append([common + link])                      # serial
        return attempts
    link = ["-shared", "-o", str(build_dir / "libcarcara_integrals.so"), "-lm"]
    return [[common + ["-fopenmp"] + link], [common + link]]


def _run_attempt(commands, log, timeout, verbose) -> bool:
    for cmd in commands:
        log.write("$ " + " ".join(cmd) + "\n")
        if verbose:
            print("[carcara] " + " ".join(cmd))
        try:
            run = subprocess.run(cmd, cwd=str(_SRC_DIR), capture_output=True,
                                 text=True, timeout=timeout, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            log.write(f"failed to run: {exc}\n")
            return False
        log.write(run.stdout)
        log.write(run.stderr)
        if run.returncode != 0:
            log.write(f"exit status {run.returncode}\n")
            return False
    return True


def build_backend(build_dir: str | os.PathLike | None = None, *,
                  verbose: bool = False,
                  timeout: float = _BUILD_TIMEOUT) -> Path | None:
    """Compile ``libcarcara_integrals`` for this machine.

    Detects the platform and tool chain (CMake when installed, otherwise the
    system C compiler; OpenMP through Homebrew's ``libomp`` on macOS, natively
    elsewhere) and builds the shared library into ``build_dir`` (default:
    ``csrc/build``, where the loader looks).  A full log is written to
    ``<build_dir>/build.log``.

    Returns
    -------
    Path or None
        The compiled library, or ``None`` when no tool chain was found or the
        compile failed (the log says why).  Never raises for a build failure.
    """
    system = platform.system()
    build_dir = Path(build_dir) if build_dir is not None else _BUILD_DIR
    try:
        build_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        if verbose:
            print(f"[carcara] cannot create {build_dir}: {exc}")
        return None
    attempts = _compile_attempts(system, build_dir)
    log_path = build_dir / "build.log"
    with open(log_path, "w") as log:
        log.write(f"# carcara C backend build on {system} "
                  f"{platform.machine()}, {time.ctime()}\n")
        if not attempts:
            log.write("no tool chain found: install cmake or a C compiler\n")
            if verbose:
                print("[carcara] no C tool chain found (cmake / cc)")
            return None
        for n, commands in enumerate(attempts, 1):
            log.write(f"# attempt {n}/{len(attempts)}\n")
            if _run_attempt(commands, log, timeout, verbose):
                break
        else:
            if verbose:
                print(f"[carcara] build failed, see {log_path}")
            return None
    for name in (_LIB_NAMES.get(system, "libcarcara_integrals.so"),
                 "libcarcara_integrals.so", "libcarcara_integrals.dylib"):
        candidate = build_dir / name
        if candidate.is_file():
            return candidate
    return None


def check_backend() -> BackendStatus:
    """Report which integral backend is in use, without trying to build."""
    if HAS_C_BACKEND:
        return BackendStatus(True, _find_library(), num_threads(), False,
                             "C backend loaded")
    return BackendStatus(False, None, None, False,
                         "C backend not built; NumPy reference kernels in use")


def ensure_backend(*, build: bool = True, verbose: bool = False) -> BackendStatus:
    """Make sure the C integral backend is available, compiling it if needed.

    Call this before integrating.  The policy is ``CARCARA_BACKEND``
    (:func:`backend_preference`): under ``"auto"`` (default) and ``"c"`` a
    missing or stale library is compiled once per process with
    :func:`build_backend` and loaded on the spot; the NumPy reference kernels
    are used only when that compile fails (``"c"`` raises instead).
    ``"numpy"`` skips the C library altogether.

    Returns
    -------
    BackendStatus
    """
    global _build_attempted
    preference = backend_preference()
    if preference == "numpy":
        if HAS_C_BACKEND:
            _unload()
        return BackendStatus(False, None, None, False,
                             "CARCARA_BACKEND=numpy: reference kernels forced")
    if HAS_C_BACKEND or _reload():
        return check_backend()
    compiled = False
    message = "C backend not built"
    if build and not _build_attempted:
        _build_attempted = True
        path = build_backend(verbose=verbose)
        if path is not None and _reload():
            compiled = True
            return BackendStatus(True, _find_library(), num_threads(), True,
                                 f"C backend compiled into {path.parent}")
        message = ("C backend compile failed" if path is None
                   else f"compiled library at {path} could not be loaded")
        message += f" (see {_BUILD_DIR / 'build.log'})"
    elif build:
        message = f"C backend compile already failed (see {_BUILD_DIR / 'build.log'})"
    if preference == "c":
        raise RuntimeError(f"{message}; CARCARA_BACKEND=c refuses the NumPy fallback")
    return BackendStatus(False, None, None, compiled,
                         message + "; NumPy reference kernels in use")


def _unload() -> None:
    global _LIB, HAS_C_BACKEND
    _LIB = None
    HAS_C_BACKEND = False


def warn_fallback(status: BackendStatus) -> None:
    """Emit (once per process) the warning that the integrals run in NumPy."""
    global _fallback_warned
    if status.available or _fallback_warned:
        return
    _fallback_warned = True
    warnings.warn(
        "the real-space integrals are running on the NumPy reference kernels: "
        + status.message + ".  They are slower and use more memory than the "
        "C backend; build it with `carcara.integrals.build_backend()` or "
        "`cmake -S src/carcara/integrals/csrc -B src/carcara/integrals/csrc/build"
        " && cmake --build src/carcara/integrals/csrc/build`.",
        RuntimeWarning, stacklevel=3)


# --------------------------------------------------------------------------- #
# Public API (C-accelerated when available, NumPy fallback otherwise).
# --------------------------------------------------------------------------- #

def num_threads() -> int | None:
    """OpenMP thread count of the C backend, or ``None`` when it is not built.

    ``None`` signals the vectorized NumPy fallback path (no explicit OpenMP
    parallelism managed here); a positive integer is the number of cores the C
    integral kernels run on.
    """
    if HAS_C_BACKEND and hasattr(_LIB, "carcara_num_threads"):
        return int(_LIB.carcara_num_threads())
    return None


def _as_shape(shape):
    """Normalize a node-count spec to a ``(nx, ny, nz)`` tuple of ints."""
    if np.isscalar(shape):
        n = int(shape)
        return (n, n, n)
    nx, ny, nz = (int(s) for s in shape)
    return (nx, ny, nz)


def one_body_matrices(psi_stack, Vext, grid):
    """Kinetic ``T`` and potential ``V`` matrices for ``M`` sampled functions.

    ``T[a, b] = <psi_a | -1/2 nabla^2 | psi_b>`` (finite-difference Laplacian),
    ``V[a, b] = <psi_a | Vext | psi_b>``.

    Parameters
    ----------
    psi_stack : (M, ngrid) complex128
        Row ``a`` holds ``psi_a`` sampled on the flattened grid.
    Vext : (ngrid,) float64
        External (e.g. electron-nuclear) potential sampled on the grid.
    grid : Grid
        The integration grid.  Its geometry -- ``shape``, ``dx``, the inverse
        metric ``metric_inverse()`` and the voxel volume ``dV`` -- selects the
        kernel: a cubic grid uses the fast ``carcara_one_body`` C path; any
        anisotropic (per-axis spacing) or non-orthogonal grid uses the general
        ``carcara_one_body_general`` C path.  Without the C library the
        vectorized NumPy kernel (which mirrors both C kernels) is used instead.
    """
    psi_stack = np.ascontiguousarray(psi_stack, dtype=np.complex128)
    Vext = np.ascontiguousarray(Vext, dtype=np.float64)
    nx, ny, nz = grid.shape
    M = psi_stack.shape[0]
    T = np.zeros((M, M), dtype=np.complex128)
    V = np.zeros((M, M), dtype=np.complex128)
    ginv = np.ascontiguousarray(grid.metric_inverse(), dtype=np.float64)
    dV = float(grid.dV)

    if HAS_C_BACKEND and grid.is_cubic:
        _LIB.carcara_one_body(psi_stack.reshape(-1), Vext, M, int(nx),
                              float(grid.dx), T.reshape(-1), V.reshape(-1))
        return T, V
    if HAS_C_BACKEND:
        _LIB.carcara_one_body_general(
            psi_stack.reshape(-1), Vext, M, int(nx), int(ny), int(nz),
            ginv.reshape(-1), dV, T.reshape(-1), V.reshape(-1))
        return T, V
    return _one_body_numpy(psi_stack, Vext, ginv, dV, (nx, ny, nz))


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
    threading.
    """
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

    if HAS_C_BACKEND:
        out = np.empty((M, P), dtype=np.complex128)
        _LIB.carcara_kb_project(psi_stack, chi_stack, M, P, ngrid,
                                float(dV), out)
        return out
    return (np.conj(psi_stack) @ chi_stack.T) * dV


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

def _shifted(field3d, axis, direction):
    """``field`` shifted by one node along ``axis``, zero-filled at the boundary.

    ``direction=+1`` fetches the ``+e_axis`` neighbor (so index 0 loses its lower
    neighbor); this mirrors the C kernel's "out-of-range neighbor is 0" rule.
    """
    out = np.roll(field3d, -direction, axis)
    sl = [slice(None)] * 3
    sl[axis] = -1 if direction > 0 else 0     # the wrapped face -> 0
    out[tuple(sl)] = 0.0
    return out


def _laplacian_general(field3d, ginv):
    """General FD Laplacian ``sum_{a,b} ginv[a,b] d_a d_b f`` (mirrors the C kernel).

    ``ginv`` is the 3x3 inverse metric ``(step^T step)^{-1}``; for an orthorhombic
    grid it is ``diag(1/dx^2, 1/dy^2, 1/dz^2)`` and only the diagonal (7-point)
    terms survive.  Out-of-range neighbors are treated as zero.
    """
    diag = -2.0 * (ginv[0, 0] + ginv[1, 1] + ginv[2, 2])
    lap = diag * field3d
    for a in range(3):                                   # diagonal 2nd derivatives
        g = ginv[a, a]
        if g != 0.0:
            lap += g * (_shifted(field3d, a, +1) + _shifted(field3d, a, -1))
    for a, b in ((0, 1), (0, 2), (1, 2)):                # mixed (cross) terms
        g = ginv[a, b]
        if g != 0.0:
            c = 0.5 * g                                  # 2*g_ab * 1/4
            pp = _shifted(_shifted(field3d, a, +1), b, +1)
            pm = _shifted(_shifted(field3d, a, +1), b, -1)
            mp = _shifted(_shifted(field3d, a, -1), b, +1)
            mm = _shifted(_shifted(field3d, a, -1), b, -1)
            lap += c * (pp - pm - mp + mm)
    return lap


def _one_body_numpy(psi_stack, Vext, ginv, dV, shape):
    M = psi_stack.shape[0]
    ginv = np.asarray(ginv, dtype=float)
    T = np.zeros((M, M), dtype=np.complex128)
    V = np.zeros((M, M), dtype=np.complex128)
    shape = _as_shape(shape)
    lap = [_laplacian_general(psi_stack[b].reshape(shape), ginv).reshape(-1)
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
