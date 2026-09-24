r"""C backend of the radial kernels behind atom and pseudopotential generation.

Profiling an iron PAW-LCAO generation (84 s) puts 67 % of it in
:func:`scipy.linalg.eigh_tridiagonal` -- the k-th eigenpair of the
uniform-grid radial equation, solved thousands of times inside the
relativistic :math:`M(\varepsilon)` loop -- and 19 % in the two Numerov
recursions, which were Python loops.  ``csrc/mandacaru_radial.c`` implements
those three kernels; this module compiles it on first use (the system C
compiler, no dependencies), loads it with :mod:`ctypes`, and routes

* :func:`tridiagonal_eigenpair` -- the k-th eigenvalue and unit eigenvector of
  a symmetric tridiagonal matrix (Sturm bisection to full precision, then
  inverse iteration);
* :func:`numerov_outward_kernel` / :func:`numerov_inward_kernel` -- the
  recursions of :func:`mandacaru.pseudopotentials.oncv.numerov_outward` and
  ``_numerov_inward``, same arithmetic in the same order.

Every kernel keeps its Python reference implementation, used when the library
cannot be built and selectable with ``MANDACARU_BACKEND=numpy`` (the switch
the integral backend already reads).  ``mandacaru-build --backend`` reports
which one is in use.
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
from numpy.ctypeslib import ndpointer

#: Must match ``MANDACARU_RADIAL_ABI`` in ``csrc/mandacaru_radial.h``.
ABI_VERSION = 2

_SRC_DIR = Path(__file__).resolve().parent / "csrc"
_BUILD_DIR = _SRC_DIR / "build"
_SOURCE = _SRC_DIR / "mandacaru_radial.c"
_LIB_NAME = {"Darwin": "libmandacaru_radial.dylib",
             "Windows": "mandacaru_radial.dll"}.get(platform.system(),
                                                    "libmandacaru_radial.so")
_F64 = ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_F64_W = ndpointer(dtype=np.float64, flags="C_CONTIGUOUS,WRITEABLE")

_LIB = None
_attempted = False             # one build attempt per process
_last_message = "not loaded yet"


def _preference() -> str:
    value = os.environ.get("MANDACARU_BACKEND", "auto").strip().lower() or "auto"
    return value if value in ("auto", "c", "numpy") else "auto"


def _bind(lib):
    lib.mandacaru_radial_abi_version.restype = ctypes.c_int
    lib.mandacaru_radial_abi_version.argtypes = []
    if lib.mandacaru_radial_abi_version() != ABI_VERSION:
        raise AttributeError("stale radial library")
    lib.mandacaru_numerov_outward.restype = None
    lib.mandacaru_numerov_outward.argtypes = [
        ctypes.c_int, _F64, _F64, ctypes.c_double, ctypes.c_int, _F64_W]
    lib.mandacaru_numerov_inward.restype = None
    lib.mandacaru_numerov_inward.argtypes = [
        ctypes.c_int, _F64, ctypes.c_double, ctypes.c_int, _F64_W]
    lib.mandacaru_tridiagonal_eigenpair.restype = ctypes.c_int
    lib.mandacaru_tridiagonal_eigenpair.argtypes = [
        ctypes.c_int, _F64, _F64, ctypes.c_int, ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), _F64_W]
    return lib


def _load(path: Path):
    try:
        return _bind(ctypes.CDLL(str(path)))
    except (OSError, AttributeError):
        return None


def build_radial_backend(*, verbose: bool = False) -> Path | None:
    """Compile ``libmandacaru_radial`` into ``csrc/build``; ``None`` on failure.

    Plain C11 and ``libm``: the system compiler is all it takes.  The log of
    the attempt is ``csrc/build/radial_build.log``.
    """
    global _last_message
    system = platform.system()
    shared = "-dynamiclib" if system == "Darwin" else "-shared"
    try:
        _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _last_message = f"cannot create {_BUILD_DIR}: {exc}"
        return None
    target = _BUILD_DIR / _LIB_NAME
    log = _BUILD_DIR / "radial_build.log"
    with open(log, "w") as out:
        for name in (os.environ.get("CC"), "cc", "clang", "gcc"):
            compiler = shutil.which(name) if name else None
            if compiler is None:
                continue
            cmd = [compiler, "-std=c11", "-O2", "-fPIC", shared,
                   f"-I{_SRC_DIR}", str(_SOURCE), "-o", str(target), "-lm"]
            out.write("$ " + " ".join(cmd) + "\n")
            if verbose:
                print("[mandacaru] " + " ".join(cmd))
            try:
                run = subprocess.run(cmd, capture_output=True, text=True,
                                     timeout=300, check=False)
            except (OSError, subprocess.SubprocessError) as exc:
                out.write(f"failed to run: {exc}\n")
                continue
            out.write(run.stdout + run.stderr)
            if run.returncode == 0 and target.is_file():
                _last_message = f"compiled with {compiler}"
                return target
            out.write(f"exit status {run.returncode}\n")
    _last_message = f"no C compiler could build it; see {log}"
    return None


def _library(build: bool = True):
    """The loaded library, building it once per process when it is missing."""
    global _LIB, _attempted, _last_message
    if _preference() == "numpy":
        _last_message = "MANDACARU_BACKEND=numpy: Python reference kernels"
        return None
    if _LIB is not None:
        return _LIB
    target = _BUILD_DIR / _LIB_NAME
    stale = (target.is_file()
             and target.stat().st_mtime < _SOURCE.stat().st_mtime)
    if target.is_file() and not stale:
        _LIB = _load(target)
    if _LIB is None and build and not _attempted:
        _attempted = True
        built = build_radial_backend()
        if built is not None:
            _LIB = _load(built)
    if _LIB is None:
        if _preference() == "c":
            raise RuntimeError(f"MANDACARU_BACKEND=c but the radial C backend "
                               f"is unavailable: {_last_message}")
        if _last_message == "not loaded yet":
            _last_message = "C library unavailable: Python reference kernels"
    else:
        _last_message = f"C backend loaded from {target}"
    return _LIB


def radial_backend_status(build: bool = True) -> tuple[bool, str]:
    """``(uses_c, message)`` -- which radial kernels are in use, and why."""
    lib = _library(build=build)
    return lib is not None, _last_message


# --------------------------------------------------------------------------- #
# The kernels.
# --------------------------------------------------------------------------- #

def tridiagonal_eigenpair(diag, off, k: int, guess: float | None = None):
    """``(value, vector)``: the ``k``-th smallest eigenvalue (0-based) of the
    symmetric tridiagonal matrix ``(diag, off)`` and its unit eigenvector.

    ``guess`` -- the previous eigenvalue of a self-consistent loop -- only
    says where to start looking; the answer does not depend on it.  The sign
    of the vector is arbitrary, as it is for
    :func:`scipy.linalg.eigh_tridiagonal`; every caller fixes it.
    """
    diag = np.ascontiguousarray(diag, dtype=np.float64)
    off = np.ascontiguousarray(off, dtype=np.float64)
    lib = _library()
    if lib is None:
        from scipy.linalg import eigh_tridiagonal
        values, vectors = eigh_tridiagonal(diag, off, select="i",
                                           select_range=(int(k), int(k)))
        return float(values[0]), vectors[:, 0]
    vector = np.empty_like(diag)
    value = ctypes.c_double(0.0)
    status = lib.mandacaru_tridiagonal_eigenpair(
        int(diag.size), diag, off, int(k),
        float("nan") if guess is None else float(guess),
        ctypes.byref(value), vector)
    if status != 0:
        raise RuntimeError(f"mandacaru_tridiagonal_eigenpair failed ({status})")
    return float(value.value), vector


def numerov_outward_kernel(f, s, h2: float, start: int, u) -> np.ndarray:
    """Run the outward recursion in place on ``u`` (seeded at ``start`` and
    ``start + 1``) and return it."""
    lib = _library()
    if lib is None or not u.flags.c_contiguous:
        a = 1.0 - h2 * f / 12.0
        b = 2.0 * (1.0 + 5.0 * h2 * f / 12.0)
        c = h2 / 12.0
        for i in range(start + 1, u.size - 1):
            u[i + 1] = (b[i] * u[i] - a[i - 1] * u[i - 1]
                        + c * (s[i + 1] + 10.0 * s[i] + s[i - 1])) / a[i + 1]
        return u
    lib.mandacaru_numerov_outward(
        int(u.size), np.ascontiguousarray(f, dtype=np.float64),
        np.ascontiguousarray(s, dtype=np.float64), float(h2), int(start), u)
    return u


def numerov_inward_kernel(f, h2: float, stop: int, u) -> np.ndarray:
    """Run the homogeneous inward recursion in place on ``u`` (seeded at its
    last two points) down to ``stop`` and return it."""
    lib = _library()
    if lib is None or not u.flags.c_contiguous:
        a = 1.0 - h2 * f / 12.0
        b = 2.0 * (1.0 + 5.0 * h2 * f / 12.0)
        for i in range(u.size - 2, stop, -1):
            u[i - 1] = (b[i] * u[i] - a[i + 1] * u[i + 1]) / a[i - 1]
        return u
    lib.mandacaru_numerov_inward(
        int(u.size), np.ascontiguousarray(f, dtype=np.float64), float(h2),
        int(stop), u)
    return u
