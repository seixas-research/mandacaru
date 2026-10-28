# -*- coding: utf-8 -*-
# file: test/test_backend_build.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The C integral backend is checked -- and compiled on demand -- before any
integration.

* :func:`check_backend` / :func:`ensure_backend` report a
  :class:`BackendStatus` consistent with the loaded library;
* a missing library triggers exactly one compile attempt per process, and a
  successful compile is loaded on the spot (``compiled=True``);
* a failed compile falls back to the NumPy kernels with one
  ``RuntimeWarning`` from the engine, and ``MANDACARU_BACKEND=c`` refuses the
  fallback;
* ``MANDACARU_BACKEND=numpy`` forces the reference kernels without building;
* :func:`build_backend` really compiles the library for this machine when a
  tool chain exists (CMake, or the bare compiler when CMake is hidden), and a
  build into the default directory is loaded on the spot;
* ``mandacaru --build-backend`` compiles on demand from the shell and reports
  what happened, exiting non-zero when only the NumPy kernels are left.
"""

import shutil
import subprocess
import sys
import sysconfig
import warnings
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_info, threadpool_limits

from mandacaru.basis import HydrogenicAtomicOrbital
from mandacaru.cli import main
from mandacaru.integrals import (BackendStatus, Grid, IntegralEngine, _backend,
                                 build_backend, check_backend, ensure_backend)


@pytest.fixture
def backend_state(monkeypatch):
    """Snapshot/restore the module globals the check mutates."""
    saved = (_backend._LIB, _backend.HAS_C_BACKEND, _backend._build_attempted,
             _backend._fallback_warned)
    monkeypatch.delenv("MANDACARU_BACKEND", raising=False)
    yield
    (_backend._LIB, _backend.HAS_C_BACKEND, _backend._build_attempted,
     _backend._fallback_warned) = saved


def _engine():
    return IntegralEngine([HydrogenicAtomicOrbital(1, 0, 0)],
                          Grid(box_size=4.0, grid_spacing=1.0, center=(0, 0, 0)))


class TestStatus:
    def test_check_matches_loaded_library(self, backend_state):
        status = check_backend()
        assert isinstance(status, BackendStatus)
        assert status.available == _backend.HAS_C_BACKEND
        assert status.compiled is False
        if status.available:
            assert status.path and status.path.endswith(
                ("libmandacaru-kernels.dylib", "libmandacaru-kernels.so",
                 "mandacaru-kernels.dll", "_native" + sysconfig.get_config_var("EXT_SUFFIX")))
            assert status.n_threads >= 1
            assert status.label.startswith(("C (", "Rust ("))
        else:
            assert status.path is None and status.n_threads is None
            assert status.label.startswith("NumPy")

    def test_engine_records_the_status_it_integrates_with(self, backend_state):
        eng = _engine()
        assert eng.backend_status.available == _backend.HAS_C_BACKEND
        assert eng.uses_c_backend == eng.backend_status.available
        assert eng.timings.backend == eng.backend_status.label

    def test_numpy_policy_forces_the_reference_kernels(self, backend_state,
                                                       monkeypatch):
        monkeypatch.setenv("MANDACARU_BACKEND", "numpy")
        monkeypatch.setattr(_backend, "build_backend",
                            lambda *a, **k: pytest.fail("must not build"))
        status = ensure_backend()
        assert status.available is False and status.compiled is False
        assert _backend.HAS_C_BACKEND is False
        eng = _engine()
        assert eng.uses_c_backend is False
        T, V = eng.one_body(lambda x, y, z: -1.0 / np.maximum(
            np.sqrt(x * x + y * y + z * z), 0.5), energy_units="Ha")
        assert np.isfinite(T).all()

    def test_invalid_policy_is_rejected(self, backend_state, monkeypatch):
        monkeypatch.setenv("MANDACARU_BACKEND", "fortran")
        with pytest.raises(ValueError, match="MANDACARU_BACKEND"):
            ensure_backend()


class TestCompileOnDemand:
    def _hide_library(self, monkeypatch):
        _backend._LIB = None
        _backend.HAS_C_BACKEND = False
        _backend._build_attempted = False
        _backend._fallback_warned = False
        monkeypatch.setattr(_backend, "_find_library", lambda: None)

    def test_missing_library_is_compiled_and_loaded(self, backend_state,
                                                    monkeypatch):
        real = check_backend()
        if not real.available:
            pytest.skip("needs a compiled library to stand in for the build")
        self._hide_library(monkeypatch)
        calls = []

        def fake_build(*args, **kwargs):
            calls.append(1)
            monkeypatch.setattr(_backend, "_find_library", lambda: real.path)
            return __import__("pathlib").Path(real.path)

        monkeypatch.setattr(_backend, "build_backend", fake_build)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            eng = _engine()                 # no fallback warning
        assert eng.backend_status.available and eng.backend_status.compiled
        assert eng.backend_status.n_threads >= 1
        assert calls == [1]
        _engine()                           # second engine: no rebuild
        assert calls == [1]

    def test_failed_compile_falls_back_once_with_a_warning(self, backend_state,
                                                           monkeypatch):
        self._hide_library(monkeypatch)
        calls = []
        monkeypatch.setattr(_backend, "build_backend",
                            lambda *a, **k: calls.append(1))
        with pytest.warns(RuntimeWarning, match="NumPy reference kernels"):
            eng = _engine()
        assert eng.backend_status.available is False
        assert "compile failed" in eng.backend_status.message
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            eng2 = _engine()                # warned once, tried once
        assert calls == [1]
        assert "already failed" in eng2.backend_status.message

    def test_c_policy_refuses_the_fallback(self, backend_state, monkeypatch):
        self._hide_library(monkeypatch)
        monkeypatch.setenv("MANDACARU_BACKEND", "c")
        monkeypatch.setattr(_backend, "build_backend", lambda *a, **k: None)
        with pytest.raises(RuntimeError, match="refuses the NumPy fallback"):
            ensure_backend()


def _threads_in_fresh_process(path) -> int:
    """Load ``path`` in a subprocess and return its thread count.

    A separate process, because a library built by a different tool chain
    may link a *different* OpenMP runtime than the one already loaded here,
    and two OpenMP runtimes in one process abort.
    """
    code = ("import ctypes, sys; from mandacaru.integrals import _backend; "
            f"lib = _backend._bind(ctypes.CDLL({str(path)!r})); "
            "print(lib.mandacaru_num_threads())")
    run = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=120, check=False)
    assert run.returncode == 0, run.stderr
    return int(run.stdout.strip())


class TestRealBuild:
    @pytest.fixture
    def toolchain(self):
        if shutil.which("cmake") is None and _backend._c_compiler() is None:
            pytest.skip("no C tool chain on this machine")

    def test_build_backend_produces_a_loadable_library(self, toolchain,
                                                       tmp_path):
        path = build_backend(tmp_path)
        assert path is not None and path.is_file()
        assert (tmp_path / "build.log").read_text().startswith("# mandacaru")
        assert _threads_in_fresh_process(path) >= 1

    def test_bare_compiler_fallback_when_cmake_is_absent(self, toolchain,
                                                         tmp_path, monkeypatch):
        if _backend._c_compiler() is None:
            pytest.skip("no bare C compiler")
        real_which = shutil.which
        monkeypatch.setattr(shutil, "which",
                            lambda n, *a, **k: None if n == "cmake"
                            else real_which(n, *a, **k))
        path = build_backend(tmp_path)
        assert path is not None and path.is_file()
        log = (tmp_path / "build.log").read_text()
        assert "cmake" not in log.split("\n", 2)[1]      # the bare compiler
        assert _threads_in_fresh_process(path) >= 1

    def test_no_toolchain_returns_none_with_a_log(self, tmp_path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda *a, **k: None)
        monkeypatch.delenv("CC", raising=False)
        assert build_backend(tmp_path) is None
        assert "no tool chain" in (tmp_path / "build.log").read_text()


class TestSyncedTree:
    """A source tree shared between macOS and Linux (Dropbox, a shared home).

    Both platforms' libraries sit in ``csrc/build`` and the CMake cache names
    whichever path configured it last.  The loader must take only its own
    platform's library, and the build must drop a cache written for another
    path instead of failing the CMake attempt.
    """

    def test_foreign_library_is_not_picked(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MANDACARU_INTEGRALS_LIB", raising=False)
        monkeypatch.setattr(_backend, "_PKG_DIR", tmp_path)
        build = tmp_path / "csrc" / "build"
        build.mkdir(parents=True)
        native = _backend._native_lib_name()
        for name in ("libmandacaru-kernels.dylib", "libmandacaru-kernels.so",
                     "mandacaru-kernels.dll"):
            if name != native:
                (build / name).write_bytes(b"foreign")
        monkeypatch.setattr(_backend, "find_library", lambda name: None)
        assert _backend._find_library() is None
        (build / native).write_bytes(b"native")
        assert _backend._find_library() == str(build / native)

    def _cache(self, build_dir, home):
        build_dir.mkdir(parents=True, exist_ok=True)
        (build_dir / "CMakeFiles").mkdir(exist_ok=True)
        (build_dir / "CMakeCache.txt").write_text(
            f"CMAKE_CACHEFILE_DIR:INTERNAL={build_dir}\n"
            f"CMAKE_HOME_DIRECTORY:INTERNAL={home}\n")

    def test_cache_from_another_path_is_discarded(self, tmp_path):
        self._cache(tmp_path, "/Users/elsewhere/csrc")
        with open(tmp_path / "build.log", "w") as log:
            _backend._discard_foreign_cmake_cache(tmp_path, log)
        assert not (tmp_path / "CMakeCache.txt").exists()
        assert not (tmp_path / "CMakeFiles").exists()
        assert "discarding" in (tmp_path / "build.log").read_text()

    def test_own_cache_is_kept(self, tmp_path):
        self._cache(tmp_path, _backend._SRC_DIR)
        with open(tmp_path / "build.log", "w") as log:
            _backend._discard_foreign_cmake_cache(tmp_path, log)
        assert (tmp_path / "CMakeCache.txt").is_file()
        assert (tmp_path / "CMakeFiles").is_dir()


class TestBuildLoadsWhatItBuilt:
    """A build into the default directory leaves the C backend *in use*.

    ``build_backend()`` used to compile and return a path without loading it,
    so ``build_backend(); check_backend()`` reported "not built" -- which read
    as a failed build in CI scripts and in the shell.
    """

    def _staged(self, monkeypatch, target, real_path, *, default=None):
        """A fake tool chain whose 'build' drops the real library in ``target``.

        ``default`` is what the loader treats as its own build directory.
        """
        target.mkdir(parents=True, exist_ok=True)
        library = target / _backend._native_lib_name()
        shutil.copy(real_path, library)
        monkeypatch.setattr(_backend, "_BUILD_DIR",
                            target if default is None else default)
        monkeypatch.setattr(_backend, "_compile_attempts",
                            lambda system, build_dir: [[[sys.executable, "-c", ""]]])
        _backend._LIB = None
        _backend.HAS_C_BACKEND = False
        return library

    def test_default_directory_build_is_loaded(self, backend_state, tmp_path,
                                               monkeypatch):
        real = check_backend()
        if not real.available:
            pytest.skip("needs a compiled library to stand in for the build")
        library = self._staged(monkeypatch, tmp_path, real.path)
        monkeypatch.setattr(_backend, "_find_library", lambda: real.path)

        assert build_backend() == library
        assert _backend.HAS_C_BACKEND is True
        assert check_backend().available is True

    def test_other_directory_build_leaves_the_globals_alone(self, backend_state,
                                                            tmp_path,
                                                            monkeypatch):
        real = check_backend()
        if not real.available:
            pytest.skip("needs a compiled library to stand in for the build")
        elsewhere = tmp_path / "elsewhere"
        self._staged(monkeypatch, elsewhere, real.path,
                     default=tmp_path / "default")
        monkeypatch.setattr(_backend, "_find_library",
                            lambda: pytest.fail("must not search"))

        assert build_backend(elsewhere) is not None
        assert _backend.HAS_C_BACKEND is False


class TestARebuiltLibraryReplacesAStaleOne:
    """A build that replaces a stale library already loaded in the process
    is loaded from a private copy: the dynamic loader would hand back the
    stale handle for the same path."""

    def test_a_fresh_load_binds_a_copy_and_leaves_no_file(self, backend_state):
        import tempfile

        real = check_backend()
        if not real.available:
            pytest.skip("needs a compiled library")
        before = set(Path(tempfile.gettempdir()).glob("mandacaru-kernels_*"))
        lib = _backend._load(fresh=True)
        assert lib is not None and lib is not _backend._LIB
        assert hasattr(lib, "mandacaru_laplacian_real")
        after = set(Path(tempfile.gettempdir()).glob("mandacaru-kernels_*"))
        assert after <= before


class TestTheRealLaplacian:
    """``mandacaru_laplacian_real``: the one-body kernels' stencil on real
    samples, equal to the NumPy reference to round-off and independent of
    the OpenMP thread count."""

    METRICS = {
        "cubic": np.diag([1 / 0.3 ** 2] * 3),
        "anisotropic": np.diag([1 / 0.3 ** 2, 1 / 0.25 ** 2, 1 / 0.35 ** 2]),
        "skewed": np.linalg.inv(np.array([[0.3, 0.05, 0.0],
                                          [0.05, 0.3, 0.02],
                                          [0.0, 0.02, 0.28]]) ** 2 + 0.0),
    }

    @pytest.mark.parametrize("metric", sorted(METRICS))
    def test_it_is_the_numpy_stencil(self, metric):
        if not check_backend().available:
            pytest.skip("needs the compiled library")
        ginv = self.METRICS[metric]
        ginv = 0.5 * (ginv + ginv.T)
        shape = (7, 6, 5)
        samples = np.random.default_rng(1).normal(size=(4, 7 * 6 * 5))
        got = _backend.laplacian_real(samples, shape, ginv)
        reference = np.stack([_backend._laplacian_general(
            row.reshape(shape), ginv).reshape(-1) for row in samples])
        assert np.abs(got - reference).max() < 1e-13 * np.abs(reference).max()

    def test_the_thread_count_does_not_change_it(self):
        if not check_backend().available:
            pytest.skip("needs the compiled library")
        ginv = self.METRICS["skewed"]
        ginv = 0.5 * (ginv + ginv.T)
        samples = np.random.default_rng(2).normal(size=(5, 9 * 8 * 7))
        results = []
        for threads in (1, 2, 3):
            with threadpool_limits(limits=threads, user_api="openmp"):
                results.append(_backend.laplacian_real(samples, (9, 8, 7),
                                                       ginv))
        assert all(np.array_equal(results[0], other) for other in results)

    def test_the_real_one_body_matrices_match_the_numpy_path(
            self, backend_state):
        """``kinetic="fd"`` on real functions: the C stencil and the NumPy
        reference give the same ``T`` and ``V`` to round-off."""
        if not check_backend().available:
            pytest.skip("needs the compiled library")
        from types import SimpleNamespace

        rng = np.random.default_rng(3)
        shape = (8, 7, 6)
        step = np.array([[0.3, 0.02, 0.0], [0.0, 0.28, 0.01], [0.0, 0.0, 0.31]])
        metric = np.linalg.inv(step.T @ step)
        grid = SimpleNamespace(shape=shape, dV=float(abs(np.linalg.det(step))),
                               metric_inverse=lambda: metric)
        psi = rng.normal(size=(6, int(np.prod(shape))))
        vext = rng.normal(size=int(np.prod(shape)))
        T_c, V_c = _backend.one_body_matrices(psi, vext, grid)
        _backend.HAS_C_BACKEND = False
        T_n, V_n = _backend.one_body_matrices(psi, vext, grid)
        assert T_c.dtype == np.float64
        assert np.abs(T_c - T_n).max() < 1e-13 * np.abs(T_n).max()
        assert np.array_equal(V_c, V_n)


class TestBuildCommand:
    """``mandacaru --build-backend``."""

    def test_reports_the_loaded_library_and_succeeds(self, backend_state,
                                                     capsys):
        if not check_backend().available:
            pytest.skip("no C backend on this machine")
        assert main(["--build-backend"]) == 0
        out = capsys.readouterr().out
        assert ("C integral backend" in out or "Rust integral backend" in out)
        assert "thread" in out

    def test_needs_no_geometry(self, backend_state, monkeypatch):
        monkeypatch.setattr(_backend, "ensure_backend",
                            lambda **k: BackendStatus(True, "/lib.so", 4, True,
                                                      "compiled"))
        assert main(["--build-backend"]) == 0     # would otherwise demand one

    def test_failed_build_prints_the_log_and_exits_nonzero(self, backend_state,
                                                           tmp_path,
                                                           monkeypatch,
                                                           capsys):
        log = tmp_path / "build.log"
        log.write_text("# mandacaru C backend build\ncc: command not found\n")
        monkeypatch.setattr(_backend, "_BUILD_DIR", tmp_path)
        monkeypatch.setattr(_backend, "build_backend", lambda *a, **k: None)
        monkeypatch.setattr(_backend, "_find_library", lambda: None)
        _backend._LIB = None
        _backend.HAS_C_BACKEND = False

        assert main(["--build-backend"]) == 1
        out = capsys.readouterr().out
        assert "unavailable" in out
        assert "cc: command not found" in out
        assert "NumPy reference kernels" in out

    def test_retries_after_an_earlier_failure(self, backend_state, monkeypatch):
        """The per-process guard must not refuse an explicit request."""
        _backend._LIB = None
        _backend.HAS_C_BACKEND = False
        _backend._build_attempted = True          # a calculation already tried
        monkeypatch.setattr(_backend, "_find_library", lambda: None)
        calls = []
        monkeypatch.setattr(_backend, "build_backend",
                            lambda *a, **k: calls.append(1))

        assert main(["--build-backend"]) == 1
        assert calls == [1]


# --------------------------------------------------------------------------- #
# BLAS beside the OpenMP kernels.
# --------------------------------------------------------------------------- #

def _blas_threads() -> set:
    np.linalg.eigh(np.eye(2))            # make sure NumPy's BLAS is loaded
    return {pool["num_threads"] for pool in threadpool_info()
            if pool["user_api"] == "blas"}


@_backend.single_threaded_blas
def _inside():
    return _blas_threads()


class TestSingleThreadedBLAS:
    """BLAS limited to one thread around the crystal workflow."""

    @pytest.fixture(autouse=True)
    def check_blas_available(self):
        if not _blas_threads():
            pytest.skip("No BLAS thread pool discovered by threadpoolctl on this system")

    def test_blas_runs_one_thread_inside(self):
        with threadpool_limits(limits=2, user_api="blas"):
            assert _inside() == {1}

    def test_the_callers_setting_comes_back(self):
        with threadpool_limits(limits=2, user_api="blas"):
            _inside()
            assert _blas_threads() == {2}

    def test_it_is_restored_when_the_function_raises(self):
        @_backend.single_threaded_blas
        def fails():
            raise ValueError("boom")

        with threadpool_limits(limits=2, user_api="blas"):
            with pytest.raises(ValueError):
                fails()
            assert _blas_threads() == {2}

    def test_nesting_keeps_one_thread_and_restores_the_outer_setting(self):
        @_backend.single_threaded_blas
        def outer():
            inner = _inside()
            return inner, _blas_threads()

        with threadpool_limits(limits=2, user_api="blas"):
            assert outer() == ({1}, {1})
            assert _blas_threads() == {2}

    def test_it_keeps_the_name_and_docstring(self):
        assert _inside.__name__ == "_inside"
        assert _inside.__wrapped__ is not None
