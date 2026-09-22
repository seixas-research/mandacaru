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
import warnings
from pathlib import Path

import numpy as np
import pytest

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
                          Grid(box_size=4.0, h=1.0, center=(0, 0, 0)))


class TestStatus:
    def test_check_matches_loaded_library(self, backend_state):
        status = check_backend()
        assert isinstance(status, BackendStatus)
        assert status.available == _backend.HAS_C_BACKEND
        assert status.compiled is False
        if status.available:
            assert status.path and status.path.endswith(
                ("libmandacaru_integrals.dylib", "libmandacaru_integrals.so",
                 "mandacaru_integrals.dll"))
            assert status.n_threads >= 1
            assert status.label.startswith("C (")
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
        library = target / Path(real_path).name
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


class TestBuildCommand:
    """``mandacaru --build-backend``."""

    def test_reports_the_loaded_library_and_succeeds(self, backend_state,
                                                     capsys):
        if not check_backend().available:
            pytest.skip("no C backend on this machine")
        assert main(["--build-backend"]) == 0
        out = capsys.readouterr().out
        assert "C integral backend" in out
        assert "OpenMP thread" in out

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
