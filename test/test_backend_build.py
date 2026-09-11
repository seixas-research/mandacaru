# -*- coding: utf-8 -*-
# file: test/test_backend_build.py

# This code is part of Carcará.
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
  ``RuntimeWarning`` from the engine, and ``CARCARA_BACKEND=c`` refuses the
  fallback;
* ``CARCARA_BACKEND=numpy`` forces the reference kernels without building;
* :func:`build_backend` really compiles the library for this machine when a
  tool chain exists (CMake, or the bare compiler when CMake is hidden).
"""

import shutil
import subprocess
import sys
import warnings

import numpy as np
import pytest

from carcara.basis import FullAtomicOrbital
from carcara.integrals import (BackendStatus, Grid, IntegralEngine, _backend,
                               build_backend, check_backend, ensure_backend)


@pytest.fixture
def backend_state(monkeypatch):
    """Snapshot/restore the module globals the check mutates."""
    saved = (_backend._LIB, _backend.HAS_C_BACKEND, _backend._build_attempted,
             _backend._fallback_warned)
    monkeypatch.delenv("CARCARA_BACKEND", raising=False)
    yield
    (_backend._LIB, _backend.HAS_C_BACKEND, _backend._build_attempted,
     _backend._fallback_warned) = saved


def _engine():
    return IntegralEngine([FullAtomicOrbital(1, 0, 0)],
                          Grid(box_size=4.0, h=1.0, center=(0, 0, 0)))


class TestStatus:
    def test_check_matches_loaded_library(self, backend_state):
        status = check_backend()
        assert isinstance(status, BackendStatus)
        assert status.available == _backend.HAS_C_BACKEND
        assert status.compiled is False
        if status.available:
            assert status.path and status.path.endswith(
                ("libcarcara_integrals.dylib", "libcarcara_integrals.so",
                 "carcara_integrals.dll"))
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
        monkeypatch.setenv("CARCARA_BACKEND", "numpy")
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
        monkeypatch.setenv("CARCARA_BACKEND", "fortran")
        with pytest.raises(ValueError, match="CARCARA_BACKEND"):
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
        monkeypatch.setenv("CARCARA_BACKEND", "c")
        monkeypatch.setattr(_backend, "build_backend", lambda *a, **k: None)
        with pytest.raises(RuntimeError, match="refuses the NumPy fallback"):
            ensure_backend()


def _threads_in_fresh_process(path) -> int:
    """Load ``path`` in a subprocess and return its thread count.

    A separate process, because a library built by a different tool chain
    may link a *different* OpenMP runtime than the one already loaded here,
    and two OpenMP runtimes in one process abort.
    """
    code = ("import ctypes, sys; from carcara.integrals import _backend; "
            f"lib = _backend._bind(ctypes.CDLL({str(path)!r})); "
            "print(lib.carcara_num_threads())")
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
        assert (tmp_path / "build.log").read_text().startswith("# carcara")
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
