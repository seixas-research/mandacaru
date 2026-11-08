"""The Rust adapter binds every kernel at import or raises a named error."""

import pytest

from mandacaru.integrals import _backend


def test_an_extension_that_cannot_be_opened_raises_a_named_error(monkeypatch):
    def refuse(path):
        raise OSError("image not found")

    monkeypatch.setattr(_backend.ctypes, "CDLL", refuse)
    with pytest.raises(RuntimeError, match="could not be opened.*reinstall Mandacaru"):
        _backend._load()


def test_an_extension_whose_kernels_do_not_bind_raises_a_named_error(monkeypatch):
    def stale(lib):
        raise AttributeError("undefined symbol: mandacaru_bloch_shell")

    monkeypatch.setattr(_backend, "_bind", stale)
    with pytest.raises(RuntimeError, match="loaded but its kernels did not bind.*reinstall Mandacaru"):
        _backend._load()


def test_every_adapter_kernel_is_bound_so_no_call_reports_not_done():
    for name in ("bloch_shell", "real_shell", "real_shell_f32", "bloch_project",
                 "bader_ascent", "num_threads"):
        assert hasattr(_backend._LIB, "mandacaru_" + name), name
    status = _backend.check_backend()
    assert status.path == _backend._LIB._mandacaru_origin and status.n_threads >= 1
    assert status.label.startswith("Rust")


