# -*- coding: utf-8 -*-
# file: test_banner.py

"""The start-up banner writes provenance to standard output (not via print)."""


from mandacaru.utils import banner




class TestBanner:
    def test_show_writes_to_stdout(self, capsys):
        banner.show()
        out = capsys.readouterr().out
        assert "version:" in out
        assert "System:" in out
        assert "Dependencies:" in out
        assert "ase version:" in out

    def test_uses_write_not_builtin_print(self):
        # The module must not shadow print with ase.parallel.parprint anymore.
        import inspect
        src = inspect.getsource(banner)
        assert "parprint" not in src
        assert "_write" in src




