# -*- coding: utf-8 -*-
# file: test_banner.py

"""The start-up banner writes provenance to standard output (not via print)."""


import importlib
import os
import subprocess
import sys

import pytest

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






class TestDependencyProbe:
    def test_a_missing_package_reads_as_not_available(self):
        assert banner._dep("mandacaru_no_such_package") == ("n/a", "")

    @pytest.mark.parametrize("name", ["ase", "numpy", "scipy", "matplotlib", "qiskit"])
    def test_the_probe_reports_what_importing_the_package_reports(self, name):
        module = importlib.import_module(name)
        assert banner._dep(name) == (str(module.__version__), os.path.dirname(module.__file__))

    def test_the_probe_does_not_import_the_package(self):
        code = (
            "import sys\n"
            "from mandacaru.utils import banner\n"
            "before = set(sys.modules)\n"
            "banner.lines()\n"
            "heavy = [m for m in ('numpy', 'scipy', 'matplotlib', 'qiskit', 'ase') "
            "if m in sys.modules and m not in before]\n"
            "assert not heavy, heavy\n"
        )
        done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
