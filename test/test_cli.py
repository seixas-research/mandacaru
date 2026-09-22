# -*- coding: utf-8 -*-
# file: test/test_cli.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The ``mandacaru`` console script.

``cli.py`` wraps the single entry point, so every option it forwards has to be
one the selected method actually acts on: an option a method would ignore is a
usage error, reported the way argparse reports one.
"""

import pytest

from mandacaru.cli import build_parser, main, solver_options


class TestCommandLineOptions:
    def _options(self, *argv):
        return solver_options(build_parser().parse_args(
            ["H2", "--cell", "6", *argv]))

    def test_typed_options_are_forwarded_for_every_method(self):
        options = self._options("--method", "vqe", "--txt", "out.txt",
                                "--max-iterations", "0")
        assert options["txt"] == "out.txt" and options["max_iterations"] == 0

    def test_an_adaptive_method_gets_the_default_pool(self):
        assert self._options("--method", "adapt-vqe")["pool"] == "fermionic"
        assert "pool" not in self._options("--method", "vqe")

    @pytest.mark.parametrize("flag", [["--txt", "o.txt"],
                                      ["--max-iterations", "3"],
                                      ["--pool", "qeb"]])
    def test_an_option_the_method_ignores_is_a_usage_error(self, flag, capsys):
        with pytest.raises(SystemExit) as raised:
            main(["H2", "--cell", "6", "--method", "vqe", "--dry-run", *flag])
        assert raised.value.code == 2
        # Either refusal is a usage error: the solver does not take the option
        # at all, or it takes it and its run() would never act on it.
        message = capsys.readouterr().err
        assert "does not take" in message or "does not write" in message

    def test_a_supported_combination_still_runs(self, capsys):
        assert main(["H2", "--cell", "6", "--method", "adapt-vqe", "--dry-run",
                     "--max-iterations", "3"]) == 0
