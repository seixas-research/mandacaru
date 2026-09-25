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


class TestLibraryVariables:
    """``--set-paw/--set-ncpp/--set-oncvpsp`` and ``--pseudo-status``, on a
    temporary home directory: the user's own shell files are never touched."""

    @pytest.fixture
    def home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("SHELL", "/bin/zsh")
        for variable in ("MANDACARU_PAW_PATH", "MANDACARU_NCPP_PATH",
                         "MANDACARU_ONCVPSP_PATH"):
            monkeypatch.delenv(variable, raising=False)
        return tmp_path

    def _checkout(self, home, name):
        path = home / name
        (path / "lda").mkdir(parents=True)
        (path / "lda" / "H.parquet").write_bytes(b"")
        return path

    def test_set_records_the_variable_and_sets_it_here(self, home, capsys):
        import os

        paw = self._checkout(home, "mandacaru-paw")
        assert main(["--set-paw", str(paw)]) == 0
        out = capsys.readouterr().out
        assert f"MANDACARU_PAW_PATH={paw} added to {home / '.zshrc'}" in out
        assert "lda/: 1 datasets" in out and "new terminal" in out
        assert f"export MANDACARU_PAW_PATH={paw}" in (home / ".zshrc").read_text()
        assert os.environ["MANDACARU_PAW_PATH"] == str(paw)

    def test_all_three_at_once_and_a_relative_path(self, home, capsys,
                                                   monkeypatch):
        monkeypatch.chdir(home)
        for name in ("mandacaru-paw", "mandacaru-ncpp", "mandacaru-oncvpsp"):
            self._checkout(home, name)
        assert main(["--set-paw", "mandacaru-paw", "--set-ncpp",
                     "mandacaru-ncpp", "--set-oncvpsp", "mandacaru-oncvpsp"]) == 0
        text = (home / ".zshrc").read_text()
        for variable, name in (("PAW", "paw"), ("NCPP", "ncpp"),
                               ("ONCVPSP", "oncvpsp")):
            # Recorded absolute, so it means the same thing from anywhere.
            assert f"MANDACARU_{variable}_PATH={home}/mandacaru-{name}" in text
        capsys.readouterr()
        assert main(["--pseudo-status"]) == 0
        assert capsys.readouterr().out.count("lda/: 1 datasets") == 3

    def test_a_missing_directory_writes_nothing(self, home, capsys):
        assert main(["--set-ncpp", str(home / "nowhere")]) == 1
        assert "is not a directory" in capsys.readouterr().out
        assert not (home / ".zshrc").exists()

    def test_an_existing_value_is_replaced_after_confirmation(self, home,
                                                              capsys,
                                                              monkeypatch):
        old = self._checkout(home, "old")
        new = self._checkout(home, "new")
        (home / ".zshrc").write_text(f"export MANDACARU_PAW_PATH={old}\n")
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda prompt: "n")
        assert main(["--set-paw", str(new)]) == 0
        assert f"left at {old}" in capsys.readouterr().out
        monkeypatch.setattr("builtins.input", lambda prompt: "")
        assert main(["--set-paw", str(new)]) == 0
        assert f"replaces {old}" in capsys.readouterr().out
        assert (home / ".zshrc").read_text() == \
            f"export MANDACARU_PAW_PATH={new}\n"

    def test_status_fails_while_a_variable_is_unset(self, home, capsys):
        assert main(["--pseudo-status"]) == 1
        assert "MANDACARU_PAW_PATH is not set" in capsys.readouterr().out

    @pytest.mark.parametrize("flag", ["--link-paw-lcao", "--link-oncvpsp"])
    def test_the_link_flags_are_gone(self, flag, capsys):
        with pytest.raises(SystemExit):
            main([flag, "/tmp"])
        assert "unrecognized arguments" in capsys.readouterr().err

    def test_a_run_without_its_library_says_how_to_fix_it(self, home, capsys):
        assert main(["H2O", "--cell", "8", "--basis", "NCPP",
                     "--dry-run"]) == 2
        assert "mandacaru --set-ncpp" in capsys.readouterr().err
