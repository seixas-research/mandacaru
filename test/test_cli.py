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

    @pytest.mark.parametrize("argv, expected", [
        ([], None),
        (["--convergence-gradient", "1e-4"], {"gradient": 1e-4}),
        (["--convergence-energy", "1e-6"], {"energy": 1e-6}),
        (["--convergence-gradient", "1e-3", "--convergence-energy", "1e-6"],
         {"gradient": 1e-3, "energy": 1e-6}),
    ])
    def test_the_convergence_flags_build_the_dictionary(self, argv, expected):
        """The flags given are the criteria used; none leaves the default."""
        options = self._options("--method", "adapt-vqe", *argv)
        assert options.get("convergence") == expected

    def test_an_adaptive_method_gets_the_default_pool(self):
        assert self._options("--method", "adapt-vqe")["pool"] == "fermionic"
        assert "pool" not in self._options("--method", "vqe")

    @pytest.mark.parametrize("flag", [["--max-iterations", "3"],
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


class TestMarkovChainOptions:
    """The ``--method vasqa`` flags, forwarded only when given."""

    def _options(self, *argv):
        return solver_options(build_parser().parse_args(
            ["H2", "--cell", "6", "--method", "vasqa", *argv]))

    def test_nothing_given_forwards_nothing(self):
        options = self._options()
        for name in ("max_steps", "max_length", "move_weights", "temperature",
                     "length_penalty", "warm_start", "seed", "proposal",
                     "proposal_temperature", "record", "proposal_model"):
            assert name not in options

    def test_every_flag_reaches_its_option(self):
        options = self._options(
            "--max-steps", "50", "--max-length", "8", "--length-penalty",
            "0.002", "--no-warm-start", "--seed", "11", "--pool", "qeb",
            "--proposal", "uniform", "--proposal-temperature", "0.5")
        assert options["proposal"] == "uniform"
        assert options["proposal_temperature"] == 0.5
        assert options["max_steps"] == 50 and options["max_length"] == 8
        assert options["length_penalty"] == 0.002
        assert options["warm_start"] is False and options["seed"] == 11
        assert options["pool"] == "qeb"

    def test_the_recording_and_model_flags_reach_valqa(self):
        options = solver_options(build_parser().parse_args(
            ["H2", "--cell", "6", "--method", "valqa", "--record", "edits",
             "--proposal-model", "model.npz"]))
        assert options["record"] == "edits"
        assert options["proposal_model"] == "model.npz"

    def test_no_record_turns_the_shared_store_off(self):
        assert self._options("--no-record")["record"] is False

    def test_one_temperature_is_fixed_and_two_anneal(self):
        assert self._options("--temperature", "0.05")["temperature"] == 0.05
        assert self._options("--temperature", "0.1", "0.001")[
            "temperature"] == {"initial": 0.1, "final": 0.001}

    def test_three_temperatures_are_a_usage_error(self, capsys):
        with pytest.raises(SystemExit) as raised:
            main(["H2", "--cell", "6", "--method", "vasqa", "--dry-run",
                  "--temperature", "1", "2", "3"])
        assert raised.value.code == 2
        assert "--temperature" in capsys.readouterr().err

    def test_named_move_weights_keep_the_others_at_one(self):
        options = self._options("--move-weight", "swap=0",
                                "--move-weight", "insert=2")
        assert options["move_weights"] == {"insert": 2.0, "delete": 1.0,
                                           "replace": 1.0, "swap": 0.0}

    @pytest.mark.parametrize("text", ["grow=1", "insert=fast", "insert"])
    def test_a_bad_move_weight_is_a_usage_error(self, text, capsys):
        with pytest.raises(SystemExit) as raised:
            build_parser().parse_args(["H2", "--move-weight", text])
        assert raised.value.code == 2

    @pytest.mark.parametrize("flag", [["--max-steps", "5"], ["--seed", "1"],
                                      ["--no-warm-start"],
                                      ["--proposal", "uniform"],
                                      ["--temperature", "0.1"]])
    def test_another_method_refuses_them(self, flag, capsys):
        with pytest.raises(SystemExit) as raised:
            main(["H2", "--cell", "6", "--method", "adapt-vqe", "--dry-run",
                  *flag])
        assert raised.value.code == 2
        assert "does not take" in capsys.readouterr().err

    def test_a_vasqa_dry_run_accepts_them(self, capsys):
        assert main(["H2", "--cell", "6", "--method", "vasqa", "--dry-run",
                     "--max-steps", "5", "--seed", "3", "--temperature",
                     "0.1", "0.01", "--move-weight", "swap=0.5"]) == 0

    def test_a_short_chain_runs_end_to_end(self, capsys):
        assert main(["H2", "--cell", "5", "--h", "0.4", "--method", "vasqa",
                     "--max-steps", "4", "--seed", "3", "--quiet"]) == 0


class TestLibraryVariables:
    """``--set-paw/--set-oncvpsp/--set-upaw`` and ``--pseudo-status``, on a
    temporary home directory: the user's own shell files are never touched."""

    @pytest.fixture
    def home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("SHELL", "/bin/zsh")
        for variable in ("MANDACARU_PAW_PATH", "MANDACARU_ONCVPSP_PATH",
                         "MANDACARU_UPAW_PATH"):
            # setenv first so monkeypatch records the original state, even
            # when unset: `--set-*` writes os.environ itself, and a bare
            # delenv of an unset variable would not undo that at teardown.
            monkeypatch.setenv(variable, "")
            monkeypatch.delenv(variable)
        return tmp_path

    def _checkout(self, home, name):
        path = home / name
        (path / "lda-sr").mkdir(parents=True)
        (path / "lda-sr" / "H.parquet").write_bytes(b"")
        return path

    def test_set_records_the_variable_and_sets_it_here(self, home, capsys):
        import os

        paw = self._checkout(home, "mandacaru-paw")
        assert main(["--set-paw", str(paw)]) == 0
        out = capsys.readouterr().out
        assert f"MANDACARU_PAW_PATH={paw} added to {home / '.zshrc'}" in out
        assert "lda-sr/: 1 datasets" in out and "new terminal" in out
        assert f"export MANDACARU_PAW_PATH={paw}" in (home / ".zshrc").read_text()
        assert os.environ["MANDACARU_PAW_PATH"] == str(paw)

    def test_both_at_once_and_a_relative_path(self, home, capsys,
                                              monkeypatch):
        monkeypatch.chdir(home)
        for name in ("mandacaru-paw", "mandacaru-oncvpsp"):
            self._checkout(home, name)
        assert main(["--set-paw", "mandacaru-paw",
                     "--set-oncvpsp", "mandacaru-oncvpsp"]) == 0
        text = (home / ".zshrc").read_text()
        for variable, name in (("PAW", "paw"), ("ONCVPSP", "oncvpsp")):
            # Recorded absolute, so it means the same thing from anywhere.
            assert f"MANDACARU_{variable}_PATH={home}/mandacaru-{name}" in text
        capsys.readouterr()
        assert main(["--pseudo-status"]) == 0
        assert capsys.readouterr().out.count("lda-sr/: 1 datasets") == 2

    def test_upaw_is_optional_but_can_be_set(self, home, capsys):
        for name in ("mandacaru-paw", "mandacaru-oncvpsp"):
            self._checkout(home, name)
        assert main(["--set-paw", str(home / "mandacaru-paw"),
                     "--set-oncvpsp", str(home / "mandacaru-oncvpsp")]) == 0
        capsys.readouterr()
        # Unset UPAW: reported as optional, and the status still passes.
        assert main(["--pseudo-status"]) == 0
        assert "optional, generated on demand" in capsys.readouterr().out
        upaw = self._checkout(home, "mandacaru-upaw")
        assert main(["--set-upaw", str(upaw)]) == 0
        assert f"MANDACARU_UPAW_PATH={upaw}" in (home / ".zshrc").read_text()

    def test_the_proposal_store_is_created_and_recorded(self, home, capsys,
                                                        monkeypatch):
        import os

        monkeypatch.setenv("MANDACARU_PROPOSAL_DATA", "")
        store = home / "valqa" / "edits"
        assert main(["--set-proposal-data", str(store)]) == 0
        assert store.is_dir()
        assert f"export MANDACARU_PROPOSAL_DATA={store}" in \
            (home / ".zshrc").read_text()
        assert os.environ["MANDACARU_PROPOSAL_DATA"] == str(store)
        assert "added to" in capsys.readouterr().out

    def test_a_missing_directory_writes_nothing(self, home, capsys):
        assert main(["--set-oncvpsp", str(home / "nowhere")]) == 1
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
        assert main(["H2O", "--cell", "8", "--basis", "ONCVPSP",
                     "--dry-run"]) == 2
        assert "mandacaru --set-oncvpsp" in capsys.readouterr().err
