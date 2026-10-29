# -*- coding: utf-8 -*-
# file: test/pseudopotentials/test_library_cli.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The ``mandacaru-paw`` console script (``--set`` and ``--status``), on a
temporary home directory: the user's own shell files are never touched."""

import os

import pytest

from mandacaru.pseudopotentials.library_cli import main


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHELL", "/bin/zsh")
    for variable in ("MANDACARU_PAW_PATH", "MANDACARU_UPAW_PATH"):
        # setenv first so monkeypatch records the original state, even when
        # unset: `--set` writes os.environ itself, and a bare delenv of an
        # unset variable would not undo that at teardown.
        monkeypatch.setenv(variable, "")
        monkeypatch.delenv(variable)
    return tmp_path


def _checkout(home, name="mandacaru-paw"):
    path = home / name
    (path / "lda-sr").mkdir(parents=True)
    (path / "lda-sr" / "H.parquet").write_bytes(b"")
    return path


def test_set_records_the_variable_and_sets_it_here(home, capsys):
    paw = _checkout(home)
    assert main(["--set", str(paw)]) == 0
    out = capsys.readouterr().out
    assert f"MANDACARU_PAW_PATH={paw} added to {home / '.zshrc'}" in out
    assert "lda-sr/: 1 datasets" in out and "new terminal" in out
    assert f"export MANDACARU_PAW_PATH={paw}" in (home / ".zshrc").read_text()
    assert os.environ["MANDACARU_PAW_PATH"] == str(paw)


def test_a_relative_path_is_recorded_absolute(home, monkeypatch):
    _checkout(home)
    monkeypatch.chdir(home)
    assert main(["--set", "mandacaru-paw"]) == 0
    assert f"MANDACARU_PAW_PATH={home}/mandacaru-paw" in \
        (home / ".zshrc").read_text()


def test_a_missing_folder_writes_nothing(home, capsys):
    assert main(["--set", str(home / "nowhere")]) == 1
    assert "is not a folder" in capsys.readouterr().out
    assert not (home / ".zshrc").exists()


def test_the_same_value_is_left_alone(home, capsys):
    paw = _checkout(home)
    assert main(["--set", str(paw)]) == 0
    before = (home / ".zshrc").read_text()
    capsys.readouterr()
    assert main(["--set", str(paw)]) == 0
    assert "already sets" in capsys.readouterr().out
    assert (home / ".zshrc").read_text() == before


def test_a_checkout_without_datasets_is_warned_about(home, capsys):
    empty = home / "empty"
    empty.mkdir()
    assert main(["--set", str(empty)]) == 0
    assert "no dataset folder yet" in capsys.readouterr().out


def test_an_existing_value_is_replaced_after_confirmation(home, capsys,
                                                          monkeypatch):
    old, new = _checkout(home, "old"), _checkout(home, "new")
    (home / ".zshrc").write_text(f"export MANDACARU_PAW_PATH={old}\n")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert main(["--set", str(new)]) == 0
    assert f"left at {old}" in capsys.readouterr().out
    assert "MANDACARU_PAW_PATH" not in os.environ
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    assert main(["--set", str(new)]) == 0
    assert f"replaces {old}" in capsys.readouterr().out
    assert (home / ".zshrc").read_text() == f"export MANDACARU_PAW_PATH={new}\n"


def test_a_different_value_without_a_terminal_is_refused(home, capsys,
                                                         monkeypatch):
    old, new = _checkout(home, "old"), _checkout(home, "new")
    (home / ".zshrc").write_text(f"export MANDACARU_PAW_PATH={old}\n")
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert main(["--set", str(new)]) == 1
    assert "no terminal to confirm" in capsys.readouterr().out
    assert (home / ".zshrc").read_text() == f"export MANDACARU_PAW_PATH={old}\n"


def test_another_shell_is_refused_with_the_line_to_add(home, capsys,
                                                       monkeypatch):
    monkeypatch.setenv("SHELL", "/usr/bin/fish")
    assert main(["--set", str(_checkout(home))]) == 1
    assert "yourself" in capsys.readouterr().out


def _status(capsys, *extra):
    code = main(["--status", "--color", "never", *extra])
    return code, capsys.readouterr().out


def test_set_upaw_records_its_own_variable(home, capsys):
    upaw = _checkout(home, "mandacaru-upaw")
    assert main(["--set-upaw", str(upaw)]) == 0
    assert f"MANDACARU_UPAW_PATH={upaw} added to" in capsys.readouterr().out
    text = (home / ".zshrc").read_text()
    assert f"export MANDACARU_UPAW_PATH={upaw}" in text
    assert "MANDACARU_PAW_PATH" not in text
    assert os.environ["MANDACARU_UPAW_PATH"] == str(upaw)


def test_set_upaw_refuses_a_missing_folder(home, capsys):
    assert main(["--set-upaw", str(home / "nowhere")]) == 1
    assert "MANDACARU_UPAW_PATH" in capsys.readouterr().out
    assert not (home / ".zshrc").exists()


def test_both_libraries_and_the_status_in_one_call(home, capsys, monkeypatch):
    monkeypatch.chdir(home)
    _checkout(home, "mandacaru-paw")
    _checkout(home, "mandacaru-upaw")
    code = main(["--set", "mandacaru-paw", "--set-upaw", "mandacaru-upaw",
                 "--status", "--color", "never"])
    out = capsys.readouterr().out
    assert code == 0
    text = (home / ".zshrc").read_text()
    # Recorded absolute, so it means the same thing from anywhere.
    assert f"MANDACARU_PAW_PATH={home}/mandacaru-paw" in text
    assert f"MANDACARU_UPAW_PATH={home}/mandacaru-upaw" in text
    # The status ran after the writes, so it already sees both.
    assert out.index("added to") < out.index("Library")
    assert "paw-lcao" in out and "upaw-lcao" in out


def test_status_draws_one_row_per_set(home, capsys, monkeypatch):
    paw = _checkout(home)
    for name in ("pbe-sr", "lda-fr"):
        (paw / name).mkdir()
        (paw / name / "H.parquet").write_bytes(b"")
    monkeypatch.setenv("MANDACARU_PAW_PATH", str(paw))
    code, out = _status(capsys)
    rows = [line for line in out.splitlines() if line.startswith("│")]
    header, *body = [[c.strip() for c in line.strip("│").split("│")]
                     for line in rows]
    assert header == ["Library", "Set", "Functional", "Treatment", "Datasets",
                      "Read by default for", "State"]
    sets = {row[1]: row for row in body if row[1] != "-"}
    assert set(sets) == {"lda-fr/", "lda-sr/", "pbe-sr/"}
    assert sets["lda-sr/"][2:5] == ["LDA", "scalar", "1"]
    assert sets["lda-fr/"][3] == "spin-orbit"
    assert sets["pbe-sr/"][5] == "PBE, HSE06, r2SCAN"
    # Only the first row of a library names it; one datum per set is partial.
    assert [row[0] for row in body if row[1] != "-"] == ["paw-lcao", "", ""]
    assert "partial" in sets["lda-sr/"][6]
    assert f"MANDACARU_PAW_PATH  {paw}" in out
    assert code == 0


def test_status_marks_a_complete_set(home, capsys, monkeypatch):
    paw = home / "mandacaru-paw"
    (paw / "lda-sr").mkdir(parents=True)
    for z in range(92):
        (paw / "lda-sr" / f"E{z}.parquet").write_bytes(b"")
    monkeypatch.setenv("MANDACARU_PAW_PATH", str(paw))
    code, out = _status(capsys)
    assert "92" in out and "complete" in out and "partial" not in out


def test_status_reports_an_unset_optional_library_without_failing(
        home, capsys, monkeypatch):
    monkeypatch.setenv("MANDACARU_PAW_PATH", str(_checkout(home)))
    code, out = _status(capsys)
    assert code == 0
    assert "optional, generated on demand" in out
    assert "mandacaru-paw --set-upaw DIR" not in out   # nothing to fix


def test_status_fails_while_the_variable_is_unset(home, capsys):
    code, out = _status(capsys)
    assert code == 1
    assert "MANDACARU_PAW_PATH" in out and "not set" in out
    assert "mandacaru-paw --set DIR" in out


def test_status_fails_when_the_variable_names_no_folder(home, capsys,
                                                        monkeypatch):
    monkeypatch.setenv("MANDACARU_PAW_PATH", str(home / "gone"))
    code, out = _status(capsys)
    assert code == 1 and "not a folder" in out


def test_status_fails_for_an_optional_library_that_names_no_folder(
        home, capsys, monkeypatch):
    monkeypatch.setenv("MANDACARU_PAW_PATH", str(_checkout(home)))
    monkeypatch.setenv("MANDACARU_UPAW_PATH", str(home / "gone"))
    code, out = _status(capsys)
    assert code == 1
    assert "mandacaru-paw --set-upaw DIR" in out


def test_a_checkout_with_no_dataset_folder_is_reported(home, capsys,
                                                       monkeypatch):
    (home / "bare").mkdir()
    monkeypatch.setenv("MANDACARU_PAW_PATH", str(home / "bare"))
    code, out = _status(capsys)
    assert code == 1 and "no dataset folder" in out


class TestColor:
    ESCAPE = "\033["

    def test_never_writes_plain_text(self, home, capsys):
        code, out = _status(capsys)
        assert self.ESCAPE not in out

    def test_always_colors_by_state(self, home, capsys, monkeypatch):
        monkeypatch.setenv("MANDACARU_PAW_PATH", str(_checkout(home)))
        main(["--status", "--color", "always"])
        out = capsys.readouterr().out
        assert "\033[1m" in out                       # bold header
        assert "\033[33m" in out                      # yellow: partial set
        assert "\033[36mlda-sr/" in out               # cyan set name
        monkeypatch.delenv("MANDACARU_PAW_PATH")
        main(["--status", "--color", "always"])
        assert "\033[31m" in capsys.readouterr().out  # red: not set

    def test_color_does_not_disturb_the_alignment(self, home, capsys,
                                                  monkeypatch):
        import re

        monkeypatch.setenv("MANDACARU_PAW_PATH", str(_checkout(home)))
        _, plain = _status(capsys)
        main(["--status", "--color", "always"])
        colored = capsys.readouterr().out
        assert re.sub(r"\033\[[0-9;]*m", "", colored) == plain

    def test_auto_follows_the_terminal_and_the_environment(self, monkeypatch):
        from mandacaru.pseudopotentials.library_cli import use_color

        class Stream:
            def __init__(self, tty):
                self.tty = tty

            def isatty(self):
                return self.tty

        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("FORCE_COLOR", raising=False)
        assert use_color("auto", Stream(True)) and not use_color("auto", Stream(False))
        monkeypatch.setenv("FORCE_COLOR", "1")
        assert use_color("auto", Stream(False))
        monkeypatch.setenv("NO_COLOR", "1")
        assert not use_color("auto", Stream(True))
        assert use_color("always") and not use_color("never")


def test_an_action_is_required(capsys):
    with pytest.raises(SystemExit) as raised:
        main([])
    assert raised.value.code == 2
    assert "at least one" in capsys.readouterr().err


def test_the_console_script_is_registered():
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    scripts = tomllib.loads((root / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["mandacaru-paw"] == "mandacaru.pseudopotentials.library_cli:main"
