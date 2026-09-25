# -*- coding: utf-8 -*-
# file: test/utils/test_shell_config.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Writing ``export NAME=value`` into ``~/.zshrc`` / ``~/.bashrc``
(:mod:`mandacaru.utils.shell_config`).  Every test works on a temporary
home directory; the user's own files are never touched."""

import os
import stat

import pytest

from mandacaru.utils.shell_config import (MARKER, ShellConfigError,
                                          read_shell_variable,
                                          set_shell_variable,
                                          shell_config_file)

NAME = "MANDACARU_PAW_PATH"


@pytest.mark.parametrize("shell, name", [("/bin/zsh", ".zshrc"),
                                         ("/usr/local/bin/bash", ".bashrc")])
def test_the_login_shell_picks_the_file(tmp_path, shell, name):
    assert shell_config_file(shell, home=str(tmp_path)) == tmp_path / name


@pytest.mark.parametrize("shell", ["/usr/bin/fish", ""])
def test_another_shell_is_refused(tmp_path, shell):
    with pytest.raises(ShellConfigError, match="yourself"):
        shell_config_file(shell, home=str(tmp_path))


def test_a_missing_file_is_created(tmp_path):
    rc = tmp_path / ".zshrc"
    update = set_shell_variable(NAME, "/data/mandacaru paw", path=rc)
    assert update.action == "added" and update.changed
    assert MARKER in rc.read_text()
    # The value is quoted, so a path with a space survives the shell.
    assert "export MANDACARU_PAW_PATH='/data/mandacaru paw'" in rc.read_text()
    assert read_shell_variable(rc, NAME) == "/data/mandacaru paw"


def test_it_appends_without_disturbing_the_file(tmp_path):
    rc = tmp_path / ".bashrc"
    rc.write_text("alias ll='ls -l'\nexport PATH=/opt/bin:$PATH")  # no final \n
    set_shell_variable(NAME, "/data/paw", path=rc)
    text = rc.read_text()
    assert text.startswith("alias ll='ls -l'\nexport PATH=/opt/bin:$PATH\n")
    assert text.endswith("export MANDACARU_PAW_PATH=/data/paw\n")


def test_the_same_value_is_left_alone(tmp_path):
    rc = tmp_path / ".zshrc"
    set_shell_variable(NAME, "/data/paw", path=rc)
    before = rc.read_text()
    asked = []
    update = set_shell_variable(NAME, "/data/paw", path=rc,
                                confirm=lambda q: asked.append(q) or True)
    assert update.action == "unchanged" and not asked
    assert rc.read_text() == before


@pytest.mark.parametrize("answer, action", [(True, "replaced"),
                                            (False, "declined")])
def test_a_different_value_is_replaced_only_when_confirmed(tmp_path, answer,
                                                           action):
    rc = tmp_path / ".zshrc"
    rc.write_text(f"export {NAME}=/old\necho done\n")
    update = set_shell_variable(NAME, "/new", path=rc,
                                confirm=lambda question: answer)
    assert update.action == action and update.previous == "/old"
    expected = "/new" if answer else "/old"
    assert read_shell_variable(rc, NAME) == expected
    assert rc.read_text().count(NAME) == 1          # replaced in place
    assert rc.read_text().endswith("echo done\n")


@pytest.mark.parametrize("typed, replaced", [("", True), ("y", True),
                                             ("YES", True), ("n", False)])
def test_the_prompt_defaults_to_yes(tmp_path, monkeypatch, typed, replaced):
    rc = tmp_path / ".zshrc"
    rc.write_text(f"export {NAME}=/old\n")
    monkeypatch.setattr("builtins.input", lambda prompt: typed)
    set_shell_variable(NAME, "/new", path=rc, interactive=True)
    assert read_shell_variable(rc, NAME) == ("/new" if replaced else "/old")


def test_without_a_terminal_an_overwrite_is_refused(tmp_path):
    rc = tmp_path / ".zshrc"
    rc.write_text(f"export {NAME}=/old\n")
    with pytest.raises(ShellConfigError, match="no terminal"):
        set_shell_variable(NAME, "/new", path=rc, interactive=False)
    assert read_shell_variable(rc, NAME) == "/old"


def test_duplicates_collapse_to_one_assignment(tmp_path):
    rc = tmp_path / ".zshrc"
    rc.write_text(f"export {NAME}=/a\n# export {NAME}=/commented\n"
                  f"{NAME}=/b\n")
    assert read_shell_variable(rc, NAME) == "/b"   # the last one wins
    set_shell_variable(NAME, "/c", path=rc, confirm=lambda q: True)
    lines = rc.read_text().splitlines()
    assert [l for l in lines if l.startswith(("export", NAME))] == \
        [f"export {NAME}=/c"]
    assert f"# export {NAME}=/commented" in lines  # comments are kept


def test_permissions_are_kept_and_a_link_is_written_through(tmp_path):
    dotfiles = tmp_path / "dotfiles"
    dotfiles.mkdir()
    real = dotfiles / "zshrc"
    real.write_text("# mine\n")
    os.chmod(real, 0o600)
    rc = tmp_path / ".zshrc"
    rc.symlink_to(real)
    set_shell_variable(NAME, "/data/paw", path=rc)
    assert rc.is_symlink()                          # still a link
    assert read_shell_variable(real, NAME) == "/data/paw"
    assert stat.S_IMODE(real.stat().st_mode) == 0o600
    assert not [p for p in dotfiles.iterdir() if p.name != "zshrc"]
