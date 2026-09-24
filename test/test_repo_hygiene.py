# -*- coding: utf-8 -*-
# file: test/test_repo_hygiene.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Nothing generated may ever land in the repository root.

Files are checked strictly: the root holds exactly ``ROOT_ALLOWED``.
Directories may additionally be anything ``.gitignore`` excludes, since git
cannot commit those -- that is what the rule protects against.

Every example, test and script writes its outputs under ``examples/data/``
(or a pytest ``tmp_path``); the repository root holds only the files listed
in ``ROOT_ALLOWED``.  This test fails the suite the moment a stray file
appears there, and it also inspects every example for a file written with a
bare relative path (which lands in whatever directory the script was run
from -- the way the leak happened once).
"""

import fnmatch
import os
import re

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The only files that may live in the repository root.
ROOT_ALLOWED = {".gitignore", ".readthedocs.yaml", "CLAUDE.md", "HISTORY.md",
                "LICENSE", "PAW_SAGA.md", "README.md", "VERSION_DESCRIPTION.md",
                "pyproject.toml"}
def gitignore_patterns():
    """Directory patterns ``.gitignore`` excludes (negations not supported)."""
    path = os.path.join(REPO, ".gitignore")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        lines = [line.strip() for line in handle]
    return [line.rstrip("/").lstrip("/") for line in lines
            if line and not line.startswith(("#", "!"))]


def is_git_ignored(name: str) -> bool:
    """Whether ``.gitignore`` excludes a root entry of this name."""
    return any(fnmatch.fnmatch(name, pattern)
               for pattern in gitignore_patterns())


#: Directories that may live in the repository root.
ROOT_DIRS_ALLOWED = {".git", ".claude", ".github", "docs", "examples", "plan",
                     "latex", "logo",
                     "src", "test", "dist", "build", ".pytest_cache",
                     ".ruff_cache", ".mypy_cache", "mandacaru.egg-info"}

# A write target given as a bare quoted file name (no directory join).
_BARE_WRITE = re.compile(
    r"""(?:savefig|open|to_csv|savetxt|np\.save|json\.dump)\(\s*["']([^"'/\\]+\.[A-Za-z0-9]+)["']""")


def test_repository_root_holds_only_the_known_files():
    stray = sorted(name for name in os.listdir(REPO)
                   if os.path.isfile(os.path.join(REPO, name))
                   and name not in ROOT_ALLOWED and not name.startswith(".DS"))
    assert not stray, (
        f"generated files leaked into the repository root: {stray}. "
        "Outputs belong under examples/data/ (or a tmp_path in tests); "
        "delete them and fix whatever wrote them.")


def test_repository_root_holds_only_the_known_directories():
    """Only the source tree plus build/cache directories git already ignores.

    A git-ignored directory (``dist``, a tool's scratch folder) can never be
    committed, so it cannot leak; an *unignored* one is a real stray.
    """
    stray = sorted(name for name in os.listdir(REPO)
                   if os.path.isdir(os.path.join(REPO, name))
                   and name not in ROOT_DIRS_ALLOWED
                   and not is_git_ignored(name))
    assert not stray, (
        f"unexpected directories in the repository root: {stray}. "
        "Generated output belongs under examples/data/ (or a tmp_path); "
        "add a deliberate scratch directory to .gitignore.")


@pytest.mark.parametrize("script", sorted(
    os.path.join(dirpath, name)
    for dirpath, _dirs, names in os.walk(os.path.join(REPO, "examples"))
    for name in names if name.endswith(".py")))
def test_examples_never_write_with_a_bare_relative_path(script):
    with open(script, encoding="utf-8") as fh:
        source = fh.read()
    hits = [m.group(1) for m in _BARE_WRITE.finditer(source)]
    assert not hits, (
        f"{os.path.relpath(script, REPO)} writes {hits} with a bare relative "
        "path, which lands in the current working directory; build the path "
        "from the script's own location (os.path.join(DATA, ...)).")
