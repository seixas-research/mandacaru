# -*- coding: utf-8 -*-
# file: test/experimental/test_link_library.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""``link_library`` puts the external ONCVPSP / PAW repositories into
``library/`` as symbolic links, directory-wise or file by file, and the family
loaders read through them.  Everything runs in a temporary library root."""

import os

import pytest

from carcara.pseudopotentials import link_library as ll


@pytest.fixture
def fake_repo(tmp_path):
    repo = tmp_path / "carcara-paw"
    repo.mkdir()
    for name in ("H.parquet", "O.parquet", "README.md", "LICENSE"):
        (repo / name).write_bytes(b"x")
    return repo


class TestDirectoryLink:
    def test_links_the_directory(self, fake_repo, tmp_path):
        root = tmp_path / "library"
        target = ll.link_library("paw", fake_repo, root=root, verbose=False)
        assert os.path.islink(target)
        assert os.path.realpath(target) == str(fake_repo)
        assert ll.status(root)["paw"][2] == 2          # datasets only, not README
        assert ll.status(root)["ncpp"][1] is None      # nothing linked there

    def test_relinking_the_same_source_is_a_no_op(self, fake_repo, tmp_path):
        root = tmp_path / "library"
        ll.link_library("paw", fake_repo, root=root, verbose=False)
        ll.link_library("paw", fake_repo, root=root, verbose=False)   # no error
        other = tmp_path / "other"
        other.mkdir(); (other / "H.parquet").write_bytes(b"y")
        with pytest.raises(FileExistsError, match="force"):
            ll.link_library("paw", other, root=root, verbose=False)
        target = ll.link_library("paw", other, root=root, force=True, verbose=False)
        assert os.path.realpath(target) == str(other)

    def test_aliases_and_errors(self, fake_repo, tmp_path):
        root = tmp_path / "library"
        assert ll.link_library("oncv", fake_repo, root=root, verbose=False).endswith("oncvpsp")
        assert ll.link_library("tm", fake_repo, root=root, verbose=False).endswith("ncpp")
        with pytest.raises(ValueError, match="unknown family"):
            ll.link_library("uspp", fake_repo, root=root)
        with pytest.raises(FileNotFoundError):
            ll.link_library("paw", tmp_path / "missing", root=root)
        empty = tmp_path / "empty"; empty.mkdir()
        with pytest.raises(FileNotFoundError, match="datasets"):
            ll.link_library("paw", empty, root=root)


class TestFileLinks:
    def test_links_each_dataset(self, fake_repo, tmp_path):
        root = tmp_path / "library"
        target = ll.link_library("paw", fake_repo, root=root, files=True, verbose=False)
        assert os.path.isdir(target) and not os.path.islink(target)
        assert sorted(os.listdir(target)) == ["H.parquet", "O.parquet"]
        assert all(os.path.islink(os.path.join(target, n)) for n in os.listdir(target))
        ll.link_library("paw", fake_repo, root=root, files=True, verbose=False)  # idempotent

    def test_switching_modes_needs_force(self, fake_repo, tmp_path):
        root = tmp_path / "library"
        ll.link_library("paw", fake_repo, root=root, verbose=False)
        with pytest.raises(FileExistsError):
            ll.link_library("paw", fake_repo, root=root, files=True, verbose=False)
        ll.link_library("paw", fake_repo, root=root, files=True, force=True, verbose=False)
        assert not os.path.islink(os.path.join(root, "paw"))


class TestCLI:
    def test_cli_links_and_reports(self, fake_repo, tmp_path, monkeypatch, capsys):
        root = tmp_path / "library"
        monkeypatch.setenv("CARCARA_PSEUDO_PATH", str(root))
        assert ll.main(["--paw", str(fake_repo)]) == 0
        out = capsys.readouterr().out
        assert "paw" in out and "2 datasets" in out and "MISSING" in out   # ncpp/oncvpsp absent
        assert ll.main(["--status"]) == 0

    def test_cli_requires_an_argument(self):
        with pytest.raises(SystemExit):
            ll.main([])


class TestLoadersThroughTheRealLinks:
    """The shipped library: NCPP is bundled; ONCVPSP/PAW are links when set up."""

    def test_families_resolve_through_links(self):
        from carcara.pseudopotentials.io import library_root
        from carcara.pseudopotentials.oncv import get_oncv
        from carcara.pseudopotentials.paw import get_paw
        st = ll.status(library_root())
        assert st["ncpp"][2] >= 89
        if st["oncvpsp"][2] == 0 or st["paw"][2] == 0:
            pytest.skip("external ONCVPSP/PAW repositories not linked on this machine")
        assert get_oncv("H").family == "oncvpsp"
        assert get_paw("H").family == "paw"
