# -*- coding: utf-8 -*-
# file: test/experimental/test_link_library.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""``link_library`` puts the external ONCVPSP / PAW repositories into
``library/`` as symbolic links, directory-wise or file by file, and the family
loaders read through them.  Everything runs in a temporary library root."""

import os

import pytest

from mandacaru.pseudopotentials import link_library as ll


@pytest.fixture
def fake_repo(tmp_path):
    repo = tmp_path / "mandacaru-paw"
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
        monkeypatch.setenv("MANDACARU_PSEUDO_PATH", str(root))
        assert ll.main(["--paw", str(fake_repo)]) == 0
        out = capsys.readouterr().out
        assert "paw" in out and "2 datasets" in out and "MISSING" in out   # ncpp/oncvpsp absent
        assert ll.main(["--status"]) == 0

    def test_cli_requires_an_argument(self):
        with pytest.raises(SystemExit):
            ll.main([])


class TestMandacaruCommand:
    """``mandacaru --link-paw DIR`` -- the same thing from the main console script.

    The datasets are too large to ship, so setting them up is the first thing a
    user does after cloning; it should not require knowing that
    ``python -m mandacaru.pseudopotentials.link_library`` exists.
    """

    def test_link_paw_links_and_verifies(self, fake_repo, tmp_path,
                                         monkeypatch, capsys):
        from mandacaru import cli

        monkeypatch.setenv("MANDACARU_PSEUDO_PATH", str(tmp_path / "library"))
        # The fake datasets are not loadable, so the command links them and
        # then reports that the *verification* failed -- which is the point:
        # a link is only useful if a dataset actually loads through it.
        assert cli.main(["--link-paw", str(fake_repo)]) == 1
        out = capsys.readouterr().out
        assert "2 datasets" in out
        assert "loading a dataset failed" in out
        assert os.path.islink(tmp_path / "library" / "paw")

    def test_link_paw_needs_no_geometry(self, fake_repo, tmp_path, monkeypatch):
        """It must not trip the 'a geometry is required' guard."""
        from mandacaru import cli

        monkeypatch.setenv("MANDACARU_PSEUDO_PATH", str(tmp_path / "library"))
        cli.main(["--link-paw", str(fake_repo)])      # no SystemExit

    def test_a_missing_directory_is_reported_not_raised(self, tmp_path,
                                                        monkeypatch, capsys):
        from mandacaru import cli

        monkeypatch.setenv("MANDACARU_PSEUDO_PATH", str(tmp_path / "library"))
        assert cli.main(["--link-paw", str(tmp_path / "nope")]) == 1
        assert "is not a directory" in capsys.readouterr().out

    def test_a_populated_directory_is_refused_and_kept(self, fake_repo,
                                                       tmp_path, monkeypatch,
                                                       capsys):
        """--link-paw replaces a link, never someone's real directory."""
        from mandacaru import cli

        root = tmp_path / "library"
        (root / "paw").mkdir(parents=True)
        (root / "paw" / "mine.txt").write_text("keep me")
        monkeypatch.setenv("MANDACARU_PSEUDO_PATH", str(root))
        assert cli.main(["--link-paw", str(fake_repo)]) == 1
        assert "non-empty directory" in capsys.readouterr().out
        assert (root / "paw" / "mine.txt").read_text() == "keep me"

    def test_relinking_to_a_new_path_succeeds(self, fake_repo, tmp_path,
                                              monkeypatch):
        """Moving the data repository must not need --force from the CLI."""
        from mandacaru import cli

        root = tmp_path / "library"
        monkeypatch.setenv("MANDACARU_PSEUDO_PATH", str(root))
        cli.main(["--link-paw", str(fake_repo)])
        moved = tmp_path / "mandacaru-paw-moved"
        moved.mkdir()
        (moved / "H.parquet").write_bytes(b"x")
        cli.main(["--link-paw", str(moved)])
        assert os.path.realpath(root / "paw") == str(moved)

    def test_status_reports_without_linking(self, tmp_path, monkeypatch, capsys):
        from mandacaru import cli

        monkeypatch.setenv("MANDACARU_PSEUDO_PATH", str(tmp_path / "library"))
        assert cli.main(["--pseudo-status"]) == 0
        out = capsys.readouterr().out
        assert out.count("MISSING") == 3      # nothing linked in a fresh root


class TestLoadersThroughTheRealLinks:
    """The shipped library: NCPP is bundled; ONCVPSP/PAW are links when set up."""

    def test_families_resolve_through_links(self):
        from mandacaru.pseudopotentials.io import library_root
        from mandacaru.pseudopotentials.oncv import get_oncv
        from mandacaru.pseudopotentials.paw import get_paw
        st = ll.status(library_root())
        assert st["ncpp"][2] >= 89
        if st["oncvpsp"][2] == 0 or st["paw"][2] == 0:
            pytest.skip("external ONCVPSP/PAW repositories not linked on this machine")
        assert get_oncv("H").family == "oncvpsp"
        assert get_paw("H").family == "paw"

    def test_mandacaru_link_paw_verifies_a_real_repository(self, tmp_path,
                                                           monkeypatch, capsys):
        """The success path of ``mandacaru --link-paw``, end to end."""
        from mandacaru import cli
        from mandacaru.pseudopotentials.io import library_root

        st = ll.status(library_root())
        if st["paw"][2] == 0:
            pytest.skip("the PAW repository is not linked on this machine")
        source = st["paw"][1]
        monkeypatch.setenv("MANDACARU_PSEUDO_PATH", str(tmp_path / "library"))
        # The dataset cache is keyed by folder, so a fresh root really reloads.
        assert cli.main(["--link-paw", source]) == 0
        assert "loaded H successfully" in capsys.readouterr().out


class TestMissingLibraryMessage:
    """A fresh install must be told how to get the datasets.

    They are too large to ship, so an empty library is the *normal* first
    state, not a corruption.  The message used to offer only
    ``build_paw_library([symbol])`` -- generating 92 elements from scratch --
    and never mentioned that linking a checkout takes a second.
    """

    @pytest.fixture
    def empty_library(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MANDACARU_PSEUDO_PATH", str(tmp_path / "library"))
        # The loaders cache by folder, so a fresh root really reloads.
        from mandacaru.pseudopotentials import oncv, paw
        monkeypatch.setattr(paw, "_CACHE", {})
        monkeypatch.setattr(oncv, "_CACHE", {})
        return tmp_path

    def test_paw_points_at_the_link_command(self, empty_library):
        from mandacaru.pseudopotentials.paw import get_paw

        with pytest.raises(FileNotFoundError) as excinfo:
            get_paw("O")
        message = str(excinfo.value)
        assert "mandacaru --link-paw" in message
        assert "mandacaru-paw" in message
        assert "mandacaru --pseudo-status" in message

    def test_oncvpsp_points_at_the_link_command(self, empty_library):
        from mandacaru.pseudopotentials.oncv import get_oncv

        with pytest.raises(FileNotFoundError) as excinfo:
            get_oncv("O")
        message = str(excinfo.value)
        assert "mandacaru --link-oncvpsp" in message
        assert "mandacaru-oncvpsp" in message

    def test_a_populated_library_still_names_the_missing_element(self, tmp_path,
                                                                 monkeypatch):
        """With datasets present, the message is about the element, not setup."""
        from mandacaru.pseudopotentials import paw

        folder = tmp_path / "library" / "paw"
        folder.mkdir(parents=True)
        (folder / "H.parquet").write_bytes(b"x")
        monkeypatch.setenv("MANDACARU_PSEUDO_PATH", str(tmp_path / "library"))
        monkeypatch.setattr(paw, "_CACHE", {})

        with pytest.raises(FileNotFoundError) as excinfo:
            paw.get_paw("O")
        message = str(excinfo.value)
        assert "Available: H" in message
        assert "build_paw_library" in message
        assert "--link-paw" not in message      # the library is not the problem
