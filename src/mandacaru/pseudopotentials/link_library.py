# -*- coding: utf-8 -*-
# file: pseudopotentials/link_library.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Link externally stored pseudopotential libraries into ``library/``.

The ONCVPSP and PAW datasets are too large for this repository (about 100 MB
and 200 MB for Z <= 92), so they live in repositories of their own --
``mandacaru-oncvpsp`` and ``mandacaru-paw`` -- as flat directories of
``<Symbol>.parquet`` files.  The loaders look for them under
``library/oncvpsp/`` and ``library/paw/`` (see :func:`.io.library_root`), and
this module puts symbolic links there so a checkout of the data repositories
anywhere on disk is enough::

    python -m mandacaru.pseudopotentials.link_library \\
        --oncvpsp ~/Repositories/mandacaru-oncvpsp --paw ~/Repositories/mandacaru-paw

or, from Python::

    from mandacaru.pseudopotentials.link_library import link_library
    link_library("paw", "~/Repositories/mandacaru-paw")

By default one link per family (``library/paw -> <repo>``); ``files=True`` /
``--files`` links each ``*.parquet`` / ``*.json`` file individually instead,
which is what you want when the target directory must stay a real directory
(some sync tools do not follow directory links).  The links are ignored by
git.  Nothing is copied.
"""

from __future__ import annotations

import argparse
import os
import sys

#: Family name -> subdirectory of the library it is served from.
FAMILY_SUBDIRS = {"ncpp": "ncpp", "oncvpsp": "oncvpsp", "paw": "paw"}
_ALIASES = {"tm": "ncpp", "ncpp-tm": "ncpp", "oncv": "oncvpsp"}
_DATA_EXTENSIONS = (".parquet", ".pq", ".json")


def _family(name: str) -> str:
    key = str(name).strip().lower()
    key = _ALIASES.get(key, key)
    if key not in FAMILY_SUBDIRS:
        raise ValueError(
            f"unknown family {name!r}; use one of {sorted(FAMILY_SUBDIRS)}")
    return key


def _data_files(source: str) -> list[str]:
    return sorted(name for name in os.listdir(source)
                  if os.path.splitext(name)[1].lower() in _DATA_EXTENSIONS)


def _remove(path: str) -> None:
    if os.path.islink(path) or os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        if os.listdir(path):
            raise FileExistsError(
                f"{path} is a non-empty directory; remove it yourself or "
                "link file by file with files=True")
        os.rmdir(path)


def link_library(family: str, source, *, root=None, files: bool = False,
                 force: bool = False, verbose: bool = True) -> str:
    """Link the ``family`` datasets stored in ``source`` into the library.

    Parameters
    ----------
    family : str
        ``"ncpp"`` (``"tm"``), ``"oncvpsp"`` (``"oncv"``) or ``"paw"``.
    source : path
        Directory holding the ``<Symbol>.parquet`` files (a checkout of the
        data repository).  ``~`` is expanded.
    root : path, optional
        The library root to link into (default :func:`.io.library_root`).
    files : bool
        Link every data file into a real ``library/<family>/`` directory
        instead of linking the directory itself.
    force : bool
        Replace an existing link / empty directory.  Without it an existing
        target that already points at ``source`` is left alone and anything
        else raises ``FileExistsError``.

    Returns
    -------
    str
        The path of the link (or linked directory) in the library.
    """
    from .io import library_root

    family = _family(family)
    source = os.path.abspath(os.path.expanduser(os.fspath(source)))
    if not os.path.isdir(source):
        raise FileNotFoundError(f"{source} is not a directory")
    data = _data_files(source)
    if not data:
        raise FileNotFoundError(f"{source} holds no .parquet/.json datasets")
    root = library_root() if root is None else os.path.abspath(os.fspath(root))
    os.makedirs(root, exist_ok=True)
    target = os.path.join(root, FAMILY_SUBDIRS[family])

    if not files:
        if os.path.islink(target) and os.path.realpath(target) == source and not force:
            if verbose:
                print(f"{family:8s} already linked: {target} -> {source}")
            return target
        if os.path.lexists(target):
            if not force:
                raise FileExistsError(
                    f"{target} exists; pass force=True / --force to replace it")
            _remove(target)
        os.symlink(source, target, target_is_directory=True)
        if verbose:
            print(f"{family:8s} {target} -> {source}  ({len(data)} datasets)")
        return target

    if os.path.islink(target):
        if not force:
            raise FileExistsError(
                f"{target} is a directory link; pass force=True to replace it "
                "with per-file links")
        os.remove(target)
    os.makedirs(target, exist_ok=True)
    linked = 0
    for name in data:
        src, dst = os.path.join(source, name), os.path.join(target, name)
        if os.path.lexists(dst):
            if os.path.islink(dst) and os.path.realpath(dst) == src and not force:
                continue
            if not force:
                raise FileExistsError(f"{dst} exists; pass force=True")
            os.remove(dst)
        os.symlink(src, dst)
        linked += 1
    if verbose:
        print(f"{family:8s} {target}/: {linked} new links, "
              f"{len(data)} datasets from {source}")
    return target


def status(root=None) -> dict:
    """``{family: (target, resolved source or None, dataset count)}``."""
    from .io import library_root

    root = library_root() if root is None else os.path.abspath(os.fspath(root))
    out = {}
    for family, sub in FAMILY_SUBDIRS.items():
        target = os.path.join(root, sub)
        if os.path.isdir(target):
            source = os.path.realpath(target) if os.path.islink(target) else target
            out[family] = (target, source, len(_data_files(target)))
        else:
            out[family] = (target, None, 0)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mandacaru.pseudopotentials.link_library",
        description="Link the external ONCVPSP / PAW (and optionally NCPP) "
                    "pseudopotential repositories into Mandacaru's library.")
    for family in FAMILY_SUBDIRS:
        parser.add_argument(f"--{family}", metavar="DIR",
                            help=f"directory with the {family} datasets")
    parser.add_argument("--files", action="store_true",
                        help="link file by file instead of the directory")
    parser.add_argument("--force", action="store_true",
                        help="replace existing links")
    parser.add_argument("--status", action="store_true",
                        help="only report what the library currently serves")
    args = parser.parse_args(argv)

    requested = {f: getattr(args, f) for f in FAMILY_SUBDIRS if getattr(args, f)}
    if not requested and not args.status:
        parser.error("give at least one of --oncvpsp / --paw / --ncpp, or --status")
    for family, source in requested.items():
        link_library(family, source, files=args.files, force=args.force)
    for family, (target, source, count) in status().items():
        state = f"{count} datasets" + (f"  ({source})" if source and source != target
                                       else "")
        print(f"{family:8s} {target}: {state if count else 'MISSING'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
