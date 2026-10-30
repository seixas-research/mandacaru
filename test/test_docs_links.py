# -*- coding: utf-8 -*-
# file: test/test_docs_links.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Cross-references in the manual must resolve, checked without Sphinx.

Read the Docs builds with ``fail_on_warning: true`` (``.readthedocs.yaml``), so
**one** unresolved link fails the whole build and the site silently keeps
serving the previous version.  That is exactly what happened once: a link
written as ``[text](run_output.md#some-heading-slug)`` pointed at a heading
anchor that does not exist, because ``myst_heading_anchors`` is not enabled in
``docs/source/conf.py`` -- MyST generates no anchors from headings at all.

The fix and the convention are the same thing: a link to a *section* points at
an **explicit MyST target** placed above the heading::

    (references-bib)=
    ## `references.bib`: what the run should cite

    ... and elsewhere: [the bibliography](#references-bib)

A Sphinx build catches this, but it is slow and needs the docs extras; these
checks are a grep and run in milliseconds, so the class of mistake cannot come
back unnoticed between builds.
"""

from __future__ import annotations

import os
import re

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(REPO, "docs", "source")

#: ``(target)=`` on a line of its own: MyST's explicit cross-reference target.
TARGET = re.compile(r"^\(([^)]+)\)=\s*$", re.M)
#: A Markdown link whose destination is a bare ``#fragment``.
FRAGMENT_LINK = re.compile(r"\]\(#([^)]+)\)")
#: A Markdown link into another page *with* a fragment.
CROSS_PAGE_FRAGMENT = re.compile(r"\]\(([^)#]+\.md)#([^)]+)\)")


def markdown_pages():
    for root, _dirs, files in os.walk(DOCS):
        for name in sorted(files):
            if name.endswith(".md"):
                yield os.path.join(root, name)


def strip_code(text: str) -> str:
    """Drop code: a link inside it is a sample of syntax, not a link.

    Fenced blocks first (they contain backticks), then inline spans -- which is
    how ``contributing_docs.md`` can quote the broken form as an example of
    what not to write.  A real link keeps its destination outside the span, so
    only its *text* is removed and the fragment is still checked.
    """
    text = re.sub(r"^```.*?^```", "", text, flags=re.M | re.S)
    return re.sub(r"`[^`\n]*`", "", text)


@pytest.fixture(scope="module")
def pages():
    return {path: strip_code(open(path, encoding="utf-8").read())
            for path in markdown_pages()}


@pytest.fixture(scope="module")
def targets(pages):
    """Every explicit target in the manual, mapped to the page defining it."""
    found: dict[str, str] = {}
    for path, text in pages.items():
        for name in TARGET.findall(text):
            found[name] = path
    return found


def test_the_manual_has_pages_to_check(pages):
    """Otherwise every test below passes for the wrong reason."""
    assert len(pages) > 10


def test_every_fragment_link_points_at_a_defined_target(pages, targets):
    missing = []
    for path, text in pages.items():
        for name in FRAGMENT_LINK.findall(text):
            if name not in targets:
                missing.append(f"{os.path.relpath(path, REPO)} -> #{name}")
    assert not missing, (
        "unresolved MyST cross-reference(s); define the target as "
        "'(name)=' on the line above the heading:\n  " + "\n  ".join(missing))


def test_no_link_relies_on_a_heading_anchor(pages):
    """``page.md#slug`` needs ``myst_heading_anchors``, which is off."""
    offenders = []
    for path, text in pages.items():
        for page, fragment in CROSS_PAGE_FRAGMENT.findall(text):
            offenders.append(
                f"{os.path.relpath(path, REPO)} -> {page}#{fragment}")
    assert not offenders, (
        "link(s) to a heading anchor, which MyST does not generate "
        "(myst_heading_anchors is not set in docs/source/conf.py); use an "
        "explicit '(target)=' and link to '#target':\n  "
        + "\n  ".join(offenders))


def test_heading_anchors_are_still_disabled():
    """The guard above is only needed while they are off; say so out loud."""
    conf = open(os.path.join(DOCS, "conf.py"), encoding="utf-8").read()
    assert "myst_heading_anchors" not in conf, (
        "myst_heading_anchors is now set -- heading slugs resolve, so "
        "test_no_link_relies_on_a_heading_anchor can be relaxed or removed.")


def test_every_relative_page_link_exists(pages):
    """A link to a page that was renamed or never existed."""
    missing = []
    for path, text in pages.items():
        for link in re.findall(r"\]\(([^)]+\.md)\)", text):
            target = os.path.normpath(os.path.join(os.path.dirname(path),
                                                   link))
            if not os.path.isfile(target):
                missing.append(f"{os.path.relpath(path, REPO)} -> {link}")
    assert not missing, "link(s) to a missing page:\n  " + "\n  ".join(missing)


# --------------------------------------------------------------------------- #
# The manual describes the program that exists.
#
# The links above cannot tell a correct page from a stale one.  These checks
# compare what the manual says with the code: a stale option, a module that was
# deleted in the migration to Rust, a method that is not in the table, or a
# constructor parameter nobody documented.
# --------------------------------------------------------------------------- #

#: Pages whose prose names removed modules on purpose (what was removed, and
#: when).  Everything else must describe the code that exists now.
HISTORICAL_PAGES = {os.path.join("development", "history.md"),
                    os.path.join("development", "native_cutover.md")}
FENCED_PYTHON = re.compile(r"^```(?:python|py)\n(.*?)^```", re.M | re.S)
URL = re.compile(r"https?://\S+")
DOTTED_NAME = re.compile(r"\bmandacaru(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
EXAMPLE_PATH = re.compile(r"\bexamples/[A-Za-z0-9_./-]+\.py\b")


def _manual_pages(historical=False):
    for path in markdown_pages():
        relative = os.path.relpath(path, DOCS)
        if historical or relative not in HISTORICAL_PAGES:
            yield relative, open(path, encoding="utf-8").read()


def _constructor():
    import inspect

    from mandacaru import Mandacaru

    return Mandacaru, set(inspect.signature(Mandacaru.__init__).parameters) - {"self"}


def test_every_mandacaru_keyword_in_the_manual_is_an_option():
    """A code sample must not pass an option the constructor does not take."""
    import ast

    _cls, options = _constructor()
    wrong = []
    for relative, text in _manual_pages():
        for block in FENCED_PYTHON.findall(text):
            try:
                tree = ast.parse(block)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "id", getattr(node.func, "attr", ""))
                if name != "Mandacaru":
                    continue
                for keyword in node.keywords:
                    if keyword.arg and keyword.arg not in options:
                        wrong.append(f"{relative}: Mandacaru({keyword.arg}=...)")
    assert not wrong, "option(s) the constructor does not take:\n  " + "\n  ".join(sorted(set(wrong)))


def test_every_mandacaru_name_in_the_manual_resolves():
    """``mandacaru.some.module.name`` must import and exist."""
    import importlib

    def resolves(dotted):
        parts = dotted.split(".")
        for cut in range(len(parts), 0, -1):
            try:
                obj = importlib.import_module(".".join(parts[:cut]))
            except Exception:
                continue
            for part in parts[cut:]:
                if not hasattr(obj, part):
                    return False
                obj = getattr(obj, part)
            return True
        return False

    missing = []
    for relative, text in _manual_pages():
        for name in sorted(set(DOTTED_NAME.findall(URL.sub("", text)))):
            if not resolves(name):
                missing.append(f"{relative}: {name}")
    assert not missing, "name(s) that do not exist:\n  " + "\n  ".join(missing)


def test_the_methods_table_matches_the_code():
    """``reference/methods.md`` lists exactly the methods ``Mandacaru`` builds."""
    from mandacaru.specifications import available_methods

    path = os.path.join(DOCS, "reference", "methods.md")
    assert os.path.isfile(path), "docs/source/reference/methods.md is the methods table"
    text = open(path, encoding="utf-8").read()
    # only the table under the '(methods-table)=' target: the page may hold others
    assert "(methods-table)=" in text, "the methods table needs the '(methods-table)=' target"
    rows, in_table = [], False
    for line in text.split("(methods-table)=", 1)[1].splitlines():
        if line.startswith("|"):
            in_table = True
            rows.append(line)
        elif in_table:
            break
    documented = set(re.findall(r"^\|\s*`([a-z][a-z-]*)`\s*\|", "\n".join(rows), re.M))
    assert documented == set(available_methods()), (
        f"documented {sorted(documented)} != code {sorted(available_methods())}")


def test_every_constructor_parameter_is_documented():
    """The class docstring is the options reference: one entry per parameter."""
    import inspect

    cls, options = _constructor()
    doc = inspect.getdoc(cls) or ""
    missing = sorted(name for name in options
                     if not re.search(rf"^{re.escape(name)}\s*:", doc, re.M))
    assert not missing, ("constructor parameter(s) with no 'name : type' entry in the "
                         "Mandacaru docstring: " + ", ".join(missing))


def test_example_files_named_in_the_manual_exist():
    missing = []
    for relative, text in _manual_pages(historical=True):
        for name in sorted(set(EXAMPLE_PATH.findall(text))):
            if not os.path.isfile(os.path.join(REPO, name)):
                missing.append(f"{relative}: {name}")
    assert not missing, "example(s) that do not exist:\n  " + "\n  ".join(missing)
