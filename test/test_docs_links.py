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
