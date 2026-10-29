# -*- coding: utf-8 -*-
# file: test_single_entry_point.py

"""The algorithms are called solely through :class:`Mandacaru`.

``Mandacaru(method=...)`` is the one way in.  The solver classes are an
internal layer: they are not exported, and nothing outside the package -- no
test, example or documentation page -- constructs or imports one.  This file
enforces both halves.
"""

import importlib.util
import io
import pathlib
import re
import tokenize

import pytest

import mandacaru
from mandacaru import Mandacaru
from mandacaru.specifications import available_methods

REPO = pathlib.Path(__file__).resolve().parents[1]
# ``BlochCalculator`` was public until the periodic driver moved behind the
# entry point; it is listed here so the sweep fails if it ever comes back.
SOLVER_CLASSES = ("RHFDriver", "UHFDriver", "DFTDriver", "HVA", "VQE",
                  "ADAPTVQE", "SubspaceVQE", "SubspaceADAPTVQE", "MCASVQE",
                  "BlochCalculator")


class TestTheCalculatorIsTheEntryPoint:
    @pytest.mark.parametrize("method", available_methods())
    def test_every_method_is_built_by_the_constructor(self, method):
        assert Mandacaru(method=method).method == method

    def test_the_default_method_is_adapt_vqe(self):
        assert Mandacaru().method == "adapt-vqe"

    def test_unknown_method_is_refused(self):
        with pytest.raises(ValueError, match="unsupported native method"):
            Mandacaru(method="qaoa")

    def test_invalid_options_are_refused_by_the_constructor(self):
        """Not at the first energy: the options are validated with the calculator."""
        with pytest.raises(ValueError, match="gradient"):
            Mandacaru(method="adapt-vqe", gradient="nope")


class TestTheSolverClassesAreInternal:
    def test_the_removed_algorithms_package_stays_removed(self):
        assert importlib.util.find_spec("mandacaru.algorithms") is None

    @pytest.mark.parametrize("name", SOLVER_CLASSES)
    def test_not_exported(self, name):
        assert not hasattr(mandacaru, name)
        assert name not in mandacaru.__all__

    @staticmethod
    def _python_sources():
        for folder in ("test", "examples"):
            yield from sorted((REPO / folder).rglob("*.py"))

    def test_no_test_or_example_constructs_or_imports_one(self):
        offenders = []
        for path in self._python_sources():
            if path.name == pathlib.Path(__file__).name:
                continue
            text = path.read_text(encoding="utf-8")
            tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
            for i, tok in enumerate(tokens):
                if tok.type != tokenize.NAME or tok.string not in SOLVER_CLASSES:
                    continue
                before = tokens[i - 1].string if i else ""
                after = tokens[i + 1].string if i + 1 < len(tokens) else ""
                line = tok.line.strip()
                if after == "(" and before not in ("class", "def", "."):
                    offenders.append(f"{path.relative_to(REPO)}:{tok.start[0]}: "
                                     f"constructs {tok.string}")
                elif line.startswith(("from ", "import ")):
                    offenders.append(f"{path.relative_to(REPO)}:{tok.start[0]}: "
                                     f"imports {tok.string}")
        assert not offenders, "\n".join(offenders)

    def test_no_documentation_page_constructs_one(self):
        pattern = re.compile(
            r"(?<![\w.])(?:%s)\(" % "|".join(SOLVER_CLASSES))
        pages = [REPO / "README.md", *sorted((REPO / "docs").rglob("*.md")),
                 *sorted((REPO / "docs").rglob("*.rst"))]
        offenders = [f"{page.relative_to(REPO)}:{number}"
                     for page in pages if page.exists()
                     for number, line in enumerate(
                         page.read_text(encoding="utf-8").splitlines(), 1)
                     if pattern.search(line)]
        assert not offenders, "\n".join(offenders)
