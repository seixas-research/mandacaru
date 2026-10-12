# -*- coding: utf-8 -*-
# file: test_single_entry_point.py

"""The algorithms are called solely through :class:`Mandacaru`.

``Mandacaru(method="vqe" | "adapt-vqe" | "subspace-vqe" | "subspace-adapt-vqe"
| "mcas-vqe")``
is the one way in.  The solver classes are the internal layer: they are not
exported, and nothing outside the package -- no test, example or documentation
page -- constructs or imports one.  This file enforces both halves, and that
the calculator's own API is explicit: it forwards nothing to the solver.
"""

import inspect
import io
import pathlib
import re
import tokenize

import numpy as np
import pytest

import mandacaru.algorithms as algorithms
from mandacaru import Mandacaru

REPO = pathlib.Path(__file__).resolve().parents[1]
# ``BlochCalculator`` was public until the periodic driver moved behind the
# entry point; it is listed here so the sweep fails if it ever comes back.
SOLVER_CLASSES = ("RHFDriver", "UHFDriver", "DFTDriver", "HVA", "VQE",
                  "ADAPTVQE", "SubspaceVQE", "SubspaceADAPTVQE", "MCASVQE",
                  "BlochCalculator")
# ``HVA`` was a method until it became ``method="vqe", ansatz="hva"``; it
# stays in SOLVER_CLASSES so the sweep fails if the class ever comes back.
METHODS = {"rhf": "RHFDriver", "uhf": "UHFDriver", "dft": "DFTDriver",
           "vqe": "VQE",
           "adapt-vqe": "ADAPTVQE", "subspace-vqe": "SubspaceVQE",
           "subspace-adapt-vqe": "SubspaceADAPTVQE", "mcas-vqe": "MCASVQE",
           "bloch-vqe": "BlochVQE", "bloch-adapt-vqe": "BlochADAPTVQE"}


#: Options a method cannot be built without (the periodic ones need a mesh).
METHOD_OPTIONS = {"bloch-vqe": {"kpts": {"size": (1, 1, 1), "gamma": True}},
                  "bloch-adapt-vqe": {"kpts": {"size": (1, 1, 1), "gamma": True}}}


class TestTheCalculatorIsTheEntryPoint:
    @pytest.mark.parametrize("method, solver", sorted(METHODS.items()))
    def test_every_method_builds_its_solver(self, method, solver):
        options = METHOD_OPTIONS.get(method, {})
        assert type(Mandacaru(method=method, **options).solver).__name__ == solver

    @pytest.mark.parametrize("spelling", ["adapt-vqe", "adaptvqe", "ADAPT_VQE",
                                          "Adapt VQE", " adapt-vqe "])
    def test_method_spelling_is_forgiving(self, spelling):
        assert Mandacaru(method=spelling).method == "adapt-vqe"

    def test_unknown_method_is_refused(self):
        with pytest.raises(ValueError, match="unknown method"):
            Mandacaru(method="qaoa")

    def test_invalid_options_are_refused_by_the_constructor(self):
        """Not at the first energy: the solver is built with the calculator."""
        with pytest.raises(ValueError, match="gradient"):
            Mandacaru(method="adapt-vqe", gradient="nope")

    def test_solver_internals_are_not_forwarded(self):
        """The calculator has an explicit API; nothing reaches the solver
        implicitly.  Solver internals are read on :attr:`Mandacaru.solver`."""
        calc = Mandacaru(method="adapt-vqe", pool="qeb", quenching=False)
        assert calc.solver.quenching is False
        assert calc.solver.gradient == "analytic"
        for name in ("quenching", "gradient", "no_such_attribute"):
            with pytest.raises(AttributeError, match=f"no attribute '{name}'"):
                getattr(calc, name)


#: Each explicit member delegating to a solver, and a method providing it.
SOLVER_MEMBERS = {
    "get_fermi_level": "dft", "get_number_of_spins": "dft",
    "get_eigenvalues": "dft", "get_ibz_k_points": "dft",
    "get_k_point_weights": "dft", "get_bz_k_points": "bloch-vqe",
    "kpoints": "rhf", "n_bands": "bloch-vqe", "dimension": "bloch-vqe",
    "symmetry": "bloch-vqe", "irreducible_zone": "bloch-vqe",
    "band_path": "bloch-vqe", "get_spectral_function": "bloch-vqe",
    "bands": "bloch-vqe", "band_weights": "bloch-vqe", "dos": "dft",
    "pdos": "dft", "fat_bands": "dft", "fermi_surface": "dft",
    "write_fermi_surface": "dft", "get_polarization": "dft",
    "polarization_result": "dft", "projectabilities": "dft",
    "wannier": "dft", "citation_keys": "rhf", "write_references": "rhf"}

#: Members whose solvers disagree on the signature (the calculator passes
#: the arguments through).
PASS_THROUGH = {"get_fermi_level"}


def _calculator(method):
    return Mandacaru(method=method, **METHOD_OPTIONS.get(method, {}))


class TestTheCalculatorAPIIsExplicit:
    """The solver's user API is spelled out on :class:`Mandacaru`."""

    @pytest.mark.parametrize("name", sorted(SOLVER_MEMBERS))
    def test_every_member_is_defined_on_the_calculator(self, name):
        assert name in vars(Mandacaru), name

    @pytest.mark.parametrize("name", sorted(set(SOLVER_MEMBERS) - PASS_THROUGH
                                            - {"kpoints", "n_bands",
                                               "dimension",
                                               "polarization_result"}))
    def test_signatures_are_the_solvers(self, name):
        solver = type(_calculator(SOLVER_MEMBERS[name]).solver)

        def parameters(function):
            return [(p.name, p.kind, p.default) for p in
                    inspect.signature(function).parameters.values()]
        assert parameters(getattr(Mandacaru, name)) == \
            parameters(getattr(solver, name))

    def test_members_delegate_to_the_solver(self):
        calc = _calculator("rhf")
        assert calc.citation_keys() == calc.solver.citation_keys()
        assert np.array_equal(calc.kpoints, calc.solver.kpoints)
        bloch = Mandacaru(method="bloch-vqe",
                          kpts={"size": (2, 1, 1), "gamma": True})
        assert np.array_equal(bloch.get_bz_k_points(),
                              bloch.solver.get_bz_k_points())

    @pytest.mark.parametrize("method, call, message", [
        ("rhf", lambda calc: calc.bands(),
         r"bands\(\) needs method='bloch-vqe' or 'bloch-adapt-vqe'; "
         r"method='rhf' does not provide it"),
        ("adapt-vqe", lambda calc: calc.dos(), r"dos\(\) needs method='dft'"),
        ("bloch-vqe", lambda calc: calc.wannier(),
         r"wannier\(\) needs a periodic method='dft' calculation"),
        ("vqe", lambda calc: calc.get_fermi_level(),
         r"get_fermi_level\(\) needs method='dft' or a Bloch method"),
        ("bloch-vqe", lambda calc: calc.get_eigenvalues(),
         r"get_eigenvalues\(\) needs method='dft'")])
    def test_an_unsupported_method_names_what_supports_it(self, method, call,
                                                          message):
        with pytest.raises(NotImplementedError, match=message):
            call(_calculator(method))

    @pytest.mark.parametrize("name", ["n_bands", "dimension",
                                      "polarization_result"])
    def test_an_unsupported_property_is_an_attribute_error(self, name):
        calc = _calculator("rhf")
        with pytest.raises(AttributeError, match=f"{name} needs"):
            getattr(calc, name)
        assert not hasattr(calc, name)

    def test_the_polarization_result_is_none_before_a_polarization(self):
        assert _calculator("dft").polarization_result is None


class TestTheSolverClassesAreInternal:
    @pytest.mark.parametrize("name", SOLVER_CLASSES)
    def test_not_exported(self, name):
        assert not hasattr(algorithms, name)
        assert name not in algorithms.__all__

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
