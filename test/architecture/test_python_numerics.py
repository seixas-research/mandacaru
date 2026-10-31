"""The numerics gate: the counter, the budget file and the ratchet.

``scripts/numerics_budget.py`` counts the numerical work still done in Python.
The counter is tested on synthetic source (every evasion must score), the
comparison on synthetic packages, and the repository against its own budget.
The date check lives in ``scripts/check_architecture.py``, outside pytest, so
the suite stays reproducible; its logic is tested here with an injected date.
"""

import copy
import datetime
import runpy
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GATE = runpy.run_path(str(ROOT / "scripts/numerics_budget.py"))
CHECKS = runpy.run_path(str(ROOT / "scripts/check_architecture.py"))
HARD = GATE["HARD_ZERO"]


def counts(source, **kwargs):
    return GATE["measure_source"](textwrap.dedent(source), **kwargs)["counts"]


def nonzero(source, **kwargs):
    return {k: v for k, v in counts(source, **kwargs).items() if v and k != "sloc"}


# ---------------------------------------------------------------------------
# What scores zero
# ---------------------------------------------------------------------------
def test_docstrings_comments_and_strings_score_zero():
    source = '''
        """np.sum(a), scipy.special.jv(1, x), eval("1"), x @ y."""
        import numpy as np  # np.einsum("ij", a)

        def f(a):
            """np.linalg.norm(a); sum(a); for i in range(3): a = 2.0 * a"""
            # np.sum(a) @ b
            text = "np.sum(a) @ b, import scipy, sum(a)"
            note = f"np.sum(a) and {1}"
            return np.asarray(a), text, note
    '''
    assert nonzero(source) == {}
    assert counts(source)["sloc"] == 5


@pytest.mark.parametrize(
    "source",
    [
        "import numpy as np\nnp.asarray(1); np.zeros(3); np.float64(1); np.pi; np.dtype(float)",
        "import numpy as np\nnp.isfinite(1.0); np.allclose(1, 2); np.errstate(all='ignore')",
        "from numpy import asarray, float64\nasarray(1)",
        "import math\nmath.pi; math.isfinite(1.0); math.inf",
        "from math import pi\nx = pi",
        "max(1, 2); abs(-1); sorted((1, 2)); sum([1, 2])",
        "from mandacaru import units\nx = units.HARTREE_TO_EV",
        "import json, os\nfrom pathlib import Path\nfrom . import sibling\nfrom ..up import thing",
        "from ase import Atoms\nfrom ase.io import read\nfrom ase.data import chemical_symbols",
        "x = 'a' * 3\ny = '%d' % 3\nz = 2 * 3",
        "import numpy as np\nx: np.typing.NDArray\ny: np.typing.ArrayLike",
        "from numpy.typing import NDArray, DTypeLike\nx: NDArray",
        "import numpy as np\nnp.float64(1.0); np.bool_(True); np.errstate(all='ignore')",
    ],
)
def test_plumbing_and_literals_score_zero(source):
    assert nonzero(source) == {}


# ---------------------------------------------------------------------------
# What must score (every evasion is a case)
# ---------------------------------------------------------------------------
CASES = {
    "numpy call": ("import numpy as np\nnp.sum([1])", {"np": 1}),
    "renamed numpy": ("import numpy as xp\nxp.linalg.norm(1)", {"np": 1}),
    "numpy submodule alias": ("import numpy.linalg as la\nla.norm(1)", {"np": 1}),
    "numpy by plain import": ("import numpy\nnumpy.fft.fft(1)", {"np": 1}),
    "from-import, used": ("from numpy import sqrt\nsqrt(2.0)", {"np": 2}),
    "from-import, unused": ("from numpy import einsum", {"np": 1}),
    "from-import submodule": ("from numpy import linalg\nlinalg.norm(1)", {"np": 2}),
    "from-import submodule path": ("from numpy.linalg import norm\nnorm(1)", {"np": 2}),
    "function-local import": (
        "def f(x):\n    import numpy as np\n    return np.sum(x)", {"np": 1}),
    "alias by assignment": ("import numpy as np\nxp = np\nxp.sum(1)", {"np": 1}),
    "submodule by assignment": (
        "import numpy as np\nlin = np.linalg\nlin.norm(1)", {"np": 2}),
    "np.lib is not plumbing": (
        "import numpy as np\nnp.lib.stride_tricks.sliding_window_view(1, 2)", {"np": 1}),
    "np.lib.scimath": ("import numpy as np\nnp.lib.scimath.sqrt(-1)", {"np": 1}),
    "np.ma is not plumbing": ("import numpy as np\nnp.ma.masked_array(1)", {"np": 1}),
    "np.testing is not plumbing": (
        "import numpy as np\nnp.testing.assert_allclose(1, 1)", {"np": 1}),
    "np.exceptions is not plumbing": (
        "import numpy as np\nnp.exceptions.AxisError", {"np": 1}),
    "np.ndarray.sum is not plumbing": ("import numpy as np\nnp.ndarray.sum(a)", {"np": 1}),
    "np.float64.dot is not plumbing": ("import numpy as np\nnp.float64.dot(a, a)", {"np": 1}),
    "np.dtype attribute path": ("import numpy as np\nx = np.dtype.kind", {"np": 1}),
    "from-imported plumbing name with a method": (
        "from numpy import float64\nfloat64.dot(a, a)", {"np": 1}),
    "alias of a plumbing name with a method": (
        "import numpy as np\nscalar = np.float64\nscalar.dot(a, a)", {"np": 1}),
    "np.random is counted": ("import numpy as np\nnp.random.default_rng(0)", {"np": 1}),
    "numpy wildcard": ("from numpy import *", {"np": 1}),
    "numpy submodule wildcard": ("from numpy.linalg import *", {"np": 1}),
    "scipy import": ("import scipy.linalg as sl", {"scipy": 1}),
    "scipy from-import": ("from scipy.special import erf", {"scipy": 1}),
    "scipy local import": ("def f():\n    from scipy import optimize", {"scipy": 1}),
    "scipy wildcard": ("from scipy import *", {"scipy": 1}),
    "math call": ("import math\nmath.sqrt(2.0)", {"math": 1}),
    "math alias": ("import math as m\nm.floor(2.0)", {"math": 1}),
    "cmath call": ("import cmath\ncmath.exp(1j)", {"math": 1}),
    "bare math name": ("from math import sqrt\nsqrt(2.0)", {"math": 2}),
    "bare math name renamed": ("from math import sqrt as root\nroot(2.0)", {"math": 2}),
    "math wildcard": ("from math import *", {"math": 1}),
    "matrix product": ("c = a @ b", {"matmul": 1}),
    "in-place matrix product": ("a @= b", {"matmul": 1}),
    "ndarray method": ("x.sum()", {"meth": 1}),
    "method on a call": ("f(x).argsort()", {"meth": 1}),
    "method list": ("x.dot(y); x.clip(0, 1); x.cumsum(); x.searchsorted(1)", {"meth": 4}),
    "native FFI": ("import ctypes", {"ctypes": 1}),
    "native FFI from": ("from cffi import FFI", {"ctypes": 1}),
    "jit": ("import numba", {"ctypes": 1}),
    "builtin sum": ("sum(x)", {"bi": 1}),
    "builtin max": ("max(a, b)", {"bi": 1}),
    "builtin sorted": ("sorted(items)", {"bi": 1}),
    "builtin round": ("round(x, 3)", {"bi": 1}),
    "builtin sum of a slice product": ("sum(x[1:] * y[:-1])", {"bi": 1, "arith": 1}),
    "builtin sum of a generator": ("sum(1 for _ in x)", {"bi": 1}),
    "eval": ("eval('1')", {"dyn": 1}),
    "exec": ("exec('x = 1')", {"dyn": 1}),
    "compile": ("compile('1', 'f', 'eval')", {"dyn": 1}),
    "__import__": ("__import__('numpy')", {"dyn": 1}),
    "import_module": (
        "import importlib\nimportlib.import_module('numpy')", {"dyn": 1}),
    "import_module renamed": (
        "from importlib import import_module as load\nload('scipy.linalg')", {"dyn": 1}),
    "import_module absolute with a package": (
        "from importlib import import_module\nimport_module('numpy', __package__)", {"dyn": 1}),
    "getattr on numpy": ("import numpy as np\ngetattr(np, 'sum')([1])", {"dyn": 1}),
    "getattr on an alias of numpy": (
        "import numpy as np\nxp = np\ngetattr(xp, 'sum')([1])", {"dyn": 1}),
    "numpy handed around": ("import numpy as np\nrun(np)", {"dyn": 1}),
    "math handed around": ("import math\nrun(math)", {"dyn": 1}),
    "globals lookup": ("globals()['np'].sum([1])", {"dyn": 1, "meth": 1}),
    "vars lookup": ("vars()['np']", {"dyn": 1}),
    "sys.modules lookup": ("import sys\nsys.modules['numpy']", {"dyn": 1}),
    "numpy name next to import_module": (
        "from importlib import import_module\nNAME = 'numpy'\nimport_module('.' + n, __name__)",
        {"dyn": 1}),
    "loop over data": (
        "import numpy as np\nfor i in range(3):\n    y = np.sqrt(i)", {"np": 1, "dloops": 1}),
    "loop over data, bare math": (
        "from math import sqrt\nfor i in r:\n    sqrt(i)", {"math": 2, "dloops": 1}),
    "loop over data, builtin": ("for i in r:\n    s = sum(a)", {"bi": 1, "dloops": 1}),
    "loop over data, method": ("while True:\n    x.sum()", {"meth": 1, "dloops": 1}),
    "loop with arithmetic only": ("for i in r:\n    y[i] = 2.0 * x[i]", {"aloops": 1, "arith": 1}),
    "comprehension over data": (
        "import numpy as np\ny = [np.sqrt(v) for v in x]", {"np": 1, "dcomp": 1}),
    "comprehension with arithmetic only": (
        "y = [2.0 * v for v in x]", {"acomp": 1, "arith": 1}),
    "generator over data": ("y = list(x.sum() for x in xs)", {"meth": 1, "dcomp": 1}),
    "operator arithmetic": ("c = a * b / d ** 2", {"arith": 3, "arithmax": 3}),
    "in-place arithmetic": ("a *= b", {"arith": 1}),
}


@pytest.mark.parametrize("source, expected", CASES.values(), ids=CASES.keys())
def test_the_counter_scores_every_evasion(source, expected):
    got = nonzero(source)
    for key, value in expected.items():
        assert got.get(key, 0) == value, (key, got)
    assert set(got) - {"arithmax"} <= set(expected) | {"arith"}, got


EXPORTER = "import numpy as np\nimport math\n"
REEXPORTS = {
    "relative": ("basis/importer.py", "from .exporter import np\ndef f(x):\n    return np.sqrt(x)\n", {"np": 1}),
    "absolute": ("basis/importer.py", "from mandacaru.basis.exporter import np\ndef f(x):\n    return np.linalg.norm(x)\n", {"np": 1}),
    "renamed": ("basis/importer.py", "from .exporter import np as xp\ndef f(x):\n    return xp.linalg.norm(x)\n", {"np": 1}),
    "the math module": ("basis/importer.py", "from .exporter import math\ndef f(x):\n    return math.sqrt(x)\n", {"math": 1}),
    "a wildcard": ("basis/importer.py", "from .exporter import *\ndef f(x):\n    return np.sum(x)\n", {"np": 1}),
    "the parent package": ("basis/sub/importer.py", "from ..exporter import np\ndef f(x):\n    return np.sqrt(x)\n", {"np": 1}),
    "a function-local import": ("basis/importer.py", "def f(x):\n    from .exporter import np\n    return np.sqrt(x)\n", {"np": 1}),
    "numpy handed around": ("basis/importer.py", "from .exporter import np\nrun(np)\n", {"dyn": 1}),
}


@pytest.mark.parametrize("where, source, expected", REEXPORTS.values(), ids=REEXPORTS.keys())
def test_a_numpy_name_re_exported_by_another_module_still_scores(tmp_path, where, source, expected):
    """The exporter holds an unused ``import numpy as np``; the importer's use must count."""
    base = "basis/exporter.py"
    measured = GATE["measure_package"](package(tmp_path, {base: EXPORTER, where: source}))
    assert {k: v for k, v in measured[base]["counts"].items() if v and k != "sloc"} == {}
    got = {k: v for k, v in measured[where]["counts"].items() if v and k not in ("sloc", "arith", "arithmax")}
    assert got == expected


def test_a_re_export_is_followed_through_a_chain_and_a_package_init(tmp_path):
    files = {
        "basis/exporter.py": EXPORTER,
        "basis/guarded.py": "from contextlib import suppress\nwith suppress(ImportError):\n    import numpy as np\n",
        "basis/guarded_user.py": "from .guarded import np\ndef f(x):\n    return np.sqrt(x)\n",
        "basis/middle.py": "from .exporter import np\n",
        "basis/__init__.py": "from .middle import np\n",
        "basis/importer.py": "from . import np\ndef f(x):\n    return np.sqrt(x)\n",
        "basis/by_name.py": "from mandacaru.basis import np\ndef f(x):\n    return np.sqrt(x)\n",
        "basis/sqrt_exporter.py": "from numpy import sqrt\n",
        "basis/sqrt_importer.py": "from .sqrt_exporter import sqrt\nsqrt(2.0)\n",
    }
    measured = GATE["measure_package"](package(tmp_path, files))
    assert measured["basis/guarded_user.py"]["counts"]["np"] == 1
    assert measured["basis/importer.py"]["counts"]["np"] == 1
    assert measured["basis/by_name.py"]["counts"]["np"] == 1
    assert measured["basis/sqrt_importer.py"]["counts"]["np"] == 1
    assert measured["basis/sqrt_exporter.py"]["counts"]["np"] == 1


def test_numpy_reached_through_a_module_attribute_scores_as_dyn(tmp_path):
    files = {"basis/exporter.py": EXPORTER,
             "basis/importer.py": "from . import exporter\ndef f(x):\n    return exporter.np.sqrt(x)\n",
             "basis/other.py": "import mandacaru.basis.exporter as e\ne.math.sqrt(1.0)\n"}
    measured = GATE["measure_package"](package(tmp_path, files))
    assert measured["basis/importer.py"]["counts"]["dyn"] == 1
    assert measured["basis/other.py"]["counts"]["dyn"] == 1


def test_a_re_exported_plumbing_name_and_an_unrelated_np_stay_clean(tmp_path):
    files = {"basis/exporter.py": EXPORTER,
             "basis/plumbing.py": "from .exporter import np\nx = np.zeros(3)\ny = np.asarray(x)\n",
             "basis/local.py": "from .helpers import np\nnp.sqrt(2.0)\n",
             "basis/helpers.py": "np = object()\n"}
    measured = GATE["measure_package"](package(tmp_path, files))
    for module in ("basis/plumbing.py", "basis/local.py"):
        assert {k: v for k, v in measured[module]["counts"].items() if v and k != "sloc"} == {}


def test_loops_over_labels_are_not_data_loops():
    source = "for name in ['a', 'b']:\n    print(name)\nlabels = [n.upper() for n in names]"
    assert nonzero(source) == {}


def test_relative_package_imports_are_not_dynamic():
    source = """
        from importlib import import_module
        import_module(".paw", __package__)
        import_module("." + attribute, __name__)
        value = getattr(import_module(module, __name__), attribute)
        globals()[name] = value
    """
    assert nonzero(source) == {}


def test_unit_conversion_and_strings_are_not_arithmetic():
    units = frozenset({"HARTREE_TO_EV", "to_bohr"})
    assert nonzero("e = energy * HARTREE_TO_EV", units=units) == {}
    assert nonzero("e = to_bohr(r) * 2.0", units=units) == {}
    assert nonzero("e *= HARTREE_TO_EV", units=units) == {}
    assert nonzero("e = energy * scale", units=units) == {"arith": 1, "arithmax": 1}


def test_arithmax_is_the_largest_function_not_the_total():
    source = """
        def f(a, b):
            return a * b

        def g(a, b):
            return a * b * a / b
    """
    got = nonzero(source)
    assert got["arith"] == 4 and got["arithmax"] == 3


def test_a_function_keeps_its_own_counters():
    source = """
        import numpy as np

        class K:
            def m(self, x):
                return np.sum(x)

        def f(x):
            def inner(y):
                return np.max(y)
            return inner(x)
    """
    detail = GATE["measure_source"](textwrap.dedent(source))["functions"]
    assert detail["K.m"] == {"np": 1}
    assert detail["f.inner"] == {"np": 1}


# ---------------------------------------------------------------------------
# The import allowlist
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "source, target",
    [
        ("import pandas", "pandas"),
        ("import pyarrow.parquet", "pyarrow.parquet"),
        ("import sympy", "sympy"),
        ("import spglib", "spglib"),
        ("def f():\n    import spglib", "spglib"),
        ("from qiskit.quantum_info import Operator", "qiskit.quantum_info"),
        ("from ase.neighborlist import neighbor_list", "ase.neighborlist"),
        ("from ase import geometry", "ase.geometry"),
        ("import ase.spacegroup", "ase.spacegroup"),
        ("import requests", "requests"),
    ],
)
def test_the_zero_tier_import_allowlist(source, target):
    measured = GATE["measure_source"](source)
    assert measured["counts"]["imp"] == 1
    assert [t for t, _ in measured["imports"]] == [target]


# ---------------------------------------------------------------------------
# The comparison with the budget, on synthetic packages
# ---------------------------------------------------------------------------
def package(tmp_path, files):
    root = tmp_path / "pkg"
    for name, source in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source))
    return root


def budget(modules=None, exceptions=None, zero_sloc=0):
    return {
        "module": copy.deepcopy(modules or {}),
        "exception": copy.deepcopy(exceptions or []),
        "tier_total": {"zero_sloc": zero_sloc},
    }


def failures(tmp_path, files, **kwargs):
    measured = GATE["measure_package"](package(tmp_path, files))
    return GATE["check_budget"](measured, budget(**kwargs))


NUMPY_TWICE = "import numpy as np\nx = np.sum([1])\ny = np.max([1])\n"
CLEAN = {"closed": "hand-off"}


def test_a_matching_budget_passes(tmp_path):
    found = failures(
        tmp_path,
        {"basis/a.py": NUMPY_TWICE, "calculation.py": "import numpy as np\nnp.sum(1)\n"},
        modules={
            "basis/a.py": {"np": 2, "sloc": 3, "wave": "W4", "dies": "2027-07-16"},
            "calculation.py": {"np": 1},
        },
        zero_sloc=3,
    )
    assert found == []


def test_a_regression_fails(tmp_path):
    found = failures(
        tmp_path, {"basis/a.py": NUMPY_TWICE},
        modules={"basis/a.py": {"np": 1, "sloc": 3, "wave": "W4", "dies": "2027-07-16"}},
        zero_sloc=3,
    )
    assert any("np 2 > budget 1" in line and "regression" in line for line in found)


def test_a_stale_hard_counter_fails(tmp_path):
    found = failures(
        tmp_path, {"basis/a.py": NUMPY_TWICE},
        modules={"basis/a.py": {"np": 3, "sloc": 3, "wave": "W4", "dies": "2027-07-16"}},
        zero_sloc=3,
    )
    assert any("np 2 < budget 3" in line and "stale" in line for line in found)


def test_a_stale_soft_ceiling_and_a_regressed_one_fail(tmp_path):
    files = {"basis/a.py": "x = 1\ny = 2\n"}
    row = {"category": "hand-off"}
    stale = failures(tmp_path, files, modules={"basis/a.py": {**row, "sloc": 5}}, zero_sloc=2)
    assert any("sloc 2 < budget 5" in line for line in stale)
    regressed = failures(tmp_path, files, modules={"basis/a.py": {**row, "sloc": 1}}, zero_sloc=2)
    assert any("sloc 2 > budget 1" in line for line in regressed)


def test_an_unlisted_zero_tier_module_is_budgeted_at_zero(tmp_path):
    found = failures(tmp_path, {"integrals/new.py": "x = 1\n"})
    assert any("integrals/new.py" in line and "sloc 1 > budget 0" in line and "unlisted" in line for line in found)
    found = failures(tmp_path, {"models/new.py": "import numpy as np\nnp.sum(1)\n"})
    assert any("models/new.py: np 1 > budget 0" in line for line in found)


def test_the_two_core_modules_and_paw_integrations_are_in_the_zero_tier():
    tier = GATE["tier_of"]
    assert {tier(m) for m in (
        "core/symmetry.py", "core/spin_orbit.py", "integrations/paw.py",
        "integrations/paw_molecular.py", "pseudopotentials/paw.py", "basis/hao.py",
        "integrals/grid.py", "models/basis.py")} == {"zero"}
    assert tier("core/mapping.py") == "q" and tier("workflows/qpe.py") == "q"
    assert tier("integrations/ase.py") == "a" and tier("calculation.py") == "a"


def test_tiers_a_and_q_are_ratchet_only(tmp_path):
    files = {"core/m.py": NUMPY_TWICE, "utils/u.py": "import numpy as np\nnp.sum(1)\n"}
    found = failures(tmp_path, files)
    assert any("core/m.py: np 2 > budget 0" in line for line in found)
    assert any("utils/u.py: np 1 > budget 0" in line for line in found)
    assert failures(tmp_path, files, modules={"core/m.py": {"np": 2}, "utils/u.py": {"np": 1}}) == []
    stale = failures(tmp_path, files, modules={"core/m.py": {"np": 3}, "utils/u.py": {"np": 1}})
    assert any("stale" in line for line in stale)
    dated = failures(tmp_path, files, modules={"core/m.py": {"np": 2, "dies": "2027-01-01"}, "utils/u.py": {"np": 1}})
    assert any("ratchet-only" in line for line in dated)


def test_a_row_for_a_missing_module_fails(tmp_path):
    found = failures(tmp_path, {"basis/a.py": "x = 1\n"},
                     modules={"basis/gone.py": {"sloc": 1, "category": "hand-off"},
                              "basis/a.py": {"sloc": 1, "category": "hand-off"}}, zero_sloc=1)
    assert any("basis/gone.py" in line and "does not exist" in line for line in found)


def test_the_tier_total_is_exact(tmp_path):
    files = {"basis/a.py": "x = 1\n"}
    row = {"basis/a.py": {"sloc": 1, "category": "hand-off"}}
    assert failures(tmp_path, files, modules=row, zero_sloc=1) == []
    assert any("total" in line and "stale" in line for line in failures(tmp_path, files, modules=row, zero_sloc=4))
    assert any("total" in line and "regression" in line for line in failures(tmp_path, files, modules=row, zero_sloc=0))


SORTED_NAMES = """
    def validate(names):
        return sorted(names)

    def other(names):
        return sorted(names)
"""


def test_an_exception_waives_exactly_the_counted_function(tmp_path):
    files = {"models/v.py": SORTED_NAMES}
    row = {"models/v.py": {"bi": 1, "sloc": 4, "wave": "W4", "dies": "2027-07-16"}}
    waiver = {"module": "models/v.py", "function": "validate", "bi": 1,
              "category": "validation", "reason": "sorts names for the error text"}
    assert failures(tmp_path, files, modules=row, exceptions=[waiver], zero_sloc=4) == []
    wrong = dict(waiver, bi=2)
    found = failures(tmp_path, files, modules=row, exceptions=[wrong], zero_sloc=4)
    assert any("waives bi = 2 but the function has 1" in line for line in found)
    gone = dict(waiver, function="vanished")
    found = failures(tmp_path, files, modules=row, exceptions=[gone], zero_sloc=4)
    assert any("function that does not exist" in line for line in found)
    no_reason = {k: v for k, v in waiver.items() if k != "reason"}
    found = failures(tmp_path, files, modules=row, exceptions=[no_reason], zero_sloc=4)
    assert any("needs a reason" in line for line in found)
    outside = dict(waiver, category="orchestration")
    found = failures(tmp_path, files, modules=row, exceptions=[outside], zero_sloc=4)
    assert any("needs wave and dies" in line for line in found)


def waiver_case(tmp_path, source, counter, **fields):
    row = {"sloc": 3, "wave": "W4", "dies": "2027-07-16"}
    waiver = {"module": "models/v.py", "function": "validate", counter: 1, "reason": "test", **fields}
    return failures(tmp_path, {"models/v.py": source}, modules={"models/v.py": row},
                    exceptions=[waiver], zero_sloc=3)


NP_IN_VALIDATE = "import numpy as np\ndef validate(x):\n    return np.sum(x)\n"
DYN_IN_VALIDATE = "def validate(x):\n    return eval(x)\n"


@pytest.mark.parametrize("counter, source, refusal", [
    ("np", NP_IN_VALIDATE, "a validation exception may waive only"),
    ("dyn", DYN_IN_VALIDATE, "dyn is never waivable in the zero tier"),
])
def test_a_closed_category_exception_cannot_waive_a_numeric_or_evasion_counter(
        tmp_path, counter, source, refusal):
    found = waiver_case(tmp_path, source, counter, category="validation")
    assert any(f"exception waives {counter}: {refusal}" in line for line in found), found
    assert any(f"{counter} 1 > budget 0" in line and "regression" in line for line in found), found


def test_a_dated_exception_may_waive_np_but_never_dyn_or_an_import_in_the_zero_tier(tmp_path):
    dated = dict(category="orchestration", wave="W4", dies="2027-07-16")
    assert waiver_case(tmp_path, NP_IN_VALIDATE, "np", **dated) == []
    found = waiver_case(tmp_path, DYN_IN_VALIDATE, "dyn", **dated)
    assert any("dyn is never waivable in the zero tier" in line for line in found)
    assert any("dyn 1 > budget 0" in line for line in found)
    found = waiver_case(tmp_path, "def validate(x):\n    import spglib\n    return 1\n", "imp", **dated)
    assert any("imp is never waivable in the zero tier" in line for line in found)


def test_the_closed_category_waivers_of_the_seed_are_all_scalar_helpers():
    seed = GATE["load_budget"]()
    for item in seed["exception"]:
        if item["category"] in GATE["CLOSED_CATEGORIES"]:
            assert {k for k in item if k in GATE["WAIVABLE"]} <= set(GATE["CLOSED_WAIVABLE"]), item


def test_an_allowed_import_is_named_by_its_row_and_must_be_used(tmp_path):
    files = {"core/symmetry.py": "import spglib\nx = 1\n"}
    row = {"sloc": 2, "imp": 0, "wave": "3b", "dies": "2027-05-07"}
    assert any("imp 1 > budget 0" in line for line in failures(
        tmp_path, files, modules={"core/symmetry.py": {k: v for k, v in row.items() if k != "imp"}}, zero_sloc=2))
    allowed = {**row, "allow_imports": ["spglib"]}
    assert failures(tmp_path, files, modules={"core/symmetry.py": allowed}, zero_sloc=2) == []
    unused = {**row, "allow_imports": ["spglib", "scipy.spatial"]}
    found = failures(tmp_path, files, modules={"core/symmetry.py": unused}, zero_sloc=2)
    assert any("scipy.spatial" in line for line in found)


def test_row_integrity(tmp_path):
    files = {"basis/a.py": "import numpy as np\nnp.sum(1)\n"}
    cases = {
        "a closed row with np": ({"np": 1, "sloc": 2, "category": "hand-off"}, "must show no"),
        "a row without a date": ({"np": 1, "sloc": 2}, "needs wave and dies"),
        "a bad date": ({"np": 1, "sloc": 2, "wave": "3a", "dies": "soon"}, "not an ISO date"),
        "an unknown category": ({"np": 1, "sloc": 2, "category": "misc"}, "not one of"),
        "a dated closed row": ({"sloc": 2, "category": "hand-off", "dies": "2027-01-01"}, "is undated"),
        "an unknown counter": ({"np": 1, "sloc": 2, "wave": "3a", "dies": "2027-01-01", "loops": 3}, "not tracked"),
    }
    for label, (row, expected) in cases.items():
        found = failures(tmp_path, files, modules={"basis/a.py": row}, zero_sloc=2)
        assert any(expected in line for line in found), (label, found)


def test_a_clean_closed_row_passes_integrity(tmp_path):
    found = failures(tmp_path, {"basis/a.py": "x = 1\n"},
                     modules={"basis/a.py": {"sloc": 1, "category": "validation"}}, zero_sloc=1)
    assert found == []


# ---------------------------------------------------------------------------
# The date check
# ---------------------------------------------------------------------------
def dated(dies):
    return budget(modules={"basis/a.py": {"sloc": 1, "wave": "3a", "dies": dies}},
                  exceptions=[{"module": "basis/a.py", "function": "f", "bi": 1,
                               "category": "validation", "reason": "x"}])


def test_the_date_check_warns_thirty_days_ahead_and_fails_past_it():
    day = datetime.date(2027, 3, 26)
    warnings, failed = GATE["date_findings"](dated("2027-03-26"), day - datetime.timedelta(days=31))
    assert warnings == [] and failed == []
    warnings, failed = GATE["date_findings"](dated("2027-03-26"), day - datetime.timedelta(days=30))
    assert len(warnings) == 1 and failed == []
    warnings, failed = GATE["date_findings"](dated("2027-03-26"), day)
    assert len(warnings) == 1 and failed == []
    warnings, failed = GATE["date_findings"](dated("2027-03-26"), day + datetime.timedelta(days=1))
    assert warnings == [] and len(failed) == 1 and "1 days past" in failed[0]


def test_architecture_check_runs_the_date_check(capsys):
    assert CHECKS["check_numerics_dates"](datetime.date(2026, 10, 30)) == []
    late = CHECKS["check_numerics_dates"](datetime.date(2099, 1, 1))
    assert late and all("past" in line for line in late)


def test_every_date_in_the_budget_is_at_least_a_stage_away():
    """The seed is dated stage end plus 28 days; nothing is due before 2026-12-11."""
    rows = GATE["load_budget"]()
    dates = [r["dies"] for r in rows["module"].values() if r.get("dies")]
    assert dates and min(dates) >= "2026-12-11"


# ---------------------------------------------------------------------------
# The monotone check against the committed budget
# ---------------------------------------------------------------------------
def committed():
    return budget(
        modules={"basis/a.py": {"np": 3, "sloc": 10, "wave": "3a", "dies": "2027-03-26",
                                "allow_imports": ["spglib"]}},
        exceptions=[{"module": "basis/a.py", "function": "f", "bi": 2, "category": "validation",
                     "reason": "x", "dies": "2027-03-26"}],
        zero_sloc=10,
    )


def test_a_budget_no_looser_than_head_passes():
    assert GATE["monotone"](committed(), committed()) == []
    lower = committed()
    lower["module"]["basis/a.py"].update(np=2, sloc=8, dies="2027-02-01")
    lower["module"]["basis/a.py"]["allow_imports"] = []
    lower["tier_total"]["zero_sloc"] = 8
    del lower["module"]["basis/a.py"]["allow_imports"]
    assert GATE["monotone"](committed(), lower) == []
    removed = committed()
    removed["module"] = {}
    removed["exception"] = []
    assert GATE["monotone"](committed(), removed) == []


@pytest.mark.parametrize(
    "edit, expected",
    [
        (lambda b: b["module"]["basis/a.py"].update(np=4), "np rose from 3 to 4"),
        (lambda b: b["module"]["basis/a.py"].update(sloc=11), "sloc rose from 10 to 11"),
        (lambda b: b["module"]["basis/a.py"].update(aloops=1), "aloops rose from 0 to 1"),
        (lambda b: b["module"]["basis/a.py"].update(dies="2027-04-01"), "dies moved later"),
        (lambda b: b["module"]["basis/a.py"].pop("dies"), "dies 2027-03-26 was removed"),
        (lambda b: b["module"]["basis/a.py"].update(dies="soon"), "is not an ISO date"),
        (lambda b: b["module"]["basis/a.py"].update(wave="W4"), "wave changed ('3a' -> 'W4')"),
        (lambda b: b["module"]["basis/a.py"].update(category="hand-off"), "category changed (None -> 'hand-off')"),
        (lambda b: b["exception"][0].pop("dies"), "dies 2027-03-26 was removed"),
        (lambda b: b["exception"][0].update(category="hand-off"), "category changed ('validation' -> 'hand-off')"),
        (lambda b: b["exception"][0].update(wave="W4"), "wave changed (None -> 'W4')"),
        (lambda b: b["module"]["basis/a.py"].update(allow_imports=["spglib", "psutil"]), "gained psutil"),
        (lambda b: b["module"].update({"basis/b.py": {"sloc": 1, "category": "hand-off"}}), "row added"),
        (lambda b: b["exception"][0].update(bi=3), "waived bi rose from 2 to 3"),
        (lambda b: b["exception"][0].update(dies="2027-05-01"), "dies moved later"),
        (lambda b: b["exception"].append({"module": "basis/a.py", "function": "g", "bi": 1}), "exception added"),
        (lambda b: b["tier_total"].update(zero_sloc=12), "tier_total.zero_sloc rose"),
    ],
)
def test_a_looser_budget_than_head_fails(edit, expected):
    new = committed()
    edit(new)
    assert any(expected in line for line in GATE["monotone"](committed(), new))


def test_relabeling_a_dated_row_into_a_closed_category_cannot_escape_its_date():
    new = committed()
    row = new["module"]["basis/a.py"]
    for field in ("wave", "dies", "allow_imports"):
        row.pop(field, None)
    row.update(np=0, category="validation")
    found = GATE["monotone"](committed(), new)
    assert any("dies 2027-03-26 was removed" in line for line in found)
    assert any("wave changed" in line for line in found) and any("category changed" in line for line in found)


def test_the_budget_is_no_looser_than_the_committed_one():
    old = GATE["committed_budget"]()
    if old is None:
        pytest.skip("no committed budget yet (git or the file is unavailable)")
    assert GATE["monotone"](old, GATE["load_budget"]()) == []


# ---------------------------------------------------------------------------
# The ratchet
# ---------------------------------------------------------------------------
def test_the_ratchet_lowers_and_never_raises(tmp_path):
    files = {
        "basis/a.py": NUMPY_TWICE,
        "basis/b.py": "x = 1\n",
        "core/m.py": "import numpy as np\nnp.sum(1)\nnp.max(1)\nnp.min(1)\n",
        "core/n.py": "x = 1\n",
    }
    measured = GATE["measure_package"](package(tmp_path, files))
    before = budget(
        modules={
            "basis/a.py": {"np": 5, "sloc": 9, "wave": "W4", "dies": "2027-07-16"},
            "basis/b.py": {"sloc": 7, "category": "hand-off"},
            "basis/gone.py": {"sloc": 1, "category": "hand-off"},
            "core/m.py": {"np": 1},
            "core/n.py": {"np": 4},
        },
        exceptions=[{"module": "basis/a.py", "function": "vanished", "bi": 1,
                     "category": "validation", "reason": "x"}],
        zero_sloc=99,
    )
    after, notes = GATE["ratchet"](measured, before)
    assert after["module"]["basis/a.py"]["np"] == 2 and after["module"]["basis/a.py"]["sloc"] == 3
    assert after["module"]["basis/b.py"]["sloc"] == 1
    assert "basis/gone.py" not in after["module"]  # the module is gone
    assert after["module"]["core/m.py"] == {"np": 1}  # a rise is never written
    assert "core/n.py" not in after["module"]  # measures zero everywhere
    assert after["exception"] == [] and after["tier_total"]["zero_sloc"] == 4
    assert any("cannot be raised" in line for line in notes)
    assert GATE["monotone"](before, after) == []


def test_the_budget_file_round_trips_and_the_ratchet_result_passes(tmp_path):
    files = {"basis/a.py": NUMPY_TWICE, "core/m.py": "import numpy as np\nnp.sum(1)\n"}
    measured = GATE["measure_package"](package(tmp_path, files))
    before = budget(
        modules={"basis/a.py": {"np": 9, "sloc": 9, "wave": "W4", "dies": "2027-07-16",
                                "allow_imports": []},
                 "core/m.py": {"np": 1}},
        zero_sloc=9,
    )
    before["module"]["basis/a.py"].pop("allow_imports")
    target = tmp_path / "budget.toml"
    after, _ = GATE["ratchet"](measured, before)
    target.write_text(GATE["dump_budget"](after))
    reloaded = GATE["load_budget"](target)
    assert reloaded == after
    assert GATE["check_budget"](measured, reloaded) == []


def test_the_command_line_fails_on_a_regression_and_passes_after_it_is_removed(tmp_path):
    pkg = package(tmp_path, {"pseudopotentials/q.py": "x = 1\n"})
    target = tmp_path / "budget.toml"
    target.write_text(GATE["dump_budget"](budget(
        modules={"pseudopotentials/q.py": {"sloc": 1, "category": "hand-off"}}, zero_sloc=1)))
    command = [sys.executable, str(ROOT / "scripts/numerics_budget.py"),
               "--package", str(pkg), "--budget", str(target)]
    assert subprocess.run(command, capture_output=True, text=True).returncode == 0
    module = pkg / "pseudopotentials/q.py"
    module.write_text("import numpy as np\nx = np.sqrt(2.0)\n")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1 and "np 1 > budget 0" in result.stdout
    module.write_text("x = 1\n")
    assert subprocess.run(command, capture_output=True, text=True).returncode == 0


# ---------------------------------------------------------------------------
# The repository
# ---------------------------------------------------------------------------
def test_the_repository_satisfies_its_numerics_budget():
    measured = GATE["measure_package"]()
    assert GATE["check_budget"](measured, GATE["load_budget"]()) == []


def test_every_zero_tier_module_with_code_has_a_row():
    measured = GATE["measure_package"]()
    rows = GATE["load_budget"]()["module"]
    missing = [m for m, x in measured.items()
               if GATE["tier_of"](m) == "zero" and x["counts"]["sloc"] and m not in rows]
    assert missing == []


def test_the_undated_rows_are_clean_after_their_waivers():
    budget_file = GATE["load_budget"]()
    for relative, row in budget_file["module"].items():
        if GATE["tier_of"](relative) == "zero" and row.get("category"):
            assert not any(row.get(k) for k in GATE["CLEAN_COUNTERS"]), relative
            assert "dies" not in row and "wave" not in row, relative


def test_the_stale_inventories_are_retired():
    assert not (ROOT / "test/migration/audit_native.py").exists()
    assert not (ROOT / "test/migration/native_cutover/audit.json").exists()
    page = (ROOT / "docs/source/development/native_cutover.md").read_text()
    assert "audit_native" not in page and "audit.json" not in page
    assert "scripts/numerics_budget.py" in page
