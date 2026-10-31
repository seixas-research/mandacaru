"""The numerics gate: how much numerical work is left in Python.

Mandacaru's numerics live in Rust. This script counts, per module of
``src/mandacaru``, the Python that still does numerical work, and compares the
count with the allowlist ``test/architecture/python_numerics_budget.toml``.
Only the AST is read, so docstrings, comments and strings score zero, and an
alias, a from-import, a function-local import or a wildcard cannot hide a call.

Hard counters (exact; a rise is a regression, a drop is stale and must be
recorded with ``--ratchet``):

``np``      NumPy attributes and from-imports outside the plumbing set. The
            WHOLE dotted path must be plumbing (``np.ndarray.sum`` is not;
            only ``np.typing.*`` is a plumbing namespace); ``np.lib``,
            ``np.ma``, ``np.testing``, ``np.exceptions`` and ``np.random``
            are not plumbing; a wildcard import counts once. A numpy or
            math name imported from another ``mandacaru`` module (a
            re-export, followed through ``from .x import np``, ``from
            mandacaru.x import np``, wildcards and chains) is numpy here too
``scipy``   any scipy import, at any depth
``matmul``  the ``@`` operator
``meth``    numeric ndarray methods (``.sum``, ``.dot``, ``.cumsum`` ...)
``math``    ``math``/``cmath`` calls, bare names from ``from math import ...``
            included; constants and ``isfinite`` are plumbing
``ctypes``  ctypes, cffi, numba and jax imports
``dyn``     eval/exec/compile/``__import__``, ``importlib.import_module`` (a
            relative or ``mandacaru`` name is a package import, not evasion),
            ``globals()``/``vars()``/``sys.modules`` indexing, a bare
            reference to the numpy or math module (``getattr(np, ...)``) and
            a module attribute named like a numpy or math alias of the
            package (``other.np.sqrt``). Never waivable in the zero tier
``dloops``  loops over data whose body holds a counted operation
``dcomp``   comprehensions whose body holds a counted operation
``bi``      zero tier only: builtin ``sum abs min max sorted pow round`` on a
            non-literal argument
``imp``     zero tier only: an import outside the allowlist

Soft counters (zero tier, exact in the same way): ``aloops``/``acomp`` (loops
and comprehensions that only do operator arithmetic), ``arith`` (operator
arithmetic outside strings and unit conversions), ``arithmax`` (the largest
``arith`` of one function) and ``sloc`` (code lines, docstrings excluded).

An unlisted module of the zero tier is budgeted at zero on every counter,
``sloc`` included, so a new file needs a row. A tier A or Q module is
ratchet-only: growth stops, no date. No pragma comment is honored; an
exemption is a row or an ``[[exception]]`` in the budget file and nothing else.

Usage::

    python scripts/numerics_budget.py                 # check (exit 1 on failure)
    python scripts/numerics_budget.py --ratchet       # lower budgets, never raise
    python scripts/numerics_budget.py --show [MODULE ...]
    python scripts/numerics_budget.py --dates [--today YYYY-MM-DD]
    python scripts/numerics_budget.py --monotone      # against git HEAD
"""

from __future__ import annotations

import argparse
import ast
import collections
import datetime
import json
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src/mandacaru"
BUDGET = ROOT / "test/architecture/python_numerics_budget.toml"
BUDGET_RELATIVE = "test/architecture/python_numerics_budget.toml"

HARD_COMMON = ("np", "scipy", "matmul", "meth", "math", "ctypes", "dyn", "dloops", "dcomp")
HARD_ZERO = HARD_COMMON + ("bi", "imp")
SOFT_ZERO = ("aloops", "acomp", "arith", "arithmax", "sloc")
TIER_KEYS = {
    "zero": HARD_ZERO + SOFT_ZERO,
    "a": HARD_COMMON,
    "q": HARD_COMMON,
}
# An undated row of the zero tier must stay in one of these closed categories
# and must show none of the hard counters (after its [[exception]] waivers).
CLOSED_CATEGORIES = ("validation", "hand-off", "unit", "result-view", "log-table")
CLEAN_COUNTERS = HARD_ZERO
ROW_FIELDS = ("category", "wave", "dies", "allow_imports")
# An exception waives the additive counters of one function; the maximum and
# the line count of a module are not additive.
WAIVABLE = HARD_ZERO + ("aloops", "acomp", "arith")
# A closed-category exception (validation, hand-off, unit, result-view,
# log-table) is the scalar helper of an option check or a printed table, so it
# may waive only these. np, scipy, matmul, meth, math, ctypes, dyn and imp need
# a dated exception.
CLOSED_WAIVABLE = ("bi", "dloops", "dcomp", "aloops", "acomp", "arith")
# In the zero tier no exception, dated or not, waives these: an evasion route
# (dyn) or an import outside the allowlist (a row's allow_imports names it).
NEVER_WAIVABLE_ZERO = ("dyn", "imp")
WARN_DAYS = 30

PLUMBING = frozenset(
    {
        # construction and hand-off
        "asarray", "array", "ascontiguousarray", "asfortranarray", "zeros", "empty",
        "ones", "full", "zeros_like", "empty_like", "ones_like", "full_like",
        "ndarray", "frombuffer", "copy", "atleast_1d", "atleast_2d", "atleast_3d",
        "reshape", "ravel", "squeeze", "expand_dims", "transpose", "swapaxes",
        "moveaxis", "broadcast_to", "concatenate", "stack", "vstack", "hstack",
        "result_type", "promote_types", "can_cast",
        # types and constants
        "dtype", "float64", "float32", "complex64", "complex128", "int8", "int16",
        "int32", "int64", "uint8", "uint32", "uint64", "intp", "bool_", "bool",
        "str_", "object_", "floating", "integer", "number", "generic",
        "complexfloating", "inexact", "signedinteger", "newaxis", "pi", "inf",
        "nan", "e", "issubdtype", "isscalar", "ndim", "shape", "size", "iinfo",
        "finfo", "typing",
        # validation, no arithmetic on data
        "isfinite", "isnan", "isinf", "array_equal", "allclose", "isclose", "all",
        "any", "errstate", "printoptions", "set_printoptions", "array2string",
        "array_repr",
        # file hand-off
        "save", "load", "savez", "savez_compressed", "loadtxt", "savetxt",
        "iscomplexobj", "isrealobj", "iscomplex", "isreal", "ComplexWarning",
        "void", "datetime64", "timedelta64",
    }
)
# A plumbing namespace: everything below it is annotations (``np.typing.NDArray``).
PLUMBING_NAMESPACES = frozenset({"typing"})
MATH_PLUMBING = frozenset({"pi", "e", "inf", "nan", "tau", "isfinite"})
NUMERIC_METHODS = frozenset(
    {
        "sum", "dot", "cumsum", "prod", "mean", "std", "var", "max", "min",
        "argmax", "argmin", "argsort", "sort", "trace", "conj", "clip", "round",
        "searchsorted", "nonzero", "outer", "inner", "cross",
    }
)
BUILTIN_NUMERICS = frozenset({"sum", "abs", "min", "max", "sorted", "pow", "round"})
DYNAMIC_BUILTINS = frozenset({"eval", "exec", "compile", "__import__"})
NATIVE_FFI = frozenset({"ctypes", "cffi", "numba", "jax"})
ARITHMETIC = (ast.Mult, ast.Div, ast.Pow, ast.FloorDiv, ast.Mod)
# ase submodules that are data tables or API, not numerics. ase.geometry,
# ase.neighborlist and ase.spacegroup are numerics: they are not here.
ASE_ALLOWED = ("ase.atoms", "ase.io", "ase.units", "ase.data", "ase.calculators", "ase.spectrum")
ASE_NAMES = frozenset({"Atoms", "io", "units", "data", "calculators", "spectrum"})
STDLIB = frozenset(sys.stdlib_module_names)


# --------------------------------------------------------------------------
# The counter
# --------------------------------------------------------------------------
def dotted(node):
    """``a.b.c`` for a pure Name/Attribute chain, else ``None``."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def is_plumbing_path(path):
    """Whether a numpy-relative dotted path is plumbing: ALL of it, not its head.

    ``float64`` is, ``float64.dot`` and ``ndarray.sum`` are not; only a
    namespace of annotations (``typing.NDArray``) may carry a tail.
    """
    head, _, rest = path.partition(".")
    return head in PLUMBING and (not rest or head in PLUMBING_NAMESPACES)


def is_string(node):
    return isinstance(node, ast.JoinedStr) or (
        isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes))
    )


def is_literal(node):
    """A constant, or a tuple/list/set/negation made only of constants."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(is_literal(item) for item in node.elts)
    if isinstance(node, ast.UnaryOp):
        return is_literal(node.operand)
    return False


def unit_names(package):
    """Public names of ``units.py``: converting by them is not numerics."""
    path = Path(package) / "units.py"
    names = set()
    if path.is_file():
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign):
                names.update(
                    t.id for t in node.targets
                    if isinstance(t, ast.Name) and not t.id.startswith("_")
                )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_"):
                    names.add(node.name)
    return frozenset(names)


def code_lines(tree):
    """Lines that carry code: statement lines minus docstrings."""
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0]))
    lines = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt) or id(node) in docstrings:
            continue
        start = node.lineno
        decorators = getattr(node, "decorator_list", None)
        if decorators:
            start = min([start] + [d.lineno for d in decorators])
        body = getattr(node, "body", None)
        if isinstance(body, list) and body:
            end = max(body[0].lineno - 1, start)
        else:
            end = node.end_lineno
        lines.update(range(start, end + 1))
    return len(lines)


def toplevel_nodes(tree):
    """The module's own statements, descending into ``if``/``try``/``with`` but not into scopes."""
    nodes, pending = [], list(tree.body)
    while pending:
        node = pending.pop(0)
        nodes.append(node)
        if isinstance(node, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
            pending.extend(node.body + getattr(node, "orelse", []) + getattr(node, "finalbody", []))
            for handler in getattr(node, "handlers", []):
                pending.extend(handler.body)
    return nodes


def imported_module_key(here, node):
    """``'basis/x'`` for a from-import of package code (relative or ``mandacaru.``), else ``None``.

    ``here`` is the importing module's package path (``('basis',)`` for
    ``basis/x.py`` and for ``basis/__init__.py``). A package ``__init__`` is
    keyed by its package (``'basis'``).
    """
    if node.level:
        base = list(here)
        if node.level - 1 > len(base):
            return None
        base = base[: len(base) - (node.level - 1)]
        parts = base + (node.module.split(".") if node.module else [])
    else:
        module = (node.module or "").split(".")
        if module[0] != "mandacaru":
            return None
        parts = module[1:]
    return "/".join(parts)


def module_key(relative):
    """The key a module is imported by: ``basis/x.py`` -> ``basis/x``, ``basis/__init__.py`` -> ``basis``."""
    stem = relative[: -len(".py")]
    return stem[: -len("/__init__")] if stem.endswith("/__init__") else ("" if stem == "__init__" else stem)


class Aliases:
    """What the numpy, math and importlib names of a set of nodes stand for.

    ``exports`` maps a module key to the alias table of that module's own
    top-level statements, so ``from .x import np`` binds ``np`` to numpy when
    ``x`` binds it. Without it a re-export hides every use of ``np``.
    """

    def __init__(self, nodes, here=(), exports=None):
        self.numpy = {}  # local name -> numpy-relative path ("" is the module)
        self.math_modules = set()
        self.math_names = {}  # local name -> function name
        self.importlib = set()
        self.import_module = set()
        self.alias_values = set()  # ids of Names that only bind an alias
        nodes = list(nodes)
        for node in nodes:
            self._imports(node, here, exports or {})
        for _ in range(3):  # `xp = np`, `lin = np.linalg`
            for node in nodes:
                self._assignments(node)

    def table(self):
        return {"numpy": dict(self.numpy), "math_modules": set(self.math_modules),
                "math_names": dict(self.math_names)}

    def _imports(self, node, here, exports):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root == "numpy":
                    if alias.asname:
                        self.numpy[alias.asname] = alias.name[len("numpy."):] if "." in alias.name else ""
                    else:
                        self.numpy["numpy"] = ""
                elif root in {"math", "cmath"}:
                    self.math_modules.add(alias.asname or root)
                elif alias.name == "importlib":
                    self.importlib.add(alias.asname or "importlib")
        elif isinstance(node, ast.ImportFrom):
            key = imported_module_key(here, node)
            if key is not None:
                self._reexports(node, exports.get(key))
                return
            module = node.module or ""
            root = module.split(".")[0]
            for alias in node.names:
                bound = alias.asname or alias.name
                if alias.name == "*":
                    continue
                if root == "numpy":
                    tail = module[len("numpy."):] if "." in module else ""
                    self.numpy[bound] = f"{tail}.{alias.name}" if tail else alias.name
                elif root in {"math", "cmath"}:
                    self.math_names[bound] = alias.name
                elif module == "importlib" and alias.name == "import_module":
                    self.import_module.add(bound)
                elif module == "importlib" and alias.name == "util":
                    self.importlib.add(bound)

    def _reexports(self, node, table):
        if not table:
            return
        for alias in node.names:
            names = list(table["numpy"]) + list(table["math_modules"]) + list(table["math_names"]) \
                if alias.name == "*" else [alias.name]
            for name in names:
                bound = name if alias.name == "*" else (alias.asname or name)
                if name in table["numpy"]:
                    self.numpy[bound] = table["numpy"][name]
                elif name in table["math_modules"]:
                    self.math_modules.add(bound)
                elif name in table["math_names"]:
                    self.math_names[bound] = table["math_names"][name]

    def _assignments(self, node):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            path = self.numpy_path(value) if value is not None else None
            mathy = isinstance(value, ast.Name) and value.id in self.math_modules
            for target in targets:
                if isinstance(target, ast.Name):
                    if path is not None:
                        self.numpy.setdefault(target.id, path)
                        self._mark_alias_value(value)
                    elif mathy:
                        self.math_modules.add(target.id)
                        self._mark_alias_value(value)

    def _mark_alias_value(self, node):
        for inner in ast.walk(node):
            if isinstance(inner, ast.Name):
                self.alias_values.add(id(inner))

    def numpy_path(self, node):
        """The numpy-relative path an expression resolves to, or ``None``."""
        name = dotted(node)
        if name is None:
            return None
        head, _, rest = name.partition(".")
        if head not in self.numpy:
            return None
        base = self.numpy[head]
        return ".".join(part for part in (base, rest) if part)


class Scan(ast.NodeVisitor):
    """One module: counts, per-function counts and the import targets."""

    def __init__(self, tree, units=frozenset(), exports=None, here=(), alias_names=frozenset()):
        self.units = units
        self.alias_names = alias_names  # names that stand for numpy or math somewhere in the package
        self.counts = collections.Counter()
        self.functions = collections.defaultdict(collections.Counter)
        self.imports = []  # disallowed targets, with line numbers
        self.stack = []
        self.events = set()  # ids of nodes that are counted data operations
        aliases = Aliases(ast.walk(tree), here, exports)
        self.aliases = aliases
        self.alias_values = aliases.alias_values
        self.numpy = aliases.numpy
        self.math_modules = aliases.math_modules
        self.math_names = aliases.math_names
        self.importlib = aliases.importlib
        self.import_module = aliases.import_module
        self.dynamic_imports = False
        self.named_imports = set()  # name arguments of the dynamic imports
        self.visit(tree)
        if self.dynamic_imports:
            # a string that names numpy/scipy/math may be fed to import_module
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and id(node) not in self.named_imports
                ):
                    root = node.value.split(".")[0]
                    if root in {"numpy", "scipy", "math", "cmath"}:
                        self.bump("dyn", node, event=False)
        self.counts["sloc"] = code_lines(tree)
        self.counts["arithmax"] = max(
            (c["arith"] for c in self.functions.values()), default=0
        )

    # -- bookkeeping --------------------------------------------------------
    def bump(self, key, node=None, event=True):
        self.counts[key] += 1
        self.functions[".".join(self.stack) or "<module>"][key] += 1
        if event and node is not None and key in {"np", "math", "meth", "matmul", "bi"}:
            self.events.add(id(node))

    # -- scopes -------------------------------------------------------------
    def _scope(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _scope

    # -- imports ------------------------------------------------------------
    def visit_Import(self, node):
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root == "scipy":
                self.bump("scipy", node)
            elif root in NATIVE_FFI:
                self.bump("ctypes", node)
            else:
                self._import_target(alias.name, node)

    def visit_ImportFrom(self, node):
        if node.level:
            return
        module = node.module or ""
        root = module.split(".")[0]
        names = [alias.name for alias in node.names]
        if root == "numpy":
            for name in names:
                path = name if name == "*" else (
                    f"{module[len('numpy.'):]}.{name}" if "." in module else name
                )
                if name == "*" or not is_plumbing_path(path):
                    self.bump("np", node)
        elif root == "scipy":
            self.bump("scipy", node)
        elif root in NATIVE_FFI:
            self.bump("ctypes", node)
        elif root in {"math", "cmath"}:
            for name in names:
                if name == "*" or name not in MATH_PLUMBING:
                    self.bump("math", node)
        elif root == "ase":
            if module == "ase":
                for name in names:
                    if name not in ASE_NAMES:
                        self._import_target(f"ase.{name}", node)
            else:
                self._import_target(module, node)
        else:
            self._import_target(module, node)

    def _import_target(self, target, node):
        root = target.split(".")[0]
        if root in {"mandacaru", "_native", "numpy", "math", "cmath"} or root in STDLIB:
            return
        if root == "ase" and any(
            target == allowed or target.startswith(allowed + ".") for allowed in ASE_ALLOWED
        ):
            return
        if root == "ase" and target == "ase":
            return
        self.counts["imp"] += 1
        self.functions[".".join(self.stack) or "<module>"]["imp"] += 1
        self.imports.append((target, node.lineno))

    # -- names, attributes, calls -------------------------------------------
    def visit_Attribute(self, node):
        name = dotted(node)
        if name is not None:
            head, _, rest = name.partition(".")
            if head in self.numpy and isinstance(node.ctx, ast.Load):
                if not is_plumbing_path(self.aliases.numpy_path(node)):
                    self.bump("np", node)
                return
            if head in self.math_modules and rest:
                if rest not in MATH_PLUMBING:
                    self.bump("math", node)
                return
            if self.alias_names.intersection(rest.split(".")):
                # `other.np.sqrt(...)`: numpy handed through a module attribute
                self.bump("dyn", node)
                return
        self.generic_visit(node)

    def visit_Name(self, node):
        if not isinstance(node.ctx, ast.Load):
            return
        name = node.id
        if name in self.numpy:
            if self.numpy[name] == "":
                if id(node) not in self.alias_values:
                    self.bump("dyn", node)  # the module itself, passed around
            elif not is_plumbing_path(self.numpy[name]) and id(node) not in self.alias_values:
                self.bump("np", node)
        elif name in self.math_modules:
            if id(node) not in self.alias_values:
                self.bump("dyn", node)
        elif name in self.math_names:
            if self.math_names[name] not in MATH_PLUMBING:
                self.bump("math", node)

    def visit_Call(self, node):
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
            if name in BUILTIN_NUMERICS and not all(
                is_literal(arg) for arg in node.args
            ):
                self.bump("bi", node)
            elif name in DYNAMIC_BUILTINS:
                self.bump("dyn", node)
                if name == "__import__":
                    self.dynamic_imports = True
                    self.named_imports.update(id(arg) for arg in node.args[:1])
            elif name in self.import_module:
                self._dynamic_import(node)
        elif isinstance(func, ast.Attribute):
            owner = dotted(func.value)
            if func.attr == "import_module" and owner in self.importlib:
                self._dynamic_import(node)
            elif func.attr in NUMERIC_METHODS:
                head = (dotted(func) or "").split(".")[0]
                if head not in self.numpy and head not in self.math_modules:
                    self.bump("meth", node)
        self.generic_visit(node)

    def _dynamic_import(self, node):
        """``import_module``: a package-relative literal is not evasion."""
        self.dynamic_imports = True
        first = node.args[0] if node.args else None
        self.named_imports.update(id(arg) for arg in node.args[:1])
        package = node.args[1] if len(node.args) > 1 else None
        for keyword in node.keywords:
            if keyword.arg == "package":
                package = keyword.value
        relative = False
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            relative = first.value.startswith((".", "mandacaru"))
        elif first is not None and isinstance(package, ast.Name) and package.id in {"__name__", "__package__"}:
            relative = not (isinstance(first, ast.Constant))
        if not relative:
            self.bump("dyn", node)

    def visit_Subscript(self, node):
        if isinstance(node.ctx, ast.Load):
            value = node.value
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id in {"globals", "vars", "locals"}
            ) or dotted(value) == "sys.modules":
                self.bump("dyn", node)
        self.generic_visit(node)

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.MatMult):
            self.bump("matmul", node)
        elif self._is_arithmetic(node):
            self.bump("arith", event=False)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        if isinstance(node.op, ast.MatMult):
            self.bump("matmul", node)
        elif isinstance(node.op, ARITHMETIC) and not (
            is_string(node.value) or self._unit(node.value)
        ):
            self.bump("arith", event=False)
        self.generic_visit(node)

    def _unit(self, node):
        if isinstance(node, ast.Name):
            return node.id in self.units
        if isinstance(node, ast.Attribute):
            return node.attr in self.units
        if isinstance(node, ast.Call):
            return self._unit(node.func)
        return False

    def _is_arithmetic(self, node):
        if not isinstance(node.op, ARITHMETIC):
            return False
        left, right = node.left, node.right
        if is_string(left) or is_string(right):
            return False
        if isinstance(left, ast.Constant) and isinstance(right, ast.Constant):
            return False
        return not (self._unit(left) or self._unit(right))

    # -- loops and comprehensions -------------------------------------------
    def _loop(self, node, loop_key, comp_key):
        self.generic_visit(node)
        data = False
        arithmetic = False
        for inner in ast.walk(node):
            if id(inner) in self.events:
                data = True
                break
            if isinstance(inner, ast.BinOp) and self._is_arithmetic(inner):
                arithmetic = True
            elif (
                isinstance(inner, ast.AugAssign)
                and isinstance(inner.op, ARITHMETIC)
                and not (is_string(inner.value) or self._unit(inner.value))
            ):
                arithmetic = True
        key = loop_key if comp_key is None else comp_key
        if data:
            self.bump("d" + key, event=False)
        elif arithmetic:
            self.bump("a" + key, event=False)

    def visit_For(self, node):
        self._loop(node, "loops", None)

    visit_AsyncFor = visit_While = visit_For

    def _comprehension(self, node):
        self._loop(node, None, "comp")

    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _comprehension


def measure_source(source, units=frozenset(), exports=None, here=(), alias_names=frozenset()):
    """Counters of one module's source: ``counts``, ``functions``, ``imports``.

    ``exports``, ``here`` and ``alias_names`` come from :func:`package_aliases`;
    without them a name imported from another module of the package is unknown.
    """
    scan = Scan(ast.parse(source), units, exports, here, alias_names)
    counts = {key: scan.counts[key] for key in HARD_ZERO + SOFT_ZERO}
    return {
        "counts": counts,
        "functions": {name: dict(c) for name, c in scan.functions.items()},
        "imports": list(scan.imports),
    }


def package_modules(package):
    package = Path(package)
    return sorted(
        p for p in package.rglob("*.py") if "__pycache__" not in p.parts
    )


def package_aliases(trees):
    """``(exports, alias_names)`` of a package, from ``{relative path: tree}``.

    ``exports[key]`` is the alias table of the top-level statements of one
    module, followed through re-exports (``from .x import np``, wildcards and
    chains) until nothing changes. ``alias_names`` is every top-level name that
    stands for numpy or math anywhere in the package.
    """
    exports, here = {}, {}
    for relative in trees:
        here[relative] = tuple(relative.split("/")[:-1])
    for _ in range(len(trees) + 1):
        changed = False
        for relative, tree in trees.items():
            table = Aliases(toplevel_nodes(tree), here[relative], exports).table()
            key = module_key(relative)
            if exports.get(key) != table:
                exports[key] = table
                changed = True
        if not changed:
            break
    alias_names = set()
    for table in exports.values():
        alias_names.update(name for name, path in table["numpy"].items() if path == "")
        alias_names.update(table["math_modules"])
    return exports, frozenset(alias_names)


def measure_package(package=PACKAGE):
    """``{relative path: measurement}`` for every module of the package."""
    package = Path(package)
    units = unit_names(package)
    sources = {
        path.relative_to(package).as_posix(): path.read_text(encoding="utf-8")
        for path in package_modules(package)
    }
    exports, alias_names = package_aliases({r: ast.parse(t) for r, t in sources.items()})
    return {
        relative: measure_source(
            text, units, exports, tuple(relative.split("/")[:-1]), alias_names
        )
        for relative, text in sources.items()
    }


def tier_of(relative):
    """``zero`` (nothing numerical may stay), ``q`` (quantum side) or ``a``."""
    name = relative.rsplit("/", 1)[-1]
    if relative.startswith(("pseudopotentials/", "basis/", "integrals/", "models/")):
        return "zero"
    if relative in {"core/symmetry.py", "core/spin_orbit.py"}:
        return "zero"
    if relative.startswith("integrations/") and relative.count("/") == 1 and name.startswith("paw"):
        return "zero"
    if relative.startswith(("backends/", "circuits/", "core/", "optimizers/", "workflows/")):
        return "q"
    return "a"


# --------------------------------------------------------------------------
# The budget file
# --------------------------------------------------------------------------
def load_budget(path=BUDGET):
    """Parsed budget: ``module`` rows, ``exception`` list, ``tier_total``."""
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "module": data.get("module", {}),
        "exception": data.get("exception", []),
        "tier_total": data.get("tier_total", {}),
    }


def _toml_value(value):
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, str):
        return json.dumps(value)
    return str(value)


HEADER = """\
# Python numerics budget, read by scripts/numerics_budget.py and by
# test/architecture/test_python_numerics.py. Hard and soft counters are exact:
# a budget above the measurement is stale, below it a regression. After a port
# lower the row (python scripts/numerics_budget.py --ratchet); only the owner
# raises a number or moves a `dies` later, by committing it.
#
# Zero tier (pseudopotentials/ basis/ integrals/ models/ integrations/paw*
# core/symmetry.py core/spin_orbit.py): a module without a row is budgeted at
# zero on every counter, sloc included. A row is dated (wave, dies) while it has
# any hard counter or policy to port; an undated row names a closed category
# (validation, hand-off, unit, result-view, log-table) and shows no hard counter.
# Tiers A and Q (everything else) are ratchet-only: growth stops, no date.
# Counters: np scipy matmul meth math ctypes dyn dloops dcomp bi imp (hard);
# aloops acomp arith arithmax sloc (soft, zero tier). Omitted means 0.
# [[exception]] entries waive the counters of one function; their counts must
# match the measurement exactly and are subtracted from the module row.
"""


def dump_budget(budget):
    """Deterministic TOML text of a budget (tomllib reads, this writes)."""
    lines = [HEADER]
    total = budget.get("tier_total", {})
    lines.append("[tier_total]")
    for key in sorted(total):
        lines.append(f"{key} = {total[key]}")
    lines.append("")
    lines.append("[module]")
    order = HARD_ZERO + SOFT_ZERO + ROW_FIELDS
    for relative in sorted(budget["module"]):
        row = budget["module"][relative]
        fields = [f"{k} = {_toml_value(row[k])}" for k in order if k in row]
        lines.append(f'"{relative}" = {{ ' + ", ".join(fields) + " }" if fields else f'"{relative}" = {{}}')
    for item in budget["exception"]:
        lines.append("")
        lines.append("[[exception]]")
        keys = ["module", "function"] + list(WAIVABLE) + ["category", "reason", "wave", "dies"]
        for key in keys:
            if key in item:
                lines.append(f"{key} = {_toml_value(item[key])}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
def waivers(budget):
    """``{module: {function: exception}}`` of the exception entries."""
    table = collections.defaultdict(dict)
    for item in budget["exception"]:
        table[item.get("module", "")][item.get("function", "")] = item
    return table


def waiver_refusal(relative, item, key):
    """Why an exception may not waive ``key``, or ``None`` when it may."""
    if tier_of(relative) == "zero" and key in NEVER_WAIVABLE_ZERO:
        return f"{key} is never waivable in the zero tier"
    if item.get("category") in CLOSED_CATEGORIES and key not in CLOSED_WAIVABLE:
        return f"a {item['category']} exception may waive only {list(CLOSED_WAIVABLE)}; {key} needs a dated exception"
    return None


def effective(relative, measurement, row, waived):
    """Counters of a module after the waivers and the row's allowed imports.

    A waiver the rules refuse (see :func:`waiver_refusal`) subtracts nothing.
    """
    counts = dict(measurement["counts"])
    for function, item in waived.get(relative, {}).items():
        have = measurement["functions"].get(function, {})
        for key in WAIVABLE:
            if key in item and have.get(key, 0) == item[key] and waiver_refusal(relative, item, key) is None:
                counts[key] -= item[key]
    allowed = set(row.get("allow_imports", ()))
    counts["imp"] -= sum(1 for target, _ in measurement["imports"] if target in allowed)
    return counts


def check_budget(measured, budget):
    """Failures of the measured package against the budget (empty means pass)."""
    failures = []
    rows = budget["module"]
    waived = waivers(budget)
    for relative in sorted(rows):
        if relative not in measured:
            failures.append(f"{relative}: budget row names a module that does not exist (delete the row)")
    for relative, measurement in sorted(measured.items()):
        tier = tier_of(relative)
        row = rows.get(relative, {})
        keys = TIER_KEYS[tier]
        for key in row:
            if key not in keys and key not in ROW_FIELDS:
                failures.append(f"{relative}: counter {key} is not tracked in tier {tier}")
        counts = effective(relative, measurement, row, waived)
        for key in keys:
            budgeted = row.get(key, 0)
            if counts[key] > budgeted:
                hint = " (unlisted zero-tier module: add a row)" if tier == "zero" and not row else ""
                failures.append(f"{relative}: {key} {counts[key]} > budget {budgeted}: regression{hint}")
            elif counts[key] < budgeted:
                failures.append(f"{relative}: {key} {counts[key]} < budget {budgeted}: stale, lower the budget (--ratchet)")
        used = {target for target, _ in measurement["imports"]}
        for target in row.get("allow_imports", ()):
            if target not in used:
                failures.append(f"{relative}: allow_imports names {target}, which the module no longer imports")
    failures.extend(check_exceptions(measured, budget))
    total = sum(m["counts"]["sloc"] for r, m in measured.items() if tier_of(r) == "zero")
    for key, value in budget["tier_total"].items():
        if key == "zero_sloc" and total != value:
            relation = "regression" if total > value else "stale, lower the budget (--ratchet)"
            failures.append(f"zero tier sloc total {total} != budget {value}: {relation}")
    if "zero_sloc" not in budget["tier_total"]:
        failures.append("tier_total.zero_sloc is missing")
    failures.extend(check_integrity(measured, budget))
    return failures


def check_exceptions(measured, budget):
    failures = []
    for item in budget["exception"]:
        where = f"{item.get('module', '?')}::{item.get('function', '?')}"
        module, function = item.get("module"), item.get("function")
        if module not in measured:
            failures.append(f"{where}: exception names a module that does not exist")
            continue
        have = measured[module]["functions"].get(function)
        if have is None:
            failures.append(f"{where}: exception names a function that does not exist")
            continue
        declared = {k: v for k, v in item.items() if k in WAIVABLE}
        if not declared:
            failures.append(f"{where}: exception waives no counter")
        for key, value in declared.items():
            if have.get(key, 0) != value:
                failures.append(f"{where}: exception waives {key} = {value} but the function has {have.get(key, 0)}")
            refusal = waiver_refusal(module, item, key)
            if refusal:
                failures.append(f"{where}: exception waives {key}: {refusal}")
        for field in ("category", "reason"):
            if not item.get(field):
                failures.append(f"{where}: exception needs a {field}")
        if not item.get("dies") and item.get("category") not in CLOSED_CATEGORIES:
            failures.append(f"{where}: an exception outside the closed categories needs wave and dies")
        if item.get("dies") and parse_date(item["dies"]) is None:
            failures.append(f"{where}: dies is not an ISO date")
    return failures


def parse_date(text):
    try:
        return datetime.date.fromisoformat(str(text))
    except ValueError:
        return None


def check_integrity(measured, budget):
    """Row rules that do not depend on the counts alone."""
    failures = []
    for relative, row in sorted(budget["module"].items()):
        tier = tier_of(relative)
        category = row.get("category")
        if tier != "zero":
            for field in ("wave", "dies", "category", "allow_imports"):
                if field in row:
                    failures.append(f"{relative}: {field} belongs to zero-tier rows; tier {tier} is ratchet-only")
            continue
        if category is not None and category not in CLOSED_CATEGORIES:
            failures.append(f"{relative}: category {category} is not one of {list(CLOSED_CATEGORIES)}")
        if category in CLOSED_CATEGORIES:
            dirty = [k for k in CLEAN_COUNTERS if row.get(k, 0)]
            if dirty:
                failures.append(f"{relative}: a {category} row must show no {dirty}")
            if row.get("dies") or row.get("wave"):
                failures.append(f"{relative}: a {category} row is undated")
        else:
            if not row.get("wave") or not row.get("dies"):
                failures.append(f"{relative}: a row outside the closed categories needs wave and dies")
            elif parse_date(row["dies"]) is None:
                failures.append(f"{relative}: dies {row['dies']} is not an ISO date")
    return failures


def date_findings(budget, today=None):
    """``(warnings, failures)`` for rows and exceptions whose ``dies`` is near or past."""
    today = today or datetime.date.today()
    warnings, failures = [], []
    entries = [(r, row) for r, row in budget["module"].items()]
    entries += [(f"{i.get('module')}::{i.get('function')}", i) for i in budget["exception"]]
    for name, entry in sorted(entries):
        dies = parse_date(entry["dies"]) if entry.get("dies") else None
        if dies is None:
            continue
        days = (dies - today).days
        if days < 0:
            failures.append(f"{name}: dies {dies} is {-days} days past (wave {entry.get('wave')}): port it, or the owner moves the date by committing it")
        elif days <= WARN_DAYS:
            warnings.append(f"{name}: dies {dies} in {days} days (wave {entry.get('wave')})")
    return warnings, failures


def label_changes(where, before, now):
    """Failures when a dated entry loses, delays or relabels its date.

    A missing ``dies`` where the committed one had a date, a later date, and a
    changed ``wave`` or ``category`` all let an entry escape its death date, so
    each fails; the owner relabels by committing.
    """
    failures = []
    if before.get("dies"):
        if not now.get("dies"):
            failures.append(f"{where}: dies {before['dies']} was removed")
        else:
            was, is_now = parse_date(before["dies"]), parse_date(now["dies"])
            if is_now is None:
                failures.append(f"{where}: dies {now['dies']} is not an ISO date")
            elif was is not None and is_now > was:
                failures.append(f"{where}: dies moved later ({before['dies']} -> {now['dies']})")
    for field in ("wave", "category"):
        if before.get(field) != now.get(field):
            failures.append(f"{where}: {field} changed ({before.get(field)!r} -> {now.get(field)!r})")
    return failures


def monotone(old, new):
    """Failures when ``new`` is looser than ``old`` (the committed budget)."""
    failures = []
    keys = HARD_ZERO + SOFT_ZERO
    for relative, row in sorted(new["module"].items()):
        before = old["module"].get(relative)
        if before is None:
            failures.append(f"{relative}: row added since the committed budget (the owner adds rows by committing)")
            continue
        for key in keys:
            if row.get(key, 0) > before.get(key, 0):
                failures.append(f"{relative}: {key} rose from {before.get(key, 0)} to {row.get(key, 0)}")
        failures.extend(label_changes(relative, before, row))
        for target in set(row.get("allow_imports", ())) - set(before.get("allow_imports", ())):
            failures.append(f"{relative}: allow_imports gained {target}")
    old_items = {(i.get("module"), i.get("function")): i for i in old["exception"]}
    for item in new["exception"]:
        where = f"{item.get('module')}::{item.get('function')}"
        before = old_items.get((item.get("module"), item.get("function")))
        if before is None:
            failures.append(f"{where}: exception added since the committed budget")
            continue
        for key in keys:
            if item.get(key, 0) > before.get(key, 0):
                failures.append(f"{where}: waived {key} rose from {before.get(key, 0)} to {item.get(key, 0)}")
        failures.extend(label_changes(where, before, item))
    for key, value in new["tier_total"].items():
        if value > old["tier_total"].get(key, 0):
            failures.append(f"tier_total.{key} rose from {old['tier_total'].get(key, 0)} to {value}")
    return failures


def committed_budget(root=ROOT):
    """The budget at git HEAD, or ``None`` when git or the file is unavailable."""
    try:
        result = subprocess.run(
            ["git", "show", f"HEAD:{BUDGET_RELATIVE}"],
            cwd=root, capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    data = tomllib.loads(result.stdout)
    return {
        "module": data.get("module", {}),
        "exception": data.get("exception", []),
        "tier_total": data.get("tier_total", {}),
    }


def ratchet(measured, budget):
    """Lower every number the measurement undercuts. Never raises one.

    Returns ``(new budget, notes)``. A row of a missing module, a row that
    measures zero everywhere, an unused allowed import and an exception whose
    function is gone are removed.
    """
    waived = waivers(budget)
    notes = []
    rows = {}
    for relative, row in budget["module"].items():
        if relative not in measured:
            notes.append(f"{relative}: removed (module is gone)")
            continue
        keys = TIER_KEYS[tier_of(relative)]
        counts = effective(relative, measured[relative], row, waived)
        new = {k: v for k, v in row.items() if k in ROW_FIELDS}
        for key in keys:
            budgeted = row.get(key, 0)
            value = min(budgeted, counts[key])
            if value < budgeted:
                notes.append(f"{relative}: {key} {budgeted} -> {value}")
            if counts[key] > budgeted:
                notes.append(f"{relative}: {key} {counts[key]} > {budgeted} cannot be raised here")
            if value:
                new[key] = value
        used = {target for target, _ in measured[relative]["imports"]}
        kept = [t for t in row.get("allow_imports", ()) if t in used]
        if kept:
            new["allow_imports"] = kept
        else:
            new.pop("allow_imports", None)
        if not any(new.get(k, 0) for k in keys):
            notes.append(f"{relative}: removed (all counters are zero)")
            continue
        rows[relative] = new
    exceptions = []
    for item in budget["exception"]:
        have = measured.get(item.get("module"), {"functions": {}})["functions"].get(item.get("function"))
        if have is None:
            notes.append(f"{item.get('module')}::{item.get('function')}: removed (function is gone)")
            continue
        new = {k: v for k, v in item.items() if k not in WAIVABLE}
        for key in WAIVABLE:
            if key in item:
                value = min(item[key], have.get(key, 0))
                if value < item[key]:
                    notes.append(f"{item['module']}::{item['function']}: {key} {item[key]} -> {value}")
                if value:
                    new[key] = value
        if any(k in new for k in WAIVABLE):
            exceptions.append(new)
    total = sum(m["counts"]["sloc"] for r, m in measured.items() if tier_of(r) == "zero")
    tier_total = dict(budget["tier_total"])
    if total < tier_total.get("zero_sloc", total):
        notes.append(f"tier_total.zero_sloc {tier_total['zero_sloc']} -> {total}")
        tier_total["zero_sloc"] = total
    return {"module": rows, "exception": exceptions, "tier_total": tier_total}, notes


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def run_check(package, budget_path):
    measured = measure_package(package)
    budget = load_budget(budget_path)
    failures = check_budget(measured, budget)
    zero = [r for r in measured if tier_of(r) == "zero"]
    print(
        f"numerics budget: {len(measured)} modules, {len(budget['module'])} rows, "
        f"{len(zero)} in the zero tier "
        f"({sum(measured[r]['counts']['sloc'] for r in zero)} sloc)"
    )
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--package", type=Path, default=PACKAGE)
    parser.add_argument("--budget", type=Path, default=BUDGET)
    parser.add_argument("--ratchet", action="store_true", help="lower budgets, never raise")
    parser.add_argument("--show", nargs="*", metavar="MODULE", help="print counters (and per-function detail for named modules)")
    parser.add_argument("--dates", action="store_true", help="warn 30 days before a dies, fail past it")
    parser.add_argument("--today", type=datetime.date.fromisoformat, default=None)
    parser.add_argument("--monotone", action="store_true", help="compare with the budget at git HEAD")
    args = parser.parse_args(argv)

    if args.show is not None:
        measured = measure_package(args.package)
        keys = HARD_ZERO + SOFT_ZERO
        print(f"{'module':52s} " + " ".join(f"{k:>8s}" for k in keys))
        for relative, m in sorted(measured.items()):
            if args.show and relative not in args.show:
                continue
            if any(m["counts"][k] for k in keys) or args.show:
                print(f"{relative:52s} " + " ".join(f"{m['counts'][k]:8d}" for k in keys))
                if args.show:
                    for function, counts in sorted(m["functions"].items()):
                        print(f"    {function}: {dict(sorted(counts.items()))}")
                    for target, line in m["imports"]:
                        print(f"    import outside the allowlist: {target} (line {line})")
        return 0
    if args.ratchet:
        measured = measure_package(args.package)
        new, notes = ratchet(measured, load_budget(args.budget))
        args.budget.write_text(dump_budget(new), encoding="utf-8")
        print("\n".join(notes) or "nothing to lower")
        return 0
    if args.dates:
        warnings, failures = date_findings(load_budget(args.budget), args.today)
        for line in warnings:
            print(f"warning: {line}", file=sys.stderr)
        print("\n".join(failures) or "no budget row is past its date")
        return 1 if failures else 0
    if args.monotone:
        old = committed_budget()
        if old is None:
            print("no committed budget to compare with (git or the file is unavailable)")
            return 0
        failures = monotone(old, load_budget(args.budget))
        print("\n".join(failures) or "the budget is no looser than HEAD")
        return 1 if failures else 0
    failures = run_check(args.package, args.budget)
    print("\n".join(failures) or "numerics budget satisfied")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
