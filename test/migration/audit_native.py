"""Inventory Python numerical candidates and their native coverage evidence.

Run directly to regenerate a reviewable JSON audit. An import/call inventory is
evidence for review, not proof that a module can safely be deleted.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARRAY_CONVERSIONS = {"array", "asarray", "ascontiguousarray", "atleast_1d",
                     "atleast_2d", "reshape", "ravel", "ndim", "iscomplexobj",
                     "isrealobj", "isscalar", "shape"}
NUMERICAL_PACKAGES = {"numpy", "scipy", "jax", "math", "numba"}


def inventory(root=ROOT):
    records = []
    for path in sorted((root / "src/mandacaru").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports[name.asname or name.name.split(".")[0]] = name.name
            elif isinstance(node, ast.ImportFrom):
                for name in node.names:
                    imports[name.asname or name.name] = f"{node.module or ''}.{name.name}"
        candidates = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = set()
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                parts, value = [], call.func
                while isinstance(value, ast.Attribute):
                    parts.append(value.attr)
                    value = value.value
                if isinstance(value, ast.Name):
                    source = imports.get(value.id, value.id)
                    target = ".".join([source, *reversed(parts)])
                    if source.split(".")[0] in NUMERICAL_PACKAGES:
                        if target.split(".")[-1] not in ARRAY_CONVERSIONS:
                            calls.add(target)
            if calls:
                candidates.append({"name": node.name, "line": node.lineno,
                                   "numerical_calls": sorted(calls)})
        records.append({"path": path.relative_to(root).as_posix(),
                        "lines": len(path.read_text().splitlines()),
                        "numerical_candidates": candidates})
    return {"schema_version": 1, "python_modules": records,
            "module_count": len(records),
            "candidate_functions": sum(len(r["numerical_candidates"]) for r in records),
            "limitations": "Conversions are excluded. Nested functions can appear under their enclosing function as well. Pure-Python arithmetic and calls to project functions require separate review."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = inventory()
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(f"{data['module_count']} modules; {data['candidate_functions']} numerical candidates")


if __name__ == "__main__":
    main()
