"""Capture source provenance, declared APIs, imports and test evidence without imports.

Run from any directory: python test/migration/inventory.py --output PATH
The inventory is static evidence, not a claim that an API or test passes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "mandacaru"
SOURCE_SUFFIXES = {".py", ".pyi", ".rs", ".c", ".h", ".toml", ".lock", ".json"}
EXCLUDED = {"__pycache__", "target", "build", "cbuild"}


def source_manifest(root: Path = ROOT) -> dict[str, str]:
    """Fingerprint source and build inputs; exclude binaries and generated trees."""
    paths = [root / "pyproject.toml"]
    paths.extend(p for p in (root / "src" / "mandacaru").rglob("*")
                 if p.is_file() and (p.suffix in SOURCE_SUFFIXES or p.name == "CMakeLists.txt")
                 and not EXCLUDED.intersection(p.relative_to(root).parts))
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def digest_manifest(manifest: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def git_revision() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def declarations(tree: ast.Module) -> dict:
    """Keep source expressions for defaults, including nonliteral defaults."""
    constants, classes, functions = {}, {}, {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and (
                        target.id.isupper() or target.id == "__all__"):
                    constants[target.id] = ast.unparse(node.value)
        if isinstance(node, ast.ClassDef):
            methods = {}
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if member.name == "__init__" or not member.name.startswith("_"):
                        methods.setdefault(member.name, []).append({
                            "arguments": ast.unparse(member.args),
                            "returns": ast.unparse(member.returns) if member.returns else None,
                            "decorators": [ast.unparse(d) for d in member.decorator_list],
                        })
            fields = {n.target.id: ast.unparse(n.annotation)
                      for n in node.body if isinstance(n, ast.AnnAssign)
                      and isinstance(n.target, ast.Name)}
            if methods or fields:
                classes[node.name] = {"methods": methods, "annotated_fields": fields}
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            functions[node.name] = {"arguments": ast.unparse(node.args),
                                    "returns": ast.unparse(node.returns) if node.returns else None}
    return {"constants": constants, "classes": classes, "functions": functions}


def imports(tree: ast.Module, module: str, package: bool) -> list[dict]:
    edges = []
    parent = module.split(".") if package else module.split(".")[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = parent[:len(parent) - node.level + 1]
                names = [".".join(base + ([node.module] if node.module else []))]
                if node.module is None:
                    names = [names[0] + "." + alias.name for alias in node.names]
            else:
                names = [node.module or ""]
        else:
            continue
        edges.extend({"module": name, "line": node.lineno} for name in names)
    return sorted(edges, key=lambda e: (e["line"], e["module"]))


def build_inventory() -> dict:
    manifest = source_manifest()
    modules, api, sizes, external, internal = {}, {}, Counter(), {}, Counter()
    subsystems = {p.name for p in SOURCE.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    for path in sorted(SOURCE.rglob("*.py")):
        if EXCLUDED.intersection(path.relative_to(ROOT).parts):
            continue
        relative = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=relative)
        parts = list(path.relative_to(ROOT / "src").with_suffix("").parts)
        package = parts[-1] == "__init__"
        module = ".".join(parts[:-1] if package else parts)
        group = parts[1] if len(parts) > 2 else "(root)"
        edges = imports(tree, module, package)
        modules[relative] = {"module": module, "lines": len(text.splitlines()),
                             "imports": edges}
        sizes[group + ".files"] += 1
        sizes[group + ".lines"] += len(text.splitlines())
        for edge in edges:
            head = edge["module"].split(".")[0]
            if head != "mandacaru":
                external.setdefault(head, set()).add(relative)
            else:
                pieces = edge["module"].split(".")
                destination = pieces[1] if len(pieces) > 1 and pieces[1] in subsystems else "(root)"
                if destination != group:
                    internal[(group, destination)] += 1
        api[relative] = declarations(tree)
    tests = {}
    for path in sorted((ROOT / "test").rglob("test_*.py")):
        if "migration" in path.relative_to(ROOT / "test").parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tests[path.relative_to(ROOT).as_posix()] = sum(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name.startswith("test_") for n in ast.walk(tree))
    return {
        "schema_version": 1,
        "evidence": "Static declarations and import sites; guarded imports included; no execution or coverage inference.",
        "git_revision": git_revision(),
        "source_sha256": digest_manifest(manifest),
        "source_files": manifest,
        "subsystems": dict(sorted(sizes.items())),
        "internal_subsystem_import_sites": [
            {"from": origin, "to": destination, "count": count}
            for (origin, destination), count in sorted(internal.items())],
        "native_sources": {suffix: {
            "files": sum(Path(name).suffix == suffix for name in manifest),
            "lines": sum(len((ROOT / name).read_text().splitlines()) for name in manifest
                         if Path(name).suffix == suffix)} for suffix in (".rs", ".c", ".h")},
        "modules": modules,
        "declared_api": api,
        "external_import_sites": {k: sorted(v) for k, v in sorted(external.items())},
        "test_functions_by_file": tests,
        "test_function_count": sum(tests.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_inventory(), indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
