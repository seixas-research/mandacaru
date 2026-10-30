"""Check architectural boundaries without importing the extension or optional SDKs."""

import ast
from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src/mandacaru"
BOUNDARY = (
    "calculation.py",
    "structure.py",
    "resources.py",
    "results.py",
    "specifications.py",
    "models/basis.py",
)
INTEGRATIONS = {
    "scipy",
    "jax",
    "qiskit",
    "braket",
    "cirq",
    "pandas",
    "pyarrow",
    "fastparquet",
    "matplotlib",
}
# Parquet is read and written by mandacaru-core; no Python package may return.
RETIRED_PARQUET_PACKAGES = {"fastparquet", "pandas", "pyarrow"}


# Follow explicit and grouped Rust paths (including `use crate::{math, paw}`).
# Strip comments and strings so explanatory text cannot create dependencies.
def rust_dependencies(source):
    source = re.sub(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"', "", source, flags=re.S)
    tokens = re.findall(r"[A-Za-z_][A-Za-z_0-9]*|::|[{};,]", source)
    dependencies = set()
    for i in range(len(tokens) - 2):
        if tokens[i : i + 2] != ["crate", "::"]:
            continue
        if tokens[i + 2] != "{":
            dependencies.add(tokens[i + 2])
            continue
        depth = 1
        expect_name = True
        for token in tokens[i + 3 :]:
            if token == "{":
                depth += 1
            elif token == "}":
                depth -= 1
                if depth == 0:
                    break
            elif depth == 1:
                if expect_name and token not in {",", "::", ";"}:
                    dependencies.add(token)
                    expect_name = False
                elif token == ",":
                    expect_name = True
    return dependencies


def cargo_dependencies(manifest):
    sections = [manifest]
    sections.extend(manifest.get("target", {}).values())
    return {
        entry.get("package", name) if isinstance(entry, dict) else name
        for section in sections
        for category in ("dependencies", "build-dependencies", "dev-dependencies")
        for name, entry in section.get(category, {}).items()
    }


def check():
    failures = []
    for filename in BOUNDARY:
        tree = ast.parse((PACKAGE / filename).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                imports = [node.module or ""]
            else:
                imports = []
            for name in imports:
                if name.split(".")[0] in INTEGRATIONS:
                    failures.append(
                        f"{filename}:{node.lineno}: integration dependency {name} in calculation boundary"
                    )
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                arguments = (
                    node.args.posonlyargs + node.args.args + node.args.kwonlyargs
                )
                for arg in arguments:
                    if arg.arg in {"engine", "engine_options"}:
                        failures.append(
                            f"{filename}:{node.lineno}: retired execution selector {arg.arg}"
                        )
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    required = {
        "numpy",
        "scipy",
        "threadpoolctl",
        "ase",
        "spglib",
        "matplotlib",
        "qiskit",
        "qiskit-ibm-runtime",
        "amazon-braket-sdk",
        "cirq",
        "pytest",
        "psutil",
    }
    declared = {
        re.split(r"[<>=!~;\[]", item)[0].strip().lower()
        for item in metadata["project"]["dependencies"]
    }
    if missing := required - declared:
        failures.append(f"required Python dependencies are missing: {sorted(missing)}")
    extras = {
        re.split(r"[<>=!~;\[]", item)[0].strip().lower()
        for items in metadata["project"].get("optional-dependencies", {}).values()
        for item in items
    }
    if retired := RETIRED_PARQUET_PACKAGES & (declared | extras):
        failures.append(
            f"Parquet is native; Python packages must not be declared: {sorted(retired)}"
        )
    for path in sorted(PACKAGE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imports = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                imports = [node.module or ""]
            else:
                continue
            for name in imports:
                if name.split(".")[0] in RETIRED_PARQUET_PACKAGES:
                    failures.append(
                        f"{path.relative_to(ROOT)}:{node.lineno}: imports retired Parquet package {name}"
                    )
    retired_extras = {
        "ase",
        "periodic",
        "visualization",
        "columnar",
        "qiskit",
        "braket",
        "cirq",
    }
    if retired_extras & metadata["project"].get("optional-dependencies", {}).keys():
        failures.append("runtime scientific and SDK dependencies must be required")
    for filename in (
        "_rust_molecular.py",
        "_rust_quantum.py",
        "hartree_fock.py",
        "dft.py",
        "spinor_dft.py",
        "periodic_dft.py",
        "vqe.py",
        "adapt_vqe.py",
        "base.py",
        "pool_driver.py",
        "subspace.py",
    ):
        if (PACKAGE / "algorithms" / filename).exists():
            failures.append(f"duplicate calculator orchestration: {filename}")
    for name in (
        "engine",
        "direct",
        "cholesky",
        "two_center",
        "localized",
        "fft",
        "buffers",
        "poisson",
    ):
        if (PACKAGE / "integrals" / (name + ".py")).exists():
            failures.append(f"legacy Python integral implementation: {name}")
    if (ROOT / ".github/workflows").exists() and any(
        (ROOT / ".github/workflows").glob("*.y*ml")
    ):
        failures.append("GitHub workflows are prohibited; use local checks")
    if any(PACKAGE.rglob("*.rs")):
        failures.append("Rust sources must live in crates/, outside the Python package")
    workspace = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]
    expected = {
        "crates/mandacaru-core",
        "crates/mandacaru-kernels",
        "crates/mandacaru-python",
    }
    if set(workspace["members"]) != expected:
        failures.append(
            "Cargo workspace must separate scientific core, kernels and Python bindings"
        )
    manifests = {
        name: tomllib.loads((ROOT / "crates" / name / "Cargo.toml").read_text())
        for name in ("mandacaru-core", "mandacaru-kernels", "mandacaru-python")
    }
    for name in ("mandacaru-core", "mandacaru-kernels"):
        dependencies = cargo_dependencies(manifests[name])
        if {"pyo3", "pyo3-build-config", "numpy", "mandacaru-python"} & dependencies:
            failures.append(
                f"{name}: scientific crates cannot depend on Python bindings"
            )
    if "mandacaru-core" in cargo_dependencies(manifests["mandacaru-kernels"]):
        failures.append("low-level kernels cannot depend on the scientific core")
    if "mandacaru-core" not in cargo_dependencies(manifests["mandacaru-python"]):
        failures.append("Python bindings must use the scientific core")
    if (PACKAGE / "algorithms").exists():
        failures.append("the former Python algorithms package must not be restored")
    core = ROOT / "crates/mandacaru-core/src"
    solvers = {"molecular", "periodic", "quantum", "paw"}
    boundaries = {
        "math": solvers | {"basis", "models"},
        "models": solvers | {"basis"},
        "basis": solvers,
        "parquet": solvers | {"basis", "models"},
    }
    for layer, forbidden in boundaries.items():
        for path in (core / layer).rglob("*.rs"):
            invalid = rust_dependencies(path.read_text()) & forbidden
            if invalid:
                failures.append(
                    f"{path.relative_to(ROOT)}: {layer} depends on {sorted(invalid)}"
                )
    # ruzstd is the one third-party Parquet dependency: pinned and declared once,
    # in the workspace, and used by mandacaru-core only.
    ruzstd = workspace.get("dependencies", {}).get("ruzstd")
    if not isinstance(ruzstd, dict) or not str(ruzstd.get("version", "")).startswith("="):
        failures.append("the workspace must pin ruzstd to an exact version")
    elif ruzstd.get("default-features") is not False:
        failures.append("the workspace must declare ruzstd with default-features = false")
    for name, manifest in manifests.items():
        entry = {
            key: value
            for section in [manifest, *manifest.get("target", {}).values()]
            for category in ("dependencies", "build-dependencies", "dev-dependencies")
            for key, value in section.get(category, {}).items()
        }.get("ruzstd")
        if name == "mandacaru-core":
            if entry != {"workspace": True}:
                failures.append("mandacaru-core must take ruzstd from the workspace")
        elif entry is not None:
            failures.append(f"{name}: only mandacaru-core may depend on ruzstd")
    lock = tomllib.loads((ROOT / "Cargo.lock").read_text())
    if sum(item["name"] == "ruzstd" for item in lock["package"]) != 1:
        failures.append("Cargo.lock must hold exactly one ruzstd package")
    hook = metadata["tool"]["hatch"]["build"]["targets"]["wheel"]["hooks"]["custom"][
        "path"
    ]
    if hook != "build_hooks/native.py" or not (ROOT / hook).is_file():
        failures.append("native packaging must use build_hooks/native.py")
    included = metadata["tool"]["hatch"]["build"]["targets"]["sdist"]["only-include"]
    for path in ("crates", "build_hooks", "Cargo.toml", "Cargo.lock"):
        if path not in included:
            failures.append(f"native source distribution omits {path}")
    calculator = (PACKAGE / "calculation.py").read_text()
    for forbidden in ("set_buffer_dir(", "set_dump_buffers(", "subprocess", "ctypes"):
        if forbidden in calculator:
            failures.append(
                f"calculator contains process-global or compiler operation: {forbidden}"
            )
    if failures:
        raise SystemExit("\n".join(failures))
    print("Architecture checks passed")


if __name__ == "__main__":
    check()
