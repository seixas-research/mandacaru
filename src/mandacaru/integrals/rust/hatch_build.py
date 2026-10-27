"""Build the PyO3 extension while preserving Hatch's project metadata hooks."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class NativeBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if self.target_name != "wheel":
            return
        root = Path(self.root)
        manifest = root / "src/mandacaru/integrals/rust/Cargo.toml"
        cargo = shutil.which("cargo")
        if cargo is None:
            raise RuntimeError("Building Mandacaru from source requires Rust/Cargo; install a platform wheel or install Rust.")
        env = dict(os.environ, PYO3_PYTHON=sys.executable)
        if env.get("CARGO_BUILD_TARGET"):
            raise RuntimeError("Cross-target wheels are not configured; unset CARGO_BUILD_TARGET for a host build.")
        subprocess.run([cargo, "build", "--locked", "--release", "--manifest-path", str(manifest),
                        "--package", "mandacaru-python"], cwd=root, env=env, check=True)
        # Cargo's machine-readable metadata resolves custom target locations.
        metadata = subprocess.run([cargo, "metadata", "--no-deps", "--format-version", "1",
                                   "--manifest-path", str(manifest)], cwd=root, env=env,
                                  check=True, capture_output=True, text=True)
        import json
        target = Path(json.loads(metadata.stdout)["target_directory"])
        if sys.platform == "win32":
            filename = "_native.dll"
        elif sys.platform == "darwin":
            filename = "lib_native.dylib"
        else:
            filename = "lib_native.so"
        artifact = target / "release" / filename
        suffix = sysconfig.get_config_var("EXT_SUFFIX")
        if not suffix or not artifact.is_file():
            raise RuntimeError(f"Native extension build did not produce {artifact}")
        destination = "mandacaru/_native" + suffix
        if version == "editable":
            # The Python package resolves from src/ in editable mode. Keep the
            # extension beside it. Hatch tags editable descriptors as any, so
            # the descriptor itself must not contain a platform binary.
            build_data["pure_python"] = True
            local = root / "src" / destination
            temporary = local.with_name(local.name + ".tmp")
            try:
                shutil.copyfile(artifact, temporary)
                os.replace(temporary, local)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            build_data["pure_python"] = False
            build_data["infer_tag"] = True
            build_data["force_include"][str(artifact)] = destination
