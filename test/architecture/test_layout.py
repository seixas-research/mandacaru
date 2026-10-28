"""Keep language and scientific dependency boundaries executable locally."""

from pathlib import Path
import runpy

import pytest

CHECKS = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "scripts/check_architecture.py")
)


def test_repository_obeys_architectural_boundaries():
    CHECKS["check"]()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("use crate::quantum::C64;", {"quantum"}),
        (
            "use crate::{math::fft::{Fft3, fast_len}, quantum::C64, Context};",
            {"math", "quantum", "Context"},
        ),
        ("crate :: { models, paw::{channel, radial} };", {"models", "paw"}),
        ('// crate::quantum\n/* crate::paw */ let s = "crate::molecular";', set()),
    ],
)
def test_rust_dependency_detection_handles_groups_and_ignores_text(source, expected):
    assert CHECKS["rust_dependencies"](source) == expected


def test_crate_dependency_detection_includes_target_and_build_dependencies():
    manifest = {
        "dependencies": {"array": {"package": "numpy", "version": "0.30"}},
        "build-dependencies": {"pyo3-build-config": "0.30"},
        "target": {"cfg(unix)": {"dependencies": {"mandacaru-core": "0.1"}}},
    }
    assert CHECKS["cargo_dependencies"](manifest) == {
        "numpy",
        "pyo3-build-config",
        "mandacaru-core",
    }
