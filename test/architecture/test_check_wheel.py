"""The wheel-content checks of ``scripts/check_wheel.py`` on synthetic wheels."""

import runpy
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CHECK = runpy.run_path(str(ROOT / "scripts/check_wheel.py"))
SCRIPTS = {"mandacaru": "mandacaru.cli:main", "mandacaru-paw": "mandacaru.pseudopotentials.library_cli:main"}

GOOD = {
    "names": ["mandacaru/__init__.py", "mandacaru/_native.cpython-314-darwin.so",
              "mandacaru-1.dist-info/METADATA", "mandacaru-1.dist-info/WHEEL",
              "mandacaru-1.dist-info/entry_points.txt"],
    "metadata": "Name: mandacaru\nRequires-Dist: numpy>=2.0.0\nRequires-Dist: scipy\nRequires-Dist: pytest-cov; extra == 'dev'\n",
    "wheel_metadata": "Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp314-cp314-macosx_26_0_arm64\n",
    "entry_points": "[console_scripts]\n" + "".join(f"{k} = {v}\n" for k, v in SCRIPTS.items()),
}


def problems(**changes):
    case = {**GOOD, **changes}
    return CHECK["content_problems"](case["names"], case["metadata"], case["wheel_metadata"],
                                     case["entry_points"], SCRIPTS)


def test_a_sound_platform_wheel_has_no_problem():
    assert problems() == []


@pytest.mark.parametrize("names", [
    [n for n in GOOD["names"] if "_native" not in n],
    GOOD["names"] + ["mandacaru/_native2.cpython-314-darwin.so"],
    ["mandacaru/other.so"] + [n for n in GOOD["names"] if "_native" not in n],
])
def test_the_binary_must_be_exactly_one_native_extension(names):
    assert problems(names=names)[0].startswith("want exactly one binary")


@pytest.mark.parametrize("extra", ["mandacaru/integrals/csrc/kernels.c", "mandacaru/header.h",
                                   "mandacaru/x.cpp", "mandacaru/x.f90",
                                   "mandacaru/integrals/libmandacaru_integrals.txt",
                                   "mandacaru/basis/csrc/notes"])
def test_c_sources_the_retired_integral_library_and_csrc_do_not_ship(extra):
    assert any("sources" in p for p in problems(names=GOOD["names"] + [extra]))


def test_the_wheel_is_a_platform_wheel():
    assert problems(wheel_metadata="Root-Is-Purelib: true\nTag: py3-none-any\n")
    assert problems(wheel_metadata="Root-Is-Purelib: false\nTag: py3-none-any\n")


@pytest.mark.parametrize("retired", ["pandas", "pyarrow", "fastparquet"])
def test_a_retired_parquet_requirement_is_refused_with_other_requirements_present(retired):
    for line in (retired, retired + ">=1", retired + "[x]", retired + " (>=1)"):
        assert any("retired requirement" in p
                   for p in problems(metadata=GOOD["metadata"] + f"Requires-Dist: {line}\n"))
    assert problems(metadata=GOOD["metadata"] + f"Requires-Dist: {retired}-extras\n") == []


def test_github_only_logo_markup_is_refused():
    assert problems(metadata=GOOD["metadata"] + "![x](a.png#gh-dark-mode-only)\n")


def test_every_declared_console_script_must_be_installed():
    assert problems(entry_points="[console_scripts]\nmandacaru = mandacaru.cli:main\n") == [
        "entry point missing: mandacaru-paw = mandacaru.pseudopotentials.library_cli:main"]


@pytest.mark.parametrize("stray", ["mandacaru/__pycache__/x.pyc", "mandacaru/rust/target/x"])
def test_build_products_are_refused(stray):
    assert any("build products" in p for p in problems(names=GOOD["names"] + [stray]))


def test_a_retired_module_is_refused():
    assert any("retired file" in p for p in problems(names=GOOD["names"] + ["mandacaru/basis/r2scan.py"]))


def test_the_wheel_is_read_from_the_zip_and_the_entry_points_come_from_pyproject(tmp_path):
    wheel = tmp_path / "mandacaru-1-cp314-cp314-macosx_26_0_arm64.whl"
    declared = CHECK["project_scripts"]()
    assert declared == SCRIPTS
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in GOOD["names"]:
            body = {"METADATA": GOOD["metadata"], "WHEEL": GOOD["wheel_metadata"],
                    "entry_points.txt": GOOD["entry_points"]}.get(name.rsplit("/", 1)[-1], "")
            archive.writestr(name, body)
    assert CHECK["wheel_problems"](wheel) == []
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("mandacaru/integrals/csrc/a.c", "")
    assert CHECK["wheel_problems"](wheel)
