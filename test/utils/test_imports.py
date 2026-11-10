"""Low-level profiling imports must not depend on prior package imports."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", [
    "mandacaru.integrals",
    "mandacaru.utils.profiling",
    "mandacaru.core",
    "mandacaru.pseudopotentials.paw",
])
def test_low_level_modules_import_in_a_fresh_process(module):
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_public_utility_exports_remain_available_in_a_fresh_process():
    """Every name of ``mandacaru.utils.__all__`` resolves, and the documented
    ones (the run-log parser, the reference table and its BibTeX renderer) are
    there; the cube and XSF writers are methods of the calculator."""
    result = subprocess.run(
        [sys.executable, "-c", "import mandacaru.utils as utils\n"
         "assert all(getattr(utils, name) is not None for name in utils.__all__)\n"
         "from mandacaru.utils import *\n"
         "assert callable(parse_output)\n"
         "assert callable(bibtex)\n"
         "assert len(REFERENCES) > 0\n"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
