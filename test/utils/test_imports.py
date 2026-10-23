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
    result = subprocess.run(
        [sys.executable, "-c", "from mandacaru.utils import *\n"
         "assert callable(write_references)\n"
         "assert callable(write_cube)\n"
         "assert callable(dump_hamiltonian)\n"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
