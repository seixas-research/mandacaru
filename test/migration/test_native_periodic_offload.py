"""Bulk arrays of a periodic calculation on disk: the same numbers as in memory.

``resources={"disk_offload_bytes": ...}`` moves the Bloch orbital stacks (and the other bulk
arrays) of a crystal to a scratch folder; energy, forces and stress must not notice, the folder
must leave with the calculation, and a scratch location that could land in the working directory
or in a ``.mandacaru`` folder must be refused.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk

from mandacaru import Mandacaru, _native

pytestmark = pytest.mark.skipif(
    not os.environ.get("MANDACARU_PAW_PATH"), reason="MANDACARU_PAW_PATH is not set"
)

BASIS = {"name": "PAW-LCAO", "size": "SZ", "filter_cutoff": 200}
SCF = {"max_iterations": 400, "tolerance": 1e-11, "density_tolerance": 1e-9, "mixing": 0.25, "history": 7,
       "width_ha": 0.3 / 27.211386245988}


def aluminum(offload):
    resources = {}
    if offload is not None:
        resources["disk_offload_bytes"] = offload
    atoms = bulk("Al", "fcc", a=4.05)
    atoms.calc = Mandacaru(
        method="dft", basis=BASIS, grid_spacing=0.6, kpts={"size": (2, 2, 2), "gamma": True},
        smearing={"method": "fermi-dirac", "width": 0.3}, trace=False, periodic_scf_options=SCF,
        parallelization={"threads": 1}, resources=resources,
    )
    return atoms


def scratch_folders():
    return {p for p in Path(os.environ.get("TMPDIR", "/tmp")).glob("mandacaru-scratch-*")}


def test_energy_and_forces_on_disk_equal_those_in_memory():
    before = scratch_folders()
    numbers = {}
    for name, offload in (("memory", 2**62), ("disk", 0)):
        atoms = aluminum(offload)
        calc = atoms.calc
        energy = atoms.get_potential_energy(force_consistent=True)
        forces = atoms.get_forces()
        numbers[name] = (energy, forces, calc.context.disk_archived_bytes, calc.context.disk_offload_bytes)
        calc.close()
        assert calc.context.memory_used_bytes == 0
    assert numbers["memory"][2] == 0
    assert numbers["disk"][2] > 0, "a zero threshold must write the Bloch stacks to disk"
    assert (numbers["memory"][3], numbers["disk"][3]) == (2**62, 0)
    assert abs(numbers["disk"][0] - numbers["memory"][0]) < 1e-12
    assert np.max(np.abs(numbers["disk"][1] - numbers["memory"][1])) < 1e-9
    del atoms, calc  # close() broke the atoms <-> calculator cycle: no collector pass is needed
    assert scratch_folders() == before, "the scratch folder must leave with the calculation"
    assert _native.Context(threads=1).disk_offload_bytes == 8 * 2**20


def test_the_offload_threshold_is_validated():
    for bad in (-1, 1.5, "8MiB", True):
        with pytest.raises(ValueError, match="disk_offload_bytes"):
            aluminum(bad)


def test_a_scratch_location_inside_the_working_directory_or_a_dot_mandacaru_folder_is_refused(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("TMPDIR", "relative-scratch")
    with pytest.raises(ValueError, match="TMPDIR"):
        aluminum(0).get_potential_energy()
    hidden = tmp_path / ".mandacaru"
    hidden.mkdir()
    monkeypatch.setenv("TMPDIR", str(hidden))
    with pytest.raises(ValueError, match=r"\.mandacaru"):
        aluminum(0).get_potential_energy()
    assert not Path("relative-scratch").exists()
    assert list(hidden.iterdir()) == []


def test_freed_memory_goes_back_to_the_system_without_changing_a_result():
    assert isinstance(_native.Context(threads=1).release_freed_memory(), bool)
    atoms = aluminum(None)
    first = atoms.get_potential_energy()
    atoms.calc.close()
    again = aluminum(None)
    again.calc.context.release_freed_memory()
    assert abs(again.get_potential_energy() - first) < 1e-12
    again.calc.close()
