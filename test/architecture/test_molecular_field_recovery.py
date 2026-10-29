"""Physical density and orbital contracts behind the ASE visualization views."""

import os
from pathlib import Path
import subprocess
import sys

from ase import Atoms
from ase.io.cube import read_cube_data
import numpy as np
import pytest

from mandacaru import Mandacaru, Viewer3D


def h2():
    return Atoms("H2", positions=[[2.5, 2.5, 2.13], [2.5, 2.5, 2.87]], cell=[5] * 3)


def test_semilocal_fields_conserve_charge_and_do_not_construct_a_hamiltonian(tmp_path):
    atoms = h2()
    with Mandacaru(
        method="dft", atoms=atoms, xc="pbe", grid_spacing=0.5, resources={"threads": 1}
    ) as calc:
        atoms.get_potential_energy()
        natural = calc.natural_orbitals()
        assert natural.n_electrons == pytest.approx(2.0)
        assert np.allclose(natural.occupations, [2.0, 0.0], atol=1e-10)
        density = calc.volumetric_field()
        assert density.integral == pytest.approx(2.0, abs=1e-11)
        assert calc.volumetric_field("difference_density").norm < 1e-24
        assert calc.volumetric_field("natural_orbital", 0).norm == pytest.approx(
            1.0, abs=1e-11
        )
        assert not calc.integrals.has_eri and not calc.native_reference.has_model
        path = tmp_path / "density.cube"
        assert calc.write_cube(path) == str(path)
        data, recovered = read_cube_data(path)
        assert np.allclose(data, density.data, atol=1e-6)
        assert np.allclose(recovered.positions, atoms.positions, atol=1e-5)
        image = tmp_path / "orbital.png"
        Viewer3D(
            calc,
            quantity="natural_orbital",
            index=0,
            grid_spacing=0.4,
            mode="scatter",
            points=500,
        ).save(image)
        assert image.stat().st_size > 1000


def test_complex_rdm_and_frozen_core_use_the_correct_orbital_conjugation():
    atoms = h2()
    with Mandacaru(
        method="rhf", atoms=atoms, grid_spacing=0.5, resources={"threads": 1}
    ) as calc:
        atoms.get_potential_energy()
        c = np.array([1.0, 1j]) / np.sqrt(2.0)
        gamma = np.zeros((4, 4), dtype=complex)
        gamma[:2, :2] = np.outer(c.conj(), c)
        result = calc.native_reference.observables(gamma=gamma, active=[0, 1])
        assert np.allclose(result["occupations"], [1.0, 0.0], atol=1e-12)
        source = calc.native_reference.snapshot()["model_orbitals"]
        lowdin = calc.integrals.snapshot()["lowdin"]
        expected = lowdin @ source @ c
        actual = result["coefficients"][:, 0]
        assert np.allclose(
            np.outer(actual, actual.conj()),
            np.outer(expected, expected.conj()),
            atol=1e-12,
        )
        with pytest.raises(ValueError, match="orbital is complex"):
            calc.native_reference.observables(gamma=gamma, quantity="natural_orbital")
        amplitude = calc.native_reference.observables(
            gamma=gamma, quantity="natural_orbital", component="modulus"
        )
        assert amplitude["norm"] == pytest.approx(1.0, abs=1e-12)
        frozen = calc.native_reference.observables(
            gamma=np.zeros((2, 2), dtype=complex),
            active=[1],
            frozen=[0],
            quantity="density",
        )
        assert np.allclose(frozen["occupations"], [2.0, 0.0], atol=1e-12)
        assert frozen["integral"] == pytest.approx(2.0, abs=1e-12)


def test_original_stretched_h2_visualization_example_writes_physical_fields(tmp_path):
    import shutil

    example = (
        Path(__file__).resolve().parents[2]
        / "examples/visualization/02_h2_stretched_cube_files.py"
    )
    target = tmp_path / example.name
    shutil.copyfile(example, target)
    run = subprocess.run(
        [sys.executable, str(target)],
        cwd=tmp_path,
        env=dict(
            os.environ, MPLBACKEND="Agg", MPLCONFIGDIR=str(tmp_path / "matplotlib")
        ),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    data = tmp_path / "data"
    assert len(list(data.glob("*.cube"))) == 3
    assert (data / "h2_difference_slice.png").stat().st_size > 1000
    density, _ = read_cube_data(data / "h2_stretched_density.cube")
    difference, _ = read_cube_data(data / "h2_stretched_difference.cube")
    with (data / "h2_stretched_density.cube").open() as cube:
        lines = [next(cube) for _ in range(6)]
    step = np.array([[float(v) for v in row.split()[1:]] for row in lines[3:6]])
    dv = abs(np.linalg.det(step))
    assert np.sum(density) * dv == pytest.approx(2.0, abs=2e-5)
    assert abs(np.sum(difference) * dv) < 1e-7
    assert np.linalg.norm(difference) > 1e-3
