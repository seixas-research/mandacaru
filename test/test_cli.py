# -*- coding: utf-8 -*-
# file: test/test_cli.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The ``mandacaru`` console script.

``cli.py`` is a thin adapter over the single entry point: it reads a geometry,
forwards its options to ``Mandacaru(...)`` and prints one JSON report.  The
tests check three things: the options it forwards are the ones given, a real
run agrees with calling ``Mandacaru`` directly, and bad input is refused the
way argparse refuses it.  The PAW library commands live in
``test/pseudopotentials/test_library_cli.py``.
"""

import json

import numpy as np
import pytest

import mandacaru
from mandacaru import Geometry, Mandacaru
from mandacaru.cli import main
from mandacaru.version import __version__

H2 = {
    "numbers": [1, 1],
    "positions": [[2.5, 2.5, 2.13], [2.5, 2.5, 2.87]],
    "cell": [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
}
QUICK = ["--method", "rhf", "--grid-spacing", "0.5", "--threads", "1"]


@pytest.fixture
def h2_json(tmp_path):
    path = tmp_path / "h2.json"
    path.write_text(json.dumps(H2))
    return str(path)


class Recorder:
    """Stands in for ``Mandacaru``: records how the CLI builds and calls it."""

    instances = []

    def __init__(self, **options):
        self.options = options
        self.calls = []
        self.results = {"forces": np.zeros((2, 3))}
        Recorder.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def _result(self):
        class Result:
            success = True

            @staticmethod
            def in_units(unit):
                return -1.0

        return Result()

    def run(self, geometry=None):
        self.calls.append(("run", geometry))
        return self._result()

    def calculate(self, geometry, properties):
        self.calls.append(("calculate", geometry, tuple(properties)))
        return self._result()


@pytest.fixture
def recorder(monkeypatch):
    Recorder.instances = []
    monkeypatch.setattr(mandacaru, "Mandacaru", Recorder)
    return Recorder


class TestForwardedOptions:
    def test_the_defaults_reach_the_calculator(self, recorder, h2_json):
        assert main([h2_json]) == 0
        options = recorder.instances[0].options
        assert options["method"] == "dft" and options["xc"] == "lda"
        assert options["basis"] == "HAO" and options["grid_spacing"] == 0.2
        assert options["resources"] == {"threads": None,
                                        "memory_limit_bytes": 512 * 1024**2}
        assert options["load_hamiltonian"] is None
        assert options["save_hamiltonian"] is None
        assert options["load_crystal"] is None

    def test_every_option_given_is_forwarded(self, recorder, h2_json):
        assert main([h2_json, "--method", "dft", "--xc", "pbe", "--basis",
                     "NAO", "--grid-spacing", "0.3", "--threads", "2",
                     "--memory-mb", "64", "--save-hamiltonian", "h.parquet"]) == 0
        options = recorder.instances[0].options
        assert (options["xc"], options["basis"]) == ("pbe", "NAO")
        assert options["grid_spacing"] == 0.3
        assert options["resources"] == {"threads": 2,
                                        "memory_limit_bytes": 64 * 1024**2}
        assert options["save_hamiltonian"] == "h.parquet"

    def test_a_cached_problem_needs_no_geometry(self, recorder, capsys):
        assert main(["--load-hamiltonian", "h.parquet", "--method", "vqe"]) == 0
        instance = recorder.instances[0]
        assert instance.options["load_hamiltonian"] == "h.parquet"
        assert instance.calls == [("run", None)]

    def test_a_cached_crystal_needs_no_geometry(self, recorder):
        assert main(["--load-crystal", "c.json"]) == 0
        assert recorder.instances[0].options["load_crystal"] == "c.json"

    def test_forces_use_the_calculate_route(self, recorder, h2_json, capsys):
        assert main([h2_json, "--forces"]) == 0
        kind, geometry, properties = recorder.instances[0].calls[0]
        assert kind == "calculate" and properties == ("energy", "forces")
        report = json.loads(capsys.readouterr().out)
        assert report["forces_ev_per_angstrom"] == [[0.0] * 3] * 2

    def test_without_forces_the_report_has_none(self, recorder, h2_json, capsys):
        assert main([h2_json]) == 0
        assert "forces_ev_per_angstrom" not in json.loads(capsys.readouterr().out)


class TestGeometryInput:
    def test_a_json_geometry_is_read_as_geometry_keywords(self, recorder,
                                                          h2_json):
        assert main([h2_json]) == 0
        kind, geometry = recorder.instances[0].calls[0]
        assert isinstance(geometry, Geometry) and geometry.numbers == (1, 1)
        assert np.allclose(geometry.positions, H2["positions"])

    def test_another_format_goes_through_ase(self, recorder, tmp_path):
        ase = pytest.importorskip("ase")
        from ase.io import write

        path = tmp_path / "h2.extxyz"
        write(path, ase.Atoms("H2", positions=H2["positions"], cell=[5] * 3))
        assert main([str(path)]) == 0
        geometry = recorder.instances[0].calls[0][1]
        assert list(geometry.get_chemical_symbols()) == ["H", "H"]

    def test_without_ase_only_json_is_accepted(self, recorder, tmp_path,
                                               monkeypatch, capsys):
        monkeypatch.setitem(__import__("sys").modules, "ase", None)
        monkeypatch.setitem(__import__("sys").modules, "ase.io", None)
        with pytest.raises(SystemExit) as raised:
            main([str(tmp_path / "h2.xyz")])
        assert raised.value.code == 2
        assert "needs the ase extra" in capsys.readouterr().err

    def test_a_json_that_is_not_a_geometry_is_refused(self, recorder, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"numbers": [1]}))
        with pytest.raises(TypeError):
            main([str(path)])


class TestRefusals:
    def test_no_geometry_and_no_cached_problem_is_a_usage_error(self, recorder,
                                                                capsys):
        with pytest.raises(SystemExit) as raised:
            main([])
        assert raised.value.code == 2
        assert "provide geometry or a cached problem" in capsys.readouterr().err

    @pytest.mark.parametrize("argv", [["--method", "ccsd"], ["--xc", "nope"],
                                      ["--threads", "many"]])
    def test_an_unknown_choice_is_refused_before_any_work(self, recorder, argv,
                                                          h2_json, capsys):
        with pytest.raises(SystemExit) as raised:
            main([h2_json, *argv])
        assert raised.value.code == 2
        assert not recorder.instances

    def test_the_version_is_the_package_version(self, capsys):
        with pytest.raises(SystemExit) as raised:
            main(["--version"])
        assert raised.value.code == 0
        assert __version__ in capsys.readouterr().out


class TestReport:
    def test_the_report_is_json_and_is_written_to_output(self, recorder,
                                                         h2_json, tmp_path,
                                                         capsys):
        out = tmp_path / "report.json"
        assert main([h2_json, "--method", "rhf", "--output", str(out)]) == 0
        printed = json.loads(capsys.readouterr().out)
        assert printed == {"energy_ev": -1.0, "converged": True,
                           "method": "rhf"}
        assert json.loads(out.read_text()) == printed


class TestAgainstTheEntryPoint:
    """No stand-in here: the CLI must report what ``Mandacaru`` computes."""

    def _direct(self, **extra):
        options = dict(method="rhf", grid_spacing=0.5,
                       resources={"threads": 1}, **extra)
        return Mandacaru(**options)

    def test_the_energy_is_the_calculators(self, h2_json, capsys):
        assert main([h2_json, *QUICK]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["converged"] is True and report["method"] == "rhf"
        geometry = Geometry(**H2)
        with self._direct() as calc:
            expected = calc.run(geometry).in_units("eV")
        assert report["energy_ev"] == pytest.approx(expected, abs=1e-8)

    def test_the_forces_are_the_calculators_and_balance(self, h2_json, capsys):
        assert main([h2_json, *QUICK, "--forces"]) == 0
        report = json.loads(capsys.readouterr().out)
        forces = np.array(report["forces_ev_per_angstrom"])
        assert forces.shape == (2, 3)
        # Two identical atoms on the z axis: equal and opposite, none sideways.
        assert np.allclose(forces[0], -forces[1], atol=1e-6)
        assert np.allclose(forces[:, :2], 0.0, atol=1e-6)
