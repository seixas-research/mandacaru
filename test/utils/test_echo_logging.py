"""Echo report serialization, standard destinations, and parser isolation."""

import numpy as np
import pytest

from mandacaru.algorithms import QuantumEchoes
from mandacaru.core import PauliSum
from mandacaru.utils import append_performance, append_quantum_echoes, parse_output
from mandacaru.utils.logging import STDOUT


def test_echo_report_roundtrip(tmp_path, capsys):
    echo = QuantumEchoes(PauliSum({"Z": -0.4}), PauliSum({"X": 0.2}))
    results = [echo.run([1, 0], t, tau_p=0.01, steps=4) for t in [0, 0.5, 1]]
    spectrum = echo.spectrum([1, 0], num_samples=128, steps_per_sample=1)
    path = str(tmp_path / "output.txt")
    append_performance(path, wall_time_s=0.123)
    before = parse_output(path)["performance"].copy()
    options = dict(order=2, steps=4, field=[0, 0, 0.2], spectrum=spectrum,
                   spectrum_steps_per_sample=1)
    append_quantum_echoes(path, results, **options)
    append_quantum_echoes(STDOUT, results, **options)
    stdout = capsys.readouterr().out
    file_text = (tmp_path / "output.txt").read_text()
    assert stdout[stdout.index("[QUANTUM ECHOES]"):] == file_text[file_text.index("[QUANTUM ECHOES]"):]
    parsed = parse_output(path)
    assert parsed["performance"] == before  # echo keys must not leak into performance
    block = parsed["quantum_echoes"]
    assert block["order"] == 2 and block["steps"] == 4
    assert block["field_au"] == [0, 0, 0.2]
    assert block["spectrum_num_samples"] == 128
    assert block["spectrum_resolution_ha"] == pytest.approx(spectrum.resolution)
    assert len(block["samples"]) == 3
    assert len(block["spectrum"]) == 128
    assert len(block["peaks"]) == 1
    for original, row in zip(results, block["samples"]):
        assert row["response"] == pytest.approx(original.response)
        assert complex(row["amplitude_real"], row["amplitude_imag"]) == pytest.approx(original.amplitude)
        assert complex(row["correlation_real"], row["correlation_imag"]) == pytest.approx(original.correlation)
    np.testing.assert_allclose([row["magnitude"] for row in block["spectrum"]], spectrum.intensities)


def test_echo_report_without_spectrum(tmp_path):
    path = str(tmp_path / "echo.txt")
    append_quantum_echoes(path, [], order=2, steps=1)
    block = parse_output(path)["quantum_echoes"]
    assert block["samples"] == []
    assert "spectrum" not in block
