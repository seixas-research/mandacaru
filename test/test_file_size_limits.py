# -*- coding: utf-8 -*-
# file: test/test_file_size_limits.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Large registers: the Hamiltonian cache stops at 50 qubits, and output.txt
leaves out the operator pool and the selected operator above 20."""

from types import SimpleNamespace

import pytest

from carcara.algorithms.base import VariationalDriver
from carcara.core.mapping import PauliSum
from carcara.core.serialization import MAX_FILE_QUBITS, save_hamiltonian
from carcara.utils.logging import (DETAILED_LOG_MAX_QUBITS, AdaptOutputLogger,
                                   parse_output)


def test_hamiltonian_file_refused_above_the_limit(tmp_path):
    save_hamiltonian(tmp_path / "ok.json", PauliSum({"Z" * MAX_FILE_QUBITS: 1.0}))
    big = tmp_path / "big.json"
    with pytest.raises(ValueError, match="MAX_FILE_QUBITS"):
        save_hamiltonian(big, PauliSum({"Z" * (MAX_FILE_QUBITS + 1): 1.0}))
    assert not big.exists()


def test_driver_skips_the_hamiltonian_file_and_keeps_running(tmp_path):
    cache = tmp_path / "h.parquet"
    driver = SimpleNamespace(n_qubits=MAX_FILE_QUBITS + 1, _save_path=str(cache),
                             load_hamiltonian=None)
    with pytest.warns(RuntimeWarning, match="qubit limit"):
        assert VariationalDriver._maybe_save_hamiltonian(driver) is None
    assert not cache.exists()


@pytest.mark.parametrize("n_qubits", [DETAILED_LOG_MAX_QUBITS,
                                      DETAILED_LOG_MAX_QUBITS + 1, 60])
def test_output_log_omits_the_pool_above_twenty_qubits(tmp_path, n_qubits):
    pool = [SimpleNamespace(label=f"op{i}", kind="double",
                            generator=PauliSum({"X" * n_qubits: 0.5j}))
            for i in range(3)]
    path = tmp_path / "output.txt"
    with AdaptOutputLogger(str(path), n_qubits=n_qubits) as log:
        log.write_metadata()
        log.write_iteration(1, pool, [0.1, 0.3, 0.2], 1, None, -1.0, 1)
        log.write_summary(True, -1.0, 1, operator_sequence=["op1"])
    text = path.read_text()
    iteration = parse_output(str(path))["iterations"][0]
    detailed = n_qubits <= DETAILED_LOG_MAX_QUBITS
    assert ("operator_pool:" in text) == detailed
    assert ("selected_operator:" in text) == detailed
    assert ("operator_sequence:" in text) == detailed
    assert ("X" * n_qubits in text) == detailed
    assert iteration["energy"] == -1.0 and "pool_size: 3" in text
    if not detailed:
        assert iteration["max_gradient"] == pytest.approx(0.3)
        assert "operator_details: omitted" in text
