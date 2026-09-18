# -*- coding: utf-8 -*-
# file: test/test_verbose_dumps.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""``verbose_operators`` / ``verbose_hamiltonian``: the two optional JSON dumps.

The run trace reports the *size* of the operator pool and of the qubit
Hamiltonian, never their contents.  These two options write the contents to
``pool.json`` / ``hamiltonian.json`` (or to a path of your choosing) so they can
still be read -- see :mod:`carcara.utils.dumps`.
"""

import json

import numpy as np
import pytest
from ase import Atoms

from carcara import Carcara
from carcara.algorithms import ADAPTVQE, VQE
from carcara.algorithms.adapt_vqe import EXPRESSIVITY_DENSE_MAX_QUBITS
from carcara.core.mapping import PauliSum
from carcara.core.serialization import MAX_FILE_QUBITS
from carcara.utils.dumps import (HAMILTONIAN_FILE, POOL_FILE, dump_hamiltonian,
                                 dump_pool, resolve_dump_path)


@pytest.fixture(scope="module")
def h2_hamiltonian():
    """H2 at 0.74 A in the minimal FAO basis, molecular-orbital form."""
    from carcara.core import MolecularIntegrals, minimal_fao_basis
    from carcara.integrals import Grid

    nuclei = [(1.0, np.array([0.0, 0.0, -0.37])),
              (1.0, np.array([0.0, 0.0, +0.37]))]
    grid = Grid(center=[0, 0, 0], box_size=6.0, h=0.25)
    integrals = MolecularIntegrals(nuclei, minimal_fao_basis(nuclei), grid)
    return integrals.molecular_hamiltonian(mo_basis=True, n_electrons=2)


def _adapt(hamiltonian, **options):
    options.setdefault("verbose", False)
    return ADAPTVQE(hamiltonian, "fermionic", num_particles=(1, 1),
                    n_spatial_orbitals=2, profile=False,
                    max_iterations=2, gradient_tolerance=1e-4, **options)


class TestPathResolution:
    def test_true_selects_the_documented_names(self):
        assert resolve_dump_path(True, POOL_FILE) == "pool.json"
        assert resolve_dump_path(True, HAMILTONIAN_FILE) == "hamiltonian.json"

    def test_false_and_none_write_nothing(self):
        assert resolve_dump_path(False, POOL_FILE) is None
        assert resolve_dump_path(None, POOL_FILE) is None

    def test_a_path_is_used_verbatim(self, tmp_path):
        target = tmp_path / "sub" / "my_pool.json"
        assert resolve_dump_path(target, POOL_FILE) == str(target)

    @pytest.mark.parametrize("bad", [0, 1, 2.5, [], ""])
    def test_other_values_are_rejected(self, bad):
        with pytest.raises((TypeError, ValueError)):
            resolve_dump_path(bad, POOL_FILE)


class TestDefaultsWriteNothing:
    def test_no_files_without_the_options(self, h2_hamiltonian, tmp_path,
                                          monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapt = _adapt(h2_hamiltonian)
        adapt.run()
        assert list(tmp_path.iterdir()) == []


class TestPoolDump:
    def test_written_at_the_documented_name(self, h2_hamiltonian, tmp_path,
                                            monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapt = _adapt(h2_hamiltonian, verbose_operators=True)
        adapt.run()
        payload = json.loads((tmp_path / POOL_FILE).read_text())
        assert payload["pool"] == "fermionic"
        assert payload["pool_class"] == "FermionicPool"
        assert payload["pool_size"] == len(adapt._pool_ops)
        assert payload["n_qubits"] == 4
        assert payload["mapping"] == "jordan_wigner"
        assert payload["num_particles"] == [1, 1]

    def test_every_operator_carries_its_generator(self, h2_hamiltonian,
                                                  tmp_path):
        target = tmp_path / "pool.json"
        adapt = _adapt(h2_hamiltonian, verbose_operators=str(target))
        adapt.run()
        payload = json.loads(target.read_text())
        assert len(payload["operators"]) == len(adapt._pool_ops)
        for entry, op in zip(payload["operators"], adapt._pool_ops):
            assert entry["label"] == op.label and entry["kind"] == op.kind
            assert entry["support"] == list(op.support)
            terms = op.generator.simplify().terms
            assert entry["num_terms"] == len(terms)
            for term in entry["generator"]:
                coeff = complex(term["real"], term["imag"])
                assert coeff == pytest.approx(complex(terms[term["pauli"]]))

    def test_written_even_when_the_run_is_not_verbose(self, h2_hamiltonian,
                                                     tmp_path, capsys):
        """The file is the *replacement* for printing the pool, not an extra."""
        target = tmp_path / "pool.json"
        _adapt(h2_hamiltonian, verbose_operators=str(target)).run()
        assert capsys.readouterr().out == ""
        assert target.is_file()


class TestHamiltonianDump:
    def test_terms_match_the_operator(self, h2_hamiltonian, tmp_path,
                                      monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapt = _adapt(h2_hamiltonian, verbose_hamiltonian=True)
        adapt.run()
        payload = json.loads((tmp_path / HAMILTONIAN_FILE).read_text())
        terms = adapt.hamiltonian.simplify().terms
        assert payload["n_qubits"] == 4
        assert payload["num_terms"] == len(terms)
        assert payload["energy_unit"] == "Ha"      # the operator's own unit
        assert payload["mapping"] == "jordan_wigner"
        for term in payload["terms"]:
            coeff = complex(term["real"], term["imag"])
            assert coeff == pytest.approx(complex(terms[term["pauli"]]))

    def test_terms_are_ordered_by_magnitude(self, h2_hamiltonian, tmp_path):
        target = tmp_path / "h.json"
        _adapt(h2_hamiltonian, verbose_hamiltonian=str(target)).run()
        magnitudes = [abs(complex(t["real"], t["imag"]))
                      for t in json.loads(target.read_text())["terms"]]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_vqe_writes_it_too(self, h2_hamiltonian, tmp_path):
        from carcara.circuits import UCCSD

        target = tmp_path / "h.json"
        VQE(h2_hamiltonian, UCCSD(2, (1, 1)), verbose=False,
            verbose_hamiltonian=str(target)).run()
        assert json.loads(target.read_text())["n_qubits"] == 4


class TestLimits:
    def test_a_wide_register_is_skipped_with_a_warning(self, tmp_path):
        wide = PauliSum({"Z" * (MAX_FILE_QUBITS + 1): 1.0})
        target = tmp_path / "h.json"
        with pytest.warns(RuntimeWarning, match="qubit limit"):
            assert dump_hamiltonian(str(target), wide) is None
        assert not target.exists()

    def test_the_pool_obeys_the_same_limit(self, tmp_path):
        from carcara.circuits.pools import PoolOperator

        wide = PoolOperator(label="op", support=(0, 1), kind="double",
                            generator=PauliSum(
                                {"X" * (MAX_FILE_QUBITS + 1): 0.5j}))
        target = tmp_path / "pool.json"
        with pytest.warns(RuntimeWarning, match="qubit limit"):
            assert dump_pool(str(target), None, [wide]) is None
        assert not target.exists()


class TestCalculator:
    def test_carcara_forwards_both_options(self, tmp_path):
        atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, 0.74)],
                      cell=[6.0] * 3)
        atoms.center()
        atoms.calc = Carcara(method="adapt-vqe",
                             basis="FAO",
                             h=0.35,
                             verbose=False,
                             verbose_operators=str(tmp_path / "pool.json"),
                             verbose_hamiltonian=str(tmp_path / "h.json"))
        atoms.get_potential_energy()
        pool = json.loads((tmp_path / "pool.json").read_text())
        hamiltonian = json.loads((tmp_path / "h.json").read_text())
        assert pool["n_qubits"] == hamiltonian["n_qubits"] == 4
        assert pool["pool_size"] >= 1 and hamiltonian["num_terms"] >= 1


class TestExpressivityColumn:
    """The ``expr`` column, and the cost guard that decides whether to fill it.

    The estimate is ``2 * EXPRESSIVITY_SAMPLES`` state preparations per
    iteration.  On the sparse / sector backends a state is compressed and that
    is free; on the dense backend it allocates the whole ``2**n`` vector, which
    at 12 qubits measured 78 s *per iteration* against 0.07 s sparse.  So
    ``"auto"`` asks only where it is cheap.
    """

    def test_small_dense_register_is_computed_and_printed(self, h2_hamiltonian,
                                                          capsys):
        adapt = _adapt(h2_hamiltonian, verbose=True, sparse=False)
        assert adapt.n_qubits <= EXPRESSIVITY_DENSE_MAX_QUBITS
        assert adapt._expressivity_wanted("auto") is True
        adapt.run()
        rows = [line for line in capsys.readouterr().out.splitlines()
                if line.strip()[:1].isdigit()]
        assert rows and all(float(row.split()[4]) >= 0.0 for row in rows)

    def test_wide_dense_register_is_skipped(self, h2_hamiltonian, monkeypatch):
        adapt = _adapt(h2_hamiltonian, sparse=False)
        monkeypatch.setattr(adapt, "n_qubits",
                            EXPRESSIVITY_DENSE_MAX_QUBITS + 1)
        assert adapt._expressivity_wanted("auto") is False

    def test_sparse_and_sector_backends_are_exempt(self, h2_hamiltonian,
                                                   monkeypatch):
        adapt = _adapt(h2_hamiltonian, sparse=True)
        monkeypatch.setattr(adapt, "n_qubits",
                            EXPRESSIVITY_DENSE_MAX_QUBITS + 10)
        assert adapt._expressivity_wanted("auto") is True

    def test_true_and_false_override_the_guard(self, h2_hamiltonian,
                                               monkeypatch):
        adapt = _adapt(h2_hamiltonian, sparse=False)
        monkeypatch.setattr(adapt, "n_qubits",
                            EXPRESSIVITY_DENSE_MAX_QUBITS + 1)
        assert adapt._expressivity_wanted(True) is True
        assert adapt._expressivity_wanted(False) is False

    def test_an_unknown_value_is_rejected(self, h2_hamiltonian):
        with pytest.raises(ValueError, match="log_expressivity"):
            _adapt(h2_hamiltonian)._expressivity_wanted("sometimes")

    def test_a_skipped_column_reads_as_a_dash(self, h2_hamiltonian, capsys):
        adapt = _adapt(h2_hamiltonian, verbose=True)
        adapt.run(log_expressivity=False)
        rows = [line for line in capsys.readouterr().out.splitlines()
                if line.strip()[:1].isdigit()]
        assert rows and all(row.split()[4] == "-" for row in rows)
