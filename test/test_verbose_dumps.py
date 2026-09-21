# -*- coding: utf-8 -*-
# file: test/test_verbose_dumps.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""``verbose_operators`` / ``verbose_hamiltonian``: the two optional JSON dumps.

The run trace reports the *size* of the operator pool and of the qubit
Hamiltonian, never their contents.  These two options write the contents to
``pool.json`` / ``hamiltonian.json`` (or to a path of your choosing) so they can
still be read -- see :mod:`mandacaru.utils.dumps`.
"""

import json

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.core.mapping import PauliSum
from mandacaru.core.serialization import MAX_FILE_QUBITS
from mandacaru.utils.dumps import (HAMILTONIAN_FILE, POOL_FILE, dump_hamiltonian,
                                   dump_pool, resolve_dump_path)


@pytest.fixture(scope="module")
def h2_hamiltonian():
    """H2 at 0.74 A in the minimal FAO basis, molecular-orbital form."""
    from mandacaru.core import MolecularIntegrals, minimal_fao_basis
    from mandacaru.integrals import Grid

    nuclei = [(1.0, np.array([0.0, 0.0, -0.37])),
              (1.0, np.array([0.0, 0.0, +0.37]))]
    grid = Grid(center=[0, 0, 0], box_size=6.0, h=0.25)
    integrals = MolecularIntegrals(nuclei, minimal_fao_basis(nuclei), grid)
    return integrals.molecular_hamiltonian(mo_basis=True, n_electrons=2)


def _adapt(hamiltonian, **options):
    options.setdefault("trace", False)
    return Mandacaru(method="adapt-vqe", hamiltonian=hamiltonian,
                     pool="fermionic", num_particles=(1, 1),
                     n_spatial_orbitals=2, profile=False, max_iterations=2,
                     gradient_tolerance=1e-4, **options)


class TestPathResolution:
    def test_true_selects_the_documented_names(self):
        assert resolve_dump_path(True, POOL_FILE) == "pool.json"
        # Not "hamiltonian.json": that is what `save_hamiltonian=True,
        # hamiltonian_format="json"` writes, and the two documents have
        # incompatible schemas -- writing both left the cache unloadable.
        assert resolve_dump_path(True, HAMILTONIAN_FILE) \
            == "hamiltonian.inspect.json"

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
        from mandacaru.circuits import UCCSD

        target = tmp_path / "h.json"
        Mandacaru(method="vqe", hamiltonian=h2_hamiltonian,
                  ansatz=UCCSD(2, (1, 1)), trace=False,
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
        from mandacaru.circuits.pools import PoolOperator

        wide = PoolOperator(label="op", support=(0, 1), kind="double",
                            generator=PauliSum(
                                {"X" * (MAX_FILE_QUBITS + 1): 0.5j}))
        target = tmp_path / "pool.json"
        with pytest.warns(RuntimeWarning, match="qubit limit"):
            assert dump_pool(str(target), None, [wide]) is None
        assert not target.exists()


class TestCalculator:
    def test_mandacaru_forwards_both_options(self, tmp_path):
        atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, 0.74)],
                      cell=[6.0] * 3)
        atoms.center()
        atoms.calc = Mandacaru(method="adapt-vqe",
                               basis="FAO",
                               h=0.35,
                               verbose_operators=str(tmp_path / "pool.json"),
                               verbose_hamiltonian=str(tmp_path / "h.json"))
        atoms.get_potential_energy()
        pool = json.loads((tmp_path / "pool.json").read_text())
        hamiltonian = json.loads((tmp_path / "h.json").read_text())
        assert pool["n_qubits"] == hamiltonian["n_qubits"] == 4
        assert pool["pool_size"] >= 1 and hamiltonian["num_terms"] >= 1


def _column(out, name):
    """Values of one column of the iteration table, read via its heading.

    The optional ``expr`` column exists only when the expressivity was asked
    for, so a fixed field index would drift.
    """
    lines = out.splitlines()
    heading = next(line for line in lines if line.split()[:1] == ["iter"])
    columns = heading.replace("energy (eV)", "energy").split()
    index = lines.index(heading)
    rows = [line.split() for line in lines[index + 2:]
            if line.strip() and line.split()[0].isdigit()]
    if name not in columns:
        return None                      # the column is not in this table
    return [dict(zip(columns, row))[name] for row in rows]


class TestExpressivityColumn:
    """The ``expr`` column: opt-in, and absent rather than blank when off.

    The estimate is ``2 * EXPRESSIVITY_SAMPLES`` state preparations per
    iteration, each applying every operator in the ansatz, so its cost is
    linear in the ansatz and quadratic over a run.  It is a diagnostic, not a
    result, so it is computed only when asked for.
    """

    def test_off_by_default(self, h2_hamiltonian, capsys):
        adapt = _adapt(h2_hamiltonian, trace=True)
        adapt.run()
        out = capsys.readouterr().out
        assert _column(out, "expr") is None
        heading = next(l for l in out.splitlines()
                       if l.split()[:1] == ["iter"])
        assert "expr" not in heading.split()

    def test_true_computes_and_prints_it(self, h2_hamiltonian, capsys):
        adapt = _adapt(h2_hamiltonian, trace=True)
        adapt.run(log_expressivity=True)
        values = _column(capsys.readouterr().out, "expr")
        assert values and all(float(v) >= 0.0 for v in values)

    def test_it_is_a_plain_boolean(self, h2_hamiltonian):
        adapt = _adapt(h2_hamiltonian)
        assert adapt._expressivity_wanted(True) is True
        assert adapt._expressivity_wanted(False) is False

    @pytest.mark.parametrize("bad", ["auto", "sometimes", 1, 0, None])
    def test_anything_else_is_rejected(self, h2_hamiltonian, bad):
        """No automatic middle setting: the caller decides once."""
        with pytest.raises(ValueError, match="log_expressivity"):
            _adapt(h2_hamiltonian)._expressivity_wanted(bad)

    def test_a_skipped_column_is_absent_not_blank(self, h2_hamiltonian,
                                                  capsys):
        """A column of nothing but "-" is worse than no column.

        Whether the expressivity is computed is a per-run setting, so when it
        is off the table simply does not carry the column.
        """
        adapt = _adapt(h2_hamiltonian, trace=True)
        adapt.run(log_expressivity=False)
        out = capsys.readouterr().out
        assert _column(out, "expr") is None
        heading = next(l for l in out.splitlines()
                       if l.split()[:1] == ["iter"])
        assert "expr" not in heading.split()
        # The rest of the table is unaffected.
        assert _column(out, "|grad|")
