# -*- coding: utf-8 -*-
# file: test/algorithms/test_two_body_representation.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""``two_body={"representation": "double-factorized", "truncation": tau}`` through
``Mandacaru``: the active-space Hamiltonian is built from the rotated Cholesky vectors
and their eigen decompositions, never from the ``n^4`` tensor.

The independent reference is the dense route (the tensor and its fermion monomials):
untruncated, both give the same qubit Hamiltonian and the same ADAPT-VQE energy; the
readers of the dense tensor refuse the factorized model by name, and the run log states
the truncation in its energy unit. The operator equality and the energy bound at stated
truncations are pinned in Rust (``crates/mandacaru-core/tests/molecular_double_factorized.rs``).
"""

from __future__ import annotations

import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.utils.logging import parse_output

H4 = dict(symbols="H4", positions=[[3, 3, 2.2 + 0.9 * i] for i in range(4)],
          cell=[6.0, 6.0, 7.5], pbc=False)
OPTIONS = dict(method="adapt-vqe", basis={"name": "HAO", "virtual_orbitals": 1},
               grid_spacing=0.4, active_space={"orbitals": 4}, pool="fermionic",
               max_iterations=4, trace=False)


def run(tmp_path, name, **two_body):
    atoms = Atoms(**H4)
    out = str(tmp_path / f"{name}.txt")
    atoms.calc = Mandacaru(**OPTIONS, txt=out,
                           two_body={"method": "cholesky", **two_body} if two_body else None)
    energy = atoms.get_potential_energy()
    return atoms.calc, energy, out


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("two_body")
    return {
        "dense": run(tmp_path, "dense"),
        "exact": run(tmp_path, "exact", representation="double-factorized", truncation=0.0),
        "truncated": run(tmp_path, "truncated", representation="double-factorized",
                         truncation=0.05),
    }


def terms(calc):
    return {label: complex(re, im) for label, re, im in calc.problem.hamiltonian_terms}


class TestTheFactorizedActiveSpace:
    def test_untruncated_it_is_the_dense_hamiltonian_and_energy(self, runs):
        dense, exact = terms(runs["dense"][0]), terms(runs["exact"][0])
        assert dense.keys() == exact.keys()
        assert max(abs(dense[k] - exact[k]) for k in dense) < 1e-10
        assert runs["exact"][1] == pytest.approx(runs["dense"][1], abs=1e-8)

    def test_the_truncation_moves_no_energy_by_more_than_it_states(self, runs):
        calc, energy, _ = runs["truncated"]
        report = calc.result.hamiltonian.two_body_report
        assert report["representation"] == "double-factorized"
        assert report["kept"] < report["eigenpairs"]
        from mandacaru import _native
        assert _native.energy_from_hartree(report["error_bound_ha"], "eV") <= 0.05 + 1e-12
        # The variational energy of a truncated Hamiltonian, from the same ansatz growth:
        # inside the stated bound of the dense one (the bound holds for every state).
        assert abs(energy - runs["dense"][1]) <= 0.05

    def test_readers_of_the_dense_tensor_refuse_it_by_name(self, runs):
        result = runs["exact"][0].result
        with pytest.raises(NotImplementedError, match="double-factorized"):
            result.hamiltonian.two_body
        with pytest.raises(NotImplementedError, match="double-factorized"):
            result.mp2_natural_orbitals()

    def test_the_run_log_states_the_truncation_in_its_energy_unit(self, runs):
        line = parse_output(runs["truncated"][2])["electrons"]["two-body integrals"]
        assert "; double-factorized active-space tensor, truncation 5.000e-02 eV, " in line
        assert "eigenpairs in" in line and "energy change bound" in line
        dense = parse_output(runs["dense"][2])["electrons"].get("two-body integrals")
        assert dense is None


class TestASavedFactorizedHamiltonian:
    """The owner (2026-11-10): "Record it in the file". The representation, the truncation
    and the bound reached travel with the Hamiltonian, and a run from the file states them."""

    @pytest.mark.parametrize("extension", ["json", "parquet"])
    def test_the_file_records_the_truncation_and_the_run_from_it_states_it(
            self, runs, tmp_path, extension):
        from mandacaru import _native
        calc = runs["truncated"][0]
        path = str(tmp_path / f"h4.{extension}")
        calc.result.save_hamiltonian(path)
        record = _native.snapshot_two_body(
            calc.context.read_snapshot(path, "mandacaru-qubit-hamiltonian"))
        report = calc.result.hamiltonian.two_body_report
        assert record["representation"] == "double-factorized"
        assert record["truncation_ev"] == pytest.approx(0.05, rel=1e-12)
        assert record["error_bound_ev"] == pytest.approx(
            _native.energy_from_hartree(report["error_bound_ha"], "eV"), rel=1e-12)
        out = str(tmp_path / f"loaded_{extension}.txt")
        Mandacaru(method="adapt-vqe", load_hamiltonian=path, pool="fermionic",
                  max_iterations=1, trace=False, txt=out).run()
        line = parse_output(out)["electrons"]["two-body integrals"]
        assert line.startswith("double-factorized active-space tensor (recorded in the loaded "
                               "Hamiltonian), truncation 5.000e-02 eV, energy change bound ")

    def test_a_dense_file_records_nothing(self, runs, tmp_path):
        from mandacaru import _native
        calc = runs["dense"][0]
        path = str(tmp_path / "h4.json")
        calc.result.save_hamiltonian(path)
        assert _native.snapshot_two_body(
            calc.context.read_snapshot(path, "mandacaru-qubit-hamiltonian")) is None
