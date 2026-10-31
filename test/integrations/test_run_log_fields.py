# -*- coding: utf-8 -*-
# file: test/integrations/test_run_log_fields.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The field dictionaries of the run log, assembled from stand-ins for the
native objects: no calculation runs here.  What a field *says* about a real
run is pinned by ``test/utils/test_logging.py``."""

from types import SimpleNamespace

import numpy as np
import pytest

from mandacaru.integrations import run_log_fields as fields
from mandacaru.units import BOHR_TO_ANGSTROM, HARTREE_TO_EV


def grid(shape, step):
    return SimpleNamespace(shape=list(shape), step_bohr=np.asarray(step, dtype=float))


def driver(**options):
    values = dict(
        _two_body_requested=False, _cholesky_tolerance=1e-10, _grid_spacing=0.35,
        _kinetic="spectral", _charge=0, _kpts_spec={"size": (1, 1, 1), "gamma": True},
        _mapping="jordan_wigner", _atomic_units=False, _scf_options={},
        _resume_lineage=None, _resume_path=None, _shots=0, _sparse=False, _sector=True,
        n_qubits=4, integrals=SimpleNamespace(grid=grid((18, 18, 18), np.eye(3) * 0.6614)))
    values.update(options)
    return SimpleNamespace(**values)


def result(problem="problem", num_particles=(1, 1)):
    return SimpleNamespace(_problem=problem, num_particles=num_particles, n_spatial_orbitals=2)


GEOMETRY = object()


class TestTheGrid:
    def test_an_orthogonal_grid_states_points_and_spacing_per_axis(self):
        text = fields.grid_text(grid((18, 20, 22), np.diag([1.0, 2.0, 3.0])))
        assert text == (
            f"18 x 20 x 22 (spacing {BOHR_TO_ANGSTROM:.4f} x "
            f"{2 * BOHR_TO_ANGSTROM:.4f} x {3 * BOHR_TO_ANGSTROM:.4f} Angstrom)")

    def test_a_skewed_grid_does_not_pass_its_diagonal_off_as_the_spacing(self):
        step = np.diag([1.0, 1.0, 1.0])
        step[0, 1] = 0.5
        text = fields.grid_text(grid((4, 4, 4), step))
        assert text == "4 x 4 x 4 (non-orthogonal step vectors)"

    def test_the_gamma_mesh_is_named_and_another_mesh_is_not_called_gamma(self):
        assert fields.kpoints_text(driver()) == "Gamma (1x1x1 Monkhorst-Pack)"
        mesh = driver(_kpts_spec={"size": (2, 2, 1), "gamma": False})
        assert fields.kpoints_text(mesh) == "2x2x1 Monkhorst-Pack"


class TestTheTwoBodyLine:
    def test_nothing_is_said_unless_the_factorization_was_requested(self):
        assert fields.two_body_text(driver()) is None

    def test_a_requested_factorization_states_tolerance_rank_and_residual(self, monkeypatch):
        monkeypatch.setattr(fields._native, "integrals_report", lambda _integrals: dict(
            cholesky_rank=3, cholesky_residual=2.5e-7, n_orbitals=2))
        text = fields.two_body_text(driver(_two_body_requested=True, _cholesky_tolerance=1e-6))
        assert text == ("Cholesky factors, tolerance 1.000e-06 Ha, rank 3 (basis functions 2), "
                        "largest residual diagonal 2.500e-07 Ha")

    def test_integrals_that_were_never_factored_say_so(self, monkeypatch):
        monkeypatch.setattr(fields._native, "integrals_report", lambda _integrals: dict(
            cholesky_rank=None, cholesky_residual=None, n_orbitals=2))
        text = fields.two_body_text(driver(_two_body_requested=True))
        assert text.endswith("not built") and "rank" not in text


class TestTheElectronsBlock:
    FULL = ("grid spacing", "grid points", "kinetic operator", "two-body integrals", "k-points",
            "charge", "spin-polarized", "reference state", "Z2 tapering", "mapping",
            "Hamiltonian", "spatial orbitals", "electrons (alpha, beta)", "qubits")

    def test_a_quantum_run_from_a_geometry_writes_the_thirteen_fields_in_order(self, monkeypatch):
        monkeypatch.setattr(fields._native, "problem_term_count", lambda _problem: 15)
        block = fields.electrons(driver(), result(), GEOMETRY)
        assert tuple(key for key, value in block.items() if value is not None) == tuple(
            key for key in self.FULL if key != "two-body integrals")
        assert block["Hamiltonian"] == "15 Pauli terms"
        assert block["mapping"] == "Jordan-Wigner" and block["qubits"] == "4"
        assert block["electrons (alpha, beta)"] == "(1, 1)"

    def test_the_two_body_line_follows_the_kinetic_operator(self, monkeypatch):
        monkeypatch.setattr(fields._native, "problem_term_count", lambda _problem: 15)
        monkeypatch.setattr(fields, "two_body_text", lambda _driver: "Cholesky factors")
        keys = list(fields.electrons(driver(), result(), GEOMETRY))
        assert keys[keys.index("kinetic operator") + 1] == "two-body integrals"

    def test_a_classical_run_describes_no_register_it_did_not_build(self):
        block = fields.electrons(driver(), result(problem=None), GEOMETRY)
        for key in ("mapping", "reference state", "Z2 tapering", "Hamiltonian", "qubits"):
            assert key not in block
        assert block["spatial orbitals"] == "2"

    def test_a_hamiltonian_without_a_geometry_has_no_discretization(self, monkeypatch):
        monkeypatch.setattr(fields._native, "problem_term_count", lambda _problem: 7)
        block = fields.electrons(driver(integrals=None), result(), None)
        for key in ("grid spacing", "grid points", "kinetic operator", "k-points", "charge"):
            assert key not in block
        assert block["Hamiltonian"] == "7 Pauli terms"

    @pytest.mark.parametrize("counts, text", [((1, 1), "False (multiplicity 1)"),
                                              ((2, 1), "True (multiplicity 2)"),
                                              ((3, 1), "True (multiplicity 3)")])
    def test_the_spin_state_is_read_from_the_particle_numbers(self, counts, text):
        block = fields.electrons(driver(), result(problem=None, num_particles=counts), GEOMETRY)
        assert block["spin-polarized"] == text

    def test_an_unlisted_mapping_is_written_as_it_is(self, monkeypatch):
        monkeypatch.setattr(fields._native, "problem_term_count", lambda _problem: 1)
        block = fields.electrons(driver(_mapping="ternary_tree"), result(), None)
        assert block["mapping"] == "ternary_tree"


class TestTheSCFBlocks:
    @staticmethod
    def scf_driver(**options):
        mean_field = lambda method, num_particles, *, max_iterations=300, tolerance=1e-9, \
            diis=True, guesses=3: None
        return driver(integrals=SimpleNamespace(mean_field=mean_field, kohn_sham=mean_field),
                      **options)

    def test_the_defaults_are_the_ones_the_native_method_declares(self):
        block = fields.scf_setup(self.scf_driver(), {"functional": None, "method": "rhf"})
        assert block["scf_method"] == "restricted Hartree-Fock (closed shell)"
        assert block["scf_max_iterations"] == 300
        assert block["scf_convergence_Hartree"].startswith("1e-09 ")
        assert block["scf_acceleration"] == "DIIS"

    def test_what_the_user_set_overrides_the_declared_default(self):
        options = {"max_iterations": 12, "tolerance": 1e-6, "diis": False}
        block = fields.scf_setup(self.scf_driver(_scf_options=options),
                                 {"functional": None, "method": "uhf"})
        assert block["scf_max_iterations"] == 12
        assert block["scf_convergence_Hartree"].startswith("1e-06 ")
        assert block["scf_acceleration"] == "none"

    def test_a_functional_names_the_kohn_sham_method(self):
        block = fields.scf_setup(self.scf_driver(), {"functional": "pbe", "method": "rks"})
        assert block["scf_method"] == "Kohn-Sham DFT (pbe)"

    def test_the_history_is_converted_to_the_log_unit_and_carries_no_clock(self):
        history = [(1, -1.0, None, 0.5), (2, -1.5, -0.5, 1e-9)]
        rows = fields.scf_rows(self.scf_driver(), {"history": history})
        assert [row["iter"] for row in rows] == [1, 2]
        assert rows[0]["dE"] is None and rows[1]["dE"] == pytest.approx(-0.5 * HARTREE_TO_EV)
        assert rows[1]["energy"] == pytest.approx(-1.5 * HARTREE_TO_EV)
        assert all("clock_time" not in row for row in rows)
        hartree = fields.scf_rows(self.scf_driver(_atomic_units=True), {"history": history})
        assert hartree[1]["energy"] == -1.5


class TestTheLineage:
    def test_a_fresh_run_has_none(self):
        assert fields.lineage(driver()) == ({}, None)

    def test_a_resumed_run_states_where_it_restarted_in_the_log_unit(self):
        resumed = dict(method="adapt-vqe", energy_ha=-1.0, num_operators=2, num_parameters=2,
                       status="{}")
        block, start = fields.lineage(driver(_resume_lineage=resumed, _resume_path="state.json"))
        assert block["resumed_from"] == "state.json" and block["restored_operators"] == 2
        assert block["restored_parameters"] == 2
        assert block["restored_energy_eV"] == f"{-HARTREE_TO_EV:.10f}"
        assert block["resume_same_hamiltonian"] is True
        assert start == pytest.approx(-HARTREE_TO_EV)
        assert "initial_ansatz" not in block


class TestTheStateVectorBackend:
    problem = SimpleNamespace(num_qubits=4, dimension=4)

    def test_a_sector_run_states_the_sector_dimension(self):
        assert fields.backend_text(driver(), self.problem) == \
            "particle-number sector, 4 states of 2^4"

    def test_the_full_register_says_amplitudes(self):
        text = fields.backend_text(driver(_sector=False), SimpleNamespace(num_qubits=4, dimension=16))
        assert text == "full register, 16 amplitudes of 2^4"

    def test_a_measured_run_holds_no_state_vector(self):
        assert fields.backend_text(driver(_shots=100), self.problem).startswith("none (shots > 0")
