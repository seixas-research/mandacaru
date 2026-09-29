# -*- coding: utf-8 -*-
# file: test_vasqa.py

"""VASQA end to end: the MCAS chain driving VQE on H2 in a minimal basis.

The chain itself is verified without a Hamiltonian in ``test_mcas.py``; this
file checks what the energies add -- that the reported state is the one the
calculator will differentiate, that a zero-angle insertion leaves the state
alone, that the sector ground state is reached and never undershot, and that
the options behave as documented.  Reached only through
``Mandacaru(method="vasqa")``.
"""

import math

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.algorithms import VASQAResult
from mandacaru.core import MolecularIntegrals, minimal_hao_basis
from mandacaru.integrals import Grid
from mandacaru.units import HARTREE_TO_EV

POOLS = ("fermionic", "qubit", "qeb", "ceo", "ceo-ovp")


@pytest.fixture(scope="module")
def h2_hamiltonian():
    R = 0.74
    nuclei = [(1.0, np.array([0.0, 0.0, -R / 2])),
              (1.0, np.array([0.0, 0.0, +R / 2]))]
    grid = Grid(center=[0.0, 0.0, 0.0], box_size=5.0, h=0.25)
    mints = MolecularIntegrals(nuclei, minimal_hao_basis(nuclei), grid)
    return mints.molecular_hamiltonian(mo_basis=True, n_electrons=2)


@pytest.fixture(scope="module")
def sector_ground_ev(h2_hamiltonian):
    """Lowest eigenvalue with two electrons, in eV."""
    m = h2_hamiltonian.map_to_qubits("jordan_wigner").to_matrix()
    m = 0.5 * (m + m.conj().T)
    two = [i for i in range(16) if bin(i).count("1") == 2]
    return float(np.linalg.eigvalsh(m[np.ix_(two, two)])[0]) * HARTREE_TO_EV


def _vasqa(h2_hamiltonian, **options):
    options = {"pool": "fermionic", "max_steps": 30, "max_length": 5,
               "seed": 7, "profile": False, "trace": False, **options}
    return Mandacaru(method="vasqa", hamiltonian=h2_hamiltonian,
                     num_particles=(1, 1), n_spatial_orbitals=2, **options)


class TestTheSearch:
    @pytest.mark.parametrize("pool", POOLS)
    def test_every_pool_reaches_the_sector_ground_state(
            self, h2_hamiltonian, sector_ground_ev, pool):
        calc = _vasqa(h2_hamiltonian, pool=pool)
        result = calc.run()
        assert isinstance(result, VASQAResult)
        assert result.optimal_energy == pytest.approx(sector_ground_ev,
                                                      abs=1e-6)

    def test_no_energy_undershoots_the_sector_ground_state(
            self, h2_hamiltonian, sector_ground_ev):
        result = _vasqa(h2_hamiltonian, pool="qubit").run()
        energies = [s.proposed_energy for s in result.steps]
        assert min(energies) >= sector_ground_ev - 1e-9

    def test_the_reported_state_is_the_solvers_ansatz(self, h2_hamiltonian):
        """Forces and densities read ``solver.ansatz`` at these parameters."""
        calc = _vasqa(h2_hamiltonian, pool="qubit")
        result = calc.run()
        solver = calc.solver
        assert len(result.operators) == len(result.optimal_parameters)
        assert [op.label for op in solver.ansatz.operators] == result.operators
        assert result.operators == [solver._pool_ops[i].label
                                    for i in result.architecture]
        energy = solver.energy(solver.ansatz.state(result.optimal_parameters))
        assert energy * HARTREE_TO_EV == pytest.approx(result.optimal_energy,
                                                       abs=1e-10)

    def test_the_trajectory_is_complete(self, h2_hamiltonian):
        result = _vasqa(h2_hamiltonian, max_steps=12).run()
        assert [s.step for s in result.steps] == list(range(1, 13))
        taken = sum(a for a, _ in result.acceptance_by_move.values())
        tried = sum(t for _, t in result.acceptance_by_move.values())
        assert tried == 12
        assert taken == sum(s.accepted for s in result.steps)
        for before, after in zip(result.steps, result.steps[1:]):
            if not after.accepted:
                # A rejection repeats the current state.
                assert after.current == before.current
        for s in result.steps:
            assert 0 <= len(s.proposed) <= 5
            if s.accepted:
                assert s.current == s.proposed

    def test_the_first_move_can_only_insert(self, h2_hamiltonian):
        result = _vasqa(h2_hamiltonian, max_steps=1).run()
        assert result.steps[0].move == "insert"


class TestTheEvaluator:
    def test_a_zero_angle_insertion_preserves_the_state(self, h2_hamiltonian):
        solver = _vasqa(h2_hamiltonian, pool="qubit").solver
        architecture = (0, 3, 5)
        theta = np.array([0.3, -0.2, 0.1])
        psi = solver._ansatz_for(architecture).state(theta)
        from mandacaru.algorithms.mcas import Action
        for slot in range(len(architecture) + 1):
            action = Action("insert", slot, operator=7)
            grown = solver._ansatz_for(action.apply(architecture))
            np.testing.assert_allclose(grown.state(action.transfer(theta)),
                                       psi, atol=1e-12)

    def test_the_empty_architecture_is_the_reference(self, h2_hamiltonian):
        calc = _vasqa(h2_hamiltonian, max_steps=0)
        result = calc.run()
        assert result.operators == []
        assert result.optimal_energy == result.reference_energy
        assert result.num_evaluations == 0

    def test_an_empty_pool_returns_the_reference(self):
        """One electron in one orbital: nothing to correlate, as in ADAPT."""
        from mandacaru.core.mapping import Fermion
        h = Fermion.from_integrals(np.diag([-0.5, -0.5]))
        calc = Mandacaru(method="vasqa", hamiltonian=h, num_particles=(1, 0),
                         n_spatial_orbitals=1, profile=False, trace=False)
        result = calc.run()
        assert result.steps == [] and result.operators == []
        assert result.optimal_energy == result.reference_energy

    def test_without_warm_start_the_cost_is_a_function_of_the_architecture(
            self, h2_hamiltonian):
        result = _vasqa(h2_hamiltonian, pool="qubit", warm_start=False,
                        max_steps=40).run()
        seen: dict = {}
        for s in result.steps:
            seen.setdefault(s.proposed, s.proposed_energy)
            assert s.proposed_energy == seen[s.proposed]

    def test_a_large_length_penalty_keeps_the_reference(self, h2_hamiltonian):
        result = _vasqa(h2_hamiltonian, length_penalty=10.0).run()
        assert result.operators == []
        assert result.optimal_cost == result.reference_energy
        assert result.best_energy < result.reference_energy

    def test_the_penalty_is_part_of_the_cost(self, h2_hamiltonian):
        result = _vasqa(h2_hamiltonian, length_penalty=0.01).run()
        for s in result.steps:
            assert s.proposed_cost == pytest.approx(
                s.proposed_energy + 0.01 * len(s.proposed))

    def test_at_zero_temperature_the_cost_never_rises(self, h2_hamiltonian):
        result = _vasqa(h2_hamiltonian, pool="qubit", temperature=0.0,
                        length_penalty=0.001, max_steps=30).run()
        costs = [s.current_cost for s in result.steps]
        assert all(b <= a + 1e-12 for a, b in zip(costs, costs[1:]))

    def test_atomic_units_report_hartree(self, h2_hamiltonian,
                                         sector_ground_ev):
        result = _vasqa(h2_hamiltonian, atomic_units=True,
                        temperature={"initial": 0.004, "final": 4e-5}).run()
        assert result.energy_unit == "Ha"
        assert result.in_units("eV") == pytest.approx(sector_ground_ev,
                                                      abs=1e-6)

    def test_the_default_temperature_is_in_ev_in_either_unit(
            self, h2_hamiltonian):
        ev = _vasqa(h2_hamiltonian, max_steps=3).run()
        ha = _vasqa(h2_hamiltonian, max_steps=3, atomic_units=True).run()
        assert ev.steps[0].temperature == pytest.approx(0.1)
        assert ha.steps[0].temperature * HARTREE_TO_EV == pytest.approx(0.1)
        assert ha.steps[-1].temperature * HARTREE_TO_EV == pytest.approx(1e-3)


class TestTheGradientProposal:
    def test_it_is_the_default_and_favors_the_steep_operator(
            self, h2_hamiltonian):
        """At the RHF reference both singles have zero gradient (Brillouin)
        and the double does not: tau = 0.2 proposes it with probability
        e^5 / (e^5 + 2) ~ 0.987."""
        firsts = [_vasqa(h2_hamiltonian, max_steps=1, seed=seed).run()
                  .steps[0].action for seed in range(10)]
        doubles = sum(action.startswith("insert(D(") for action in firsts)
        assert doubles >= 8, firsts

    def test_the_uniform_proposal_draws_the_singles_as_often(
            self, h2_hamiltonian):
        firsts = [_vasqa(h2_hamiltonian, max_steps=1, seed=seed,
                         proposal="uniform").run().steps[0].action
                  for seed in range(30)]
        singles = sum(action.startswith("insert(S(") for action in firsts)
        assert 10 <= singles <= 30        # 2/3 expected, ~20

    def test_one_screening_per_new_state(self, h2_hamiltonian):
        gradient = _vasqa(h2_hamiltonian, max_steps=12).run()
        assert gradient.num_screenings == 13      # the reference + 12
        uniform = _vasqa(h2_hamiltonian, max_steps=12,
                         proposal="uniform").run()
        assert uniform.num_screenings == 0

    def test_memoized_states_are_screened_once(self, h2_hamiltonian):
        result = _vasqa(h2_hamiltonian, max_steps=40, warm_start=False).run()
        assert result.num_screenings == result.num_architectures

    def test_the_setup_block_names_it(self, h2_hamiltonian, tmp_path):
        from mandacaru.utils.logging import parse_output, reset_log
        path = tmp_path / "output.txt"
        reset_log(str(path))
        _vasqa(h2_hamiltonian, max_steps=3, txt=str(path),
               proposal_temperature=0.5).run()
        log = parse_output(str(path))
        assert log["setup"]["proposal"].startswith(
            "gradient softmax (tau 0.5 of max |grad|)")
        assert int(log["summary"]["gradient_screenings"]) == 4


class TestReproducibility:
    def test_the_same_seed_gives_the_same_chain(self, h2_hamiltonian):
        a = _vasqa(h2_hamiltonian, pool="qubit", max_steps=15).run()
        b = _vasqa(h2_hamiltonian, pool="qubit", max_steps=15).run()
        assert [(s.action, s.accepted) for s in a.steps] == \
            [(s.action, s.accepted) for s in b.steps]
        assert a.optimal_energy == b.optimal_energy

    def test_another_seed_gives_another_chain(self, h2_hamiltonian):
        a = _vasqa(h2_hamiltonian, pool="qubit", max_steps=15, seed=1).run()
        b = _vasqa(h2_hamiltonian, pool="qubit", max_steps=15, seed=2).run()
        assert [s.action for s in a.steps] != [s.action for s in b.steps]


class TestOptions:
    @pytest.mark.parametrize("options, error", [
        ({"min_length": 1}, "min_length"),
        ({"max_length": 0}, "min_length < max_length"),
        ({"max_steps": -1}, "max_steps"),
        ({"temperature": -1.0}, "temperature"),
        ({"length_penalty": -0.1}, "length_penalty"),
        ({"warm_start": "yes"}, "warm_start"),
        ({"move_weights": {"insert": 1.0}}, "insert and delete"),
        ({"proposal": "learned"}, "unknown proposal"),
        ({"proposal_temperature": 0.0}, "softmax temperature"),
    ])
    def test_bad_options_are_refused_by_the_constructor(self, options, error):
        with pytest.raises(ValueError, match=error):
            Mandacaru(method="vasqa", **options)

    @pytest.mark.parametrize("option", ["checkpoint", "resume"])
    def test_checkpoints_are_refused(self, option, tmp_path):
        with pytest.raises(NotImplementedError):
            Mandacaru(method="vasqa", **{option: str(tmp_path / "x")})

    @pytest.mark.parametrize("option", [{"tetris": True},
                                        {"gradient": "analytic"},
                                        {"max_iterations": 5}])
    def test_adapt_growth_options_are_not_accepted(self, option):
        with pytest.raises(TypeError, match="does not take"):
            Mandacaru(method="vasqa", **option)


class TestRunLog:
    """``txt=`` holds ADAPT-VQE's blocks, with ``[MARKOV CHAIN]`` for
    ``[ITERATIONS]``, and reads back through the protocol's parser."""

    def _log(self, h2_hamiltonian, tmp_path, **options):
        from mandacaru.utils.logging import parse_output, reset_log
        path = tmp_path / "output.txt"
        reset_log(str(path))
        result = _vasqa(h2_hamiltonian, pool="qubit", max_steps=15,
                        txt=str(path), **options).run()
        return result, parse_output(str(path)), path.read_text()

    def test_one_row_per_step_matching_the_result(self, h2_hamiltonian,
                                                  tmp_path):
        result, log, text = self._log(h2_hamiltonian, tmp_path)
        assert "[MARKOV CHAIN]" in text and "[ITERATIONS]" not in text
        assert "VASQA (QubitPool)" in text
        rows = log["markov_chain"]
        assert [r["step"] for r in rows] == list(range(1, 16))
        for row, step in zip(rows, result.steps):
            assert row["move"] == step.move
            assert row["accepted"] == step.accepted
            assert row["L"] == len(step.proposed)
            assert row["energy"] == pytest.approx(step.proposed_energy,
                                                  abs=1e-6)
            assert row["current"] == pytest.approx(step.current_energy,
                                                   abs=1e-6)
            assert row["energy_unit"] == "eV"

    def test_de_is_measured_from_the_state_before_the_step(
            self, h2_hamiltonian, tmp_path):
        result, log, _ = self._log(h2_hamiltonian, tmp_path)
        before = result.reference_energy
        for row, step in zip(log["markov_chain"], result.steps):
            assert row["dE"] == pytest.approx(step.proposed_energy - before,
                                              abs=1e-6)
            before = step.current_energy

    def test_six_decimals_and_no_cost_column(self, h2_hamiltonian, tmp_path):
        _r, log, text = self._log(h2_hamiltonian, tmp_path,
                                  length_penalty=0.01)
        assert "dF" not in log["markov_chain"][0]
        row = text.split("[MARKOV CHAIN]")[1].splitlines()[3].split()
        # step time move L energy dE current ...: six decimals each.
        for cell in (row[4], row[5], row[6]):
            assert len(cell.split(".")[1]) == 6

    def test_setup_and_summary_blocks(self, h2_hamiltonian, tmp_path):
        result, log, _ = self._log(h2_hamiltonian, tmp_path)
        assert log["setup"]["pool"] == "qubit"
        assert int(log["setup"]["max_steps"]) == 15
        assert log["setup"]["seed"] == "7"
        summary = log["summary"]
        assert "converged" not in summary
        assert float(summary["optimal_energy_eV"]) == pytest.approx(
            result.optimal_energy, abs=1e-9)
        assert summary["operators"] == ", ".join(result.operators)
        assert int(summary["chain_steps"]) == 15

    def test_a_trace_prints_the_same_blocks(self, h2_hamiltonian, capsys):
        _vasqa(h2_hamiltonian, max_steps=3, trace=True).run()
        out = capsys.readouterr().out
        assert "[OPTIMIZATION SETUP]" in out and "[MARKOV CHAIN]" in out


class TestCalculatorMode:
    def test_energy_and_forces_from_a_geometry(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        atoms.center(vacuum=2.5)
        atoms.calc = Mandacaru(method="vasqa", basis="HAO", h=0.4,
                               max_steps=10, seed=3, profile=False,
                               trace=False)
        energy = atoms.get_potential_energy()
        assert math.isfinite(energy)
        assert isinstance(atoms.calc.result, VASQAResult)
        assert energy == pytest.approx(atoms.calc.result.optimal_energy)
        forces = atoms.get_forces()
        assert forces.shape == (2, 3) and np.all(np.isfinite(forces))
        # Newton's third law along the bond.
        assert forces[0, 2] == pytest.approx(-forces[1, 2], abs=1e-3)
