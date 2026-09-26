# -*- coding: utf-8 -*-
# file: test/utils/test_logging.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The run report: one renderer, `[BLOCK]` by `[BLOCK]`.

:mod:`mandacaru.utils.logging` renders a run as an ordered set of blocks --
``[SYSTEM]``, ``[BASIS]``, ``[ELECTRONS]``, ``[OPTIMIZATION SETUP]``,
``[ITERATIONS]``, the summary, ``[FORCES]``, ``[PERFORMANCE]`` and the
relaxation footer -- and writes them to whatever destinations
``log_targets`` names: the ``txt=`` file, standard output, or both.  One class
per block, plus the lifetime rules (append across geometry steps, indentation,
nothing said twice) and the round trip through :func:`parse_output`.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from mandacaru.algorithms import Mandacaru
from mandacaru.core import MolecularIntegrals, minimal_hao_basis
from mandacaru.integrals import Grid
from mandacaru.optimizers import DEFAULT_OPTIMIZER
from mandacaru.utils import AdaptOutputLogger, parse_output
import os
from mandacaru.core.mapping import PauliSum


#: A direct-mode (no geometry) problem: the smallest register that carries a
#: real ansatz, in Hartree so the numbers read as the solver stores them.
DIRECT = dict(num_particles=(1, 1), n_spatial_orbitals=2, atomic_units=True,
              trace=False, profile=False)

def h2(distance=0.74, cell=6.0):
    """H2 centered in a cubic cell (the grid is cut from it)."""
    middle = cell / 2
    return Atoms("H2", positions=[[middle, middle, middle - distance / 2],
                                  [middle, middle, middle + distance / 2]],
                 cell=[cell] * 3)


@pytest.fixture(scope="module")
def h2_hamiltonian():
    """A 4-qubit H2 Hamiltonian, built once for the whole module."""
    R = 0.74
    nuclei = [(1.0, np.array([0.0, 0.0, -R / 2])),
              (1.0, np.array([0.0, 0.0, +R / 2]))]
    grid = Grid(center=[0, 0, 0], box_size=5.0, h=0.35)
    integrals = MolecularIntegrals(nuclei, minimal_hao_basis(nuclei), grid)
    return integrals.molecular_hamiltonian(mo_basis=True, n_electrons=2)


def _h2_adapt(hamiltonian, **kwargs):
    """A direct-mode ADAPT-VQE on that Hamiltonian -- the log's producer."""
    return Mandacaru(method="adapt-vqe", hamiltonian=hamiltonian,
                     pool="fermionic", num_particles=(1, 1),
                     n_spatial_orbitals=2, profile=False, **kwargs)


class TestAdaptOutputProtocol:
    def test_default_optimizer_is_cobyla(self, h2_hamiltonian):
        # Requirement 6: the default classical optimizer must be COBYLA.
        assert _h2_adapt(h2_hamiltonian).optimizer.method == \
            DEFAULT_OPTIMIZER

    def test_output_file_written_and_parseable(self, h2_hamiltonian, tmp_path):
        R = 0.74
        geom = Atoms("H2", positions=[[0, 0, -R / 2], [0, 0, R / 2]],
                     cell=[[6, 0, 0], [1, 7, 0], [0, 0, 5]], pbc=True)
        out = str(tmp_path / "output.txt")
        adapt = _h2_adapt(h2_hamiltonian, max_iterations=6,
                          gradient_tolerance=1e-4, txt=out)
        result = adapt.run(geometry=geom, log_expressivity=True)

        parsed = parse_output(out)

        # Metadata block: initial geometry + explicit unit-cell parameters.
        assert parsed["system"]["n_atoms"] == "2"
        assert parsed["system"]["cell_present"] == "True"
        assert parsed["system"]["units"] == "Angstrom"       # req 1: default A
        assert "cell_lengths" in parsed["system"]
        assert "cell_angles" in parsed["system"]

        # Optimization setup block -- energies default to eV (requirement 1).
        assert parsed["setup"]["classical_optimizer"] == DEFAULT_OPTIMIZER
        assert parsed["setup"]["energy_unit"] == "eV"
        assert "reference_energy_eV" in parsed["setup"]

        # One row per iteration, every tracked property a column.
        assert len(parsed["iterations"]) == result.num_operators >= 1
        for index, it in enumerate(parsed["iterations"], start=1):
            assert it["index"] == index
            assert it["selected_operator"]                 # 3. selected operator
            assert it["expressivity_E"] != "-"             # 4. expressivity E
            assert it["energy_unit"] == "eV"               # 1. eV default
            assert it["max_gradient"] is not None          # 2. gradient
            assert it["cnot_count"] and it["circuit_depth"]
        text = open(out, encoding="utf-8").read()
        # The pool's size and type are stated once, before the table.
        assert "pool_size:" in text and "[ITERATIONS]" in text

    def test_only_the_selected_operator_and_the_pool_size_are_logged(
            self, h2_hamiltonian, tmp_path):
        # Requirement 3, as revised 2026-09-17: the log names the operator that
        # was selected and how large the pool is, and does not list the pool --
        # `verbose_operators=True` writes that to pool.json instead.
        out = str(tmp_path / "output.txt")
        adapt = _h2_adapt(h2_hamiltonian, max_iterations=4,
                          gradient_tolerance=1e-4, txt=out)
        result = adapt.run()
        text = open(out, encoding="utf-8").read()
        setup, table = text.split("[ITERATIONS]", 1)
        # Pool type and size before the table, each stated once.
        assert f"pool_size: {len(adapt._pool_ops)}" in setup
        assert "pool: " in setup and "pool_class: " in setup
        # The table names the selected operator per row and nothing else.
        assert "operator_pool:" not in text
        assert "(selected)" not in text
        rows = [l for l in table.splitlines() if l.strip()[:1].isdigit()]
        assert len(rows) == len(result.operators)
        for row, label in zip(rows, result.operators):
            assert row.split()[-1] == label

    def test_verbose_operators_writes_the_pool_file(self, h2_hamiltonian,
                                                    tmp_path):
        """The pool leaves the log for a JSON file, written once."""
        import json

        pool_file = tmp_path / "pool.json"
        adapt = _h2_adapt(h2_hamiltonian, max_iterations=2,
                          gradient_tolerance=1e-4,
                          verbose_operators=str(pool_file))
        adapt.run()
        payload = json.loads(pool_file.read_text())
        assert payload["pool"] == adapt.pool.name
        assert payload["pool_size"] == len(adapt._pool_ops) == \
            len(payload["operators"])
        assert payload["n_qubits"] == adapt.n_qubits
        first = payload["operators"][0]
        assert first["label"] == adapt._pool_ops[0].label
        assert first["kind"] == adapt._pool_ops[0].kind
        assert first["generator"] and all(
            set(term["pauli"]) <= set("IXYZ") for term in first["generator"])

    def test_summary_reports_final_parameterization(self, h2_hamiltonian, tmp_path):
        # Requirement 5: richer summary (expressivity, gates, CNOTs, depth, ...).
        out = str(tmp_path / "output.txt")
        # profile=True: the summary reports gate counts.
        adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                          pool="fermionic", num_particles=(1, 1),
                          n_spatial_orbitals=2, profile=True, max_iterations=4,
                          gradient_tolerance=1e-4, txt=out)
        result = adapt.run(log_expressivity=True)  # the expressivity is opt-in
        summary = parse_output(out)["summary"]
        for key in ("optimal_energy_eV", "reference_energy_eV", "num_operators",
                    "num_parameters", "final_expressivity_E", "cnot_count",
                    "circuit_depth", "total_gates", "one_qubit_gates",
                    "cost_evaluations"):
            assert key in summary, key
        # The operator sequence is not repeated in the summary: the iteration
        # table above already names each selected operator, in order.
        assert "operator_sequence" not in summary
        assert [it["selected_operator"] for it in parse_output(out)["iterations"]] \
            == list(result.operators)

    def test_atomic_units_switch(self, h2_hamiltonian, tmp_path):
        # Requirement 1: atomic units used only when explicitly requested.
        out = str(tmp_path / "output.txt")
        geom = Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]])
        adapt = _h2_adapt(h2_hamiltonian, atomic_units=True, max_iterations=4,
                          gradient_tolerance=1e-4, txt=out)
        adapt.run(geometry=geom)
        parsed = parse_output(out)
        assert parsed["system"]["units"] == "Bohr"
        assert parsed["setup"]["energy_unit"] == "Ha"
        assert "reference_energy_Ha" in parsed["setup"]

    def test_runs_without_geometry(self, h2_hamiltonian, tmp_path):
        # The protocol must still write cleanly when no geometry is supplied.
        out = str(tmp_path / "output.txt")
        _h2_adapt(h2_hamiltonian, max_iterations=4, gradient_tolerance=1e-4,
                  txt=out).run()
        parsed = parse_output(out)
        assert parsed["system"]["cell_present"] == "False"
        assert parsed["system"]["geometry"] == "(not provided)"


class TestAdaptOutputLogger:
    def test_logger_cell_parameters(self, tmp_path):
        out = str(tmp_path / "log.txt")
        with AdaptOutputLogger(out) as logger:
            logger.write_system(
                symbols=["H", "H"], positions=[[0, 0, 0], [0, 0, 0.74]],
                cell=np.diag([5.0, 6.0, 7.0]))
        parsed = parse_output(out)
        assert parsed["system"]["cell_present"] == "True"
        assert parsed["system"]["cell_lengths"].startswith("a=5")


class TestLogAppendsAcrossSteps:
    """A geometry optimization writes one file, not one file per step.

    Every step of a relaxation runs a complete ADAPT-VQE and so builds its own
    logger; truncating the file each time left only the last step's iterations.
    The first logger of a path in a process truncates and writes the banner,
    every later one appends a numbered block.
    """

    @staticmethod
    def _block(path, step):
        """Write one minimal energy block to ``path`` and return the logger."""
        from types import SimpleNamespace

        from mandacaru.core.mapping import PauliSum
        pool = [SimpleNamespace(label="op0", kind="double",
                                generator=PauliSum({"XXXX": 0.5j}))]
        with AdaptOutputLogger(path, n_qubits=4) as logger:
            logger.write_system(symbols=["H", "H"],
                                  positions=[[0, 0, 0], [0, 0, 0.7 + step / 100]])
            logger.write_optimizer_setup("COBYLA", -1.0)
            logger.write_iteration(1, pool, [0.3], 0, None, -1.0 - step, 1)
            logger.write_summary(True, -1.0 - step, 1)
        return logger

    def test_banner_is_written_once_at_the_top(self, tmp_path):
        from mandacaru.utils import banner

        out = str(tmp_path / "output.txt")
        for step in (1, 2, 3):
            self._block(out, step)
        text = open(out, encoding="utf-8").read()
        # The banner is provenance of the *file*: once, before the first block.
        assert text.count("developed by:") == 1
        # The banner opens with a blank line, so "starts with its first line"
        # would hold for any file: ask for the wordmark itself, at the top.
        opening = "\n".join(banner.lines()[:4])
        assert banner.WORDMARK[0] in opening
        assert text.startswith(opening)
        assert text.index("developed by:") < text.index("[SYSTEM]")

    def test_each_step_appends_a_numbered_block(self, tmp_path):
        out = str(tmp_path / "output.txt")
        steps = [self._block(out, step).step for step in (1, 2, 3)]
        assert steps == [1, 2, 3]

        parsed = parse_output(out)
        assert len(parsed["steps"]) == 3
        # Nothing earlier was erased: every step's iterations are still there.
        assert [block["iterations"][0]["energy"] for block in parsed["steps"]] \
            == [-2.0, -3.0, -4.0]
        # The top level describes the last step, as a single-point log always did.
        assert parsed["system"]["step"] == "3"
        assert parsed["iterations"][0]["energy"] == -4.0
        # Only the later blocks carry the step in their title.
        text = open(out, encoding="utf-8").read()
        assert text.count("geometry step 2") == 1
        assert "geometry step 1" not in text

    def test_reset_log_starts_a_fresh_file(self, tmp_path):
        from mandacaru.utils import log_steps, reset_log

        out = str(tmp_path / "output.txt")
        self._block(out, 1)
        self._block(out, 2)
        assert log_steps(out) == 2

        # What a notebook cell re-run (or a driver loop wanting its own file)
        # needs: forget the path, and the next logger truncates and re-banners.
        reset_log(out)
        assert log_steps(out) == 0
        assert self._block(out, 9).step == 1
        text = open(out, encoding="utf-8").read()
        assert text.count("developed by:") == 1
        assert len(parse_output(out)["steps"]) == 1

    def test_append_is_overridable(self, tmp_path):
        out = str(tmp_path / "output.txt")
        self._block(out, 1)
        with AdaptOutputLogger(out, append=False) as logger:
            logger.write_system()
        # An explicit append=False truncates whatever the step counter says.
        assert len(parse_output(out)["steps"]) == 1


class TestBlockIndentation:
    """Section markers sit at column 0; what they contain is indented 4 spaces.

    One level per level of nesting: a block's keys are indented once, and the
    rows of a table (or the atoms under ``geometry:``) once more, so the
    structure of the file can be read off the left margin.
    """

    @pytest.fixture(scope="class")
    def log(self, tmp_path_factory):
        from types import SimpleNamespace

        from mandacaru.core.mapping import PauliSum
        from mandacaru.utils import append_forces
        from mandacaru.utils.logging import INDENT

        out = str(tmp_path_factory.mktemp("indent") / "output.txt")
        pool = [SimpleNamespace(label="op0", kind="double",
                                generator=PauliSum({"XXXX": 0.5j}))]
        with AdaptOutputLogger(out, n_qubits=4) as logger:
            logger.write_system(symbols=["O", "H"],
                                  positions=[[0, 0, 0], [0, 0.97, 0]],
                                  cell=np.diag([10.0, 10.0, 10.0]))
            logger.write_optimizer_setup("COBYLA", -476.6)
            logger.write_iteration(1, pool, [0.3], 0, None, -477.0, 1)
            logger.write_summary(True, -477.0, 1)
        append_forces(out, ["O", "H"], [[0.0, 0.9, 0.0], [0.0, -0.85, 0.0]],
                      hellmann_feynman=[[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
        assert INDENT == "    "
        return open(out, encoding="utf-8").read().splitlines()

    def test_markers_and_rules_stay_at_column_zero(self, log):
        for marker in ("[SYSTEM]", "[OPTIMIZATION SETUP]", "[ITERATIONS]",
                       "[VARIATIONAL QUANTUM SUMMARY]", "[FORCES]"):
            assert marker in log, marker
        assert all(not line.startswith(" ")
                   for line in log if set(line) == {"="})

    @pytest.mark.parametrize("key", ["step:", "units:", "n_atoms:", "geometry:",
                                     "cell_present:", "cell_vectors:",
                                     "cell_lengths:", "classical_optimizer:",
                                     "energy_unit:",
                                     "converged:", "num_operators:",
                                     "forces:", "max_force:", "net_force:"])
    def test_a_blocks_keys_are_indented_one_level(self, log, key):
        lines = [line for line in log if line.strip().startswith(key)]
        assert lines, key
        for line in lines:
            assert line.startswith("    ") and not line.startswith("     ")

    def test_nested_rows_are_indented_one_level_further(self, log):
        # The geometry and the cell vectors are the contents of the keyed line
        # above them, and their first field is left-aligned: exactly 8 spaces.
        left = [line for line in log
                if line.lstrip().startswith(("O  ", "H  ", "a1 =", "a2 =",
                                             "a3 ="))]
        assert len(left) == 5                        # 2 atoms + 3 cell vectors
        for line in left:
            assert line.startswith("        ") and not line.startswith(" " * 9)

        # A force table sits at the same level, but its columns are
        # right-aligned inside their fields, so only the structural part of the
        # indentation can be asserted.
        table = [line for line in log
                 if line.lstrip().startswith(("atom", "1 O", "2 H"))]
        assert len(table) == 6                       # 2 tables x (head + 2 rows)
        for line in table:
            assert line.startswith("        ")

    def test_the_iteration_table_is_indented_with_its_rows(self, log):
        table = log[log.index("[ITERATIONS]") + 1:]
        heading, rule, row = table[0], table[1], table[2]
        assert heading.startswith("    iter")
        # The rule keeps the heading's width, so the columns still line up.
        assert rule == "    " + "-" * len(heading.strip())
        assert row.startswith("       1 ")           # 4 + the right-aligned int

    def test_blank_separators_carry_no_trailing_whitespace(self, log):
        assert all(line == "" or line.strip() for line in log)


class TestSystemBlock:
    """``[SYSTEM]`` records the geometry the step was run on.

    A cell and periodicity are different facts -- Mandacaru always needs a cell
    (it is the box the real-space grid is cut from), while ``pbc`` says whether
    the system repeats along each of its vectors -- so the block reports both.
    Initial magnetic moments select the spin state, so a run given them has to
    say so.
    """

    def _system(self, tmp_path, name, **atoms_options):
        out = str(tmp_path / f"{name}.txt")
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]],
                      cell=[6.0] * 3, **atoms_options)
        atoms.center()
        atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", h=0.45,
                               pool="fermionic", max_iterations=1,
                               txt=out, trace=False, profile=False)
        atoms.get_potential_energy()
        return parse_output(out)["system"]

    def test_periodicity_is_reported_per_direction(self, tmp_path):
        system = self._system(tmp_path, "full", pbc=True)
        assert system["pbc"].startswith("a=True b=True c=True")
        assert "3-D" in system["pbc"]

    def test_a_partly_periodic_cell_names_its_directions(self, tmp_path):
        system = self._system(tmp_path, "chain", pbc=[True, False, False])
        assert system["pbc"].startswith("a=True b=False c=False")
        assert "1-D" in system["pbc"] and "periodic along a" in system["pbc"]

    def test_a_molecule_says_non_periodic(self, tmp_path):
        system = self._system(tmp_path, "molecule", pbc=False)
        assert system["pbc"] == "a=False b=False c=False (non-periodic)"
        # ... and still has a cell, which is the point of reporting both.
        assert system["cell_present"] == "True"

    def test_initial_magnetic_moments_are_reported_as_a_list(self, tmp_path):
        system = self._system(tmp_path, "triplet", pbc=True, magmoms=[1, -1])
        assert system["initial_magnetic_moments"] == "[1.0, -1.0]"

    def test_the_list_reads_back_as_the_literal_that_was_passed(self,
                                                                tmp_path):
        """It is the same list a caller hands to ``Atoms(magmoms=...)``."""
        import ast
        system = self._system(tmp_path, "roundtrip", pbc=True,
                              magmoms=[0.5, -0.25])
        assert ast.literal_eval(system["initial_magnetic_moments"]) == \
            [0.5, -0.25]

    def test_no_moments_is_a_list_of_zeros_not_silence(self, tmp_path):
        """A reader must be able to tell nobody set them from nobody looked."""
        system = self._system(tmp_path, "closed", pbc=True)
        assert system["initial_magnetic_moments"] == "[0.0, 0.0]"

    def test_a_geometry_without_them_reports_them_as_unknown(self):
        """A ``(symbols, positions)`` pair carries neither fact."""
        from mandacaru.utils.logging import _magmom_text, _pbc_text
        assert _pbc_text(None) == "(not provided)"
        assert _magmom_text(None) == "(not provided)"


class TestElectronsBlock:
    """``[ELECTRONS]`` records which Hamiltonian the iterations belong to.

    With the trace routed to the log file, this block is the *only* place the
    configuration is written down, so it has to carry everything needed to know
    what was solved: the discretization, the encoding, and the size of the
    register and Hamiltonian that came out.
    """

    # No "basis": the [BASIS] block owns it (TestBasisBlock below).
    FIELDS = ("grid spacing", "grid points", "kinetic operator", "k-points",
              "charge", "spin-polarized", "reference state", "frozen core",
              "active space", "Z2 tapering", "mapping", "Hamiltonian",
              "spatial orbitals", "electrons (alpha, beta)", "qubits")

    @pytest.fixture(scope="class")
    def run(self, tmp_path_factory):
        from mandacaru import Mandacaru

        out = str(tmp_path_factory.mktemp("electrons") / "output.txt")
        atoms = Atoms("H2", positions=[[3, 3, 2.63], [3, 3, 3.37]],
                      cell=[6.0, 6.0, 6.0])
        atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", h=0.35,
                               pool="fermionic", max_iterations=2,
                               gradient_tolerance=1e-3, txt=out)
        atoms.get_potential_energy()
        return out, atoms.calc

    def test_every_field_is_present_and_in_order(self, run):
        out, _calc = run
        block = parse_output(out)["electrons"]
        assert tuple(block) == self.FIELDS

    def test_the_values_describe_this_run(self, run):
        out, calc = run
        block = parse_output(out)["electrons"]
        assert "basis" not in block
        assert block["grid spacing"] == "0.35 Angstrom (requested)"
        # What the cell turned the request into: 6 Angstrom / 0.35 -> 17 steps.
        assert block["grid points"].startswith("18 x 18 x 18 (spacing 0.35")
        assert block["charge"] == "0"
        assert block["reference state"] == "hartree-fock"
        assert block["frozen core"] == "none"
        # Never blank: a run that puts every orbital on the register says so, so
        # a reader can tell the truncation was absent rather than unrecorded.
        assert block["active space"] == "none (every orbital on the register)"
        assert block["Z2 tapering"] == "none"
        assert block["kinetic operator"] == "finite difference"
        assert "Monkhorst-Pack" in block["k-points"]
        assert block["spin-polarized"] == "False (multiplicity 1)"
        # The transformation's own name, not the identifier the code uses.
        assert block["mapping"] == "Jordan-Wigner"
        assert block["qubits"] == str(calc.n_qubits) == "4"
        assert block["electrons (alpha, beta)"] == str(calc.num_particles)
        assert block["spatial orbitals"] == "2"
        assert block["Hamiltonian"] == \
            f"{len(calc.hamiltonian.simplify().terms)} Pauli terms"

    def test_the_problem_is_not_repeated_in_the_setup_block(self, run):
        # [ELECTRONS] owns the problem; [OPTIMIZATION SETUP] owns the optimizer
        # and the pool.  Saying the mapping twice, in two spellings, is how the
        # two blocks drift apart.
        out, _calc = run
        setup = parse_output(out)["setup"]
        assert "mapping" not in setup and "num_particles" not in setup
        assert {"pool", "pool_class", "pool_size"} <= set(setup)

    def test_the_system_block_is_named_system(self, run):
        out, _calc = run
        text = open(out, encoding="utf-8").read()
        assert "[SYSTEM]" in text
        # The block order is the reading order: what, then how, then the run.
        assert text.index("[SYSTEM]") < text.index("[ELECTRONS]") \
            < text.index("[OPTIMIZATION SETUP]") < text.index("[ITERATIONS]")


class TestBasisBlock:
    """``[BASIS]``: the basis that *ran*, not the options that were typed.

    A Mandacaru basis is built at run time, so its defaults (PAW-LCAO is filtered),
    the radii an ``energy_shift`` resolves to and the dataset files are decided
    below the calculator.  The block is where they are written down -- it is
    what a comparison against another code is made from.
    """

    @pytest.fixture(scope="class")
    def paw(self, tmp_path_factory):
        from mandacaru import Mandacaru
        from mandacaru.pseudopotentials import get_paw
        try:
            get_paw("H")
        except (FileNotFoundError, ValueError):
            pytest.skip("MANDACARU_PAW_PATH is not configured")
        out = str(tmp_path_factory.mktemp("basis") / "output.txt")
        atoms = Atoms("H2", positions=[[4, 4, 3.63], [4, 4, 4.37]],
                      cell=[8.0, 8.0, 8.0])
        atoms.calc = Mandacaru(method="adapt-vqe",
                               basis={"name": "PAW-LCAO", "size": "DZ",
                                      "energy_shift": 0.1},
                               h=0.30, pool="fermionic", max_iterations=1,
                               txt=out)
        atoms.get_potential_energy()
        return out, atoms.calc

    def test_it_sits_between_the_system_and_the_electrons(self, paw):
        out, _calc = paw
        markers = [line.strip() for line in open(out)
                   if line.startswith("[")]
        assert markers[:3] == ["[SYSTEM]", "[BASIS]", "[ELECTRONS]"]

    def test_defaults_the_user_never_typed_are_recorded(self, paw):
        out, _calc = paw
        block = parse_output(out)["basis"]
        assert block["name"] == "PAW-LCAO" and block["family"].startswith("PAW-LCAO (")
        assert block["size"] == "DZ"
        # The tail-norm scheme is the default, and the line names it rather
        # than printing a bare number: a tail *norm* and a squared-norm
        # fraction are not comparable.  The output never names another code.
        assert block["zeta_split"].startswith("tail_norm 0.16, 0.3, 0.6 (norm")
        assert "GPAW" not in open(out).read()
        # The shell follows the confinement: the Gaussian when confined.
        assert block["polarization"].startswith("gaussian")
        assert block["energy_shift"] == "0.1 eV"
        assert "A = 12 Ha" in block["confinement_potential"]
        # On by default for PAW-LCAO, with the cutoff the grid resolved it to.
        assert block["filter"].startswith("filtered")
        assert "Bohr^-1" in block["filter_cutoff"]
        assert block["local_potential"].startswith("range-separated")
        assert block["dataset_xc"].startswith("LDA")
        # The folder is stated once, by name and path; the table names files.
        assert block["directory"].startswith("lda (")
        assert block["basis_functions"] == "4"

    def test_the_tables_read_back_typed(self, paw):
        from mandacaru.pseudopotentials import get_paw
        from mandacaru.pseudopotentials.confinement import confined_orbital

        out, _calc = paw
        block = parse_output(out)["basis"]
        (dataset,) = block["datasets"]
        assert dataset["symbol"] == "H" and dataset["Z_ion"] == 1
        assert dataset["source"] in ("H.parquet", "H.json")
        (orbital,) = block["orbitals"]
        expected = confined_orbital(get_paw("H"), 0, 0.1)
        assert orbital["l"] == 0 and orbital["zetas"] == 2
        assert orbital["r_c_Bohr"] == pytest.approx(expected.r_c, abs=1e-4)
        assert orbital["shift_eV"] == pytest.approx(0.1, abs=1e-3)
        assert orbital["eps_basis_eV"] - orbital["eps_free_eV"] == \
            pytest.approx(orbital["shift_eV"], abs=1e-5)
        assert block["functions"] == [
            {"symbol": "H", "atoms": 2, "functions_per_atom": 2}]

    def test_an_unconfined_basis_says_so(self, tmp_path):
        from mandacaru import Mandacaru
        from mandacaru.pseudopotentials import get_paw
        try:
            get_paw("H")
        except (FileNotFoundError, ValueError):
            pytest.skip("MANDACARU_PAW_PATH is not configured")
        out = str(tmp_path / "output.txt")
        atoms = Atoms("H2", positions=[[4, 4, 3.63], [4, 4, 4.37]],
                      cell=[8.0, 8.0, 8.0])
        atoms.calc = Mandacaru(method="adapt-vqe",
                               basis={"name": "PAW-LCAO", "energy_shift": None},
                               h=0.30, pool="fermionic", max_iterations=1,
                               txt=out)
        atoms.get_potential_energy()
        block = parse_output(out)["basis"]
        assert block["energy_shift"] == "unconfined"
        assert "confinement_potential" not in block
        assert block["orbitals"][0]["r_c_Bohr"] == "unconfined"

    def test_an_all_electron_basis_has_one_too(self, tmp_path):
        from mandacaru import Mandacaru
        out = str(tmp_path / "output.txt")
        atoms = Atoms("H2", positions=[[3, 3, 2.63], [3, 3, 3.37]],
                      cell=[6.0, 6.0, 6.0])
        atoms.calc = Mandacaru(method="adapt-vqe",
                               basis={"name": "HAO", "virtual_orbitals": 1},
                               h=0.35, pool="fermionic", max_iterations=1,
                               txt=out)
        atoms.get_potential_energy()
        block = parse_output(out)["basis"]
        assert block["name"] == "HAO" and block["family"] == "all-electron"
        assert "virtual_orbitals" in block["options"]
        assert block["basis_functions"] == "4"
        assert block["functions"] == [
            {"symbol": "H", "atoms": 2, "functions_per_atom": 2}]

    def test_direct_mode_has_none(self, tmp_path, h2_hamiltonian):
        # No basis was built, so a block describing one would be a guess.
        from mandacaru import Mandacaru
        out = str(tmp_path / "output.txt")
        Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                  pool="fermionic", num_particles=(1, 1),
                  n_spatial_orbitals=2, profile=False, max_iterations=1,
                  txt=out, trace=False).run()
        assert "[BASIS]" not in open(out).read()
        assert "basis" not in parse_output(out)


class TestOptimizationSetupBlock:
    """``[OPTIMIZATION SETUP]`` says how the run was configured.

    The ADAPT screening gradient can be computed three ways and the choice
    changes both the cost and the truncation error of the ``|grad|`` column, so
    it is a first-class line of the block -- ``gradient_method``, next to the
    ``gradient_tol`` it is compared against -- and not, as it was, the last of
    the pool lines under the name ``screening_gradient``.
    """

    #: The block's lines, in the order they are written: the classical
    #: optimizer and its budget, the gradient group, the pool, the loop's
    #: starting point.  Lineage (a resumed run) closes it.
    FIELDS = ("classical_optimizer", "max_iterations",
              "gradient_method", "gradient_formula", "gradient_tol",
              "gradient_units",
              "pool", "pool_class", "pool_size",
              # How the ansatz grows, how a growth step is optimized and where
              # it executes: the log is the only record when the
              # standard-output header is off.
              "growth", "pruning",
              "reoptimize_all_parameters", "state_vector_backend", "device",
              "backend_provider", "circuit_execution", "shots",
              "circuit_profiling",
              "energy_unit", "reference_energy_eV")

    def _log(self, hamiltonian, tmp_path, **kwargs):
        out = str(tmp_path / "output.txt")
        Mandacaru(method="adapt-vqe", hamiltonian=hamiltonian,
                  pool="fermionic", num_particles=(1, 1),
                  n_spatial_orbitals=2, profile=False, max_iterations=2,
                  gradient_tolerance=1e-3, txt=out, trace=False,
                  **kwargs).run()
        return out

    def test_the_groups_are_written_in_order(self, h2_hamiltonian, tmp_path):
        parsed = parse_output(self._log(h2_hamiltonian, tmp_path))
        setup = parsed["setup"]
        assert tuple(setup) == self.FIELDS
        # The top level is the last step, so every step carries the block.
        assert parsed["steps"][-1]["setup"] == setup

    @pytest.mark.parametrize("gradient", ["analytic", "finite_difference",
                                          "parameter_shift"])
    def test_every_method_is_named_in_its_canonical_spelling(
            self, h2_hamiltonian, tmp_path, gradient):
        out = self._log(h2_hamiltonian, tmp_path, gradient=gradient)
        setup = parse_output(out)["setup"]
        assert setup["gradient_method"] == gradient
        # And in the file itself, at one level of indentation like every other
        # keyed line of a block.
        assert f"\n    gradient_method: {gradient}\n" in \
            open(out, encoding="utf-8").read()

    def test_a_hyphenated_request_is_logged_canonically(self, h2_hamiltonian,
                                                        tmp_path):
        """The log records the method, not how the user spelled it."""
        out = self._log(h2_hamiltonian, tmp_path, gradient="parameter-shift")
        assert parse_output(out)["setup"]["gradient_method"] == \
            "parameter_shift"

    def test_the_method_precedes_the_tolerance_it_qualifies(
            self, h2_hamiltonian, tmp_path):
        text = open(self._log(h2_hamiltonian, tmp_path),
                    encoding="utf-8").read()
        assert text.index("gradient_method:") < text.index("gradient_formula:") \
            < text.index("gradient_tol:") < text.index("gradient_units:")

    @pytest.mark.parametrize("gradient", ["analytic", "finite_difference",
                                          "parameter_shift"])
    def test_the_formula_says_what_the_method_computes(self, h2_hamiltonian,
                                                       tmp_path, gradient):
        """A name alone does not tell a reader what was evaluated."""
        from mandacaru.algorithms.adapt_vqe import GRADIENT_FORMULAS

        out = self._log(h2_hamiltonian, tmp_path, gradient=gradient)
        setup = parse_output(out)["setup"]
        # Parseable although the value contains "=" and ","; the parser splits
        # on the first colon, and no formula contains one.
        assert setup["gradient_formula"] == GRADIENT_FORMULAS[gradient]

    def test_the_retired_key_is_gone_from_what_is_written(self, h2_hamiltonian,
                                                          tmp_path):
        """A mandated replacement is carried through -- no leftover alias."""
        text = open(self._log(h2_hamiltonian, tmp_path),
                    encoding="utf-8").read()
        assert "screening_gradient" not in text

    def test_an_old_log_still_reads_back(self, tmp_path):
        """Writers stop producing a retired key; readers stay tolerant of it."""
        old = tmp_path / "old_output.txt"
        old.write_text("[OPTIMIZATION SETUP]\n"
                       "    classical_optimizer: COBYLA\n"
                       "    gradient_tol: 0.001\n"
                       "    gradient_units: Hartree\n"
                       "    screening_gradient: parameter-shift\n",
                       encoding="utf-8")
        setup = parse_output(str(old))["setup"]
        assert setup["gradient_method"] == "parameter-shift"
        # The fixture above is a *file*, so it reads back what it says, not
        # what today's default would have written.
        assert setup["classical_optimizer"] == "COBYLA"

    def test_a_sparse_run_logs_what_it_actually_screened_with(
            self, h2_hamiltonian, tmp_path):
        """The sparse pool never forms the shift estimators' eigendecompositions.

        It screens analytically whatever was asked for, and the block has to
        report what ran -- with the request named, so the override is visible.
        """
        out = self._log(h2_hamiltonian, tmp_path, gradient="parameter_shift",
                        sparse=True)
        setup = parse_output(out)["setup"]
        assert setup["gradient_method"] == "analytic"
        assert "parameter_shift" in setup["gradient_formula"]

    def test_a_resumed_run_states_it_too(self, h2_hamiltonian, tmp_path):
        """Lineage closes the block; the gradient group is unaffected by it."""
        from mandacaru.utils.logging import reset_log

        checkpoint = str(tmp_path / "state.json")
        Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                  pool="fermionic", num_particles=(1, 1), n_spatial_orbitals=2,
                  profile=False, max_iterations=1, checkpoint=checkpoint,
                  trace=False).run()
        out = str(tmp_path / "resumed.txt")
        reset_log(out)
        Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                  pool="fermionic", num_particles=(1, 1), n_spatial_orbitals=2,
                  profile=False, max_iterations=3, gradient="finite-difference",
                  resume=checkpoint, txt=out, trace=False).run()

        setup = parse_output(out)["setup"]
        assert setup["gradient_method"] == "finite_difference"
        # The starting ansatz is the lineage's to state, not a second line.
        assert "initial_ansatz" not in setup
        keys = list(setup)
        # The four groups stay contiguous and lineage stays last.
        assert keys[:len(self.FIELDS)] == list(self.FIELDS)
        assert keys[len(self.FIELDS):] == [
            "resumed_from", "restored_operators", "restored_parameters",
            "restored_energy_eV", "resume_same_hamiltonian"]

    def test_the_trace_uses_the_same_vocabulary(self, h2_hamiltonian, capsys):
        """A reader who saw the terminal recognizes the file, and vice versa.

        Literally the same keys now -- the terminal *is* the block protocol,
        written to standard output instead of to a file.
        """
        Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                  pool="fermionic", num_particles=(1, 1), n_spatial_orbitals=2,
                  profile=False, max_iterations=1, gradient="parameter_shift",
                  trace=True).run()
        printed = capsys.readouterr().out
        assert "[OPTIMIZATION SETUP]" in printed
        assert "gradient_method: parameter_shift" in printed
        assert "parameter-shift" not in printed.split("formula")[0]


class TestNothingIsSaidTwice:
    """Each fact has one owner: a value written in two blocks can disagree
    with itself, and a reader has to decide which one to believe."""

    @pytest.fixture(scope="class")
    def text(self, tmp_path_factory):
        from mandacaru import Mandacaru
        out = str(tmp_path_factory.mktemp("owners") / "output.txt")
        atoms = Atoms("H2", positions=[[3, 3, 2.63], [3, 3, 3.37]],
                      cell=[6.0, 6.0, 6.0])
        atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", h=0.35,
                               pool="fermionic", max_iterations=2, txt=out)
        atoms.get_forces()
        return out

    def test_the_optimizer_belongs_to_the_setup_block(self, text):
        parsed = parse_output(text)
        assert parsed["setup"]["classical_optimizer"] == DEFAULT_OPTIMIZER
        assert "classical_optimizer" not in parsed["summary"]
        assert open(text).read().count("classical_optimizer:") == 1

    def test_the_basis_belongs_to_the_basis_block(self, text):
        parsed = parse_output(text)
        assert "basis" not in parsed["electrons"]
        assert parsed["basis"]["name"] == "HAO"

    def test_what_only_the_terminal_used_to_say_is_in_the_setup(self, text):
        setup = parse_output(text)["setup"]
        assert setup["reoptimize_all_parameters"] == "True"
        assert setup["shots"].startswith("0 (exact")
        assert "amplitudes" in setup["state_vector_backend"] \
            or "sector" in setup["state_vector_backend"]
        for key in ("device", "backend_provider", "circuit_execution"):
            assert key in setup

    def test_an_unprofiled_run_says_why_its_gate_columns_are_empty(
            self, tmp_path, h2_hamiltonian):
        # The H2O sample log once showed 20 rows of "-" under 1q / cnot /
        # depth with nothing in the file to explain them: the example passed
        # profile=False.  The setup block now states it either way.
        from mandacaru import Mandacaru
        logs = {}
        for profile in (True, False):
            out = str(tmp_path / f"profile_{profile}.txt")
            Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                      pool="fermionic", num_particles=(1, 1),
                      n_spatial_orbitals=2, profile=profile, max_iterations=1,
                      txt=out, trace=False).run()
            logs[profile] = parse_output(out)
        assert logs[True]["setup"]["circuit_profiling"] == "True"
        assert logs[True]["iterations"][0]["cnot_count"] not in (None, "-")
        assert logs[False]["iterations"][0]["cnot_count"] == "-"
        off = logs[False]["setup"]["circuit_profiling"]
        assert off.startswith("False") and "profile=True" in off
        assert "cnot_count" not in logs[False]["summary"]

    def test_the_two_wall_times_are_adjacent(self, text):
        lines = [line.strip().split(":")[0] for line in open(text)]
        index = lines.index("wall_time_s")
        assert lines[index + 1] == "solver_wall_time_s"


class TestForcesBlock:
    """The forces of a geometry step are logged with its energies."""

    FORCES = np.array([[0.0, 0.5, -1.5], [0.0, -0.4, 1.5], [0.0, -0.1, 0.02]])
    HF = np.array([[0.0, -2.0, 3.0], [0.0, 1.0, -1.5], [0.0, 1.0, -1.5]])
    PULAY = np.array([[0.0, 1.5, -1.5], [0.0, -0.6, 0.0], [0.0, -0.9, 1.48]])

    def _write(self, path, **kwargs):
        from mandacaru.utils import append_forces

        append_forces(path, ["O", "H", "H"], self.FORCES,
                      hellmann_feynman=self.HF, pulay=self.PULAY, **kwargs)
        return parse_output(path)["forces"]

    def test_vectors_and_breakdown_round_trip(self, tmp_path):
        out = str(tmp_path / "output.txt")
        block = self._write(out, step=1, extra={"force_method": "rdm"})

        assert block["symbols"] == ["O", "H", "H"]
        assert np.allclose(block["forces"], self.FORCES)
        assert np.allclose(block["hellmann_feynman"], self.HF)
        assert np.allclose(block["pulay"], self.PULAY)
        assert block["units"] == "eV/Angstrom"
        assert block["force_method"] == "rdm"
        assert block["step"] == 1

    def test_norms_summarize_the_step(self, tmp_path):
        out = str(tmp_path / "output.txt")
        block = self._write(out, step=1)
        norms = np.linalg.norm(self.FORCES, axis=1)
        # max_force is what an ASE optimizer converges on; net_force is the
        # translational residual a free molecule must not have.
        assert block["max_force"] == pytest.approx(norms.max(), abs=1e-7)
        assert block["rms_force"] == pytest.approx(
            np.sqrt((norms ** 2).mean()), abs=1e-7)
        assert block["net_force"] == pytest.approx(
            np.abs(self.FORCES.sum(axis=0)).max(), abs=1e-7)

    def test_the_block_lands_under_the_step_it_belongs_to(self, tmp_path):
        out = str(tmp_path / "output.txt")
        TestLogAppendsAcrossSteps._block(out, 1)
        self._write(out)                   # step read from the path's counter
        TestLogAppendsAcrossSteps._block(out, 2)
        self._write(out)

        parsed = parse_output(out)
        assert [block["forces"]["step"] for block in parsed["steps"]] == [1, 2]

    def test_a_breakdown_is_optional(self, tmp_path):
        from mandacaru.utils import append_forces

        out = str(tmp_path / "output.txt")
        append_forces(out, ["H", "H"], [[0.0, 0.0, 0.3], [0.0, 0.0, -0.3]])
        block = parse_output(out)["forces"]
        assert np.allclose(block["forces"], [[0, 0, 0.3], [0, 0, -0.3]])
        assert "hellmann_feynman" not in block and "pulay" not in block


class TestPerformanceBlock:
    """Every evaluation records where its time and memory went."""

    @pytest.fixture(scope="class")
    def relaxation(self, tmp_path_factory):
        from ase.optimize import BFGS

        from mandacaru import Mandacaru

        out = str(tmp_path_factory.mktemp("perf") / "output.txt")
        atoms = Atoms("H2", positions=[[3, 3, 2.6], [3, 3, 3.4]],
                      cell=[6.0, 6.0, 6.0])
        atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", h=0.35,
                               pool="fermionic", max_iterations=4,
                               gradient_tolerance=1e-3, txt=out)
        BFGS(atoms, logfile=None).run(fmax=0.05, steps=1)
        return out

    def test_one_block_per_step_closing_it(self, relaxation):
        parsed = parse_output(relaxation)
        assert len(parsed["steps"]) >= 2
        for block in parsed["steps"]:
            assert "performance" in block
        # The performance block closes the step: it comes after the forces.
        text = open(relaxation, encoding="utf-8").read()
        assert text.index("[FORCES]") < text.index("[PERFORMANCE]")

    def test_the_gradient_is_one_of_the_stages(self, relaxation):
        # The whole point of the calculator writing this block: on a real
        # relaxation the nuclear gradient is the largest stage, and a block
        # closed when the solver finished would leave it out entirely.
        stages = parse_output(relaxation)["performance"]["stages_s"]
        assert "nuclear gradient (forces)" in stages
        assert any(name.startswith("integration:") for name in stages)
        assert "parameter optimization" in stages
        assert all(seconds >= 0.0 for seconds in stages.values())

    def test_times_add_up(self, relaxation):
        performance = parse_output(relaxation)["performance"]
        stages = performance["stages_s"]
        # Every figure in the block is written to four decimals, so a sum
        # rebuilt from the parsed values agrees only to that granularity.
        rounding = 2e-4
        assert performance["total_s"] == pytest.approx(sum(stages.values()),
                                                       abs=rounding)
        # The step's wall clock covers the stages plus what nobody timed
        # (Hamiltonian construction, the mapping, materialization).
        assert performance["wall_time_s"] >= performance["total_s"] - rounding
        assert performance["untimed_s"] == pytest.approx(
            performance["wall_time_s"] - performance["total_s"], abs=rounding)

    def test_resources_are_recorded(self, relaxation):
        performance = parse_output(relaxation)["performance"]
        assert performance["integration_backend"] in ("C (OpenMP)", "NumPy")
        assert isinstance(performance["cpu_count"], int)
        assert performance["peak_memory_MiB"] > 0
        # There is no distributed parallelism, and the block says so rather
        # than leaving it to be inferred.
        assert "not used" in performance["mpi"]

    def test_no_qpu_keys_without_a_processor(self, relaxation):
        # Nothing ran on hardware, so nothing is claimed for it.
        performance = parse_output(relaxation)["performance"]
        assert not [key for key in performance if key.startswith("qpu_")]

    def test_a_direct_run_writes_its_own_block(self, h2_hamiltonian, tmp_path):
        # With no calculator to defer to, the solver writes the block itself.
        out = str(tmp_path / "output.txt")
        _h2_adapt(h2_hamiltonian, max_iterations=2, gradient_tolerance=1e-4,
                  txt=out).run()
        performance = parse_output(out)["performance"]
        assert "parameter optimization" in performance["stages_s"]
        assert performance["wall_time_s"] > 0
        # No forces here, so no gradient stage -- the block reports what ran.
        assert "nuclear gradient (forces)" not in performance["stages_s"]


class TestQPUAccounting:
    """What a run spent on a processor, read from the provider that ran it."""

    class _Job:
        @staticmethod
        def job_id():
            return "abc123"

        @staticmethod
        def metrics():
            return {"usage": {"quantum_seconds": 27.0, "seconds": 31.5}}

    def test_reports_the_jobs_quantum_seconds(self):
        from mandacaru.backends.providers import qpu_usage

        provider = SimpleNamespace(device_spec="ibm_fez", shots=4096,
                                   last_job=self._Job())
        usage = qpu_usage(provider, wall_time_s=51.25)
        assert usage["qpu_device"] == "ibm_fez"
        assert usage["qpu_shots"] == 4096
        assert usage["qpu_jobs"] == 1
        assert usage["qpu_job_ids"] == "abc123"
        assert usage["qpu_seconds"] == 27.0          # the metered QPU time
        assert usage["qpu_billed_seconds"] == 31.5
        assert usage["qpu_wall_time_s"] == 51.25

    def test_a_local_provider_reports_only_what_it_knows(self):
        from mandacaru.backends.providers import qpu_usage

        provider = SimpleNamespace(device_spec="statevector", shots=0,
                                   last_job=None)
        usage = qpu_usage(provider, wall_time_s=1.5)
        # No job, so no quantum seconds are invented -- just the wall clock.
        assert usage == {"qpu_device": "statevector", "qpu_wall_time_s": 1.5}

    def test_no_provider_reports_nothing(self):
        from mandacaru.backends.providers import qpu_usage

        assert qpu_usage(None, wall_time_s=9.0) == {}

    def test_a_job_that_cannot_answer_does_not_break_the_log(self):
        from mandacaru.backends.providers import qpu_usage

        class Hostile:
            def job_id(self):
                raise RuntimeError("service unreachable")

            def metrics(self):
                raise RuntimeError("service unreachable")

        provider = SimpleNamespace(device_spec="ibm_fez", shots=100,
                                   last_job=Hostile())
        usage = qpu_usage(provider, wall_time_s=2.0)
        # The run already finished; a provider that cannot answer must not
        # cost it its log.
        assert usage["qpu_jobs"] == 1 and "qpu_seconds" not in usage


class TestRelaxationLog:
    """End to end: a BFGS relaxation writes one continuous, complete log."""

    @pytest.fixture(scope="class")
    def relaxation(self, tmp_path_factory):
        from ase.optimize import BFGS

        from mandacaru import Mandacaru

        out = str(tmp_path_factory.mktemp("relax") / "output.txt")
        atoms = Atoms("H2", positions=[[3, 3, 2.6], [3, 3, 3.4]],
                      cell=[6.0, 6.0, 6.0])
        atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", h=0.35,
                               pool="fermionic", max_iterations=4,
                               gradient_tolerance=1e-3, txt=out)
        opt = BFGS(atoms, logfile=None)
        opt.run(fmax=0.05, steps=2)
        return out, atoms, opt

    def test_every_step_is_in_the_file(self, relaxation):
        out, _atoms, opt = relaxation
        parsed = parse_output(out)
        # One block per force evaluation, and BFGS counts one per step it took.
        assert len(parsed["steps"]) == opt.nsteps + 1 >= 2
        # Each block is complete: the iterations *and* the forces of that step.
        for block in parsed["steps"]:
            assert block["iterations"] and block["summary"]["converged"] == "True"
            assert block["forces"]["symbols"] == ["H", "H"]
        assert open(out, encoding="utf-8").read().count("developed by:") == 1

    def test_the_logged_force_is_the_reported_force(self, relaxation):
        out, atoms, _opt = relaxation
        block = parse_output(out)["forces"]
        forces = atoms.get_forces()
        assert np.allclose(block["forces"], forces, atol=1e-8)
        assert block["max_force"] == pytest.approx(
            float(np.linalg.norm(forces, axis=1).max()), abs=1e-7)
        # The breakdown the calculator exposes is the breakdown it logged.
        result = atoms.calc.force_result
        assert np.allclose(block["hellmann_feynman"], result.hellmann_feynman,
                           atol=1e-8)
        assert np.allclose(block["pulay"], result.pulay, atol=1e-8)

    def test_energies_follow_the_trajectory(self, relaxation):
        out, atoms, _opt = relaxation
        parsed = parse_output(out)
        # The last block's summary is the energy ASE reports for this geometry.
        last = float(parsed["summary"]["optimal_energy_eV"])
        assert last == pytest.approx(atoms.get_potential_energy(), abs=1e-8)
        # A relaxation goes downhill: the blocks are ordered in time, not sorted.
        energies = [float(block["summary"]["optimal_energy_eV"])
                    for block in parsed["steps"]]
        assert energies[-1] <= energies[0] + 1e-9


class TestGeometryOptimizationSummary:
    """A relaxation closes its log with the trajectory seen as one thing.

    ASE never tells a calculator that a relaxation is over -- the optimizer just
    stops calling it -- so the summary is written either by an explicit
    :meth:`Mandacaru.write_optimization_summary` or, for a plain script, by the
    interpreter-exit hook that call also disarms.
    """

    @pytest.fixture(scope="class")
    def relaxed(self, tmp_path_factory):
        from ase.optimize import BFGS

        from mandacaru import Mandacaru

        out = str(tmp_path_factory.mktemp("relaxsummary") / "output.txt")
        atoms = Atoms("H2", positions=[[3, 3, 2.6], [3, 3, 3.4]],
                      cell=[6.0, 6.0, 6.0])
        atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", h=0.35,
                               pool="fermionic", max_iterations=4,
                               gradient_tolerance=1e-3, txt=out)
        opt = BFGS(atoms, logfile=None)
        opt.run(fmax=0.05, steps=2)
        assert atoms.calc.write_optimization_summary(optimizer=opt) is True
        return out, atoms, opt

    def test_the_summary_describes_the_whole_trajectory(self, relaxed):
        out, atoms, _opt = relaxed
        block = parse_output(out)["optimization"]
        steps = parse_output(out)["steps"]
        assert block["geometry_steps"] == len(steps) == len(atoms.calc.trajectory)
        # First and last energies are the run's, and the energy went down.
        assert block["final_energy_eV"] == pytest.approx(
            atoms.get_potential_energy(), abs=1e-8)
        assert block["energy_change_eV"] < 0
        assert block["final_max_force"] < block["initial_max_force"]

    def test_the_convergence_table_is_the_step_history(self, relaxed):
        out, _atoms, _opt = relaxed
        parsed = parse_output(out)
        rows = parsed["optimization"]["convergence"]
        assert [row["step"] for row in rows] == list(range(1, len(rows) + 1))
        # Each row is the step whose own blocks are above it in the file.
        for row, step in zip(rows, parsed["steps"]):
            assert row["energy"] == pytest.approx(
                float(step["summary"]["optimal_energy_eV"]), abs=1e-8)
            assert row["max_force"] == pytest.approx(step["forces"]["max_force"],
                                                     abs=1e-7)

    def test_the_relaxed_geometry_is_the_final_one(self, relaxed):
        out, atoms, _opt = relaxed
        geometry = parse_output(out)["optimization"]["relaxed_geometry"]
        assert [row[0] for row in geometry] == atoms.get_chemical_symbols()
        assert np.allclose([row[1:] for row in geometry], atoms.get_positions(),
                           atol=1e-9)

    def test_the_footer_states_the_verdict_against_fmax(self, relaxed):
        out, _atoms, opt = relaxed
        status = parse_output(out)["completion"]["status"]
        # The optimizer was handed over, so the footer knows what converged meant.
        assert status.startswith("converged" if opt.converged()
                                 else "NOT converged")
        assert f"{opt.fmax:.6f}" in status

    def test_it_is_written_once(self, relaxed):
        out, atoms, opt = relaxed
        text = open(out, encoding="utf-8").read()
        assert text.count("[GEOMETRY OPTIMIZATION SUMMARY]") == 1
        assert text.count("[RELAXATION COMPLETE]") == 1
        # A second call -- and the exit hook -- find it already written.
        assert atoms.calc.write_optimization_summary(optimizer=opt) is False
        atoms.calc._write_summary_at_exit()
        assert open(out, encoding="utf-8").read() == text

    def test_the_step_summaries_are_the_variational_ones(self, relaxed):
        out, _atoms, _opt = relaxed
        text = open(out, encoding="utf-8").read()
        # The per-step block is the *variational* summary; the relaxation's own
        # summary is the single block at the end.
        assert text.count("[VARIATIONAL QUANTUM SUMMARY]") == \
            len(parse_output(out)["steps"])

    def test_a_single_point_gets_no_optimization_summary(self, tmp_path):
        from mandacaru import Mandacaru

        out = str(tmp_path / "output.txt")
        atoms = Atoms("H2", positions=[[3, 3, 2.6], [3, 3, 3.4]],
                      cell=[6.0, 6.0, 6.0])
        atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", h=0.35,
                               pool="fermionic", max_iterations=2,
                               gradient_tolerance=1e-3, txt=out)
        atoms.get_forces()
        # One geometry is not a trajectory; there is nothing to summarize.
        assert atoms.calc.write_optimization_summary() is False
        assert "[GEOMETRY OPTIMIZATION SUMMARY]" not in open(out).read()
        assert atoms.calc.write_optimization_summary(force=True) is True


class TestStandardOutputIsTheASETable:
    """stdout carries the evolution of energies and forces, nothing else.

    The report has a destination -- ``txt=<path>`` -- so standard output is
    left to what an ASE optimizer prints there, the same split GPAW makes with
    ``txt=``.  Without a log file standard output is the only destination there
    is, so **the same blocks** are printed to it.
    """

    @staticmethod
    def _run(tmp_path, capsys, **options):
        from mandacaru import Mandacaru

        atoms = Atoms("H2", positions=[[3, 3, 2.6], [3, 3, 3.4]],
                      cell=[6.0, 6.0, 6.0])
        atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", h=0.35,
                               pool="fermionic", max_iterations=2,
                               gradient_tolerance=1e-3, **options)
        atoms.get_potential_energy()
        return capsys.readouterr().out

    def test_a_log_file_silences_the_trace(self, tmp_path, capsys):
        out = self._run(tmp_path, capsys, txt=str(tmp_path / "output.txt"))
        for noise in ("ADAPT-VQE", "operator pool", "Timings", "iter"):
            assert noise not in out, noise
        assert out.strip() == ""

    def test_without_a_log_file_the_trace_is_printed(self, tmp_path, capsys):
        out = self._run(tmp_path, capsys)
        # The only report there is, so it must not be silent -- and it is the
        # *same* report, block for block, that `txt=` would have written.
        for block in ("[SYSTEM]", "[BASIS]", "[ELECTRONS]",
                      "[OPTIMIZATION SETUP]", "[ITERATIONS]",
                      "[VARIATIONAL QUANTUM SUMMARY]", "[PERFORMANCE]"):
            assert block in out, block
        assert "reference_energy_eV" in out

    def test_the_screen_and_the_file_carry_the_same_report(self, tmp_path,
                                                           capsys):
        """One renderer, two destinations: the blocks must be identical.

        They used to be two renderers -- an older key/value header and a
        width-adaptive table on the terminal against the block protocol in the
        file -- which drifted apart option by option.  Reproduced here by
        running the same problem twice and diffing everything that is not
        machine- or clock-dependent.
        """

        from mandacaru.utils.logging import STDOUT, reset_log

        reset_log(STDOUT)
        printed = self._run(tmp_path, capsys)
        log = tmp_path / "same.txt"
        self._run(tmp_path, capsys, txt=str(log))
        written = log.read_text()
        reset_log(STDOUT)

        def blocks(text):
            # From `[SYSTEM]` (the banner reaches stdout from `_show_banner`
            # and the file from the logger, so it is not part of the report)
            # up to `[PERFORMANCE]`, which is wall times and peak RSS: those
            # differ between two runs of the same problem by construction, so
            # comparing them would be testing the clock.
            body = text[text.index("[SYSTEM]"):]
            end = body.find("[PERFORMANCE]")
            if end != -1:
                body = body[:end]
            return [line.rstrip() for line in body.splitlines() if line.strip()]

        assert blocks(printed) == blocks(written)

    def test_trace_overrides_the_automatic_choice(self, tmp_path, capsys):
        printed = self._run(tmp_path, capsys, trace=True,
                            txt=str(tmp_path / "with_trace.txt"))
        assert "ADAPT-VQE" in printed
        quiet = self._run(tmp_path, capsys, trace=False)
        assert quiet.strip() == ""

    def test_trace_must_be_a_boolean_or_none(self):
        from mandacaru import Mandacaru

        with pytest.raises(TypeError, match="trace must be"):
            Mandacaru(method="adapt-vqe", trace="yes")

    def test_verbose_is_refused_and_points_at_trace(self):
        from mandacaru import Mandacaru

        with pytest.raises(TypeError, match="trace="):
            Mandacaru(method="adapt-vqe", verbose=False)

    def test_the_log_still_has_everything(self, tmp_path, capsys):
        """Silencing stdout must route the detail, not discard it."""
        out = str(tmp_path / "output.txt")
        self._run(tmp_path, capsys, txt=out)
        parsed = parse_output(out)
        assert parsed["iterations"] and parsed["summary"]["converged"]
        assert parsed["performance"]["wall_time_s"] > 0


class TestVerbosePauliOutput:
    def test_no_pauli_strings_are_dumped(self, h2_hamiltonian, capsys):
        adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                          pool="ceo", num_particles=(1, 1),
                          n_spatial_orbitals=2, profile=False, trace=True,
                          max_iterations=3, gradient_tolerance=1e-6)
        adapt.run()
        out = capsys.readouterr().out
        # The Hamiltonian is summarized by size only -- its Pauli expansion runs
        # to thousands of lines for a realistic active space.
        assert "Hamiltonian: 15 Pauli terms" in out
        assert "* ZIII" not in out
        # Nor is the selected operator's generator dumped per iteration.
        assert "ansatz operator (Pauli strings)" not in out
        assert "[iter 1]" not in out
        # The operator is still reachable programmatically.
        assert "ZIII" in adapt.hamiltonian.simplify().terms
        # Nor is the pool listed: its name and size are all the trace carries.
        assert "pool: ceo" in out
        assert f"pool_size: {len(adapt._pool_ops)}" in out
        assert "operator_pool" not in out

    def test_iterations_are_single_aligned_lines(self, h2_hamiltonian,
                                                capsys):
        adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                          pool="ceo", num_particles=(1, 1),
                          n_spatial_orbitals=2, profile=True, trace=True,
                          max_iterations=3, gradient_tolerance=1e-6)
        result = adapt.run(log_expressivity=True)   # the `expr` column is opt-in
        out = capsys.readouterr().out

        # A column heading precedes the per-iteration rows: one column per
        # property computed at that step.  The headings are read off the table
        # rather than hard-coded, because the optional `expr` column exists
        # only when the expressivity was asked for.
        lines = out.splitlines()
        heading = next(line for line in lines if line.split()[:1] == ["iter"])
        # "energy (eV)" is one column but two whitespace-separated tokens.
        columns = heading.replace("energy (eV)", "energy").split()
        for column in ("iter", "time", "dE", "|grad|", "expr", "cnot", "1q",
                       "depth", "operator"):
            assert column in columns
        assert "type" not in columns

        index = lines.index(heading)
        # One line per grown operator, each carrying every column.
        rows = [line for line in lines[index + 2:]
                if line.strip() and line.split()[0].isdigit()]
        assert len(rows) == result.num_operators
        for index, row in enumerate(rows, start=1):
            fields = row.split()
            assert int(fields[0]) == index
            assert len(fields) == len(columns)
            # The label is written in full: this is the log protocol, which
            # never abbreviates what a reader would have to guess at.
            assert fields[-1] in result.operators
            cell = dict(zip(columns, fields))
            assert float(cell["expr"]) >= 0.0
            assert int(cell["1q"]) > 0
            assert int(cell["cnot"]) > 0 and int(cell["depth"]) > 0
        # The rows are column-aligned: the operator column starts at one offset.
        starts = {row.index(row.split()[-1]) for row in rows}
        assert len(starts) == 1

    def test_the_table_does_not_depend_on_the_terminal(self, h2_hamiltonian,
                                                       capsys, monkeypatch):
        """Every column, at every terminal width.

        The trace used to be a second, width-adaptive renderer that dropped
        columns to fit (``dE``, then ``1q``, ``depth``, ``steps``, ``cnot``)
        and elided long operator labels.  It is now the log protocol itself,
        printed rather than written, so a narrow terminal wraps the row instead
        of silently losing a column -- and the screen can be compared with the
        file line for line, which is the point.
        """
        headings = {}
        for columns in ("80", "200"):
            monkeypatch.setenv("COLUMNS", columns)
            adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                              pool="ceo", num_particles=(1, 1),
                              n_spatial_orbitals=2, profile=True, trace=True,
                              max_iterations=2, gradient_tolerance=1e-6)
            adapt.run(log_expressivity=True)     # the `expr` column is opt-in
            out = capsys.readouterr().out
            headings[columns] = next(line.split() for line in out.splitlines()
                                     if line.split()[:1] == ["iter"])
        assert headings["80"] == headings["200"]
        assert set(" ".join(headings["80"]).replace("energy (eV)",
                                                    "energy").split()) == {
            "iter", "time", "energy", "expr", "dE", "|grad|", "steps", "cnot",
            "1q", "depth", "operator"}

    def test_verbose_false_is_silent(self, h2_hamiltonian, capsys):
        adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                          pool="ceo", num_particles=(1, 1),
                          n_spatial_orbitals=2, profile=False, trace=False,
                          max_iterations=3, gradient_tolerance=1e-6)
        adapt.run()
        assert capsys.readouterr().out == ""


class TestMeasurementBlock:
    def test_the_log_carries_the_energy_ase_returned(self, tmp_path):
        from mandacaru.backends.providers import QiskitProvider

        out = str(tmp_path / "output.txt")
        atoms = h2()
        atoms.calc = Mandacaru(
            method="adapt-vqe", basis="HAO", h=0.45, pool="fermionic",
            max_iterations=2, gradient_tolerance=1e-3, txt=out, profile=False,
            measurement_provider=QiskitProvider(device="statevector", shots=0))
        energy = atoms.get_potential_energy()
        block = parse_output(out)["measurement"]
        assert float(block["energy_eV"]) == pytest.approx(energy, abs=1e-9)
        assert block["reported_by"].startswith("ASE")
        # The optimization's own value is kept, under its own name.
        assert "variational_energy_eV" in block


class TestCompletionIsNotAssumed:
    def test_the_exit_hook_does_not_claim_completion(self, tmp_path):
        out = str(tmp_path / "output.txt")
        atoms = h2()
        atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", h=0.45,
                               pool="fermionic", max_iterations=2, profile=False,
                               gradient_tolerance=1e-3, txt=out)
        atoms.get_forces()
        atoms.positions[1, 2] += 0.03
        atoms.get_forces()
        # Reaching interpreter exit says the process ended, not that the
        # optimization finished: exit handlers run after an exception too.
        atoms.calc._write_summary_at_exit()
        status = parse_output(out)["completion"]["status"]
        assert "not signaled" in status
        # It must not *claim* a verdict.  The word appears only in the clause
        # explaining that reaching exit does not establish one.
        assert not status.startswith(("converged", "NOT converged", "finished"))

    def test_an_empty_trajectory_is_not_forced(self, tmp_path):
        calc = Mandacaru(method="adapt-vqe", basis="HAO",
                         txt=str(tmp_path / "output.txt"))
        # Used to raise IndexError.
        assert calc.write_optimization_summary(force=True) is False

    def test_a_failed_write_can_be_retried(self, tmp_path, monkeypatch):
        out = str(tmp_path / "output.txt")
        atoms = h2()
        atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", h=0.45,
                               pool="fermionic", max_iterations=2, profile=False,
                               gradient_tolerance=1e-3, txt=out)
        atoms.get_forces()
        atoms.positions[1, 2] += 0.03
        atoms.get_forces()

        import mandacaru.utils.logging as logging_module
        def explode(*args, **kwargs):
            raise OSError("disk full")
        monkeypatch.setattr(logging_module, "append_optimization_summary",
                            explode)
        with pytest.raises(OSError):
            atoms.calc.write_optimization_summary()
        # The state must not say "written" after a write that failed.
        monkeypatch.undo()
        assert atoms.calc.write_optimization_summary() is True


class TestLogLifetime:
    def test_a_missing_parent_is_created_without_consuming_a_step(self, tmp_path):
        from mandacaru.utils.logging import AdaptOutputLogger, log_steps

        path = str(tmp_path / "deep" / "nested" / "output.txt")
        with AdaptOutputLogger(path, n_qubits=4) as logger:
            logger.write_system()
            assert logger.step == 1
        assert log_steps(path) == 1
        assert os.path.exists(path)


class TestParserPreservesData:
    @pytest.fixture
    def written(self, tmp_path):
        from types import SimpleNamespace

        from mandacaru.utils.logging import AdaptOutputLogger

        path = str(tmp_path / "output.txt")
        cell = np.array([[4.0, 0.0, 0.0], [0.5, 5.0, 0.0], [0.0, 0.0, 6.0]])
        pool = [SimpleNamespace(label="custom operator with spaces",
                                kind="double",
                                generator=PauliSum({"XXXX": 0.5j}))]
        with AdaptOutputLogger(path, n_qubits=4) as logger:
            logger.write_system(symbols=["H", "He"],
                                positions=[[1.0, 2.0, 3.0], [0.0, 0.0, 0.74]],
                                cell=cell)
            logger.write_iteration(1, pool, [0.3], 0, None, -1.5, 7)
        return path, cell

    def test_the_geometry_survives_as_numbers(self, written):
        path, _cell = written
        geometry = parse_output(path)["system"]["geometry"]
        assert [row[0] for row in geometry] == ["H", "He"]
        assert np.allclose([row[1:] for row in geometry],
                           [[1.0, 2.0, 3.0], [0.0, 0.0, 0.74]])

    def test_the_cell_vectors_survive(self, written):
        path, cell = written
        # Lengths and angles cannot recover the orientation; the vectors can.
        assert np.allclose(parse_output(path)["system"]["cell_vectors"], cell)

    def test_an_operator_label_may_contain_spaces(self, written):
        path, _cell = written
        entry = parse_output(path)["iterations"][0]
        assert entry["selected_operator"] == "custom operator with spaces"
        assert entry["energy"] == pytest.approx(-1.5)

    def test_an_unrecorded_field_is_not_invented(self, written):
        path, _cell = written
        # The writer took num_parameters=7 and does not serialize it: reported
        # as unknown rather than as the iteration index.
        assert parse_output(path)["iterations"][0]["num_parameters"] is None


class TestGradientProvenance:
    def test_the_gradient_units_are_stated(self, tmp_path):
        out = str(tmp_path / "output.txt")
        Mandacaru(method="adapt-vqe",
                  hamiltonian=PauliSum({"ZIII": 1.0, "XXII": 0.2}),
                  pool="fermionic", txt=out, max_iterations=1, **DIRECT).run()
        assert parse_output(out)["setup"]["gradient_units"] == "Hartree"

    def test_a_resumed_run_records_its_lineage(self, tmp_path):
        state = str(tmp_path / "state.json")
        out = str(tmp_path / "output.txt")
        hamiltonian = PauliSum({"ZIII": 1.0, "XXII": 0.2})
        Mandacaru(method="adapt-vqe", hamiltonian=hamiltonian,
                  pool="fermionic", checkpoint=state, max_iterations=1,
                  **DIRECT).run()
        Mandacaru(method="adapt-vqe", hamiltonian=hamiltonian,
                  pool="fermionic", resume=state, txt=out, max_iterations=2,
                  **DIRECT).run()
        setup = parse_output(out)["setup"]
        assert setup["resumed_from"] == state
        assert int(setup["restored_operators"]) == 1
        assert setup["resume_same_hamiltonian"] == "True"
        # The lineage, not a separate line, says what the run started from.
        assert int(setup["restored_parameters"]) >= 1
        assert "initial_ansatz" not in setup



# --------------------------------------------------------------------------- #
# The [ITERATIONS] columns: time, dE and the gradient's fixed-point format.
# --------------------------------------------------------------------------- #

class TestIterationColumns:
    """``iter`` then ``time`` (``HH:MM:SS``), ``dE`` from the previous row (the
    first from the reference energy), ``|grad|`` to six decimals, no ``type``."""

    @pytest.fixture
    def rows(self, tmp_path):
        import re

        path = str(tmp_path / "output.txt")
        pool = [SimpleNamespace(label="D(0,2->1,3)", kind="double",
                                generator=PauliSum({"XXXY": 0.5j}))]
        with AdaptOutputLogger(path, n_qubits=4) as logger:
            logger.write_optimizer_setup("BFGS", reference_energy=-1.0)
            logger.write_iteration(1, pool, [0.25], 0, None, -1.25, 1)
            logger.write_iteration(2, pool, [3.2e-7], 0, None, -1.5, 2)
        text = open(path, encoding="utf-8").read()
        table = text.split("[ITERATIONS]", 1)[1].splitlines()
        heading = table[1].split()
        assert re.fullmatch(r"\d\d:\d\d:\d\d", table[3].split()[1])
        return heading, parse_output(path)["iterations"], table

    def test_the_order_puts_the_time_second_and_drops_the_type(self, rows):
        heading, _entries, _table = rows
        assert heading[:2] == ["iter", "time"]
        assert heading.index("dE") < heading.index("|grad|")
        assert "type" not in heading

    def test_de_is_from_the_previous_row_and_first_from_the_reference(
            self, rows):
        _heading, entries, _table = rows
        assert entries[0]["delta_energy"] == pytest.approx(-0.25)
        assert entries[1]["delta_energy"] == pytest.approx(-0.25)
        assert entries[0]["time"] and entries[1]["time"]

    def test_the_gradient_has_six_decimals(self, rows):
        _heading, entries, table = rows
        cells = [line.split() for line in table[3:5]]
        column = rows[0].index("|grad|") - 1   # "energy (eV)": two heading tokens
        assert [row[column] for row in cells] == ["0.250000", "0.000000"]
        assert entries[0]["max_gradient"] == pytest.approx(0.25)
