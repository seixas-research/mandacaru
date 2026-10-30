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

from mandacaru import Mandacaru
from mandacaru.optimizers import DEFAULT_OPTIMIZER
from mandacaru.utils import Logger, parse_output
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


def _h2_adapt(hamiltonian, **kwargs):
    """A direct-mode ADAPT-VQE on that Hamiltonian -- the log's producer."""
    return Mandacaru(method="adapt-vqe", hamiltonian=hamiltonian,
                     pool="fermionic", num_particles=(1, 1),
                     n_spatial_orbitals=2, profile=False, **kwargs)


class TestLogger:
    def test_logger_cell_parameters(self, tmp_path):
        out = str(tmp_path / "log.txt")
        with Logger(out) as logger:
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
        with Logger(path, n_qubits=4) as logger:
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
        with Logger(out, append=False) as logger:
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
        with Logger(out, n_qubits=4) as logger:
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
        atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", grid_spacing=0.45,
                               pool="fermionic", max_iterations=1,
                               txt=out, trace=False, profile=False)
        atoms.get_potential_energy()
        return parse_output(out)["system"]

    @staticmethod
    def _periodic_system(tmp_path, name, pbc):
        """The block as written for a periodic geometry.

        A periodic geometry is calculated only by the native DFT, so the
        periodicity text is exercised on the writer itself."""
        out = str(tmp_path / f"{name}.txt")
        with Logger(out) as logger:
            logger.write_system(symbols=["H", "H"],
                                positions=[[3, 3, 2.63], [3, 3, 3.37]],
                                cell=np.diag([6.0] * 3), pbc=pbc)
        return parse_output(out)["system"]

    def test_periodicity_is_reported_per_direction(self, tmp_path):
        system = self._periodic_system(tmp_path, "full", pbc=[True] * 3)
        assert system["pbc"].startswith("a=True b=True c=True")
        assert "3-D" in system["pbc"]

    def test_a_partly_periodic_cell_names_its_directions(self, tmp_path):
        system = self._periodic_system(tmp_path, "chain",
                                       pbc=[True, False, False])
        assert system["pbc"].startswith("a=True b=False c=False")
        assert "1-D" in system["pbc"] and "periodic along a" in system["pbc"]

    def test_a_molecule_says_non_periodic(self, tmp_path):
        system = self._system(tmp_path, "molecule", pbc=False)
        assert system["pbc"] == "a=False b=False c=False (non-periodic)"
        # ... and still has a cell, which is the point of reporting both.
        assert system["cell_present"] == "True"

    def test_initial_magnetic_moments_are_reported_as_a_list(self, tmp_path):
        system = self._system(tmp_path, "triplet", magmoms=[1, -1])
        assert system["initial_magnetic_moments"] == "[1.0, -1.0]"

    def test_the_list_reads_back_as_the_literal_that_was_passed(self,
                                                                tmp_path):
        """It is the same list a caller hands to ``Atoms(magmoms=...)``."""
        import ast
        system = self._system(tmp_path, "roundtrip", magmoms=[0.5, -0.25])
        assert ast.literal_eval(system["initial_magnetic_moments"]) == \
            [0.5, -0.25]

    def test_no_moments_is_a_list_of_zeros_not_silence(self, tmp_path):
        """A reader must be able to tell nobody set them from nobody looked."""
        system = self._system(tmp_path, "closed")
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
              "charge", "spin-polarized", "reference state", "Z2 tapering",
              "mapping", "Hamiltonian",
              "spatial orbitals", "electrons (alpha, beta)", "qubits")

    @pytest.fixture(scope="class")
    def run(self, tmp_path_factory):
        from mandacaru import Mandacaru

        out = str(tmp_path_factory.mktemp("electrons") / "output.txt")
        atoms = Atoms("H2", positions=[[3, 3, 2.63], [3, 3, 3.37]],
                      cell=[6.0, 6.0, 6.0])
        atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", grid_spacing=0.35,
                               pool="fermionic", max_iterations=2,
                               convergence={"gradient": 1e-3}, txt=out)
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
        # The [ACTIVE SPACE] block owns the frozen core and the truncation,
        # and is written only when the register was reduced.
        assert "frozen core" not in block and "active space" not in block
        assert "active_space" not in parse_output(out)
        assert block["Z2 tapering"] == "none"
        assert block["kinetic operator"] == "spectral"
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

    def test_cholesky_factors_have_one_line(self, tmp_path):
        """A factored run says how it was factored -- tolerance, rank and the
        largest residual diagonal -- once, in this block, right after the
        kinetic operator; the dense run above has no such line."""
        from mandacaru import Mandacaru

        out = str(tmp_path / "output.txt")
        atoms = Atoms("H2", positions=[[3, 3, 2.63], [3, 3, 3.37]],
                      cell=[6.0, 6.0, 6.0])
        atoms.calc = Mandacaru(method="rhf", basis="HAO", grid_spacing=0.35,
                               two_body={"method": "cholesky",
                                         "tolerance": 1e-6}, txt=out)
        atoms.get_potential_energy()
        parsed = parse_output(out)
        block = parsed["electrons"]
        keys = list(block)
        assert keys[keys.index("kinetic operator") + 1] == "two-body integrals"
        factors = atoms.calc.solver._gradient_context["integrals"]._cholesky
        line = block["two-body integrals"]
        assert line == factors.summary()
        assert line.startswith("Cholesky factors, tolerance 1.000e-06 Ha, "
                               f"rank {factors.rank} (")
        assert f"largest residual diagonal {factors.max_residual:.3e} Ha" \
            in line
        # One owner: no other block repeats the rank or the tolerance.
        text = open(out, encoding="utf-8").read()
        assert text.count(f"rank {factors.rank}") == 1
        assert text.count("1.000e-06 Ha") == 1

    def test_the_dense_run_has_no_two_body_line(self, run):
        out, _calc = run
        assert "two-body integrals" not in parse_output(out)["electrons"]


class TestBasisPrecisionLine:
    """The storage precision has one owner in the log: ``[BASIS]``."""

    @pytest.mark.parametrize("precision", ["float64", "float32"])
    def test_the_precision_is_written_once_in_the_basis_block(self, tmp_path, precision):
        import re

        out = str(tmp_path / "output.txt")
        atoms = Atoms("H2", positions=[[3, 3, 2.63], [3, 3, 3.37]],
                      cell=[6.0, 6.0, 6.0])
        atoms.calc = Mandacaru(method="rhf", basis="HAO", grid_spacing=0.5,
                               precision=precision, resources={"threads": 1},
                               txt=out)
        atoms.get_potential_energy()
        step = parse_output(out)
        assert step["basis"]["precision"] == precision
        text = open(out, encoding="utf-8").read()
        assert len(re.findall(r"^\s*precision:", text, flags=re.M)) == 1
        lines = text.splitlines()
        start = lines.index("[BASIS]")
        end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("["))
        assert f"precision: {precision}" in "\n".join(lines[start:end])


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
                               grid_spacing=0.30, pool="fermionic", max_iterations=1,
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
        # The first_zeta scheme is the default, and the line names it rather
        # than printing a bare number: a tail *norm* and a squared-norm
        # fraction are not comparable.
        assert block["zeta_split"].startswith(
            "first_zeta: tail norms 0.16, 0.3, 0.6 (norm")
        # The shell follows the confinement: the Gaussian when confined.
        assert block["polarization_method"].startswith("gaussian")
        assert block["energy_shift"] == "0.1 eV"
        assert "A = 12 Ha" in block["confinement_potential"]
        # On by default for PAW-LCAO, with the cutoff the grid resolved it to.
        assert "Bohr^-1" in block["filter_cutoff"]
        assert block["filter_cutoff"].endswith("filtered (auto: 1 x Nyquist)")
        assert block["local_potential"].startswith("range-separated")
        assert block["dataset_xc"].startswith("LDA")
        # The folder is stated once, by name and path; the table names files.
        assert block["paw_dataset"].startswith("lda-sr (")
        assert block["basis_functions"] == "4"

    def test_the_tables_read_back_typed(self, paw):
        from mandacaru.pseudopotentials import get_paw
        from mandacaru.pseudopotentials.confinement import confined_orbital

        out, _calc = paw
        block = parse_output(out)["basis"]
        (dataset,) = block["datasets"]
        assert dataset["symbol"] == "H" and dataset["Z_ion"] == 1
        assert dataset["source"] in ("H.parquet", "H.json")
        # The digest of the bytes that were read, not of the name.
        import hashlib
        with open(get_paw("H").source, "rb") as handle:
            assert dataset["sha256"] == hashlib.sha256(handle.read()).hexdigest()
        # One entry per channel of the dataset: since 2026-10-25 hydrogen has
        # an (empty, unconfined) 2p channel beside the 1s.  The basis itself
        # is still the 1s alone: the function count below does not move.
        orbital, p_channel = block["orbitals"]
        assert p_channel["symbol"] == "H" and p_channel["l"] == 1
        assert p_channel["r_c_Bohr"] == "unconfined"
        assert p_channel["shift_eV"] == 0.0
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
                               grid_spacing=0.30, pool="fermionic", max_iterations=1,
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
                               grid_spacing=0.35, pool="fermionic", max_iterations=1,
                               txt=out)
        atoms.get_potential_energy()
        block = parse_output(out)["basis"]
        assert block["name"] == "HAO" and block["family"] == "all-electron"
        assert "virtual_orbitals" in block["options"]
        assert block["basis_functions"] == "4"
        assert block["functions"] == [
            {"symbol": "H", "atoms": 2, "functions_per_atom": 2}]


class TestOptimizationSetupBlock:
    """``[OPTIMIZATION SETUP]`` says how the run was configured.

    The ADAPT screening gradient can be computed three ways and the choice
    changes both the cost and the truncation error of the ``|grad|`` column, so
    it is a first-class line of the block -- ``gradient_method``, next to the
    ``convergence_gradient`` it is compared against -- and not, as it was, the last of
    the pool lines under the name ``screening_gradient``.
    """

    #: The block's lines, in the order they are written: the classical
    #: optimizer, its budget and the convergence criteria, the gradient group,
    #: the pool, the loop's starting point.  Lineage (a resumed run) closes it.
    FIELDS = ("classical_optimizer", "max_iterations", "convergence",
              # How the [ITERATIONS] table marks a value meeting its threshold.
              "convergence_marker",
              "gradient_method", "gradient_formula", "convergence_gradient",
              "gradient_units",
              "pool", "pool_class", "pool_size",
              # How the ansatz grows, how a growth step is optimized and where
              # it executes: the log is the only record when the
              # standard-output header is off.
              "growth", "pruning", "screening",
              "reoptimize_all_parameters", "parameter_gradient",
              "state_vector_backend", "device",
              "backend_provider", "circuit_execution", "shots",
              "circuit_profiling",
              "energy_unit", "reference_energy_eV")

    def _log(self, hamiltonian, tmp_path, **kwargs):
        out = str(tmp_path / "output.txt")
        Mandacaru(method="adapt-vqe", hamiltonian=hamiltonian,
                  pool="fermionic", num_particles=(1, 1),
                  n_spatial_orbitals=2, profile=False, max_iterations=2,
                  convergence={"gradient": 1e-3}, txt=out, trace=False,
                  **kwargs).run()
        return out


    def test_an_old_log_still_reads_back(self, tmp_path):
        """Writers stop producing a retired key; readers stay tolerant of it."""
        old = tmp_path / "old_output.txt"
        old.write_text("[OPTIMIZATION SETUP]\n"
                       "    classical_optimizer: COBYLA\n"
                       "    convergence_gradient: 0.001\n"
                       "    gradient_units: Hartree\n"
                       "    screening_gradient: parameter-shift\n",
                       encoding="utf-8")
        setup = parse_output(str(old))["setup"]
        assert setup["gradient_method"] == "parameter-shift"
        # The fixture above is a *file*, so it reads back what it says, not
        # what today's default would have written.
        assert setup["classical_optimizer"] == "COBYLA"


class TestNothingIsSaidTwice:
    """Each fact has one owner: a value written in two blocks can disagree
    with itself, and a reader has to decide which one to believe."""

    @pytest.fixture(scope="class")
    def text(self, tmp_path_factory):
        from mandacaru import Mandacaru
        out = str(tmp_path_factory.mktemp("owners") / "output.txt")
        atoms = Atoms("H2", positions=[[3, 3, 2.63], [3, 3, 3.37]],
                      cell=[6.0, 6.0, 6.0])
        atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", grid_spacing=0.35,
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
        atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", grid_spacing=0.35,
                               pool="fermionic", max_iterations=4,
                               convergence={"gradient": 1e-3}, txt=out)
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
        atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", grid_spacing=0.35,
                               pool="fermionic", max_iterations=4,
                               convergence={"gradient": 1e-3}, txt=out)
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
        atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", grid_spacing=0.35,
                               pool="fermionic", max_iterations=4,
                               convergence={"gradient": 1e-3}, txt=out)
        opt = BFGS(atoms, logfile=None)
        opt.run(fmax=0.05, steps=2)
        assert atoms.calc.write_optimization_summary(optimizer=opt) is True
        return out, atoms, opt

    def test_the_summary_describes_the_whole_trajectory(self, relaxed):
        out, atoms, _opt = relaxed
        block = parse_output(out)["optimization"]
        steps = parse_output(out)["steps"]
        assert block["geometry_steps"] == len(steps)
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
        # A second call finds it already written.
        assert atoms.calc.write_optimization_summary(optimizer=opt) is False
        assert open(out, encoding="utf-8").read() == text

    def test_the_exit_hook_finds_it_already_written(self, relaxed):
        out, atoms, _opt = relaxed
        text = open(out, encoding="utf-8").read()
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
        atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", grid_spacing=0.35,
                               pool="fermionic", max_iterations=2,
                               convergence={"gradient": 1e-3}, txt=out)
        atoms.get_forces()
        # One geometry is not a trajectory; there is nothing to summarize.
        assert atoms.calc.write_optimization_summary() is False
        assert "[GEOMETRY OPTIMIZATION SUMMARY]" not in open(out).read()
        assert atoms.calc.write_optimization_summary(force=True) is True


class TestStandardOutputIsTheASETable:
    """stdout carries the evolution of energies and forces, nothing else.

    The report has a destination -- ``txt=<path>`` -- so standard output is
    left to what an ASE optimizer prints there, the usual split for
    ``txt=``.  Without a log file standard output is the only destination there
    is, so **the same blocks** are printed to it.
    """

    @staticmethod
    def _run(tmp_path, capsys, **options):
        from mandacaru import Mandacaru

        atoms = Atoms("H2", positions=[[3, 3, 2.6], [3, 3, 3.4]],
                      cell=[6.0, 6.0, 6.0])
        atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", grid_spacing=0.35,
                               pool="fermionic", max_iterations=2,
                               convergence={"gradient": 1e-3}, **options)
        atoms.get_potential_energy()
        return capsys.readouterr().out

    def test_a_log_file_silences_the_trace(self, tmp_path, capsys):
        out = self._run(tmp_path, capsys, txt=str(tmp_path / "output.txt"))
        for noise in ("ADAPT-VQE", "operator pool", "Timings", "iter"):
            assert noise not in out, noise
        assert out.strip() == ""

    def test_without_a_log_file_the_trace_is_printed(self, tmp_path, capsys):
        out = self._run(tmp_path, capsys, trace=True)
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
        printed = self._run(tmp_path, capsys, trace=True)
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
        assert "[ITERATIONS]" in printed
        quiet = self._run(tmp_path, capsys, trace=False)
        assert quiet.strip() == ""

    def test_trace_must_be_a_boolean(self):
        from mandacaru import Mandacaru

        with pytest.raises(TypeError, match="require booleans"):
            Mandacaru(method="adapt-vqe", trace="yes")

    def test_verbose_is_refused_and_points_at_trace(self):
        from mandacaru import Mandacaru

        with pytest.raises(TypeError, match="verbose"):
            Mandacaru(method="adapt-vqe", verbose=False)

    def test_the_log_still_has_everything(self, tmp_path, capsys):
        """Silencing stdout must route the detail, not discard it."""
        out = str(tmp_path / "output.txt")
        self._run(tmp_path, capsys, txt=out)
        parsed = parse_output(out)
        assert parsed["iterations"] and parsed["summary"]["converged"]
        assert parsed["performance"]["wall_time_s"] > 0


class TestCompletionIsNotAssumed:
    def test_the_exit_hook_does_not_claim_completion(self, tmp_path):
        out = str(tmp_path / "output.txt")
        atoms = h2()
        atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", grid_spacing=0.45,
                               pool="fermionic", max_iterations=2, profile=False,
                               convergence={"gradient": 1e-3}, txt=out)
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
        atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", grid_spacing=0.45,
                               pool="fermionic", max_iterations=2, profile=False,
                               convergence={"gradient": 1e-3}, txt=out)
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
        from mandacaru.utils.logging import Logger, log_steps

        path = str(tmp_path / "deep" / "nested" / "output.txt")
        with Logger(path, n_qubits=4) as logger:
            logger.write_system()
            assert logger.step == 1
        assert log_steps(path) == 1
        assert os.path.exists(path)


class TestParserPreservesData:
    @pytest.fixture
    def written(self, tmp_path):
        from types import SimpleNamespace

        from mandacaru.utils.logging import Logger

        path = str(tmp_path / "output.txt")
        cell = np.array([[4.0, 0.0, 0.0], [0.5, 5.0, 0.0], [0.0, 0.0, 6.0]])
        pool = [SimpleNamespace(label="custom operator with spaces",
                                kind="double",
                                generator=PauliSum({"XXXX": 0.5j}))]
        with Logger(path, n_qubits=4) as logger:
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
        assert parse_output(out)["setup"]["gradient_units"] == "Ha/rad"

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
        with Logger(path, n_qubits=4) as logger:
            logger.write_optimizer_setup("BFGS", reference_energy=-1.0)
            logger.write_iteration(1, pool, [0.25], 0, None, -1.25, 1)
            logger.write_iteration(2, pool, [3.2e-7], 0, None, -1.5, 2)
        text = open(path, encoding="utf-8").read()
        table = text.split("[ITERATIONS]", 1)[1].splitlines()
        heading = re.split(r" {4,}", table[1].strip())
        assert re.fullmatch(r"\d\d:\d\d:\d\d", table[3].split()[1])
        return heading, parse_output(path)["iterations"], table

    def test_the_order_puts_the_time_second_and_drops_the_type(self, rows):
        import re

        heading, _entries, table = rows
        assert heading[:2] == ["iter", "time (HH:MM:SS)"]
        assert heading.index("dE (eV)") < heading.index("|grad| (eV)")
        assert "type" not in heading
        assert re.split(r" {4,}", table[1].strip()) == [
            "iter", "time (HH:MM:SS)", "energy (eV)", "dE (eV)", "|grad| (eV)", "steps",
            "1q", "cnot", "depth", "operator"]
        raw_heading = table[1]
        assert (raw_heading.index("dE (eV)")
                - raw_heading.index("energy (eV)") - len("energy (eV)")) == 12

    def test_de_is_from_the_previous_row_and_first_from_the_reference(
            self, rows):
        _heading, entries, _table = rows
        assert entries[0]["delta_energy"] == pytest.approx(-0.25)
        assert entries[1]["delta_energy"] == pytest.approx(-0.25)
        assert entries[0]["time"] and entries[1]["time"]

    def test_the_gradient_has_six_decimals(self, rows):
        _heading, entries, table = rows
        cells = [line.split() for line in table[3:5]]
        column = rows[0].index("|grad| (eV)")
        assert [row[column] for row in cells] == ["6.802847", "0.000009"]
        assert entries[0]["max_gradient"] == pytest.approx(0.25)

    @pytest.mark.parametrize("label", ["long operator " * 20, "operator\nsecond line"])
    def test_long_operator_names_keep_one_line_per_iteration(self, tmp_path, label):
        path = tmp_path / "operators.txt"
        pool = [SimpleNamespace(label=label + str(i)) for i in range(2)]
        with Logger(str(path), n_qubits=4) as logger:
            for i in range(2):
                logger.write_iteration(i + 1, pool, [0.1, 0.2], i, None, -1.0, i + 1)
        table = path.read_text().split("[ITERATIONS]\n")[1].splitlines()
        assert len(table) == 4  # heading, rule and exactly two data rows
        assert [line.split()[-1] for line in table[2:]] == ["pool[0]", "pool[1]"]
        assert [row["selected_operator"] for row in parse_output(str(path))["iterations"]] == [
            "pool[0]", "pool[1]"]
        assert pool[0].label == label + "0"  # display never changes the full label


