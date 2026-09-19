# -*- coding: utf-8 -*-
# file: test_adapt_args_gradient.py

"""ADAPTVQE argument surface, gradient strategies, the device registry and the
FullAtomicOrbitals (FAO) basis.

Covers the features added on top of the ADAPT-VQE driver:

* selectable ``gradient`` strategies -- ``"analytic"`` (the default),
  ``"finite_difference"`` and ``"parameter-shift"``, the latter two matching the
  exact analytic gradient they estimate;
* the ``ADAPTVQE`` argument surface (``pool``, ``basis``, ``mapping``,
  ``gradient``, ``device``) and its basis-driven Hamiltonian builder;
* the ``device`` registry (AER_simulator vs the reserved ibm-quantum).
"""

import numpy as np
import pytest
from ase import Atoms

from mandacaru.algorithms import Mandacaru
from mandacaru.circuits import AdaptAnsatz
from mandacaru.backends import available_devices, is_simulator, normalize_device
from mandacaru.integrals import Grid


# --------------------------------------------------------------------------- #
# The FullAtomicOrbitals (FAO) rename.
# --------------------------------------------------------------------------- #

class TestFAONaming:
    def test_fao_names_exist(self):
        from mandacaru.basis import FAOBasisSet, FullAtomicOrbital
        from mandacaru.core import MolecularIntegrals, minimal_fao_basis
        assert FullAtomicOrbital is not None
        assert FAOBasisSet is not None
        assert MolecularIntegrals is not None
        assert minimal_fao_basis is not None

    def test_factory_builds_fao(self):
        from mandacaru.basis import BasisSet, FAOBasisSet
        assert isinstance(BasisSet.build("FAO"), FAOBasisSet)
        assert BasisSet.build("STO-3G").name == "STO-3G"
        with pytest.raises(ValueError):
            BasisSet.build("hydrogenic")      # the alias no longer resolves


# --------------------------------------------------------------------------- #
# Shared H2 Hamiltonian (FAO / MO basis).
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def h2_hamiltonian():
    from mandacaru.core import MolecularIntegrals, minimal_fao_basis
    R = 0.74
    nuclei = [(1.0, np.array([0.0, 0.0, -R / 2])),
              (1.0, np.array([0.0, 0.0, +R / 2]))]
    grid = Grid(center=[0, 0, 0], box_size=6.0, h=0.25)
    integrals = MolecularIntegrals(nuclei, minimal_fao_basis(nuclei), grid)
    return integrals.molecular_hamiltonian(mo_basis=True, n_electrons=2)


# --------------------------------------------------------------------------- #
# Gradient strategies.
# --------------------------------------------------------------------------- #

class TestGradientStrategies:
    @pytest.mark.parametrize("pool", ["fermionic", "qubit", "qeb", "ceo"])
    def test_finite_difference_matches_analytic(self, h2_hamiltonian, pool):
        adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                          pool=pool, num_particles=(1, 1),
                          n_spatial_orbitals=2, profile=False,
                          gradient="finite_difference")
        psi = AdaptAnsatz(adapt.n_qubits, adapt.pool.occupied_orbitals).state(
            np.zeros(0))
        g_an = adapt._analytic_gradients(psi)
        g_fd = adapt._finite_difference_gradients(psi)
        np.testing.assert_allclose(g_fd, g_an, atol=1e-6)

    @pytest.mark.parametrize("pool", ["fermionic", "qubit", "qeb", "ceo"])
    def test_parameter_shift_is_exact(self, h2_hamiltonian, pool):
        adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                          pool=pool, num_particles=(1, 1),
                          n_spatial_orbitals=2, profile=False,
                          gradient="parameter-shift")
        psi = AdaptAnsatz(adapt.n_qubits, adapt.pool.occupied_orbitals).state(
            np.zeros(0))
        g_an = adapt._analytic_gradients(psi)
        g_ps = adapt._parameter_shift_gradients(psi)
        np.testing.assert_allclose(g_ps, g_an, atol=1e-9)

    def test_analytic_is_the_default_and_screens_analytically(
            self, h2_hamiltonian):
        """The exact derivative, not an estimate of it, out of the box."""
        adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                          pool="ceo", num_particles=(1, 1),
                          n_spatial_orbitals=2, profile=False)
        assert adapt.gradient == "analytic"
        psi = AdaptAnsatz(adapt.n_qubits, adapt.pool.occupied_orbitals).state(
            np.zeros(0))
        np.testing.assert_allclose(adapt._gradients(psi),
                                   adapt._analytic_gradients(psi), atol=0)

    def test_the_eigendecomposition_is_built_only_when_asked_for(
            self, h2_hamiltonian):
        """|pool| dense diagonalizations are the shift estimators' cost alone."""
        adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                          pool="fermionic", num_particles=(1, 1),
                          n_spatial_orbitals=2, profile=False)
        psi = AdaptAnsatz(adapt.n_qubits, adapt.pool.occupied_orbitals).state(
            np.zeros(0))
        adapt._gradients(psi)
        assert adapt._pool_eig is None
        adapt._parameter_shift_gradients(psi)
        assert len(adapt._pool_eig) == len(adapt._pool_matrices)

    def test_every_gradient_reaches_fci(self, h2_hamiltonian):
        m = h2_hamiltonian.map_to_qubits("jordan_wigner").to_matrix()
        exact = float(np.linalg.eigvalsh(0.5 * (m + m.conj().T)).min())
        for grad in ("analytic", "finite_difference", "parameter-shift"):
            adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                              pool="ceo", num_particles=(1, 1),
                              n_spatial_orbitals=2, profile=False,
                              gradient=grad, max_iterations=10,
                              gradient_tolerance=1e-4)
            res = adapt.run()
            assert abs(res.in_units("Ha") - exact) < 1e-4, grad

    def test_invalid_gradient_rejected(self, h2_hamiltonian):
        with pytest.raises(ValueError):
            Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                      pool="ceo", num_particles=(1, 1), n_spatial_orbitals=2,
                      gradient="nope")


# --------------------------------------------------------------------------- #
# Argument surface + basis-driven builder.
# --------------------------------------------------------------------------- #

class TestArgumentSurface:
    @pytest.mark.parametrize("pool", ["ceo", "fermionic", "qubit", "qeb"])
    def test_pool_options(self, pool):
        atoms = Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]])
        atoms.calc = Mandacaru(method="adapt-vqe", pool=pool, basis="FAO",
                               grid=Grid(center=[0, 0, 0], box_size=6.0, h=0.3),
                               max_iterations=8, gradient_tolerance=1e-3)
        assert np.isfinite(atoms.get_total_energy())

    @pytest.mark.parametrize("mapping",
                             ["jordan_wigner", "parity", "bravyi_kitaev"])
    def test_mapping_options_reach_fci(self, mapping):
        atoms = Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]])
        atoms.calc = Mandacaru(method="adapt-vqe", pool="fermionic",
                               basis="FAO", mapping=mapping,
                               grid=Grid(center=[0, 0, 0], box_size=6.0, h=0.25),
                               max_iterations=8, gradient_tolerance=1e-3)
        energy_ev = atoms.get_total_energy()
        h = atoms.calc.hamiltonian.to_matrix()
        exact = float(np.linalg.eigvalsh(0.5 * (h + h.conj().T)).min())
        exact_ev = exact * 27.211386245988
        assert abs(energy_ev - exact_ev) < 1e-3, mapping

    def test_run_defaults_come_from_constructor(self, h2_hamiltonian):
        # max_iterations / gradient_tolerance / output are constructor args and
        # supply the defaults for run().
        adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                          pool="ceo", num_particles=(1, 1),
                          n_spatial_orbitals=2, profile=False,
                          max_iterations=3, gradient_tolerance=1e-2)
        assert adapt.max_iterations == 3
        assert adapt.gradient_tolerance == 1e-2
        res = adapt.run()                       # no args -> uses the defaults
        assert res.num_operators <= 3

    def test_output_constructor_arg_writes_file(self, h2_hamiltonian, tmp_path):
        from mandacaru.utils import parse_output
        out = str(tmp_path / "output.txt")
        adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                          pool="fermionic", num_particles=(1, 1),
                          n_spatial_orbitals=2, profile=False,
                          max_iterations=4, gradient_tolerance=1e-3,
                          output=out)
        adapt.run()                             # output taken from constructor
        parsed = parse_output(out)
        assert parsed["setup"]["classical_optimizer"] == "COBYLA"
        assert len(parsed["iterations"]) >= 1

    def test_basis_option_sets_qubit_count(self):
        # FAO on LiH -> Li{1s,2s} + H{1s} = 3 orbitals -> 6 qubits.
        atoms = Atoms("LiH", positions=[[0, 0, -0.8], [0, 0, 0.8]])
        atoms.calc = Mandacaru(method="adapt-vqe", pool="ceo", basis="FAO",
                               grid=Grid(center=[0, 0, 0], box_size=7.0, h=0.3),
                               max_iterations=4, gradient_tolerance=1e-2)
        atoms.get_total_energy()
        assert atoms.calc.n_qubits == 6
        assert atoms.calc.num_particles == (2, 2)


# --------------------------------------------------------------------------- #
# Optimizer selection.
# --------------------------------------------------------------------------- #

class TestOptimizerOption:
    def test_default_is_cobyla(self, h2_hamiltonian):
        adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                          pool="ceo", num_particles=(1, 1),
                          n_spatial_orbitals=2, profile=False)
        assert adapt.optimizer.method == "COBYLA"

    @pytest.mark.parametrize(
        "name",
        ["SPSA", "COBYLA", "Nelder-Mead", "SLSQP", "Adam", "L-BFGS-B"])
    def test_named_optimizers_build(self, h2_hamiltonian, name):
        adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                          pool="ceo", num_particles=(1, 1),
                          n_spatial_orbitals=2, profile=False, optimizer=name)
        assert adapt.optimizer.method == name

    def test_optimizer_instance_passthrough(self, h2_hamiltonian):
        from mandacaru.optimizers import Optimizer
        opt = Optimizer("L-BFGS-B", maxiter=500)
        adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                          pool="ceo", num_particles=(1, 1),
                          n_spatial_orbitals=2, profile=False, optimizer=opt)
        assert adapt.optimizer is opt

    def test_unknown_optimizer_rejected(self, h2_hamiltonian):
        with pytest.raises(ValueError):
            Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                      pool="ceo", num_particles=(1, 1), n_spatial_orbitals=2,
                      optimizer="nope")


# --------------------------------------------------------------------------- #
# Standard-output Pauli-string trace.
# --------------------------------------------------------------------------- #

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
        assert "qubit Hamiltonian" in out and "15 Pauli terms" in out
        assert "* ZIII" not in out
        # Nor is the selected operator's generator dumped per iteration.
        assert "ansatz operator (Pauli strings)" not in out
        assert "[iter 1]" not in out
        # The operator is still reachable programmatically.
        assert "ZIII" in adapt.hamiltonian.simplify().terms
        # Nor is the pool listed: its name and size are all the trace carries.
        assert "operator pool" in out
        assert f"ceo ({len(adapt._pool_ops)} operators)" in out
        assert "operator_pool" not in out

    def test_iterations_are_single_aligned_lines(self, h2_hamiltonian, capsys,
                                                monkeypatch):
        monkeypatch.setenv("COLUMNS", "100")
        adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                          pool="ceo", num_particles=(1, 1),
                          n_spatial_orbitals=2, profile=True, trace=True,
                          max_iterations=3, gradient_tolerance=1e-6)
        result = adapt.run(log_expressivity=True)   # the `expr` column is opt-in
        out = capsys.readouterr().out

        # A column heading precedes the per-iteration rows: one column per
        # property computed at that step.  The headings are read off the table
        # rather than hard-coded, because which optional columns fit depends on
        # the terminal width (see _iteration_layout).
        lines = out.splitlines()
        heading = next(line for line in lines if line.split()[:1] == ["iter"])
        # "E (eV)" is one column but two whitespace-separated tokens.
        columns = heading.replace("E (eV)", "energy").split()
        for column in ("iter", "|grad|", "expr", "cnot", "1q", "depth",
                       "type", "operator"):
            assert column in columns

        index = lines.index(heading)
        # One line per grown operator, each carrying every column.
        rows = [line for line in lines[index + 2:]
                if line.strip() and line.split()[0].isdigit()]
        assert len(rows) == result.num_operators
        for index, row in enumerate(rows, start=1):
            fields = row.split()
            assert int(fields[0]) == index
            assert len(fields) == len(columns)
            # A long label is elided (MAX_LABEL_WIDTH) so it cannot cost the
            # data columns; match its stem against the grown operators.
            stem = fields[-1].rstrip("\u2026")
            assert any(op.startswith(stem) for op in result.operators)
            cell = dict(zip(columns, fields))
            # The `type` cell drops the pool-name prefix
            # ("fermionic-double" -> "double").  A pool whose kind *is* the
            # pool name (CEO tags everything "ceo") would make the column a
            # constant copy of the header, so there it is read off the label.
            step = result.iterations[index - 1]
            shown = cell["type"]
            if step.operator_kind.endswith(shown):
                pass                                   # prefix stripped
            else:
                assert shown in ("single", "double")
                double = "{D(" in step.operator_label \
                    or step.operator_label.startswith("D(")
                assert double == (shown == "double")
            assert float(cell["expr"]) >= 0.0
            assert int(cell["1q"]) > 0
            assert int(cell["cnot"]) > 0 and int(cell["depth"]) > 0
        # The rows are column-aligned: the operator column starts at one offset.
        starts = {row.index(row.split()[-1]) for row in rows}
        assert len(starts) == 1

    def test_every_line_fits_the_terminal(self, h2_hamiltonian, capsys,
                                          monkeypatch):
        """One iteration is one *line*: a wrapped row is not a row.

        The table was 126 characters wide in an 80-column terminal, so every
        iteration spilled onto two visual lines -- the columns were right and
        the output was still unreadable.  ``_iteration_layout`` now drops the
        derivable columns (``npar``, then ``dE``) until the row fits.
        """
        monkeypatch.setenv("COLUMNS", "80")
        adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                          pool="ceo", num_particles=(1, 1),
                          n_spatial_orbitals=2, profile=True, trace=True,
                          max_iterations=3, gradient_tolerance=1e-6)
        adapt.run()
        lines = capsys.readouterr().out.splitlines()
        # From the run banner onward: the start-up banner prints the working
        # directory, whose length is the user's path, not ours.
        start = next(i for i, line in enumerate(lines)
                     if line.startswith("=" * 70))
        for line in lines[start:]:
            assert len(line) <= 80, f"{len(line)} chars: {line!r}"

    def test_a_wide_terminal_keeps_every_column(self, h2_hamiltonian, capsys,
                                                monkeypatch):
        monkeypatch.setenv("COLUMNS", "200")
        adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                          pool="ceo", num_particles=(1, 1),
                          n_spatial_orbitals=2, profile=True, trace=True,
                          max_iterations=2, gradient_tolerance=1e-6)
        adapt.run(log_expressivity=True)            # the `expr` column is opt-in
        out = capsys.readouterr().out
        heading = next(line for line in out.splitlines()
                       if line.split()[:1] == ["iter"])
        assert set(heading.replace("E (eV)", "energy").split()) == {
            "iter", "|grad|", "energy", "dE", "expr", "cnot", "1q",
            "depth", "type", "operator"}

    def test_verbose_false_is_silent(self, h2_hamiltonian, capsys):
        adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                          pool="ceo", num_particles=(1, 1),
                          n_spatial_orbitals=2, profile=False, trace=False,
                          max_iterations=3, gradient_tolerance=1e-6)
        adapt.run()
        assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------- #
# Timing / memory / cores profiling (requirements 2-3).
# --------------------------------------------------------------------------- #

class TestADAPTProfiling:
    def test_result_carries_stage_timings(self, h2_hamiltonian):
        adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                          pool="ceo", num_particles=(1, 1),
                          n_spatial_orbitals=2, profile=False, trace=False,
                          max_iterations=3, gradient_tolerance=1e-6)
        res = adapt.run()
        t = res.timings
        assert t is not None
        assert "gradient screening" in t["stages_s"]
        assert "parameter optimization" in t["stages_s"]
        assert t["wall_time_s"] is not None
        assert t["peak_memory_mb"] > 0.0

    def test_summary_prints_timings_cores_memory(self, h2_hamiltonian, capsys):
        adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                          pool="ceo", num_particles=(1, 1),
                          n_spatial_orbitals=2, profile=False, trace=True,
                          max_iterations=3, gradient_tolerance=1e-6)
        adapt.run()
        out = capsys.readouterr().out
        assert "Timings (wall-clock)" in out
        assert "gradient screening" in out
        assert "parameter optimization" in out
        assert "cores (OpenMP threads)" in out
        assert "peak memory" in out

    def test_calculator_summary_includes_integration(self, capsys):
        atoms = Atoms("H2", positions=[[3, 3, 2.63], [3, 3, 3.37]],
                      cell=[[6, 0, 0], [0, 6, 0], [0, 0, 6]], pbc=True)
        atoms.calc = Mandacaru(method="adapt-vqe", pool="ceo", basis="FAO",
                               h=0.4, trace=True, max_iterations=4,
                               gradient_tolerance=1e-3)
        atoms.get_total_energy()
        out = capsys.readouterr().out
        assert "integration:" in out
        assert atoms.calc.result.integration_profile is not None


# --------------------------------------------------------------------------- #
# Monkhorst-Pack k-points and placement-invariance (PBC-aware grid).
# --------------------------------------------------------------------------- #

class TestKPoints:
    def test_default_is_gamma(self):
        adapt = Mandacaru(method="adapt-vqe", pool="ceo", basis="FAO")
        assert adapt.kpts == (1, 1, 1)
        assert len(adapt.kpoints) == 1
        np.testing.assert_allclose(adapt.kpoints[0], [0.0, 0.0, 0.0])

    def test_mesh_generated_via_ase_monkhorst_pack(self):
        from ase.dft.kpoints import monkhorst_pack
        adapt = Mandacaru(method="adapt-vqe", pool="ceo", basis="FAO",
                          kpts=(2, 2, 1))
        np.testing.assert_allclose(adapt.kpoints, monkhorst_pack((2, 2, 1)))

    def test_non_gamma_rejected_at_run(self, h2_hamiltonian):
        adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                          pool="ceo", num_particles=(1, 1),
                          n_spatial_orbitals=2, kpts=(2, 1, 1))
        with pytest.raises(NotImplementedError, match="Monkhorst-Pack"):
            adapt.run()

    def test_invalid_kpts_rejected(self):
        with pytest.raises(ValueError):
            Mandacaru(method="adapt-vqe", pool="ceo", basis="FAO", kpts=(1, 1))       # not length-3

    def test_dict_spec_with_gamma_centering(self):
        # ASE dict form {"size": ..., "gamma": True}: Gamma-centered mesh.
        adapt = Mandacaru(method="adapt-vqe", pool="ceo", basis="FAO",
                          kpts={"size": (2, 2, 1), "gamma": True})
        assert adapt.kpts == (2, 2, 1)
        assert adapt.kpts_gamma is True
        # Gamma-centering shifts the even-axis mesh so it includes the Gamma point.
        assert any(np.allclose(k, [0, 0, 0]) for k in adapt.kpoints)

    def test_dict_gamma_only(self):
        adapt = Mandacaru(method="adapt-vqe", pool="ceo", basis="FAO",
                          kpts={"size": (1, 1, 1), "gamma": True})
        assert len(adapt.kpoints) == 1 and adapt.kpts_gamma is True


class TestSpinAndInitialState:
    def test_spin_defaults_false(self):
        assert Mandacaru(method="adapt-vqe", pool="ceo", basis="FAO").spin is False

    def test_spin_flag_stored(self):
        assert Mandacaru(method="adapt-vqe", pool="ceo", basis="FAO",
                         spin=True).spin is True

    def test_even_electron_spin_polarized_matches_closed_shell(self):
        # For a singlet (even electrons) spin-polarized == closed-shell.
        atoms = Atoms("H2", positions=[[3, 3, 2.63], [3, 3, 3.37]],
                      cell=[[6, 0, 0], [0, 6, 0], [0, 0, 6]], pbc=True)

        def energy(spin):
            atoms.calc = Mandacaru(method="adapt-vqe", pool="ceo", basis="FAO",
                                   h=0.4, spin=spin, trace=False,
                                   max_iterations=6, gradient_tolerance=1e-3)
            return atoms.get_total_energy()

        assert energy(False) == pytest.approx(energy(True), abs=1e-6)

    def test_initial_state_default_is_hartree_fock(self):
        assert Mandacaru(method="adapt-vqe", pool="ceo", basis="FAO").initial_state == "hartree-fock"

    def test_initial_state_none_is_hartree_fock(self):
        assert Mandacaru(method="adapt-vqe", pool="ceo", basis="FAO",
                         initial_state=None).initial_state == "hartree-fock"

    def test_unknown_initial_state_rejected(self):
        with pytest.raises(ValueError):
            Mandacaru(method="adapt-vqe", pool="ceo", basis="FAO",
                      initial_state="random")


class TestPlacementInvariance:
    def test_energy_independent_of_position_in_cell(self):
        # PBC-aware grid: the molecule is centered on itself, so placing H2 at the
        # cell corner vs. the cell center gives the same Hamiltonian and energy.
        def energy(pos):
            atoms = Atoms("H2", positions=pos,
                          cell=[[8, 0, 0], [0, 8, 0], [0, 0, 8]], pbc=True)
            atoms.calc = Mandacaru(method="adapt-vqe", pool="ceo", basis="FAO",
                                   h=0.35, trace=False, max_iterations=6,
                                   gradient_tolerance=1e-3)
            return atoms.get_total_energy()

        corner = energy([[0, 0, -0.37], [0, 0, 0.37]])
        center = energy([[4, 4, 3.63], [4, 4, 4.37]])
        assert corner == pytest.approx(center, abs=1e-6)


# --------------------------------------------------------------------------- #
# Device registry.
# --------------------------------------------------------------------------- #

class TestDeviceRegistry:
    def test_aer_is_default_and_simulator(self):
        adapt = Mandacaru(method="adapt-vqe", pool="ceo", basis="FAO")
        assert adapt.device == "AER_simulator"
        assert is_simulator("AER_simulator")

    def test_aliases_normalize(self):
        assert normalize_device("aer") == "AER_simulator"
        assert normalize_device("statevector") == "AER_simulator"
        assert normalize_device("ibmq") == "ibm-quantum"

    def test_unknown_device_rejected(self):
        with pytest.raises(ValueError):
            Mandacaru(method="adapt-vqe", pool="ceo", device="quantum-thing")

    def test_ibm_quantum_listed_but_not_simulator(self):
        assert "ibm-quantum" in available_devices()
        assert not is_simulator("ibm-quantum")
