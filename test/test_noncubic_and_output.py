# -*- coding: utf-8 -*-
# file: test_noncubic_and_output.py

"""Non-cubic (cell-aware) grids, ASE integration and the ADAPT output.txt protocol.

Covers the three features added on top of the cubic integral core:

* :class:`~mandacaru.integrals.Grid` accepting an anisotropic ``box_size`` or a
  full ``cell`` tensor, with the integral engine still recovering the reference
  physics (H 1s on-site repulsion = 5/8 Ha) on a non-cubic box;
* the real-space grid generated directly from an ASE ``Atoms`` unit cell
  (:func:`mandacaru.algorithms._hamiltonian_from_atoms.grid_from_cell`);
* :class:`~mandacaru.algorithms.ADAPTVQE` writing a structured, live-parseable
  ``output.txt`` as the ADAPT loop runs.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from mandacaru.algorithms import Mandacaru
from mandacaru.algorithms._hamiltonian_from_atoms import grid_from_cell
from mandacaru.basis import FullAtomicOrbital
from mandacaru.core import MolecularIntegrals, minimal_fao_basis
from mandacaru.integrals import Grid, IntegralEngine
from mandacaru.utils import AdaptOutputLogger, parse_output


# --------------------------------------------------------------------------- #
# Non-cubic grids.
# --------------------------------------------------------------------------- #

def _eri_00(grid):
    """H 1s on-site repulsion <00|00> on ``grid`` (Hartree)."""
    orb = FullAtomicOrbital(1, 0, 0, Z=1.0, center=[0.0, 0.0, 0.0], units="bohr")
    eng = IntegralEngine([orb], grid)
    return float(np.real(eng.two_body(method="fft", energy_units="Ha")[0, 0, 0, 0]))


class TestNonCubicGrid:
    def test_cubic_backward_compatible(self):
        g = Grid(center=[0, 0, 0], box_size=8.0, h=0.20, units="bohr")
        assert g.is_cubic
        assert g.shape == (g.points, g.points, g.points)
        assert g.size == g.points ** 3

    def test_anisotropic_box_is_non_cubic(self):
        g = Grid(center=[0, 0, 0], box_size=[8.0, 9.0, 7.0], h=0.20, units="bohr")
        assert not g.is_cubic
        nx, ny, nz = g.shape
        assert nx != ny and ny != nz
        # Uniform spacing across axes is required by the finite-difference core.
        assert g.X.shape == (nx, ny, nz)

    def test_uniform_spacing_on_non_cubic(self):
        g = Grid(center=[0, 0, 0], box_size=[8.0, 9.0, 7.0], h=0.20, units="bohr")
        dx = np.diff(g.X[:, 0, 0])
        dy = np.diff(g.Y[0, :, 0])
        dz = np.diff(g.Z[0, 0, :])
        np.testing.assert_allclose(dx, g.dx)
        np.testing.assert_allclose(dy, g.dx)
        np.testing.assert_allclose(dz, g.dx)

    def test_cell_tensor_orthorhombic(self):
        cell = np.diag([16.0, 18.0, 14.0])
        g = Grid(center=[0, 0, 0], box_size=0.0, h=0.20, units="bohr", cell=cell)
        assert not g.is_cubic
        assert g.shape[0] < g.shape[1]     # 16 < 18 -> fewer x nodes than y

    def test_cell_tensor_non_orthogonal_bounding_box(self):
        # A skewed (non-orthogonal) cell is still enclosed by the grid.
        cell = np.array([[16.0, 0.0, 0.0],
                         [2.0, 18.0, 0.0],
                         [0.0, 0.0, 14.0]])
        g = Grid(center=[0, 0, 0], box_size=0.0, h=0.25, units="bohr", cell=cell)
        assert g.size > 0
        assert g.dx == pytest.approx(0.25)

    def test_eri_matches_cubic_reference(self):
        # The H 1s self-repulsion must be 5/8 Ha independent of box shape.
        cubic = _eri_00(Grid(center=[0, 0, 0], box_size=8.0, h=0.20, units="bohr"))
        rect = _eri_00(Grid(center=[0, 0, 0], box_size=[8.0, 9.0, 7.0],
                            h=0.20, units="bohr"))
        cell = _eri_00(Grid(center=[0, 0, 0], box_size=0.0, h=0.20, units="bohr",
                            cell=np.diag([16.0, 18.0, 14.0])))
        assert cubic == pytest.approx(0.625, abs=2e-3)
        assert rect == pytest.approx(0.625, abs=2e-3)
        assert cell == pytest.approx(0.625, abs=2e-3)
        assert rect == pytest.approx(cubic, abs=1e-4)

    def test_kinetic_energy_non_cubic(self):
        # <1s|-1/2 nabla^2|1s> = 1/2 Ha for hydrogen, on a non-cubic grid.
        orb = FullAtomicOrbital(1, 0, 0, Z=1.0, center=[0, 0, 0], units="bohr")
        g = Grid(center=[0, 0, 0], box_size=[8.0, 9.0, 7.0], h=0.20, units="bohr")
        eng = IntegralEngine([orb], g)
        zero_v = lambda x, y, z: np.zeros(np.broadcast(x, y, z).shape)
        T, _ = eng.one_body(zero_v, energy_units="Ha")
        assert float(np.real(T[0, 0])) == pytest.approx(0.5, abs=1e-2)


class TestVaryingResolution:
    """Per-axis (varying) resolution and non-orthogonal cells (C-backend req)."""

    def test_per_axis_resolution_grid(self):
        # A length-3 h sets a different spacing on each axis.
        g = Grid(center=[0, 0, 0], box_size=8.0, h=[0.20, 0.25, 0.30],
                 units="bohr")
        assert (g.dx, g.dy, g.dz) == pytest.approx((0.20, 0.25, 0.30))
        assert g.is_orthorhombic and not g.is_cubic
        assert g.dV == pytest.approx(0.20 * 0.25 * 0.30)
        # Inverse metric is diagonal 1/d^2 for an orthorhombic grid.
        np.testing.assert_allclose(
            g.metric_inverse(),
            np.diag([1 / 0.20 ** 2, 1 / 0.25 ** 2, 1 / 0.30 ** 2]), atol=1e-12)

    def test_kinetic_energy_varying_resolution(self):
        # The generalized (per-axis) Laplacian still recovers <1s|T|1s> = 1/2 Ha.
        orb = FullAtomicOrbital(1, 0, 0, Z=1.0, center=[0, 0, 0], units="bohr")
        g = Grid(center=[0, 0, 0], box_size=8.0, h=[0.20, 0.25, 0.30],
                 units="bohr")
        T, _ = IntegralEngine([orb], g).one_body(
            lambda x, y, z: np.zeros(np.broadcast(x, y, z).shape),
            energy_units="Ha")
        assert float(np.real(T[0, 0])) == pytest.approx(0.5, abs=2e-2)

    def test_skewed_non_orthogonal_grid(self):
        # skew=True samples the actual (non-orthogonal) lattice; the step matrix
        # is non-diagonal and dV = |det(step)|.
        cell = np.array([[16.0, 0.0, 0.0],
                         [3.0, 16.0, 0.0],
                         [0.0, 1.0, 16.0]])
        g = Grid(center=[0, 0, 0], cell=cell, h=0.4, units="bohr", skew=True)
        assert not g.is_orthorhombic
        assert g.dV == pytest.approx(abs(np.linalg.det(g.step)))
        # Cross-term Laplacian recovers the hydrogen kinetic energy.
        orb = FullAtomicOrbital(1, 0, 0, Z=1.0, center=[0, 0, 0], units="bohr")
        T, _ = IntegralEngine([orb], g).one_body(
            lambda x, y, z: np.zeros(np.broadcast(x, y, z).shape),
            energy_units="Ha")
        assert float(np.real(T[0, 0])) == pytest.approx(0.5, abs=3e-2)

    def test_c_and_numpy_kernels_agree(self):
        # The C general kernel and the NumPy fallback must stay in lockstep on
        # anisotropic and skewed grids.
        import mandacaru.integrals._backend as backend
        if not backend.HAS_C_BACKEND:
            pytest.skip("C backend not built")
        orb = FullAtomicOrbital(1, 0, 0, Z=1.0, center=[0, 0, 0.2], units="bohr")
        for g in (Grid(center=[0, 0, 0], box_size=6.0, h=[0.25, 0.30, 0.35],
                       units="bohr"),
                  Grid(center=[0, 0, 0],
                       cell=np.array([[12., 0, 0], [3., 12., 0], [0, 1., 12.]]),
                       h=0.4, units="bohr", skew=True)):
            psi = np.stack([orb.sample(g)])
            vext = np.zeros(g.size)
            saved = backend.HAS_C_BACKEND
            try:
                backend.HAS_C_BACKEND = True
                Tc, _ = backend.one_body_matrices(psi, vext, g)
                backend.HAS_C_BACKEND = False
                Tn, _ = backend.one_body_matrices(psi, vext, g)
            finally:
                backend.HAS_C_BACKEND = saved
            np.testing.assert_allclose(Tc, Tn, atol=1e-10)


# --------------------------------------------------------------------------- #
# Grid generated from an ASE cell.
# --------------------------------------------------------------------------- #

class TestGridFromCell:
    def test_grid_spans_the_cell_and_is_non_cubic(self):
        cell = [[10.0, 0, 0], [0, 12.0, 0], [0, 0, 8.0]]  # Angstrom, orthorhombic
        atoms = Atoms("H", positions=[[5, 6, 4]], cell=cell, pbc=True)
        grid = grid_from_cell(atoms, h=0.25)
        assert not grid.is_cubic          # 10 != 12 != 8
        # h is Angstrom; the grid spacing is that value converted to Bohr.
        assert grid.dx == pytest.approx(0.25 * 1.8897259886)

    def test_grid_requires_a_cell(self):
        atoms = Atoms("H", positions=[[0, 0, 0]])          # no unit cell
        with pytest.raises(ValueError, match="no unit cell"):
            grid_from_cell(atoms, h=0.2)


# --------------------------------------------------------------------------- #
# ADAPT-VQE output.txt protocol.
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def h2_hamiltonian():
    R = 0.74
    nuclei = [(1.0, np.array([0.0, 0.0, -R / 2])),
              (1.0, np.array([0.0, 0.0, +R / 2]))]
    grid = Grid(center=[0, 0, 0], box_size=5.0, h=0.35)
    integrals = MolecularIntegrals(nuclei, minimal_fao_basis(nuclei), grid)
    return integrals.molecular_hamiltonian(mo_basis=True, n_electrons=2)


def _h2_adapt(hamiltonian, **kwargs):
    return Mandacaru(method="adapt-vqe", hamiltonian=hamiltonian,
                     pool="fermionic", num_particles=(1, 1),
                     n_spatial_orbitals=2, profile=False, **kwargs)


class TestAdaptOutputProtocol:
    def test_default_optimizer_is_cobyla(self, h2_hamiltonian):
        # Requirement 6: the default classical optimizer must be COBYLA.
        assert _h2_adapt(h2_hamiltonian).optimizer.method == "COBYLA"

    def test_output_file_written_and_parseable(self, h2_hamiltonian, tmp_path):
        R = 0.74
        geom = Atoms("H2", positions=[[0, 0, -R / 2], [0, 0, R / 2]],
                     cell=[[6, 0, 0], [1, 7, 0], [0, 0, 5]], pbc=True)
        out = str(tmp_path / "output.txt")
        adapt = _h2_adapt(h2_hamiltonian, max_iterations=6,
                          gradient_tolerance=1e-4, output=out)
        result = adapt.run(geometry=geom, log_expressivity=True)

        parsed = parse_output(out)

        # Metadata block: initial geometry + explicit unit-cell parameters.
        assert parsed["system"]["n_atoms"] == "2"
        assert parsed["system"]["cell_present"] == "True"
        assert parsed["system"]["units"] == "Angstrom"       # req 1: default A
        assert "cell_lengths" in parsed["system"]
        assert "cell_angles" in parsed["system"]

        # Optimization setup block -- energies default to eV (requirement 1).
        assert parsed["setup"]["classical_optimizer"] == "COBYLA"
        assert parsed["setup"]["energy_unit"] == "eV"
        assert "reference_energy_eV" in parsed["setup"]

        # One row per iteration, every tracked property a column.
        assert len(parsed["iterations"]) == result.num_operators >= 1
        for index, it in enumerate(parsed["iterations"], start=1):
            assert it["index"] == index
            assert it["selected_operator"]                 # 3. selected operator
            assert it["operator_kind"]
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
                          gradient_tolerance=1e-4, output=out)
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
                          gradient_tolerance=1e-4, output=out)
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
                          gradient_tolerance=1e-4, output=out)
        adapt.run(geometry=geom)
        parsed = parse_output(out)
        assert parsed["system"]["units"] == "Bohr"
        assert parsed["setup"]["energy_unit"] == "Ha"
        assert "reference_energy_Ha" in parsed["setup"]

    def test_runs_without_geometry(self, h2_hamiltonian, tmp_path):
        # The protocol must still write cleanly when no geometry is supplied.
        out = str(tmp_path / "output.txt")
        _h2_adapt(h2_hamiltonian, max_iterations=4, gradient_tolerance=1e-4,
                  output=out).run()
        parsed = parse_output(out)
        assert parsed["system"]["cell_present"] == "False"
        assert parsed["system"]["geometry"] == "(not provided)"


class TestCEOLabels:
    def test_ceo_labels_unique_and_descriptive(self):
        from mandacaru.circuits import build_pool
        labels = [op.label for op in build_pool("ceo", 3, (2, 2)).operators()]
        # Requirement 4: no collisions, and each label names its support.
        assert len(labels) == len(set(labels))
        assert all(lbl.startswith("CEO[q") for lbl in labels)


class TestADAPTVQECalculator:
    def test_class_named_all_caps(self):
        # The driver class is ADAPTVQE (all caps); the old AdaptVQE alias is gone.
        import mandacaru.algorithms as algs
        assert not hasattr(algs, "AdaptVQE")
        assert not hasattr(algs, "AdaptVQEResult")

    def test_ase_calculator_get_total_energy(self, tmp_path):
        # Requirement 7: attach ADAPTVQE as an ASE calculator; get_total_energy
        # runs the simulation and returns eV.
        def builder(atoms):
            nuclei = [(float(Z), np.asarray(R)) for Z, R in
                      zip(atoms.get_atomic_numbers(), atoms.get_positions())]
            grid = Grid(center=[0, 0, 0], box_size=5.0, h=0.35)
            H = MolecularIntegrals(
                nuclei, minimal_fao_basis(nuclei), grid
            ).molecular_hamiltonian(mo_basis=True, n_electrons=2)
            return H, (1, 1), 2

        atoms = Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]])
        atoms.calc = Mandacaru(method="adapt-vqe", pool="ceo",
                               hamiltonian_builder=builder, max_iterations=6,
                               gradient_tolerance=1e-4)
        energy_ev = atoms.get_total_energy()
        result = atoms.calc.result

        # ASE returns eV, and so does the result object itself.
        assert result.energy_unit == "eV"
        assert energy_ev == pytest.approx(result.optimal_energy, rel=1e-9)
        # And match the exact FCI of the built Hamiltonian (Hartree -> eV).
        h = atoms.calc.hamiltonian.to_matrix()
        exact = float(np.linalg.eigvalsh(0.5 * (h + h.conj().T)).min())
        assert result.optimal_energy == pytest.approx(
            exact * 27.211386245988, abs=1e-4 * 27.211386245988)
        assert result.in_units("Ha") == pytest.approx(exact, abs=1e-4)

    def test_calculator_builds_from_default_basis(self):
        # With the default basis="FAO", no explicit builder is needed: the
        # calculator builds the Hamiltonian from the geometry itself.
        atoms = Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]])
        atoms.calc = Mandacaru(method="adapt-vqe", pool="ceo", basis="FAO",
                               grid=Grid(center=[0, 0, 0], box_size=6.0, h=0.30),
                               max_iterations=6, gradient_tolerance=1e-3)
        energy = atoms.get_total_energy()
        assert np.isfinite(energy)
        assert atoms.calc.n_qubits == 4        # H2 in FAO -> 2 orbitals

    def test_ibm_quantum_device_requires_shots(self):
        # Real hardware never returns a state vector: refused up front.
        with pytest.raises(ValueError, match="shots > 0"):
            Mandacaru(method="adapt-vqe", pool="ceo", basis="FAO",
                      device="ibm-quantum")

    def test_grid_auto_generated_from_cell(self):
        # No explicit grid: the calculator builds one from atoms.cell at
        # resolution h, and the run still reaches a finite energy.
        atoms = Atoms("H2", positions=[[3, 3, 2.63], [3, 3, 3.37]],
                      cell=[[6, 0, 0], [0, 6, 0], [0, 0, 6]], pbc=True)
        atoms.calc = Mandacaru(method="adapt-vqe", pool="ceo", basis="FAO",
                               h=0.30, max_iterations=6,
                               gradient_tolerance=1e-3)
        assert np.isfinite(atoms.get_total_energy())
        assert atoms.calc.n_qubits == 4

    def test_grid_requires_cell_when_not_given(self):
        # Without an explicit grid AND without a unit cell, grid auto-generation
        # is impossible -> a clear error.
        atoms = Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]])  # no cell
        atoms.calc = Mandacaru(method="adapt-vqe", pool="ceo", basis="FAO",
                               max_iterations=4, gradient_tolerance=1e-3)
        with pytest.raises(ValueError, match="no unit cell"):
            atoms.get_total_energy()


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


# --------------------------------------------------------------------------- #
# One log per run, one block per geometry step (the relaxation protocol).
# --------------------------------------------------------------------------- #

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
        assert text.startswith(banner.lines()[0])
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
                                     "energy_unit:", "initial_ansatz:",
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


class TestRelaxationLog:
    """End to end: a BFGS relaxation writes one continuous, complete log."""

    @pytest.fixture(scope="class")
    def relaxation(self, tmp_path_factory):
        from ase.optimize import BFGS

        from mandacaru import Mandacaru

        out = str(tmp_path_factory.mktemp("relax") / "output.txt")
        atoms = Atoms("H2", positions=[[3, 3, 2.6], [3, 3, 3.4]],
                      cell=[6.0, 6.0, 6.0])
        atoms.calc = Mandacaru(method="adapt-vqe", basis="FAO", h=0.35,
                               pool="fermionic", max_iterations=4,
                               gradient_tolerance=1e-3, output=out)
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


# --------------------------------------------------------------------------- #
# Performance accounting, and the split between the log and standard output.
# --------------------------------------------------------------------------- #

class TestPerformanceBlock:
    """Every evaluation records where its time and memory went."""

    @pytest.fixture(scope="class")
    def relaxation(self, tmp_path_factory):
        from ase.optimize import BFGS

        from mandacaru import Mandacaru

        out = str(tmp_path_factory.mktemp("perf") / "output.txt")
        atoms = Atoms("H2", positions=[[3, 3, 2.6], [3, 3, 3.4]],
                      cell=[6.0, 6.0, 6.0])
        atoms.calc = Mandacaru(method="adapt-vqe", basis="FAO", h=0.35,
                               pool="fermionic", max_iterations=4,
                               gradient_tolerance=1e-3, output=out)
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
                  output=out).run()
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


class TestStandardOutputIsTheASETable:
    """stdout carries the evolution of energies and forces, nothing else.

    The detail has a destination -- ``output=<path>`` -- so standard output is
    left to what an ASE optimizer prints there, the same split GPAW makes with
    ``txt=``.  Without a log file the trace is the only report there is, so it
    is printed.
    """

    @staticmethod
    def _run(tmp_path, capsys, **options):
        from mandacaru import Mandacaru

        atoms = Atoms("H2", positions=[[3, 3, 2.6], [3, 3, 3.4]],
                      cell=[6.0, 6.0, 6.0])
        atoms.calc = Mandacaru(method="adapt-vqe", basis="FAO", h=0.35,
                               pool="fermionic", max_iterations=2,
                               gradient_tolerance=1e-3, **options)
        atoms.get_potential_energy()
        return capsys.readouterr().out

    def test_a_log_file_silences_the_trace(self, tmp_path, capsys):
        out = self._run(tmp_path, capsys, output=str(tmp_path / "output.txt"))
        for noise in ("ADAPT-VQE", "operator pool", "Timings", "iter"):
            assert noise not in out, noise
        assert out.strip() == ""

    def test_without_a_log_file_the_trace_is_printed(self, tmp_path, capsys):
        out = self._run(tmp_path, capsys)
        # The only report there is, so it must not be silent.
        assert "ADAPT-VQE" in out and "Timings" in out
        assert "Hartree-Fock reference" in out

    def test_trace_overrides_the_automatic_choice(self, tmp_path, capsys):
        printed = self._run(tmp_path, capsys, trace=True,
                            output=str(tmp_path / "with_trace.txt"))
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
        self._run(tmp_path, capsys, output=out)
        parsed = parse_output(out)
        assert parsed["iterations"] and parsed["summary"]["converged"]
        assert parsed["performance"]["wall_time_s"] > 0


class TestElectronsBlock:
    """``[ELECTRONS]`` records which Hamiltonian the iterations belong to.

    With the trace routed to the log file, this block is the *only* place the
    configuration is written down, so it has to carry everything needed to know
    what was solved: the discretization, the encoding, and the size of the
    register and Hamiltonian that came out.
    """

    FIELDS = ("basis", "grid spacing", "kinetic operator", "k-points",
              "spin-polarized", "mapping", "Hamiltonian",
              "spatial orbitals", "electrons (alpha, beta)", "qubits")

    @pytest.fixture(scope="class")
    def run(self, tmp_path_factory):
        from mandacaru import Mandacaru

        out = str(tmp_path_factory.mktemp("electrons") / "output.txt")
        atoms = Atoms("H2", positions=[[3, 3, 2.63], [3, 3, 3.37]],
                      cell=[6.0, 6.0, 6.0])
        atoms.calc = Mandacaru(method="adapt-vqe", basis="FAO", h=0.35,
                               pool="fermionic", max_iterations=2,
                               gradient_tolerance=1e-3, output=out)
        atoms.get_potential_energy()
        return out, atoms.calc

    def test_every_field_is_present_and_in_order(self, run):
        out, _calc = run
        block = parse_output(out)["electrons"]
        assert tuple(block) == self.FIELDS

    def test_the_values_describe_this_run(self, run):
        out, calc = run
        block = parse_output(out)["electrons"]
        assert block["basis"] == "FAO"
        assert block["grid spacing"] == "0.35 Angstrom"
        assert block["kinetic operator"] == "finite difference"
        assert "Monkhorst-Pack" in block["k-points"]
        assert block["spin-polarized"] == "False"
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
              "energy_unit", "reference_energy_eV", "initial_ansatz")

    def _log(self, hamiltonian, tmp_path, **kwargs):
        out = str(tmp_path / "output.txt")
        Mandacaru(method="adapt-vqe", hamiltonian=hamiltonian,
                  pool="fermionic", num_particles=(1, 1),
                  n_spatial_orbitals=2, profile=False, max_iterations=2,
                  gradient_tolerance=1e-3, output=out, trace=False,
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
                  resume=checkpoint, output=out, trace=False).run()

        setup = parse_output(out)["setup"]
        assert setup["gradient_method"] == "finite_difference"
        assert setup["initial_ansatz"].startswith("resumed")
        keys = list(setup)
        # The four groups stay contiguous and lineage stays last.
        assert keys[:len(self.FIELDS)] == list(self.FIELDS)
        assert keys[len(self.FIELDS):] == [
            "resumed_from", "restored_operators", "restored_energy_eV",
            "resume_same_hamiltonian"]

    def test_the_trace_uses_the_same_vocabulary(self, h2_hamiltonian, capsys):
        """A reader who saw the terminal recognizes the file, and vice versa."""
        Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                  pool="fermionic", num_particles=(1, 1), n_spatial_orbitals=2,
                  profile=False, max_iterations=1, gradient="parameter_shift",
                  trace=True).run()
        printed = capsys.readouterr().out
        assert "gradient method" in printed and "parameter_shift" in printed
        assert "parameter-shift" not in printed.split("formula")[0]


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
        atoms.calc = Mandacaru(method="adapt-vqe", basis="FAO", h=0.35,
                               pool="fermionic", max_iterations=4,
                               gradient_tolerance=1e-3, output=out)
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
        atoms.calc = Mandacaru(method="adapt-vqe", basis="FAO", h=0.35,
                               pool="fermionic", max_iterations=2,
                               gradient_tolerance=1e-3, output=out)
        atoms.get_forces()
        # One geometry is not a trajectory; there is nothing to summarize.
        assert atoms.calc.write_optimization_summary() is False
        assert "[GEOMETRY OPTIMIZATION SUMMARY]" not in open(out).read()
        assert atoms.calc.write_optimization_summary(force=True) is True
