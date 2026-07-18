# -*- coding: utf-8 -*-
# file: test_noncubic_and_output.py

"""Non-cubic (cell-aware) grids, ASE integration and the ADAPT output.txt protocol.

Covers the three features added on top of the cubic integral core:

* :class:`~carcara.integrals.Grid` accepting an anisotropic ``box_size`` or a
  full ``cell`` tensor, with the integral engine still recovering the reference
  physics (H 1s on-site repulsion = 5/8 Ha) on a non-cubic box;
* the real-space grid generated directly from an ASE ``Atoms`` unit cell
  (:func:`carcara.algorithms._hamiltonian_from_atoms.grid_from_cell`);
* :class:`~carcara.algorithms.ADAPTVQE` writing a structured, live-parseable
  ``output.txt`` as the ADAPT loop runs.
"""

import numpy as np
import pytest
from ase import Atoms

from carcara.algorithms import ADAPTVQE
from carcara.algorithms._hamiltonian_from_atoms import grid_from_cell
from carcara.basis import FullAtomicOrbital
from carcara.core import MolecularIntegrals, minimal_fao_basis
from carcara.integrals import Grid, IntegralEngine
from carcara.utils import AdaptOutputLogger, parse_output


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
        import carcara.integrals._backend as backend
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
    return ADAPTVQE(hamiltonian, "fermionic", num_particles=(1, 1),
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
        result = adapt.run(geometry=geom)

        parsed = parse_output(out)

        # Metadata block: initial geometry + explicit unit-cell parameters.
        assert parsed["metadata"]["n_atoms"] == "2"
        assert parsed["metadata"]["cell_present"] == "True"
        assert parsed["metadata"]["units"] == "Angstrom"       # req 1: default A
        assert "cell_lengths" in parsed["metadata"]
        assert "cell_angles" in parsed["metadata"]

        # Optimization setup block -- energies default to eV (requirement 1).
        assert parsed["setup"]["classical_optimizer"] == "COBYLA"
        assert parsed["setup"]["energy_unit"] == "eV"
        assert "reference_energy_eV" in parsed["setup"]

        # At least one iteration, each with all four tracked metrics.
        assert len(parsed["iterations"]) == result.num_operators >= 1
        for it in parsed["iterations"]:
            assert it["selected_operator"]                 # 3. selected operator
            assert "expressivity_E" in it                  # 4. expressivity E
            assert it["energy_unit"] == "eV"               # 1. eV default
            assert len(it["pool"]) >= 1                     # 1. pool Pauli strings
            for pauli in it["pool"]:                        # Pauli-string format
                assert any(P in pauli for P in "XYZ") or pauli == "0"

    def test_selected_operator_reported_separately_from_pool(self, h2_hamiltonian,
                                                             tmp_path):
        # Requirement 3: the selected operator is a distinct block, and the pool
        # is a separate listing.
        out = str(tmp_path / "output.txt")
        _h2_adapt(h2_hamiltonian, max_iterations=4, gradient_tolerance=1e-4,
                  output=out).run()
        text = open(out, encoding="utf-8").read()
        assert "selected_operator:" in text
        assert "operator_pool:" in text
        # The selected-operator block precedes the pool listing in every block.
        block = text.split("[ITERATION 1]", 1)[1]
        assert block.index("selected_operator:") < block.index("operator_pool:")
        assert "(selected)" in block                       # cross-ref marker
        assert "|grad|=" in block                          # 2. gradient magnitudes

    def test_summary_reports_final_parameterization(self, h2_hamiltonian, tmp_path):
        # Requirement 5: richer summary (expressivity, gates, CNOTs, depth, ...).
        out = str(tmp_path / "output.txt")
        adapt = ADAPTVQE(h2_hamiltonian, "fermionic", num_particles=(1, 1),
                         n_spatial_orbitals=2, profile=True,  # profile for gates
                         max_iterations=4, gradient_tolerance=1e-4, output=out)
        adapt.run()
        summary = parse_output(out)["summary"]
        for key in ("optimal_energy_eV", "reference_energy_eV", "num_operators",
                    "num_parameters", "final_expressivity_E", "cnot_count",
                    "circuit_depth", "total_gates", "one_qubit_gates",
                    "cost_evaluations", "operator_sequence"):
            assert key in summary, key

    def test_atomic_units_switch(self, h2_hamiltonian, tmp_path):
        # Requirement 1: atomic units used only when explicitly requested.
        out = str(tmp_path / "output.txt")
        geom = Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]])
        adapt = _h2_adapt(h2_hamiltonian, atomic_units=True, max_iterations=4,
                          gradient_tolerance=1e-4, output=out)
        adapt.run(geometry=geom)
        parsed = parse_output(out)
        assert parsed["metadata"]["units"] == "Bohr"
        assert parsed["setup"]["energy_unit"] == "Ha"
        assert "reference_energy_Ha" in parsed["setup"]

    def test_runs_without_geometry(self, h2_hamiltonian, tmp_path):
        # The protocol must still write cleanly when no geometry is supplied.
        out = str(tmp_path / "output.txt")
        _h2_adapt(h2_hamiltonian, max_iterations=4, gradient_tolerance=1e-4,
                  output=out).run()
        parsed = parse_output(out)
        assert parsed["metadata"]["cell_present"] == "False"
        assert parsed["metadata"]["geometry"] == "(not provided)"


class TestCEOLabels:
    def test_ceo_labels_unique_and_descriptive(self):
        from carcara.circuits import build_pool
        labels = [op.label for op in build_pool("ceo", 3, (2, 2)).operators()]
        # Requirement 4: no collisions, and each label names its support.
        assert len(labels) == len(set(labels))
        assert all(lbl.startswith("CEO[q") for lbl in labels)


class TestADAPTVQECalculator:
    def test_class_named_all_caps(self):
        # The driver class is ADAPTVQE (all caps); the old AdaptVQE alias is gone.
        import carcara.algorithms as algs
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
        atoms.calc = ADAPTVQE(pool="ceo", hamiltonian_builder=builder,
                              max_iterations=6, gradient_tolerance=1e-4)
        energy_ev = atoms.get_total_energy()
        result = atoms.calc.adapt_result

        # ASE returns eV; it must equal the Ha result converted to eV.
        expected_ev = result.optimal_energy * 27.211386245988
        assert energy_ev == pytest.approx(expected_ev, rel=1e-9)
        # And match the exact FCI of the built Hamiltonian.
        h = atoms.calc.hamiltonian.to_matrix()
        exact = float(np.linalg.eigvalsh(0.5 * (h + h.conj().T)).min())
        assert result.optimal_energy == pytest.approx(exact, abs=1e-4)

    def test_calculator_builds_from_default_basis(self):
        # With the default basis="FAO", no explicit builder is needed: the
        # calculator builds the Hamiltonian from the geometry itself.
        atoms = Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]])
        atoms.calc = ADAPTVQE(pool="ceo", basis="FAO",
                              grid=Grid(center=[0, 0, 0], box_size=6.0, h=0.30),
                              max_iterations=6, gradient_tolerance=1e-3)
        energy = atoms.get_total_energy()
        assert np.isfinite(energy)
        assert atoms.calc.n_qubits == 4        # H2 in FAO -> 2 orbitals

    def test_ibm_quantum_device_not_runnable(self):
        atoms = Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]])
        atoms.calc = ADAPTVQE(pool="ceo", basis="FAO", device="ibm-quantum")
        with pytest.raises(NotImplementedError):
            atoms.get_total_energy()

    def test_grid_auto_generated_from_cell(self):
        # No explicit grid: the calculator builds one from atoms.cell at
        # resolution h, and the run still reaches a finite energy.
        atoms = Atoms("H2", positions=[[3, 3, 2.63], [3, 3, 3.37]],
                      cell=[[6, 0, 0], [0, 6, 0], [0, 0, 6]], pbc=True)
        atoms.calc = ADAPTVQE(pool="ceo", basis="FAO", h=0.30,
                              max_iterations=6, gradient_tolerance=1e-3)
        assert np.isfinite(atoms.get_total_energy())
        assert atoms.calc.n_qubits == 4

    def test_grid_requires_cell_when_not_given(self):
        # Without an explicit grid AND without a unit cell, grid auto-generation
        # is impossible -> a clear error.
        atoms = Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]])  # no cell
        atoms.calc = ADAPTVQE(pool="ceo", basis="FAO",
                              max_iterations=4, gradient_tolerance=1e-3)
        with pytest.raises(ValueError, match="no unit cell"):
            atoms.get_total_energy()


class TestAdaptOutputLogger:
    def test_logger_cell_parameters(self, tmp_path):
        out = str(tmp_path / "log.txt")
        with AdaptOutputLogger(out) as logger:
            logger.write_metadata(
                symbols=["H", "H"], positions=[[0, 0, 0], [0, 0, 0.74]],
                cell=np.diag([5.0, 6.0, 7.0]))
        parsed = parse_output(out)
        assert parsed["metadata"]["cell_present"] == "True"
        assert parsed["metadata"]["cell_lengths"].startswith("a=5")
