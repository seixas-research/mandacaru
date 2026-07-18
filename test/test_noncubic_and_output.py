# -*- coding: utf-8 -*-
# file: test_noncubic_and_output.py

"""Non-cubic (cell-aware) grids, ASE integration and the ADAPT output.txt protocol.

Covers the three features added on top of the cubic integral core:

* :class:`~carcara.integrals.Grid` accepting an anisotropic ``box_size`` or a
  full ``cell`` tensor, with the integral engine still recovering the reference
  physics (H 1s on-site repulsion = 5/8 Ha) on a non-cubic box;
* :class:`~carcara.wavefunction.Wavefunction` built directly from an ASE
  ``Atoms`` object (elements / positions / cell);
* :class:`~carcara.algorithms.AdaptVQE` writing a structured, live-parseable
  ``output.txt`` as the ADAPT loop runs.
"""

import numpy as np
import pytest
from ase import Atoms

from carcara.algorithms import ADAPTVQE, AdaptVQE
from carcara.basis import HydrogenicOrbital
from carcara.core import HydrogenicIntegrals, minimal_hydrogenic_basis
from carcara.integrals import Grid, IntegralEngine
from carcara.utils import AdaptOutputLogger, parse_output
from carcara.wavefunction import Wavefunction


# --------------------------------------------------------------------------- #
# Non-cubic grids.
# --------------------------------------------------------------------------- #

def _eri_00(grid):
    """H 1s on-site repulsion <00|00> on ``grid`` (Hartree)."""
    orb = HydrogenicOrbital(1, 0, 0, Z=1.0, center=[0.0, 0.0, 0.0], units="bohr")
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
        orb = HydrogenicOrbital(1, 0, 0, Z=1.0, center=[0, 0, 0], units="bohr")
        g = Grid(center=[0, 0, 0], box_size=[8.0, 9.0, 7.0], h=0.20, units="bohr")
        eng = IntegralEngine([orb], g)
        zero_v = lambda x, y, z: np.zeros(np.broadcast(x, y, z).shape)
        T, _ = eng.one_body(zero_v, energy_units="Ha")
        assert float(np.real(T[0, 0])) == pytest.approx(0.5, abs=1e-2)


# --------------------------------------------------------------------------- #
# ASE integration.
# --------------------------------------------------------------------------- #

class TestASEIntegration:
    def test_from_ase_molecule(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        wf = Wavefunction.from_ase(atoms)
        assert wf.n_atoms == 2
        assert wf.all_symbols == ["H", "H"]
        assert wf.Z == 1
        assert not wf.has_cell
        assert wf.cell is None

    def test_from_ase_extracts_positions(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        wf = Wavefunction.from_ase(atoms)
        assert wf.all_positions_bohr.shape == (2, 3)
        # 0.74 Angstrom -> Bohr on the second atom's x.
        assert wf.all_positions_bohr[1, 0] == pytest.approx(0.74 * 1.8897259886)

    def test_from_ase_crystal_cell(self):
        cell = [[0.0, 2.7, 2.7], [2.7, 0.0, 2.7], [2.7, 2.7, 0.0]]
        atoms = Atoms("Si2", positions=[[0, 0, 0], [1.35, 1.35, 1.35]],
                      cell=cell, pbc=True)
        wf = Wavefunction.from_ase(atoms)
        assert wf.has_cell
        assert wf.cell.shape == (3, 3)
        np.testing.assert_allclose(wf.cell, cell)

    def test_grid_from_cell(self):
        cell = [[10.0, 0, 0], [0, 12.0, 0], [0, 0, 8.0]]  # Angstrom, orthorhombic
        atoms = Atoms("H", positions=[[5, 6, 4]], cell=cell, pbc=True)
        wf = Wavefunction.from_ase(atoms)
        grid = wf.grid_from_cell(h=0.5)
        assert not grid.is_cubic          # 10 != 12 != 8
        assert grid.dx == pytest.approx(0.5)

    def test_grid_from_cell_requires_cell(self):
        wf = Wavefunction.from_ase(Atoms("H", positions=[[0, 0, 0]]))
        with pytest.raises(ValueError):
            wf.grid_from_cell()

    def test_legacy_xyz_still_works(self, tmp_path):
        from ase.io import write
        path = str(tmp_path / "h2.xyz")
        write(path, Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]))
        wf = Wavefunction(path, atom_index=0)   # positional str path (legacy)
        assert wf.n_atoms == 2
        assert not wf.has_cell


# --------------------------------------------------------------------------- #
# ADAPT-VQE output.txt protocol.
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def h2_hamiltonian():
    R = 0.74
    nuclei = [(1.0, np.array([0.0, 0.0, -R / 2])),
              (1.0, np.array([0.0, 0.0, +R / 2]))]
    grid = Grid(center=[0, 0, 0], box_size=5.0, h=0.35)
    integrals = HydrogenicIntegrals(nuclei, minimal_hydrogenic_basis(nuclei), grid)
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
        adapt = _h2_adapt(h2_hamiltonian)
        result = adapt.run(max_iterations=6, gradient_tol=1e-4,
                           output_file=out, geometry=geom)

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
        _h2_adapt(h2_hamiltonian).run(max_iterations=4, gradient_tol=1e-4,
                                      output_file=out)
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
                         n_spatial_orbitals=2, profile=True)  # profile for gates
        adapt.run(max_iterations=4, gradient_tol=1e-4, output_file=out)
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
        adapt = _h2_adapt(h2_hamiltonian, atomic_units=True)
        adapt.run(max_iterations=4, gradient_tol=1e-4, output_file=out,
                  geometry=geom)
        parsed = parse_output(out)
        assert parsed["metadata"]["units"] == "Bohr"
        assert parsed["setup"]["energy_unit"] == "Ha"
        assert "reference_energy_Ha" in parsed["setup"]

    def test_runs_without_geometry(self, h2_hamiltonian, tmp_path):
        # The protocol must still write cleanly when no geometry is supplied.
        out = str(tmp_path / "output.txt")
        _h2_adapt(h2_hamiltonian).run(max_iterations=4, gradient_tol=1e-4,
                                      output_file=out)
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
    def test_class_renamed_with_alias(self):
        # Requirement 2: class is ADAPTVQE (all caps); AdaptVQE stays as an alias.
        assert AdaptVQE is ADAPTVQE

    def test_ase_calculator_get_total_energy(self, tmp_path):
        # Requirement 7: attach ADAPTVQE as an ASE calculator; get_total_energy
        # runs the simulation and returns eV.
        def builder(atoms):
            nuclei = [(float(Z), np.asarray(R)) for Z, R in
                      zip(atoms.get_atomic_numbers(), atoms.get_positions())]
            grid = Grid(center=[0, 0, 0], box_size=5.0, h=0.35)
            H = HydrogenicIntegrals(
                nuclei, minimal_hydrogenic_basis(nuclei), grid
            ).molecular_hamiltonian(mo_basis=True, n_electrons=2)
            return H, (1, 1), 2

        atoms = Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]])
        atoms.calc = ADAPTVQE(pool="ceo", hamiltonian_builder=builder,
                              run_options={"max_iterations": 6,
                                           "gradient_tol": 1e-4})
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
                              run_options={"max_iterations": 6,
                                           "gradient_tol": 1e-3})
        energy = atoms.get_total_energy()
        assert np.isfinite(energy)
        assert atoms.calc.n_qubits == 4        # H2 in FAO -> 2 orbitals

    def test_ibm_quantum_device_not_runnable(self):
        atoms = Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]])
        atoms.calc = ADAPTVQE(pool="ceo", basis="FAO", device="ibm-quantum")
        with pytest.raises(NotImplementedError):
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
