# -*- coding: utf-8 -*-
# file: test_adapt_args_gradient.py

"""ADAPTVQE argument surface, gradient strategies, the device registry and the
FullAtomicOrbitals (FAO) basis.

Covers the features added on top of the ADAPT-VQE driver:

* selectable ``gradient`` strategies -- ``"finite_difference"`` and
  ``"parameter-shift"`` -- both matching the exact analytic gradient;
* the ``ADAPTVQE`` argument surface (``pool``, ``basis``, ``mapping``,
  ``gradient``, ``device``) and its basis-driven Hamiltonian builder;
* the ``device`` registry (AER_simulator vs the reserved ibm-quantum).
"""

import numpy as np
import pytest
from ase import Atoms

from carcara.algorithms import ADAPTVQE
from carcara.circuits import AdaptAnsatz
from carcara.backends import available_devices, is_simulator, normalize_device
from carcara.integrals import Grid


# --------------------------------------------------------------------------- #
# The FullAtomicOrbitals (FAO) rename.
# --------------------------------------------------------------------------- #

class TestFAONaming:
    def test_fao_names_exist(self):
        from carcara.basis import FAOBasisSet, FullAtomicOrbital
        from carcara.core import MolecularIntegrals, minimal_fao_basis
        assert FullAtomicOrbital is not None
        assert FAOBasisSet is not None
        assert MolecularIntegrals is not None
        assert minimal_fao_basis is not None

    def test_factory_builds_fao(self):
        from carcara.basis import BasisSet, FAOBasisSet
        assert isinstance(BasisSet.build("FAO"), FAOBasisSet)
        assert BasisSet.build("STO-3G").name == "STO-3G"
        with pytest.raises(ValueError):
            BasisSet.build("hydrogenic")      # the alias no longer resolves


# --------------------------------------------------------------------------- #
# Shared H2 Hamiltonian (FAO / MO basis).
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def h2_hamiltonian():
    from carcara.core import MolecularIntegrals, minimal_fao_basis
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
        adapt = ADAPTVQE(h2_hamiltonian, pool, num_particles=(1, 1),
                         n_spatial_orbitals=2, profile=False,
                         gradient="finite_difference")
        psi = AdaptAnsatz(adapt.n_qubits, adapt.pool.occupied_orbitals).state(
            np.zeros(0))
        g_an = adapt._analytic_gradients(psi)
        g_fd = adapt._finite_difference_gradients(psi)
        np.testing.assert_allclose(g_fd, g_an, atol=1e-6)

    @pytest.mark.parametrize("pool", ["fermionic", "qubit", "qeb", "ceo"])
    def test_parameter_shift_is_exact(self, h2_hamiltonian, pool):
        adapt = ADAPTVQE(h2_hamiltonian, pool, num_particles=(1, 1),
                         n_spatial_orbitals=2, profile=False,
                         gradient="parameter-shift")
        psi = AdaptAnsatz(adapt.n_qubits, adapt.pool.occupied_orbitals).state(
            np.zeros(0))
        g_an = adapt._analytic_gradients(psi)
        g_ps = adapt._parameter_shift_gradients(psi)
        np.testing.assert_allclose(g_ps, g_an, atol=1e-9)

    def test_both_gradients_reach_fci(self, h2_hamiltonian):
        m = h2_hamiltonian.map_to_qubits("jordan_wigner").to_matrix()
        exact = float(np.linalg.eigvalsh(0.5 * (m + m.conj().T)).min())
        for grad in ("finite_difference", "parameter-shift"):
            adapt = ADAPTVQE(h2_hamiltonian, "ceo", num_particles=(1, 1),
                             n_spatial_orbitals=2, profile=False, gradient=grad,
                             max_iterations=10, gradient_tolerance=1e-4)
            res = adapt.run()
            assert abs(res.optimal_energy - exact) < 1e-4, grad

    def test_invalid_gradient_rejected(self, h2_hamiltonian):
        with pytest.raises(ValueError):
            ADAPTVQE(h2_hamiltonian, "ceo", num_particles=(1, 1),
                     n_spatial_orbitals=2, gradient="nope")


# --------------------------------------------------------------------------- #
# Argument surface + basis-driven builder.
# --------------------------------------------------------------------------- #

class TestArgumentSurface:
    @pytest.mark.parametrize("pool", ["ceo", "fermionic", "qubit", "qeb"])
    def test_pool_options(self, pool):
        atoms = Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]])
        atoms.calc = ADAPTVQE(pool=pool, basis="FAO",
                              grid=Grid(center=[0, 0, 0], box_size=6.0, h=0.3),
                              max_iterations=8, gradient_tolerance=1e-3)
        assert np.isfinite(atoms.get_total_energy())

    @pytest.mark.parametrize("mapping",
                             ["jordan_wigner", "parity", "bravyi_kitaev"])
    def test_mapping_options_reach_fci(self, mapping):
        atoms = Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]])
        atoms.calc = ADAPTVQE(pool="fermionic", basis="FAO", mapping=mapping,
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
        adapt = ADAPTVQE(h2_hamiltonian, "ceo", num_particles=(1, 1),
                         n_spatial_orbitals=2, profile=False,
                         max_iterations=3, gradient_tolerance=1e-2)
        assert adapt.max_iterations == 3
        assert adapt.gradient_tolerance == 1e-2
        res = adapt.run()                       # no args -> uses the defaults
        assert res.num_operators <= 3

    def test_output_constructor_arg_writes_file(self, h2_hamiltonian, tmp_path):
        from carcara.utils import parse_output
        out = str(tmp_path / "output.txt")
        adapt = ADAPTVQE(h2_hamiltonian, "fermionic", num_particles=(1, 1),
                         n_spatial_orbitals=2, profile=False,
                         max_iterations=4, gradient_tolerance=1e-3, output=out)
        adapt.run()                             # output taken from constructor
        parsed = parse_output(out)
        assert parsed["setup"]["classical_optimizer"] == "COBYLA"
        assert len(parsed["iterations"]) >= 1

    def test_basis_option_sets_qubit_count(self):
        # FAO on LiH -> Li{1s,2s} + H{1s} = 3 orbitals -> 6 qubits.
        atoms = Atoms("LiH", positions=[[0, 0, -0.8], [0, 0, 0.8]])
        atoms.calc = ADAPTVQE(pool="ceo", basis="FAO",
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
        adapt = ADAPTVQE(h2_hamiltonian, "ceo", num_particles=(1, 1),
                         n_spatial_orbitals=2, profile=False)
        assert adapt.optimizer.method == "COBYLA"

    @pytest.mark.parametrize(
        "name",
        ["SPSA", "COBYLA", "Nelder-Mead", "SLSQP", "Adam", "L-BFGS-B"])
    def test_named_optimizers_build(self, h2_hamiltonian, name):
        adapt = ADAPTVQE(h2_hamiltonian, "ceo", num_particles=(1, 1),
                         n_spatial_orbitals=2, profile=False, optimizer=name)
        assert adapt.optimizer.method == name

    def test_optimizer_instance_passthrough(self, h2_hamiltonian):
        from carcara.optimizers import Optimizer
        opt = Optimizer("L-BFGS-B", maxiter=500)
        adapt = ADAPTVQE(h2_hamiltonian, "ceo", num_particles=(1, 1),
                         n_spatial_orbitals=2, profile=False, optimizer=opt)
        assert adapt.optimizer is opt

    def test_unknown_optimizer_rejected(self, h2_hamiltonian):
        with pytest.raises(ValueError):
            ADAPTVQE(h2_hamiltonian, "ceo", num_particles=(1, 1),
                     n_spatial_orbitals=2, optimizer="nope")


# --------------------------------------------------------------------------- #
# Standard-output Pauli-string trace.
# --------------------------------------------------------------------------- #

class TestVerbosePauliOutput:
    def test_hamiltonian_and_ansatz_pauli_strings_printed(self, h2_hamiltonian,
                                                          capsys):
        adapt = ADAPTVQE(h2_hamiltonian, "ceo", num_particles=(1, 1),
                         n_spatial_orbitals=2, profile=False, verbose=True,
                         max_iterations=3, gradient_tolerance=1e-6)
        adapt.run()
        out = capsys.readouterr().out
        # The qubit Hamiltonian is echoed as Pauli strings ...
        assert "Qubit Hamiltonian" in out
        assert "* ZIII" in out
        # ... and each iteration prints the selected operator's generator.
        assert "ansatz operator (Pauli strings)" in out
        assert out.count("[iter 1]") == 1

    def test_verbose_false_is_silent(self, h2_hamiltonian, capsys):
        adapt = ADAPTVQE(h2_hamiltonian, "ceo", num_particles=(1, 1),
                         n_spatial_orbitals=2, profile=False, verbose=False,
                         max_iterations=3, gradient_tolerance=1e-6)
        adapt.run()
        assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------- #
# Timing / memory / cores profiling (requirements 2-3).
# --------------------------------------------------------------------------- #

class TestADAPTProfiling:
    def test_result_carries_stage_timings(self, h2_hamiltonian):
        adapt = ADAPTVQE(h2_hamiltonian, "ceo", num_particles=(1, 1),
                         n_spatial_orbitals=2, profile=False, verbose=False,
                         max_iterations=3, gradient_tolerance=1e-6)
        res = adapt.run()
        t = res.timings
        assert t is not None
        assert "gradient screening" in t["stages_s"]
        assert "parameter optimization" in t["stages_s"]
        assert t["wall_time_s"] is not None
        assert t["peak_memory_mb"] > 0.0

    def test_summary_prints_timings_cores_memory(self, h2_hamiltonian, capsys):
        adapt = ADAPTVQE(h2_hamiltonian, "ceo", num_particles=(1, 1),
                         n_spatial_orbitals=2, profile=False, verbose=True,
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
        atoms.calc = ADAPTVQE(pool="ceo", basis="FAO", h=0.4, verbose=True,
                              max_iterations=4, gradient_tolerance=1e-3)
        atoms.get_total_energy()
        out = capsys.readouterr().out
        assert "integration:" in out
        assert atoms.calc.adapt_result.integration_profile is not None


# --------------------------------------------------------------------------- #
# Monkhorst-Pack k-points and placement-invariance (PBC-aware grid).
# --------------------------------------------------------------------------- #

class TestKPoints:
    def test_default_is_gamma(self):
        adapt = ADAPTVQE(pool="ceo", basis="FAO")
        assert adapt.kpts == (1, 1, 1)
        assert len(adapt.kpoints) == 1
        np.testing.assert_allclose(adapt.kpoints[0], [0.0, 0.0, 0.0])

    def test_mesh_generated_via_ase_monkhorst_pack(self):
        from ase.dft.kpoints import monkhorst_pack
        adapt = ADAPTVQE(pool="ceo", basis="FAO", kpts=(2, 2, 1))
        np.testing.assert_allclose(adapt.kpoints, monkhorst_pack((2, 2, 1)))

    def test_non_gamma_rejected_at_run(self, h2_hamiltonian):
        adapt = ADAPTVQE(h2_hamiltonian, "ceo", num_particles=(1, 1),
                         n_spatial_orbitals=2, kpts=(2, 1, 1))
        with pytest.raises(NotImplementedError, match="Monkhorst-Pack"):
            adapt.run()

    def test_invalid_kpts_rejected(self):
        with pytest.raises(ValueError):
            ADAPTVQE(pool="ceo", basis="FAO", kpts=(1, 1))       # not length-3

    def test_dict_spec_with_gamma_centering(self):
        # ASE dict form {"size": ..., "gamma": True}: Gamma-centred mesh.
        adapt = ADAPTVQE(pool="ceo", basis="FAO",
                         kpts={"size": (2, 2, 1), "gamma": True})
        assert adapt.kpts == (2, 2, 1)
        assert adapt.kpts_gamma is True
        # Gamma-centring shifts the even-axis mesh so it includes the Gamma point.
        assert any(np.allclose(k, [0, 0, 0]) for k in adapt.kpoints)

    def test_dict_gamma_only(self):
        adapt = ADAPTVQE(pool="ceo", basis="FAO",
                         kpts={"size": (1, 1, 1), "gamma": True})
        assert len(adapt.kpoints) == 1 and adapt.kpts_gamma is True


class TestSpinAndInitialState:
    def test_spin_defaults_false(self):
        assert ADAPTVQE(pool="ceo", basis="FAO").spin is False

    def test_spin_flag_stored(self):
        assert ADAPTVQE(pool="ceo", basis="FAO", spin=True).spin is True

    def test_even_electron_spin_polarized_matches_closed_shell(self):
        # For a singlet (even electrons) spin-polarized == closed-shell.
        atoms = Atoms("H2", positions=[[3, 3, 2.63], [3, 3, 3.37]],
                      cell=[[6, 0, 0], [0, 6, 0], [0, 0, 6]], pbc=True)

        def energy(spin):
            atoms.calc = ADAPTVQE(pool="ceo", basis="FAO", h=0.4, spin=spin,
                                  verbose=False, max_iterations=6,
                                  gradient_tolerance=1e-3)
            return atoms.get_total_energy()

        assert energy(False) == pytest.approx(energy(True), abs=1e-6)

    def test_initial_state_default_is_hartree_fock(self):
        assert ADAPTVQE(pool="ceo", basis="FAO").initial_state == "hartree-fock"

    def test_initial_state_none_is_hartree_fock(self):
        assert ADAPTVQE(pool="ceo", basis="FAO",
                        initial_state=None).initial_state == "hartree-fock"

    def test_unknown_initial_state_rejected(self):
        with pytest.raises(ValueError):
            ADAPTVQE(pool="ceo", basis="FAO", initial_state="random")


class TestPlacementInvariance:
    def test_energy_independent_of_position_in_cell(self):
        # PBC-aware grid: the molecule is centred on itself, so placing H2 at the
        # cell corner vs. the cell centre gives the same Hamiltonian and energy.
        def energy(pos):
            atoms = Atoms("H2", positions=pos,
                          cell=[[8, 0, 0], [0, 8, 0], [0, 0, 8]], pbc=True)
            atoms.calc = ADAPTVQE(pool="ceo", basis="FAO", h=0.35, verbose=False,
                                  max_iterations=6, gradient_tolerance=1e-3)
            return atoms.get_total_energy()

        corner = energy([[0, 0, -0.37], [0, 0, 0.37]])
        centre = energy([[4, 4, 3.63], [4, 4, 4.37]])
        assert corner == pytest.approx(centre, abs=1e-6)


# --------------------------------------------------------------------------- #
# Device registry.
# --------------------------------------------------------------------------- #

class TestDeviceRegistry:
    def test_aer_is_default_and_simulator(self):
        adapt = ADAPTVQE(pool="ceo", basis="FAO")
        assert adapt.device == "AER_simulator"
        assert is_simulator("AER_simulator")

    def test_aliases_normalize(self):
        assert normalize_device("aer") == "AER_simulator"
        assert normalize_device("statevector") == "AER_simulator"
        assert normalize_device("ibmq") == "ibm-quantum"

    def test_unknown_device_rejected(self):
        with pytest.raises(ValueError):
            ADAPTVQE(pool="ceo", device="quantum-thing")

    def test_ibm_quantum_listed_but_not_simulator(self):
        assert "ibm-quantum" in available_devices()
        assert not is_simulator("ibm-quantum")
