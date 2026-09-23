# -*- coding: utf-8 -*-
# file: test_vasqe.py

"""VASQE (experimental): ADAPT-VQE with stochastic (softmax) operator selection.

VASQE lives in :mod:`mandacaru.experimental` and registers its methods with the
unified calculator on import; the stable code never names it.  Everything that
exercises VASQE -- selection, schedules, quenching, the Hamiltonian cache, the
periodic calculator, the packaging rules -- lives in this file.

Checks the selection probabilities and temperature schedules, that VASQE reduces
to ADAPT-VQE at low temperature (reaching FCI on H2), that selection is genuinely
stochastic at high temperature, and that the excited-state extensions (deflation
and subspace) inherit the stochastic selection.
"""

import numpy as np
import pytest

import mandacaru.experimental  # noqa: F401  (registers the methods)
from mandacaru.algorithms.adapt_vqe import ADAPTVQEResult
from mandacaru.experimental import (
    SubspaceVASQE,
    TEMPERATURE_SCHEDULES,
    VASQE,
    VASQEResult,
    annealed_temperature,
    softmax_selection_probabilities,
)
from mandacaru.core import MolecularIntegrals, minimal_hao_basis
from mandacaru.integrals import Grid
from mandacaru.optimizers import Optimizer
from mandacaru.units import HARTREE_TO_EV
from mandacaru import Mandacaru


# --------------------------------------------------------------------------- #
# Shared H2 fixture (MO basis).
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def h2_hamiltonian():
    R = 0.74
    nuclei = [(1.0, np.array([0.0, 0.0, -R / 2])),
              (1.0, np.array([0.0, 0.0, +R / 2]))]
    grid = Grid(center=[0.0, 0.0, 0.0], box_size=5.0, h=0.25)
    mints = MolecularIntegrals(nuclei, minimal_hao_basis(nuclei), grid)
    return mints.molecular_hamiltonian(mo_basis=True, n_electrons=2)


@pytest.fixture(scope="module")
def h2_fci(h2_hamiltonian):
    """FCI ground state in eV (the Hamiltonian is Hartree; results are eV)."""
    m = h2_hamiltonian.map_to_qubits("jordan_wigner").to_matrix()
    return float(np.linalg.eigvalsh(0.5 * (m + m.conj().T)).min()) * HARTREE_TO_EV


#: Result-side tolerances, converted from the historical Hartree values.
TOL_1E4 = 1e-4 * HARTREE_TO_EV
TOL_1E5 = 1e-5 * HARTREE_TO_EV
TOL_1E6 = 1e-6 * HARTREE_TO_EV
TOL_1E8 = 1e-8 * HARTREE_TO_EV


def _vasqe(h2_hamiltonian, **kwargs):
    kwargs.setdefault("optimizer", Optimizer("L-BFGS", maxiter=2000))
    return Mandacaru(method="vasqe", hamiltonian=h2_hamiltonian,
                     pool="fermionic", num_particles=(1, 1),
                     n_spatial_orbitals=2, trace=False, profile=False,
                     gradient_tolerance=1e-6, **kwargs)


# --------------------------------------------------------------------------- #
# Softmax selection probabilities.
# --------------------------------------------------------------------------- #

class TestSoftmax:
    def test_low_temperature_is_argmax(self):
        g = np.array([0.1, 0.9, 0.4])
        p = softmax_selection_probabilities(g, 1e-6)
        assert np.argmax(p) == 1                    # largest |g|
        assert p[1] == pytest.approx(1.0, abs=1e-9)

    def test_high_temperature_is_uniform(self):
        g = np.array([0.1, 0.9, 0.4])
        p = softmax_selection_probabilities(g, 1e6)
        np.testing.assert_allclose(p, np.full(3, 1 / 3), atol=1e-3)

    def test_probabilities_normalized(self):
        g = np.array([0.2, 0.5, 0.7, 0.01])
        p = softmax_selection_probabilities(g, 0.3)
        assert p.sum() == pytest.approx(1.0)
        assert np.all(p >= 0)

    def test_uses_absolute_gradients(self):
        # sign must not matter: |−0.9| beats |0.1|.
        p = softmax_selection_probabilities(np.array([-0.9, 0.1]), 0.05)
        assert np.argmax(p) == 0

    def test_favors_larger_gradient(self):
        p = softmax_selection_probabilities(np.array([0.3, 0.6]), 1.0)
        assert p[1] > p[0]                          # larger |g| -> larger P


# --------------------------------------------------------------------------- #
# Temperature schedules.
# --------------------------------------------------------------------------- #

class TestSchedules:
    def test_constant_ignores_final(self):
        for k in range(5):
            assert annealed_temperature("constant", 2.0, 0.01, k, 5) == 2.0

    @pytest.mark.parametrize("schedule", ["linear", "exponential", "logarithmic"])
    def test_annealing_endpoints(self, schedule):
        assert annealed_temperature(schedule, 2.0, 0.05, 0, 6) == pytest.approx(2.0)
        assert annealed_temperature(schedule, 2.0, 0.05, 5, 6) == pytest.approx(0.05)

    @pytest.mark.parametrize("schedule", ["linear", "exponential", "logarithmic"])
    def test_monotone_decreasing(self, schedule):
        temps = [annealed_temperature(schedule, 2.0, 0.05, k, 8) for k in range(8)]
        assert all(a >= b - 1e-12 for a, b in zip(temps, temps[1:]))

    def test_exponential_cools_faster_than_linear_early(self):
        lin = annealed_temperature("linear", 2.0, 0.02, 1, 8)
        exp = annealed_temperature("exponential", 2.0, 0.02, 1, 8)
        assert exp < lin                            # geometric drop is steeper early

    def test_step_clamped(self):
        # step beyond the horizon stays at the final temperature.
        assert annealed_temperature("linear", 2.0, 0.05, 99, 6) == pytest.approx(0.05)

    def test_unknown_schedule_raises(self):
        with pytest.raises(ValueError):
            annealed_temperature("cubic", 1.0, 0.1, 0, 5)


# --------------------------------------------------------------------------- #
# VASQE driver.
# --------------------------------------------------------------------------- #

class TestVASQE:
    def test_low_temperature_reduces_to_adapt(self, h2_hamiltonian, h2_fci):
        # tau -> 0 always picks the largest-gradient operator: ADAPT-VQE / FCI.
        r = _vasqe(h2_hamiltonian, temperature=1e-4).run()
        assert isinstance(r, VASQEResult)
        assert r.energy_unit == "eV"
        assert r.optimal_energy == pytest.approx(h2_fci, abs=TOL_1E6)

    def test_result_records_schedule(self, h2_hamiltonian):
        r = _vasqe(h2_hamiltonian, temperature=1.5, final_temperature=0.02,
                   schedule="linear", max_iterations=6).run()
        assert r.schedule == "linear"
        assert r.initial_temperature == 1.5
        assert r.final_temperature == 0.02
        assert len(r.temperatures) == r.num_operators   # one tau per grown op

    def test_result_is_adaptvqe_result_subclass(self, h2_hamiltonian):
        r = _vasqe(h2_hamiltonian, temperature=0.1).run()
        assert isinstance(r, ADAPTVQEResult)            # inherits ADAPT fields
        assert r.metrics is None or hasattr(r.metrics, "num_operators")

    def test_reproducible_with_seed(self, h2_hamiltonian):
        r1 = _vasqe(h2_hamiltonian, temperature=3.0, seed=7, max_iterations=5).run()
        r2 = _vasqe(h2_hamiltonian, temperature=3.0, seed=7, max_iterations=5).run()
        assert r1.operators == r2.operators             # same seed -> same picks

    def test_high_temperature_selection_is_stochastic(self, h2_hamiltonian):
        # With a hot temperature different seeds should not all agree on the first
        # picked operator (the pool has several operators to choose from).
        first_ops = set()
        for seed in range(12):
            r = _vasqe(h2_hamiltonian, temperature=50.0, seed=seed,
                       max_iterations=1).run()
            if r.operators:
                first_ops.add(r.operators[0])
        assert len(first_ops) >= 2                       # genuinely random choice

    def test_annealing_reaches_fci(self, h2_hamiltonian, h2_fci):
        r = _vasqe(h2_hamiltonian, temperature=2.0, final_temperature=1e-3,
                   schedule="exponential", max_iterations=10, seed=1).run()
        assert r.optimal_energy == pytest.approx(h2_fci, abs=TOL_1E6)

    def test_invalid_schedule_rejected(self, h2_hamiltonian):
        with pytest.raises(ValueError):
            _vasqe(h2_hamiltonian, schedule="parabolic")

    def test_nonpositive_temperature_rejected(self, h2_hamiltonian):
        with pytest.raises(ValueError):
            _vasqe(h2_hamiltonian, temperature=0.0)

    def test_is_ase_calculator(self):
        from ase import Atoms
        atoms = Atoms("H2", positions=[[4.0, 4.0, 3.63], [4.0, 4.0, 4.37]],
                      cell=[[8.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]], pbc=True)
        atoms.calc = Mandacaru(method="vasqe", basis="HAO", h=0.30,
                               temperature=1e-3, trace=False, profile=False,
                               gradient_tolerance=1e-4, max_iterations=10)
        atoms.get_potential_energy()
        assert isinstance(atoms.calc.result, VASQEResult)


# --------------------------------------------------------------------------- #
# Excited-state extensions (deflation + subspace) inherit stochastic selection.
# --------------------------------------------------------------------------- #

class TestVASQEExcitedStates:
    def test_energy_levels_deflation(self, h2_hamiltonian, h2_fci):
        levels = _vasqe(h2_hamiltonian, temperature=1e-3).energy_levels(2)
        m = h2_hamiltonian.map_to_qubits("jordan_wigner").to_matrix()
        spectrum = (np.sort(np.linalg.eigvalsh(0.5 * (m + m.conj().T)).real)
                    * HARTREE_TO_EV)
        assert levels.ground_state_energy == pytest.approx(h2_fci, abs=TOL_1E6)
        for e in levels.energies:                        # true eigenvalues
            assert float(np.min(np.abs(spectrum - e))) < TOL_1E5

    def test_subspace_vasqe(self, h2_hamiltonian, h2_fci):
        sv = Mandacaru(method="subspace-vasqe", hamiltonian=h2_hamiltonian,
                       pool="fermionic", num_states=2, num_particles=(1, 1),
                       n_spatial_orbitals=2, temperature=0.5,
                       optimizer=Optimizer("L-BFGS", maxiter=2000),
                       trace=False, profile=False, gradient_tolerance=1e-6,
                       max_iterations=20, seed=1)
        result = sv.run()
        assert result.num_states == 2
        assert result.energies[0] == pytest.approx(h2_fci, abs=TOL_1E4)
        assert result.energies[1] >= result.energies[0] - 1e-9

    def test_subspace_vasqe_uses_stochastic_selection(self, h2_hamiltonian):
        # SubspaceVASQE must route selection through VASQE._select_operator.
        sv = Mandacaru(method="subspace-vasqe", hamiltonian=h2_hamiltonian,
                       pool="fermionic", num_states=2, num_particles=(1, 1),
                       n_spatial_orbitals=2, temperature=1.0, trace=False,
                       profile=False)
        assert sv._select_operator.__qualname__.startswith("VASQE")


# --------------------------------------------------------------------------- #
# Selection hook is behavior-preserving for ADAPTVQE.
# --------------------------------------------------------------------------- #

def test_schedule_names_exported():
    assert TEMPERATURE_SCHEDULES == ("constant", "exponential", "linear",
                                     "logarithmic")


# --------------------------------------------------------------------------- #
# Packaging: VASQE is experimental, ADAPT-VQE is the default.
# --------------------------------------------------------------------------- #

class TestExperimentalPackaging:
    def test_not_exported_from_stable_algorithms(self):
        import mandacaru.algorithms as algorithms
        for name in ("VASQE", "VASQEResult", "SubspaceVASQE"):
            assert not hasattr(algorithms, name)
            assert name not in algorithms.__all__

    def test_registered_with_the_calculator(self):
        from mandacaru.algorithms import (DEFAULT_METHOD, STABLE_METHODS,
                                          available_methods,
                                          experimental_methods, resolve_method)
        assert DEFAULT_METHOD == "adapt-vqe"
        assert "vasqe" in experimental_methods()
        assert "subspace-vasqe" in experimental_methods()
        assert not set(experimental_methods()) & set(STABLE_METHODS)
        assert set(available_methods()) == (set(STABLE_METHODS)
                                            | set(experimental_methods()))
        assert resolve_method("vasqe")[1] is VASQE
        assert resolve_method("subspace-vasqe")[1] is SubspaceVASQE

    def test_stable_code_never_names_vasqe(self):
        import pathlib
        import mandacaru
        root = pathlib.Path(mandacaru.__file__).parent
        for path in root.rglob("*.py"):
            if "experimental" in path.parts:
                continue
            assert "vasqe" not in path.read_text().lower(), path

    def test_calculators_default_to_adapt_vqe(self):
        calc = Mandacaru()
        assert calc.method == "adapt-vqe"
        assert type(calc.solver).__name__ == "ADAPTVQE"

    def test_calculator_and_dry_run_accept_the_method(self):
        from ase import Atoms
        from mandacaru.algorithms import Mandacaru
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[6.0] * 3)
        for method in ("vasqe", "subspace-vasqe"):
            atoms.calc = Mandacaru(method=method, dry_run=True)
            assert np.isnan(atoms.get_potential_energy())
            assert atoms.calc.dry_run_result.method == method


# --------------------------------------------------------------------------- #
# Moved from the stable test files: quenching, the Hamiltonian cache and the
# periodic Bloch calculator, each exercised with VASQE.
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def lih_cache(tmp_path_factory):
    from ase import Atoms
    path = str(tmp_path_factory.mktemp("cache") / "lih.json")
    atoms = Atoms("LiH", positions=[[0, 0, 0], [0, 0, 1.6]], cell=[7.0] * 3)
    atoms.calc = Mandacaru(method="adapt-vqe", pool="qeb", basis="HAO", h=0.4,
                           trace=False, profile=False, max_iterations=1,
                           save_hamiltonian=path, hamiltonian_format="json")
    atoms.get_potential_energy()
    return path


class TestVASQEQuenching:
    def test_default_is_quenched(self):
        assert Mandacaru(method="vasqe").quenching is True

    def test_quenched_vasqe_freezes_earlier_parameters(self, lih_cache):
        result = Mandacaru(method="vasqe", pool="qeb",
                           load_hamiltonian=lih_cache, trace=False,
                           profile=False, quenching=False, max_iterations=3,
                           gradient_tolerance=1e-8, temperature=1e-6, seed=0).run()
        assert result.num_operators >= 2

    def test_low_temperature_quenched_vasqe_tracks_quenched_adapt(self,
                                                                   lih_cache):
        """tau -> 0 reduces VASQE to ADAPT-VQE, quenching policy included."""
        common = dict(pool="qeb", load_hamiltonian=lih_cache, trace=False,
                      profile=False, quenching=False, max_iterations=3,
                      gradient_tolerance=1e-8)
        adapt = Mandacaru(method="adapt-vqe", **common).run()
        vasqe = Mandacaru(method="vasqe", **common, temperature=1e-6, seed=0).run()
        # Symmetry-degenerate operators tie on their gradient, so the *set*
        # selected (and the energy) must match, not the tie-broken order.
        assert set(vasqe.operators) == set(adapt.operators)
        assert vasqe.optimal_energy == pytest.approx(adapt.optimal_energy,
                                                     abs=TOL_1E8)


class TestHamiltonianCache:
    def test_vasqe_loads_a_cached_hamiltonian(self, lih_cache):
        result = Mandacaru(method="vasqe", pool="fermionic",
                           load_hamiltonian=lih_cache, trace=False,
                           profile=False, max_iterations=2, temperature=1e-6).run()
        assert isinstance(result, VASQEResult)


class TestBloch:
    def test_total_energy_through_the_periodic_method(self):
        """The periodic layer composes over a *registered* method too.

        ``bloch-vasqe`` is registered by this package, not by the stable
        resolver, so this is what keeps ``register_method`` decoupled from
        which methods the periodic driver knows about.
        """
        from ase import Atoms
        from mandacaru.algorithms import Mandacaru
        atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]],
                      cell=[[1.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                      pbc=[True, False, False])
        atoms.calc = Mandacaru(method="bloch-vasqe",
                               kpts={"size": (2, 1, 1), "gamma": True},
                               basis="HAO", h=0.40,
                               optimizer=Optimizer("L-BFGS", maxiter=2000),
                               temperature=1.0, max_iterations=6,
                               gradient_tolerance=1e-3, profile=False,
                               trace=False)
        e_cell = atoms.get_potential_energy()
        result = atoms.calc.result
        assert isinstance(result, VASQEResult) and np.isfinite(e_cell)
        assert len(result.temperatures) == result.num_operators
