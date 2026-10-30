"""Molecular precision controls must select owned native storage."""
from ase import Atoms
from ase.build import bulk, molecule
import numpy as np
import pytest

from mandacaru import Mandacaru
from mandacaru.units import HARTREE_TO_EV


def test_native_precision_changes_sample_storage_and_preserves_total_energy():
    atoms = Atoms("H2", positions=[[2.5, 2.5, 2.1], [2.5, 2.5, 2.9]], cell=[5.] * 3)
    energies = []
    samples = []
    for precision, dtype in (("float64", np.float64), ("float32", np.float32)):
        with Mandacaru(method="rhf", atoms=atoms, basis="HAO", precision=precision,
                       grid_spacing=.5, resources={"threads": 1}) as calc:
            energies.append(atoms.get_potential_energy())
            snapshot = calc.integrals.snapshot(include_samples=True)
            assert snapshot["precision"] == precision
            assert snapshot["samples"].dtype == dtype
            assert snapshot["orthonormal_samples"].dtype == dtype
            assert snapshot["one_body"].dtype == np.float64
            samples.append(snapshot["samples"])
            assert calc.result.success
    assert samples[1].nbytes * 2 == samples[0].nbytes
    assert energies[1] == pytest.approx(energies[0], abs=1e-5)
    assert np.array_equal(samples[1], samples[0].astype(np.float32))


@pytest.mark.parametrize("name", ["single", "double", "Float32", "FLOAT64", " float64",
                                  "quadruple", "", None])
def test_invalid_precision_is_rejected_before_setup(name):
    with pytest.raises(ValueError, match="precision must be 'float32' or 'float64'"):
        Mandacaru(precision=name)


@pytest.mark.parametrize("name", ["float32", "float64"])
def test_the_two_precision_names_are_accepted(name):
    assert Mandacaru(method="dft", precision=name)._precision == name


def test_the_default_precision_is_float64():
    assert Mandacaru()._precision == "float64"


def test_periodic_float32_precision_and_repeated_ase_cell_updates():
    atoms = bulk("Al", "fcc", a=4.05)
    with Mandacaru(method="dft", xc="lda", atoms=atoms, basis="PAW-LCAO",
                   grid_spacing=.8, precision="float32",
                   resources={"threads": 1}) as calc:
        original = atoms.get_potential_energy()
        assert calc.result.success
        assert calc.periodic_problem.precision == "float32"
        assert calc.result.scf.n_iterations > 0
        assert atoms.get_magnetic_moment() == pytest.approx(0.)
        atoms.set_cell(atoms.cell * 1.01, scale_atoms=True)
        updated = atoms.get_potential_energy()
        assert calc.result.success
        assert updated != pytest.approx(original, abs=1e-5)
        assert not calc.calculation_required(atoms, ["energy"])


def _h2():
    return Atoms("H2", positions=[[2.5, 2.5, 2.1], [2.5, 2.5, 2.9]], cell=[5.] * 3)


class TestRouteRefusals:
    """What float32 cannot honor raises before any setup or SCF."""

    @pytest.mark.parametrize("source", [
        {"load_hamiltonian": "unused.parquet"},
        {"hamiltonian": object(), "n_spatial_orbitals": 2, "num_particles": (1, 1)},
        {"native_model": object()},
    ])
    def test_a_cached_problem_on_a_circuit_device_builds_nothing_to_store(self, source):
        with pytest.raises(ValueError, match="builds neither"):
            Mandacaru(method="vqe", precision="float32", shots=100, **source)

    def test_the_circuit_device_is_not_refused_for_a_geometry_run(self):
        """float32 then acts on the orbital samples of the integral build."""
        assert Mandacaru(method="vqe", precision="float32", shots=100)._precision == "float32"

    def test_a_classical_method_accepts_float32_for_the_sample_build(self):
        assert Mandacaru(method="rhf", precision="float32")._precision == "float32"
        assert Mandacaru(method="dft", precision="float32")._precision == "float32"

    def test_forces_of_a_quantum_method_are_refused_before_anything_runs(self, monkeypatch):
        atoms = _h2()
        calc = Mandacaru(method="vqe", precision="float32", shots=100,
                         resources={"threads": 1})
        ran = []
        monkeypatch.setattr(calc, "run", lambda *a, **k: ran.append(a))
        monkeypatch.setattr(calc, "_mean_field", lambda *a, **k: ran.append(a))
        with pytest.raises(ValueError, match="quantum-method forces.*1e-3 Bohr"):
            calc.calculate(atoms, properties=("energy", "forces"))
        assert not ran
        assert calc.result is None and calc.integrals is None and calc.results == {}

    def test_a_cached_energy_does_not_let_float32_forces_through(self):
        """The refusal precedes the cached-result lookup."""
        atoms = _h2()
        atoms.calc = Mandacaru(method="rhf", basis={"name": "PAW-LCAO"},
                               precision="float32", grid_spacing=.5,
                               resources={"threads": 1})
        atoms.calc.result = object()
        atoms.calc.results = {"energy": 0.0}
        with pytest.raises(ValueError, match="PAW-LCAO forces"):
            atoms.calc.calculate(atoms, properties=("energy", "forces"))

    def test_float64_forces_pass_the_refusal(self, monkeypatch):
        calc = Mandacaru(method="rhf", basis={"name": "PAW-LCAO"}, precision="float64",
                         grid_spacing=.5, resources={"threads": 1})
        reached = []

        def stop(*args, **kwargs):
            reached.append(1)
            raise RuntimeError("stop")

        monkeypatch.setattr(calc, "run", stop)
        with pytest.raises(RuntimeError, match="stop"):
            calc.calculate(_h2(), properties=("energy", "forces"))
        assert reached


class TestQuantumFloors:
    """A setting the complex64 state cannot resolve raises, naming the floor."""

    @property
    def energy_floor(self):
        from mandacaru import _native

        return _native.FLOAT32_ENERGY_FLOOR_HA

    @property
    def gradient_floor(self):
        from mandacaru import _native

        return _native.FLOAT32_GRADIENT_FLOOR

    def test_the_floors_are_positive_rust_constants_in_hartree(self):
        assert 0 < self.energy_floor < 1e-3
        assert 0 < self.gradient_floor < 1e-3

    @pytest.mark.parametrize("method", ["vqe", "adapt-vqe", "subspace-vqe",
                                        "subspace-adapt-vqe"])
    def test_an_optimizer_tolerance_below_the_floor_raises(self, method):
        below = self.energy_floor / 100
        with pytest.raises(ValueError, match=rf"optimizer tol {below:g} Ha.*float32 energy floor {self.energy_floor:g} Ha"):
            Mandacaru(method=method, precision="float32",
                      optimizer={"method": "SLSQP", "tol": below})

    def test_a_tolerance_at_the_floor_is_accepted(self):
        calc = Mandacaru(method="vqe", precision="float32",
                         optimizer={"method": "SLSQP", "tol": self.energy_floor})
        assert calc._optimizer["tol"] == self.energy_floor

    @pytest.mark.parametrize("method", ["adapt-vqe", "subspace-adapt-vqe"])
    def test_an_adapt_gradient_threshold_below_the_floor_raises(self, method):
        below = self.gradient_floor / 100
        with pytest.raises(ValueError, match=rf"gradient.*{below:g}.*floor {self.gradient_floor:g} Ha/rad"):
            Mandacaru(method=method, precision="float32", convergence={"gradient": below})

    def test_an_adapt_energy_threshold_below_the_floor_raises(self):
        from mandacaru import _native

        below_ev = self.energy_floor * _native.HARTREE_TO_EV / 100
        with pytest.raises(ValueError, match=rf"convergence\['energy'\].*floor {self.energy_floor:g} Ha"):
            Mandacaru(method="adapt-vqe", precision="float32",
                      convergence={"energy": below_ev})

    def test_a_tolerance_of_none_resolves_below_the_floor_and_raises_before_the_scf(self):
        with pytest.raises(ValueError, match=rf"tol=None.*float32 energy floor {self.energy_floor:g} Ha"):
            Mandacaru(method="vqe", precision="float32",
                      optimizer={"method": "SLSQP", "tol": None})
        # SPSA without a tolerance runs its whole budget and tests nothing.
        Mandacaru(method="vqe", precision="float32",
                  optimizer={"method": "SPSA", "tol": None})

    def test_the_same_settings_are_fine_in_float64(self):
        Mandacaru(method="adapt-vqe", convergence={"gradient": 1e-7, "energy": 1e-9},
                  optimizer={"method": "SLSQP", "tol": 1e-12})

    def test_an_unset_tolerance_resolves_to_the_floor_and_float64_keeps_its_own(self):
        assert Mandacaru(method="vqe", precision="float32")._optimizer["tol"] == self.energy_floor
        assert Mandacaru(method="vqe")._optimizer["tol"] == 1e-12

    def test_a_circuit_device_has_no_state_vector_and_no_floor(self):
        calc = Mandacaru(method="adapt-vqe", precision="float32", shots=100,
                         convergence={"gradient": 1e-7})
        assert calc._optimizer["tol"] is None

    @pytest.mark.parametrize("source", [
        {"load_hamiltonian": "unused.parquet"},
        {"hamiltonian": object(), "n_spatial_orbitals": 2, "num_particles": (1, 1)},
    ])
    def test_cached_problems_are_accepted_on_the_state_vector_device(self, source):
        assert Mandacaru(method="vqe", precision="float32", **source)._precision == "float32"


class TestQuantumStateVector:
    """float32 reaches the solver: a complex64 state, energy within the gate."""

    def test_h2_vqe_state_is_complex64_and_the_energy_is_inside_the_gate(self):
        energies, dtypes = {}, {}
        for precision in ("float64", "float32"):
            atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, .74]], cell=[6.] * 3)
            with Mandacaru(method="vqe", basis="HAO", grid_spacing=.35,
                           precision=precision, resources={"threads": 1}) as calc:
                atoms.calc = calc
                energies[precision] = atoms.get_potential_energy() / HARTREE_TO_EV
                assert calc.problem.precision == precision
                dtypes[precision] = calc.result.state().dtype
        assert dtypes == {"float64": np.complex128, "float32": np.complex64}
        assert energies["float32"] == pytest.approx(energies["float64"], abs=5e-4)

    def test_forces_of_a_state_vector_method_are_refused_before_the_scf(self, monkeypatch):
        atoms = _h2()
        calc = Mandacaru(method="vqe", precision="float32", resources={"threads": 1})
        ran = []
        monkeypatch.setattr(calc, "run", lambda *a, **k: ran.append(a))
        with pytest.raises(ValueError, match="quantum-method forces"):
            calc.calculate(atoms, properties=("energy", "forces"))
        assert not ran and calc.result is None and calc.integrals is None


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    """One Al crystal file per precision."""
    folder = tmp_path_factory.mktemp("crystals")
    paths = {}
    for precision in ("float64", "float32"):
        path = str(folder / f"al_{precision}.json")
        atoms = bulk("Al", "fcc", a=4.05)
        with Mandacaru(method="dft", xc="lda", atoms=atoms, basis="PAW-LCAO",
                       grid_spacing=.8, precision=precision, save_crystal=path,
                       resources={"threads": 1}):
            atoms.get_potential_energy()
        paths[precision] = path
    return paths


class TestCrystalFilePrecision:
    """A crystal file carries its precision; the constructor must equal it."""

    @pytest.mark.parametrize("stored, asked", [("float32", "float64"),
                                               ("float64", "float32")])
    def test_a_mismatch_raises_naming_both_values(self, files, stored, asked):
        with Mandacaru(method="dft", load_crystal=files[stored], precision=asked,
                       resources={"threads": 1}) as calc:
            with pytest.raises(ValueError) as raised:
                calc.run()
            message = str(raised.value)
            assert f"stores {stored}" in message and f"precision={asked!r}" in message
            assert calc.result is None

    @pytest.mark.parametrize("name", ["float64", "float32"])
    def test_the_matching_precision_loads_and_solves(self, files, name):
        with Mandacaru(method="dft", load_crystal=files[name], precision=name,
                       resources={"threads": 1}) as calc:
            calc.run()
            assert calc.result.success
            assert calc.periodic_problem.precision == name


class TestMemoryBudget:
    """With 0.75 x the float64 peak, float64 cannot run and float32 can."""

    @staticmethod
    def _run(precision, limit):
        atoms = molecule("H2O")
        atoms.center(vacuum=4.0)
        with Mandacaru(method="rhf", basis={"name": "NAO", "size": "DZ"},
                       grid_spacing=.25, precision=precision,
                       resources={"threads": 2, "memory_limit_bytes": limit}) as calc:
            atoms.calc = calc
            energy = atoms.get_potential_energy()
            return energy / HARTREE_TO_EV, calc.context.memory_peak_bytes

    def test_float32_fits_a_budget_float64_exceeds(self):
        energy64, peak64 = self._run("float64", 2 * 1024**3)
        limit = int(0.75 * peak64)
        with pytest.raises(MemoryError):
            self._run("float64", limit)
        energy32, peak32 = self._run("float32", limit)
        assert peak32 <= limit
        assert energy32 == pytest.approx(energy64, abs=1e-5)
