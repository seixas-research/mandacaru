# -*- coding: utf-8 -*-
# file: test_profiling.py

"""Profiling of the simulation stages: timing, peak memory and core count.

Covers the resource-reporting features:

* the C backend exposes its OpenMP thread count (``_backend.num_threads``);
* the real-space integral engine times its one-/two-body stages and records the
  cores / backend / peak memory (``IntegralEngine.integration_profile``);
* :class:`~mandacaru.utils.profiling.Timings` accumulates named stage wall-times.
"""

import numpy as np
import pytest

from mandacaru.core import MolecularIntegrals, minimal_hao_basis
from mandacaru.integrals import Grid, IntegralEngine
from mandacaru.integrals import _backend
from mandacaru.utils.profiling import Timings, backend_cores, peak_memory_mb
from ase import Atoms
from mandacaru.algorithms import Mandacaru


class TestBackendCores:
    def test_num_threads_reports_positive_int_when_c_backend(self):
        n = _backend.num_threads()
        if _backend.HAS_C_BACKEND:
            assert isinstance(n, int) and n >= 1
            assert backend_cores() == n
        else:
            assert n is None

    def test_peak_memory_is_positive(self):
        assert peak_memory_mb() > 0.0


class TestTimings:
    def test_stage_accumulates(self):
        t = Timings(n_cores=4, backend="C (OpenMP)")
        with t.time("stage-a"):
            _ = sum(range(1000))
        with t.time("stage-a"):
            _ = sum(range(1000))
        with t.time("stage-b"):
            _ = sum(range(1000))
        assert set(t.stages) == {"stage-a", "stage-b"}
        assert t.total == pytest.approx(sum(t.stages.values()))
        d = t.as_dict()
        assert d["n_cores"] == 4 and d["backend"] == "C (OpenMP)"
        assert "peak_memory_mb" in d

    def test_format_report_mentions_cores_and_memory(self):
        t = Timings(n_cores=8, backend="C (OpenMP)")
        t.add("only-stage", 0.01)
        t.wall_time = 0.02
        report = t.format_report()
        assert "cores (OpenMP threads)" in report
        assert "peak memory" in report
        assert "total wall time" in report


@pytest.fixture
def h2_integrals():
    nuclei = [(1.0, np.array([0.0, 0.0, -0.37])),
              (1.0, np.array([0.0, 0.0, 0.37]))]
    grid = Grid(center=[0, 0, 0], box_size=5.0, h=0.3)
    return MolecularIntegrals(nuclei, minimal_hao_basis(nuclei), grid)


@pytest.fixture(scope="module")
def h2_hamiltonian():
    """A 4-qubit H2 Hamiltonian -- something for the driver to time."""
    R = 0.74
    nuclei = [(1.0, np.array([0.0, 0.0, -R / 2])),
              (1.0, np.array([0.0, 0.0, +R / 2]))]
    grid = Grid(center=[0, 0, 0], box_size=5.0, h=0.35)
    integrals = MolecularIntegrals(nuclei, minimal_hao_basis(nuclei), grid)
    return integrals.molecular_hamiltonian(mo_basis=True, n_electrons=2)


class TestIntegrationProfile:
    def test_engine_times_one_and_two_body(self):
        orb = minimal_hao_basis([(1.0, np.array([0.0, 0.0, 0.0]))])
        g = Grid(center=[0, 0, 0], box_size=5.0, h=0.4)
        eng = IntegralEngine(orb, g)
        eng.one_body(lambda x, y, z: np.zeros(np.broadcast(x, y, z).shape),
                     energy_units="Ha")
        eng.two_body(method="fft", energy_units="Ha")
        prof = eng.integration_profile()
        assert "one-body integrals" in prof["stages_s"]
        assert any("two-body" in k for k in prof["stages_s"])
        assert prof["total_s"] >= 0.0
        assert prof["peak_memory_mb"] > 0.0
        # Cores: an int with the C backend, None with the NumPy fallback.
        if _backend.HAS_C_BACKEND:
            assert prof["n_cores"] == _backend.num_threads()
            assert prof["backend"].startswith("C (")
        else:
            assert prof["n_cores"] is None

    def test_molecular_integrals_exposes_profile(self, h2_integrals):
        h2_integrals.molecular_hamiltonian(mo_basis=True, n_electrons=2)
        prof = h2_integrals.integration_profile()
        assert prof["stages_s"] and prof["total_s"] >= 0.0


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
        assert "[PERFORMANCE]" in out
        assert "gradient screening" in out
        assert "parameter optimization" in out
        assert "openmp_threads" in out
        assert "peak_memory_MiB" in out

    def test_calculator_summary_includes_integration(self, capsys):
        atoms = Atoms("H2", positions=[[3, 3, 2.63], [3, 3, 3.37]],
                      cell=[[6, 0, 0], [0, 6, 0], [0, 0, 6]], pbc=True)
        atoms.calc = Mandacaru(method="adapt-vqe", pool="ceo", basis="HAO",
                               h=0.4, trace=True, max_iterations=4,
                               gradient_tolerance=1e-3)
        atoms.get_total_energy()
        out = capsys.readouterr().out
        assert "integration:" in out
        assert atoms.calc.result.integration_profile is not None
