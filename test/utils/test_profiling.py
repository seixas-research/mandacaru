# -*- coding: utf-8 -*-
# file: test_profiling.py

"""Profiling of the simulation stages: timing, peak memory and the processor layout.

Covers the resource-reporting features:

* the native engine exposes its default worker count (``_native.num_threads``);
* a calculation times its integral stages and records its processor layout and
  peak memory;
* :class:`~mandacaru.utils.profiling.Timings` accumulates named stage wall-times.
"""

import pytest

from mandacaru import _native
from mandacaru.resources import layout_label
from mandacaru.utils.profiling import Timings, peak_memory_mb
from ase import Atoms
from mandacaru import Mandacaru


class TestBackendCores:
    def test_num_threads_reports_positive_int_when_native_backend(self):
        n = _native.num_threads()
        assert isinstance(n, int) and n >= 1

    def test_the_layout_label_names_every_level_and_the_processor_count(self):
        context = _native.Context(threads=2, kpts=3, memory_limit_bytes=1 << 20)
        assert layout_label(context) == "kpts 3 x bands 1 x threads 2 (6 workers)"
        assert context.parallelization == {"kpts": 3, "bands": 1, "threads": 2}
        assert context.workers == 6

    def test_the_cores_for_a_grid_product_are_the_hardware_threads(self):
        assert _native.grid_blas_threads() >= 1

    def test_peak_memory_is_positive(self):
        assert peak_memory_mb() > 0.0


class TestTimings:
    def test_stage_accumulates(self):
        t = Timings(parallelization="kpts 2 x bands 1 x threads 2 (4 workers)")
        with t.time("stage-a"):
            _ = sum(range(1000))
        with t.time("stage-a"):
            _ = sum(range(1000))
        with t.time("stage-b"):
            _ = sum(range(1000))
        assert set(t.stages) == {"stage-a", "stage-b"}
        assert t.total == pytest.approx(sum(t.stages.values()))
        d = t.as_dict()
        assert d["parallelization"] == "kpts 2 x bands 1 x threads 2 (4 workers)"
        assert "peak_memory_mb" in d

    def test_format_report_mentions_the_layout_and_memory(self):
        t = Timings(parallelization="kpts 1 x bands 1 x threads 8 (8 workers)")
        t.add("only-stage", 0.01)
        t.wall_time = 0.02
        report = t.format_report()
        assert "parallelization" in report and "8 workers" in report
        assert "peak memory" in report
        assert "total wall time" in report
        resources = t.resources()
        assert resources["parallelization"] == "kpts 1 x bands 1 x threads 8 (8 workers)"
        assert "threads" not in resources


class TestADAPTProfiling:
    def test_calculator_summary_includes_integration(self, capsys):
        atoms = Atoms("H2", positions=[[3, 3, 2.63], [3, 3, 3.37]],
                      cell=[[6, 0, 0], [0, 6, 0], [0, 0, 6]])
        atoms.calc = Mandacaru(method="adapt-vqe", pool="ceo", basis="HAO",
                               grid_spacing=0.4, trace=True, max_iterations=4,
                               convergence={"gradient": 1e-3})
        atoms.get_total_energy()
        out = capsys.readouterr().out
        assert "integration:" in out
        assert atoms.calc.result.timings is not None


class TestExclusiveProfiling:
    def test_nested_stages_do_not_double_count(self, monkeypatch):
        from mandacaru.utils import profiling
        clock = iter([0.0, 3.0, 7.0, 10.0])
        monkeypatch.setattr(profiling.time, "perf_counter", lambda: next(clock))
        timings = Timings()
        with timings.exclusive("outer"):
            with timings.exclusive("inner"):
                pass
        assert timings.stages == {"inner": 4.0, "outer": 6.0}
        assert timings.total == 10.0

    def test_scope_and_stage_are_restored_after_an_exception(self):
        from mandacaru.utils.profiling import profile_stage
        timings = Timings()

        @profile_stage("failed stage")
        def fail():
            raise RuntimeError("expected failure")

        with pytest.raises(RuntimeError):
            with timings.activate():
                fail()
        recorded = dict(timings.stages)
        with pytest.raises(RuntimeError):
            fail()
        assert timings.stages == recorded
        assert timings._frames == []
