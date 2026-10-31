"""Ensure migration evidence cannot silently accept incomplete or invalid runs."""

import copy
import json
import math
from pathlib import Path

import pytest

from . import baseline, inventory, suite


def report():
    sample = {"status": "ok", "values": {"energy_ev": -30.0, "tensor": [[1.0, 2.0]]},
              "contract": {"energy_unit": "eV"}, "checks": {"converged": True},
              "compute_wall_s": 1.0}
    return {"schema_version": 1, "cases_sha256": "inputs", "fixture_sha256": "fixture",
            "cases": {"h2": {"status": "ok", "definition": {"atol": 1e-7, "rtol": 1e-9},
                              "dataset_inputs": [], "samples": [sample]}}}


def test_comparison_allows_roundoff_but_ignores_iteration_and_timing_changes():
    reference = report()
    candidate = copy.deepcopy(reference)
    sample = candidate["cases"]["h2"]["samples"][0]
    sample["values"]["energy_ev"] += 1e-8
    sample["compute_wall_s"] = 20.0
    sample["diagnostics"] = {"num_evaluations": 45}
    assert not baseline.compare(reference, candidate)


@pytest.mark.parametrize("mutation", ["missing_case", "skipped", "empty", "failed_repeat",
                                     "changed_input", "changed_dataset", "units", "shape",
                                     "missing_value", "energy", "nan", "false_check", "missing_check"])
def test_comparison_rejects_incomplete_or_incompatible_evidence(mutation):
    reference = report()
    candidate = copy.deepcopy(reference)
    case = candidate["cases"]["h2"]
    sample = case["samples"][0]
    if mutation == "missing_case":
        candidate["cases"].clear()
    elif mutation == "skipped":
        case["status"] = "skipped"
    elif mutation == "empty":
        case["samples"] = []
    elif mutation == "failed_repeat":
        case["samples"].append({**sample, "status": "failed"})
    elif mutation == "changed_input":
        candidate["fixture_sha256"] = "other"
    elif mutation == "changed_dataset":
        case["dataset_inputs"] = [{"sha256": "other"}]
    elif mutation == "units":
        sample["contract"]["energy_unit"] = "Ha"
    elif mutation == "shape":
        sample["values"]["tensor"] = [1.0, 2.0]
    elif mutation == "missing_value":
        del sample["values"]["tensor"]
    elif mutation == "energy":
        sample["values"]["energy_ev"] += 1e-3
    elif mutation == "nan":
        sample["values"]["energy_ev"] = math.nan
    elif mutation == "false_check":
        sample["checks"]["converged"] = False
    elif mutation == "missing_check":
        sample["checks"].clear()
    assert baseline.compare(reference, candidate)


def test_reference_repeats_must_also_be_consistent():
    reference = report()
    bad = copy.deepcopy(reference["cases"]["h2"]["samples"][0])
    bad["values"]["energy_ev"] += 0.1
    reference["cases"]["h2"]["samples"].append(bad)
    assert baseline.compare(reference, report())


def test_empty_or_truthy_invariant_evidence_is_rejected():
    empty = report()
    empty["cases"] = {}
    assert baseline.compare(empty, empty)
    invalid = report()
    invalid["cases"]["h2"]["samples"][0]["checks"]["converged"] = "false"
    assert baseline.compare(report(), invalid)


def test_nonfinite_evidence_cannot_be_written(tmp_path):
    with pytest.raises(ValueError):
        baseline.write_json(tmp_path / "result.json", {"energy": math.inf})


def test_controlled_environment_cannot_inherit_an_alternate_library_or_record_store(monkeypatch, tmp_path):
    monkeypatch.setenv("MANDACARU_INTEGRALS_LIB", "/other/library")
    monkeypatch.setenv("MANDACARU_PROPOSAL_DATA", "/user/training")
    monkeypatch.setenv("OMP_NUM_THREADS", "24")
    env = baseline.controlled_env("rust", 2, tmp_path)
    assert "MANDACARU_INTEGRALS_LIB" not in env
    assert "MANDACARU_PROPOSAL_DATA" not in env
    assert all(env[key] == "2" for key in baseline.THREAD_ENV)
    assert env["MANDACARU_BACKEND"] == "rust"


def test_source_fingerprint_tracks_content_and_excludes_build_products(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    src = tmp_path / "src" / "mandacaru"
    src.mkdir(parents=True)
    code = src / "a.py"
    code.write_text("x = 1\n")
    target = src / "rust" / "target"
    target.mkdir(parents=True)
    (target / "generated.rs").write_text("generated")
    before = inventory.source_manifest(tmp_path)
    assert len(before) == 2
    code.write_text("x = 2\n")
    assert inventory.digest_manifest(before) != inventory.digest_manifest(inventory.source_manifest(tmp_path))


def test_capability_index_and_workloads_have_valid_references():
    data = json.loads((baseline.HERE / "capabilities.json").read_text())
    case_list = baseline.cases()
    ids = {case["id"] for case in case_list}
    assert len(ids) == len(case_list)
    for case in case_list:
        assert case["atol"] > 0 and case["rtol"] >= 0
    for group in data["groups"]:
        assert set(group["reference_cases"]) <= ids
        for path in group["sources"] + group["tests"]:
            assert (baseline.ROOT / path).is_file(), path


LIVE_BACKEND_KEYS = {"available", "message", "threads", "library_sha256"}


def test_a_sample_records_the_native_backend_the_runner_writes_today():
    from mandacaru.integrals._backend import check_backend
    status = check_backend()
    record = baseline.backend_record(status)
    assert set(record) == LIVE_BACKEND_KEYS and "is_rust" not in record
    assert record["available"] is True and record["threads"] == status.n_threads >= 1
    assert record["message"] == status.message
    assert record["library_sha256"] == baseline.fingerprint(Path(status.path))


def test_checked_reference_corpus_is_complete_and_internally_consistent():
    reference = baseline.HERE / "reference"
    numpy = baseline.read_json(reference / "numpy.json")
    rust = baseline.read_json(reference / "rust.json")
    assert set(numpy["cases"]) == {case["id"] for case in baseline.cases()}
    assert not baseline.compare(numpy, rust)
    inventory_record = baseline.read_json(reference / "inventory.json")
    assert inventory.digest_manifest(inventory_record["source_files"]) == inventory_record["source_sha256"]
    for report_data, native in ((numpy, False), (rust, True)):
        assert report_data["cases_sha256"] == baseline.fingerprint(baseline.CASES_PATH)
        assert report_data["fixture_sha256"] == baseline.fingerprint(baseline.FIXTURE)
        for case in report_data["cases"].values():
            assert len(case["samples"]) == 3
            for sample in case["samples"]:
                # ``is_rust`` is a RETIRED key, kept only because these two files are frozen
                # records of the two routes that existed when they were captured (numpy.json
                # is the removed NumPy route, rust.json the Rust one). ``baseline.py`` no
                # longer writes it; do not edit a frozen reference to drop it.
                assert set(sample["native_backend"]) == LIVE_BACKEND_KEYS | {"is_rust"}
                assert sample["native_backend"]["is_rust"] is native
                assert sample["environment"]["source_sha256"] == inventory_record["source_sha256"]
    comparison = baseline.read_json(reference / "comparison.json")
    assert comparison["passed"] and not comparison["differences"]
    for name, digest in comparison["reports_sha256"].items():
        assert baseline.fingerprint(reference / name) == digest


def test_run_records_worker_timeout_as_failure(tmp_path, monkeypatch):
    import argparse
    import subprocess
    monkeypatch.setattr(baseline, "cases", lambda: [
        {"id": "timeout", "tier": "quick", "atol": 1e-7, "rtol": 1e-9}])
    class Process:
        pid = 12345

        def __init__(self, *args, **kwargs):
            self.first_wait = True

        def wait(self, timeout=None):
            if self.first_wait:
                self.first_wait = False
                raise subprocess.TimeoutExpired("worker", timeout)
            return -9

    killed = []
    monkeypatch.setattr(baseline.subprocess, "Popen", Process)
    monkeypatch.setattr(baseline.os, "killpg", lambda pid, sig: killed.append(pid))
    args = argparse.Namespace(case=None, tier="quick", backend="numpy", threads=1,
                              repeat=1, timeout=1, output=tmp_path / "run.json")
    assert baseline.run(args) == 1
    assert baseline.read_json(args.output)["cases"]["timeout"]["samples"][0]["status"] == "timeout"
    assert killed == [12345]


def test_junit_summary_preserves_failures_errors_and_skips(tmp_path):
    junit = tmp_path / "suite.xml"
    junit.write_text('<testsuites><testsuite><testcase name="pass" time="1"/>'
                     '<testcase name="fail"><failure message="energy differs"/></testcase>'
                     '<testcase name="error"><error message="missing dataset"/></testcase>'
                     '<testcase name="skip"><skipped message="optional"/></testcase>'
                     '</testsuite></testsuites>')
    summary = suite.summarize(junit)
    assert summary["executed"] == 4 and summary["passed"] == 1
    assert summary["failures"][0]["message"] == "energy differs"
    assert summary["errors"][0]["message"] == "missing dataset"
    assert summary["skips"][0]["message"] == "optional"
    assert suite.summarize(tmp_path / "absent.xml")["missing_junit"]
    junit.write_text("<incomplete")
    assert suite.summarize(junit)["invalid_junit"]


@pytest.mark.parametrize("timed_out", [False, True])
def test_suite_capture_distinguishes_known_failures_from_an_incomplete_run(tmp_path, monkeypatch, timed_out):
    import argparse
    import subprocess
    from pathlib import Path

    class Process:
        pid = 12345

        def __init__(self, command, **kwargs):
            self.waits = 0
            path = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml=")))
            path.write_text('<testsuites><testsuite><testcase name="existing_failure">'
                            '<failure message="reference failure"/></testcase></testsuite></testsuites>')

        def wait(self, timeout=None):
            self.waits += 1
            if timed_out and self.waits == 1:
                raise subprocess.TimeoutExpired("pytest", timeout)
            return 2 if timed_out else 1

    monkeypatch.setattr(suite.subprocess, "Popen", Process)
    monkeypatch.setattr(suite.os, "killpg", lambda *args: None)
    monkeypatch.setattr(suite, "environment", lambda: {})
    args = argparse.Namespace(output=tmp_path / "suite.json", backend="numpy", threads=2,
                              timeout=1, markers="not slow", path=[])
    assert suite.run(args) == (124 if timed_out else 1)
    captured = baseline.read_json(args.output)
    assert captured["complete"] is not timed_out
    assert captured["results"]["failures"][0]["message"] == "reference failure"
    assert all(value == "2" for value in captured["environment"]["thread_environment"].values())
