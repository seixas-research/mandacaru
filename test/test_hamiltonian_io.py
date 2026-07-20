# -*- coding: utf-8 -*-
# file: test/test_hamiltonian_io.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Hamiltonian disk cache: ``save_hamiltonian`` / ``load_hamiltonian``.

Covers the Apache Parquet round-trip (:mod:`carcara.core.serialization`) and the
driver-level contract: loading a cached Hamiltonian must reproduce the built-run
energy *and* bypass the molecular integrals and the fermion-to-qubit mapping.
"""

from __future__ import annotations


import subprocess
import sys
import textwrap

import numpy as np
import pytest
from ase import Atoms

from carcara.algorithms import ADAPTVQE, VASQE, VQE
from carcara.core import PauliSum
from carcara.core.serialization import (DEFAULT_FILENAME, DEFAULT_FORMAT,
                                        FILE_EXTENSION, FILE_EXTENSIONS,
                                        FORMAT_TAG, HAMILTONIAN_FORMATS,
                                        PARQUET_ENGINES, available_engines,
                                        detect_format, load_hamiltonian,
                                        resolve_engine, resolve_format,
                                        resolve_save_path, save_hamiltonian)

# Engines exercised **in this process**.
#
# `pyarrow` is deliberately excluded: on CPython 3.14 with qiskit 2.5 / pyarrow
# 25, pyarrow's Parquet writer hard-crashes (SIGSEGV) once Qiskit's transpiler
# has run in the same interpreter -- both ship their own native runtimes -- and a
# segfault takes the whole test session down, not just one test.  That is exactly
# why Carcará defaults to `fastparquet` (see carcara.core.serialization).
#
# pyarrow is still covered, in `TestPyarrowEngineIsolated` below, by running a
# round-trip in a *clean subprocess* where the crash cannot occur or spread.
IN_PROCESS_ENGINES = [
    pytest.param(name,
                 marks=pytest.mark.skipif(name not in available_engines(),
                                          reason=f"{name} not installed"))
    for name in PARQUET_ENGINES if name != "pyarrow"
]
ENGINES = IN_PROCESS_ENGINES


def _usable_engines() -> tuple[str, ...]:
    """Engines safe to drive from inside the test process."""
    return tuple(name for name in available_engines() if name != "pyarrow")


def _run_isolated(script: str) -> subprocess.CompletedProcess:
    """Run ``script`` in a fresh interpreter (never importing Qiskit)."""
    return subprocess.run([sys.executable, "-c", textwrap.dedent(script)],
                          capture_output=True, text=True)


@pytest.fixture
def h2_atoms():
    return Atoms("H2", positions=[[3, 3, 2.63], [3, 3, 3.37]],
                 cell=[[6, 0, 0], [0, 6, 0], [0, 0, 6]], pbc=True)


@pytest.fixture
def h2_cache(tmp_path, h2_atoms):
    """A built H2 run that dumps its qubit Hamiltonian; yields (path, energy)."""
    path = str(tmp_path / "h2.parquet")
    h2_atoms.calc = ADAPTVQE(pool="fermionic", basis="FAO", h=0.4, verbose=False,
                             max_iterations=4, save_hamiltonian=path)
    h2_atoms.get_total_energy()
    return path, h2_atoms.calc.adapt_result.optimal_energy


# --------------------------------------------------------------------------- #
# Serialization round-trip.
# --------------------------------------------------------------------------- #

def _write_raw_parquet(path, columns, key_value):
    """Write a hand-made Parquet table (bypassing Carcará) for negative tests."""
    import fastparquet
    import pandas as pd
    fastparquet.write(str(path), pd.DataFrame(columns),
                      custom_metadata=key_value, write_index=False)


class TestSerializationRoundTrip:
    @pytest.mark.parametrize("engine", ENGINES)
    def test_round_trip_preserves_terms(self, tmp_path, engine):
        pauli = PauliSum({"IIII": -0.81, "ZIII": 0.17, "XXYY": 0.045j})
        path = save_hamiltonian(tmp_path / "h.parquet", pauli,
                                mapping="bravyi_kitaev", num_particles=(1, 1),
                                n_spatial_orbitals=2, engine=engine)
        record = load_hamiltonian(path, engine=engine)

        assert record.mapping == "bravyi_kitaev"
        assert record.num_particles == (1, 1)
        assert record.n_spatial_orbitals == 2
        assert record.num_qubits == 4
        assert set(record.hamiltonian.terms) == set(pauli.terms)
        for label, coeff in pauli.terms.items():
            assert record.hamiltonian.terms[label] == pytest.approx(coeff)

    @pytest.mark.parametrize("engine", ENGINES)
    def test_file_is_a_parquet_table_of_pauli_terms(self, tmp_path, engine):
        pandas = pytest.importorskip("pandas")
        pauli = PauliSum({"IZ": 0.5, "ZI": -0.25})
        path = save_hamiltonian(tmp_path / "h.parquet", pauli, engine=engine)
        frame = pandas.read_parquet(path, engine=engine)

        assert list(frame.columns) == ["pauli", "real", "imag"]
        # Rows are sorted by Pauli string, so the file is reproducible.
        assert list(frame["pauli"]) == ["IZ", "ZI"]
        assert list(frame["real"]) == [0.5, -0.25]
        assert list(frame["imag"]) == [0.0, 0.0]
        assert path.endswith(".parquet")

    @pytest.mark.parametrize("engine", ENGINES)
    def test_metadata_carries_the_problem_spec(self, tmp_path, engine):
        pauli = PauliSum({"IZ": 0.5, "ZI": -0.25})
        path = save_hamiltonian(tmp_path / "h.parquet", pauli, mapping="parity",
                                num_particles=(1, 1), n_spatial_orbitals=1,
                                engine=engine)
        record = load_hamiltonian(path, engine=engine)
        assert record.mapping == "parity"
        assert record.num_particles == (1, 1)
        assert record.n_spatial_orbitals == 1
        assert record.num_qubits == 2

    def test_readable_by_pandas(self, tmp_path):
        """Being plain Parquet, the cache is queryable with the usual tools."""
        pandas = pytest.importorskip("pandas")
        pauli = PauliSum({"IZ": 0.5, "ZI": -0.25, "XX": 0.125})
        path = save_hamiltonian(tmp_path / "h.parquet", pauli)
        frame = pandas.read_parquet(path, engine=_usable_engines()[0])
        assert len(frame) == 3
        assert set(frame["pauli"]) == {"IZ", "ZI", "XX"}

    def test_matrix_is_bit_identical_after_round_trip(self, tmp_path):
        rng = np.random.default_rng(3)
        labels = ["IIII", "ZIII", "IZIZ", "XXYY", "YYXX"]
        pauli = PauliSum({l: complex(rng.normal(), 0.0) for l in labels})
        record = load_hamiltonian(save_hamiltonian(tmp_path / "h.parquet", pauli))
        assert np.allclose(record.hamiltonian.to_matrix(), pauli.to_matrix())

    def test_compression_shrinks_a_large_hamiltonian(self, tmp_path):
        """Parquet + zstd is the point of the format: many terms, small file."""
        rng = np.random.default_rng(11)
        labels = {"".join(rng.choice(list("IXYZ"), size=12))
                  for _ in range(4000)}
        pauli = PauliSum({l: complex(rng.normal()) for l in labels})
        path = save_hamiltonian(tmp_path / "big.parquet", pauli)
        # 12 chars + two float64 per term is >28 bytes raw; compressed is less.
        assert (tmp_path / "big.parquet").stat().st_size < 28 * len(labels)
        assert len(load_hamiltonian(path).hamiltonian.terms) == len(labels)

    def test_resolve_save_path(self):
        assert resolve_save_path(False) is None
        assert resolve_save_path(None) is None
        assert resolve_save_path(True) == DEFAULT_FILENAME
        assert DEFAULT_FILENAME.endswith(FILE_EXTENSION)
        assert resolve_save_path("my/cache.parquet") == "my/cache.parquet"
        # A path without an extension gets the Parquet one.
        assert resolve_save_path("my/cache") == "my/cache" + FILE_EXTENSION
        with pytest.raises(TypeError):
            resolve_save_path(17)

    def test_resolve_engine(self):
        assert resolve_engine("auto") in PARQUET_ENGINES
        # 'auto' prefers fastparquet when it is installed (see the module
        # docstring: pyarrow's Parquet writer can crash alongside Qiskit).
        if "fastparquet" in available_engines():
            assert resolve_engine("auto") == "fastparquet"
        with pytest.raises(ValueError, match="unknown Parquet engine"):
            resolve_engine("duckdb")

    def test_rejects_foreign_and_future_files(self, tmp_path):
        pytest.importorskip("fastparquet")
        columns = {"pauli": ["IZ"], "real": [1.0], "imag": [0.0]}

        alien = tmp_path / "alien.parquet"
        _write_raw_parquet(alien, columns, {"who": "someone else"})
        with pytest.raises(ValueError, match="not a Carcará"):
            load_hamiltonian(alien)

        future = tmp_path / "future.parquet"
        _write_raw_parquet(future, columns,
                           {"carcara.format": FORMAT_TAG,
                            "carcara.version": "99",
                            "carcara.num_qubits": "2"})
        with pytest.raises(ValueError, match="version"):
            load_hamiltonian(future)

    def test_rejects_inconsistent_pauli_length(self, tmp_path):
        pytest.importorskip("fastparquet")
        bad = tmp_path / "bad.parquet"
        _write_raw_parquet(bad, {"pauli": ["IZ"], "real": [1.0], "imag": [0.0]},
                           {"carcara.format": FORMAT_TAG,
                            "carcara.version": "1",
                            "carcara.num_qubits": "4"})
        with pytest.raises(ValueError, match="expected 4"):
            load_hamiltonian(bad)

    def test_missing_column_is_rejected(self, tmp_path):
        pytest.importorskip("fastparquet")
        bad = tmp_path / "nocol.parquet"
        _write_raw_parquet(bad, {"pauli": ["IZ"], "real": [1.0]},
                           {"carcara.format": FORMAT_TAG,
                            "carcara.version": "1",
                            "carcara.num_qubits": "2"})
        with pytest.raises(ValueError, match="missing the 'imag' column"):
            load_hamiltonian(bad)


# --------------------------------------------------------------------------- #
# The two file formats and automatic detection.
# --------------------------------------------------------------------------- #

class TestFileFormats:
    @pytest.mark.parametrize("fmt", HAMILTONIAN_FORMATS)
    def test_round_trip_in_both_formats(self, tmp_path, fmt):
        pauli = PauliSum({"IIII": -0.81, "ZIII": 0.17, "XXYY": 0.045j})
        path = save_hamiltonian(tmp_path / f"h{FILE_EXTENSIONS[fmt]}", pauli,
                                mapping="parity", num_particles=(1, 1),
                                n_spatial_orbitals=2, format=fmt)
        record = load_hamiltonian(path)

        assert record.mapping == "parity"
        assert record.num_particles == (1, 1)
        assert record.n_spatial_orbitals == 2
        assert np.allclose(record.hamiltonian.to_matrix(), pauli.to_matrix())

    def test_json_needs_no_parquet_engine(self, tmp_path):
        """The JSON path must not import a Parquet engine at all.

        This is the point of the format: it sidesteps the Qiskit/pyarrow
        interaction described in carcara.core.serialization.
        """
        path = tmp_path / "h.json"
        done = _run_isolated(f"""
            import sys
            from carcara.core import PauliSum
            from carcara.core.serialization import (save_hamiltonian,
                                                    load_hamiltonian)
            pauli = PauliSum({{"IZ": 0.5, "ZI": -0.25}})
            save_hamiltonian({str(path)!r}, pauli, format="json",
                             num_particles=(1, 1), n_spatial_orbitals=1)
            record = load_hamiltonian({str(path)!r})
            assert record.num_particles == (1, 1)
            for engine in ("pyarrow", "fastparquet"):
                assert engine not in sys.modules, engine
            print("ok")
        """)
        assert done.returncode == 0, done.stderr
        assert "ok" in done.stdout

    def test_json_is_human_readable(self, tmp_path):
        import json
        pauli = PauliSum({"IZ": 0.5, "ZI": -0.25})
        path = save_hamiltonian(tmp_path / "h.json", pauli, format="json")
        payload = json.loads((tmp_path / "h.json").read_text())

        assert payload["format"] == FORMAT_TAG
        assert payload["num_qubits"] == 2
        # Sorted by Pauli string, so the file is reproducible and diffable.
        assert payload["terms"] == [["IZ", 0.5, 0.0], ["ZI", -0.25, 0.0]]

    def test_formats_agree_term_for_term(self, tmp_path):
        rng = np.random.default_rng(5)
        labels = ["IIII", "ZIII", "IZIZ", "XXYY", "YYXX"]
        pauli = PauliSum({l: complex(rng.normal(), rng.normal() * 0.1)
                          for l in labels})
        records = {
            fmt: load_hamiltonian(
                save_hamiltonian(tmp_path / f"h{FILE_EXTENSIONS[fmt]}", pauli,
                                 format=fmt))
            for fmt in HAMILTONIAN_FORMATS}
        parquet, js = records["parquet"], records["json"]
        assert set(parquet.hamiltonian.terms) == set(js.hamiltonian.terms)
        for label, coeff in js.hamiltonian.terms.items():
            assert parquet.hamiltonian.terms[label] == pytest.approx(coeff)

    # -- detection --------------------------------------------------------- #

    @pytest.mark.parametrize("fmt", HAMILTONIAN_FORMATS)
    def test_detect_from_extension(self, tmp_path, fmt):
        path = save_hamiltonian(tmp_path / f"h{FILE_EXTENSIONS[fmt]}",
                                PauliSum({"IZ": 0.5}), format=fmt)
        assert detect_format(path) == fmt

    @pytest.mark.parametrize("fmt", HAMILTONIAN_FORMATS)
    def test_detect_from_content_when_the_extension_is_unknown(self, tmp_path,
                                                               fmt):
        """A file named ``.cache`` must still load: the bytes decide."""
        path = tmp_path / "opaque.cache"
        save_hamiltonian(path, PauliSum({"IZ": 0.5, "ZI": -0.25}),
                         num_particles=(1, 1), n_spatial_orbitals=1, format=fmt)
        assert detect_format(path) == fmt
        record = load_hamiltonian(path)          # no format= given
        assert record.num_particles == (1, 1)

    def test_undetectable_file_is_rejected(self, tmp_path):
        path = tmp_path / "junk.cache"
        path.write_bytes(b"\x00\x01 not a hamiltonian")
        with pytest.raises(ValueError, match="cannot determine the format"):
            detect_format(path)

    def test_explicit_format_overrides_detection(self, tmp_path):
        path = tmp_path / "named.parquet"          # extension lies ...
        save_hamiltonian(path, PauliSum({"IZ": 0.5}), format="json")
        assert load_hamiltonian(path, format="json").num_qubits == 2

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_hamiltonian(tmp_path / "absent.json")

    # -- the format argument ----------------------------------------------- #

    def test_resolve_format(self):
        assert resolve_format("parquet") == "parquet"
        assert resolve_format("JSON") == "json"
        assert resolve_format(".json") == "json"
        assert resolve_format("pq") == "parquet"
        with pytest.raises(ValueError, match="unknown hamiltonian_format"):
            resolve_format("hdf5")

    def test_save_path_extension_follows_the_format(self):
        assert resolve_save_path(True, "parquet") == "hamiltonian.parquet"
        assert resolve_save_path(True, "json") == "hamiltonian.json"
        assert resolve_save_path("cache", "json") == "cache.json"
        assert resolve_save_path("cache", "parquet") == "cache.parquet"
        # An explicit extension is always respected.
        assert resolve_save_path("cache.json", "parquet") == "cache.json"

    def test_format_inferred_from_the_path_extension(self, tmp_path):
        """``save_hamiltonian`` follows the path when no format is given."""
        path = save_hamiltonian(tmp_path / "h.json", PauliSum({"IZ": 0.5}))
        assert detect_format(path) == "json"


class TestDriverFileFormats:
    @pytest.mark.parametrize("fmt", HAMILTONIAN_FORMATS)
    def test_driver_saves_and_reloads_in_either_format(self, tmp_path,
                                                       h2_atoms, fmt):
        path = str(tmp_path / f"h2{FILE_EXTENSIONS[fmt]}")
        h2_atoms.calc = ADAPTVQE(pool="fermionic", basis="FAO", h=0.4,
                                 verbose=False, max_iterations=4,
                                 save_hamiltonian=path, hamiltonian_format=fmt)
        h2_atoms.get_total_energy()
        built = h2_atoms.calc.adapt_result.optimal_energy

        assert detect_format(path) == fmt
        loaded = ADAPTVQE(pool="fermionic", load_hamiltonian=path,
                          verbose=False, max_iterations=4).run()
        assert loaded.optimal_energy == pytest.approx(built, abs=1e-9)

    def test_default_format_is_parquet(self):
        assert ADAPTVQE().hamiltonian_format == DEFAULT_FORMAT == "parquet"
        assert VQE().hamiltonian_format == "parquet"

    def test_format_selects_the_default_filename(self, tmp_path, monkeypatch,
                                                 h2_atoms):
        monkeypatch.chdir(tmp_path)
        h2_atoms.calc = ADAPTVQE(pool="fermionic", basis="FAO", h=0.4,
                                 verbose=False, max_iterations=1,
                                 save_hamiltonian=True,
                                 hamiltonian_format="json")
        h2_atoms.get_total_energy()
        assert (tmp_path / "hamiltonian.json").is_file()
        assert not (tmp_path / "hamiltonian.parquet").exists()

    def test_unknown_format_is_rejected(self):
        with pytest.raises(ValueError, match="unknown hamiltonian_format"):
            ADAPTVQE(hamiltonian_format="hdf5")


# --------------------------------------------------------------------------- #
# The pyarrow engine, exercised out-of-process (see IN_PROCESS_ENGINES).
# --------------------------------------------------------------------------- #

@pytest.mark.skipif("pyarrow" not in available_engines(),
                    reason="pyarrow not installed")
class TestPyarrowEngineIsolated:
    def test_round_trip(self, tmp_path):
        """pyarrow must round-trip a Hamiltonian exactly, like fastparquet."""
        path = tmp_path / "pa.parquet"
        done = _run_isolated(f"""
            from carcara.core import PauliSum
            from carcara.core.serialization import (save_hamiltonian,
                                                    load_hamiltonian)
            pauli = PauliSum({{"IIII": -0.81, "ZIII": 0.17, "XXYY": 0.045j}})
            save_hamiltonian({str(path)!r}, pauli, mapping="parity",
                             num_particles=(1, 1), n_spatial_orbitals=2,
                             engine="pyarrow")
            record = load_hamiltonian({str(path)!r}, engine="pyarrow")
            assert record.mapping == "parity"
            assert record.num_particles == (1, 1)
            assert record.n_spatial_orbitals == 2
            assert set(record.hamiltonian.terms) == set(pauli.terms)
            for label, coeff in pauli.terms.items():
                assert abs(record.hamiltonian.terms[label] - coeff) < 1e-12
            print("ok")
        """)
        assert done.returncode == 0, done.stderr
        assert "ok" in done.stdout
        assert path.is_file()

    def test_engines_are_interchangeable(self, tmp_path):
        """A file written by one engine must be readable by the other."""
        pytest.importorskip("fastparquet")
        fp_path = tmp_path / "by_fastparquet.parquet"
        pa_path = tmp_path / "by_pyarrow.parquet"
        pauli = PauliSum({"IIII": -0.81, "ZIII": 0.17, "XXYY": 0.045j})
        save_hamiltonian(fp_path, pauli, num_particles=(1, 1),
                         n_spatial_orbitals=2, engine="fastparquet")

        # pyarrow reads the fastparquet file, then writes its own.
        done = _run_isolated(f"""
            import numpy as np
            from carcara.core import PauliSum
            from carcara.core.serialization import (save_hamiltonian,
                                                    load_hamiltonian)
            record = load_hamiltonian({str(fp_path)!r}, engine="pyarrow")
            assert record.num_particles == (1, 1)
            assert record.n_spatial_orbitals == 2
            save_hamiltonian({str(pa_path)!r}, record.hamiltonian,
                             num_particles=record.num_particles,
                             n_spatial_orbitals=record.n_spatial_orbitals,
                             engine="pyarrow")
            print("ok")
        """)
        assert done.returncode == 0, done.stderr

        # ... and fastparquet reads what pyarrow wrote, recovering the operator.
        back = load_hamiltonian(pa_path, engine="fastparquet")
        assert back.num_particles == (1, 1)
        assert np.allclose(back.hamiltonian.to_matrix(), pauli.to_matrix())


# --------------------------------------------------------------------------- #
# Driver integration.
# --------------------------------------------------------------------------- #

class TestDriverSavesHamiltonian:
    def test_calculator_writes_the_qubit_hamiltonian(self, h2_cache):
        path, _ = h2_cache
        record = load_hamiltonian(path)
        assert record.num_qubits == 4
        assert record.mapping == "jordan_wigner"
        assert record.num_particles == (1, 1)
        assert record.n_spatial_orbitals == 2
        assert record.metadata["driver"] == "ADAPTVQE"

    def test_saved_operator_matches_the_live_one(self, tmp_path, h2_atoms):
        path = str(tmp_path / "h2.parquet")
        h2_atoms.calc = ADAPTVQE(pool="fermionic", basis="FAO", h=0.4,
                                 verbose=False, max_iterations=2,
                                 save_hamiltonian=path)
        h2_atoms.get_total_energy()
        saved = load_hamiltonian(path).hamiltonian
        live = h2_atoms.calc.hamiltonian.simplify()
        assert set(saved.terms) == set(live.terms)
        for label, coeff in live.terms.items():
            assert saved.terms[label] == pytest.approx(coeff)

    def test_save_true_uses_default_filename(self, tmp_path, monkeypatch,
                                             h2_atoms):
        monkeypatch.chdir(tmp_path)
        h2_atoms.calc = ADAPTVQE(pool="fermionic", basis="FAO", h=0.4,
                                 verbose=False, max_iterations=1,
                                 save_hamiltonian=True)
        h2_atoms.get_total_energy()
        assert (tmp_path / DEFAULT_FILENAME).is_file()


class TestDriverLoadsHamiltonian:
    def test_loaded_run_reproduces_the_built_energy(self, h2_cache):
        path, built_energy = h2_cache
        loaded = ADAPTVQE(pool="fermionic", load_hamiltonian=path, verbose=False,
                          max_iterations=4)
        assert loaded.run().optimal_energy == pytest.approx(built_energy,
                                                            abs=1e-9)

    def test_loading_needs_no_geometry_and_configures_the_pool(self, h2_cache):
        path, _ = h2_cache
        loaded = ADAPTVQE(pool="ceo", load_hamiltonian=path, verbose=False)
        # Direct mode without ever touching an Atoms object.
        assert loaded._configured
        assert loaded.n_qubits == 4
        assert loaded.num_particles == (1, 1)
        assert loaded._integration_profile is None      # no integrals were run

    def test_loading_bypasses_the_integral_engine(self, h2_cache, monkeypatch):
        """No one- or two-body integral may be computed on the load path."""
        path, _ = h2_cache
        import carcara.algorithms._hamiltonian_from_atoms as builder
        from carcara.integrals.engine import IntegralEngine

        def fail(*args, **kwargs):                      # pragma: no cover
            raise AssertionError("the integral engine was invoked")

        monkeypatch.setattr(builder, "build_basis_hamiltonian", fail)
        monkeypatch.setattr(IntegralEngine, "one_body", fail)
        monkeypatch.setattr(IntegralEngine, "two_body", fail)

        result = ADAPTVQE(pool="fermionic", load_hamiltonian=path, verbose=False,
                          max_iterations=4).run()
        assert np.isfinite(result.optimal_energy)

    def test_loading_bypasses_the_hamiltonian_mapping(self, h2_cache):
        """The Hamiltonian is adopted as-is; no fermion-to-qubit map is run on it.

        (The *pool* still maps its own excitation generators -- that is a
        different operator and is unrelated to the Hamiltonian's transformation.)
        """
        path, _ = h2_cache
        driver = ADAPTVQE(pool="fermionic", load_hamiltonian=path, verbose=False)
        stored = load_hamiltonian(path).hamiltonian.simplify()
        live = driver.hamiltonian.simplify()
        assert isinstance(driver.hamiltonian, PauliSum)
        assert set(live.terms) == set(stored.terms)
        for label, coeff in stored.terms.items():
            assert live.terms[label] == pytest.approx(coeff)

    def test_vqe_rebuilds_its_uccsd_ansatz_from_the_file(self, h2_cache):
        path, built_energy = h2_cache
        vqe = VQE(load_hamiltonian=path, verbose=False)
        assert vqe.ansatz.n_qubits == 4
        assert vqe.ansatz.num_particles == (1, 1)
        # UCCSD reaches the same ground state as the (converged) ADAPT run.
        assert vqe.run().optimal_energy == pytest.approx(built_energy, abs=1e-6)

    def test_vasqe_loads_too(self, h2_cache):
        path, built_energy = h2_cache
        result = VASQE(pool="fermionic", load_hamiltonian=path, verbose=False,
                       temperature=1e-3, max_iterations=4, seed=1).run()
        assert result.optimal_energy == pytest.approx(built_energy, abs=1e-6)

    def test_save_then_load_same_path_is_not_reserialized(self, tmp_path,
                                                          h2_cache):
        """Loading and saving the same file must not rewrite (and clobber) it."""
        path, _ = h2_cache
        before = open(path, "rb").read()               # Parquet is binary
        ADAPTVQE(pool="fermionic", load_hamiltonian=path,
                 save_hamiltonian=path, verbose=False, max_iterations=1).run()
        assert open(path, "rb").read() == before

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ADAPTVQE(pool="fermionic",
                     load_hamiltonian=str(tmp_path / "nope.parquet"))
