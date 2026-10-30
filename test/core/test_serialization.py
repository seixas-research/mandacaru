# -*- coding: utf-8 -*-
# file: test/test_hamiltonian_io.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Hamiltonian disk cache: ``save_hamiltonian`` / ``load_hamiltonian``.

Covers the Apache Parquet round-trip (:mod:`mandacaru.core.serialization`) and the
driver-level contract: loading a cached Hamiltonian must reproduce the built-run
energy *and* bypass the molecular integrals and the fermion-to-qubit mapping.
"""

from __future__ import annotations


import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
from mandacaru.units import HARTREE_TO_EV
import pytest
from ase import Atoms

from mandacaru import Mandacaru, _native
from mandacaru.core import PauliSum
from mandacaru.core.serialization import (DEFAULT_FILENAME, DEFAULT_FORMAT,
                                          FILE_EXTENSIONS, FORMAT_TAG,
                                          HAMILTONIAN_FORMATS, detect_format,
                                          load_hamiltonian, resolve_format,
                                          resolve_save_path, save_hamiltonian)

#: Frozen foreign-writer files (fastparquet, pyarrow) and hand-made negatives.
PARQUET_REFERENCE = Path(__file__).parents[1] / "migration/reference/parquet"


def _run_isolated(script: str) -> subprocess.CompletedProcess:
    """Run ``script`` in a fresh, isolated interpreter."""
    return subprocess.run([sys.executable, "-I", "-c", textwrap.dedent(script)],
                          capture_output=True, text=True)


@pytest.fixture
def h2_atoms():
    return Atoms("H2", positions=[[3, 3, 2.63], [3, 3, 3.37]],
                 cell=[[6, 0, 0], [0, 6, 0], [0, 0, 6]], pbc=False)


@pytest.fixture
def h2_cache(tmp_path, h2_atoms):
    """A built H2 run that dumps its qubit Hamiltonian; yields (path, energy)."""
    path = str(tmp_path / "h2.parquet")
    h2_atoms.calc = Mandacaru(method="adapt-vqe", pool="fermionic",
                              basis="HAO", grid_spacing=0.4, trace=False,
                              max_iterations=4, save_hamiltonian=path)
    h2_atoms.get_total_energy()
    return path, h2_atoms.calc.result.optimal_energy


# --------------------------------------------------------------------------- #
# Serialization round-trip.
# --------------------------------------------------------------------------- #

class TestSerializationRoundTrip:
    def test_round_trip_preserves_terms(self, tmp_path):
        pauli = PauliSum({"IIII": -0.81, "ZIII": 0.17, "XXYY": 0.045j})
        path = save_hamiltonian(tmp_path / "h.parquet", pauli,
                                mapping="bravyi_kitaev", num_particles=(1, 1),
                                n_spatial_orbitals=2)
        record = load_hamiltonian(path)

        assert record.mapping == "bravyi_kitaev"
        assert record.num_particles == (1, 1)
        assert record.n_spatial_orbitals == 2
        assert record.num_qubits == 4
        assert set(record.hamiltonian.terms) == set(pauli.terms)
        for label, coeff in pauli.terms.items():
            assert record.hamiltonian.terms[label] == pytest.approx(coeff)

    def test_file_is_a_parquet_table_of_pauli_terms(self, tmp_path):
        pauli = PauliSum({"IZ": 0.5, "ZI": -0.25})
        path = save_hamiltonian(tmp_path / "h.parquet", pauli)
        table = _native.read_parquet(_native.Context(), path)

        assert list(table["columns"]) == ["pauli", "real", "imag"]
        # Rows are sorted by Pauli string, so the file is reproducible.
        assert table["columns"]["pauli"] == ["IZ", "ZI"]
        assert table["columns"]["real"].tolist() == [0.5, -0.25]
        assert table["columns"]["imag"].tolist() == [0.0, 0.0]
        assert path.endswith(".parquet")
        with open(path, "rb") as handle:
            data = handle.read()
        assert data[:4] == data[-4:] == b"PAR1"

    def test_metadata_carries_the_problem_spec(self, tmp_path):
        pauli = PauliSum({"IZ": 0.5, "ZI": -0.25})
        path = save_hamiltonian(tmp_path / "h.parquet", pauli, mapping="parity",
                                num_particles=(1, 1), n_spatial_orbitals=1)
        record = load_hamiltonian(path)
        assert record.mapping == "parity"
        assert record.num_particles == (1, 1)
        assert record.n_spatial_orbitals == 1
        assert record.num_qubits == 2
        footer = _native.read_parquet_metadata(_native.Context(), path)
        assert footer["metadata"]["mandacaru.format"] == FORMAT_TAG
        assert footer["metadata"]["mandacaru.num_qubits"] == "2"

    @pytest.mark.parametrize("compression", ["zstd", "uncompressed", "none",
                                             "ZSTD"])
    def test_compression_names(self, tmp_path, compression):
        pauli = PauliSum({"IZ": 0.5, "ZI": -0.25})
        path = save_hamiltonian(tmp_path / "h.parquet", pauli,
                                compression=compression)
        assert load_hamiltonian(path).hamiltonian.terms == pauli.terms

    def test_an_unknown_compression_raises_and_writes_nothing(self, tmp_path):
        with pytest.raises(ValueError, match="zstd"):
            save_hamiltonian(tmp_path / "h.parquet", PauliSum({"IZ": 0.5}),
                             compression="gzip")
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize("keyword", ["engine", "codec"])
    def test_the_retired_selectors_are_gone(self, tmp_path, keyword):
        with pytest.raises(TypeError):
            save_hamiltonian(tmp_path / "h.parquet", PauliSum({"IZ": 0.5}),
                             **{keyword: "fastparquet"})
        with pytest.raises(TypeError):
            load_hamiltonian(tmp_path / "h.parquet", **{keyword: "fastparquet"})

    def test_writing_twice_gives_the_same_bytes(self, tmp_path):
        pauli = PauliSum({"IZ": 0.5, "ZI": -0.25, "XX": 0.125j})
        first = save_hamiltonian(tmp_path / "a.parquet", pauli)
        second = save_hamiltonian(tmp_path / "b.parquet", pauli)
        assert Path(first).read_bytes() == Path(second).read_bytes()

    def test_matrix_is_bit_identical_after_round_trip(self, tmp_path):
        rng = np.random.default_rng(3)
        labels = ["IIII", "ZIII", "IZIZ", "XXYY", "YYXX"]
        pauli = PauliSum({l: complex(rng.normal(), 0.0) for l in labels})
        record = load_hamiltonian(save_hamiltonian(tmp_path / "h.parquet", pauli))
        assert np.allclose(record.hamiltonian.to_matrix(), pauli.to_matrix())

    @pytest.mark.parametrize("name", ["h.parquet", "h.json"])
    def test_every_coefficient_survives_bit_for_bit(self, tmp_path, name):
        """The coefficients go through a JSON text on the way to the file;
        a parser that is not round-trip exact moved about one value in six
        by one unit in the last place."""
        rng = np.random.default_rng(20261030)
        labels = sorted({"".join(rng.choice(list("IXYZ"), size=9))
                         for _ in range(4000)})
        scale = 10.0 ** rng.integers(-3, 4, size=len(labels))
        real = rng.normal(size=len(labels)) * scale
        imag = rng.normal(size=len(labels)) * 3.0
        pauli = PauliSum({l: complex(r, i)
                          for l, r, i in zip(labels, real, imag)})
        record = load_hamiltonian(save_hamiltonian(tmp_path / name, pauli))
        terms = record.hamiltonian.terms
        assert sorted(terms) == labels
        got = np.array([terms[l] for l in labels])
        assert got.real.view(np.uint64).tolist() == real.view(np.uint64).tolist()
        assert got.imag.view(np.uint64).tolist() == imag.view(np.uint64).tolist()

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
        assert DEFAULT_FILENAME.endswith(FILE_EXTENSIONS[DEFAULT_FORMAT])
        assert resolve_save_path("my/cache.parquet") == "my/cache.parquet"
        # A path without an extension gets the Parquet one.
        assert resolve_save_path("my/cache") == "my/cache" + FILE_EXTENSIONS["parquet"]
        with pytest.raises(TypeError):
            resolve_save_path(17)

    def test_rejects_foreign_and_future_files(self):
        alien = PARQUET_REFERENCE / "negative/hamiltonian_kv_wrong_format.parquet"
        with pytest.raises(ValueError, match="not a Mandacaru"):
            load_hamiltonian(alien)

        for name in ("hamiltonian_kv_version_2", "hamiltonian_kv_version_0"):
            future = PARQUET_REFERENCE / f"negative/{name}.parquet"
            with pytest.raises(ValueError, match="version"):
                load_hamiltonian(future)

    def test_rejects_inconsistent_pauli_length(self):
        bad = PARQUET_REFERENCE / "negative/hamiltonian_wrong_pauli_length.parquet"
        with pytest.raises(ValueError, match="Pauli labels must match the "
                                             "register and coefficients must "
                                             "be finite"):
            load_hamiltonian(bad)

    def test_missing_column_is_rejected(self):
        bad = PARQUET_REFERENCE / "negative/hamiltonian_missing_imag.parquet"
        with pytest.raises(ValueError, match=r"is missing the 'imag' column "
                                             r"\(found \["):
            load_hamiltonian(bad)

    def test_a_register_wider_than_fifty_qubits_is_rejected(self):
        bad = PARQUET_REFERENCE / "negative/hamiltonian_qubits_51.parquet"
        with pytest.raises(ValueError, match="50"):
            load_hamiltonian(bad)

    def test_a_non_finite_coefficient_is_rejected(self):
        bad = PARQUET_REFERENCE / "negative/hamiltonian_nonfinite_inf.parquet"
        with pytest.raises(ValueError, match="finite"):
            load_hamiltonian(bad)

    def test_a_non_finite_coefficient_creates_no_file(self, tmp_path):
        pauli = PauliSum({"IZ": 1.0})
        pauli.terms["IZ"] = complex("inf")      # slipped past the constructor
        with pytest.raises(ValueError, match="finite|JSON compliant"):
            save_hamiltonian(tmp_path / "h.parquet", pauli)
        assert list(tmp_path.iterdir()) == []

    def test_more_than_fifty_qubits_creates_no_file(self, tmp_path):
        pauli = PauliSum({"I" * 51: 1.0})
        with pytest.raises(ValueError, match="50"):
            save_hamiltonian(tmp_path / "h.parquet", pauli)
        assert list(tmp_path.iterdir()) == []

    def test_a_truncated_file_raises_at_every_cut(self, tmp_path):
        path = save_hamiltonian(tmp_path / "h.parquet",
                                PauliSum({"IZ": 0.5, "ZI": -0.25}),
                                num_particles=(1, 1), n_spatial_orbitals=1)
        data = Path(path).read_bytes()
        cut = tmp_path / "cut.parquet"
        for length in range(0, len(data), 7):
            cut.write_bytes(data[:length])
            with pytest.raises(ValueError):
                load_hamiltonian(cut)

    def test_mandacaru_defaults_for_a_file_with_only_the_required_keys(self):
        record = load_hamiltonian(
            PARQUET_REFERENCE / "hamiltonian/semantic_kv_defaults_only_required.parquet")
        assert record.mapping == "jordan_wigner"
        assert record.num_particles is None
        assert record.n_spatial_orbitals is None
        assert record.metadata == {}


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

    def test_no_python_parquet_package_is_needed(self, tmp_path):
        """Parquet and JSON, a pseudopotential and a Qiskit transpile run with
        ``pandas``, ``pyarrow`` and ``fastparquet`` unimportable, and no
        Mandacaru module so much as asks for them."""
        fixture = PARQUET_REFERENCE / "pseudopotential/fastparquet_zstd.parquet"
        done = _run_isolated(f"""
            import importlib.abc, os, sys
            BLOCKED = {{"pandas", "pyarrow", "fastparquet"}}
            attempts = []
            class Block(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split(".")[0] in BLOCKED:
                        frame = sys._getframe(1)
                        while frame and frame.f_code.co_filename.startswith(
                                "<frozen importlib"):
                            frame = frame.f_back
                        attempts.append((fullname,
                                         frame.f_globals.get("__name__")))
                        raise ModuleNotFoundError(fullname, name=fullname)
            sys.meta_path.insert(0, Block())
            from mandacaru.core import PauliSum
            from mandacaru.core.serialization import (save_hamiltonian,
                                                    load_hamiltonian)
            from mandacaru.pseudopotentials.io import (load_pseudopotential,
                                                       save_pseudopotential)
            pauli = PauliSum({{"IZ": 0.5, "ZI": -0.25}})
            for extension in (".parquet", ".json"):
                path = save_hamiltonian(os.path.join({str(tmp_path)!r},
                                                     "h" + extension), pauli,
                                        num_particles=(1, 1),
                                        n_spatial_orbitals=1)
                record = load_hamiltonian(path)
                assert record.num_particles == (1, 1)
                assert record.hamiltonian.terms == pauli.terms
            loaded = load_pseudopotential({str(fixture)!r})
            again = save_pseudopotential(
                loaded, os.path.join({str(tmp_path)!r}, "pp.parquet"), stride=1)
            assert load_pseudopotential(again).symbol == loaded.symbol
            from qiskit import QuantumCircuit, transpile
            circuit = QuantumCircuit(2)
            circuit.h(0)
            circuit.cx(0, 1)
            compiled = transpile(circuit, basis_gates=["u", "cx"],
                                 optimization_level=1)
            assert compiled.count_ops().get("cx") == 1
            mandacaru = [a for a in attempts
                         if str(a[1]).split(".")[0] == "mandacaru"]
            assert not mandacaru, mandacaru
            assert not [m for m in sys.modules
                        if m.split(".")[0] in BLOCKED]
            print("ok")
        """)
        assert done.returncode == 0, done.stderr
        assert "ok" in done.stdout

    def test_json_is_human_readable(self, tmp_path):
        import json
        pauli = PauliSum({"IZ": 0.5, "ZI": -0.25})
        save_hamiltonian(tmp_path / "h.json", pauli, format="json")
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
        with pytest.warns(RuntimeWarning, match="will not match"):
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
        h2_atoms.calc = Mandacaru(method="adapt-vqe", pool="fermionic",
                                  basis="HAO", grid_spacing=0.4, trace=False,
                                  max_iterations=4, save_hamiltonian=path,
                                  hamiltonian_format=fmt)
        h2_atoms.get_total_energy()
        built = h2_atoms.calc.result.optimal_energy

        assert detect_format(path) == fmt
        loaded = Mandacaru(method="adapt-vqe", pool="fermionic",
                           load_hamiltonian=path, trace=False,
                           max_iterations=4).run()
        assert loaded.optimal_energy == pytest.approx(built, abs=1e-9 * HARTREE_TO_EV)

    def test_default_format_is_parquet(self, tmp_path):
        """A loaded Hamiltonian saved to a bare name is written as Parquet."""
        assert DEFAULT_FORMAT == "parquet"
        source = save_hamiltonian(tmp_path / "source.parquet",
                                  PauliSum({"IIII": -0.5, "ZIII": 0.25}),
                                  mapping="jordan_wigner", num_particles=(1, 1),
                                  n_spatial_orbitals=2)
        bare = str(tmp_path / "bare")
        Mandacaru(method="adapt-vqe", pool="fermionic", load_hamiltonian=source,
                  save_hamiltonian=bare, trace=False, max_iterations=1).run()
        assert detect_format(bare) == "parquet"

    def test_format_selects_the_written_file(self, tmp_path, h2_atoms):
        """``hamiltonian_format="json"`` writes JSON, not Parquet, from a geometry."""
        path = str(tmp_path / "hamiltonian.json")
        h2_atoms.calc = Mandacaru(method="adapt-vqe", pool="fermionic",
                                  basis="HAO", grid_spacing=0.4, trace=False,
                                  max_iterations=1, save_hamiltonian=path,
                                  hamiltonian_format="json")
        h2_atoms.get_total_energy()
        assert open(path, "rb").read(1) == b"{"
        assert not (tmp_path / "hamiltonian.parquet").exists()

    def test_a_geometry_run_flags_a_name_that_contradicts_the_format(
            self, tmp_path, h2_atoms):
        """The geometry route warns like ``save_hamiltonian`` does."""
        path = str(tmp_path / "wrong.json")
        h2_atoms.calc = Mandacaru(method="adapt-vqe", pool="fermionic",
                                  basis="HAO", grid_spacing=0.4, trace=False,
                                  max_iterations=1, save_hamiltonian=path,
                                  hamiltonian_format="parquet")
        with pytest.warns(RuntimeWarning, match="will not match"):
            h2_atoms.get_total_energy()
        assert open(path, "rb").read(4) == b"PAR1"

    def test_unknown_format_is_rejected(self):
        with pytest.raises(ValueError, match="unknown hamiltonian_format"):
            Mandacaru(method="adapt-vqe", hamiltonian_format="hdf5")

    def test_the_path_extension_outranks_the_driver_default(self, tmp_path,
                                                            h2_atoms):
        """A file named '.json' must not be handed Parquet bytes."""
        path = str(tmp_path / "named.json")
        h2_atoms.calc = Mandacaru(method="adapt-vqe", pool="fermionic",
                                  basis="HAO", grid_spacing=0.4, trace=False,
                                  max_iterations=1, save_hamiltonian=path)   # default: parquet
        h2_atoms.get_total_energy()
        assert open(path, "rb").read(1) == b"{"
        assert detect_format(path) == "json"

    def test_a_name_that_contradicts_the_content_is_flagged(self, tmp_path):
        """An explicit format still wins, but the mismatch is not silent."""
        with pytest.warns(RuntimeWarning, match="will not match"):
            path = save_hamiltonian(tmp_path / "wrong.json",
                                    PauliSum({"ZZ": 1.0}), format="parquet")
        with pytest.warns(RuntimeWarning, match="content is 'parquet'"):
            assert load_hamiltonian(path).hamiltonian.terms["ZZ"] == 1.0

    def test_a_mislabeled_file_is_read_as_what_it_is(self, tmp_path):
        """Written by an older build; the bytes decide, with a warning."""
        real = save_hamiltonian(tmp_path / "real.parquet",
                                PauliSum({"ZZ": 1.0, "XX": 0.5}))
        lying = tmp_path / "lying.json"
        lying.write_bytes(open(real, "rb").read())
        with pytest.warns(RuntimeWarning, match="content is 'parquet'"):
            assert detect_format(lying) == "parquet"
        assert load_hamiltonian(lying).hamiltonian.terms["XX"] == 0.5


class TestTaperedRecords:
    """A tapered Hamiltonian is 2M - 2 qubits wide; the file has to say so."""

    def _tapered(self, tmp_path, fmt, atoms):
        path = str(tmp_path / f"tapered{FILE_EXTENSIONS[fmt]}")
        atoms.calc = Mandacaru(method="adapt-vqe", pool="fermionic",
                               basis="HAO", grid_spacing=0.4,
                               mapping="parity_reduced", trace=False,
                               profile=False, max_iterations=4,
                               save_hamiltonian=path)
        return path, atoms.get_total_energy()

    @pytest.mark.parametrize("fmt", HAMILTONIAN_FORMATS)
    def test_the_reduction_round_trips(self, tmp_path, h2_atoms, fmt):
        path, energy = self._tapered(tmp_path, fmt, h2_atoms)
        record = load_hamiltonian(path)
        assert record.mapping == "parity_reduced"
        assert record.num_qubits == 2

        # A driver told nothing but the file reproduces the run exactly.
        loaded = Mandacaru(method="adapt-vqe", pool="fermionic",
                           load_hamiltonian=path, trace=False, profile=False,
                           max_iterations=4)
        result = loaded.run()
        assert loaded.problem.mapping == "parity_reduced" and loaded.n_qubits == 2
        assert result.optimal_energy == pytest.approx(energy, abs=1e-9)

    def test_an_untapered_file_will_not_pretend(self, tmp_path, h2_atoms):
        path = str(tmp_path / "plain.json")
        h2_atoms.calc = Mandacaru(method="adapt-vqe", pool="fermionic",
                                  basis="HAO", grid_spacing=0.4, trace=False,
                                  profile=False, max_iterations=1,
                                  save_hamiltonian=path)
        h2_atoms.get_total_energy()
        assert load_hamiltonian(path).mapping != "parity_reduced"
        with pytest.raises(ValueError, match="mapping disagrees"):
            Mandacaru(method="adapt-vqe", pool="fermionic",
                      load_hamiltonian=path, mapping="parity_reduced",
                      trace=False).run()

    def test_mapping_is_the_only_reduction_setting(self, tmp_path):
        import json
        path = save_hamiltonian(tmp_path / "mapping.json",
                                PauliSum({"II": 1.0}),
                                mapping="parity_reduced",
                                num_particles=(1, 1), n_spatial_orbitals=2)
        payload = json.loads(open(path).read())
        assert payload["mapping"] == "parity_reduced"
        assert "two_qubit_reduction" not in payload
        assert load_hamiltonian(path).mapping == "parity_reduced"


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
        assert record.metadata["source_engine"] == "rust"

    def test_saved_operator_matches_the_live_one(self, tmp_path, h2_atoms):
        path = str(tmp_path / "h2.parquet")
        h2_atoms.calc = Mandacaru(method="adapt-vqe", pool="fermionic",
                                  basis="HAO", grid_spacing=0.4, trace=False,
                                  max_iterations=2, save_hamiltonian=path)
        h2_atoms.get_total_energy()
        saved = load_hamiltonian(path).hamiltonian
        live = {label: complex(re, im) for label, re, im
                in h2_atoms.calc.problem.hamiltonian_terms}
        assert set(saved.terms) == set(live)
        for label, coeff in live.items():
            assert saved.terms[label] == pytest.approx(coeff)

    def test_the_default_filename_is_parquet(self, tmp_path, monkeypatch,
                                             h2_atoms):
        monkeypatch.chdir(tmp_path)
        h2_atoms.calc = Mandacaru(method="adapt-vqe", pool="fermionic",
                                  basis="HAO", grid_spacing=0.4, trace=False,
                                  max_iterations=1,
                                  save_hamiltonian=DEFAULT_FILENAME)
        h2_atoms.get_total_energy()
        assert (tmp_path / DEFAULT_FILENAME).is_file()
        assert detect_format(tmp_path / DEFAULT_FILENAME) == "parquet"


class TestDriverLoadsHamiltonian:
    def test_loaded_run_reproduces_the_built_energy(self, h2_cache):
        path, built_energy = h2_cache
        loaded = Mandacaru(method="adapt-vqe", pool="fermionic",
                           load_hamiltonian=path, trace=False,
                           max_iterations=4)
        assert loaded.run().optimal_energy == pytest.approx(
            built_energy, abs=1e-9 * HARTREE_TO_EV)

    def test_loading_needs_no_geometry_and_configures_the_pool(self, h2_cache):
        path, _ = h2_cache
        loaded = Mandacaru(method="adapt-vqe", pool="ceo",
                           load_hamiltonian=path, trace=False, max_iterations=1)
        # Direct mode without ever touching an Atoms object.
        loaded.run()
        assert loaded.n_qubits == 4
        assert loaded.num_particles == (1, 1)
        assert loaded.native_reference is None                # no SCF was run

    def test_loading_bypasses_the_integral_engine(self, h2_cache, monkeypatch):
        """No one- or two-body integral may be computed on the load path."""
        path, _ = h2_cache
        def fail(*args, **kwargs):                      # pragma: no cover
            raise AssertionError("the mean-field engine was invoked")

        monkeypatch.setattr(Mandacaru, "_mean_field", fail)

        result = Mandacaru(method="adapt-vqe", pool="fermionic",
                           load_hamiltonian=path, trace=False,
                           max_iterations=4).run()
        assert np.isfinite(result.optimal_energy)

    def test_loading_bypasses_the_hamiltonian_mapping(self, h2_cache):
        """The Hamiltonian is adopted as-is; no fermion-to-qubit map is run on it.

        (The *pool* still maps its own excitation generators -- that is a
        different operator and is unrelated to the Hamiltonian's transformation.)
        """
        path, _ = h2_cache
        driver = Mandacaru(method="adapt-vqe", pool="fermionic",
                           load_hamiltonian=path, trace=False)
        driver.run()
        stored = load_hamiltonian(path).hamiltonian.simplify()
        live = {label: complex(re, im) for label, re, im
                in driver.problem.hamiltonian_terms}
        assert set(live) == set(stored.terms)
        for label, coeff in stored.terms.items():
            assert live[label] == pytest.approx(coeff)

    def test_vqe_rebuilds_its_uccsd_ansatz_from_the_file(self, h2_cache):
        path, built_energy = h2_cache
        vqe = Mandacaru(method="vqe", load_hamiltonian=path, trace=False)
        # UCCSD reaches the same ground state as the (converged) ADAPT run.
        assert vqe.run().optimal_energy == pytest.approx(
            built_energy, abs=1e-6 * HARTREE_TO_EV)
        assert vqe.n_qubits == 4
        assert vqe.num_particles == (1, 1)

    def test_save_then_load_same_path_is_not_reserialized(self, tmp_path,
                                                          h2_cache):
        """Loading and saving the same file must not rewrite (and clobber) it."""
        path, _ = h2_cache
        before = open(path, "rb").read()               # Parquet is binary
        Mandacaru(method="adapt-vqe", pool="fermionic", load_hamiltonian=path,
                  save_hamiltonian=path, trace=False, max_iterations=1).run()
        assert open(path, "rb").read() == before

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Mandacaru(method="adapt-vqe", pool="fermionic",
                      load_hamiltonian=str(tmp_path / "nope.parquet")).run()


class TestDriverWithAFrozenCache:
    """The load/save contract on a frozen LiH cache, which needs no geometry."""

    LIH = PARQUET_REFERENCE / "hamiltonian/fastparquet_zstd_lih_cache.parquet"

    def _own_copy(self, tmp_path):
        """The frozen file re-saved by this build (a fixture written by the
        replaced stack has other bytes, though the same content)."""
        record = load_hamiltonian(self.LIH)
        return save_hamiltonian(tmp_path / "lih.parquet", record.hamiltonian,
                                mapping=record.mapping,
                                num_particles=record.num_particles,
                                n_spatial_orbitals=record.n_spatial_orbitals)

    def test_save_then_load_same_path_is_not_reserialized(self, tmp_path):
        """Loading and saving the same file leaves its bytes unchanged."""
        path = self._own_copy(tmp_path)
        before = Path(path).read_bytes()
        Mandacaru(method="adapt-vqe", pool="fermionic", load_hamiltonian=path,
                  save_hamiltonian=path, trace=False, max_iterations=1).run()
        assert Path(path).read_bytes() == before

    @pytest.mark.parametrize("fmt", HAMILTONIAN_FORMATS)
    def test_a_loaded_run_saves_the_same_operator_in_either_format(
            self, tmp_path, fmt):
        out = str(tmp_path / f"out{FILE_EXTENSIONS[fmt]}")
        Mandacaru(method="adapt-vqe", pool="fermionic",
                  load_hamiltonian=self.LIH, save_hamiltonian=out,
                  hamiltonian_format=fmt, trace=False, max_iterations=1).run()
        assert detect_format(out) == fmt
        assert (Path(out).read_bytes()[:4] == b"PAR1") == (fmt == "parquet")
        saved, frozen = load_hamiltonian(out), load_hamiltonian(self.LIH)
        assert saved.hamiltonian.terms == frozen.hamiltonian.terms
        assert saved.num_particles == frozen.num_particles == (1, 1)

    def test_a_frozen_cache_drives_a_run(self):
        result = Mandacaru(method="adapt-vqe", pool="fermionic",
                           load_hamiltonian=self.LIH, trace=False,
                           max_iterations=4).run()
        assert np.isfinite(result.optimal_energy)


class TestZeroHamiltonianWidth:
    @pytest.mark.parametrize("fmt", ["json", "parquet"])
    def test_a_zero_operator_keeps_its_register(self, tmp_path, fmt):
        path = str(tmp_path / f"zero.{fmt}")
        save_hamiltonian(path, PauliSum({}, num_qubits=4), format=fmt,
                         num_particles=(1, 1), n_spatial_orbitals=2)
        record = load_hamiltonian(path)
        assert record.hamiltonian.num_qubits == 4

    def test_terms_that_cancel_keep_their_register(self, tmp_path):
        path = str(tmp_path / "cancel.json")
        # A term whose coefficient canceled to zero is dropped on simplify();
        # the register width must survive it.  (A dict literal cannot repeat a
        # key, so the canceled term is written as the zero it sums to.)
        save_hamiltonian(path, PauliSum({"ZIII": 0.0}),
                         num_particles=(1, 1), n_spatial_orbitals=2)
        assert load_hamiltonian(path).hamiltonian.num_qubits == 4
