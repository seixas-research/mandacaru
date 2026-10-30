# -*- coding: utf-8 -*-
# file: test/migration/test_native_parquet.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The native Parquet reader and writer against the frozen references.

``reference/parquet/references.json`` was written on 2026-10-29 with the stack
this port replaced (fastparquet 2026.5.0, pandas, pyarrow 26.0.0).  It holds,
for every one of the 431 shipped pseudopotential files (368 PAW-LCAO plus 63
UPAW-LCAO), the SHA-256 of each column's float64 bytes and of the key/value
metadata text, and for the 98 small fixtures under ``reference/parquet`` what a
conforming reader must return or refuse.  ``old_api_results.json`` records what
the old Python readers returned for the same fixtures.

A ported routine reproduces the number it replaces: every column below must be
bit-identical.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pytest

from mandacaru import _native
from mandacaru.core.serialization import load_hamiltonian
from mandacaru.pseudopotentials import io as pseudo_io
from mandacaru.pseudopotentials.io import load_pseudopotential

REFERENCE = Path(__file__).parent / "reference" / "parquet"
REFERENCES = json.loads((REFERENCE / "references.json").read_text())
OLD_API = json.loads((REFERENCE / "old_api_results.json").read_text())["fixtures"]
CORPUS = REFERENCES["corpus_files"]
FIXTURES = REFERENCES["fixtures"]

#: Where each corpus prefix lives; ``None`` when the variable is unset.
LIBRARY_ROOTS = {
    "paw": "MANDACARU_PAW_PATH",
    "upaw": "MANDACARU_UPAW_PATH",
}
HAS_PYARROW = importlib.util.find_spec("pyarrow") is not None


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def float64_digest(values) -> str:
    return sha256_of(np.ascontiguousarray(values, dtype="<f8").tobytes())


def string_digest(values) -> str:
    return sha256_of("\0".join(values).encode("utf-8"))


def read(path, context=None):
    context = context or _native.Context(threads=1)
    return _native.read_parquet(context, str(path))


# --------------------------------------------------------------------------- #
# The 431 shipped files: bit-identical columns and metadata.
# --------------------------------------------------------------------------- #

def corpus_path(key: str) -> Path:
    """``paw/lda-sr/Si.parquet`` -> ``$MANDACARU_PAW_PATH/lda-sr/Si.parquet``."""
    prefix, _, rest = key.partition("/")
    variable = LIBRARY_ROOTS[prefix]
    root = os.environ.get(variable, "").strip()
    if not root:
        pytest.skip(f"{variable} is not set")
    path = Path(root) / rest
    if not Path(root).is_dir():
        pytest.skip(f"{root} is not a directory")
    return path


def test_the_frozen_corpus_is_the_one_the_plan_names():
    totals = REFERENCES["corpus_totals"]
    assert len(CORPUS) == totals["files"] == 431
    assert sum(len(entry["columns"]) for entry in CORPUS.values()) == totals[
        "columns"] == 11994
    assert totals["files_per_set"] == {
        "paw/lda-fr": 92, "paw/lda-sr": 92, "paw/pbe-fr": 92,
        "paw/pbe-sr": 92, "upaw/lda-sr": 63}


@pytest.fixture(scope="module")
def context():
    return _native.Context(threads=1)


@pytest.mark.parametrize("key", sorted(CORPUS))
def test_shipped_file_is_bit_identical(key, context):
    expected = CORPUS[key]
    path = corpus_path(key)
    assert path.is_file(), f"{path} is missing from the library"
    assert path.stat().st_size == expected["bytes"]

    table = _native.read_parquet(context, str(path))
    assert table["num_rows"] == expected["rows"]
    names = list(table["columns"])
    assert names == [column[0] for column in expected["columns"]]
    for name, rows, digest in expected["columns"]:
        values = table["columns"][name]
        assert isinstance(values, np.ndarray) and values.dtype == np.float64
        assert values.size == rows
        assert float64_digest(values) == digest, f"{key}: column {name!r}"

    metadata = dict(table["metadata"])
    text = metadata.pop(expected["kv"]["key"])
    encoded = text.encode("utf-8")
    assert len(encoded) == expected["kv"]["bytes"]
    assert sha256_of(encoded) == expected["kv"]["sha256"]
    assert expected["kv"]["scalars"]["format"] == json.loads(text)["format"]
    assert sorted(json.loads(text)) == expected["kv"]["top_level_keys"]
    # Whatever else the writer left in the footer (the key fastparquet adds).
    assert sorted(metadata) == sorted(expected["kv_other_keys"])
    for other, record in expected["kv_other_keys"].items():
        assert len(metadata[other].encode("utf-8")) == record["bytes"]
        assert sha256_of(metadata[other].encode("utf-8")) == record["sha256"]
    assert context.memory_used_bytes == 0

    footer = _native.read_parquet_metadata(context, str(path))
    assert footer["num_rows"] == expected["rows"]
    assert footer["created_by"] == expected["created_by"]
    assert footer["metadata"] == table["metadata"]


# --------------------------------------------------------------------------- #
# The fixtures a conforming reader reads.
# --------------------------------------------------------------------------- #

READABLE = sorted(name for name, info in FIXTURES.items()
                  if info["expect_new_reader"] == "ok")
REFUSED = sorted(name for name, info in FIXTURES.items()
                 if info["expect_new_reader"] == "error")
HAMILTONIANS = [name for name in READABLE if name.startswith("hamiltonian/")]
PSEUDOPOTENTIALS = [name for name in READABLE
                    if name.startswith("pseudopotential/")]


def test_the_fixture_inventory_is_complete():
    assert len(FIXTURES) == 98 == len(OLD_API)
    assert len(READABLE) == 38 and len(REFUSED) == 60
    for name, info in FIXTURES.items():
        assert (REFERENCE / name).stat().st_size == info["bytes"] < 20_000
        assert sha256_of((REFERENCE / name).read_bytes()) == info["sha256"]


@pytest.mark.parametrize("name", READABLE)
def test_readable_fixture_matches_what_pyarrow_decoded(name):
    context = _native.Context(threads=1)
    table = read(REFERENCE / name, context)
    decoded = FIXTURES[name]["decoded_by_pyarrow"]
    assert list(table["columns"]) == list(decoded)
    assert table["num_rows"] == FIXTURES[name]["structure"]["num_rows"]
    for column, record in decoded.items():
        values = table["columns"][column]
        assert len(values) == record["rows"] == table["num_rows"]
        if record["type"] == "double":
            assert isinstance(values, np.ndarray)
            assert float64_digest(values) == record["sha256_float64_bytes"]
        else:       # "string", or "binary" for a column without the annotation
            assert isinstance(values, list)
            assert string_digest(values) == record["sha256_utf8_nul_joined"]
    assert context.memory_used_bytes == 0


@pytest.mark.parametrize("name", READABLE)
def test_readable_fixture_footer_matches(name):
    footer = _native.read_parquet_metadata(_native.Context(threads=1),
                                           str(REFERENCE / name))
    recorded = FIXTURES[name]["structure"]
    assert footer["num_rows"] == recorded["num_rows"]
    assert footer["num_row_groups"] == recorded["num_row_groups"]
    assert [column for column, _ in footer["columns"]] == [
        column["name"] for column in recorded["schema"]]


def terms_digest(terms) -> str:
    return sha256_of("\n".join(
        f"{label} {complex(c).real.hex()} {complex(c).imag.hex()}"
        for label, c in sorted(terms.items())).encode("utf-8"))


def old_result(name, api):
    """What the replaced reader returned (fastparquet, else pyarrow), if it did."""
    for engine in ("fastparquet", "pyarrow"):
        outcome = OLD_API[name][api][engine]
        if outcome.get("ok"):
            return outcome["result"]
    return None


@pytest.mark.parametrize("name", HAMILTONIANS)
def test_hamiltonian_loads_as_the_old_reader_loaded_it(name):
    record = load_hamiltonian(REFERENCE / name)
    expected = old_result(name, "load_hamiltonian")
    if expected is None:
        # The old readers refused these (an unannotated byte-array column, a
        # missing mapping key); the native reader is specified to accept them
        # and to return the 27 terms of the LiH reference.
        expected = old_result("hamiltonian/fastparquet_zstd.parquet",
                              "load_hamiltonian")
        assert name in {"hamiltonian/pyarrow_binary_unannotated.parquet",
                        "hamiltonian/semantic_kv_defaults_only_required.parquet"}
    assert record.num_qubits == expected["num_qubits"]
    assert len(record.hamiltonian.terms) == expected["n_terms"]
    assert terms_digest(record.hamiltonian.terms) == expected["terms_sha256"]
    if name != "hamiltonian/semantic_kv_defaults_only_required.parquet":
        assert record.mapping == expected["mapping"]
        assert (None if record.num_particles is None
                else list(record.num_particles)) == expected["num_particles"]
        assert record.n_spatial_orbitals == expected["n_spatial_orbitals"]
        assert record.metadata == expected["metadata"]
    for label, real, imag in expected.get("terms", []):
        coefficient = record.hamiltonian.terms[label]
        assert (coefficient.real.hex(), coefficient.imag.hex()) == (real, imag)


def test_a_file_with_only_the_required_keys_takes_the_documented_defaults():
    record = load_hamiltonian(
        REFERENCE / "hamiltonian/semantic_kv_defaults_only_required.parquet")
    assert record.mapping == "jordan_wigner"
    assert record.num_particles is None
    assert record.n_spatial_orbitals is None
    assert record.metadata == {}
    assert record.num_qubits == 4


def test_an_empty_hamiltonian_keeps_its_register_width():
    for name in ("hamiltonian/fastparquet_empty.parquet",
                 "hamiltonian/pyarrow_empty.parquet"):
        record = load_hamiltonian(REFERENCE / name)
        assert record.hamiltonian.terms == {}
        assert record.hamiltonian.num_qubits == 4


def test_a_register_of_forty_eight_qubits_survives_the_dictionary_fallback():
    record = load_hamiltonian(
        REFERENCE / "hamiltonian/pyarrow_dictionary_fallback.parquet")
    assert record.num_qubits == 48 and len(record.hamiltonian.terms) == 3200


@pytest.mark.parametrize("name", PSEUDOPOTENTIALS)
def test_pseudopotential_fixture_loads_as_the_old_reader_loaded_it(name):
    expected = old_result(name, "load_pseudopotential")
    table = read(REFERENCE / name)
    recorded = FIXTURES[name]["kv"]["mandacaru.pseudopotential"]
    text = table["metadata"]["mandacaru.pseudopotential"].encode("utf-8")
    assert (len(text), sha256_of(text)) == (recorded["bytes"], recorded["sha256"])

    payload = pseudo_io._read_parquet(str(REFERENCE / name))
    tables = payload.pop("radial_tables")
    # The recorded digest covers what the dataset codec receives: the payload
    # without its radial tables and without the three identity keys.
    scalars = {key: value for key, value in payload.items()
               if key not in ("format", "version", "family")}
    assert sha256_of(json.dumps(scalars, sort_keys=True).encode("utf-8")
                     ) == expected["payload_scalars_sha256"]
    assert list(tables) == expected["radial_columns"]
    for column, record in expected["radial_tables"].items():
        assert len(tables[column]) == record["rows"]
        assert float64_digest(tables[column]) == record["sha256"]
    dataset = load_pseudopotential(REFERENCE / name)
    assert dataset.symbol == expected["symbol"]
    assert type(dataset).__name__ == expected["class"]


# --------------------------------------------------------------------------- #
# The fixtures a conforming reader refuses.
# --------------------------------------------------------------------------- #

#: Refusals whose reason lives in the Mandacaru layer, not in the container:
#: the generic reader returns the table and ``load_hamiltonian`` refuses it.
SEMANTIC = {name for name in REFUSED
            if name.startswith("negative/hamiltonian_")
            or name == "negative/column_double_pauli.parquet"}

#: Where the recorded hint is not a word of the native message (the message
#: says "extends to byte N, past the data area" for a chunk outside the file).
HINT_DIFFERS = {
    "negative/footer_chunk_size_huge.parquet": "past the data area",
    "negative/footer_page_offset_outside_file.parquet": "past the data area",
}


def hint_found(hints, message) -> bool:
    lowered = message.lower()
    return any(hint.lower() in lowered for hint in hints)


@pytest.mark.parametrize("name", REFUSED)
def test_refused_fixture_raises_from_load_hamiltonian(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises((ValueError, MemoryError)):
            load_hamiltonian(REFERENCE / name,
                             resources={"memory_limit_bytes": 1 << 30})


@pytest.mark.parametrize("name", sorted(set(REFUSED) - SEMANTIC))
def test_refused_container_names_its_reason_and_releases_memory(name):
    context = _native.Context(threads=1, memory_limit_bytes=1 << 30)
    with pytest.raises((ValueError, MemoryError)) as raised:
        _native.read_parquet(context, str(REFERENCE / name))
    message = str(raised.value)
    hints = FIXTURES[name]["new_reader_error_hint"]
    if name in HINT_DIFFERS:
        assert HINT_DIFFERS[name] in message
    else:
        assert hint_found(hints, message), (hints, message)
    assert context.memory_used_bytes == 0


@pytest.mark.parametrize("name", sorted(SEMANTIC))
def test_refused_hamiltonian_names_its_reason(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(ValueError) as raised:
            load_hamiltonian(REFERENCE / name)
    hints = FIXTURES[name]["new_reader_error_hint"]
    assert hint_found(hints, str(raised.value)), (hints, str(raised.value))


def test_a_footer_claiming_two_to_the_forty_rows_hits_the_budget_first():
    context = _native.Context(threads=1, memory_limit_bytes=1 << 20)
    with pytest.raises(MemoryError):
        _native.read_parquet(context, str(
            REFERENCE / "negative/footer_rows_2e40.parquet"))
    assert context.memory_used_bytes == 0


def test_read_metadata_never_touches_the_data_pages(tmp_path):
    source = (REFERENCE / "hamiltonian/fastparquet_zstd.parquet").read_bytes()
    footer = _native.read_parquet_metadata(
        _native.Context(threads=1), str(REFERENCE / "hamiltonian/fastparquet_zstd.parquet"))
    length = int.from_bytes(source[-8:-4], "little")
    corrupted = bytearray(source)
    for i in range(4, len(source) - 8 - length):
        corrupted[i] ^= 0xFF
    path = tmp_path / "pages_destroyed.parquet"
    path.write_bytes(bytes(corrupted))
    assert _native.read_parquet_metadata(
        _native.Context(threads=1), str(path)) == footer
    with pytest.raises(ValueError):
        read(path)


# --------------------------------------------------------------------------- #
# What Mandacaru writes.
# --------------------------------------------------------------------------- #

SPECIAL = np.array([0.0, -0.0, 1.0, -1.5, np.inf, -np.inf, np.nan,
                    5e-324, -5e-324, 2.2250738585072014e-308,
                    1.7976931348623157e308, np.pi])
STRINGS = ["", "IXYZ", "naïve", "日本語", "a" * 5000, "tab\tnewline\n", "I"]


def special_columns():
    rows = len(SPECIAL)
    return {"value": SPECIAL.copy(),
            "label": [STRINGS[i % len(STRINGS)] for i in range(rows)],
            "other": np.arange(rows, dtype=np.float64)[::-1].copy()}


@pytest.mark.parametrize("compression", ["zstd", "uncompressed", "none"])
def test_written_columns_come_back_bit_exact(tmp_path, compression):
    columns = special_columns()
    metadata = {"b.key": "second", "a.key": "first", "unicode": "ção"}
    path = tmp_path / "t.parquet"
    context = _native.Context(threads=1)
    _native.write_parquet(context, str(path), columns, metadata, compression)
    assert context.memory_used_bytes == 0
    table = read(path)
    assert list(table["columns"]) == ["value", "label", "other"]
    assert table["columns"]["value"].tobytes() == columns["value"].tobytes()
    assert table["columns"]["label"] == columns["label"]
    assert table["columns"]["other"].tobytes() == columns["other"].tobytes()
    assert table["metadata"] == metadata
    data = path.read_bytes()
    assert data[:4] == data[-4:] == b"PAR1"


def test_the_default_codec_is_zstd_and_the_bytes_are_deterministic(tmp_path):
    columns = special_columns()
    context = _native.Context(threads=1)
    paths = [tmp_path / f"{i}.parquet" for i in range(3)]
    _native.write_parquet(context, str(paths[0]), columns, {"k": "v"})
    _native.write_parquet(context, str(paths[1]), columns, {"k": "v"}, "zstd")
    _native.write_parquet(context, str(paths[2]), columns, {"k": "v"}, "ZSTD")
    assert paths[0].read_bytes() == paths[1].read_bytes() == paths[2].read_bytes()
    assert b"mandacaru-parquet 1" in paths[0].read_bytes()


def test_a_large_table_spans_row_groups_and_pages(tmp_path):
    rng = np.random.default_rng(7)
    rows = (1 << 20) + 70_000          # one more row group than the default size
    columns = {"x": rng.normal(size=rows),
               "label": ["IXYZ"[i % 4] * 8 for i in range(rows)]}
    path = tmp_path / "big.parquet"
    context = _native.Context(threads=1)
    _native.write_parquet(context, str(path), columns, {})
    footer = _native.read_parquet_metadata(context, str(path))
    assert footer["num_rows"] == rows and footer["num_row_groups"] == 2
    table = read(path)
    assert table["columns"]["x"].tobytes() == columns["x"].tobytes()
    assert table["columns"]["label"] == columns["label"]
    assert context.memory_used_bytes == 0


def test_an_empty_table_round_trips(tmp_path):
    path = tmp_path / "empty.parquet"
    _native.write_parquet(_native.Context(threads=1), str(path),
                          {"x": np.zeros(0), "s": []}, {"k": "v"})
    table = read(path)
    assert table["num_rows"] == 0
    assert table["columns"]["x"].size == 0 and table["columns"]["s"] == []
    assert table["metadata"] == {"k": "v"}


@pytest.mark.parametrize("columns, match", [
    ({"x": np.zeros(3), "y": np.zeros(2)}, "values"),
    ({"x": np.zeros(3, dtype=np.float32)}, "float64"),
    ({"x": np.zeros(6)[::2]}, "contiguous"),
    ({"x": np.zeros((2, 2))}, "one-dimensional|1-D|dimension"),
    ({"x": [1, 2]}, "str"),
    ({"": np.zeros(2)}, "name"),
])
def test_invalid_columns_raise_and_write_nothing(tmp_path, columns, match):
    context = _native.Context(threads=1)
    with pytest.raises((ValueError, TypeError), match=f"(?i){match}"):
        _native.write_parquet(context, str(tmp_path / "t.parquet"), columns, {})
    assert list(tmp_path.iterdir()) == []
    assert context.memory_used_bytes == 0


def test_an_unknown_codec_names_the_accepted_ones(tmp_path):
    with pytest.raises(ValueError, match="zstd") as raised:
        _native.write_parquet(_native.Context(threads=1),
                              str(tmp_path / "t.parquet"),
                              {"x": np.zeros(2)}, {}, "gzip")
    assert "uncompressed" in str(raised.value)
    assert list(tmp_path.iterdir()) == []


def test_a_failed_write_keeps_the_old_file_and_leaves_no_stage_file(tmp_path):
    path = tmp_path / "t.parquet"
    context = _native.Context(threads=1)
    _native.write_parquet(context, str(path), {"x": np.arange(4.0)}, {})
    before = path.read_bytes()
    with pytest.raises(ValueError):
        _native.write_parquet(context, str(path),
                              {"x": np.zeros(3), "y": np.zeros(2)}, {})
    assert path.read_bytes() == before
    assert [entry.name for entry in tmp_path.iterdir()] == ["t.parquet"]


def test_a_memory_limit_too_small_for_the_table_raises(tmp_path):
    context = _native.Context(threads=1, memory_limit_bytes=64)
    with pytest.raises(MemoryError):
        _native.write_parquet(context, str(tmp_path / "t.parquet"),
                              {"x": np.arange(10_000.0)}, {})
    assert context.memory_used_bytes == 0
    assert list(tmp_path.iterdir()) == []


def test_a_memory_limit_too_small_for_the_file_raises_on_read(tmp_path):
    path = tmp_path / "t.parquet"
    _native.write_parquet(_native.Context(threads=1), str(path),
                          {"x": np.arange(10_000.0)}, {})
    context = _native.Context(threads=1, memory_limit_bytes=64)
    with pytest.raises(MemoryError):
        read(path, context)
    assert context.memory_used_bytes == 0


def test_the_python_objects_of_a_text_column_are_charged_to_the_budget(tmp_path):
    """The ``str`` objects a text column becomes are far larger than the
    Rust table they come from; they count against the memory budget."""
    path = tmp_path / "t.parquet"
    labels = [f"{i % 9999:04d}" for i in range(300_000)]
    _native.write_parquet(_native.Context(threads=1), str(path),
                          {"label": labels}, {})
    # The Rust table (about 3.6 MB) fits; 300,000 Python strings (21 MB) do not.
    small = _native.Context(threads=1, memory_limit_bytes=12 << 20)
    with pytest.raises(MemoryError):
        read(path, small)
    assert small.memory_used_bytes == 0
    roomy = _native.Context(threads=1, memory_limit_bytes=96 << 20)
    assert read(path, roomy)["columns"]["label"] == labels
    assert roomy.memory_peak_bytes >= 20 << 20


def test_the_metadata_dict_is_charged_to_the_budget(tmp_path):
    path = tmp_path / "t.parquet"
    metadata = {f"key-{i}": "v" * 40 for i in range(60_000)}
    _native.write_parquet(_native.Context(threads=1), str(path),
                          {"x": np.arange(3.0)}, metadata)
    for call in (read, lambda p, c: _native.read_parquet_metadata(c, str(p))):
        small = _native.Context(threads=1, memory_limit_bytes=6 << 20)
        with pytest.raises(MemoryError):
            call(path, small)
        assert small.memory_used_bytes == 0
        roomy = _native.Context(threads=1, memory_limit_bytes=96 << 20)
        assert call(path, roomy)["metadata"] == metadata


def test_the_columns_are_handed_over_without_a_copy(tmp_path):
    """A float64 column moves into its array: the table's bytes are not held
    twice while the arrays are built."""
    path = tmp_path / "t.parquet"
    values = np.random.default_rng(1).normal(size=2_000_000)
    _native.write_parquet(_native.Context(threads=1), str(path), {"x": values}, {},
                          "uncompressed")
    context = _native.Context(threads=1)
    table = read(path, context)
    assert table["columns"]["x"].tobytes() == values.tobytes()
    # table (16 MB) + one chunk (16 MB) + page scratch; a copy would add 16 MB more.
    assert context.memory_peak_bytes < 3 * values.nbytes


@pytest.mark.parametrize("threads", [1, 4])
def test_a_small_thread_stack_cannot_kill_the_process(tmp_path, threads):
    """The zstd codec keeps large state on the stack; on a 128 KiB thread it
    used to abort the whole interpreter (SIGBUS) instead of raising."""
    script = """
        import sys, threading
        import numpy as np
        from mandacaru import _native
        path = sys.argv[1]
        columns = {"x": np.arange(300_000.0) % 97, "label": ["IXYZ"] * 1000 + ["Z"] * 299_000}
        outcome = []
        def work():
            context = _native.Context(threads=int(sys.argv[2]))
            _native.write_parquet(context, path, columns, {"k": "v"})
            table = _native.read_parquet(context, path)
            outcome.append(table["columns"]["x"].tobytes() == columns["x"].tobytes()
                           and table["columns"]["label"] == columns["label"])
        threading.stack_size(128 * 1024)
        thread = threading.Thread(target=work)
        thread.start(); thread.join()
        assert outcome == [True], outcome
        print("ok")
    """
    done = subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(script),
         str(tmp_path / "t.parquet"), str(threads)],
        capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, (done.returncode, done.stderr)
    assert "ok" in done.stdout


def test_a_missing_file_raises_the_operating_system_error(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError, OSError)):
        read(tmp_path / "absent.parquet")


# --------------------------------------------------------------------------- #
# Interoperability with a foreign implementation (pyarrow, in a subprocess).
#
# pyarrow is not a dependency; when it is installed it is exercised in a
# separate interpreter that never imports Mandacaru or Qiskit, because pyarrow
# imported into a process that has run Qiskit's transpiler can crash it.
# --------------------------------------------------------------------------- #

pyarrow_only = pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")


def run_pyarrow(script: str, *arguments) -> dict:
    done = subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(script), *map(str, arguments)],
        capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


READ_WITH_PYARROW = """
    import hashlib, json, sys
    import numpy as np
    import pyarrow.parquet as pq
    table = pq.read_table(sys.argv[1])
    out = {"num_rows": table.num_rows, "types": {}, "digests": {},
           "metadata": {}}
    for field in table.schema:
        column = table.column(field.name)
        out["types"][field.name] = str(field.type)
        if str(field.type) == "double":
            data = np.asarray(column.to_numpy(), dtype="<f8").tobytes()
        else:
            data = "\\0".join(column.to_pylist()).encode("utf-8")
        out["digests"][field.name] = hashlib.sha256(data).hexdigest()
    meta = pq.read_metadata(sys.argv[1]).metadata or {}
    out["metadata"] = {k.decode(): v.decode() for k, v in meta.items()
                       if not k.startswith(b"ARROW")}
    out["created_by"] = pq.read_metadata(sys.argv[1]).created_by
    out["row_groups"] = pq.read_metadata(sys.argv[1]).num_row_groups
    print(json.dumps(out))
"""

WRITE_WITH_PYARROW = """
    import hashlib, json, sys
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    rng = np.random.default_rng(11)
    rows = 5000
    labels = ["".join(rng.choice(list("IXYZ"), size=12)) for _ in range(rows)]
    real = rng.normal(size=rows)
    imag = rng.normal(size=rows) * 1e-3
    table = pa.table({"pauli": labels, "real": real, "imag": imag})
    out = {"real": hashlib.sha256(real.astype("<f8").tobytes()).hexdigest(),
           "imag": hashlib.sha256(imag.astype("<f8").tobytes()).hexdigest(),
           "pauli": hashlib.sha256("\\0".join(labels).encode()).hexdigest(),
           "files": {}}
    directory = sys.argv[1]
    options = {
        "zstd_dictionary": dict(compression="zstd", use_dictionary=True),
        "zstd_plain": dict(compression="zstd", use_dictionary=False),
        "snappy_dictionary": dict(compression="snappy", use_dictionary=True),
        "snappy_plain": dict(compression="snappy", use_dictionary=False),
        "none_plain": dict(compression="none", use_dictionary=False),
        "v2_pages": dict(compression="zstd", data_page_version="2.0"),
        "v1_pages_small": dict(compression="zstd", data_page_size=2048,
                               data_page_version="1.0"),
        "row_groups": dict(compression="zstd", row_group_size=1200),
        "format_1_0": dict(compression="snappy", version="1.0"),
        "checksums": dict(compression="zstd", write_page_checksum=True),
    }
    for name, kwargs in options.items():
        path = f"{directory}/{name}.parquet"
        pq.write_table(table, path, **kwargs)
        out["files"][name] = path
    print(json.dumps(out))
"""


@pyarrow_only
def test_pyarrow_reads_what_mandacaru_writes(tmp_path):
    from mandacaru.core import PauliSum
    from mandacaru.core.serialization import save_hamiltonian

    rng = np.random.default_rng(3)
    labels = {"".join(rng.choice(list("IXYZ"), size=10)) for _ in range(900)}
    pauli = PauliSum({label: complex(rng.normal(), rng.normal() * 1e-2)
                      for label in labels})
    for compression in ("zstd", "uncompressed"):
        path = save_hamiltonian(tmp_path / f"{compression}.parquet", pauli,
                                mapping="parity", num_particles=(1, 1),
                                n_spatial_orbitals=5, compression=compression)
        theirs = run_pyarrow(READ_WITH_PYARROW, path)
        ours = read(path)
        assert theirs["num_rows"] == ours["num_rows"] == len(labels)
        assert theirs["types"] == {"pauli": "string", "real": "double",
                                   "imag": "double"}
        assert theirs["digests"]["pauli"] == string_digest(ours["columns"]["pauli"])
        assert theirs["digests"]["real"] == float64_digest(ours["columns"]["real"])
        assert theirs["digests"]["imag"] == float64_digest(ours["columns"]["imag"])
        assert theirs["metadata"] == ours["metadata"]
        assert theirs["created_by"] == "mandacaru-parquet 1"

    fixture = REFERENCE / "pseudopotential/fastparquet_zstd.parquet"
    again = pseudo_io.save_pseudopotential(load_pseudopotential(fixture),
                                           tmp_path / "pp.parquet", stride=1)
    theirs = run_pyarrow(READ_WITH_PYARROW, again)
    ours = read(again)
    assert set(theirs["types"].values()) == {"double"}
    assert theirs["metadata"] == ours["metadata"]
    for name, values in ours["columns"].items():
        assert theirs["digests"][name] == float64_digest(values)


@pyarrow_only
def test_pyarrow_reads_a_table_with_special_values(tmp_path):
    path = tmp_path / "special.parquet"
    columns = special_columns()
    _native.write_parquet(_native.Context(threads=1), str(path), columns,
                          {"k": "v"})
    theirs = run_pyarrow(READ_WITH_PYARROW, path)
    assert theirs["types"] == {"value": "double", "label": "string",
                               "other": "double"}
    assert theirs["digests"]["value"] == float64_digest(columns["value"])
    assert theirs["digests"]["label"] == string_digest(columns["label"])
    assert theirs["metadata"] == {"k": "v"}


@pyarrow_only
def test_mandacaru_reads_what_pyarrow_writes(tmp_path):
    written = run_pyarrow(WRITE_WITH_PYARROW, tmp_path)
    for name, path in written["files"].items():
        table = read(path)
        assert table["num_rows"] == 5000, name
        assert float64_digest(table["columns"]["real"]) == written["real"], name
        assert float64_digest(table["columns"]["imag"]) == written["imag"], name
        assert string_digest(table["columns"]["pauli"]) == written["pauli"], name


@pyarrow_only
def test_mandacaru_reads_an_empty_pyarrow_table_written_without_a_dictionary(tmp_path):
    """pyarrow stores zero rows without a dictionary as chunks whose offset
    is 0 and whose size is 0; they hold no byte and must still load."""
    written = run_pyarrow("""
        import json, sys
        import pyarrow as pa, pyarrow.parquet as pq
        schema = pa.schema([("pauli", pa.string()), ("real", pa.float64()),
                            ("imag", pa.float64())])
        files = {}
        for compression in ("zstd", "snappy", "none"):
            for version in ("1.0", "2.6"):
                path = f"{sys.argv[1]}/empty_{compression}_{version}.parquet"
                table = pa.table({"pauli": pa.array([], pa.string()),
                                  "real": pa.array([], pa.float64()),
                                  "imag": pa.array([], pa.float64())})
                pq.write_table(table.replace_schema_metadata({"mandacaru.num_qubits": "4"}),
                               path, compression=compression, use_dictionary=False,
                               version=version)
                files[f"{compression}_{version}"] = path
        print(json.dumps(files))
    """, tmp_path)
    for name, path in written.items():
        table = read(path)
        assert table["num_rows"] == 0, name
        assert list(table["columns"]) == ["pauli", "real", "imag"], name
        assert table["metadata"]["mandacaru.num_qubits"] == "4", name
        footer = _native.read_parquet_metadata(_native.Context(threads=1), path)
        assert footer["num_rows"] == 0, name


# --------------------------------------------------------------------------- #
# The process-level guarantee: no Python Parquet package is imported.
# --------------------------------------------------------------------------- #

def test_the_native_parquet_path_imports_no_python_parquet_package():
    script = """
        import sys
        import mandacaru
        from mandacaru import _native
        from mandacaru.core import serialization
        from mandacaru.pseudopotentials import io
        import numpy
        retired = [m for m in sys.modules
                   if m.split(".")[0] in ("fastparquet", "pyarrow", "pandas")]
        assert not retired, retired
        print("ok")
    """
    done = subprocess.run([sys.executable, "-I", "-c", textwrap.dedent(script)],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert "ok" in done.stdout
