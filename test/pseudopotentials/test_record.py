# -*- coding: utf-8 -*-
# file: test/pseudopotentials/test_record.py

# This code is part of Mandacaru.
# MIT License

"""The dataset record across the native boundary (:mod:`mandacaru.pseudopotentials.record`): a
generated dataset is written and read back through ``mandacaru-core`` in both formats, the
storage rule (stride, ten digits) is the core's, the defects block carries the two keys of the
native generator, and a record the file reader refuses raises."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from mandacaru import _native
from mandacaru.pseudopotentials.defects import GhostStateWarning
from mandacaru.pseudopotentials.generation import generate_paw
from mandacaru.pseudopotentials.native_context import context
from mandacaru.pseudopotentials import record
from mandacaru.pseudopotentials.io import STRIDE


@pytest.fixture(scope="module")
def hydrogen():
    return generate_paw("H")


def tables_of(pp):
    return record.record_of(pp)[1]


class TestRoundTrip:
    @pytest.mark.parametrize("fmt", ["parquet", "json"])
    def test_stride_one_keeps_the_record_to_ten_digits(self, hydrogen, tmp_path, fmt):
        path = record.write_dataset(hydrogen, tmp_path / f"H.{fmt}", format=fmt)
        back = record.read_dataset(path)
        before, after = tables_of(hydrogen), tables_of(back)
        assert before.keys() == after.keys()
        for name in before:
            scale = max(1.0, float(np.max(np.abs(before[name]))))
            assert np.max(np.abs(before[name] - after[name])) <= 5e-10 * scale, name
        assert json.loads(record.record_of(back)[0]) == json.loads(record.record_of(hydrogen)[0])
        assert back.family == hydrogen.family and back.xc == "lda"
        for l, channel in hydrogen.channels.items():
            assert np.array_equal(back.channels[l].coupling, channel.coupling)
            assert np.array_equal(back.channels[l].overlap_correction, channel.overlap_correction)

    def test_the_two_formats_hold_the_same_record(self, hydrogen, tmp_path):
        parquet = record.read_dataset(record.write_dataset(hydrogen, tmp_path / "H.parquet", stride=4))
        plain = record.read_dataset(record.write_dataset(hydrogen, tmp_path / "H.json", stride=4))
        a, b = tables_of(parquet), tables_of(plain)
        assert all(np.array_equal(a[name], b[name]) for name in a)
        assert json.loads(record.record_of(parquet)[0]) == json.loads(record.record_of(plain)[0])

    def test_a_library_stride_keeps_every_fourth_point_from_the_fourth(self, hydrogen, tmp_path):
        back = record.read_dataset(record.write_dataset(hydrogen, tmp_path / "H.parquet",
                                                        stride=STRIDE))
        assert back.r.size == hydrogen.r.size // STRIDE
        assert np.allclose(back.r, hydrogen.r[STRIDE - 1::STRIDE], rtol=1e-9)
        assert back.r[0] == pytest.approx(STRIDE * hydrogen.r[0], rel=1e-9)

    def test_the_format_follows_the_extension(self, hydrogen, tmp_path):
        record.write_dataset(hydrogen, tmp_path / "a.json")
        assert (tmp_path / "a.json").read_text().startswith("{")
        record.write_dataset(hydrogen, tmp_path / "b.parquet")
        assert (tmp_path / "b.parquet").read_bytes()[:4] == b"PAR1"

    def test_an_unknown_format_raises(self, hydrogen, tmp_path):
        with pytest.raises(ValueError, match="unknown pseudopotential format"):
            record.write_dataset(hydrogen, tmp_path / "H.dat", format="hdf5")

    def test_nothing_is_left_beside_the_file(self, hydrogen, tmp_path):
        record.write_dataset(hydrogen, tmp_path / "H.parquet")
        assert sorted(p.name for p in tmp_path.iterdir()) == ["H.parquet"]


class TestDefects:
    def test_the_native_keys_survive_the_file_and_warn_on_load(self, hydrogen, tmp_path):
        import dataclasses

        flagged = dataclasses.replace(
            hydrogen, defects={"ghosts": {}, "phases": {}, "missing": {1: 0.0345},
                               "branch_ghosts": {"1:-2": -3.7362}})
        path = record.write_dataset(flagged, tmp_path / "H.parquet")
        with pytest.warns(GhostStateWarning, match="kappa=-2 ghost -3.74 Ha"):
            back = record.read_dataset(path)
        assert back.defects["missing"] == {1: 0.0345}
        assert back.defects["branch_ghosts"] == {"1:-2": -3.7362}

    def test_a_clean_dataset_loads_silently(self, hydrogen, tmp_path):
        path = record.write_dataset(hydrogen, tmp_path / "H.parquet")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert record.read_dataset(path).defects == {}


class TestRefusals:
    def test_a_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no such pseudopotential file"):
            record.read_dataset(tmp_path / "nothing.parquet")

    def test_a_key_the_format_does_not_have_raises(self, hydrogen):
        payload, tables = record.record_of(hydrogen)
        meta = json.loads(payload)
        meta["surprise"] = 1
        with pytest.raises(ValueError, match="surprise"):
            _native.paw_record_write(context(), "/nonexistent/x.parquet", json.dumps(meta), tables)

    def test_a_table_the_payload_does_not_name_raises(self, hydrogen, tmp_path):
        payload, tables = record.record_of(hydrogen)
        tables = {**tables, "stray": np.zeros(3)}
        with pytest.raises(ValueError):
            _native.paw_record_write(context(), str(tmp_path / "x.parquet"), payload, tables)
        assert list(tmp_path.iterdir()) == []

    def test_a_table_that_is_not_float64_raises(self, hydrogen, tmp_path):
        payload, tables = record.record_of(hydrogen)
        tables["r"] = tables["r"].astype(np.float32)
        with pytest.raises(ValueError, match="float64"):
            _native.paw_record_write(context(), str(tmp_path / "x.parquet"), payload, tables)

    def test_a_version_from_the_future_raises(self, hydrogen, tmp_path):
        payload, tables = record.record_of(hydrogen)
        meta = json.loads(payload)
        meta["version"] = record.FORMAT_VERSION + 1
        with pytest.raises(ValueError, match="newer than this build"):
            _native.paw_record_write(context(), str(tmp_path / "x.parquet"), json.dumps(meta), tables)

    def test_free_waves_cannot_be_stored(self, hydrogen, tmp_path):
        import dataclasses

        free = dataclasses.replace(hydrogen, norm_deficit=None)
        with pytest.raises(ValueError, match="norm_deficit"):
            record.write_dataset(free, tmp_path / "H.parquet")
        assert list(tmp_path.iterdir()) == []


class TestLocalPotential:
    def test_the_native_spline_is_the_table_inside_and_the_ion_beyond(self, hydrogen):
        r, v = hydrogen.r, hydrogen.v_local
        values, slopes = _native.paw_local_value(context(), r, v, hydrogen.valence_charge,
                                                     np.array([r[10], r[-1], 2 * r[-1]]))
        assert values[0] == pytest.approx(v[10], rel=1e-12)
        assert values[1] == pytest.approx(v[-1], rel=1e-12)
        assert values[2] == pytest.approx(-hydrogen.valence_charge / (2 * r[-1]), rel=1e-12)
        assert slopes[2] == pytest.approx(hydrogen.valence_charge / (2 * r[-1]) ** 2, rel=1e-12)
        flat = _native.paw_local_value(context(), r, v, 1.0, np.array([0.5 * r[0]]))
        assert flat[1][0] == 0.0 and flat[0][0] == pytest.approx(v[0], rel=1e-12)
