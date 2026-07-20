# -*- coding: utf-8 -*-
# file: test/test_pseudo_io.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Pseudopotential serialization: Parquet, JSON, and format auto-detection."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from carcara.basis.pseudo_io import (DEFAULT_FORMAT, FILE_EXTENSIONS,
                                     PARQUET_MAGIC, PSEUDO_FORMATS, STRIDE,
                                     available_elements, detect_format,
                                     generation_points, get_pseudopotential,
                                     library_elements, library_file,
                                     load_pseudopotential, resolve_format,
                                     save_pseudopotential)


@pytest.fixture(scope="module")
def silicon():
    """A real pseudopotential from the bundled library."""
    return get_pseudopotential("Si")


def assert_same(left, right):
    """Every numerical table and scalar survived the round trip."""
    assert left.symbol == right.symbol
    assert left.atomic_number == right.atomic_number
    assert left.local_l == right.local_l
    assert left.valence_charge == pytest.approx(right.valence_charge)
    for name in ("r", "v_local", "valence_density"):
        np.testing.assert_allclose(getattr(left, name), getattr(right, name))
    assert sorted(left.channels) == sorted(right.channels)
    for l, channel in left.channels.items():
        other = right.channels[l]
        assert channel.n == other.n
        assert channel.eigenvalue == pytest.approx(other.eigenvalue)
        assert channel.r_cut == pytest.approx(other.r_cut)
        np.testing.assert_allclose(channel.pseudo_radial, other.pseudo_radial)
        np.testing.assert_allclose(channel.v_ionic, other.v_ionic)
    assert sorted(left.projectors) == sorted(right.projectors)
    for l, projector in left.projectors.items():
        np.testing.assert_allclose(projector, right.projectors[l])
    for l, energy in left.kb_energies.items():
        assert energy == pytest.approx(right.kb_energies[l])


class TestFormatResolution:
    """Naming the formats."""

    def test_default_is_parquet(self):
        assert DEFAULT_FORMAT == "parquet"

    @pytest.mark.parametrize("spec, expected", [
        ("parquet", "parquet"), ("PARQUET", "parquet"), (".parquet", "parquet"),
        ("pq", "parquet"), ("json", "json"), (" JSON ", "json"),
    ])
    def test_resolve(self, spec, expected):
        assert resolve_format(spec) == expected

    def test_unknown_format_is_rejected(self):
        with pytest.raises(ValueError, match="unknown pseudopotential format"):
            resolve_format("hdf5")


class TestRoundTrip:
    """Saving and loading must not change the pseudopotential."""

    @pytest.mark.parametrize("fmt", PSEUDO_FORMATS)
    def test_round_trip_is_exact(self, silicon, tmp_path, fmt):
        path = save_pseudopotential(
            silicon, tmp_path / f"si{FILE_EXTENSIONS[fmt]}", format=fmt)
        assert_same(silicon, load_pseudopotential(path))

    @pytest.mark.parametrize("fmt", PSEUDO_FORMATS)
    def test_saving_is_idempotent(self, silicon, tmp_path, fmt):
        """The regression: a stride applied on every write compounds.

        ``save`` used to decimate by :data:`STRIDE` unconditionally, so a
        load-then-save cycle quietly quartered the resolution each time.
        """
        first = load_pseudopotential(save_pseudopotential(
            silicon, tmp_path / f"a{FILE_EXTENSIONS[fmt]}", format=fmt))
        second = load_pseudopotential(save_pseudopotential(
            first, tmp_path / f"b{FILE_EXTENSIONS[fmt]}", format=fmt))
        assert second.r.size == silicon.r.size
        assert_same(first, second)

    def test_the_two_formats_agree(self, silicon, tmp_path):
        """Parquet and JSON are interchangeable, not merely both loadable."""
        as_parquet = load_pseudopotential(save_pseudopotential(
            silicon, tmp_path / "si.parquet", format="parquet"))
        as_json = load_pseudopotential(save_pseudopotential(
            silicon, tmp_path / "si.json", format="json"))
        assert_same(as_parquet, as_json)

    def test_explicit_stride_decimates(self, silicon, tmp_path):
        path = save_pseudopotential(silicon, tmp_path / "si.parquet", stride=4)
        assert load_pseudopotential(path).r.size == -(-silicon.r.size // 4)

    def test_parquet_is_smaller_than_json(self, silicon, tmp_path):
        parquet = save_pseudopotential(silicon, tmp_path / "si.parquet")
        as_json = save_pseudopotential(silicon, tmp_path / "si.json")
        assert os.path.getsize(parquet) < os.path.getsize(as_json)


class TestFormatDetection:
    """Loading must not require being told the format."""

    def test_extension_selects_the_format(self, silicon, tmp_path):
        for fmt, extension in FILE_EXTENSIONS.items():
            path = save_pseudopotential(silicon, tmp_path / f"si{extension}",
                                        format=fmt)
            assert detect_format(path) == fmt

    def test_pq_extension_is_parquet(self, silicon, tmp_path):
        path = save_pseudopotential(silicon, tmp_path / "si.pq",
                                    format="parquet")
        assert detect_format(path) == "parquet"

    @pytest.mark.parametrize("fmt", PSEUDO_FORMATS)
    def test_magic_bytes_when_the_extension_is_unhelpful(self, silicon,
                                                         tmp_path, fmt):
        path = save_pseudopotential(tmp_path and silicon,
                                    tmp_path / "mystery.dat", format=fmt)
        assert detect_format(path) == fmt
        assert_same(silicon, load_pseudopotential(path))

    def test_parquet_files_carry_the_magic_number(self, silicon, tmp_path):
        path = save_pseudopotential(silicon, tmp_path / "si.parquet")
        with open(path, "rb") as handle:
            assert handle.read(4) == PARQUET_MAGIC

    def test_explicit_format_overrides_detection(self, silicon, tmp_path):
        path = save_pseudopotential(silicon, tmp_path / "si.dat", format="json")
        assert_same(silicon, load_pseudopotential(path, format="json"))

    def test_unrecognizable_content_is_reported(self, tmp_path):
        path = tmp_path / "junk.dat"
        path.write_bytes(b"\x00\x01 not a pseudopotential")
        with pytest.raises(ValueError, match="cannot determine the format"):
            detect_format(path)

    def test_missing_file_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            detect_format(tmp_path / "absent.dat")

    def test_json_stays_human_readable(self, silicon, tmp_path):
        """The point of keeping JSON: it can be inspected without a reader."""
        path = save_pseudopotential(silicon, tmp_path / "si.json", format="json")
        with open(path) as handle:
            payload = json.load(handle)
        assert payload["symbol"] == "Si"
        assert payload["atomic_number"] == 14


class TestArrowStringIsolation:
    """Reading Parquet must not drag pyarrow in behind fastparquet's back."""

    def test_reads_keep_pandas_off_arrow_strings(self, silicon, tmp_path):
        """The regression: pandas 3 defaults ``future.infer_string`` on.

        fastparquet builds its column index through pandas, so with that option
        the index is a pyarrow string array -- pulling pyarrow into a process
        that chose fastparquet precisely to avoid it.  After Qiskit's
        ``transpile`` has run, decoding those column names raises
        ``UnicodeDecodeError`` or ``ArrowException``, which broke *every*
        pseudopotential load in a full test session.
        """
        pd = pytest.importorskip("pandas")
        if not hasattr(pd.options.future, "infer_string"):
            pytest.skip("this pandas has no infer_string option")

        from carcara.core.serialization import native_pandas_strings

        with native_pandas_strings():
            assert pd.options.future.infer_string is False
        # and the setting is restored afterwards
        with native_pandas_strings():
            pass
        assert pd.options.future.infer_string is not None

    def test_load_works_with_arrow_strings_enabled(self, silicon, tmp_path):
        pd = pytest.importorskip("pandas")
        if not hasattr(pd.options.future, "infer_string"):
            pytest.skip("this pandas has no infer_string option")
        path = save_pseudopotential(silicon, tmp_path / "si.parquet")
        with pd.option_context("future.infer_string", True):
            assert_same(silicon, load_pseudopotential(path))


class TestLibrary:
    """The bundled library under ``pseudos/``."""

    def test_covers_z_below_90(self):
        elements = library_elements()
        assert len(elements) == 89
        assert elements[0] == "H" and elements[-1] == "Ac"

    def test_every_element_is_present(self):
        assert set(available_elements()) == set(library_elements())

    def test_library_files_are_parquet(self):
        assert library_file("Si").endswith(".parquet")

    def test_library_file_finds_either_format(self, tmp_path):
        (tmp_path / "Si.json").write_text("{}")
        assert library_file("Si", tmp_path).endswith(".json")

    def test_generation_points_scale_with_z(self):
        assert generation_points(1) >= 6000
        assert generation_points(80) > generation_points(8)

    def test_stride_is_not_the_save_default(self):
        """It applies to library generation only -- see the idempotence test."""
        assert STRIDE > 1

    @pytest.mark.parametrize("symbol, valence", [
        ("H", 1), ("C", 4), ("O", 6), ("Si", 4),
        ("Fe", 8),                      # 3d^6 4s^2, not a 2-electron atom
        ("Zn", 12),
    ])
    def test_valence_charges(self, symbol, valence):
        assert get_pseudopotential(symbol).valence_charge == pytest.approx(
            valence)

    @pytest.mark.parametrize("symbol", ["H", "C", "O", "Si", "Fe"])
    def test_library_entries_are_physical(self, symbol):
        pp = get_pseudopotential(symbol)
        assert np.all(np.isfinite(pp.v_local))
        assert pp.v_local[0] > -1e4          # bounded at the origin, unlike -Z/r
        assert np.all(np.diff(pp.r) > 0)
        assert pp.local_l in pp.channels

    def test_transition_metals_have_a_d_channel(self):
        """Iron's chemistry lives in 3d; a pseudopotential without it is wrong."""
        assert 2 in get_pseudopotential("Fe").channels
