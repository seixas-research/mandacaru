# -*- coding: utf-8 -*-
# file: test/test_pseudo_io.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Pseudopotential serialization: Parquet, JSON, and format auto-detection."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from mandacaru.pseudopotentials.paw import get_paw, paw_library_path
from mandacaru.pseudopotentials.io import (DEFAULT_FORMAT, FILE_EXTENSIONS,
                                     PARQUET_MAGIC, PSEUDO_FORMATS, STRIDE,
                                     available_elements, detect_format,
                                     library_elements, library_file,
                                     load_pseudopotential, resolve_format,
                                     save_pseudopotential)


@pytest.fixture(scope="module")
def silicon():
    """A real dataset from the PAW-LCAO library ($MANDACARU_PAW_PATH)."""
    return get_paw("Si")


def assert_same(left, right):
    """Every numerical table and scalar survived the round trip."""
    assert left.symbol == right.symbol
    assert left.atomic_number == right.atomic_number
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
    assert sorted(left.projectors) == sorted(right.projectors)
    for l, projector in left.projectors.items():
        np.testing.assert_allclose(projector, right.projectors[l])
    assert left.relativistic_exchange == right.relativistic_exchange


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
        assert load_pseudopotential(path).r.size == silicon.r.size // 4

    def test_a_decimated_grid_starts_one_step_from_the_origin(self):
        """The PAW-LCAO grids are r_k = k h; every Numerov
        diagnostic of a loaded dataset prepends r = 0 and takes r[1] - r[0]
        as its step, so the stored grid must be k (stride h) -- not
        h, 5h, 9h, ..., whose first step differs from the rest."""
        from mandacaru.pseudopotentials.io import _table

        h = 0.0025
        r = np.asarray(_table(h * np.arange(1, 4001), 4))
        steps = np.diff(np.concatenate([[0.0], r]))
        assert np.allclose(steps, 4 * h, rtol=1e-8)

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

        from mandacaru.core.serialization import native_pandas_strings

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
    """The library layout every family shares."""

    def test_covers_z_up_to_92(self):
        elements = library_elements()
        assert len(elements) == 92
        assert elements[0] == "H" and elements[-1] == "U"

    def test_every_element_is_present(self):
        assert set(available_elements(paw_library_path())) == set(
            library_elements())

    def test_library_files_are_parquet(self):
        assert library_file("Si", paw_library_path()).endswith(".parquet")

    def test_library_file_finds_either_format(self, tmp_path):
        (tmp_path / "Si.json").write_text("{}")
        assert library_file("Si", tmp_path).endswith(".json")

    def test_stride_is_not_the_save_default(self):
        """It applies to library generation only -- see the idempotence test."""
        assert STRIDE > 1


class TestRelativisticExchangeRecord:
    """Whether the functional carried the relativistic exchange correction is
    stored, because rescreening a loaded file has to match its unscreening."""

    def test_a_relativistic_dataset_carries_it(self, silicon):
        assert silicon.relativity == "scalar" and silicon.relativistic_exchange

    def test_it_survives_the_round_trip(self, silicon, tmp_path):
        loaded = load_pseudopotential(save_pseudopotential(
            silicon, tmp_path / "si.parquet"))
        assert loaded.relativistic_exchange


def _pbe_library():
    """``$MANDACARU_PAW_PATH/pbe-sr``, or a skip when the checkout has no
    PBE set (an older ``mandacaru-paw``)."""
    from mandacaru.pseudopotentials.environment import LibraryPathError

    try:
        folder = paw_library_path(xc="pbe", must_exist=False)
    except LibraryPathError as error:
        pytest.skip(str(error))
    if not os.path.isdir(folder):
        pytest.skip(f"no PBE PAW-LCAO set at {folder}")
    return folder


class TestPBELibrary:
    """The shipped scalar-relativistic PBE set, ``pbe-sr/``."""

    def test_every_element_is_present(self):
        assert set(available_elements(_pbe_library())) == set(
            library_elements())

    @pytest.mark.parametrize("symbol", ["H", "Li", "O", "Na", "Fe", "Cu",
                                        "Ag", "Pb"])
    def test_a_sample_loads_as_pbe_without_ghosts(self, symbol):
        """Selected by functional, read from ``pbe-sr/``, recorded as a
        scalar-relativistic PBE dataset with relativistic exchange, and its
        generalized spectrum holds no level below any channel's reference."""
        from mandacaru.pseudopotentials.partial_waves import ghost_errors
        from mandacaru.pseudopotentials.paw import _paw_levels

        folder = _pbe_library()
        pp = get_paw(symbol, xc="pbe")
        assert os.path.dirname(pp.source) == os.path.realpath(folder)
        assert (pp.xc, pp.relativity) == ("pbe", "scalar")
        assert pp.relativistic_exchange
        assert not (pp.defects or {}).get("ghosts")
        assert ghost_errors(pp, _paw_levels, with_atom=False) == {}

    @pytest.mark.parametrize("symbol, n", [("Sc", 4), ("Zr", 5), ("Ta", 6)])
    def test_the_early_d_block_carries_its_semicore_shell(self, symbol, n):
        """Without the (n-1)s (n-1)p shell the local potential scattered the
        empty p of Sc, Ti, Y-Tc and Ta 0.2-1.5 rad off, PBE ScH came out 11 %
        long and bcc Mo 12 %; with it they build clean and within 0.7 %
        (HISTORY.md, 2026-10-11)."""
        _pbe_library()
        pp = get_paw(symbol, xc="pbe")
        assert pp.semicore_subshells == ((n - 1, 0), (n - 1, 1))
        assert 1 in pp.channels and not pp.defects

    @pytest.mark.parametrize("symbol", ["Sc", "Ti", "Ga", "Ge"])
    def test_a_filtered_3d_function_projects_onto_its_own_wave(self, symbol):
        """The filtered (grid_spacing = 0.20 Angstrom) 3d basis function of a single
        atom projects onto its own first d projector as 1 and onto the
        second as 0, to 0.1.  The semicore Sc and Ti at their default d
        spheres were off by 0.21 / 0.35, the PBE Ga and Ge without the d
        repair by 0.47 / 1.27 (HISTORY.md, 2026-10-11)."""
        from ase import Atoms

        from mandacaru.algorithms._hamiltonian_from_atoms import (
            build_basis_hamiltonian)

        folder = _pbe_library()
        atoms = Atoms(symbol, positions=[[3.5, 3.5, 3.5]], cell=[7.0] * 3)
        _H, _p, _n, _profile, context = build_basis_hamiltonian(
            atoms, {"name": "PAW-LCAO", "size": "SZ",
                    "projector_basis": "dual", "paw_dataset": folder},
            None, 0.20, 0, None, spin=True)
        integrals = context["integrals"]
        C = np.asarray(integrals.projections())
        columns = {(q.m, q.index): k for k, q in
                   enumerate(integrals.kb_projectors) if q.l == 2}
        row = int(np.argmax(np.abs(C[:, columns[0, 0]])))
        assert C[row, columns[0, 0]].real == pytest.approx(1.0, abs=0.1)
        assert abs(C[row, columns[0, 1]]) < 0.1

    def test_the_functional_selects_the_set(self):
        """``xc`` picks the folder; the default stays the LDA set."""
        folder = _pbe_library()
        assert paw_library_path(xc="pbe") == folder
        assert os.path.basename(paw_library_path()) == "lda-sr"
        lda, pbe = get_paw("O"), get_paw("O", xc="pbe")
        assert (lda.xc, pbe.xc) == ("lda", "pbe")
        assert not np.allclose(lda.v_local[:100], pbe.v_local[:100])

    def test_the_calculator_reads_it_by_folder_name(self):
        """``Mandacaru(paw_dataset="pbe-sr")`` puts the PBE datasets under a
        PBE Kohn-Sham calculation without the mismatch warning."""
        import warnings

        from ase import Atoms

        from mandacaru import Mandacaru

        _pbe_library()
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.75]],
                      cell=[6.0] * 3)
        atoms.center()
        atoms.calc = Mandacaru(method="dft", xc="pbe", grid_spacing=0.3,
                               basis={"name": "PAW-LCAO", "size": "SZ"},
                               paw_dataset="pbe-sr")
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            energy = atoms.get_potential_energy()
        assert np.isfinite(energy)
        integrals = atoms.calc.solver._gradient_context["integrals"]
        assert {d.xc for d in integrals.pseudopotentials} == {"pbe"}


def _pbe_dirac_library():
    """``$MANDACARU_PAW_PATH/pbe-fr``, or a skip when the checkout has no
    PBE spin-orbit set (an older ``mandacaru-paw``)."""
    from mandacaru.pseudopotentials.environment import (LibraryPathError,
                                                        library_folder)

    try:
        folder = library_folder("paw-lcao", "pbe-fr")
    except (LibraryPathError, FileNotFoundError, ValueError) as error:
        pytest.skip(str(error))
    if not os.path.isdir(folder):
        pytest.skip(f"no PBE spin-orbit PAW-LCAO set at {folder}")
    return folder


class TestPBEDiracLibrary:
    """The shipped PBE set with the spin-orbit term, ``pbe-fr/``."""

    def test_every_element_is_present(self):
        assert set(available_elements(_pbe_dirac_library())) == set(
            library_elements())

    @pytest.mark.parametrize("symbol", ["H", "O", "Ti", "Au", "Pb", "Bi",
                                        "Pa", "U"])
    def test_a_sample_loads_as_dirac_pbe_without_ghosts(self, symbol):
        """Recorded as a PBE dataset with the Dirac treatment and the
        relativistic exchange, a spin-orbit term for every channel with
        l >= 1 (the j = l - 1/2 reference not above the j = l + 1/2 one),
        and no ghost."""
        from mandacaru.pseudopotentials.paw import _paw_levels
        from mandacaru.pseudopotentials.partial_waves import ghost_errors

        folder = _pbe_dirac_library()
        pp = get_paw(symbol, paw_dataset=folder)
        assert os.path.dirname(pp.source) == os.path.realpath(folder)
        assert (pp.xc, pp.relativity) == ("pbe", "dirac")
        assert pp.relativistic_exchange
        assert set(pp.spin_orbit) == {l for l in pp.channels if l >= 1}
        for l, term in pp.spin_orbit.items():
            down, up = term["reference_energies"]
            assert down[0] <= up[0]
        assert not (pp.defects or {}).get("ghosts")
        assert ghost_errors(pp, _paw_levels, with_atom=False) == {}

    def test_protactinium_is_built_on_bound_branches(self):
        """One 5f branch of Pa's scalar configuration is unbound in the Dirac
        atom; ``pbe-fr/`` builds it on 5f2 6d1 7s2, with a d channel and both
        5f branches bound."""
        pp = get_paw("Pa", paw_dataset=_pbe_dirac_library())
        assert sorted(pp.channels) == [0, 2, 3]
        assert all(energy[0] < 0.0
                   for energy in pp.spin_orbit[3]["reference_energies"])

    def test_lda_fr_holds_protactinium_too(self):
        """``lda-fr/`` has all 92 elements since Pa was added on the same
        reference (2026-10-14)."""
        from mandacaru.pseudopotentials.environment import (LibraryPathError,
                                                            library_folder)

        try:
            folder = library_folder("paw-lcao", "lda-fr")
        except (LibraryPathError, FileNotFoundError, ValueError) as error:
            pytest.skip(str(error))
        if not os.path.isdir(folder):
            pytest.skip(f"no spin-orbit LDA set at {folder}")
        assert set(available_elements(folder)) == set(library_elements())
        pp = get_paw("Pa", paw_dataset=folder)
        assert (pp.xc, pp.relativity) == ("lda", "dirac")
        assert sorted(pp.channels) == [0, 2, 3] and not pp.defects
        assert all(energy[0] < 0.0
                   for energy in pp.spin_orbit[3]["reference_energies"])

    def test_the_set_is_not_the_scalar_one(self):
        """Same functional, different folder: the spin-orbit term is the
        difference, and a Kohn-Sham calculation still reads ``pbe-sr/``."""
        folder = _pbe_dirac_library()
        scalar, dirac = get_paw("Au", xc="pbe"), get_paw("Au", paw_dataset=folder)
        assert (scalar.relativity, dirac.relativity) == ("scalar", "dirac")
        assert not scalar.spin_orbit and dirac.spin_orbit
        assert paw_library_path(xc="pbe", relativity="dirac").endswith("pbe-fr")


class TestLibraryLoader:
    """``load_library_dataset`` behind ``get_paw``: the folder's name is a
    promise about what is in it, and the cache follows the file."""

    def _copy(self, tmp_path, folder, symbol="H"):
        import shutil

        target = tmp_path / folder
        target.mkdir()
        shutil.copy(library_file(symbol, paw_library_path()),
                    target / f"{symbol}.parquet")
        return target

    def test_a_scalar_dataset_in_a_dirac_folder_is_refused(self, tmp_path):
        """A scalar file copied into ``lda-fr/`` would silently drop the
        spin-orbit term that folder promises."""
        folder = self._copy(tmp_path, "lda-fr")
        with pytest.raises(ValueError, match="relativity='scalar' in the "
                                             "lda-fr/ folder"):
            get_paw("H", paw_dataset=str(folder))

    def test_a_matching_folder_and_a_folder_of_ones_own_load(self, tmp_path):
        assert get_paw("H", paw_dataset=str(self._copy(
            tmp_path, "lda-sr"))).relativity == "scalar"
        assert get_paw("H", paw_dataset=str(self._copy(
            tmp_path, "mine"))).relativity == "scalar"

    def test_a_replaced_file_is_read_again(self, tmp_path):
        """The cache is keyed by the file's identity: a dataset rebuilt under
        a running session is not served from memory."""
        from dataclasses import replace

        folder = self._copy(tmp_path, "lda-sr")
        first = get_paw("H", paw_dataset=str(folder))
        assert get_paw("H", paw_dataset=str(folder)) is first
        path = save_pseudopotential(
            replace(first, one_center_energy=first.one_center_energy + 1.0),
            folder / "H.parquet")
        stamp = os.stat(path).st_mtime_ns + 10 ** 9
        os.utime(path, ns=(stamp, stamp))
        second = get_paw("H", paw_dataset=str(folder))
        assert second is not first
        assert second.one_center_energy == pytest.approx(
            first.one_center_energy + 1.0)
