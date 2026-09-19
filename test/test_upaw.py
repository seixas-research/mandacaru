# -*- coding: utf-8 -*-
# file: test/test_upaw.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The unitary PAW family (``basis="UPAW"``), Ivanov et al. arXiv:2408.03159.

UPAW is PAW with the orthonormality constraint ``O_ij = <phi_i|phi_j> -
<phi~_i|phi~_j> = 0``, i.e. Carcará's ``norm_deficit = 0``: the transformation
is unitary, so the pseudo states are orthonormal and the overlap operator is the
identity.  It is an **option**, not the default -- the measurements that decided
that are in ``smooth_partial_waves``' docstring and are pinned below, because
they are the reason a future reader should not promote it.
"""

import warnings

import numpy as np
import pytest
from ase import Atoms

from carcara import Carcara
from carcara.pseudopotentials.families import family_names, resolve_family
from carcara.pseudopotentials.paw import (UPAW_FAMILY, build_upaw_library,
                                          generate_paw, generate_upaw, get_upaw)

CELL = 10.0
CENTRE = CELL / 2 + 0.011


def h2(distance=0.74):
    return Atoms("H2", positions=[[CENTRE, CENTRE, CENTRE - distance / 2],
                                  [CENTRE, CENTRE, CENTRE + distance / 2]],
                 cell=[CELL] * 3)


@pytest.fixture(scope="module")
def hydrogen():
    """One UPAW dataset, generated once (a few tenths of a second)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return generate_upaw("H")


class TestUnitarity:
    """What makes it *unitary*: the overlap correction is zero."""

    def test_the_overlap_correction_vanishes(self, hydrogen):
        for channel in hydrogen.channels.values():
            q = np.asarray(channel.overlap_correction, dtype=float)
            assert np.abs(q).max() < 1e-9, q

    def test_the_overlap_operator_is_the_identity(self, hydrogen):
        # 1 + sum |p> q <p| with q = 0.  PAW's is bounded below by 1 + eps.
        for channel in hydrogen.channels.values():
            assert channel.overlap_minimum == pytest.approx(1.0, abs=1e-9)

    def test_paw_keeps_a_nonzero_correction(self):
        """The contrast: the default family deliberately gives up some norm."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            paw = generate_paw("H")
        q = max(np.abs(np.asarray(c.overlap_correction)).max()
                for c in paw.channels.values())
        assert q > 1e-3

    def test_the_dataset_is_still_sound(self, hydrogen):
        """Unitarity must not buy a ghost state or a broken projector."""
        from carcara.pseudopotentials.paw import check_paw_channel

        report = check_paw_channel(hydrogen, 0)
        assert abs(report["eigenvalue_error"]) < 1e-6
        assert report["duality_error"] < 1e-8
        assert report["nodes"] == 0

    def test_the_molecular_overlap_is_unaugmented(self):
        """`S = S~ + C q C†` collapses to the plain smooth overlap."""
        from carcara.algorithms._hamiltonian_from_atoms import \
            build_basis_hamiltonian
        from carcara.integrals import Grid

        grid = Grid(center=[CENTRE] * 3, box_size=8.0, h=0.3, units="angstrom")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            context = build_basis_hamiltonian(h2(), "UPAW", grid, 0.3, 0, None)[4]
        integrals = context["integrals"]
        assert np.abs(integrals.overlap()
                      - integrals.bare_overlap()).max() < 1e-10


class TestFamilyRegistration:
    def test_it_is_a_basis_name(self):
        assert UPAW_FAMILY in family_names()
        for spelling in ("UPAW", "upaw", "unitary-paw"):
            assert resolve_family(spelling).name == UPAW_FAMILY

    def test_it_is_not_norm_conserving_and_takes_the_paw_options(self):
        spec = resolve_family("UPAW")
        assert spec.norm_conserving is False
        assert {"size", "split_norm", "directory", "projector_basis"} \
            <= set(spec.options)

    def test_the_generator_refuses_a_norm_deficit(self):
        # It *is* the definition of the family.
        with pytest.raises(TypeError, match="norm_deficit"):
            generate_upaw("H", norm_deficit=0.05)

    def test_a_dataset_knows_which_family_it_belongs_to(self, hydrogen):
        assert hydrogen.family == UPAW_FAMILY

    def test_a_paw_file_is_refused_as_upaw(self, tmp_path):
        from carcara.pseudopotentials.io import (library_file,
                                                 save_pseudopotential)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            save_pseudopotential(generate_paw("H"),
                                 library_file("H", str(tmp_path)))
        with pytest.raises(ValueError, match="family"):
            get_upaw("H", str(tmp_path))

    def test_a_missing_dataset_in_a_named_directory_is_not_generated(self,
                                                                     tmp_path):
        # An explicit directory is a statement that the library is there.
        with pytest.raises(FileNotFoundError, match="build_upaw_library"):
            get_upaw("Li", str(tmp_path))


class TestCalculatorPath:
    def test_the_basis_name_runs_end_to_end(self):
        atoms = h2()
        atoms.calc = Carcara(method="adapt-vqe", basis="UPAW", h=0.3,
                             pool="fermionic", max_iterations=8,
                             gradient_tolerance=1e-5, profile=False,
                             trace=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            energy = atoms.get_potential_energy()
        assert np.isfinite(energy) and energy < 0
        assert atoms.calc.n_qubits == 4

    def test_the_dry_run_names_the_family(self):
        estimate = Carcara(method="adapt-vqe",
                           basis={"name": "UPAW", "size": "DZP"}).dry_run(h2())
        assert "UPAW" in estimate.basis
        assert estimate.n_qubits == 20

    def test_it_cannot_be_mixed_with_another_family(self):
        water = Atoms("OH2", positions=[[CENTRE, CENTRE, CENTRE],
                                        [CENTRE, CENTRE + 0.77, CENTRE + 0.59],
                                        [CENTRE, CENTRE - 0.77, CENTRE + 0.59]],
                      cell=[CELL] * 3)
        for spec in ({"H": "UPAW", "O": "PAW"}, {"H": "UPAW", "O": "FAO"}):
            with pytest.raises(ValueError, match="per-element basis"):
                Carcara(method="adapt-vqe", basis=spec).dry_run(water)

    def test_a_built_library_is_used_instead_of_generating(self, tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            written = build_upaw_library(("H",), directory=str(tmp_path),
                                         verbose=False)
        assert len(written) == 1
        # From the library: no generation warning this time.
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            loaded = get_upaw("H", str(tmp_path))
        assert loaded.family == UPAW_FAMILY
        assert max(np.abs(np.asarray(c.overlap_correction)).max()
                   for c in loaded.channels.values()) < 1e-9


class TestWhyItIsNotTheDefault:
    """The measurements behind the choice, so a future reader has them pinned.

    Full comparison in CLAUDE.md; these are the two that matter and are cheap.
    """

    def test_the_constraint_only_fixes_the_monopole(self):
        """UPAW zeroes L = 0 and leaves the higher multipoles -- on oxygen
        L = 2 is *larger* than PAW's, which is why the compensation machinery
        (and its force derivatives) cannot be deleted."""
        from carcara.algorithms._hamiltonian_from_atoms import \
            build_basis_hamiltonian
        from carcara.integrals import Grid

        water = Atoms("OH2", positions=[[CENTRE, CENTRE, CENTRE],
                                        [CENTRE, CENTRE + 0.77, CENTRE + 0.59],
                                        [CENTRE, CENTRE - 0.77, CENTRE + 0.59]],
                      cell=[CELL] * 3)
        grid = Grid(center=[CENTRE] * 3, box_size=8.0, h=0.3, units="angstrom")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            integrals = build_basis_hamiltonian(water, "UPAW", grid, 0.3, 0,
                                                None)[4]["integrals"]
        moments = integrals.compensation_moments()
        by_l: dict[int, float] = {}
        for (_atom, L, _M), block in moments.items():
            by_l[L] = max(by_l.get(L, 0.0),
                          float(np.abs(np.asarray(block)).max()))
        assert by_l[0] < 1e-9                      # the monopole is gone
        assert max(by_l.get(L, 0.0) for L in (1, 2)) > 1e-4   # the rest is not

    def test_paw_is_still_the_default(self):
        from carcara.algorithms._hamiltonian_from_atoms import resolve_basis

        # Nothing here may change what `basis="PAW"` means.
        assert resolve_basis("PAW")[0].lower() == "paw"
        assert resolve_family("PAW").name == "paw"
