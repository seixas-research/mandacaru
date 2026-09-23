# -*- coding: utf-8 -*-
# file: test/test_upaw.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The unitary PAW-LCAO family (``basis="UPAW-LCAO"``), Ivanov et al. arXiv:2408.03159.

UPAW-LCAO is PAW-LCAO with the orthonormality constraint ``O_ij = <phi_i|phi_j> -
<phi~_i|phi~_j> = 0``, i.e. Mandacaru's ``norm_deficit = 0``: the transformation
is unitary, so the pseudo states are orthonormal and the overlap operator is the
identity.  It is an **option**, not the default -- the measurements that decided
that are in ``smooth_partial_waves``' docstring and are pinned below, because
they are the reason a future reader should not promote it.
"""

import warnings

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.pseudopotentials.families import family_names, resolve_family
from mandacaru.pseudopotentials.paw import (UPAW_FAMILY, build_upaw_library,
                                            generate_paw, generate_upaw, get_upaw)

CELL = 10.0
CENTER = CELL / 2 + 0.011


def h2(distance=0.74):
    return Atoms("H2", positions=[[CENTER, CENTER, CENTER - distance / 2],
                                  [CENTER, CENTER, CENTER + distance / 2]],
                 cell=[CELL] * 3)


@pytest.fixture(scope="module")
def hydrogen():
    """One UPAW-LCAO dataset, generated once (a few tenths of a second)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return generate_upaw("H")


class TestUnitarity:
    """What makes it *unitary*: the overlap correction is zero.

    **Exactly zero only without relativity**, and that is a statement about
    the construction rather than a tolerance to be loosened.  ``q`` is the
    charge the smooth density is missing, so it is a plain inner product; the
    norm the Vanderbilt condition conserves is the M-weighted one (a
    Wronskian -- :func:`~mandacaru.pseudopotentials.oncv.norm_targets`).  The
    two are the same matrix at ``M = 1`` and differ at ``O(c^-2)`` when they
    are not, so a relativistic UPAW-LCAO dataset cannot be both exactly
    unitary *and* Vanderbilt-consistent.  Measured on hydrogen: ``5.7e-15``
    with ``relativity="none"``, ``6.4e-5`` with the scalar-relativistic
    default.  The physics is kept and the unitarity degrades, because an
    asymmetric coupling is the worse trade.
    """

    def test_the_overlap_correction_vanishes_without_relativity(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = generate_upaw("H", relativity="none", nlcc=False)
        for channel in dataset.channels.values():
            q = np.asarray(channel.overlap_correction, dtype=float)
            assert np.abs(q).max() < 1e-9, q

    def test_relativity_costs_it_only_order_c_squared(self, hydrogen):
        """Small, and the size of the relativistic correction itself."""
        q = max(np.abs(np.asarray(c.overlap_correction)).max()
                for c in hydrogen.channels.values())
        assert 1e-9 < q < 1e-3
        # ... and the default PAW-LCAO deficit stays two orders larger, so the
        # contrast this family exists to make survives.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            paw = generate_paw("H")
        deficit = max(np.abs(np.asarray(c.overlap_correction)).max()
                      for c in paw.channels.values())
        assert deficit > 100 * q

    def test_the_overlap_operator_is_the_identity_without_relativity(self):
        # 1 + sum |p> q <p| with q = 0 exactly.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = generate_upaw("H", relativity="none", nlcc=False)
        for channel in dataset.channels.values():
            assert channel.overlap_minimum == pytest.approx(1.0, abs=1e-9)

    def test_relativity_leaves_it_close_to_the_identity(self, hydrogen):
        """0.99166 for hydrogen, against exactly 1 non-relativistically.

        The 8e-3 is `q` times the projectors' Gram matrix, and `q` is 6.4e-5:
        the projectors are large, so a small correction to the metric is
        visible in its smallest eigenvalue.  What matters is that it stays
        well above :data:`~mandacaru.pseudopotentials.paw.OVERLAP_MINIMUM`,
        which is what the generator refuses below."""
        for channel in hydrogen.channels.values():
            assert 0.98 < channel.overlap_minimum < 1.0

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
        from mandacaru.pseudopotentials.paw import check_paw_channel

        report = check_paw_channel(hydrogen, 0)
        # 5.5e-8 non-relativistically, 1.3e-6 with the scalar-relativistic
        # default.  The degradation is real and is the price of a *smooth,
        # non-relativistic* system reproducing a relativistic reference: UPAW-LCAO
        # sets q = 0 exactly, so there is no overlap correction left to absorb
        # the O(c^-2) difference between the M-weighted norm the condition
        # conserves and the plain inner product.  It is 3.5e-5 eV.
        assert abs(report["eigenvalue_error"]) < 1e-5
        assert report["duality_error"] < 1e-8
        assert report["nodes"] == 0

    def test_the_molecular_overlap_is_unaugmented(self):
        """`S = S~ + C q C†` collapses to the plain smooth overlap.

        Exactly, and therefore only without relativity -- see this class's
        docstring.  A scalar-relativistic UPAW-LCAO dataset carries
        `q = 6.4e-5` and its overlap *is* augmented, by that much.
        """
        from mandacaru.algorithms._hamiltonian_from_atoms import \
            build_basis_hamiltonian
        from mandacaru.integrals import Grid

        grid = Grid(center=[CENTER] * 3, box_size=8.0, h=0.3, units="angstrom")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            context = build_basis_hamiltonian(h2(), "UPAW-LCAO", grid, 0.3, 0, None)[4]
        integrals = context["integrals"]
        augmentation = np.abs(integrals.overlap()
                              - integrals.bare_overlap()).max()
        # Bounded by q times the projections, and q is 6.4e-5 here.  It was
        # < 1e-10 while the datasets were non-relativistic; the bound below
        # is what the scalar-relativistic default leaves, and it is still
        # three orders under what PAW-LCAO augments by.
        assert augmentation < 1e-3


class TestFamilyRegistration:
    def test_it_is_a_basis_name(self):
        assert UPAW_FAMILY in family_names()
        for spelling in ("UPAW-LCAO", "upaw-lcao", "unitary-paw-lcao"):
            assert resolve_family(spelling).name == UPAW_FAMILY

    def test_it_is_not_norm_conserving_and_takes_the_paw_options(self):
        spec = resolve_family("UPAW-LCAO")
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
        from mandacaru.pseudopotentials.io import (library_file,
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
        atoms.calc = Mandacaru(method="adapt-vqe", basis="UPAW-LCAO", h=0.3,
                               pool="fermionic", max_iterations=8,
                               gradient_tolerance=1e-5, profile=False,
                               trace=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            energy = atoms.get_potential_energy()
        assert np.isfinite(energy) and energy < 0
        assert atoms.calc.n_qubits == 4

    def test_the_dry_run_names_the_family(self):
        estimate = Mandacaru(method="adapt-vqe",
                             basis={"name": "UPAW-LCAO", "size": "DZP"}).dry_run(h2())
        assert "UPAW-LCAO" in estimate.basis
        assert estimate.n_qubits == 20

    def test_it_cannot_be_mixed_with_another_family(self):
        water = Atoms("OH2", positions=[[CENTER, CENTER, CENTER],
                                        [CENTER, CENTER + 0.77, CENTER + 0.59],
                                        [CENTER, CENTER - 0.77, CENTER + 0.59]],
                      cell=[CELL] * 3)
        for spec in ({"H": "UPAW-LCAO", "O": "PAW-LCAO"}, {"H": "UPAW-LCAO", "O": "HAO"}):
            with pytest.raises(ValueError, match="per-element basis"):
                Mandacaru(method="adapt-vqe", basis=spec).dry_run(water)

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
                   for c in loaded.channels.values()) < 1e-3


@pytest.mark.slow
class TestWhyItIsNotTheDefault:
    """The measurements behind the choice, so a future reader has them pinned.

    These are the two that matter and are cheap.
    """

    def test_the_constraint_only_fixes_the_monopole(self):
        """UPAW-LCAO zeroes L = 0 and leaves the higher multipoles -- on oxygen
        L = 2 is *larger* than PAW-LCAO's, which is why the compensation machinery
        (and its force derivatives) cannot be deleted."""
        from mandacaru.algorithms._hamiltonian_from_atoms import \
            build_basis_hamiltonian
        from mandacaru.integrals import Grid

        water = Atoms("OH2", positions=[[CENTER, CENTER, CENTER],
                                        [CENTER, CENTER + 0.77, CENTER + 0.59],
                                        [CENTER, CENTER - 0.77, CENTER + 0.59]],
                      cell=[CELL] * 3)
        grid = Grid(center=[CENTER] * 3, box_size=8.0, h=0.3, units="angstrom")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            integrals = build_basis_hamiltonian(water, "UPAW-LCAO", grid, 0.3, 0,
                                                None)[4]["integrals"]
        moments = integrals.compensation_moments()
        by_l: dict[int, float] = {}
        for (_atom, L, _M), block in moments.items():
            by_l[L] = max(by_l.get(L, 0.0),
                          float(np.abs(np.asarray(block)).max()))
        # The monopole is what the constraint removes -- but it removes it
        # from the norm the constraint is written in.  `norm_deficit = 0` is
        # imposed on the conserved (M-weighted) norm, while the compensation
        # monopole is built from the plain charge, and the two part company at
        # O(c^-2) once the reference atom is relativistic.  Measured here:
        # 2.5e-5 under the scalar-relativistic default, against < 1e-9 while
        # the datasets were non-relativistic.  The exact zero is pinned on a
        # non-relativistic dataset in `TestWhatMakesItUnitary`; what belongs
        # here is that it is still far below the higher multipoles, so the
        # monopole is suppressed rather than merely small: measured, 2.5e-5
        # against 2.1e-2, a factor of 852.
        assert by_l[0] < 1e-4
        rest = max(by_l.get(L, 0.0) for L in (1, 2))
        assert rest > 1e-4 and by_l[0] < rest / 100.0

    def test_paw_is_still_the_default(self):
        from mandacaru.algorithms._hamiltonian_from_atoms import resolve_basis

        # Nothing here may change what `basis="PAW-LCAO"` means.
        assert resolve_basis("PAW-LCAO")[0].lower() == "paw-lcao"
        assert resolve_family("PAW-LCAO").name == "paw-lcao"
