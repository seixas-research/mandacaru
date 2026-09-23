# -*- coding: utf-8 -*-
# file: test/core/test_symmetry.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Space groups and the irreducible Brillouin zone: :mod:`mandacaru.core.symmetry`.

spglib supplies the symmetry *operations*; the mesh reduction is done here, in
Mandacaru's own ``[-1/2, 1/2)`` Gamma-centered convention.  These tests pin that
split -- the reduction has to be right for lattices spglib would describe with a
different grid convention, and it has to refuse to merge points that are not
actually in the mesh.

Nothing here builds a Hamiltonian, so the file costs a fraction of a second.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.dft.kpoints import monkhorst_pack

from mandacaru.core.symmetry import (IrreducibleZone, SymmetryInfo,
                                     crystal_symmetry, irreducible_kpoints)


def mesh(size):
    offset = np.array([0.5 / n if n % 2 == 0 else 0.0 for n in size])
    return monkhorst_pack(size) + offset


def square(a=2.5, vacuum=12.0):
    return Atoms("H", positions=[[0, 0, 0]],
                 cell=[[a, 0, 0], [0, a, 0], [0, 0, vacuum]],
                 pbc=[True, True, False])


def hexagonal(a=2.5, vacuum=12.0):
    return Atoms("H", positions=[[0, 0, 0]],
                 cell=[[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0],
                       [0, 0, vacuum]],
                 pbc=[True, True, False])


def fcc(a=4.0):
    cell = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    return Atoms("Cu", positions=[[0, 0, 0]], cell=cell, pbc=True)


class TestTheSpaceGroup:
    @pytest.mark.parametrize("atoms, symbol, number", [
        (square(), "P4/mmm", 123),
        (hexagonal(), "P6/mmm", 191),
        (fcc(), "Fm-3m", 225),
    ])
    def test_it_is_recognized(self, atoms, symbol, number):
        info = crystal_symmetry(atoms)
        assert info.international == symbol
        assert info.number == number

    def test_a_slab_reports_the_three_dimensional_group(self):
        """Vacuum makes a 2-D lattice a tall 3-D one, and that is correct.

        The basis functions really are lattice-summed in all three directions,
        so the operations that act on the mesh are the 3-D ones.
        """
        info = crystal_symmetry(square())
        assert info.n_operations == 16

    def test_the_rotations_are_integer_matrices_of_determinant_plus_or_minus_one(self):
        info = crystal_symmetry(fcc())
        assert info.rotations.dtype.kind in "iu"
        dets = np.linalg.det(info.rotations.astype(float))
        assert np.all(np.abs(np.abs(dets) - 1.0) < 1e-9)

    def test_symprec_is_recorded_because_the_answer_depends_on_it(self):
        assert crystal_symmetry(square(), symprec=1e-3).symprec == 1e-3

    def test_the_summary_names_the_group(self):
        assert "P4/mmm" in crystal_symmetry(square()).summary()


class TestSpglibIsConfiguredToRaise:
    """``require_spglib`` turns off spglib's legacy error handling.

    With ``spglib.error.OLD_ERROR_HANDLING`` left at its default ``True``,
    spglib stashes a failure in a module-level string, returns a sentinel, and
    emits a ``DeprecationWarning`` on **every** call -- which made three
    symmetry lookups produce three warnings.  spglib says the flag will be
    removed, so Mandacaru opts in once, at the single point where spglib is
    imported.
    """

    def test_no_deprecation_warning_per_call(self):
        import warnings

        atoms = square()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(3):
                crystal_symmetry(atoms)
        deprecations = [record for record in caught
                        if issubclass(record.category, DeprecationWarning)]
        assert deprecations == []

    def test_the_flag_is_off_after_importing(self):
        from mandacaru.core.symmetry import require_spglib

        require_spglib()
        from spglib import error as spglib_error
        assert spglib_error.OLD_ERROR_HANDLING is False

    def test_a_failure_raises_instead_of_returning_a_sentinel(self):
        """A degenerate cell must fail loudly, not come back as ``None``."""
        atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]],
                      cell=np.zeros((3, 3)), pbc=True)
        with pytest.raises(Exception) as excinfo:
            crystal_symmetry(atoms)
        assert "failed" in str(excinfo.value).lower()


class TestTheIrreducibleZone:
    def test_the_weights_sum_to_the_mesh(self):
        zone = irreducible_kpoints(mesh((4, 4, 1)), crystal_symmetry(square()))
        assert int(zone.weights.sum()) == 16
        assert zone.n_kpoints == 16

    def test_a_square_two_by_two_mesh_reduces_to_three(self):
        """Gamma, the X pair and M: the textbook wedge of a square lattice."""
        zone = irreducible_kpoints(mesh((2, 2, 1)), crystal_symmetry(square()))
        assert len(zone.points) == 3
        assert sorted(zone.weights.tolist()) == [1, 1, 2]

    def test_every_representative_is_a_mesh_point(self):
        full = mesh((4, 4, 1))
        zone = irreducible_kpoints(full, crystal_symmetry(square()))
        for point, index in zip(zone.points, zone.indices):
            assert point == pytest.approx(full[index], abs=1e-12)

    def test_expand_spreads_a_per_wedge_quantity(self):
        full = mesh((2, 2, 1))
        zone = irreducible_kpoints(full, crystal_symmetry(square()))
        spread = zone.expand(np.arange(len(zone.points)))
        assert spread.shape == (4,)
        assert spread.tolist() == zone.mapping.tolist()

    def test_expand_carries_a_trailing_shape(self):
        """A whole spectrum per k-point expands with the same call."""
        zone = irreducible_kpoints(mesh((2, 2, 1)), crystal_symmetry(square()))
        spectra = np.random.default_rng(0).normal(size=(len(zone.points), 7))
        assert zone.expand(spectra).shape == (4, 7)

    def test_expand_refuses_the_wrong_length(self):
        zone = irreducible_kpoints(mesh((2, 2, 1)), crystal_symmetry(square()))
        with pytest.raises(ValueError, match="one entry per irreducible"):
            zone.expand(np.zeros(len(zone.points) + 1))

    def test_time_reversal_merges_k_and_minus_k(self):
        """A chain has no point-group operation relating +-k inside the mesh.

        Time reversal is what does it, so switching it off must give strictly
        more representatives.
        """
        chain = Atoms("H", positions=[[0, 0, 0]],
                      cell=np.diag([1.0, 8.0, 8.0]), pbc=[True, False, False])
        full = mesh((6, 1, 1))
        info = crystal_symmetry(chain)
        with_tr = irreducible_kpoints(full, info, time_reversal=True)
        without = irreducible_kpoints(full, info, time_reversal=False)
        assert len(with_tr.points) <= len(without.points)

    def test_the_identity_alone_reduces_nothing(self):
        identity = SymmetryInfo(number=1, international="P1", point_group="1",
                                rotations=np.eye(3, dtype=int)[None],
                                translations=np.zeros((1, 3)), symprec=0.0)
        full = mesh((3, 3, 1))
        zone = irreducible_kpoints(full, identity, time_reversal=False)
        assert len(zone.points) == len(full)
        assert zone.mapping.tolist() == list(range(len(full)))

    def test_it_never_merges_a_point_the_mesh_does_not_contain(self):
        """A 2x1x1 mesh of a square lattice is not closed under C4.

        C4 sends ``(1/2, 0, 0)`` to ``(0, 1/2, 0)``, which is not sampled, so
        no merge may happen -- the reduction must only ever group points that
        are both present.
        """
        full = mesh((2, 1, 1))
        zone = irreducible_kpoints(full, crystal_symmetry(square()))
        assert len(zone.points) == len(full)

    @pytest.mark.parametrize("atoms, size", [
        (hexagonal(), (3, 3, 1)),
        (fcc(), (2, 2, 2)),
        (fcc(), (4, 4, 4)),
    ])
    def test_non_orthogonal_lattices_reduce_consistently(self, atoms, size):
        full = mesh(size)
        zone = irreducible_kpoints(full, crystal_symmetry(atoms))
        assert int(zone.weights.sum()) == len(full)
        assert len(zone.points) <= len(full)
        # Every mapped point must really be in its representative's orbit.
        rotations = zone.symmetry.rotations
        for index, slot in enumerate(zone.mapping):
            star = rotations.transpose(0, 2, 1) @ full[index]
            star = np.concatenate([star, -star])
            folded = star - np.round(star)
            target = zone.points[slot]
            difference = folded - target
            assert np.any(np.all(np.abs(difference - np.round(difference))
                                 < 1e-8, axis=1))

    def test_a_malformed_mesh_is_refused(self):
        with pytest.raises(ValueError, match=r"shape \(nk, 3\)"):
            irreducible_kpoints(np.zeros((4, 2)), crystal_symmetry(square()))

    def test_the_summary_reports_the_saving(self):
        zone = irreducible_kpoints(mesh((4, 4, 1)), crystal_symmetry(square()))
        text = zone.summary()
        assert "irreducible" in text and "P4/mmm" in text

    def test_nobody_checked_is_not_the_same_as_zero(self):
        """``symmetry_residual`` starts as None, not 0.0."""
        zone = irreducible_kpoints(mesh((2, 2, 1)), crystal_symmetry(square()))
        assert zone.symmetry_residual is None
        assert isinstance(zone, IrreducibleZone)
