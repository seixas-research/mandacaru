# -*- coding: utf-8 -*-
# file: test/test_pseudopotential_engine.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Pseudopotentials through the integral engine, the drivers and the forces.

The library itself is tested in ``test_pseudopotential.py``; this file covers the
plumbing: the on-disk library, the Kleinman-Bylander projection kernel (C and
NumPy), the valence-only Hamiltonian, the ``pseudopotentials=`` driver argument,
and the force terms.

The headline check is :class:`TestGridPathologyIsCured`: with all-electron
oxygen, the spurious force on an isolated atom *grows* as the grid is refined
(4820 -> 10041 eV/Angstrom from h = 0.20 to 0.10).  With a pseudopotential it
must shrink instead.  That reversal is the whole point of the exercise.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from carcara.algorithms import ADAPTVQE, QuantumCalculator, VQE
from carcara.algorithms._hamiltonian_from_atoms import build_basis_hamiltonian
from carcara.basis.pseudo_io import (LIBRARY_ELEMENTS, available_elements,
                                     default_library_path,
                                     get_pseudopotential, library_file,
                                     load_pseudopotential,
                                     save_pseudopotential)
from carcara.basis.pseudo_orbital import (KBProjector, PseudoAtomicOrbital,
                                          kb_projectors, pseudo_basis,
                                          valence_electrons)
from carcara.integrals import Grid, Potentials, _backend
from carcara.units import BOHR_TO_ANGSTROM

pytestmark = pytest.mark.skipif(
    not available_elements(), reason="pseudopotential library not generated")


def lone_atom(symbol, grid, offset=0.37):
    """An atom placed deliberately off a grid node."""
    shift = offset * grid.dx * BOHR_TO_ANGSTROM
    return Atoms(symbol, positions=[[shift, 0.0, 0.0]])


# --------------------------------------------------------------------------- #
# The library.
# --------------------------------------------------------------------------- #

class TestLibrary:
    def test_ships_the_light_elements(self):
        """Everything with Z < 18 must be available."""
        present = set(available_elements())
        assert set(LIBRARY_ELEMENTS) <= present, \
            set(LIBRARY_ELEMENTS) - present

    @pytest.mark.parametrize("symbol", ["H", "C", "O", "Si", "Cl"])
    def test_loads_with_the_right_valence(self, symbol):
        from ase.data import atomic_numbers

        pp = get_pseudopotential(symbol)
        assert pp.symbol == symbol
        assert pp.atomic_number == atomic_numbers[symbol]
        assert 0 < pp.valence_charge <= pp.atomic_number
        assert np.isfinite(pp.v_local).all()

    def test_core_is_actually_removed(self):
        """Oxygen keeps 6 of its 8 electrons; the 1s pair is gone."""
        assert get_pseudopotential("O").valence_charge == 6.0
        assert get_pseudopotential("Si").valence_charge == 4.0

    def test_round_trip_preserves_the_tables(self, tmp_path):
        original = get_pseudopotential("O")
        path = save_pseudopotential(original, tmp_path / "O.json")
        reloaded = load_pseudopotential(path)

        assert reloaded.valence_charge == original.valence_charge
        assert reloaded.local_l == original.local_l
        assert set(reloaded.channels) == set(original.channels)
        # Saving is lossless: the library was decimated once at generation, and
        # a later save must not decimate again.  See test_pseudo_io.py.
        assert np.allclose(reloaded.r, original.r, rtol=1e-9)
        assert np.allclose(reloaded.v_local, original.v_local, rtol=1e-8)

    def test_rejects_a_foreign_file(self, tmp_path):
        alien = tmp_path / "X.json"
        alien.write_text('{"format": "something-else"}')
        with pytest.raises(ValueError, match="not a Carcará pseudopotential"):
            load_pseudopotential(alien)

    def test_missing_element_is_reported_helpfully(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Regenerate the library"):
            get_pseudopotential("Fe", directory=tmp_path)

    def test_library_lives_where_documented(self):
        assert default_library_path().endswith("pseudos")
        assert library_file("O").endswith("O.parquet")     # Parquet by default


# --------------------------------------------------------------------------- #
# Kleinman-Bylander projection kernel.
# --------------------------------------------------------------------------- #

class TestProjectionKernel:
    def test_c_backend_matches_numpy(self):
        rng = np.random.default_rng(0)
        psi = rng.normal(size=(5, 400)) + 1j * rng.normal(size=(5, 400))
        chi = rng.normal(size=(3, 400)) + 1j * rng.normal(size=(3, 400))
        got = _backend.kb_projections(psi, chi, 0.017)
        expected = (np.conj(psi) @ chi.T) * 0.017
        assert np.allclose(got, expected, atol=1e-12)

    def test_handles_no_projectors(self):
        psi = np.ones((3, 10), dtype=complex)
        assert _backend.kb_projections(psi, np.zeros((0, 10)), 1.0).shape == (3, 0)

    def test_rejects_mismatched_grids(self):
        with pytest.raises(ValueError, match="sampled on"):
            _backend.kb_projections(np.ones((2, 10), dtype=complex),
                                    np.ones((1, 7), dtype=complex), 1.0)


# --------------------------------------------------------------------------- #
# Local potential and basis.
# --------------------------------------------------------------------------- #

class TestLocalPotentialAndBasis:
    def test_local_potential_is_finite_where_coulomb_diverges(self):
        pp = get_pseudopotential("O")
        potentials = Potentials([(pp.valence_charge, np.zeros(3))],
                                pseudopotentials=[pp], units="bohr")
        probe = np.array([1e-4, 0.5, 1.0])
        smooth = potentials.pseudopotential(probe, probe * 0, probe * 0)
        bare = potentials.nuclear_potential(probe, probe * 0, probe * 0)

        assert np.isfinite(smooth).all()
        assert abs(smooth[0]) < 100.0            # bounded at the origin
        assert abs(bare[0]) > 10_000.0           # -Z_ion/r is not

    def test_local_potential_matches_the_ionic_tail_far_out(self):
        pp = get_pseudopotential("O")
        potentials = Potentials([(pp.valence_charge, np.zeros(3))],
                                pseudopotentials=[pp], units="bohr")
        probe = np.array([15.0])
        smooth = potentials.pseudopotential(probe, probe * 0, probe * 0)
        # ~0.1 % residual: the unscreening subtracts the valence Hartree term,
        # whose tail is not perfectly enclosed by the generation grid.
        assert smooth[0] == pytest.approx(-pp.valence_charge / 15.0, rel=5e-3)

    def test_requires_pseudopotentials(self):
        potentials = Potentials([(6.0, np.zeros(3))], units="bohr")
        with pytest.raises(ValueError, match="no pseudopotentials"):
            potentials.pseudopotential(np.array([1.0]), np.array([0.0]),
                                       np.array([0.0]))

    def test_valence_basis_size(self):
        """O contributes s + 3p; each H one s."""
        potentials = {s: get_pseudopotential(s) for s in ("O", "H")}
        functions, owners = pseudo_basis(["O", "H", "H"],
                                         np.zeros((3, 3)), potentials)
        assert len(functions) == 4 + 1 + 1
        assert owners == [0, 0, 0, 0, 1, 2]
        assert all(isinstance(f, PseudoAtomicOrbital) for f in functions)
        assert valence_electrons(["O", "H", "H"], potentials) == 8.0

    def test_projectors_carry_their_atom_and_energy(self):
        potentials = {"O": get_pseudopotential("O")}
        projectors = kb_projectors(["O"], np.zeros((1, 3)), potentials)
        assert projectors and all(isinstance(p, KBProjector) for p in projectors)
        for projector in projectors:
            assert projector.atom_index == 0
            assert np.isfinite(projector.kb_energy)

    def test_orbitals_are_normalizable_and_decay(self):
        pp = get_pseudopotential("O")
        orbital = PseudoAtomicOrbital(pp, 1, 0, center=[0, 0, 0], units="bohr")
        assert orbital.radial(np.array([100.0]))[0] == 0.0
        assert abs(orbital.evaluate(np.array([0.8]), np.array([0.0]),
                                    np.array([0.3]))[0]) > 0


# --------------------------------------------------------------------------- #
# Hamiltonian assembly and the driver argument.
# --------------------------------------------------------------------------- #

class TestHamiltonianAndDrivers:
    def test_valence_only_hamiltonian(self):
        water = Atoms("OH2", positions=[[0, 0, 0], [0, 0.76, 0.59],
                                        [0, -0.76, 0.59]])
        grid = Grid(center=water.get_positions().mean(axis=0), box_size=8.0,
                    h=0.30)
        _h, num_particles, n_orbitals, _profile, context = \
            build_basis_hamiltonian(water, "FAO", grid, 0.30, 0, None,
                                    pseudopotentials=True)
        assert n_orbitals == 6                      # O(s+3p) + 2 H(s)
        assert context["n_electrons"] == 8          # 6 + 1 + 1 valence
        assert num_particles == (4, 4)
        assert context["kb_projectors"]

    def test_kb_matrix_is_hermitian_and_low_rank(self):
        atoms = Atoms("O", positions=[[0, 0, 0]])
        grid = Grid(center=[0, 0, 0], box_size=6.0, h=0.30)
        *_rest, context = build_basis_hamiltonian(atoms, "FAO", grid, 0.30, 0,
                                                  None, pseudopotentials=True)
        integrals = context["integrals"]
        nonlocal_matrix = integrals.kb_nonlocal()

        assert np.allclose(nonlocal_matrix, nonlocal_matrix.conj().T)
        # Rank is bounded by the number of projectors -- that is the whole point
        # of the separable form.
        rank = np.linalg.matrix_rank(nonlocal_matrix, tol=1e-8)
        assert rank <= len(context["kb_projectors"])

    def test_engine_flags_the_pseudopotential_path(self):
        atoms = Atoms("O", positions=[[0, 0, 0]])
        grid = Grid(center=[0, 0, 0], box_size=6.0, h=0.30)
        *_rest, context = build_basis_hamiltonian(atoms, "FAO", grid, 0.30, 0,
                                                  None, pseudopotentials=True)
        assert context["integrals"].uses_pseudopotentials

    @pytest.mark.parametrize("driver", [VQE, ADAPTVQE])
    def test_driver_accepts_the_argument(self, driver):
        assert driver(pseudopotentials=True).pseudopotentials is True
        assert driver().pseudopotentials is False

    def test_frozen_core_is_rejected_as_redundant(self):
        atoms = Atoms("O", positions=[[0, 0, 0]])
        grid = Grid(center=[0, 0, 0], box_size=6.0, h=0.30)
        with pytest.raises(ValueError, match="redundant with pseudopotentials"):
            build_basis_hamiltonian(atoms, "FAO", grid, 0.30, 0, None,
                                    frozen_core=True, pseudopotentials=True)

    def test_end_to_end_energy(self):
        """H2 with pseudopotentials: two valence electrons, four qubits."""
        atoms = Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]])
        grid = Grid(center=[0, 0, 0], box_size=6.0, h=0.20)
        atoms.calc = VQE(basis="FAO", grid=grid, pseudopotentials=True,
                         verbose=False)
        energy = atoms.get_potential_energy()
        assert np.isfinite(energy)
        assert atoms.calc.n_qubits == 4


# --------------------------------------------------------------------------- #
# Forces.
# --------------------------------------------------------------------------- #

class TestPseudopotentialForces:
    def test_forces_are_finite_and_balanced(self):
        atoms = Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]])
        grid = Grid(center=[0, 0, 0], box_size=6.0, h=0.20)
        atoms.calc = QuantumCalculator(driver="vqe", basis="FAO", grid=grid,
                                       pseudopotentials=True)
        forces = atoms.get_forces()
        assert np.isfinite(forces).all()
        # Newton's third law on a two-atom molecule.
        assert np.abs(forces.sum(axis=0)).max() < 0.05 * np.abs(forces).max()

    def test_force_breakdown_is_available(self):
        atoms = Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]])
        grid = Grid(center=[0, 0, 0], box_size=6.0, h=0.20)
        atoms.calc = QuantumCalculator(driver="vqe", basis="FAO", grid=grid,
                                       pseudopotentials=True)
        atoms.get_forces()
        local, pulay = atoms.calc.get_force_breakdown()
        assert local.shape == (2, 3) and pulay.shape == (2, 3)


class TestGridPathologyIsCured:
    """The reversal that motivated the whole pseudopotential effort.

    An isolated atom feels no force by symmetry.  All-electron oxygen gets that
    wrong by ~5000 eV/Angstrom, and *worse* as the grid is refined, because the
    1s core (length scale a0/Z = 0.066 A) is never resolved.  With the core
    removed the error must shrink instead.
    """

    @staticmethod
    def _isolated_force(spacing, pseudopotentials):
        grid = Grid(center=[0, 0, 0], box_size=6.0, h=spacing)
        atoms = lone_atom("O", grid)
        atoms.calc = QuantumCalculator(
            driver="adapt-vqe", basis="FAO", grid=grid,
            pseudopotentials=pseudopotentials,
            frozen_core=not pseudopotentials, pool="qeb",
            max_iterations=6, gradient_tolerance=1e-3, profile=False)
        atoms.get_potential_energy()
        return float(np.abs(atoms.get_forces()).max())

    def test_converges_with_grid_refinement(self):
        """The sign of the trend is what changed: refinement now helps."""
        coarse = self._isolated_force(0.20, pseudopotentials=True)
        fine = self._isolated_force(0.15, pseudopotentials=True)
        assert fine < coarse, f"{coarse:.1f} -> {fine:.1f}"

    def test_far_smaller_than_all_electron(self):
        pseudo = self._isolated_force(0.15, pseudopotentials=True)
        all_electron = self._isolated_force(0.15, pseudopotentials=False)
        assert pseudo < all_electron / 20.0, (pseudo, all_electron)


# --------------------------------------------------------------------------- #
# The basis argument under pseudopotentials.
# --------------------------------------------------------------------------- #

class TestPseudoBasisSize:
    """`size` selects the zeta hierarchy; `basis` is no longer ignored."""

    @staticmethod
    def _water():
        return (["O", "H", "H"],
                np.array([[0, 0, 0], [0, 0.76, 0.59], [0, -0.76, 0.59]]))

    def _basis(self, size):
        from carcara.basis.pseudo_io import get_pseudopotential
        from carcara.basis.pseudo_orbital import pseudo_basis

        symbols, positions = self._water()
        pots = {s: get_pseudopotential(s) for s in set(symbols)}
        return pseudo_basis(symbols, positions, pots, size=size)

    @pytest.mark.parametrize("size, count", [
        ("SZ", 6),          # O(s+p) + 2 H(s)
        ("DZ", 12),         # each doubled
        ("DZP", 23),        # + O d shell (5) + 2 x H p shell (3)
    ])
    def test_function_counts(self, size, count):
        functions, owners = self._basis(size)
        assert len(functions) == count
        assert len(owners) == count

    def test_owners_track_the_atoms(self):
        functions, owners = self._basis("DZP")
        assert set(owners) == {0, 1, 2}
        assert owners == sorted(owners)          # grouped per atom

    def test_single_zeta_path_is_unchanged(self):
        from carcara.basis.pseudo_orbital import PseudoAtomicOrbital

        functions, _ = self._basis("SZ")
        assert all(isinstance(f, PseudoAtomicOrbital) for f in functions)

    def test_first_zeta_comes_from_the_pseudopotential(self):
        """It must be the pseudized orbital the KB projectors were built from."""
        from carcara.basis.pseudo_io import get_pseudopotential

        oxygen = get_pseudopotential("O")
        functions, _ = self._basis("DZP")
        first = next(f for f in functions
                     if f.l == 0 and getattr(f, "zeta", 1) == 1)
        expected = oxygen.channels[0].pseudo_radial
        np.testing.assert_allclose(first.radial(oxygen.r), expected, atol=1e-8)


class TestBasisArgumentIsHonoured:
    """Silently substituting a different basis is worse than refusing."""

    def test_size_is_forwarded(self):
        from carcara.algorithms._hamiltonian_from_atoms import (
            _merge_pseudo_basis_options)

        merged = _merge_pseudo_basis_options({"name": "NAO", "size": "DZP"}, {})
        assert merged["size"] == "DZP"

    def test_the_pseudo_family_is_accepted_by_name(self):
        from carcara.algorithms._hamiltonian_from_atoms import (
            _merge_pseudo_basis_options)

        assert _merge_pseudo_basis_options({"name": "PP", "size": "DZ"},
                                           {})["size"] == "DZ"

    def test_the_default_basis_passes_through(self):
        from carcara.algorithms._hamiltonian_from_atoms import (
            _merge_pseudo_basis_options)

        assert _merge_pseudo_basis_options("FAO", {}) == {}

    @pytest.mark.parametrize("basis", ["6-31G(d)", "GTO",
                                       {"name": "NAO", "energy_shift": 0.03}])
    def test_incompatible_bases_are_refused(self, basis):
        """An all-electron radial function cannot pair with a pseudopotential."""
        from carcara.algorithms._hamiltonian_from_atoms import (
            _merge_pseudo_basis_options)

        with pytest.raises(ValueError, match="cannot be used with pseudopot"):
            _merge_pseudo_basis_options(basis, {})
