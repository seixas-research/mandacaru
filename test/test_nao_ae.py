# -*- coding: utf-8 -*-
# file: test/test_nao_ae.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""All-electron numerical atomic orbitals (NAO-AE).

Pins the construction in :mod:`carcara.basis.nao_ae`: the smooth confining
wall, the hydrogen-like sizing rule, the all-electron minimal basis, the tier
specification, the per-channel Gram-Schmidt, the factory wiring, and the
variational payoff of the tiers at Hartree-Fock.
"""

import numpy as np
import pytest

from carcara.basis import BasisSet, NAOAEBasisSet
from carcara.basis import nao_ae
from carcara.basis.multizeta import TabulatedOrbital
from carcara.basis.nao_ae import (RadialFunction, build_species,
                                  confined_atom, confinement_potential,
                                  effective_charge_for_radius,
                                  hydrogenic_function, hydrogenic_mean_radius,
                                  mean_radius, minimal_functions,
                                  orthonormalize, tier_specification)
from carcara.core import MolecularIntegrals
from carcara.integrals import Grid
from carcara.units import to_bohr

R = np.linspace(0.005, 12.0, 2400)


def _channel_overlaps(functions, l):
    ch = [f for f in functions if f.l == l]
    return np.array([[np.trapezoid(a.u * b.u, a.r) for b in ch] for a in ch])


# --------------------------------------------------------------------------- #
# The wall.
# --------------------------------------------------------------------------- #

class TestConfinementPotential:
    def test_zero_inside_the_onset(self):
        v = confinement_potential(R, 5.0, 2.0)
        assert np.all(v[R <= 5.0] == 0.0)

    def test_rises_monotonically_to_the_cap(self):
        v = confinement_potential(R, 5.0, 2.0)
        ramp = (R > 5.0) & (R < 7.0)
        assert np.all(np.diff(v[ramp]) >= 0.0)
        assert np.all(v[R >= 7.0] == nao_ae.WALL_CAP)
        assert v[ramp].max() <= nao_ae.WALL_CAP

    def test_smooth_at_the_onset(self):
        # Every derivative vanishes at r0: just past it the wall is negligible.
        v = confinement_potential(np.array([5.0 + 0.05]), 5.0, 2.0)
        assert v[0] < 1e-15

    def test_scale_is_linear(self):
        r = np.array([6.0])
        assert confinement_potential(r, 5.0, 2.0, scale=3.0) == pytest.approx(
            3.0 * confinement_potential(r, 5.0, 2.0, scale=1.0))

    def test_invalid_geometry(self):
        with pytest.raises(ValueError):
            confinement_potential(R, 5.0, 0.0)


# --------------------------------------------------------------------------- #
# Hydrogen-like sizing and functions.
# --------------------------------------------------------------------------- #

class TestHydrogenic:
    def test_mean_radius_round_trip(self):
        assert hydrogenic_mean_radius(1, 0, 1.0) == 1.5
        assert hydrogenic_mean_radius(2, 1, 1.0) == 5.0
        for n, l, target in ((2, 1, 1.4), (3, 2, 1.26), (3, 0, 4.0)):
            z = effective_charge_for_radius(n, l, target)
            assert hydrogenic_mean_radius(n, l, z) == pytest.approx(target)

    def test_unconfined_hydrogen_matches_the_analytic_state(self):
        # A far wall leaves the exact 1s untouched (up to the radial grid).
        f = hydrogenic_function(R, 1, 0, 1.0, onset=10.0, width=2.0,
                                tail_norm=None)
        assert f.energy == pytest.approx(-0.5, abs=0.01)
        assert mean_radius(f.r, f.u) == pytest.approx(1.5, rel=0.01)
        assert np.trapezoid(f.u ** 2, f.r) == pytest.approx(1.0, rel=1e-6)

    @pytest.mark.parametrize("n, l", [(2, 0), (3, 1), (3, 0), (4, 2)])
    def test_node_count(self, n, l):
        f = hydrogenic_function(R, n, l, 2.0, onset=8.0, width=2.0,
                                tail_norm=None)
        interior = f.u[(f.r > 0.05) & (f.r < 8.0)]
        interior = interior[np.abs(interior) > 1e-6]
        assert int(np.sum(np.diff(np.sign(interior)) != 0)) == n - l - 1

    def test_strictly_zero_beyond_the_wall(self):
        f = hydrogenic_function(R, 2, 0, 1.0, onset=4.0, width=1.5,
                                tail_norm=None)
        assert np.all(f.u[f.r >= 5.5] == 0.0)
        assert np.any(np.abs(f.u[f.r < 4.0]) > 0.1)

    def test_tail_norm_shortens_compact_functions_only(self):
        compact = hydrogenic_function(R, 3, 2, 8.0, onset=6.0, width=1.0)
        assert compact.onset < 6.0
        assert np.trapezoid(compact.u[compact.r > compact.onset] ** 2,
                            compact.r[compact.r > compact.onset]) < 5e-3
        diffuse = hydrogenic_function(R, 2, 0, 1.0, onset=4.0, width=1.0)
        assert diffuse.onset == 4.0

    def test_quantum_numbers_validated(self):
        with pytest.raises(ValueError):
            hydrogenic_function(R, 1, 1, 1.0, onset=4.0, width=1.0)
        with pytest.raises(ValueError):
            hydrogenic_mean_radius(2, 2, 1.0)


# --------------------------------------------------------------------------- #
# The all-electron minimal basis.
# --------------------------------------------------------------------------- #

class TestMinimalBasis:
    def test_hydrogen_1s_reproduces_the_exact_energy(self):
        atom = confined_atom(1, to_bohr(3.0, "angstrom"),
                             to_bohr(1.0, "angstrom"))
        (f,) = minimal_functions(atom, to_bohr(3.0, "angstrom"),
                                 to_bohr(1.0, "angstrom"))
        assert (f.n, f.l, f.kind) == (1, 0, "minimal")
        du = np.gradient(f.u, f.r)
        energy = 0.5 * np.trapezoid(du ** 2, f.r) - np.trapezoid(f.u ** 2 / f.r, f.r)
        assert energy == pytest.approx(-0.5, abs=0.02)   # radial-grid limited

    def test_core_shells_are_included(self):
        r0, w = to_bohr(3.0, "angstrom"), to_bohr(1.0, "angstrom")
        atom = confined_atom(8, r0, w)
        shells = [(f.n, f.l) for f in minimal_functions(atom, r0, w)]
        assert shells == [(1, 0), (2, 0), (2, 1)]
        for f in minimal_functions(atom, r0, w):
            assert np.trapezoid(f.u ** 2, f.r) == pytest.approx(1.0, rel=1e-6)
            assert np.all(f.u[f.r >= r0 + w] == 0.0)

    def test_confinement_stays_out_of_the_reported_potential(self):
        r0, w = to_bohr(3.0, "angstrom"), to_bohr(1.0, "angstrom")
        confined = confined_atom(8, r0, w)
        assert confined.details["confined"] is True
        # The 1s of oxygen never reaches the wall; only the slightly confined
        # valence density (through the Hartree term) shifts it, by < 1 mHa.
        from carcara.basis.atomic_solver import solve_atom
        free = solve_atom(8)
        assert confined.eigenvalues[(1, 0)] == pytest.approx(
            free.eigenvalues[(1, 0)], abs=5e-3)
        assert np.all(confined.v_effective[confined.r > r0 + w] < 1.0)


# --------------------------------------------------------------------------- #
# Tiers.
# --------------------------------------------------------------------------- #

class TestTiers:
    @staticmethod
    def _atom(Z):
        return confined_atom(Z, to_bohr(3.0, "angstrom"),
                             to_bohr(1.0, "angstrom"))

    def test_tier_zero_is_empty(self):
        assert tier_specification(self._atom(8), 0) == []

    def test_hydrogen_tier_one(self):
        spec = tier_specification(self._atom(1), 1)
        kinds = {(n, l): kind for n, l, _z, kind in spec}
        assert kinds == {(2, 1): "polarization", (2, 0): "diffuse"}

    def test_oxygen_tier_one_and_two(self):
        one = tier_specification(self._atom(8), 1)
        assert [(n, l, k) for n, l, _z, k in one] == [
            (3, 2, "polarization"), (3, 0, "diffuse"), (3, 1, "diffuse")]
        two = tier_specification(self._atom(8), 2)
        assert two[:3] == one
        assert [(n, l, k) for n, l, _z, k in two[3:]] == [
            (4, 3, "polarization"), (4, 2, "polarization"),
            (2, 0, "contracted"), (2, 1, "contracted")]
        assert all(z > 0 for _n, _l, z, _k in two)

    def test_sizes_come_from_the_valence_radius(self):
        atom = self._atom(8)
        radii = nao_ae.valence_radii(atom)
        (n, l, z, _k) = tier_specification(atom, 1)[0]     # the 3d polarization
        assert hydrogenic_mean_radius(n, l, z) == pytest.approx(radii[-1])
        (n, l, z, _k) = tier_specification(atom, 1)[2]     # the diffuse 3p
        assert hydrogenic_mean_radius(n, l, z) == pytest.approx(2.0 * radii[1])

    def test_semicore_d_counts_as_valence(self):
        spec = tier_specification(self._atom(26), 1)      # Fe: 3d6 4s2
        ls = {l for _n, l, _z, _k in spec}
        assert 3 in ls and 2 in ls and 0 in ls              # f polarization

    def test_unknown_tier(self):
        with pytest.raises(ValueError):
            tier_specification(self._atom(1), 3)


# --------------------------------------------------------------------------- #
# Orthonormalization.
# --------------------------------------------------------------------------- #

class TestOrthonormalize:
    @pytest.fixture(scope="class")
    def hydrogen(self):
        return build_species(1, tier=2)

    def test_channels_are_orthonormal(self, hydrogen):
        for l in {f.l for f in hydrogen}:
            S = _channel_overlaps(hydrogen, l)
            assert np.allclose(S, np.eye(len(S)), atol=1e-10)

    def test_minimal_function_leads_and_is_unchanged(self, hydrogen):
        assert hydrogen[0].kind == "minimal"
        r0, w = to_bohr(3.0, "angstrom"), to_bohr(1.0, "angstrom")
        (raw,) = minimal_functions(confined_atom(1, r0, w), r0, w)
        assert np.allclose(hydrogen[0].u, raw.u)

    def test_duplicates_are_rejected(self, hydrogen):
        copy = RadialFunction(r=hydrogen[0].r, u=hydrogen[0].u.copy(), n=9,
                              l=0, kind="custom")
        assert len(orthonormalize(list(hydrogen) + [copy])) == len(hydrogen)
        with pytest.raises(RuntimeError):
            copy.kind = "minimal"
            orthonormalize([hydrogen[0], copy])

    def test_channels_are_independent(self, hydrogen):
        s = [f for f in hydrogen if f.l == 0]
        p = [f for f in hydrogen if f.l == 1]
        assert len(s) >= 2 and len(p) >= 2


# --------------------------------------------------------------------------- #
# The factory / basis set.
# --------------------------------------------------------------------------- #

class TestBasisSet:
    @pytest.mark.parametrize("name", ["NAO-AE", "naoae", "AE-NAO", "nao_ae"])
    def test_names(self, name):
        assert isinstance(BasisSet.build(name), NAOAEBasisSet)

    @pytest.mark.parametrize("symbol, tier, count", [
        ("H", 0, 1), ("H", 1, 5), ("H", 2, 14),
        ("O", 0, 5), ("O", 1, 14), ("O", 2, 30),
        ("Li", 0, 2), ("Li", 1, 6),
    ])
    def test_function_counts(self, symbol, tier, count):
        assert len(BasisSet.build("NAO-AE", tier=tier).atom(symbol)) == count

    def test_default_tier_is_one(self):
        assert BasisSet.build("NAO-AE").tier == 1

    def test_orbitals_are_tabulated_and_confined(self):
        bset = BasisSet.build("NAO-AE", tier=1, onset=2.5, width=1.0)
        orbitals = bset.atom("H", center=[0, 0, 0])
        wall = to_bohr(3.5, "angstrom")
        assert all(isinstance(o, TabulatedOrbital) for o in orbitals)
        assert all(o.r_c == pytest.approx(wall) for o in orbitals)
        x = np.linspace(0.0, 8.0, 400)
        for o in orbitals:
            values = o.evaluate(x, 0.0 * x, 0.0 * x)
            assert np.all(values[x >= wall] == 0.0)
        assert [o.l for o in orbitals] == [0, 1, 1, 1, 0] or \
            sorted(o.l for o in orbitals) == [0, 0, 1, 1, 1]

    def test_polarization_flag(self):
        orbitals = BasisSet.build("NAO-AE", tier=1).atom("O")
        assert {o.l for o in orbitals if o.polarization} == {2}
        assert all(not o.polarization for o in orbitals if o.l < 2)

    def test_normalized_on_a_real_space_grid(self):
        grid = Grid(center=[0, 0, 0], box_size=8.0, h=0.20)
        orbitals = BasisSet.build("NAO-AE", tier=1).atom("H")
        for o in orbitals:
            psi = o.sample(grid)
            assert np.vdot(psi, psi).real * grid.dV == pytest.approx(1.0,
                                                                     rel=0.03)

    def test_centering(self):
        (o,) = BasisSet.build("NAO-AE", tier=0).atom("H", center=[0, 0, 1.0])
        assert np.allclose(o.center, [0, 0, to_bohr(1.0, "angstrom")])

    def test_species_is_cached_and_described(self):
        bset = BasisSet.build("NAO-AE")
        assert bset.species("H") is bset.species(1)
        text = bset.describe("H")
        assert "1s (atomic)" in text and "polarization" in text
        assert "NAOAEBasisSet(tier=1" in repr(bset)

    def test_extra_functions(self):
        bset = BasisSet.build("NAO-AE", tier=0, extra=[(2, 1, 2.5)])
        assert len(bset.atom("H")) == 4
        assert bset.species("H")[-1].kind == "custom"

    def test_wall_must_fit_the_radial_grid(self):
        with pytest.raises(ValueError, match="beyond the radial grid"):
            BasisSet.build("NAO-AE", onset=12.0, width=3.0).atom("H")
        with pytest.raises(ValueError):
            BasisSet.build("NAO-AE", tier=5)

    def test_qubit_estimate_uses_it(self):
        from ase.build import molecule
        from carcara.algorithms import estimate_qubits
        water = molecule("H2O")
        water.center(vacuum=3.0)
        est = estimate_qubits(water, basis={"name": "NAO-AE", "tier": 1})
        assert est.per_atom == [("O", 14), ("H", 5), ("H", 5)]
        assert est.n_qubits == 48

    def test_cannot_be_mixed_with_a_pseudopotential_family(self):
        from carcara.algorithms._hamiltonian_from_atoms import \
            resolve_pseudo_basis
        with pytest.raises(ValueError, match="cannot mix a pseudopotential"):
            resolve_pseudo_basis("per-element",
                                 {"O": {"name": "NAO-AE"}, "H": "NCPP"},
                                 ["O", "H"])


# --------------------------------------------------------------------------- #
# Variational payoff.
# --------------------------------------------------------------------------- #

class TestVariational:
    @staticmethod
    def _rhf(**options):
        grid = Grid(center=[0, 0, 0], box_size=6.0, h=0.20)
        basis = BasisSet.build("NAO-AE", **options)
        functions, nuclei = [], []
        for position in ([0, 0, -0.37], [0, 0, 0.37]):
            functions += basis.atom("H", center=position, units="angstrom")
            nuclei.append((1.0, np.asarray(position)))
        integrals = MolecularIntegrals(
            nuclei, functions, grid,
            softening=0.5 * min(grid.dx, grid.dy, grid.dz))
        result = integrals.hartree_fock(2)
        return result.electronic_energy + integrals.nuclear_repulsion

    def test_tier_one_lowers_the_energy(self):
        minimal = self._rhf(tier=0, onset=2.0, width=0.8)
        tier1 = self._rhf(tier=1, onset=2.0, width=0.8)
        assert tier1 < minimal
        assert minimal - tier1 > 0.02                  # a chemically real gain

    def test_minimal_energy_is_sensible(self):
        # RHF/minimal H2 sits near -1.1 Ha; anything far off means a broken
        # radial function reached the integral engine.
        energy = self._rhf(tier=0, onset=2.0, width=0.8)
        assert -1.20 < energy < -1.00
