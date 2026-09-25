# -*- coding: utf-8 -*-
# file: test/test_paw_energy_shift.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The ``energy_shift`` of a PAW-LCAO basis: confined pseudo-atomic orbitals.

``basis={"name": "PAW-LCAO", "size": "DZP", "energy_shift": 0.1}`` replaces the
free-atom first zeta by the orbital of the atom in GPAW's smooth confining
potential, with the cutoff radius fixed by how far (in eV) the eigenvalue may
rise.  These tests pin the option's contract, the radial solver, the defaults
that must not move, and the molecule path including the forces.
"""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.pseudopotentials import get_paw
from mandacaru.pseudopotentials.confinement import (
    CONFINEMENT_INNER_FRACTION, MIN_ENERGY_SHIFT, confined_energy,
    confined_orbital, confinement_potential, confinement_radius,
    energy_shift_label, energy_shift_of, first_zeta_factory, free_energy,
    validate_energy_shift)
from mandacaru.pseudopotentials.orbitals import pseudo_basis
from mandacaru.units import HARTREE_TO_EV

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _library_present() -> bool:
    try:
        get_paw("H")
    except (FileNotFoundError, ValueError):
        return False
    return True


needs_library = pytest.mark.skipif(not _library_present(),
                                   reason="MANDACARU_PAW_PATH is not configured")


def h2(distance=0.74, cell=8.0):
    # Off the grid nodes on purpose: a symmetric placement hides egg-box terms.
    center = np.array([cell / 2 + 0.013, cell / 2 - 0.021, cell / 2 + 0.007])
    half = np.array([0.0, 0.0, distance / 2])
    return Atoms("H2", positions=[center - half, center + half],
                 cell=[cell] * 3)


class TestOption:
    @pytest.mark.parametrize("spec", [None, False, 0, 0.0])
    def test_off_spellings(self, spec):
        assert validate_energy_shift(spec) is None
        assert energy_shift_label(spec) == "unconfined"

    def test_a_number_is_electronvolts(self):
        assert validate_energy_shift(0.1) == 0.1
        assert energy_shift_label(0.1) == "energy_shift 0.1 eV"

    @pytest.mark.parametrize("spec", [-0.1, MIN_ENERGY_SHIFT / 2, True, "0.1",
                                      float("nan"), float("inf")])
    def test_bad_values_are_refused(self, spec):
        with pytest.raises(ValueError, match="energy_shift"):
            validate_energy_shift(spec)

    def test_per_element_mapping(self):
        spec = {"o": 0.2, "*": 0.1}
        assert energy_shift_of(spec, "O") == 0.2
        assert energy_shift_of(spec, "H") == 0.1
        assert energy_shift_of({"O": 0.2}, "H") is None
        assert validate_energy_shift({"O": None, "H": 0}) is None

    def test_the_factory_is_none_when_off(self):
        assert first_zeta_factory(None) is None
        assert first_zeta_factory({"H": 0}) is None


class TestConfiningPotential:
    def test_shape(self):
        r_c = 6.0
        r = np.linspace(0.01, 7.0, 701)
        v = confinement_potential(r, r_c)
        inner = CONFINEMENT_INNER_FRACTION * r_c
        assert np.all(v[r <= inner] == 0.0)             # the atom is untouched
        inside = (r > inner) & (r < r_c)
        assert np.all(np.diff(v[inside]) > 0.0)         # a monotone wall
        assert np.all(np.isinf(v[r >= r_c]))
        # Flat where it starts: no kink is handed to the orbital.
        assert v[inside][0] < 1e-12


@needs_library
class TestRadialSolver:
    def test_a_distant_wall_recovers_the_free_atom(self):
        pp = get_paw("H")
        assert confined_energy(pp, 0, 25.0) == pytest.approx(
            free_energy(pp, 0), abs=5e-6)

    @pytest.mark.parametrize("symbol, l", [("H", 0), ("O", 0), ("O", 1)])
    def test_the_requested_shift_is_achieved(self, symbol, l):
        orbital = confined_orbital(get_paw(symbol), l, 0.1)
        assert orbital.achieved_shift == pytest.approx(0.1, abs=1e-3)
        assert (orbital.energy - orbital.free_energy) * HARTREE_TO_EV == \
            pytest.approx(orbital.achieved_shift)

    def test_a_larger_shift_is_a_smaller_radius(self):
        pp = get_paw("H")
        radii = [confinement_radius(pp, 0, shift)
                 for shift in (0.01, 0.1, 0.3)]
        assert radii[0] > radii[1] > radii[2] > 2.0
        # GPAW's hydrogen dzp basis at the same 0.1 eV sits at ~6.4 Bohr; the
        # datasets differ, the recipe does not.
        assert radii[1] == pytest.approx(6.68, abs=0.05)

    def test_one_shift_gives_each_orbital_its_own_radius(self):
        oxygen = get_paw("O")
        s, p = (confined_orbital(oxygen, l, 0.1).r_c for l in (0, 1))
        hydrogen = confined_orbital(get_paw("H"), 0, 0.1).r_c
        assert s < p < hydrogen                 # compact 2s, diffuse H 1s

    def test_the_orbital_vanishes_at_its_radius_and_stays_close(self):
        pp = get_paw("H")
        orbital = confined_orbital(pp, 0, 0.1)
        r = np.asarray(pp.r)
        assert np.all(orbital.radial[r >= orbital.r_c] == 0.0)
        # The soft wall takes the orbital to zero smoothly, not with a kink.
        just_inside = (r < orbital.r_c) & (r > orbital.r_c - 0.1)
        assert np.max(np.abs(orbital.radial[just_inside])) < 5e-5
        free = np.asarray(pp.channels[0].pseudo_radial)
        overlap = np.trapezoid(orbital.radial * free * r * r, r) / np.sqrt(
            np.trapezoid(orbital.radial ** 2 * r * r, r)
            * np.trapezoid(free ** 2 * r * r, r))
        assert 0.99 < overlap < 1.0             # confined, not another function

    def test_it_is_cached_per_dataset(self):
        pp = get_paw("H")
        assert confined_orbital(pp, 0, 0.1) is confined_orbital(pp, 0, 0.1)

    def test_a_shift_that_would_cut_into_the_sphere_is_refused(self):
        with pytest.raises(ValueError, match="too large.*largest usable"):
            confinement_radius(get_paw("H"), 0, 500.0)


@needs_library
class TestBasis:
    def _functions(self, **kwargs):
        pp = {"H": get_paw("H")}
        functions, _owners = pseudo_basis(
            ["H"], np.zeros((1, 3)), pp, **kwargs)
        return pp["H"], functions

    def test_a_hook_that_declines_changes_nothing(self):
        r = np.linspace(0.05, 12.0, 60)
        _pp, stored = self._functions(size="DZP")
        _pp, declined = self._functions(
            size="DZP", first_zeta=lambda symbol, pp, l: None)
        for a, b in zip(stored, declined):
            np.testing.assert_array_equal(a.radial(r), b.radial(r))

    @pytest.mark.parametrize("size, count", [("SZ", 1), ("DZ", 2), ("DZP", 5)])
    def test_every_function_inherits_the_cutoff(self, size, count):
        pp, functions = self._functions(size=size,
                                        first_zeta=first_zeta_factory(0.1))
        r_c = confined_orbital(pp, 0, 0.1).r_c
        assert len(functions) == count          # the size hierarchy is intact
        beyond = np.linspace(r_c + 1e-3, r_c + 5.0, 40)
        # Not exactly zero: the basis function is a cubic spline through the
        # table, which rings at the 1e-8 level in the first interval past r_c.
        for function in functions:
            assert np.max(np.abs(function.radial(beyond))) < 1e-6


#: H2 through the PAW-LCAO family, h = 0.25 Angstrom, fermionic pool.
#: Re-measured 2026-09-23, when scalar-relativistic reference atoms and the
#: nonlinear core correction became the generation defaults and the library
#: was rebuilt:  free -30.242543 -> -30.228345, confined -30.709314 ->
#: -30.695169.  The *gain* from confining, 0.4668 eV, is what this file is
#: really about and it moved by under a milli-electronvolt.
#:
#: Re-measured again when the shipped hydrogen dataset was regenerated (it
#: predated ``PAWChannel.norm_correction``): free -> -30.228191, confined ->
#: -30.695442.  The gain becomes 0.4673 eV -- still under a milli-electronvolt
#: of movement, which is the point.
UNCONFINED_EV = -30.228191
CONFINED_EV = -30.695442
#: The same molecule in the DZP basis with a Gaussian polarization shell,
#: 20 qubits.  Re-measured in the same rebuild: -33.374833 -> -33.360087.
DZP_GAUSSIAN_EV = -33.360024
#: The same, with the confinement switched off (orbital polarization).
#: -32.884195 -> -32.869555 in the same rebuild.
DZP_UNCONFINED_EV = -32.869439


@needs_library
class TestMolecule:
    def _energy(self, basis, **kwargs):
        atoms = h2()
        atoms.calc = Mandacaru(method="adapt-vqe",
                               basis=basis, h=0.25, pool="fermionic",
                               trace=False, **kwargs)
        return atoms.get_potential_energy(), atoms

    def test_the_default_is_gpaws_energy_shift(self):
        from mandacaru.pseudopotentials.confinement import DEFAULT_ENERGY_SHIFT

        assert DEFAULT_ENERGY_SHIFT == 0.1
        default, _ = self._energy({"name": "PAW-LCAO"})
        explicit, _ = self._energy({"name": "PAW-LCAO", "energy_shift": 0.1})
        assert default == explicit

    @pytest.mark.parametrize("off", [None, False, 0])
    def test_it_can_be_switched_off(self, off):
        energy, atoms = self._energy({"name": "PAW-LCAO", "energy_shift": off})
        assert energy == pytest.approx(UNCONFINED_EV, abs=2e-5)
        assert atoms.calc.solver._gradient_context["confinement"] == {}

    def test_confinement_improves_the_minimal_basis(self):
        # A free-atom orbital is too diffuse for a molecule; a mild confinement
        # contracts it, and the variational energy drops.  Pinned so that a
        # change to the confining potential or the root search is noticed.
        free, _ = self._energy({"name": "PAW-LCAO", "energy_shift": None})
        confined, _ = self._energy({"name": "PAW-LCAO", "energy_shift": 0.1})
        assert free == pytest.approx(UNCONFINED_EV, abs=2e-5)
        assert confined == pytest.approx(CONFINED_EV, abs=2e-5)
        assert free - confined == pytest.approx(0.4673, abs=1e-3)

    def test_the_context_reports_the_radii(self):
        _, atoms = self._energy({"name": "PAW-LCAO", "energy_shift": 0.1})
        context = atoms.calc.solver._gradient_context
        orbital = context["confinement"]["H"][0]
        assert orbital.r_c == pytest.approx(6.68, abs=0.05)
        assert context["options"]["energy_shift"] == 0.1

    def test_forces_differentiate_the_confined_energy(self):
        basis = {"name": "PAW-LCAO", "energy_shift": 0.1}
        _, atoms = self._energy(basis, project_translation=False)
        analytic = atoms.get_forces()[1, 2]
        step = 0.005
        energies = []
        for sign in (+1, -1):
            moved = h2()
            moved.positions[1, 2] += sign * step
            # The grid of the reference geometry: the finite difference must
            # differentiate the same discretized energy the gradient does.
            moved.calc = Mandacaru(method="adapt-vqe",
                                   basis=basis, h=0.25, pool="fermionic",
                                   trace=False,
                                   grid=atoms.calc.solver._gradient_context[
                                       "integrals"].grid)
            energies.append(moved.get_potential_energy())
        numeric = -(energies[0] - energies[1]) / (2 * step)
        assert analytic == pytest.approx(numeric, abs=5e-3)


@needs_library
class TestPlumbing:
    def test_a_bad_value_is_refused_by_the_constructor(self):
        with pytest.raises(ValueError, match="energy_shift"):
            Mandacaru(method="adapt-vqe",
                      basis={"name": "PAW-LCAO", "energy_shift": -1})

    def test_norm_conserving_families_do_not_have_it(self):
        with pytest.raises(ValueError, match="unknown option.*energy_shift"):
            Mandacaru(method="adapt-vqe",
                      basis={"name": "NCPP", "energy_shift": 0.1})

    def test_upaw_has_it(self):
        Mandacaru(method="adapt-vqe",
                  basis={"name": "UPAW-LCAO", "energy_shift": 0.1})

    def test_the_dry_run_names_it(self):
        calc = Mandacaru(method="adapt-vqe",
                         basis={"name": "PAW-LCAO", "size": "DZP",
                                "energy_shift": 0.1})
        estimate = calc.dry_run(h2())
        assert "energy_shift 0.1 eV" in estimate.basis
        assert estimate.n_qubits == 20          # the count does not change

    def test_per_element_form(self):
        # An element left out of the mapping gets the family's default:
        # leaving an option out never means turning it off.
        calc = Mandacaru(method="adapt-vqe",
                         basis={"O": {"name": "PAW-LCAO", "energy_shift": 0.2},
                                "H": "PAW-LCAO"})
        water = Atoms("OH2", positions=[[4, 4, 4], [4, 4.77, 4.59],
                                        [4, 3.23, 4.59]], cell=[8.0] * 3)
        assert "energy_shift {*: 0.1, O: 0.2} eV" in calc.dry_run(water).basis


# What GPAW's own generator prints for the same recipe -- measured 2026-09-20
# with GPAW's `BasisMaker.from_symbol(symbol, xc="LDA").generate(2, 1,
# energysplit=0.1)`: the sz cutoff, the dz split radius and the polarization
# Gaussian's characteristic length, per valence channel (Bohr).  The datasets
# differ (each code pseudizes its own atom), so agreement is close, not exact.
GPAW_REFERENCE = {
    "H": {"r_c": {0: 6.64}, "r_split": {0: 3.67}, "l_pol": 1, "r_char": 1.395},
    "O": {"r_c": {0: 4.375, 1: 5.34}, "r_split": {0: 2.30, 1: 2.89},
          "l_pol": 2, "r_char": 1.125},
}


@needs_library
class TestTheGPAWRecipe:
    """Same ``size``, ``energy_shift``, confining potential, split scheme and
    polarization function as GPAW's basis generator: the radii must land where
    GPAW's do."""

    @pytest.mark.parametrize("symbol", sorted(GPAW_REFERENCE))
    def test_cutoff_and_split_radii(self, symbol):
        from mandacaru.basis.multizeta import GPAW_TAIL_NORMS, split_radius

        pp, reference = get_paw(symbol), GPAW_REFERENCE[symbol]
        r = np.asarray(pp.r)
        for l, expected in reference["r_c"].items():
            orbital = confined_orbital(pp, l, 0.1)
            assert orbital.r_c == pytest.approx(expected, abs=0.06)
            # GPAW's tail norm is a *norm*: 0.16 -> 0.0256 of the squared norm.
            split = split_radius(r, orbital.radial, GPAW_TAIL_NORMS[0] ** 2)
            assert split == pytest.approx(reference["r_split"][l], abs=0.08)

    @pytest.mark.parametrize("symbol", sorted(GPAW_REFERENCE))
    def test_the_polarization_gaussian(self, symbol):
        from mandacaru.pseudopotentials.confinement import (
            gaussian_polarization)

        pp, reference = get_paw(symbol), GPAW_REFERENCE[symbol]
        shell = gaussian_polarization(pp, 0.1)
        assert shell.l == reference["l_pol"]
        assert shell.r_char == pytest.approx(reference["r_char"], abs=0.005)
        # Its cutoff is the base orbital's, at the basis's own energy shift.
        assert shell.r_cut == confined_orbital(pp, shell.base_l, 0.1).r_c
        r = np.asarray(pp.r)
        assert np.trapezoid(shell.radial ** 2 * r * r, r) == pytest.approx(1.0)
        assert np.all(shell.radial[r >= shell.r_cut] == 0.0)

    def test_the_quasi_gaussian_closes_smoothly(self):
        from mandacaru.pseudopotentials.confinement import quasi_gaussian

        r = np.linspace(0.0, 6.0, 6001)
        f = quasi_gaussian(r, 0.5, 5.0)
        assert f[0] == pytest.approx(1.0 - (1 + 0.5 * 25) * np.exp(-12.5))
        edge = np.searchsorted(r, 5.0)
        assert abs(f[edge - 1]) < 1e-9                      # value -> 0
        assert abs(f[edge - 1] - f[edge - 2]) / 1e-3 < 1e-5  # slope -> 0
        assert np.all(f[edge:] == 0.0)

    def test_the_missing_channel_is_polarized_first(self):
        from mandacaru.pseudopotentials.confinement import polarization_channel

        assert polarization_channel({0: None}) == 1
        assert polarization_channel({0: None, 1: None}) == 2
        # A 4s/3d transition metal gets a p shell, as in GPAW -- not an f.
        assert polarization_channel({0: None, 2: None}) == 1


@needs_library
class TestSplitSchemes:
    def _split_radii(self, **kwargs):
        pp = {"H": get_paw("H")}
        functions, _ = pseudo_basis(["H"], np.zeros((1, 3)), pp, size="TZ",
                                    first_zeta=first_zeta_factory(0.1),
                                    **kwargs)
        r = np.linspace(0.0, 8.0, 8001)
        # Where each extra zeta ends.  The threshold sits above the 1e-8 a
        # cubic spline rings with in the first interval past a function's end.
        return [float(r[np.nonzero(np.abs(f.radial(r)) > 1e-5)[0][-1]])
                for f in functions[1:]]

    def test_gpaw_splits_every_zeta_from_the_first(self):
        from mandacaru.basis.multizeta import GPAW_TAIL_NORMS

        second, third = self._split_radii(tail_norms=GPAW_TAIL_NORMS)
        assert second == pytest.approx(3.68, abs=0.1)
        assert third < second

    def test_the_two_schemes_differ(self):
        from mandacaru.basis.multizeta import GPAW_TAIL_NORMS

        gpaw = self._split_radii(tail_norms=GPAW_TAIL_NORMS)
        siesta = self._split_radii(split_norm=0.15)
        # 0.15 of the squared norm is far more tail than a 0.16 norm (0.0256).
        assert siesta[0] < gpaw[0] - 0.5

    def test_the_default_is_gpaw_and_split_norm_selects_siesta(self):
        from mandacaru.basis.multizeta import (GPAW_TAIL_NORMS,
                                               resolve_split_scheme)

        assert resolve_split_scheme() == (None, GPAW_TAIL_NORMS)
        assert resolve_split_scheme(tail_norm=0.2) == (None, (0.2, 0.3, 0.6))
        assert resolve_split_scheme(split_norm=0.15) == (0.15, None)
        with pytest.raises(ValueError, match="not both"):
            resolve_split_scheme(split_norm=0.15, tail_norm=0.16)
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            resolve_split_scheme(tail_norm=1.5)

    def test_the_all_electron_nao_family_follows(self):
        from mandacaru.basis import BasisSet
        from mandacaru.basis.multizeta import GPAW_TAIL_NORMS

        assert BasisSet.build("NAO").tail_norms == GPAW_TAIL_NORMS
        siesta = BasisSet.build("NAO", split_norm=0.15)
        assert siesta.tail_norms is None and siesta.split_norm == 0.15


@needs_library
class TestNewOptionsEndToEnd:
    def _energy(self, basis):
        atoms = h2()
        atoms.calc = Mandacaru(method="adapt-vqe",
                               basis=basis, h=0.25, pool="fermionic",
                               trace=False)
        return atoms.get_potential_energy(), atoms.calc

    def test_gaussian_polarization_runs_and_is_reported(self):
        energy, calc = self._energy({"name": "PAW-LCAO", "size": "DZP",
                                     "energy_shift": 0.1,
                                     "polarization": "gaussian"})
        assert calc.n_qubits == 20
        assert energy == pytest.approx(DZP_GAUSSIAN_EV, abs=5e-5)
        shell = calc.solver._gradient_context["polarization"]["H"]
        assert shell.l == 1 and shell.r_char == pytest.approx(1.396, abs=0.005)

    def test_gaussian_polarization_needs_a_confined_orbital(self):
        # Asked for explicitly with the confinement switched off: refused.
        with pytest.raises(ValueError, match="needs an energy_shift"):
            Mandacaru(method="adapt-vqe",
                      basis={"name": "PAW-LCAO", "size": "DZP",
                             "energy_shift": None,
                             "polarization": "gaussian"})
        # With the family's default confinement it needs nothing else.
        Mandacaru(method="adapt-vqe",
                  basis={"name": "PAW-LCAO", "size": "DZP",
                         "polarization": "gaussian"})

    def test_the_polarization_shell_follows_the_confinement(self):
        from mandacaru.pseudopotentials.confinement import resolve_polarization

        # Left unwritten: Gaussian where the orbital is confined ...
        assert resolve_polarization(None, 0.1) == "gaussian"
        assert resolve_polarization(None, {"O": 0.2}, "O") == "gaussian"
        # ... and the orbital shell where it is not, instead of an error.
        assert resolve_polarization(None, None) == "orbital"
        assert resolve_polarization(None, {"O": 0.2}, "H") == "orbital"
        # Written, it is what was written.
        assert resolve_polarization("orbital", 0.1) == "orbital"
        default = self._energy({"name": "PAW-LCAO", "size": "DZP"})[0]
        gaussian = self._energy({"name": "PAW-LCAO", "size": "DZP",
                                 "polarization": "gaussian"})[0]
        assert default == gaussian == pytest.approx(DZP_GAUSSIAN_EV,
                                                    abs=5e-5)
        unconfined = self._energy({"name": "PAW-LCAO", "size": "DZP",
                                   "energy_shift": None})[0]
        assert unconfined == pytest.approx(DZP_UNCONFINED_EV, abs=5e-5)

    def test_the_confinement_is_an_option_with_gpaws_default(self):
        from mandacaru.pseudopotentials.confinement import validate_confinement

        assert validate_confinement(None) == (12.0, 0.6)
        pp = get_paw("H")
        softer = confined_orbital(pp, 0, 0.1, (6.0, 0.6)).r_c
        assert softer < confined_orbital(pp, 0, 0.1).r_c    # a lower wall
        for bad in ((12.0, 1.5), (-1.0, 0.6), "gpaw", (12.0,)):
            with pytest.raises(ValueError, match="confinement"):
                Mandacaru(method="adapt-vqe",
                          basis={"name": "PAW-LCAO", "confinement": bad})

    def test_both_split_options_are_refused_together(self):
        with pytest.raises(ValueError, match="not both"):
            Mandacaru(method="adapt-vqe",
                      basis={"name": "PAW-LCAO", "size": "DZ", "tail_norm": 0.16,
                             "split_norm": 0.15})

    def test_the_dry_run_names_the_gaussian_shell(self):
        calc = Mandacaru(method="adapt-vqe",
                         basis={"name": "PAW-LCAO", "size": "DZP",
                                "energy_shift": 0.1,
                                "polarization": "gaussian"})
        assert "gaussian polarization" in calc.dry_run(h2()).basis
