"""Library gate: a PAW-LCAO dataset must hold its elemental crystal together.

The atomic gates of the generator (ghost states, scattering phases, atom
levels) never put a dataset in a solid.  A valence-only vanadium with a
3.0 Bohr s sphere and a 1.16 Bohr d sphere passed all of them and had no
minimum in bcc V on a grid of 14 points per primitive vector, unnoticed
(HISTORY.md, 2026-10-20: K23 and the solid-state audit).

The protocol (corrected 2026-10-21, WP-1 and WP-3): a primitive cell of the
element's crystal (or a proxy at its atomic volume), the same real-space grid
COUNT at every volume, Fermi-Dirac 0.1 eV, a coarse k-mesh and lattice
constants from -12 % to +12 % around experiment -- with the basis-function
**filter cutoff pinned**.  The default cutoff follows the grid (``pi / h``), so
in a fixed-count scan the basis function itself changes with the lattice and
a compact projector turns that into tens of eV of one-center energy: the
minimum of an audit run with the default cutoff can be an artifact (it hid
the real expansion bias of Mg, Ag, Cd and Hg and invented the failures of
Fe-Cu).  The pin is the cutoff of the coarsest grid of the scan, the
scale 1.12 point, held for every scale.  The gate asks only whether an
interior minimum *exists*; it is deliberately weaker than a lattice-constant
test.

Four tiers:

* rule tests (fast): the flag thresholds, the minimum test and the fits,
  against analytic curves, no SCF;
* library tests (fast, need ``MANDACARU_PAW_PATH``): every installed dataset
  declaring a semicore shell has its channel and partial waves, and the
  generator table agrees with the file;
* one-center integrals: the reference of a scattering channel grows outside
  the sphere (WP-0: Cu ``V`` 1e14 Ha before the cut in ``onecenter.py``), so
  no installed dataset may have a one-center term above 1e3 Ha.  The terms
  cost a reference-atom solve (about 20-30 s per element, measured), so the
  slow tier checks Cu (the worst case) and ``MANDACARU_LIBRARY_AUDIT=1``
  checks both libraries;
* sentinels (``-m slow``, about two and a half minutes with the Cu check): Al,
  V and Ta crystals at SZP and five points.  The sweep of the whole library
  (``MANDACARU_LIBRARY_AUDIT=1``, hours, quiet machine) runs DZP at seven
  points.

``KNOWN_FAILURES`` is the ONE table of the elements that still have no
minimum under the pinned protocol, with the reason and the date of the
measurement (WP-3 re-gate, DZP, 14 points).  The sweep marks them ``xfail``
(not strict: a curve that ends within one point of the edge flips with SCF
noise).  Fix a dataset, re-run the sweep and delete its row.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.data import atomic_numbers, chemical_symbols, reference_states
from scipy.optimize import curve_fit

from mandacaru import Mandacaru
from mandacaru.basis.filtering import cutoff_energy_ev

BOHR = 0.529177210903
EV_PER_A3_IN_GPA = 160.21766208

#: The sentinel scan: five points (an interior minimum needs three) at SZP.
SENTINEL_SCALES = (0.88, 0.94, 1.0, 1.06, 1.12)
#: The sweep scan: seven points at DZP.
SWEEP_SCALES = (0.88, 0.9067, 0.9533, 1.0, 1.0467, 1.0933, 1.12)
POINTS_PER_VECTOR = 14

#: Proxy crystals at the experimental atomic volume where the ground state is
#: molecular or complex: ``symbol: (structure, volume per atom in A^3)``.
PROXIES = {
    "He": ("fcc", 35.0), "B": ("diamond", 7.3), "Mn": ("bcc", 12.26),
    "Ga": ("fcc", 19.6), "Pm": ("hcp", 33.2), "Sm": ("hcp", 32.9),
    "Rn": ("fcc", 83.8), "Fr": ("bcc", 125.0), "Ra": ("bcc", 68.2),
    "U": ("bcc", 20.9),
}
#: Not a crystal in this gate: molecular elements (a bond-length scan is the
#: right test there) and the two chains and A7 phases, which need internal
#: coordinates.
NOT_A_CRYSTAL = {"H", "N", "O", "F", "Cl", "Br", "I", "At", "P", "S", "Se", "Te"}

#: Cheap sentinels: an s-p metal, the semicore vanadium that was the original
#: defect, and a heavy 5d metal.
SENTINELS = ("Al", "V", "Ta")

#: Why an element has no minimum (WP-3, 2026-10-21, DZP, 14 points, pinned).
NOBLE = "van der Waals solid: no valence binding to find; gate on the dimer"
LANTHANIDE = "lanthanide, 4f frozen: p and d channels without projectors"
HEAVY_S = "heavy s / actinide: no p channel, 3.5-5 Bohr spheres"
EXPANSION = "energy falls on expansion at every pin: a real, compact-sphere defect"
COMPRESSION = "energy falls on compression: grid-sensitive compact p block"
UNRELIABLE = "no usable minimum (fit flagged unreliable)"
MEASURED = "2026-10-21"

#: THE known-failure table: ``(library, symbol) -> (reason, date)``.
KNOWN_FAILURES = {
    # lda-sr
    **{("lda-sr", s): (NOBLE, MEASURED) for s in ("He", "Ne", "Ar", "Kr", "Xe", "Rn")},
    **{("lda-sr", s): (COMPRESSION, MEASURED) for s in ("Ga", "Ge", "As", "Sn", "Sb")},
    **{("lda-sr", s): (EXPANSION, MEASURED) for s in ("Cd", "Ba", "Lu", "Hg")},
    **{("lda-sr", s): (LANTHANIDE, MEASURED)
       for s in ("Ce", "Pm", "Gd", "Tb", "Dy", "Ho", "Er", "Tm")},
    **{("lda-sr", s): (HEAVY_S, MEASURED) for s in ("Ra", "Ac", "Th", "Pa", "U")},
    # pbe-sr
    **{("pbe-sr", s): (NOBLE, MEASURED) for s in ("He", "Ne", "Xe", "Rn")},
    **{("pbe-sr", s): (COMPRESSION, MEASURED) for s in ("Ga", "Sn")},
    **{("pbe-sr", s): (EXPANSION, MEASURED)
       for s in ("Mg", "Zn", "Cd", "In", "Cs", "Ba", "Hg", "Tl")},
    **{("pbe-sr", s): (LANTHANIDE, MEASURED)
       for s in ("Ce", "Sm", "Gd", "Dy", "Ho", "Tm")},
    **{("pbe-sr", s): (HEAVY_S, MEASURED) for s in ("Ra", "Ac", "Pa", "U")},
}

#: One-center integrals of an installed dataset must be of the size of a
#: Hartree-scale quantity; the growing-reference bug gave 1e4 to 1e23.
ONE_CENTER_BOUND_HARTREE = 1.0e3


def crystal_of(symbol: str) -> Atoms:
    """The primitive cell the gate scans (experiment, or the proxy's volume)."""
    if symbol in PROXIES:
        structure, volume = PROXIES[symbol]
        if structure == "fcc":
            return bulk(symbol, "fcc", a=(4 * volume) ** (1 / 3))
        if structure == "bcc":
            return bulk(symbol, "bcc", a=(2 * volume) ** (1 / 3))
        if structure == "diamond":
            return bulk(symbol, "diamond", a=(8 * volume) ** (1 / 3), cubic=False)
        c_over_a = 1.614
        a = (2 * volume / (np.sqrt(3) / 2 * c_over_a)) ** (1 / 3)
        return bulk(symbol, "hcp", a=a, c=a * c_over_a)
    if symbol == "Sn":
        return bulk("Sn", "diamond", a=6.489, cubic=False)       # alpha-Sn
    reference = reference_states[atomic_numbers[symbol]]
    structure = reference["symmetry"]
    if structure in ("fcc", "bcc", "diamond", "sc"):
        return bulk(symbol, structure, a=reference["a"], cubic=False)
    if structure == "hcp":
        return bulk(symbol, "hcp", a=reference["a"],
                    c=reference["a"] * reference["c/a"])
    if structure in ("bct", "tetragonal") and symbol in ("In", "Pa"):
        a = reference["a"]
        c = a * reference["c/a"]
        cell = [[-a / 2, a / 2, c / 2], [a / 2, -a / 2, c / 2],
                [a / 2, a / 2, -c / 2]]
        return Atoms(symbol, cell=cell, pbc=True, scaled_positions=[[0, 0, 0]])
    if structure == "rhombohedral":
        return bulk(symbol)
    raise KeyError(symbol)


# --------------------------------------------------------------------------- #
# The decision rules (no SCF).
# --------------------------------------------------------------------------- #

def has_interior_minimum(scales, energies) -> bool:
    """Whether the lowest energy is not an end point and the quartic fit's
    minimum is inside the scanned range as well."""
    scales = np.asarray(scales, dtype=float)
    energies = np.asarray(energies, dtype=float)
    lowest = int(np.argmin(energies))
    if lowest in (0, len(energies) - 1):
        return False
    fine = np.linspace(scales[0], scales[-1], 4001)
    fitted = np.polyval(np.polyfit(scales, energies, 4), fine)
    return 0 < int(np.argmin(fitted)) < len(fine) - 1


def flag_of(error_percent, has_minimum: bool = True,
            bulk_modulus_gpa: float | None = None) -> str:
    """OK below 3 %, SUSPECT 3-6 %, FAIL above 6 % or without a minimum.

    An OK with a bulk modulus above 700 GPa is demoted to SUSPECT (WP-3):
    such a curve is a drifting basis, not a stiff solid.
    """
    if not has_minimum or error_percent is None:
        return "FAIL"
    size = abs(float(error_percent))
    if size >= 6.0:
        return "FAIL"
    if size >= 3.0:
        return "SUSPECT"
    if bulk_modulus_gpa is not None and bulk_modulus_gpa > 700.0:
        return "SUSPECT"
    return "OK"


def quartic_minimum(scales, energies) -> tuple[float, np.ndarray]:
    """``(scale of the minimum, quartic coefficients)`` of the fit in the
    lattice scale; ``nan`` when the minimum is not inside the scanned range."""
    scales = np.asarray(scales, dtype=float)
    coefficients = np.polyfit(scales, np.asarray(energies, dtype=float), 4)
    fine = np.linspace(scales[0], scales[-1], 20001)
    index = int(np.argmin(np.polyval(coefficients, fine)))
    if index in (0, len(fine) - 1):
        return float("nan"), coefficients
    return float(fine[index]), coefficients


def quartic_bulk_modulus(scales, energies, volume0: float) -> float:
    """Bulk modulus (GPa) at the minimum of the quartic fit, ``volume0`` the
    volume per atom (A^3) at scale 1: ``B = V d2E/dV2`` with ``V = V0 s^3``."""
    s0, c = quartic_minimum(scales, energies)
    if math.isnan(s0):
        return float("nan")
    # dE/dV = E'(s) / (3 V0 s^2); at the minimum E' = 0, so
    # d2E/dV2 = E''(s) / (3 V0 s^2)^2.
    second = np.polyval(np.polyder(c, 2), s0)
    dv_ds = 3.0 * volume0 * s0 ** 2
    return float(volume0 * s0 ** 3 * second / dv_ds ** 2 * EV_PER_A3_IN_GPA)


def birch_murnaghan_energy(volume, e0, v0, b0, b1):
    """Third-order Birch-Murnaghan energy (eV, A^3, B0 in eV/A^3)."""
    eta = (v0 / volume) ** (2.0 / 3.0) - 1.0
    return e0 + 9.0 * v0 * b0 / 16.0 * (eta ** 3 * b1
                                        + eta ** 2 * (6.0 - 4.0 * (eta + 1.0)))


def birch_murnaghan_fit(volumes, energies) -> tuple[float, float, float]:
    """``(V0 in A^3, B0 in GPa, B0')`` of a third-order Birch-Murnaghan fit."""
    volumes = np.asarray(volumes, dtype=float)
    energies = np.asarray(energies, dtype=float)
    start = (energies.min(), volumes[np.argmin(energies)], 1.0, 4.0)
    (_e0, v0, b0, b1), _ = curve_fit(birch_murnaghan_energy, volumes, energies,
                                     p0=start, maxfev=20000)
    return float(v0), float(b0 * EV_PER_A3_IN_GPA), float(b1)


def pinned_filter_ev(lengths, scales, points_per_vector=POINTS_PER_VECTOR) -> float:
    """The filter cutoff (eV) held at every scale: the ``pi / h`` of the
    coarsest grid of the scan (largest scale), ``h`` the spacing of the
    shortest cell vector."""
    spacing = float(min(lengths)) * max(scales) / points_per_vector / BOHR
    return cutoff_energy_ev(math.pi / spacing)


def energies_per_atom(symbol: str, paw_dataset: str, size: str = "SZP",
                      scales=SENTINEL_SCALES):
    """The energy-volume curve of the gate (eV per atom), filter pinned."""
    reference = crystal_of(symbol)
    lengths = np.linalg.norm(reference.cell, axis=1)
    mesh = tuple(int(np.clip(round(20.0 / length), 3, 8)) for length in lengths)
    cutoff = pinned_filter_ev(lengths, scales)
    out = []
    for scale in scales:
        atoms = reference.copy()
        atoms.set_cell(reference.cell * scale, scale_atoms=True)
        atoms.calc = Mandacaru(
            method="dft", xc="pbe" if paw_dataset.startswith("pbe") else "lda",
            grid_spacing=float(lengths.min() * scale / POINTS_PER_VECTOR),
            trace=False, paw_dataset=paw_dataset,
            basis={"name": "PAW-LCAO", "size": size, "filter_cutoff": cutoff},
            kpts={"size": mesh, "gamma": True},
            smearing={"method": "fermi-dirac", "width": 0.1})
        out.append(atoms.get_potential_energy() / len(atoms))
    return np.array(out)


class TestTheMinimumTest:
    """The decision rule alone, without an SCF."""

    scales = np.array(SWEEP_SCALES)

    def test_a_parabola_inside_the_range_has_a_minimum(self):
        assert has_interior_minimum(self.scales, (self.scales - 0.97) ** 2)

    @pytest.mark.parametrize("scales", (SENTINEL_SCALES, SWEEP_SCALES))
    def test_five_points_are_enough_to_see_a_minimum(self, scales):
        s = np.array(scales)
        assert has_interior_minimum(s, (s - 1.0) ** 2)

    @pytest.mark.parametrize("sign", (+1, -1))
    def test_a_monotonic_curve_has_none(self, sign):
        assert not has_interior_minimum(self.scales, sign * self.scales)

    def test_a_curve_whose_minimum_sits_past_the_edge_has_none(self):
        assert not has_interior_minimum(self.scales, (self.scales - 0.80) ** 2)

    def test_every_element_but_the_listed_ones_has_a_crystal(self):
        for z in range(1, 93):
            symbol = chemical_symbols[z]
            if symbol in NOT_A_CRYSTAL:
                continue
            assert len(crystal_of(symbol)) >= 1, symbol


class TestTheFlags:
    @pytest.mark.parametrize("error, expected", [
        (0.0, "OK"), (2.99, "OK"), (-2.99, "OK"), (3.0, "SUSPECT"),
        (-4.5, "SUSPECT"), (5.99, "SUSPECT"), (6.0, "FAIL"), (-8.5, "FAIL")])
    def test_thresholds(self, error, expected):
        assert flag_of(error) == expected

    def test_no_minimum_is_a_fail_whatever_the_error(self):
        assert flag_of(0.1, has_minimum=False) == "FAIL"
        assert flag_of(None) == "FAIL"

    def test_an_unphysically_stiff_ok_is_demoted(self):
        assert flag_of(1.0, bulk_modulus_gpa=699.0) == "OK"
        assert flag_of(1.0, bulk_modulus_gpa=701.0) == "SUSPECT"
        assert flag_of(7.0, bulk_modulus_gpa=701.0) == "FAIL"


class TestTheFits:
    volume0, b0, b1, e0 = 13.0, 0.9, 4.2, -100.0       # B0 = 144 GPa

    def energy(self, volumes):
        return birch_murnaghan_energy(volumes, self.e0, self.volume0, self.b0, self.b1)

    def test_birch_murnaghan_recovers_an_analytic_curve(self):
        scales = np.array(SWEEP_SCALES)
        volumes = self.volume0 * scales ** 3
        v0, b0, b1 = birch_murnaghan_fit(volumes, self.energy(volumes))
        assert v0 == pytest.approx(self.volume0, rel=1e-6)
        assert b0 == pytest.approx(self.b0 * EV_PER_A3_IN_GPA, rel=1e-5)
        assert b1 == pytest.approx(self.b1, rel=1e-4)

    def test_the_quartic_finds_the_minimum_and_modulus_of_a_quartic_curve(self):
        # E(s) = c2 (s - 1)^2 + c3 (s - 1)^3 + c4 (s - 1)^4: the quartic fit is exact.
        s = np.array(SWEEP_SCALES)
        c2, c3, c4 = 8.0, -6.0, 5.0
        energy = lambda x: c2 * (x - 1) ** 2 + c3 * (x - 1) ** 3 + c4 * (x - 1) ** 4
        s0, _ = quartic_minimum(s, energy(s))
        assert s0 == pytest.approx(1.0, abs=1e-3)
        # d2E/dV2 = E''(1) / (3 V0)^2 = 2 c2 / (3 V0)^2, B = V0 * that.
        expected = self.volume0 * 2 * c2 / (3 * self.volume0) ** 2 * EV_PER_A3_IN_GPA
        assert quartic_bulk_modulus(s, energy(s), self.volume0) == pytest.approx(
            expected, rel=2e-3)

    def test_the_quartic_agrees_with_birch_murnaghan_on_a_physical_curve(self):
        # The quartic in the scale is not exact for a Birch-Murnaghan curve;
        # over -12..+12 % it carries a few percent in B (WP-1: +-7 % on real data).
        s = np.array(SWEEP_SCALES)
        energies = self.energy(self.volume0 * s ** 3)
        s0, _ = quartic_minimum(s, energies)
        assert s0 == pytest.approx(1.0, abs=2e-3)
        b = quartic_bulk_modulus(s, energies, self.volume0)
        assert b == pytest.approx(self.b0 * EV_PER_A3_IN_GPA, rel=0.03)

    def test_a_curve_without_a_minimum_gives_no_modulus(self):
        s = np.array(SWEEP_SCALES)
        assert math.isnan(quartic_bulk_modulus(s, -s, 13.0))


class TestThePinnedFilter:
    def test_the_pin_is_the_coarsest_grid_of_the_scan(self):
        lengths = [2.5, 2.5, 2.5]
        pin = pinned_filter_ev(lengths, SENTINEL_SCALES)
        h = 2.5 * 1.12 / POINTS_PER_VECTOR / BOHR
        assert pin == pytest.approx(cutoff_energy_ev(math.pi / h))
        # finer points of the scan would have asked for a higher cutoff
        assert pin < pinned_filter_ev(lengths, (0.88,))

    def test_vanadium_reproduces_the_wp1_value(self):
        lengths = np.linalg.norm(crystal_of("V").cell, axis=1)
        assert pinned_filter_ev(lengths, SWEEP_SCALES) == pytest.approx(859.0, abs=3.0)


class TestTheKnownFailureTable:
    def test_every_row_names_a_real_crystal_element_with_a_reason_and_date(self):
        for (library, symbol), (reason, date) in KNOWN_FAILURES.items():
            assert library in ("lda-sr", "pbe-sr")
            assert symbol in chemical_symbols and symbol not in NOT_A_CRYSTAL
            assert reason and len(date) == 10

    def test_the_sentinels_are_never_known_failures(self):
        assert not {s for _, s in KNOWN_FAILURES} & set(SENTINELS)


# --------------------------------------------------------------------------- #
# The installed library (needs MANDACARU_PAW_PATH; no SCF).
# --------------------------------------------------------------------------- #

def _library_cases():
    root = os.environ.get("MANDACARU_PAW_PATH")
    if not root:
        return []
    from mandacaru.pseudopotentials.io import available_elements
    return [pytest.param(library, symbol, id=f"{library}-{symbol}")
            for library in ("lda-sr", "pbe-sr")
            for symbol in available_elements(os.path.join(root, library))]


def _dataset(library: str, symbol: str):
    from mandacaru.pseudopotentials.paw import get_paw
    folder = os.path.join(os.environ["MANDACARU_PAW_PATH"], library)
    return get_paw(symbol, folder, library[:3])


@pytest.mark.parametrize("library, symbol", _library_cases())
def test_a_declared_semicore_has_its_channel_and_partial_waves(library, symbol):
    pp = _dataset(library, symbol)
    from mandacaru.pseudopotentials.onecenter import _occupations
    for _n, l in pp.semicore_subshells:
        assert l in pp.channels, f"{library} {symbol}: no l={l} channel for its semicore"
        # the semicore wave and the valence wave of the channel
        assert len(_occupations(pp, l)) >= 2, (
            f"{library} {symbol}: the l={l} channel holds one partial wave only")


@pytest.mark.parametrize("library, symbol", _library_cases())
def test_the_generator_table_semicore_is_what_the_dataset_declares(library, symbol):
    from mandacaru.pseudopotentials.paw import DEFAULT_SEMICORE_SUBSHELLS
    pp = _dataset(library, symbol)
    declared = tuple(sorted(tuple(x) for x in pp.semicore_subshells))
    tabulated = tuple(sorted(DEFAULT_SEMICORE_SUBSHELLS.get(symbol, ())))
    assert declared == tabulated, f"{library} {symbol}: {declared} vs table {tabulated}"


def _assert_one_center_bounded(library, symbol):
    from mandacaru.pseudopotentials.onecenter import exact_one_center_terms
    terms = exact_one_center_terms(_dataset(library, symbol))
    for l, (V, K, D0) in terms.items():
        for name, block in (("V", V), ("K", K), ("D0", D0)):
            peak = float(np.abs(block).max())
            assert np.isfinite(peak) and peak < ONE_CENTER_BOUND_HARTREE, (
                f"{library} {symbol} l={l}: max|{name}| = {peak:.3g} Ha "
                "(a scattering reference grows outside the sphere)")


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("MANDACARU_PAW_PATH"),
                    reason="needs the PAW-LCAO library")
def test_copper_one_center_terms_are_bounded():
    """Copper's p channel was the worst growing reference (1e14 Ha)."""
    _assert_one_center_bounded("lda-sr", "Cu")


@pytest.mark.skipif(not os.environ.get("MANDACARU_LIBRARY_AUDIT"),
                    reason="about 20-30 s per element; set MANDACARU_LIBRARY_AUDIT=1")
@pytest.mark.parametrize("library, symbol", _library_cases())
def test_no_dataset_has_an_unbounded_one_center_integral(library, symbol):
    _assert_one_center_bounded(library, symbol)


# --------------------------------------------------------------------------- #
# The crystals (SCF).
# --------------------------------------------------------------------------- #

@pytest.mark.slow
@pytest.mark.parametrize("symbol", SENTINELS)
def test_the_sentinel_crystals_keep_an_energy_minimum(symbol):
    curve = energies_per_atom(symbol, "lda-sr")
    assert has_interior_minimum(SENTINEL_SCALES, curve), (
        f"{symbol}: no energy minimum between 0.88 and 1.12 of the experimental "
        f"lattice, filter pinned ({np.round(curve - curve.min(), 3).tolist()} eV)")


def _sweep_cases():
    cases = []
    for library in ("lda-sr", "pbe-sr"):
        for z in range(1, 93):
            symbol = chemical_symbols[z]
            if symbol in NOT_A_CRYSTAL:
                continue
            marks = [pytest.mark.slow]
            known = KNOWN_FAILURES.get((library, symbol))
            if known:
                marks.append(pytest.mark.xfail(
                    strict=False, reason=f"{known[0]} ({known[1]})"))
            cases.append(pytest.param(library, symbol, marks=marks,
                                      id=f"{library}-{symbol}"))
    return cases


@pytest.mark.skipif(not os.environ.get("MANDACARU_LIBRARY_AUDIT"),
                    reason="the library sweep takes hours; set MANDACARU_LIBRARY_AUDIT=1")
@pytest.mark.parametrize("library, symbol", _sweep_cases())
def test_every_element_of_the_library_has_an_energy_minimum(library, symbol):
    curve = energies_per_atom(symbol, library, size="DZP", scales=SWEEP_SCALES)
    assert has_interior_minimum(SWEEP_SCALES, curve), (
        f"{library} {symbol}: no energy minimum between 0.88 and 1.12 of the "
        f"experimental lattice, DZP, filter pinned "
        f"({np.round(curve - curve.min(), 3).tolist()} eV)")
