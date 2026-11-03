# -*- coding: utf-8 -*-
# file: test/experimental/test_paw.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The PAW-LCAO family (experimental): Bloechl's projector augmented-wave method
in its frozen-core form, with exact one-center terms for wave-function methods
and linearized ones for Kohn-Sham.

Atomic checks (radial, cheap) on freshly generated H, Li and O -- projector
duality, the generalized eigenproblem, the reconstruction of the all-electron
wave (the transformation the method is named after), logarithmic derivatives
-- and the molecular checks on H2 (0.74 A, grid_spacing = 0.25 A, 6 A cell) and LiH
(1.6 A, grid_spacing = 0.30 / 0.25 A, 8 A cell) against pinned energies.  The overlap
correction makes these the first runs through the augmented overlap
``S + C q C^dagger`` with a nonzero ``q``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.pseudopotentials import PSEUDO_FAMILIES, PAWChannel, PAWDataset, family_names, generate_paw, get_paw, load_pseudopotential, lookup_family, paw_library_path, paw_spectrum, reconstruct_ae, report_paw, resolve_family, save_pseudopotential
from mandacaru.pseudopotentials.diagnostics import diagnose, ghost_levels, intruding_s_miss
from mandacaru.pseudopotentials.io import library_elements
from mandacaru.pseudopotentials.environment import LibraryPathError
from mandacaru.pseudopotentials.io import available_elements, detect_format
from mandacaru.units import HARTREE_TO_EV

# --------------------------------------------------------------------------- #
# Test systems.
# --------------------------------------------------------------------------- #

H2_H, H2_CELL = 0.25, 6.0
LIH_H, LIH_CELL = 0.30, 8.0


def h2():
    atoms = Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]],
                  cell=[H2_CELL] * 3)
    atoms.center()
    return atoms


def lih():
    atoms = Atoms("LiH", positions=[[0, 0, 0], [0, 0, 1.6]],
                  cell=[LIH_CELL] * 3)
    atoms.center()
    return atoms


SYSTEMS = {"H2": (h2, H2_H), "LiH": (lih, LIH_H)}
#: KB projectors on the grid, two per channel and m.  The K21 lithium
#: (2026-10-11) carries an s and a p channel, 2 + 2 x 3 = 8.  Since 2026-10-25
#: hydrogen carries the same two channels (the p channel is empty, but its
#: projectors are still built), so it too has 8 per atom: 2 x 8 for H2 and
#: 8 + 8 for LiH.  Before that H had an s channel only (2 per atom: H2 4, LiH 10).
N_PROJECTORS = {"H2": 16, "LiH": 16}

#: PAW energies with the shipped library (H: rc 1.30, deficit 0.05,
#: Delta 0.5; Li: rc 2.60, deficit 0.02, Delta 0.5).  Re-measured 2026-09-17,
#: when the compensation charge gained the electron-ion attraction it was
#: missing (``one_body_augmentation``): it had been paying the Hartree
#: repulsion of the augmentation charge with no attraction to the ion, so
#: every PAW energy was too high by an amount that scales with that charge.
#: The values before that fix are kept for provenance in
#: ``before_compensation_attraction``.
#:
#: Re-measured again 2026-09-20, when the **Fourier filter became the PAW
#: default** (``default_options = {"filter_cutoff": True}``): every basis function is
#: now band-limited to the grid's Nyquist wave-vector before it is sampled.
#: The shift is tiny on these s-valence systems -- H2 -1.1 meV (the filtered
#: function is marginally *better*), LiH +14.1 meV -- because their orbitals
#: were nearly band-limited already; the oxygen-bearing systems move by
#: hundreds of meV.  ``basis={"name": "paw-lcao", "filter_cutoff": False}`` reproduces
#: ``PAW_BEFORE_BASIS_FILTER`` exactly.
#:
#: These are the values of **both** 2026-09-20 defaults together: the filtered
#: basis *and* the atom-centered quadrature of the local potential's short-range
#: part (``PAWIntegrals.exact_local_potential``).  Each alone is pinned below,
#: so either change can be audited separately.
#:
#: Re-measured 2026-09-23, when scalar-relativistic reference atoms and the
#: nonlinear core correction became the generation defaults and the library
#: was rebuilt.  H2 moves by ~0.5 mHa, LiH by ~2.6 mHa -- lithium has a 1s
#: core for the correction to act on and hydrogen does not.
#:
#: Re-measured again later the same day, when the shipped hydrogen dataset was
#: regenerated: it predated ``PAWChannel.norm_correction``, so its stored
#: overlap correction was the M-weighted norm where the charge belongs
#: (6.4e-5 for H).  Every number here moves by ~5 uHa, and only through
#: hydrogen -- lithium's dataset is unchanged.
#:
#: Re-measured 2026-09-27 against the library regenerated on 2026-09-25 (all
#: four tables below move together, H2 by ~8 uHa and LiH by ~11 uHa; the
#: old pins described files no current generator reproduces, dev_plan/PAW_SAGA.md
#: section 7).  Old -> new, rhf / adapt:
#:   PAW                      H2 -1.094865 / -1.108616 -> -1.094873 / -1.108624
#:                            LiH -0.769928 / -0.777671 -> -0.769939 / -0.777683
#:   PAW_FILTER_ONLY          H2 -1.094911 / -1.108663 -> -1.094919 / -1.108671
#:                            LiH -0.769909 / -0.777653 -> -0.769921 / -0.777665
#:   PAW_EXACT_LOCAL_ONLY     H2 -1.094880 / -1.108646 -> -1.094887 / -1.108652
#:                            LiH -0.770388 / -0.778105 -> -0.770399 / -0.778117
#:   PAW_BEFORE_BASIS_FILTER  H2 -1.094874 / -1.108638 -> -1.094880 / -1.108644
#:                            LiH -0.770427 / -0.778141 -> -0.770438 / -0.778153
#:
#: Re-measured 2026-09-30 against the libraries rebuilt 2026-09-28/30
#: (relativistic exchange, the intruding-1s check; HISTORY.md).  All four
#: tables move together, H2 by ~20 uHa and LiH by ~16 uHa:
#:   PAW                      H2 -1.094873 / -1.108624 -> -1.094853 / -1.108603
#:                            LiH -0.769939 / -0.777683 -> -0.769924 / -0.777664
#:   PAW_FILTER_ONLY          H2 -1.094919 / -1.108671 -> -1.094899 / -1.108650
#:                            LiH -0.769921 / -0.777665 -> -0.769906 / -0.777647
#:   PAW_EXACT_LOCAL_ONLY     H2 -1.094887 / -1.108652 -> -1.094867 / -1.108632
#:                            LiH -0.770399 / -0.778117 -> -0.770383 / -0.778098
#:   PAW_BEFORE_BASIS_FILTER  H2 -1.094880 / -1.108644 -> -1.094860 / -1.108624
#:                            LiH -0.770438 / -0.778153 -> -0.770423 / -0.778134
#:
#: Re-measured 2026-10-05: the isolated Coulomb kernel became spectral at the
#: singularity (the grid's own O(h^2) error of 1/r gone; H2 +1.85 mHa, LiH
#: +2.6 mHa) and the libraries were regenerated with the continuous PZ81
#: (HISTORY 2026-10-04/05).  Old -> new, rhf / adapt:
#:   PAW                      H2 -1.094853 / -1.108603 -> -1.093006 / -1.107128
#:                            LiH -0.769924 / -0.777664 -> -0.767303 / -0.775581
#:   PAW_FILTER_ONLY          H2 -1.094899 / -1.108650 -> -1.093052 / -1.107175
#:                            LiH -0.769906 / -0.777647 -> -0.767286 / -0.775564
#:   PAW_EXACT_LOCAL_ONLY     H2 -1.094867 / -1.108632 -> -1.093020 / -1.107156
#:                            LiH -0.770383 / -0.778098 -> -0.767757 / -0.776009
#:   PAW_BEFORE_BASIS_FILTER  H2 -1.094860 / -1.108624 -> -1.093013 / -1.107148
#:                            LiH -0.770423 / -0.778134 -> -0.767796 / -0.776045
#:
#: Re-measured 2026-10-11 (W2), LiH only: the K21 lithium (a p channel and a
#: partial core; HISTORY "K21 libraries installed").  H2 does not move.
#:   PAW                      LiH -0.767303 / -0.775581 -> -0.761836 / -0.769347
#:   PAW_FILTER_ONLY          LiH -0.767286 / -0.775564 -> -0.761814 / -0.769327
#:   PAW_EXACT_LOCAL_ONLY     LiH -0.767757 / -0.776009 -> -0.762302 / -0.769787
#:   PAW_BEFORE_BASIS_FILTER  LiH -0.767796 / -0.776045 -> -0.762339 / -0.769822
#:
#: Re-measured 2026-10-12: the spheres of a wave-function Hamiltonian became
#: exact (the one-center Coulomb tensor in the two-body operator, the frozen
#: LDA functional out of D^ion, the core's exchange in, no one-center
#: constant; HISTORY.md "PAW-LCAO Hamiltonian: one-center exact exchange").
#: Absolute energies move by +29 mHa (H2) and +16 mHa (LiH); the
#: correlation energies by -0.6 (H2) and -2.3 mHa (LiH).  Old -> new, rhf / adapt:
#:   PAW                      H2 -1.093006 / -1.107128 -> -1.063691 / -1.078462
#:                            LiH -0.761836 / -0.769347 -> -0.745519 / -0.755279
#:   PAW_FILTER_ONLY          H2 -1.093052 / -1.107175 -> -1.063737 / -1.078509
#:                            LiH -0.761814 / -0.769327 -> -0.745498 / -0.755261
#:   PAW_EXACT_LOCAL_ONLY     H2 -1.093020 / -1.107156 -> -1.063765 / -1.078565
#:                            LiH -0.762302 / -0.769787 -> -0.745941 / -0.755682
#:   PAW_BEFORE_BASIS_FILTER  H2 -1.093013 / -1.107148 -> -1.063758 / -1.078557
#:                            LiH -0.762339 / -0.769822 -> -0.745976 / -0.755714
#:
#: Re-measured 2026-10-13: the molecular default kinetic operator became
#: spectral (HISTORY.md "Phase N step 3: the spectral molecular default").
#: The stencil had under-estimated the kinetic energy: H2 rises by 8.8 mHa,
#: LiH by 17.4 mHa.  Old -> new, rhf / adapt:
#:   PAW                      H2 -1.063691 / -1.078462 -> -1.054916 / -1.068832
#:                            LiH -0.745519 / -0.755279 -> -0.728080 / -0.740849
#:   PAW_FILTER_ONLY          H2 -1.063737 / -1.078509 -> -1.054962 / -1.068879
#:                            LiH -0.745498 / -0.755261 -> -0.728060 / -0.740832
#:   PAW_EXACT_LOCAL_ONLY     H2 -1.063765 / -1.078565 -> -1.054969 / -1.068908
#:                            LiH -0.745941 / -0.755682 -> -0.728270 / -0.741042
#:   PAW_BEFORE_BASIS_FILTER  H2 -1.063758 / -1.078557 -> -1.054962 / -1.068900
#:                            LiH -0.745976 / -0.755714 -> -0.728303 / -0.741072
#:
#: Re-measured 2026-10-25: the hydrogen dataset was regenerated with an
#: augmentation radius of 1.0 Bohr (it was 1.3) and an empty p channel, so H
#: has 8 projectors per atom instead of 2.  The s channel is unchanged in
#: kind; the dataset is ghost-free (phases < 0.007 rad).  H2 rises by ~0.9
#: mHa and LiH moves by less than 1 mHa.  Old -> new, rhf / adapt:
#:   PAW                      H2 -1.054916 / -1.068832 -> -1.054076 / -1.068077
#:                            LiH -0.728080 / -0.740849 -> -0.727676 / -0.740494
#:   PAW_FILTER_ONLY          H2 -1.054962 / -1.068879 -> -1.053951 / -1.067952
#:                            LiH -0.728060 / -0.740832 -> -0.727634 / -0.740461
#:   PAW_EXACT_LOCAL_ONLY     H2 -1.054969 / -1.068908 -> -1.053949 / -1.067991
#:                            LiH -0.728270 / -0.741042 -> -0.728544 / -0.741325
#:   PAW_BEFORE_BASIS_FILTER  H2 -1.054962 / -1.068900 -> -1.054000 / -1.068047
#:                            LiH -0.728303 / -0.741072 -> -0.728325 / -0.741133
PAW = {"H2": {"rhf": -1.054076, "adapt": -1.068077},
       "LiH": {"rhf": -0.727676, "adapt": -0.740494}}
#: The filter alone (``exact_local_potential = False``).
PAW_FILTER_ONLY = {
    "H2": {"rhf": -1.053951, "adapt": -1.067952},
    "LiH": {"rhf": -0.727634, "adapt": -0.740461}}
#: The exact local potential alone (``filter_cutoff=False``).
PAW_EXACT_LOCAL_ONLY = {
    "H2": {"rhf": -1.053949, "adapt": -1.067991},
    "LiH": {"rhf": -0.728544, "adapt": -0.741325}}
#: Neither: ``filter_cutoff=False`` and ``exact_local_potential = False`` -- the
#: pre-2026-09-20 model.  Not wrong, just grid-sampled throughout; pinned so
#: both flips are auditable.
PAW_BEFORE_BASIS_FILTER = {
    "H2": {"rhf": -1.054000, "adapt": -1.068047},
    "LiH": {"rhf": -0.728325, "adapt": -0.741133}}
#: The family's **default basis** since 2026-09-20: the four tables above with
#: the first zeta *confined* (``energy_shift = 0.1`` eV; H 1s at
#: 6.68 Bohr, Li 2s at 11.20).  A mild confinement lowers a minimal basis's
#: energy -- free-atom orbitals are too diffuse for a molecule -- by 19.3 mHa
#: for H2 and 1.5 mHa for LiH.  ``{"energy_shift": None}`` reproduces ``PAW``.
#: Re-measured 2026-09-23 with the rebuilt library (see ``PAW`` above):
#:   H2  adapt -1.128495 -> -1.127975   LiH adapt -0.781824 -> -0.779022
#: and 2026-09-27 (regenerated library, as above):
#:   H2  adapt -1.127975 -> -1.128000   LiH adapt -0.779022 -> -0.779032
#: and when the confinement shift became measured from the dataset's own free
#: level:  H2 adapt -1.128000 -> -1.127986   LiH adapt -0.779032 -> -0.779033
#: and 2026-09-30 (libraries rebuilt 2026-09-28/30, as above):
#:   H2  adapt -1.127986 -> -1.127967   LiH adapt -0.779033 -> -0.779013
#: and 2026-10-05 (spectral Coulomb kernel, regenerated libraries, as above):
#:   H2  adapt -1.127967 -> -1.126352   LiH adapt -0.779013 -> -0.776997
#: and 2026-10-11 (W2, the K21 lithium): LiH adapt -0.776997 -> -0.769610,
#: and its ``rhf`` bound, which the shift left below the ADAPT energy,
#: re-measured with the confined basis: -0.773118 -> -0.761345.  With this
#: lithium the confinement lowers LiH by only 0.26 mHa.
#: The H2 ``rhf`` entry was only an upper bound (``PIN_TOL``) and not
#: re-measured until 2026-10-12, when the exact spheres moved both entries
#: (as above; ``rhf`` with the confined basis):
#:   H2  rhf -1.118043 -> -1.080995, adapt -1.126352 -> -1.096169
#:   LiH rhf -0.761345 -> -0.745614, adapt -0.769610 -> -0.756290
#: and 2026-10-13, the spectral kinetic default (as above):
#:   H2  rhf -1.080995 -> -1.067378, adapt -1.096169 -> -1.081798
#:   LiH rhf -0.745614 -> -0.726425, adapt -0.756290 -> -0.739036
#: With the spectral operator the confinement still lowers H2 (by 13.0
#: mHa) but raises LiH (by 1.8 mHa): the free-atom lithium orbitals are no
#: longer too diffuse for this minimal basis.
#: and 2026-10-25 (the regenerated hydrogen dataset, as above):
#:   H2  rhf -1.067378 -> -1.066706, adapt -1.081798 -> -1.081229
#:   LiH rhf -0.726425 -> -0.726037, adapt -0.739036 -> -0.738683
PAW_DEFAULT = {"H2": {"rhf": -1.066706, "adapt": -1.081229},
               "LiH": {"rhf": -0.726037, "adapt": -0.738683}}
#: Same table before the 2026-09-17 fix (do not restore -- they are wrong).
PAW_BEFORE_COMPENSATION_ATTRACTION = {
    "H2": {"rhf": -1.053292, "adapt": -1.067402},
    "LiH": {"rhf": -0.760451, "adapt": -0.768954}}
PIN_TOL = 2e-3
#: sqrt(4 pi): the monopole moment is a Y_00 coefficient (see TestOverlap).
SQRT_4PI = np.sqrt(4.0 * np.pi)

_GENERATED: dict = {}


def generated(symbol: str) -> PAWDataset:
    """Freshly generated dataset, cached.

    Nonrelativistic: the checks below are properties of the PAW transformation, not
    of the relativistic treatment of the reference atom."""
    if symbol not in _GENERATED:
        _GENERATED[symbol] = generate_paw(symbol, relativity="none")
    return _GENERATED[symbol]


_CHECKS: dict = {}


def checks(symbol: str, l: int) -> dict:
    """The levels of one channel of the generated dataset and the scattering phases of the
    whole diagnosis, cached per dataset."""
    if symbol not in _CHECKS:
        _CHECKS[symbol] = diagnose(generated(symbol), levels=True)
    found = _CHECKS[symbol]
    return {"spectrum": found["levels"][l], "phases": found["phases"][l],
            "found": found}


def _fci(hamiltonian) -> float:
    m = hamiltonian.map_to_qubits("jordan_wigner").to_matrix()
    return float(np.linalg.eigvalsh(0.5 * (m + m.conj().T)).min())


def _unconfined(basis):
    """``basis`` with the confinement switched off unless it says otherwise.

    This file validates the PAW *machinery* -- the transformation, the
    projections, the augmentation, the four auditable energy tables above --
    and all of that was established, and is pinned, with the dataset's own
    free-atom partial wave as the first zeta.  Since 2026-09-20 the family's
    default basis is a **confined** one (``energy_shift = 0.1`` eV);
    `TestDefaultBasis` pins that, and `test_paw_energy_shift.py`
    tests the confinement itself.
    """
    options = {"name": basis} if isinstance(basis, str) else dict(basis)
    options.setdefault("energy_shift", None)
    return options


def _total(integrals, n_electrons) -> float:
    """RHF total energy: electronic + ion-ion + the frozen one-center constant."""
    rhf = integrals.hartree_fock(n_electrons)
    return (rhf.electronic_energy + integrals.nuclear_repulsion
            + getattr(integrals, "constant_energy", 0.0))


# --------------------------------------------------------------------------- #
# (a) Atomic validation on the radial grid.
# --------------------------------------------------------------------------- #

CHANNELS = [("H", 0), ("C", 0), ("O", 0), ("O", 1)]


class TestAtomic:
    @pytest.mark.parametrize("symbol, l", CHANNELS)
    def test_projectors_are_dual_to_the_smooth_waves(self, symbol, l):
        pp = generated(symbol)
        channel = pp.channels[l]
        assert isinstance(channel, PAWChannel)
        assert len(channel.projectors) == 2 == len(channel.pseudo_waves) \
            == len(channel.ae_waves) == len(channel.raw_projectors)
        # <p_i|phi~_j> = delta_ij, recomputed here from the stored tables.
        r = pp.r
        duality = np.array([[np.trapezoid(p * f * r * r, r)
                             for f in channel.pseudo_waves]
                            for p in channel.projectors])
        print(f"\n{symbol} l={l}: duality error (stored) "
              f"{channel.duality_error:.1e}, from the tables "
              f"{np.abs(duality - np.eye(2)).max():.1e}")
        assert channel.duality_error < 1e-8
        # Trapezoid, kink at r_c: 1.8e-4 for hydrogen's 1.0 Bohr sphere.
        assert np.abs(duality - np.eye(2)).max() < 3e-4
        # Projectors live inside r_c.
        outside = r > channel.r_cut
        for p in channel.projectors + channel.raw_projectors:
            assert np.abs(p[outside]).max() == 0.0
            assert np.abs(p[~outside]).max() > 1e-3

    @pytest.mark.parametrize("symbol, l", CHANNELS)
    def test_overlap_correction_is_nonzero_and_positive(self, symbol, l):
        """Not norm-conserving -- that is the point of PAW -- with a positive
        definite ``q`` so the overlap operator is bounded below by one."""
        pp = generated(symbol)
        channel = pp.channels[l]
        q = np.asarray(channel.overlap_correction)
        print(f"\n{symbol} l={l}: q = {np.round(q, 5).tolist()}, "
              f"min eig S = {channel.overlap_minimum:.3f}")
        assert q.shape == (2, 2) and np.allclose(q, q.T)
        assert abs(q[0, 0]) > 1e-3
        assert np.linalg.eigvalsh(q).min() > 0.0
        assert channel.overlap_minimum >= 1.0 - 1e-9
        # q = deficit x the all-electron inner Gram matrix, by construction.
        r_in = pp.r[pp.r <= channel.r_cut]
        gram = np.array([[np.trapezoid(a[:r_in.size] * b[:r_in.size] * r_in ** 2,
                                       r_in) for b in channel.ae_waves]
                         for a in channel.ae_waves])
        assert np.allclose(q, pp.norm_deficit * gram, atol=2e-4)

    @pytest.mark.parametrize("symbol, l", CHANNELS)
    def test_one_center_matrices_are_consistent(self, symbol, l):
        """D^scr = B + eps q equals Delta T + Delta V^scr and is symmetric
        (the generalized Wronskian identity of the partial waves)."""
        channel = generated(symbol).channels[l]
        dT, dV = channel.kinetic_difference, channel.potential_difference
        print(f"\n{symbol} l={l}: asymmetry {channel.asymmetry:.1e}, "
              f"|D - dT - dV| {channel.consistency_error:.1e}, "
              f"dT = {np.round(dT, 4).tolist()}")
        assert channel.asymmetry < 1e-6
        assert channel.consistency_error < 1e-10
        assert np.allclose(channel.coupling_screened, dT + dV, atol=1e-10)
        assert np.allclose(dT, dT.T) and np.allclose(dV, dV.T)
        assert np.abs(dT).max() > 1e-2                    # genuinely nonzero
        # Unscreening removes exactly the Hartree screening of the augmentation.
        pp = generated(symbol)
        assert np.allclose(channel.coupling,
                           channel.coupling_screened
                           - channel.overlap_correction * pp.hartree_screening)

    @pytest.mark.parametrize("symbol, l", CHANNELS)
    def test_generalized_eigenproblem_reproduces_the_bound_state(self, symbol, l):
        """(T + v~ + sum |p> D <p|) c = eps (1 + sum |p> q <p|) c gives eps_1
        as its lowest eigenvalue, with no ghost state and the next state a
        box state above zero."""
        pp = generated(symbol)
        e1 = pp.channels[l].eigenvalue
        spectrum = checks(symbol, l)["spectrum"]
        print(f"\n{symbol} l={l}: eps_1 = {e1:+.6f}, spectrum "
              f"{np.round(spectrum, 6).tolist()}, error {spectrum[0] - e1:+.1e} Ha")
        assert abs(spectrum[0] - e1) < 1e-4
        assert spectrum[1] > 0.0
        assert paw_spectrum(pp, l, n_states=2) == pytest.approx(spectrum, abs=1e-9)

    @pytest.mark.parametrize("symbol, l", CHANNELS)
    def test_the_channel_scatters_like_the_atom(self, symbol, l):
        """The smooth logarithmic derivative against the all-electron one over the
        windows around the two references: the phase errors (rad) are the generator's
        own acceptance (0.05 near the reference, 0.3 farther out), and the diagnosis
        finds nothing wrong with the dataset."""
        c = checks(symbol, l)
        near, far = c["phases"]
        print(f"\n{symbol} l={l}: phase error {near:.1e} / {far:.1e} rad")
        assert near < 0.05 and far < 0.3
        assert c["found"]["wrong_phases"] == {} and c["found"]["ghosts"] == {}

    def test_reference_waves_reconstruct_exactly(self):
        """Duality makes the bound smooth wave map onto the all-electron one
        exactly (weights delta_i1)."""
        pp = generated("O")
        for l in (0, 1):
            channel = pp.channels[l]
            r = pp.r[(pp.r > 0.0) & (pp.r <= 8.0)]
            u = np.interp(r, pp.r, channel.pseudo_radial) * r
            rec = reconstruct_ae(pp, l, r, u)
            ae = np.interp(r, pp.r, channel.ae_waves[0]) * r
            assert np.abs(rec - ae).max() < 1e-6

    @pytest.mark.parametrize("symbol", ["H", "C", "O"])
    def test_unscreening_and_compensation(self, symbol):
        pp = generated(symbol)
        r = pp.r
        shell = 4.0 * np.pi * r * r
        # Smooth valence + compensation is neutral with the ion.
        n_smooth = np.trapezoid(pp.valence_density * shell, r)
        from mandacaru import _native
        from mandacaru.pseudopotentials.native_context import context

        g = np.asarray(_native.paw_shape_function(context(), r, pp.compensation_radius, 0))
        g = g / (4.0 * np.pi)                             # the engine's L=0 shape carries 4 pi
        # Trapezoid on the radial grid: 1.2e-6 off for hydrogen's 1.0 Bohr sphere.
        assert np.trapezoid(g * shell, r) == pytest.approx(1.0, abs=5e-6)
        assert n_smooth + pp.compensation_charge == pytest.approx(
            pp.valence_charge, abs=1e-3)
        assert pp.compensation_charge > 1e-3
        assert np.all(g[r >= pp.compensation_radius] == 0.0)
        # The ionic local potential tends to -Z_ion/r.
        far = r > 20.0
        assert np.allclose(pp.v_local[far], -pp.valence_charge / r[far],
                           atol=2e-3)
        assert np.isfinite(pp.v_local_screened).all()
        # Reference bookkeeping: the constant is exactly the difference.
        e = pp.energies
        assert e["reference_valence"] - e["pseudo_atom"] == pytest.approx(
            pp.one_center_energy, abs=1e-12)
        print(f"\n{symbol}: Q^ = {pp.compensation_charge:+.5f} e, E_1c = "
              f"{pp.one_center_energy:+.6f} Ha, reference valence "
              f"{e['reference_valence']:+.6f} Ha, omitted core-valence xc "
              f"{e['core_valence_xc_omitted']:+.4f} Ha")
        if symbol == "H":
            assert np.all(pp.core_density == 0.0)
        else:
            assert np.trapezoid(pp.core_density * shell, r) == pytest.approx(
                pp.atomic_number - pp.valence_charge, abs=1e-3)
            inside = r < pp.compensation_radius
            assert np.abs(pp.smooth_core_density[inside]).max() \
                < np.abs(pp.core_density[inside]).max()
            assert np.allclose(pp.smooth_core_density[~inside],
                               pp.core_density[~inside])

    def test_projector_bases_are_the_same_operator(self):
        pp = generated("O")
        for l in (0, 1):
            inside = pp.r <= pp.channels[l].r_cut       # projectors vanish beyond
            dual, D, q = pp.projector_set(l, "dual")
            raw, D_raw, q_raw = pp.projector_set(l, "raw")
            dual = [p[inside] for p in dual]
            raw = [p[inside] for p in raw]
            op_dual = sum(D[i, j] * np.outer(dual[i], dual[j])
                          for i in range(2) for j in range(2))
            op_raw = sum(D_raw[i, j] * np.outer(raw[i], raw[j])
                         for i in range(2) for j in range(2))
            assert np.abs(op_dual - op_raw).max() < 1e-9 * np.abs(op_dual).max()
            ov_dual = sum(q[i, j] * np.outer(dual[i], dual[j])
                          for i in range(2) for j in range(2))
            ov_raw = sum(q_raw[i, j] * np.outer(raw[i], raw[j])
                         for i in range(2) for j in range(2))
            assert np.abs(ov_dual - ov_raw).max() < 1e-9 * np.abs(ov_dual).max()
        with pytest.raises(ValueError, match="unknown projector basis"):
            pp.projector_set(0, "other")

    def test_projectors_reach_the_local_radius(self):
        """Aluminum at its own cutoffs (s 2.68, p 3.43 Bohr) and the default
        local radius: r_cl follows the larger cutoff, the s projectors extend
        past their own r_cut to it, and the channel is still dual, symmetric
        and ghost-free (at the old repair's 5 Ha; the repair now shortens
        aluminum's local radius below its s cutoff instead)."""
        pp = generate_paw("Al", ghosts="keep", norm_deficit=0.0,
                          local_shift=5.0, relativity="none")
        s, p = pp.channels[0], pp.channels[1]
        assert pp.r_cut_local > s.r_cut
        assert pp.projector_radius(0) == pytest.approx(pp.r_cut_local)
        assert pp.projector_radius(1) == pytest.approx(p.r_cut)
        shell = (pp.r > s.r_cut + 0.05) & (pp.r < pp.r_cut_local - 0.05)
        assert np.abs(np.asarray(s.projectors)[:, shell]).max() > 0.0
        assert np.all(np.asarray(s.projectors)[:, pp.r > pp.r_cut_local] == 0.0)
        assert s.duality_error < 1e-10 and s.asymmetry < 1e-6
        assert diagnose(pp)["ghosts"] == {}

    def test_the_local_potential_alone_scatters_like_the_atom(self):
        """K21: aluminum's d channel has no projectors.  Repaired with a 5 Ha
        raise over 0.9 x its p cutoff (the old search's choice) it scattered
        d 0.52 rad off and put fcc Al at 4.48 Angstrom; the search now
        shortens the local radius at no raise instead (HISTORY.md,
        2026-10-06, "K21 diagnosed")."""
        repaired = generated("Al")
        assert repaired.local_shift == 0.0
        assert repaired.r_cut_local < 0.9 * repaired.channels[1].r_cut
        errors = diagnose(repaired)["unprojected"]
        assert set(errors) == {2}
        assert errors[2] < 0.2                      # the generator's acceptance
        raised = generate_paw("Al", ghosts="keep", norm_deficit=0.0,
                              local_shift=5.0, relativity="none")
        assert diagnose(raised)["unprojected"][2] > 0.4

    def test_the_local_potential_alone_binds_no_ghost(self):
        """Iron's p channel has no projectors.  With r_cl at 0.9 x the compact
        3d cutoff, the bare all-electron well bound a p level at -4.6 Ha
        (4p: -0.05); with r_cl at the 4s cutoff it binds nothing extra.

        The valence-only iron: the shipped one now carries its 3s and 3p
        (the semicore table, 2026-10-21), whose p channel has
        projectors, so this guards the channel without projectors by
        building iron with ``semicore_subshells=()`` (old -4.6 Ha ghost
        reproduced: -4.58 Ha)."""
        valence_only = {"semicore_subshells": ()}
        # The historical construction: the compact local radius and no raise
        # of the local potential (iron's 10 Ha entry went with the semicore
        # p channel; here the default raise is 0).
        old = generate_paw("Fe", ghosts="keep", norm_deficit=0.0,
                           r_cut_local=0.828, local_shift=0.0,
                           relativity="none", **valence_only)
        assert diagnose(old)["ghosts"][1] < -4.0
        new = generate_paw("Fe", ghosts="keep", norm_deficit=0.0,
                           relativity="none", **valence_only)
        assert diagnose(new)["ghosts"] == {}

    def test_semicore_sodium_carries_its_2s2p_in_valence(self, tmp_path):
        """K21: with an s channel alone no local potential was right for
        sodium's p scattering.  The 2s2p in valence give the p channel a
        bound reference and its projectors, and the s channel two bound
        references (2s, 3s), with no raise of the local potential."""
        from mandacaru.pseudopotentials.io import library_file

        pp = generated("Na")
        assert pp.valence_charge == 9.0
        assert pp.semicore_subshells == ((2, 0), (2, 1))
        assert pp.channels[0].occupations == [2.0, 1.0]
        assert pp.channels[1].occupations == [6.0, 0.0]
        assert pp.local_shift == 0.0
        assert diagnose(pp)["ghosts"] == {}
        # The 2s, 3s and 2p levels of the generalized problem are the
        # atom's, in order: nothing extra below or between them.
        s_levels = paw_spectrum(pp, 0, n_states=3)
        assert s_levels[:2] == pytest.approx(
            pp.channels[0].reference_energies, abs=1e-3)
        assert paw_spectrum(pp, 1)[0] == pytest.approx(
            pp.channels[1].reference_energies[0], abs=1e-3)
        errors = diagnose(pp)["unprojected"]
        assert max(errors.values()) < 0.1
        # The cutoff comes from the 3s, not from the compact 2s.
        assert 2.0 < pp.channels[0].r_cut < 2.6
        path = save_pseudopotential(
            pp, library_file("Na", str(tmp_path), "json"), format="json")
        loaded = load_pseudopotential(path)
        assert loaded.semicore_subshells == pp.semicore_subshells
        assert [c.occupations for c in loaded.channels.values()] == \
            [c.occupations for c in pp.channels.values()]

    def test_the_semicore_basis_has_every_occupied_state(self):
        """2s, 3s and 2p each get a function; only the 3s is split and
        polarized (DZP: 2s, 3s x 2, 2p, a p shell from the 3s)."""
        from mandacaru.pseudopotentials.orbitals import (occupied_states,
                                                         pseudo_basis)
        pp = generated("Na")
        states = [(s.n, s.l, s.index, s.semicore) for s in occupied_states(pp)]
        assert states == [(2, 0, 0, True), (3, 0, 1, False), (2, 1, 0, True)]
        minimal, _ = pseudo_basis(["Na"], [[0, 0, 0]], {"Na": pp}, size="SZ")
        assert [(f.n, f.l) for f in minimal] == [(2, 0), (3, 0)] + [(2, 1)] * 3
        assert minimal[1].eigenvalue == pytest.approx(
            pp.channels[0].reference_energies[1])
        dzp, _ = pseudo_basis(["Na"], [[0, 0, 0]], {"Na": pp}, size="DZP")
        assert [f.l for f in dzp] == [0, 0, 0] + [1] * 6

    def test_a_semicore_channel_confines_its_valence_state(self):
        """The second eigenstate of sodium's s channel is the 3s: confined at
        0.1 eV it sits 0.1 eV above the free 3s, far outside the 2s's wall."""
        from mandacaru.pseudopotentials.confinement import (confined_orbital,
                                                            free_energy)
        pp = generated("Na")
        valence = confined_orbital(pp, 0, 0.1, state=1)
        assert valence.achieved_shift == pytest.approx(0.1, abs=1e-3)
        assert free_energy(pp, 0, 1) == pytest.approx(
            pp.channels[0].reference_energies[1], abs=2e-3)
        assert valence.r_c > 2.0 * confined_orbital(pp, 0, 0.1).r_c

    def test_semicore_subshells_are_validated(self):
        """The shell must be a filled core subshell directly below the valence one of its l
        (the engine's partition of the reference atom)."""
        pp = generate_paw("Na", ghosts="keep", relativity="none",
                          semicore_subshells=[(2, 1), (2, 0)])
        assert pp.semicore_subshells == ((2, 0), (2, 1))
        with pytest.raises(ValueError, match="not an occupied core"):
            generate_paw("Na", ghosts="keep", relativity="none", semicore_subshells=[(3, 1)])
        with pytest.raises(ValueError, match="not directly below"):
            generate_paw("Na", ghosts="keep", relativity="none", semicore_subshells=[(1, 0)])

    @staticmethod
    def _nodes(values):
        values = np.asarray(values)
        values = values[np.abs(values) > 1e-6 * np.abs(values).max()]
        return int(np.count_nonzero(np.diff(np.sign(values))))

    def test_lithium_p_channel_is_anchored_at_its_2p(self):
        """Lithium carries an empty p channel (DEFAULT_HIGHEST_CHANNEL).
        Anchored at the 2s energy its smooth waves gained a node and
        couplings of -2 Ha; the generator falls back to the atom's own 2p
        level (-0.042 Ha)."""
        pp = generated("Li")
        channel = pp.channels[1]
        assert channel.reference_energies[0] == pytest.approx(-0.0416,
                                                              abs=2e-3)
        assert np.abs(np.asarray(channel.coupling)).max() < 0.5
        r = np.asarray(pp.r)
        inside = (r > 0.0) & (r <= channel.r_cut)
        for wave in channel.pseudo_waves:
            assert self._nodes(np.asarray(wave)[inside]) == 0

    def test_lithium_p_scatters_like_the_atom(self):
        found = diagnose(generated("Li"))
        assert found["phases"][1][0] < 0.01 and found["phases"][1][1] < 0.05
        assert found["wrong_phases"] == {}

    def test_beryllium_p_scatters_like_the_atom_and_stays_out_of_the_basis(
            self):
        """Without p projectors beryllium's local potential scattered p
        0.28-0.42 rad off at every ghost-free local radius (PBE Be was
        flagged; PBE BeO 10.7 % long).  Its p channel (DEFAULT_HIGHEST_CHANNEL)
        brings that to 0.02 rad; the bound 2p stays out of the basis
        (SCATTERING_ONLY_CHANNELS), so a minimal Be keeps one orbital."""
        from mandacaru.pseudopotentials.orbitals import basis_states

        pp = generated("Be")
        assert 1 in pp.channels and not pp.defects
        near, far = diagnose(pp)["phases"][1]
        assert near < 0.02 and far < 0.05
        assert [state.l for state in basis_states(pp)] == [0]

    def test_lithium_smooth_core_carries_the_core_valence_overlap(self):
        """Pseudized at r_g = 2.6 Bohr the smooth core held 0.02 electrons;
        at DEFAULT_CORE_RADII["Li"] = 1.5 it holds 0.37."""
        pp = generated("Li")
        assert pp.nlcc["r_nlcc"] == pytest.approx(1.5, abs=0.01)
        r = np.asarray(pp.r)
        electrons = np.trapezoid(pp.smooth_core_density * 4 * np.pi * r * r, r)
        assert 0.3 < electrons < 0.45
        assert np.all(pp.smooth_core_density >= 0.0)

    def test_calcium_keeps_its_d_anchor(self):
        """Calcium's d channel pseudizes without a node at its 4s energy:
        the fallback leaves it alone."""
        pp = generated("Ca")
        assert pp.channels[2].reference_energies[0] == pytest.approx(
            pp.channels[0].reference_energies[1], abs=1e-6)

    def test_a_second_reference_can_be_placed_per_channel(self):
        """``second_reference={l: energy}`` moves one channel's scattering
        reference to an absolute energy and leaves the others at the bound
        state plus ``energy_offset``; one at or below the bound state is
        refused (DEFAULT_SECOND_REFERENCES, the 4f of W-Hg)."""
        default = generate_paw("O", ghosts="keep", relativity="none")
        moved = generate_paw("O", ghosts="keep", relativity="none",
                             second_reference={1: 0.5})
        assert moved.channels[1].reference_energies[1] == pytest.approx(0.5)
        assert moved.channels[1].reference_energies[0] == pytest.approx(
            default.channels[1].reference_energies[0])
        assert np.allclose(moved.channels[0].reference_energies,
                           default.channels[0].reference_energies)
        with pytest.raises(ValueError, match="not above the bound state"):
            generate_paw("O", ghosts="keep", relativity="none",
                         second_reference={1: -2.0})

    def test_report_runs_generated_and_loaded(self):
        text = report_paw(generated("H"))
        assert "l=0x2" in text and "phase=" in text and "E_1c" in text
        assert "2 projectors" in repr(generated("H").channels[0])
        loaded = get_paw("H")
        assert "phase=" in report_paw(loaded)          # the engine solves the atom again


# --------------------------------------------------------------------------- #
# The shipped library.
# --------------------------------------------------------------------------- #

#: One element from each population the 2026-09-24 ghost census found
#: ghosted (first rows, d, f, p).
#: The generator's floor for the smallest eigenvalue of the overlap operator.
OVERLAP_MINIMUM = 0.1
#: How far below its reference a channel's lowest level may sit before it is a ghost
#: (a file has no all-electron atom to settle it, and its stride-4 tables move a level by up to
#: 1e-3 Ha, Cu; a ghost sits tens of mHa and deeper).
GHOST_BAND = 5e-3
#: An excited level this far below zero in a compact channel is a ghost (the engine's margin).
EXCITED_GHOST_MARGIN = 0.1


def file_only_ghosts(pp) -> dict:
    """Channels with one occupied reference whose lowest level lies below it: the ghost test
    that needs no all-electron atom (``ghost_levels``)."""
    levels = ghost_levels(pp)
    return {l: levels[l][0] - c.reference_energies[0] for l, c in pp.channels.items()
            if sum(1 for o in c.occupations if o > 0.0) == 1
            and levels[l][0] < c.reference_energies[0] - GHOST_BAND}


GHOST_SWEEP = ["B", "Na", "Cl", "Fe", "Cu", "Ga", "Ba", "La", "W", "Bi"]


class TestLibrary:
    def test_shipped_elements(self):
        # The mandacaru-paw checkout ($MANDACARU_PAW_PATH/lda, all 92
        # elements); at least the six generated in-repo must be there.
        try:
            shipped = available_elements(paw_library_path())
        except LibraryPathError:
            pytest.skip("MANDACARU_PAW_PATH is not configured here")
        assert {"C", "F", "H", "Li", "N", "O"} <= set(shipped)

    def test_protactinium_and_uranium_reference_an_occupied_6d(self):
        # The aufbau 5f / 7s2 reference of Pa and U leaves no 6d, so the d
        # channel scattered 0.26-0.52 rad off; the override gives it a bound
        # reference, and the shipped files carry a d channel (the table of the
        # overrides is the engine's, pinned by crates/mandacaru-core/tests/paw_recipe.rs).
        for symbol in ("Pa", "U"):
            try:
                pp = get_paw(symbol)
            except LibraryPathError:
                pytest.skip("MANDACARU_PAW_PATH is not configured here")
            assert 2 in pp.channels

    @pytest.mark.parametrize("symbol", ["H", "Li", "C", "N", "O", "F"])
    def test_shipped_file_loads(self, symbol):
        pp = get_paw(symbol)
        assert isinstance(pp, PAWDataset) and pp.family == "paw-lcao"
        for l, channel in pp.channels.items():
            assert len(pp.projectors[l]) == 2
            assert np.asarray(pp.coupling[l]).shape == (2, 2)
            assert np.asarray(pp.overlap_correction[l]).shape == (2, 2)
            assert channel.duality_error < 1e-8
            # Not ">= 1": an augmentation charge q_ii may be negative (the
            # K21 lithium's second s partial wave carries 1e-4 more norm
            # than its all-electron wave, q_22 = -0.0001, so its minimum is
            # 0.99911).  What matters is that the PAW transformation stays
            # far from singular, the generator's own floor.
            assert channel.overlap_minimum >= OVERLAP_MINIMUM

    @pytest.mark.parametrize("symbol", ["H", "Li", "O"])
    def test_shipped_file_has_no_ghost(self, symbol):
        pp = get_paw(symbol)
        for l, channel in pp.channels.items():
            spectrum = paw_spectrum(pp, l)
            if symbol == "H" and l == 1:
                # Hydrogen's empty p channel is projected at the 1s level;
                # it binds nothing there, so the lowest p state is a positive
                # one: no state at or below the reference is the whole test.
                assert spectrum[0] > channel.eigenvalue + 0.1
                continue
            assert abs(spectrum[0] - channel.eigenvalue) < 1e-4
            assert spectrum[1] > 0.0

    @pytest.mark.parametrize("symbol", GHOST_SWEEP)
    def test_no_shipped_channel_holds_a_ghost(self, symbol):
        """An extra state below the reference, with the reference level
        displaced to second place (the engine's ghost test, here without the all-electron
        atom: :func:`file_only_ghosts`), in one element of every population the 2026-09-24
        census found ghosted.  The whole library is the slow test below."""
        assert file_only_ghosts(get_paw(symbol)) == {}

    @pytest.mark.parametrize("symbol", ["Tc", "Ta", "W", "Re", "Os", "Ir",
                                        "Pt", "Au", "Hg"])
    def test_no_compact_channel_binds_an_excited_ghost(self, symbol):
        """The 4f of W-Hg, the 4p of Tc and the semicore 5p of Ta bound a
        second level 1 Ha above their state (W -0.15 ... Hg -2.5 Ha, Tc
        -0.60, Ta -0.46) that the atom does not have: their second
        reference sat below zero.  None of these atoms binds a second f or
        p level below -EXCITED_GHOST_MARGIN, so a file-only check suffices
        (DEFAULT_SECOND_REFERENCES)."""
        pp = get_paw(symbol)
        for l, channel in pp.channels.items():
            if sum(1 for o in channel.occupations if o > 0.0) != 1:
                continue
            assert paw_spectrum(pp, l)[1] > -EXCITED_GHOST_MARGIN, l

    def test_lithium_smooth_waves_are_nodeless(self):
        """The PBE lithium shipped until 2026-09-27 had both 2s partial waves
        cross zero at 1.22 Bohr, inside r_c = 2.60: a legitimate minimum of
        the norm-constrained fit, invisible to every atomic check, and fatal
        in a molecule (``DEFAULT_N_BESSEL_BY_DATASET``).  The shipped
        LDA-SR lithium is held to the same shape."""
        pp = get_paw("Li")
        r = np.asarray(pp.r)
        channel = pp.channels[0]
        inside = (r > 0.0) & (r < channel.r_cut)
        for wave in channel.pseudo_waves:
            values = np.asarray(wave)[inside]
            signs = np.sign(values[np.abs(values) > 1e-3 * np.abs(values).max()])
            assert np.all(signs == signs[0])

    @pytest.mark.slow
    @pytest.mark.xfail(strict=True, raises=ValueError, reason=(
        "the native molecular RHF refuses a core other than empty or a closed 1s "
        "('native PAW frozen-core exchange currently requires ...'): a Cu molecule is not "
        "reachable (PAW_MIGRATION_LOG.md, A7 / S2; open item of the molecular route)"))
    def test_copper_binds_hydrogen(self):
        """The PBE copper built at deficit 0.1 collapsed CuH as the hydrogen
        entered its sphere: -1555 eV at 1.30 Angstrom, 108 eV below 1.60 (RHF
        SZ, grid_spacing = 0.20).  At deficit 0, like the LDA one, 1.60 is the lower of
        the two, by a fraction of an eV."""
        energies = {}
        for distance in (1.3, 1.6):
            atoms = Atoms("CuH", positions=[[0, 0, 0], [0, 0, distance]],
                          cell=[8.0] * 3)
            atoms.center()
            atoms.calc = Mandacaru(method="rhf", grid_spacing=0.20, paw_dataset="lda-sr",
                                   basis={"name": "PAW-LCAO", "size": "SZ"})
            energies[distance] = atoms.get_potential_energy()
        assert energies[1.6] < energies[1.3] < energies[1.6] + 5.0

    @pytest.mark.parametrize("symbol", ["Fe", "Co", "Ni", "Cu", "Sn"])
    def test_the_s_projectors_represent_an_intruding_hydrogen(self, symbol):
        """Expand the l = 0 part of a hydrogen 1s centered at the sphere's
        edge in the s channel's partial waves and projectors, and compare with
        the function itself inside the sphere.  At 8 Bessel functions (and for
        copper at deficit 0.1) the expansion missed it by 2.4-13 times its own
        norm (tin: 150), the metal's one-center density held two to three
        times the molecule's electrons, and the hydride collapsed by 150-580 eV
        (DEFAULT_N_BESSEL_BY_DATASET, DEFAULT_CUTOFFS, DEFAULT_NORM_DEFICITS);
        now it is under one."""
        # The search refuses a construction above the tolerance (1); these five were
        # the first repaired.
        assert intruding_s_miss(get_paw(symbol)) < 1.0


    @pytest.mark.slow
    @pytest.mark.parametrize("symbol", library_elements())
    def test_no_shipped_dataset_holds_a_ghost(self, symbol):
        assert file_only_ghosts(get_paw(symbol)) == {}

    def test_shipped_h_matches_a_fresh_generation(self):
        # The shipped library is scalar-relativistic, so the fresh dataset is
        # the generation default; the native atom solver cannot build it yet.
        shipped, fresh = get_paw("H"), generate_paw("H")
        assert np.allclose(shipped.coupling[0], fresh.coupling[0], rtol=1e-6)
        assert np.allclose(shipped.overlap_correction[0],
                           fresh.overlap_correction[0], rtol=1e-6)
        assert shipped.one_center_energy == pytest.approx(
            fresh.one_center_energy, rel=1e-6)
        assert shipped.r.size * 4 == pytest.approx(fresh.r.size, abs=4)

    def test_a_file_without_the_norm_correction_is_refused(self, tmp_path):
        """A file written before the charge and the M-weighted norm were distinguished
        has no ``norm_correction``, and its overlap correction holds the norm where the
        charge belongs.  It cannot be repaired on load, and the engine no longer guesses:
        it refuses the file, naming the key (no recovery, no warning)."""
        import json

        from mandacaru.pseudopotentials.io import library_file

        path = save_pseudopotential(generated("O"), library_file("O", str(tmp_path), "json"),
                                    format="json")
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        for entry in payload["channels"].values():
            entry.pop("norm_correction")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        with pytest.raises(ValueError, match="norm_correction"):
            load_pseudopotential(path)

    def test_a_missing_dataset_names_the_family(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="PAW-LCAO"):
            get_paw("Xe", paw_dataset=str(tmp_path))


# --------------------------------------------------------------------------- #
# (d) io round trip, family resolution, dry run.
# --------------------------------------------------------------------------- #

class TestIO:
    def test_a_flagged_dataset_warns_whoever_loads_it(self, tmp_path):
        """A ``ghosts="flag"`` dataset records what its generator could not
        remove; the file keeps it, and every load warns."""
        import copy

        from mandacaru.pseudopotentials.defects import GhostStateWarning
        pp = copy.copy(generated("H"))
        pp.defects = {"ghosts": {0: -5.03}, "phases": {3: (0.09, 0.09)}}
        assert "GHOSTED" in repr(pp) and "SCATTERING OFF" in repr(pp)
        path = save_pseudopotential(pp, tmp_path / "H.parquet")
        with pytest.warns(GhostStateWarning, match="l=0 ghost -5.03 Ha"):
            back = load_pseudopotential(path)
        assert back.defects == {"ghosts": {0: -5.03}, "phases": {3: (0.09, 0.09)}}
        with warnings.catch_warnings():
            warnings.simplefilter("error", GhostStateWarning)
            clean = load_pseudopotential(save_pseudopotential(
                generated("H"), tmp_path / "clean.parquet"))
        assert clean.defects == {}

    @pytest.mark.parametrize("fmt", ["json", "parquet"])
    def test_round_trip(self, tmp_path, fmt):
        pp = generated("O")
        path = save_pseudopotential(pp, tmp_path / f"O.{fmt}")
        assert detect_format(path) == fmt
        back = load_pseudopotential(path)
        assert isinstance(back, PAWDataset)
        assert back.family == "paw-lcao"
        assert back.symbol == "O" and back.valence_charge == 6.0
        for name in ("r_cut_local", "local_shift", "compensation_radius",
                     "compensation_charge", "hartree_screening",
                     "one_center_energy", "q_cut", "energy_offset",
                     "norm_deficit"):
            assert getattr(back, name) == pytest.approx(getattr(pp, name))
        assert back.energies == pytest.approx(pp.energies)
        for name in ("r", "v_local", "v_local_screened", "valence_density",
                     "core_density", "smooth_core_density"):
            assert np.allclose(getattr(back, name), getattr(pp, name),
                               rtol=1e-9, atol=1e-12)
        for l in (0, 1):
            a, b = pp.channels[l], back.channels[l]
            assert b.n == a.n and b.r_cut == a.r_cut
            assert b.reference_energies == a.reference_energies
            for name in ("overlap_correction", "kinetic_difference",
                         "potential_difference", "coupling",
                         "coupling_screened", "vanderbilt"):
                assert np.allclose(getattr(b, name), getattr(a, name))
            for name in ("ae_waves", "pseudo_waves", "projectors",
                         "raw_projectors"):
                for x, y in zip(getattr(b, name), getattr(a, name)):
                    assert np.allclose(x, y, rtol=1e-9, atol=1e-12)
            assert b.duality_error == a.duality_error
            assert b.overlap_minimum == a.overlap_minimum
            assert np.allclose(back.overlap_correction[l],
                               pp.overlap_correction[l])
        # Save-load-save is idempotent.
        again = load_pseudopotential(save_pseudopotential(back,
                                                          tmp_path / f"O2.{fmt}"))
        assert np.array_equal(again.projectors[1][1], back.projectors[1][1])

class TestResolution:
    @pytest.mark.parametrize("name", ["paw-lcao", "PAW-LCAO", " Paw-LCAO "])
    def test_names(self, name):
        assert resolve_family(name) is PSEUDO_FAMILIES["paw-lcao"]
        assert lookup_family(name) is PSEUDO_FAMILIES["paw-lcao"]
        assert "paw-lcao" in family_names()

    def test_unknown_family_lists_them_all(self):
        with pytest.raises(ValueError, match="'paw-lcao'.*'upaw-lcao'"):
            resolve_family("gth")


    @pytest.mark.parametrize("method", ["vqe", "adapt-vqe"])
    def test_drivers_accept_the_family(self, method):
        assert Mandacaru(method=method, basis="PAW-LCAO").basis == "PAW-LCAO"
        sized = {"name": "PAW-LCAO", "size": "DZ"}
        assert Mandacaru(method=method, basis=sized).basis == sized

    @pytest.mark.xfail(strict=True, reason=(
        "no code path raises \"'frozen' core is redundant\" any more: a frozen core on a "
        "PAW basis is accepted silently (PAW_MIGRATION_LOG.md, A7 / S2: a lost refusal "
        "of the molecular cutover, reported to the owner)"))
    @pytest.mark.parametrize("method", ["vqe", "adapt-vqe"])
    def test_a_frozen_core_is_refused_on_a_paw_basis(self, method):
        with pytest.raises(ValueError, match="'frozen' core is redundant"):
            Mandacaru(method=method, basis="PAW-LCAO",
                      active_space={"frozen": "auto"})


# --------------------------------------------------------------------------- #
# (b) Compensation matrices and the augmented overlap.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# (c) Molecular energies; hardness.
# --------------------------------------------------------------------------- #

class TestMolecular:


    # LiH only since 2026-10-13: with the spectral kinetic default the
    # filter moves H2's RHF energy by 1.6e-7 Ha (-1.1 meV on the stencil),
    # below the six decimals the tables are written at.  Hydrogen's
    # functions are nearly band-limited already, and the spectral operator,
    # unlike the stencil, is not biased by what the filter removes.
    @pytest.mark.parametrize("name", ["LiH"])
    def test_the_filter_default_is_not_a_no_op(self, name):
        """The pinned tables really differ: the filter moves the energy."""
        old = PAW_BEFORE_BASIS_FILTER[name]
        shift = HARTREE_TO_EV * (PAW_FILTER_ONLY[name]["rhf"] - old["rhf"])
        print(f"\n{name}: PAW filter on - off = {1000 * shift:+.1f} meV (RHF)")
        assert abs(shift) > 1e-4               # the default is not a no-op

    @pytest.mark.parametrize("name", sorted(SYSTEMS))
    def test_adapt_vqe(self, name):
        factory, h = SYSTEMS[name]
        atoms = factory()
        atoms.calc = Mandacaru(method="adapt-vqe", basis="PAW-LCAO", grid_spacing=h,
                               pool="qeb", max_iterations=4,
                               profile=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            atoms.get_potential_energy()
        result = atoms.calc.result
        energy = result.in_units("Ha")          # the pins are Hartree; results eV
        print(f"\n{name}: PAW ADAPT-VQE {result.optimal_energy:.4f} eV "
              f"({energy:.6f} Ha)")
        assert atoms.calc.n_qubits == 4
        assert result.energy_unit == "eV"
        assert np.isfinite(result.optimal_energy)
        # Through the calculator with `basis="PAW-LCAO"`: the family's default
        # basis, confined since 2026-09-20.
        assert energy == pytest.approx(PAW_DEFAULT[name]["adapt"], abs=1e-5)
        assert energy <= PAW_DEFAULT[name]["rhf"] + PIN_TOL
        # ... and the confinement is what moved it from the unconfined pin:
        # down for H2.  For LiH it moves the energy up since the spectral
        # kinetic default (2026-10-13, see PAW_DEFAULT), so only the move
        # is checked there.
        if name == "H2":
            assert energy < PAW[name]["adapt"]
        else:
            assert abs(energy - PAW[name]["adapt"]) > 1e-4


# --------------------------------------------------------------------------- #
# The spheres of a Hartree-Fock Hamiltonian.
# --------------------------------------------------------------------------- #

class TestExactSpheres:
    """What the spheres carry for Hartree-Fock: the one-center Coulomb tensor
    and the core's exchange instead of the dataset's frozen functional
    (HISTORY.md, 2026-10-12)."""

    def test_hydrogen_is_free_of_self_interaction(self):
        """One electron: the energy is its orbital energy, and with the
        spectral kinetic operator that is -1/2 Ha to the basis error."""
        atoms = Atoms("H", positions=[(0.0, 0.0, 0.0)])
        atoms.center(vacuum=5.0)
        atoms.calc = Mandacaru(method="uhf", spin=True, grid_spacing=0.2,
                               basis={"name": "PAW-LCAO", "size": "QZ2P",
                                      "energy_shift": None})
        energy = atoms.get_potential_energy() / HARTREE_TO_EV
        epsilon = float(np.min(atoms.calc.result.scf.mo_energies[0]))
        assert energy == pytest.approx(epsilon, abs=1e-8)
        assert -0.5 < energy < -0.499
