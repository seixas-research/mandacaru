# -*- coding: utf-8 -*-
# file: test/experimental/test_paw.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The PAW family (experimental): Bloechl's projector augmented-wave method
in its frozen-core, linearized one-center form.

Atomic checks (radial, cheap) on freshly generated H, Li and O -- projector
duality, the generalized eigenproblem, the reconstruction of the all-electron
wave (the transformation the method is named after), logarithmic derivatives
-- and the molecular checks on H2 (0.74 A, h = 0.25 A, 6 A cell) and LiH
(1.6 A, h = 0.30 / 0.25 A, 8 A cell) against the Troullier-Martins and
ONCVPSP energies pinned in ``test_ncpp_family`` / ``test_oncvpsp``.  The
overlap correction makes these the first runs through the augmented overlap
``S + C q C^dagger`` with a nonzero ``q``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from ase import Atoms

from mandacaru.algorithms import ADAPTVQE, Mandacaru, VQE
from mandacaru.algorithms._hamiltonian_from_atoms import (
    build_basis_hamiltonian, resolve_basis, resolve_pseudo_basis)
from mandacaru.algorithms.dry_run import estimate_qubits
from mandacaru.core.hamiltonian import projector_blocks
from mandacaru.pseudopotentials import (
    PSEUDO_FAMILIES, PAWChannel, PAWDataset, PAWIntegrals, check_paw_channel,
    compensation_coulomb, compensation_potential, compensation_shape,
    family_names, generate_paw, get_oncv, get_paw, get_pseudopotential,
    load_pseudopotential, log_derivative_ae, log_derivative_paw,
    lookup_family, paw_eigenstate, paw_library_path,
    paw_spectrum, reconstruct_ae, report_paw, resolve_family,
    save_pseudopotential)
from mandacaru.pseudopotentials import paw
from mandacaru.pseudopotentials.io import (available_elements,
                                                       default_library_path,
                                                       detect_format)
from mandacaru.units import HARTREE_TO_EV

# --------------------------------------------------------------------------- #
# Test systems (identical to test_ncpp_family / test_oncvpsp).
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

#: Troullier-Martins and ONCVPSP energies (Hartree) pinned by the sibling
#: test files, same grids.
TM = {"H2": {"rhf": -1.061096245397, "adapt": -1.075333384673},
      "LiH": {"rhf": -0.729555411706, "adapt": -0.740838985744}}
ONCV = {"H2": {"rhf": -1.044179, "adapt": -1.058561},
        "LiH": {"rhf": -0.774343, "adapt": -0.782341}}
#: PAW energies with the shipped library (H: rc 1.30, deficit 0.05,
#: Delta 0.5; Li: rc 2.60, deficit 0.02, Delta 0.5).  Re-measured 2026-09-17,
#: when the compensation charge gained the electron-ion attraction it was
#: missing (``one_body_augmentation``): it had been paying the Hartree
#: repulsion of the augmentation charge with no attraction to the ion, so
#: every PAW energy was too high by an amount that scales with that charge.
#: The values before that fix are kept for provenance in
#: ``before_compensation_attraction``.
PAW = {"H2": {"rhf": -1.095396, "adapt": -1.109168},
       "LiH": {"rhf": -0.773001, "adapt": -0.780800}}
#: Same table before the 2026-09-17 fix (do not restore -- they are wrong).
PAW_BEFORE_COMPENSATION_ATTRACTION = {
    "H2": {"rhf": -1.053292, "adapt": -1.067402},
    "LiH": {"rhf": -0.760451, "adapt": -0.768954}}
PIN_TOL = 2e-3
#: sqrt(4 pi): the monopole moment is a Y_00 coefficient (see TestOverlap).
SQRT_4PI = np.sqrt(4.0 * np.pi)
#: Agreement asked of the three families: 0.1 Ha (2.7 eV).
FAMILY_TOL = 0.1

_GENERATED: dict = {}


def generated(symbol: str) -> PAWDataset:
    """Freshly generated dataset (with its all-electron atom), cached."""
    if symbol not in _GENERATED:
        _GENERATED[symbol] = generate_paw(symbol)
    return _GENERATED[symbol]


_CHECKS: dict = {}


def checks(symbol: str, l: int) -> dict:
    """``check_paw_channel`` of the generated dataset, cached per channel."""
    if (symbol, l) not in _CHECKS:
        _CHECKS[(symbol, l)] = check_paw_channel(generated(symbol), l)
    return _CHECKS[(symbol, l)]


def _fci(hamiltonian) -> float:
    m = hamiltonian.map_to_qubits("jordan_wigner").to_matrix()
    return float(np.linalg.eigvalsh(0.5 * (m + m.conj().T)).min())


def _build(name, basis, h=None):
    factory, default_h = SYSTEMS[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return build_basis_hamiltonian(factory(), basis, None,
                                       default_h if h is None else h, 0, None)


def _total(integrals, n_electrons) -> float:
    """RHF total energy: electronic + ion-ion + the frozen one-center constant."""
    rhf = integrals.hartree_fock(n_electrons)
    return (rhf.electronic_energy + integrals.nuclear_repulsion
            + getattr(integrals, "constant_energy", 0.0))


# --------------------------------------------------------------------------- #
# (a) Atomic validation on the radial grid.
# --------------------------------------------------------------------------- #

CHANNELS = [("H", 0), ("Li", 0), ("O", 0), ("O", 1)]


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
        assert np.abs(duality - np.eye(2)).max() < 1e-4     # trapezoid, kink at r_c
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
        c = checks(symbol, l)
        print(f"\n{symbol} l={l}: eps_1 = {e1:+.6f}, spectrum "
              f"{np.round(c['spectrum'], 6).tolist()}, error "
              f"{c['eigenvalue_error']:+.1e} Ha")
        assert abs(c["eigenvalue_error"]) < 1e-4
        assert c["spectrum"][1] > 0.0
        assert c["nodes"] == 0

    @pytest.mark.parametrize("symbol, l", CHANNELS)
    def test_reconstruction_of_the_all_electron_wave(self, symbol, l):
        """phi = phi~ + sum_i (phi_i - phi~_i) <p_i|phi~> applied to the lowest
        smooth eigenfunction gives the all-electron orbital everywhere."""
        c = checks(symbol, l)
        print(f"\n{symbol} l={l}: |phi_rec - phi_AE| = "
              f"{c['reconstruction_error']:.1e} (smooth wave deviates by "
              f"{c['smooth_deviation']:.2e})")
        assert c["reconstruction_error"] < 1e-4
        assert c["smooth_deviation"] > 1e-2          # the smooth wave differs
        assert c["tail_error"] < 1e-6                # ... only inside r_c

    @pytest.mark.parametrize("symbol, l", CHANNELS)
    def test_logarithmic_derivatives(self, symbol, l):
        pp = generated(symbol)
        e1, e2 = pp.channels[l].reference_energies
        errors = checks(symbol, l)["log_derivative_errors"]
        line = "  ".join(f"L({e:+.3f})={la:+.4f} dL={err:.1e}"
                         for e, (err, la) in errors.items())
        print(f"\n{symbol} l={l}: {line}")
        assert errors[e1][0] < 1e-3
        assert errors[e2][0] < 1e-3
        mid_error, mid_l = errors[0.5 * (e1 + e2)]
        assert mid_error / max(1.0, abs(mid_l)) < 2e-2

    def test_log_derivative_functions_agree_by_construction(self):
        pp = generated("H")
        e1 = pp.channels[0].eigenvalue
        l_ae = log_derivative_ae(pp.r, pp.atom.v_effective, 0, e1,
                                 pp.channels[0].r_cut, 1.0)
        assert abs(log_derivative_paw(pp, 0, e1) - l_ae) < 1e-5

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

    @pytest.mark.parametrize("symbol", ["H", "Li", "O"])
    def test_unscreening_and_compensation(self, symbol):
        pp = generated(symbol)
        r = pp.r
        shell = 4.0 * np.pi * r * r
        # Smooth valence + compensation is neutral with the ion.
        n_smooth = np.trapezoid(pp.valence_density * shell, r)
        g = pp.compensation_shape(r)
        assert np.trapezoid(g * shell, r) == pytest.approx(1.0, abs=1e-6)
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

    def test_compensation_shape_functions(self):
        r_g = 1.3
        r = np.linspace(0.0, 3.0, 3001)
        g = compensation_shape(r, r_g)
        v = compensation_potential(r, r_g)
        assert np.trapezoid(4.0 * np.pi * r * r * g, r) == pytest.approx(1.0, abs=1e-8)
        assert np.allclose(v[r > r_g], 1.0 / r[r > r_g])
        assert np.isfinite(v[0]) and v[0] == pytest.approx(315.0 / 128.0 / r_g)
        # Poisson: the analytic potential is the Hartree potential of g.
        from mandacaru.basis.atomic_solver import hartree_potential
        rr = r[1:]
        assert np.abs(hartree_potential(rr, g[1:]) - v[1:]).max() < 1e-4
        # Disjoint spheres interact as point charges; the self-energy is
        # 4 pi int g V r^2 dr; overlapping spheres are in between.
        assert compensation_coulomb(r_g, r_g, 3.0) == pytest.approx(1.0 / 3.0)
        self_energy = np.trapezoid(4.0 * np.pi * r * r * g * v, r)
        assert compensation_coulomb(r_g, r_g, 0.0) == pytest.approx(
            self_energy, rel=1e-6)
        # Inside the sphere V_g(r) < 1/r, so overlapping charges interact
        # less than point charges (and less than at coincidence).
        overlapping = compensation_coulomb(r_g, r_g, 1.4)
        assert 0.9 / 1.4 < overlapping < 1.0 / 1.4 < self_energy
        # Continuity at the touching distance.
        assert compensation_coulomb(r_g, r_g, 2 * r_g - 1e-9) == pytest.approx(
            1.0 / (2 * r_g), rel=1e-6)

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

    def test_free_waves_are_refused_when_the_overlap_is_singular(self):
        """Without the norm condition the overlap operator of Li goes
        singular; the generator says so instead of shipping a ghost."""
        with pytest.raises(RuntimeError, match="overlap operator"):
            generate_paw("Li", norm_deficit=None, atom=generated("Li").atom)

    def test_report_runs_generated_and_loaded(self):
        text = report_paw(generated("H"))
        assert "l=0x2" in text and "|dL|" in text and "E_1c" in text
        assert "2 projectors" in repr(generated("H").channels[0])
        loaded = get_paw("H")
        assert loaded.atom is None
        assert "|dL|" not in report_paw(loaded)        # needs the atom


# --------------------------------------------------------------------------- #
# The shipped library.
# --------------------------------------------------------------------------- #

class TestLibrary:
    def test_shipped_elements(self):
        # The external paw repository (all 92 elements) linked into
        # library/paw; at least the six generated in-repo must be there.
        shipped = available_elements(paw_library_path())
        if not shipped:
            pytest.skip("external paw repository not linked on this machine")
        assert {"C", "F", "H", "Li", "N", "O"} <= set(shipped)
        assert "paw" not in available_elements(default_library_path())

    @pytest.mark.parametrize("symbol", ["H", "Li", "C", "N", "O", "F"])
    def test_shipped_file_loads(self, symbol):
        pp = get_paw(symbol)
        assert isinstance(pp, PAWDataset) and pp.family == "paw"
        for l, channel in pp.channels.items():
            assert len(pp.projectors[l]) == 2
            assert np.asarray(pp.coupling[l]).shape == (2, 2)
            assert np.asarray(pp.overlap_correction[l]).shape == (2, 2)
            assert channel.duality_error < 1e-8
            assert channel.overlap_minimum >= 1.0 - 1e-9

    @pytest.mark.parametrize("symbol", ["H", "Li", "O"])
    def test_shipped_file_has_no_ghost(self, symbol):
        pp = get_paw(symbol)
        for l, channel in pp.channels.items():
            spectrum = paw_spectrum(pp, l)
            assert abs(spectrum[0] - channel.eigenvalue) < 1e-4
            assert spectrum[1] > 0.0

    def test_shipped_h_matches_a_fresh_generation(self):
        shipped, fresh = get_paw("H"), generated("H")
        assert np.allclose(shipped.coupling[0], fresh.coupling[0], rtol=1e-6)
        assert np.allclose(shipped.overlap_correction[0],
                           fresh.overlap_correction[0], rtol=1e-6)
        assert shipped.one_center_energy == pytest.approx(
            fresh.one_center_energy, rel=1e-6)
        assert shipped.r.size * 4 == pytest.approx(fresh.r.size, abs=4)

    def test_loaders_refuse_the_other_families(self, tmp_path):
        with pytest.raises(ValueError, match="belongs to family 'paw'"):
            PSEUDO_FAMILIES["ncpp"].get("H", paw_library_path())
        with pytest.raises(ValueError, match="not 'oncvpsp'"):
            get_oncv("H", paw_library_path())
        with pytest.raises(ValueError, match="not 'paw'"):
            get_paw("H", default_library_path())
        with pytest.raises(FileNotFoundError, match="PAW"):
            get_paw("Xe", directory=str(tmp_path))


# --------------------------------------------------------------------------- #
# (d) io round trip, family resolution, dry run.
# --------------------------------------------------------------------------- #

class TestIO:
    @pytest.mark.parametrize("fmt", ["json", "parquet"])
    def test_round_trip(self, tmp_path, fmt):
        pp = generated("O")
        path = save_pseudopotential(pp, tmp_path / f"O.{fmt}")
        assert detect_format(path) == fmt
        back = load_pseudopotential(path)
        assert isinstance(back, PAWDataset)
        assert back.family == "paw" and back.atom is None
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

    def test_other_families_load_unchanged(self):
        tm = get_pseudopotential("O")
        assert type(tm).__name__ == "PseudoPotential" and tm.family == "ncpp"
        oncv = get_oncv("O")
        assert type(oncv).__name__ == "ONCVPseudoPotential"
        assert oncv.family == "oncvpsp" and len(oncv.projectors[0]) == 2

    def test_a_plain_record_named_paw_keeps_the_tm_layout(self, tmp_path):
        import copy
        pp = copy.copy(get_pseudopotential("H"))
        pp.family = "paw"
        for fmt in ("json", "parquet"):
            back = load_pseudopotential(
                save_pseudopotential(pp, tmp_path / f"H.{fmt}"))
            assert type(back).__name__ == "PseudoPotential"
            assert back.family == "paw"


class TestResolution:
    @pytest.mark.parametrize("name", ["paw", "PAW", " Paw "])
    def test_names(self, name):
        assert resolve_family(name) is PSEUDO_FAMILIES["paw"]
        assert lookup_family(name) is PSEUDO_FAMILIES["paw"]
        assert "paw" in family_names()

    def test_unknown_family_lists_all_three(self):
        with pytest.raises(ValueError, match="'ncpp'.*'oncvpsp'.*'paw'"):
            resolve_family("gth")

    def test_spec_and_basis_selection(self):
        spec = PSEUDO_FAMILIES["paw"]
        assert spec.norm_conserving is False and spec.aliases == ()
        assert spec.label == "PAW"
        assert spec.options == ("size", "split_norm", "directory",
                                "projector_basis")
        assert spec.get("H").family == "paw"
        assert spec.generate is not None and spec.build is paw.build_paw
        name, options = resolve_basis({"name": "PAW", "size": "DZ",
                                       "projector_basis": "raw"})
        family, options = resolve_pseudo_basis(name, options, ["H"])
        assert family is spec
        assert options == {"size": "DZ", "projector_basis": "raw"}
        with pytest.raises(ValueError, match="cannot mix a pseudopotential"):
            resolve_pseudo_basis("per-element", {"H": "PAW", "Li": "6-31G(d)"},
                                 ["Li", "H"])

    @pytest.mark.parametrize("driver", [VQE, ADAPTVQE])
    def test_drivers_accept_the_family(self, driver):
        assert driver(basis="paw").basis == "paw"
        assert driver(basis={"name": "paw", "size": "DZ"}).basis == \
            {"name": "paw", "size": "DZ"}
        with pytest.raises(ValueError, match="frozen_core is redundant"):
            driver(basis="PAW", frozen_core=True)

    def test_dry_run(self):
        estimate = estimate_qubits(h2(), basis="paw")
        assert estimate.n_qubits == 4 and estimate.num_particles == (1, 1)
        assert any("PAW family" in note for note in estimate.notes)
        assert estimate_qubits(lih(), basis={"name": "paw"}).n_qubits == 4
        dz = estimate_qubits(h2(), basis={"name": "PAW", "size": "DZ"})
        assert dz.n_qubits == 8
        atoms = h2()
        atoms.calc = Mandacaru(method="adapt-vqe", basis="paw",
                               h=H2_H, dry_run=True)
        assert np.isnan(atoms.get_potential_energy())
        assert atoms.calc.dry_run_result.n_qubits == 4
        assert Mandacaru(method="vqe", basis={"name": "PAW", "size": "DZ"},
                         h=LIH_H).dry_run(lih()).n_qubits == 8


# --------------------------------------------------------------------------- #
# (b) The augmented overlap on H2.
# --------------------------------------------------------------------------- #

class TestOverlap:
    def test_augmented_overlap_on_h2(self):
        _H, _p, n_orb, _pr, context = _build("H2", "paw")
        ints = context["integrals"]
        assert isinstance(ints, PAWIntegrals) and context["family"] == "paw"
        assert n_orb == 2 and len(ints.kb_projectors) == 4
        S, S_bare = ints.overlap(), ints.bare_overlap()
        w = np.linalg.eigvalsh(S)
        print(f"\nH2: eig(S_aug) = {np.round(w, 5)}, eig(S_bare) = "
              f"{np.round(np.linalg.eigvalsh(S_bare), 5)}")
        assert w.min() > 0.0
        assert np.abs(S - S.conj().T).max() < 1e-12
        assert np.abs(S - S_bare).max() > 1e-3           # q really acts
        X = ints._lowdin_x()
        assert np.abs(X.conj().T @ S @ X - np.eye(2)).max() < 1e-12
        # The q blocks are nonzero: the smooth partial waves are not
        # norm-conserving.
        Q = ints.nonlocal_overlap_matrix()
        assert Q.shape == (4, 4)
        blocks = projector_blocks(ints.kb_projectors)
        assert all(len(v) == 2 for v in blocks.values())
        for positions in blocks.values():
            block = Q[np.ix_(positions, positions)]
            assert np.abs(block).max() > 1e-3
            assert np.allclose(block, block.T)
        assert ints.nonlocal_overlap is not None
        assert np.abs(ints.kb_nonlocal()).max() > 1e-3
        C = ints.projections()
        assert np.abs(S - (S_bare + C @ Q @ C.conj().T)).max() < 1e-12

    def test_on_site_projections_are_exact(self):
        """<phi~_1|p_i> = delta_i1 on the atom the function sits on, from the
        radial tables rather than the grid; the moments then reproduce q_11."""
        _H, _p, _n, _pr, context = _build("H2", "paw")
        ints = context["integrals"]
        pp = get_paw("H")
        C = ints.projections()
        _f, _D, q_raw = pp.projector_set(0, "raw")
        B = np.asarray(pp.channels[0].vanderbilt)
        # raw basis: <phi~_1|chi_k> = B_1k ; dual basis would be delta_1k.
        # (1e-4: the library tables are decimated to 0.02 Bohr; the grid
        # value of these entries was off by 0.7-3x.)
        for atom in (0, 1):
            cols = [p for p, pr in enumerate(ints.kb_projectors)
                    if pr.atom_index == atom]
            assert np.allclose(C[atom, cols].real, B[0], atol=1e-4)
            assert np.abs(C[atom, cols].imag).max() < 1e-12
        # Moments and their Coulomb matrix are keyed by multipole channel
        # (atom, L, M) since the expansion went beyond the monopole, and they
        # are coefficients of Y_LM: the monopole carries 1/sqrt(4 pi) and the
        # Coulomb matrix the compensating 4 pi, so every product is unchanged.
        moments = ints.compensation_moments()
        assert ints.multipole_channels() == [(0, 0, 0), (1, 0, 0)]  # H: s only
        q11 = pp.overlap_correction[0][0, 0]
        for atom in (0, 1):
            assert moments[(atom, 0, 0)][atom, atom].real * SQRT_4PI \
                == pytest.approx(q11, abs=1e-4)
        assert np.allclose(sum(moments.values()) * SQRT_4PI,
                           C @ ints.nonlocal_overlap_matrix() @ C.conj().T)
        assert ints.constant_energy == pytest.approx(2 * pp.one_center_energy)

    def test_projector_basis_does_not_change_the_energy(self):
        raw = _build("H2", {"name": "paw", "projector_basis": "raw"})
        dual = _build("H2", {"name": "paw", "projector_basis": "dual"})
        e_raw = _total(raw[4]["integrals"], 2)
        e_dual = _total(dual[4]["integrals"], 2)
        assert e_raw == pytest.approx(e_dual, abs=1e-8)
        assert _fci(raw[0]) == pytest.approx(_fci(dual[0]), abs=1e-8)
        assert raw[4]["integrals"].kb_projectors[0].projector_basis == "raw"

    def test_augmented_two_body_tensor(self):
        _H, _p, _n, _pr, context = _build("H2", "paw")
        ints = context["integrals"]
        aug = ints.two_body_augmentation()
        assert aug.shape == (2, 2, 2, 2) and np.abs(aug).max() > 1e-3
        # Hermitian pair densities: <pq|rs> = <rs|pq>^* and <pq|rs> = <qp|sr>.
        assert np.abs(aug - np.conj(aug.transpose(2, 3, 0, 1))).max() < 1e-12
        assert np.abs(aug - aug.transpose(1, 0, 3, 2)).max() < 1e-12
        U = ints.compensation_coulomb()
        d = np.linalg.norm(ints._potentials.nuclei[0][1]
                           - ints._potentials.nuclei[1][1])
        r_g = get_paw("H").compensation_radius
        assert r_g == pytest.approx(1.3, abs=1e-3)
        # 4 pi against the old monopole-only form; see the moments above.
        monopole = 4.0 * np.pi
        assert U[0, 1] == pytest.approx(monopole
                                        * compensation_coulomb(r_g, r_g, d))
        assert U[0, 0] == U[1, 1] > U[0, 1] > monopole / (2 * r_g)


# --------------------------------------------------------------------------- #
# (c) Molecular energies against TM and ONCV; hardness.
# --------------------------------------------------------------------------- #

class TestMolecular:
    @pytest.mark.parametrize("name", sorted(SYSTEMS))
    def test_rhf_and_fci_against_the_other_families(self, name):
        H, particles, n_orb, _profile, context = _build(name, "paw")
        ints = context["integrals"]
        assert n_orb == 2 and particles == (1, 1)
        assert len(ints.kb_projectors) == 4
        e_rhf = _total(ints, context["n_electrons"])
        e_fci = _fci(H)
        print(f"\n{name}: PAW RHF {e_rhf * HARTREE_TO_EV:.4f} eV, FCI "
              f"{e_fci * HARTREE_TO_EV:.4f} eV  ({e_rhf:.6f} / {e_fci:.6f} Ha; "
              f"ONCV RHF {ONCV[name]['rhf']:.6f}, TM RHF {TM[name]['rhf']:.6f}; "
              f"PAW - ONCV = {(e_rhf - ONCV[name]['rhf']) * HARTREE_TO_EV:+.3f} eV, "
              f"PAW - TM = {(e_rhf - TM[name]['rhf']) * HARTREE_TO_EV:+.3f} eV)")
        assert np.isfinite(e_rhf) and np.isfinite(e_fci)
        assert e_fci <= e_rhf + 1e-9
        assert abs(e_rhf - ONCV[name]["rhf"]) < FAMILY_TOL
        assert abs(e_fci - ONCV[name]["adapt"]) < FAMILY_TOL
        assert abs(e_rhf - TM[name]["rhf"]) < FAMILY_TOL
        assert e_rhf == pytest.approx(PAW[name]["rhf"], abs=PIN_TOL)
        assert e_fci == pytest.approx(PAW[name]["adapt"], abs=PIN_TOL)

    @pytest.mark.parametrize("name", sorted(SYSTEMS))
    def test_adapt_vqe(self, name):
        factory, h = SYSTEMS[name]
        atoms = factory()
        atoms.calc = Mandacaru(method="adapt-vqe", basis="paw", h=h,
                               pool="qeb", max_iterations=4,
                               profile=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            atoms.get_potential_energy()
        result = atoms.calc.result
        energy = result.in_units("Ha")          # the pins are Hartree; results eV
        print(f"\n{name}: PAW ADAPT-VQE {result.optimal_energy:.4f} eV "
              f"({energy:.6f} Ha; ONCV {ONCV[name]['adapt']:.6f}, TM "
              f"{TM[name]['adapt']:.6f})")
        assert atoms.calc.n_qubits == 4
        assert result.energy_unit == "eV"
        assert np.isfinite(result.optimal_energy)
        assert energy == pytest.approx(PAW[name]["adapt"], abs=PIN_TOL)
        assert energy <= PAW[name]["rhf"] + PIN_TOL
        assert abs(energy - ONCV[name]["adapt"]) < FAMILY_TOL
        assert abs(energy - TM[name]["adapt"]) < FAMILY_TOL

    @pytest.mark.parametrize("name", sorted(SYSTEMS))
    def test_hardness_at_a_quarter_angstrom(self, name):
        ints = _build(name, "paw", h=0.25)[4]["integrals"]
        basis = ints.resolution_ratios
        projectors = ints.kb_resolution_ratios
        print(f"\n{name} (h=0.25 A): PAW basis T_grid/T_exact = "
              f"{np.round(basis, 3)}, projector norm ratios = "
              f"{np.round(projectors, 3)}")
        assert np.all((basis > 0.75) & (basis < 1.25))
        assert projectors.shape == (4,)
        assert np.all((projectors > 0.75) & (projectors < 1.25))

    def test_lih_is_stable_against_the_grid(self):
        """The on-site projections being exact, the energy drifts smoothly
        with the grid (egg-box) instead of collapsing between nodes."""
        e30 = _total(_build("LiH", "paw", h=0.30)[4]["integrals"], 2)
        e25 = _total(_build("LiH", "paw", h=0.25)[4]["integrals"], 2)
        print(f"\nLiH PAW RHF: h=0.30 {e30:.6f}, h=0.25 {e25:.6f} Ha")
        assert abs(e25 - e30) < 0.02
        assert abs(e25 - ONCV["LiH"]["rhf"]) < FAMILY_TOL

    def test_size_hierarchy_is_variational(self):
        sz = _build("H2", "paw")[4]["integrals"]
        dz = _build("H2", {"name": "paw", "size": "DZ"})
        ints = dz[4]["integrals"]
        assert dz[2] == 4 and len(ints.kb_projectors) == 4
        e_sz = _total(sz, 2)
        e_dz = _total(ints, 2)
        print(f"\nH2 RHF: SZ {e_sz:.6f}  DZ {e_dz:.6f} Ha")
        assert e_dz < e_sz
        assert np.linalg.eigvalsh(ints.overlap()).min() > 0.0

    def test_first_zeta_is_the_smooth_partial_wave(self):
        _H, _p, _n, _pr, context = _build("H2", "paw")
        fn = context["integrals"].basis[0]
        pp = get_paw("H")
        r = np.linspace(0.05, 4.0, 50)
        assert np.allclose(fn.radial(r), np.interp(r, pp.r,
                                                   pp.channels[0].pseudo_radial),
                           atol=2e-4)
