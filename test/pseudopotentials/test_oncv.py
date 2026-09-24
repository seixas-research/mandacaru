# -*- coding: utf-8 -*-
# file: test/experimental/test_oncvpsp.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The ONCVPSP family (experimental): Hamann's optimized norm-conserving
Vanderbilt pseudopotentials with two projectors per channel.

Atomic checks (radial, cheap) on freshly generated H, Li and O; the molecular
checks on H2 (0.74 A, h = 0.25 A, 6 A cell) and LiH (1.6 A, h = 0.30 A, 8 A
cell) against the Troullier-Martins energies pinned in ``test_ncpp_family``.
H and Li carry two s projectors here, so H2/LiH genuinely exercise the 2x2
Vanderbilt blocks of the general separable form.
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest
from ase import Atoms

from mandacaru.algorithms import Mandacaru
from mandacaru.algorithms._hamiltonian_from_atoms import (
    build_basis_hamiltonian, resolve_basis, resolve_pseudo_basis)
from mandacaru.algorithms.dry_run import estimate_qubits
from mandacaru.core.hamiltonian import projector_blocks
from mandacaru.pseudopotentials import (
    PSEUDO_FAMILIES, ONCVChannel, ONCVPseudoPotential, check_oncv_channel,
    diagonalized_projectors, family_names, generate_oncv, get_oncv,
    load_pseudopotential, log_derivative_ae, log_derivative_ps,
    lookup_family, oncv_library_path, radial_spectrum,
    report_oncv, resolve_family, save_pseudopotential)
from mandacaru.pseudopotentials.io import library_elements
from mandacaru.pseudopotentials.io import (available_elements,
                                                       default_library_path,
                                                       detect_format)
from mandacaru.pseudopotentials import oncv

# --------------------------------------------------------------------------- #
# Test systems (identical to test_ncpp_family, whose TM energies we compare to).
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

#: Troullier-Martins energies (Hartree) pinned in test_ncpp_family.py.
TM = {"H2": {"rhf": -1.061096245397, "adapt": -1.075333384673},
      "LiH": {"rhf": -0.729555411706, "adapt": -0.740838985744}}
#: ONCVPSP energies measured 2026-09-23 with the shipped library
#: (H rc = 1.30, Li rc = 2.60 Bohr, Delta = 1 Ha, q_c = 5 Bohr^-1).
#:
#: Re-pinned when scalar-relativistic reference atoms and the nonlinear core
#: correction became the generation defaults:
#:
#:   H2  rhf  -1.044179 -> -1.043772   adapt  -1.058561 -> -1.058151
#:   LiH rhf  -0.774343 -> -0.764921   adapt  -0.782341 -> -0.772577
#:
#: Hydrogen moves by under a milliHartree -- it has no core for the
#: correction to act on, and its relativistic shift is 6.7e-6 Ha.  Lithium
#: moves by 0.010 Ha (0.27 eV), which is the 1s core it does have.
ONCV = {"H2": {"rhf": -1.043772, "adapt": -1.058151},
        "LiH": {"rhf": -0.764921, "adapt": -0.772577}}
PIN_TOL = 2e-3
TM_TOL = 0.05

_GENERATED: dict = {}


def generated(symbol: str) -> ONCVPseudoPotential:
    """Freshly generated potential (with its all-electron atom), cached."""
    if symbol not in _GENERATED:
        _GENERATED[symbol] = generate_oncv(symbol)
    return _GENERATED[symbol]


def _fci(hamiltonian) -> float:
    m = hamiltonian.map_to_qubits("jordan_wigner").to_matrix()
    return float(np.linalg.eigvalsh(0.5 * (m + m.conj().T)).min())


def _build(name, basis):
    factory, h = SYSTEMS[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return build_basis_hamiltonian(factory(), basis, None, h, 0, None)


# --------------------------------------------------------------------------- #
# (a) Atomic validation on the radial grid.
# --------------------------------------------------------------------------- #

CHANNELS = [("H", 0), ("Li", 0), ("O", 0), ("O", 1)]


@pytest.mark.slow
class TestAtomic:
    @pytest.mark.parametrize("symbol, l", CHANNELS)
    def test_channel_reproduces_the_reference(self, symbol, l):
        pp = generated(symbol)
        channel = pp.channels[l]
        assert isinstance(channel, ONCVChannel)
        assert len(channel.projectors) == 2 and len(channel.pseudo_waves) == 2
        checks = check_oncv_channel(pp, l)
        e1, e2 = channel.reference_energies
        print(f"\n{symbol} l={l}: rc={channel.r_cut:.3f} eps=({e1:+.5f}, "
              f"{e2:+.5f}) eps err={checks['eigenvalue_error']:+.1e} "
              f"tail={checks['tail_error']:.1e} norm={checks['norm_matrix_error']:.1e} "
              f"asym={checks['vanderbilt_asymmetry']:.1e} "
              f"E_res={[f'{e:.1e}' for e in checks['residual_kinetic']]} "
              f"D={np.round(channel.coupling, 3).tolist()}")
        # No ghost below eps_1 and eps_1 reproduced as the lowest state.
        assert abs(checks["eigenvalue_error"]) < 1e-4
        assert checks["spectrum"][1] > e1 + 0.05
        # Pseudo wave == all-electron wave beyond rc.
        assert checks["tail_error"] < 1e-6
        # Generalized norm conservation <phi_i|phi_j>_{rc}.
        assert checks["norm_matrix_error"] < 1e-8
        # Symmetric Vanderbilt matrix (the consequence of the above).
        assert checks["vanderbilt_asymmetry"] < 1e-6
        assert np.allclose(channel.vanderbilt, channel.vanderbilt.T)
        assert np.allclose(channel.coupling @ channel.vanderbilt, np.eye(2),
                           atol=1e-8)
        assert checks["nodes"] == 0
        assert all(np.isfinite(e) and e >= 0 for e in checks["residual_kinetic"])

    @pytest.mark.parametrize("symbol, l", CHANNELS)
    def test_logarithmic_derivatives(self, symbol, l):
        """Transferability: L_ps(E) == L_ae(E) at eps_2 and in between."""
        pp = generated(symbol)
        checks = check_oncv_channel(pp, l)
        e1, e2 = pp.channels[l].reference_energies
        errors = checks["log_derivative_errors"]
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
                                 pp.channels[0].r_cut, 1.0,
                                 pp.relativity)
        assert abs(log_derivative_ps(pp, 0, e1) - l_ae) < 1e-5

    def test_projectors_vanish_beyond_rc_and_are_two_per_channel(self):
        pp = generated("O")
        for l, channel in pp.channels.items():
            outside = pp.r > channel.r_cut
            for chi in channel.projectors:
                assert np.abs(chi[outside]).max() == 0.0
                assert np.abs(chi[~outside]).max() > 1e-3
            assert np.asarray(pp.coupling[l]).shape == (2, 2)
            assert abs(pp.coupling[l][0, 1]) > 1e-3        # genuinely coupled

    def test_diagonalized_form_is_the_same_operator(self):
        pp = generated("H")
        chis, energies = diagonalized_projectors(pp, 0)
        raw = sum(D_ij * np.outer(ci, cj)
                  for (i, ci) in enumerate(pp.projectors[0])
                  for (j, cj) in enumerate(pp.projectors[0])
                  for D_ij in [pp.coupling[0][i, j]])
        diag = sum(d * np.outer(c, c) for c, d in zip(chis, energies))
        assert np.abs(raw - diag).max() < 1e-9 * np.abs(raw).max()

    def test_report_runs_generated_and_loaded(self):
        text = report_oncv(generated("H"))
        assert "l=0x2" in text and "|dL|" in text
        assert "2 projectors" in repr(generated("H").channels[0])
        loaded = get_oncv("H")
        assert loaded.atom is None
        assert "|dL|" not in report_oncv(loaded)     # needs the atom

    def test_local_potential_is_smooth_and_ionic(self):
        pp = generated("O")
        v = pp.v_local_screened
        assert np.isfinite(v).all() and abs(v[1] - v[0]) < 1e-3   # V'(0) = 0
        outside = pp.r > pp.r_cut_local
        assert np.abs(v[outside] - pp.atom.v_effective[outside]).max() == 0.0
        far = pp.r > 20.0
        assert np.allclose(pp.v_local[far], -pp.valence_charge / pp.r[far],
                           atol=2e-3)
        assert pp.local_l == -1 and pp.kb_energies == {}


# --------------------------------------------------------------------------- #
# The shipped library.
# --------------------------------------------------------------------------- #

#: One element from each population the 2026-09-24 ghost census found
#: ghosted (PAW-LCAO only, both families, ONCVPSP only, first rows, d, f, p).
GHOST_SWEEP = ["B", "Na", "Cl", "Fe", "Cu", "Ga", "Ba", "La", "W", "Bi"]


class TestLibrary:
    def test_shipped_elements(self):
        # The external oncvpsp repository (all 92 elements) linked into
        # library/oncvpsp; at least the six generated in-repo must be there.
        shipped = available_elements(oncv_library_path())
        if not shipped:
            pytest.skip("external oncvpsp repository not linked on this machine")
        assert {"C", "F", "H", "Li", "N", "O"} <= set(shipped)
        # A subdirectory: the TM library listing is untouched.
        assert "oncvpsp" not in available_elements(default_library_path())

    @pytest.mark.parametrize("symbol", ["H", "Li", "C", "N", "O", "F"])
    def test_shipped_file_loads_with_two_projectors(self, symbol):
        pp = get_oncv(symbol)
        assert isinstance(pp, ONCVPseudoPotential) and pp.family == "oncvpsp"
        for l, channel in pp.channels.items():
            assert len(pp.projectors[l]) == 2
            assert np.asarray(pp.coupling[l]).shape == (2, 2)
            assert len(channel.residual_kinetic) == 2

    @pytest.mark.parametrize("symbol", ["H", "Li", "O"])
    def test_shipped_file_has_no_ghost(self, symbol):
        pp = get_oncv(symbol)
        for l, channel in pp.channels.items():
            spectrum = radial_spectrum(pp, l)
            assert abs(spectrum[0] - channel.eigenvalue) < 1e-4

    @pytest.mark.parametrize("symbol", GHOST_SWEEP)
    def test_no_shipped_channel_holds_a_ghost(self, symbol):
        """An extra state below the reference, with the reference level
        displaced to second place (:func:`~mandacaru.pseudopotentials.oncv.
        ghost_errors`), in one element of every population the 2026-09-24
        census found ghosted.  The whole library is the slow test below."""
        from mandacaru.pseudopotentials import oncv
        assert oncv.ghost_errors(get_oncv(symbol), oncv._oncv_levels) == {}

    @pytest.mark.slow
    @pytest.mark.parametrize("symbol", library_elements())
    def test_no_shipped_dataset_holds_a_ghost(self, symbol):
        from mandacaru.pseudopotentials import oncv
        assert oncv.ghost_errors(get_oncv(symbol), oncv._oncv_levels) == {}

    def test_shipped_h_matches_a_fresh_generation(self):
        shipped, fresh = get_oncv("H"), generated("H")
        assert np.allclose(shipped.coupling[0], fresh.coupling[0], rtol=1e-6)
        assert shipped.channels[0].residual_kinetic == pytest.approx(
            fresh.channels[0].residual_kinetic, rel=1e-6)
        assert shipped.r.size * 4 == pytest.approx(fresh.r.size, abs=4)

    def test_loaders_refuse_the_other_family(self, tmp_path):
        with pytest.raises(ValueError, match="belongs to family 'oncvpsp'"):
            PSEUDO_FAMILIES["ncpp"].get("H", oncv_library_path())
        with pytest.raises(ValueError, match="not 'oncvpsp'"):
            get_oncv("H", default_library_path())
        with pytest.raises(FileNotFoundError, match="ONCVPSP"):
            get_oncv("Xe", directory=str(tmp_path))


# --------------------------------------------------------------------------- #
# (d) io round trip, family resolution, dry run.
# --------------------------------------------------------------------------- #

class TestIO:
    @pytest.mark.parametrize("fmt", ["json", "parquet"])
    def test_round_trip(self, tmp_path, fmt):
        pp = generated("H")
        path = save_pseudopotential(pp, tmp_path / f"H.{fmt}")
        assert detect_format(path) == fmt
        back = load_pseudopotential(path)
        assert isinstance(back, ONCVPseudoPotential)
        assert back.family == "oncvpsp" and back.atom is None
        assert back.symbol == "H" and back.valence_charge == 1.0
        assert back.r_cut_local == pp.r_cut_local and back.q_cut == pp.q_cut
        assert back.energy_offset == pp.energy_offset
        for name in ("r", "v_local", "v_local_screened", "valence_density"):
            assert np.allclose(getattr(back, name), getattr(pp, name),
                               rtol=1e-9, atol=1e-12)
        a, b = pp.channels[0], back.channels[0]
        assert b.n == a.n and b.r_cut == a.r_cut and b.occupation == a.occupation
        assert b.reference_energies == a.reference_energies
        assert b.residual_kinetic == a.residual_kinetic
        assert np.allclose(b.coupling, a.coupling) and np.allclose(b.vanderbilt,
                                                                  a.vanderbilt)
        for x, y in zip(b.pseudo_waves, a.pseudo_waves):
            assert np.allclose(x, y, rtol=1e-9, atol=1e-12)
        for x, y in zip(b.projectors, a.projectors):
            assert np.allclose(x, y, rtol=1e-9, atol=1e-12)
        assert np.allclose(back.projectors[0][1], pp.projectors[0][1])
        # Save-load-save is idempotent.
        again = load_pseudopotential(save_pseudopotential(back,
                                                          tmp_path / f"H2.{fmt}"))
        assert np.array_equal(again.projectors[0][0], back.projectors[0][0])

    def test_tm_files_still_load_as_tm(self):
        from mandacaru.pseudopotentials import get_pseudopotential
        pp = get_pseudopotential("O")
        assert type(pp).__name__ == "PseudoPotential" and pp.family == "ncpp"
        assert isinstance(pp.projectors[0], np.ndarray)


class TestResolution:
    @pytest.mark.parametrize("name", ["oncvpsp", "ONCVPSP", "oncv", "Oncv"])
    def test_names(self, name):
        assert resolve_family(name) is PSEUDO_FAMILIES["oncvpsp"]
        assert lookup_family(name) is PSEUDO_FAMILIES["oncvpsp"]
        assert "oncv" in family_names() and "oncvpsp" in family_names()

    def test_spec_and_basis_selection(self):
        spec = PSEUDO_FAMILIES["oncvpsp"]
        assert spec.norm_conserving and spec.aliases == ("oncv",)
        assert spec.label == "ONCVPSP"
        assert spec.get("H").family == "oncvpsp"
        assert spec.generate is not None
        name, options = resolve_basis({"name": "oncvpsp", "size": "DZP"})
        family, options = resolve_pseudo_basis(name, options, ["H"])
        assert family is spec and options == {"size": "DZP"}
        assert resolve_pseudo_basis(*resolve_basis("6-31G(d)"), ["H"])[0] is None
        with pytest.raises(ValueError, match="cannot mix a pseudopotential"):
            resolve_pseudo_basis("per-element", {"H": "oncv", "Li": "6-31G(d)"},
                                 ["Li", "H"])

    @pytest.mark.parametrize("method", ["vqe", "adapt-vqe"])
    def test_drivers_accept_the_family(self, method):
        assert Mandacaru(method=method, basis="oncv").basis == "oncv"
        sized = {"name": "oncvpsp", "size": "DZ"}
        assert Mandacaru(method=method, basis=sized).basis == sized
        with pytest.raises(ValueError, match="unknown option"):
            Mandacaru(method=method,
                      basis={"name": "oncv", "projector_basis": "raw"})

    def test_dry_run(self):
        estimate = estimate_qubits(h2(), basis="oncv")
        assert estimate.n_qubits == 4 and estimate.num_particles == (1, 1)
        assert any("ONCVPSP family" in note for note in estimate.notes)
        assert estimate_qubits(lih(), basis={"name": "oncv"}).n_qubits == 4
        dzp = estimate_qubits(h2(), basis={"name": "oncv", "size": "DZP"})
        assert dzp.n_qubits == 20                       # (2 s + 3 p) x 2 atoms
        atoms = h2()
        atoms.calc = Mandacaru(method="adapt-vqe", basis="oncv",
                               h=H2_H, dry_run=True)
        assert np.isnan(atoms.get_potential_energy())
        assert atoms.calc.dry_run_result.n_qubits == 4
        assert Mandacaru(method="vqe", basis="oncvpsp", h=H2_H).dry_run(lih()).n_qubits == 4


# --------------------------------------------------------------------------- #
# (b) Hardness on the coarse grid and (c) molecular energies vs TM.
# --------------------------------------------------------------------------- #

class TestMolecular:
    @pytest.mark.parametrize("name", sorted(SYSTEMS))
    def test_hardness_on_the_coarse_grid(self, name):
        oncv_ints = _build(name, "oncv")[4]["integrals"]
        tm_ints = _build(name, "tm")[4]["integrals"]
        basis = oncv_ints.resolution_ratios
        projectors = oncv_ints.kb_resolution_ratios
        print(f"\n{name} (h={SYSTEMS[name][1]} A): ONCV basis T_grid/T_exact = "
              f"{np.round(basis, 3)}, projector norm ratios = "
              f"{np.round(projectors, 3)}; TM basis = "
              f"{np.round(tm_ints.resolution_ratios, 3)}, TM projectors: none")
        assert np.all((basis > 0.75) & (basis < 1.25))
        assert projectors.shape == (4,)
        assert np.all((projectors > 0.75) & (projectors < 1.25))
        assert tm_ints.kb_resolution_ratios is None

    @pytest.mark.parametrize("name", sorted(SYSTEMS))
    def test_energies_against_troullier_martins(self, name):
        H, particles, n_orb, _profile, context = _build(name, "oncv")
        integrals = context["integrals"]
        assert context["family"] == "oncvpsp"
        assert n_orb == 2 and particles == (1, 1)
        # Two radial projectors per (atom, l=0, m=0): four in all.
        projectors = integrals.kb_projectors
        assert len(projectors) == 4
        assert sorted(p.index for p in projectors) == [0, 0, 1, 1]
        blocks = projector_blocks(projectors)
        assert all(len(v) == 2 for v in blocks.values())
        D = integrals.nonlocal_coupling_matrix()
        assert D.shape == (4, 4)
        for positions in blocks.values():
            block = D[np.ix_(positions, positions)]
            assert abs(block[0, 1]) > 1e-3 and np.allclose(block, block.T)
        h_nl = integrals.kb_nonlocal()
        assert np.abs(h_nl).max() > 1e-3
        assert np.abs(h_nl - h_nl.conj().T).max() < 1e-12

        rhf = integrals.hartree_fock(context["n_electrons"])
        e_rhf = rhf.electronic_energy + integrals.nuclear_repulsion
        e_fci = _fci(H)
        print(f"\n{name}: ONCV RHF {e_rhf:.6f}  FCI {e_fci:.6f}  | TM RHF "
              f"{TM[name]['rhf']:.6f}  ADAPT {TM[name]['adapt']:.6f}  | "
              f"diff RHF {e_rhf - TM[name]['rhf']:+.4f}  FCI-vs-ADAPT "
              f"{e_fci - TM[name]['adapt']:+.4f} Ha")
        assert np.isfinite(e_rhf) and np.isfinite(e_fci)
        assert e_fci <= e_rhf + 1e-9
        assert abs(e_rhf - TM[name]["rhf"]) < TM_TOL
        assert abs(e_fci - TM[name]["adapt"]) < TM_TOL
        assert e_rhf == pytest.approx(ONCV[name]["rhf"], abs=PIN_TOL)

    @pytest.mark.parametrize("name", sorted(SYSTEMS))
    def test_adapt_vqe(self, name):
        factory, h = SYSTEMS[name]
        atoms = factory()
        atoms.calc = Mandacaru(method="adapt-vqe", basis="oncv", h=h,
                               pool="qeb", max_iterations=4,
                               profile=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            atoms.get_potential_energy()
        result = atoms.calc.result
        energy = result.in_units("Ha")          # the pins are Hartree; results eV
        print(f"\n{name}: ONCV ADAPT-VQE {energy:.6f} Ha "
              f"(TM {TM[name]['adapt']:.6f}, diff "
              f"{energy - TM[name]['adapt']:+.4f})")
        assert atoms.calc.n_qubits == 4
        assert result.energy_unit == "eV"
        assert np.isfinite(result.optimal_energy)
        assert energy == pytest.approx(ONCV[name]["adapt"], abs=PIN_TOL)
        assert abs(energy - TM[name]["adapt"]) < TM_TOL
        assert energy <= ONCV[name]["rhf"] + PIN_TOL

    def test_size_hierarchy_is_variational(self):
        sz = _build("H2", "oncv")[4]["integrals"]
        dzp = _build("H2", {"name": "oncv", "size": "DZP"})
        ints = dzp[4]["integrals"]
        assert dzp[2] == 10 and len(ints.kb_projectors) == 4
        e_sz = sz.hartree_fock(2).electronic_energy + sz.nuclear_repulsion
        e_dzp = ints.hartree_fock(2).electronic_energy + ints.nuclear_repulsion
        print(f"\nH2 RHF: SZ {e_sz:.6f}  DZP {e_dzp:.6f} Ha")
        assert e_dzp < e_sz

    def test_first_zeta_is_the_bound_pseudo_wave(self):
        _H, _p, _n, _pr, context = _build("H2", "oncv")
        fn = context["integrals"].basis[0]
        pp = get_oncv("H")
        r = np.linspace(0.05, 4.0, 50)
        assert np.allclose(fn.radial(r), np.interp(r, pp.r,
                                                   pp.channels[0].pseudo_radial),
                           atol=2e-4)


# --------------------------------------------------------------------------- #
# Construction internals.
# --------------------------------------------------------------------------- #

class TestConstruction:
    def test_bessel_wavevectors_interleave(self):
        qs = oncv.bessel_wavevectors(0, 1.3, 8)
        x = qs * 1.3
        assert np.all(np.diff(x) > 0)
        assert x[0] == pytest.approx(np.pi, abs=1e-10)         # j_0 zero
        assert x[1] == pytest.approx(4.493409458, abs=1e-8)    # j_0' zero
        j, dj, _d2, _d3 = oncv.bessel_derivatives(0, x)
        assert np.abs(j[::2]).max() < 1e-12 and np.abs(dj[1::2]).max() < 1e-12

    def test_bessel_derivatives_against_finite_differences(self):
        x = np.linspace(0.5, 12.0, 40)
        for l in (0, 1, 2):
            j, dj, d2j, d3j = oncv.bessel_derivatives(l, x)
            h = 1e-3
            fd1 = (oncv.spherical_jn(l, x + h) - oncv.spherical_jn(l, x - h)) / (2 * h)
            fd2 = (oncv.spherical_jn(l, x + h) - 2 * j + oncv.spherical_jn(l, x - h)) / h ** 2
            assert np.abs(fd1 - dj).max() < 1e-6
            assert np.abs(fd2 - d2j).max() < 1e-5
            fd3 = (oncv.bessel_derivatives(l, x + h)[2]
                   - oncv.bessel_derivatives(l, x - h)[2]) / (2 * h)
            assert np.abs(fd3 - d3j).max() < 1e-5

    def test_constrained_minimum_satisfies_its_constraints(self):
        rng = np.random.default_rng(1)
        n = 8
        M = rng.normal(size=(n, n))
        K = M @ M.T
        G = np.eye(n) + 0.1 * (M + M.T)
        G = G @ G.T
        A = rng.normal(size=(3, n))
        b = rng.normal(size=3)
        k = rng.normal(size=n)
        c = oncv.constrained_minimum(K, k, A, b, G, 5.0)
        assert np.abs(A @ c - b).max() < 1e-10
        assert c @ G @ c == pytest.approx(5.0, abs=1e-9)
        # KKT stationarity in the null space of A: Z^T (K c + k) = lam Z^T G c.
        _u, _s, vt = np.linalg.svd(A)
        Z = vt[3:].T
        g_obj, g_con = Z.T @ (K @ c + k), Z.T @ (G @ c)
        lam = (g_obj @ g_con) / (g_con @ g_con)
        assert np.abs(g_obj - lam * g_con).max() < 1e-8 * max(1.0, abs(lam))
        # Global: no feasible point (found by moving in the null space and
        # solving the norm condition for the step length) is lower.
        f = lambda v: v @ K @ v + 2 * k @ v
        for _ in range(50):
            y = Z @ rng.normal(size=n - 3)
            # (c + t y)^T G (c + t y) = 5  ->  quadratic in t
            qa, qb = y @ G @ y, 2 * c @ G @ y
            t = -qb / qa                        # the other intersection
            trial = c + t * y
            assert trial @ G @ trial == pytest.approx(5.0, abs=1e-8)
            assert f(trial) >= f(c) - 1e-9

    def test_numerov_bound_state_matches_the_atom(self):
        pp = generated("Li")
        atom = pp.atom
        u, energy = oncv.bound_state(atom.r, atom.v_effective, 0,
                                     atom.eigenvalues[(2, 0)], 3.0)
        assert abs(energy - atom.eigenvalues[(2, 0)]) < 5e-4      # FD vs Numerov
        assert np.trapezoid(u * u, atom.r) == pytest.approx(1.0, abs=1e-9)
        assert np.sum(np.diff(np.sign(u[(atom.r > 0.02) & (atom.r < 10)])) != 0) == 1

    def test_generation_grid_scales_with_z(self):
        assert oncv.generation_points(1) == 6000
        assert oncv.generation_points(8) == 12000
        assert generated("O").r.size == 12000 and generated("H").r.size == 6000

    def test_asymmetric_b_is_refused(self):
        pp = generated("H")
        pw = copy.deepcopy(oncv.optimize_pseudo_waves(
            pp.r, pp.atom.v_effective, 0,
            [pp.channels[0].pseudo_waves[0], pp.channels[0].pseudo_waves[1]],
            pp.channels[0].reference_energies, pp.channels[0].r_cut))
        # A uniform rescaling of c_2 would keep B symmetric (B_12 and B_21
        # both scale with it); distorting one coefficient breaks the matching
        # at r_c and with it the Wronskian identity behind the symmetry.
        pw.coefficients[1] = pw.coefficients[1].copy()
        pw.coefficients[1][0] += 0.05
        with pytest.raises(RuntimeError, match="asymmetric"):
            oncv.assemble_channel(pp.r, pw, pp.v_local_screened, n=1)
        channel = oncv.assemble_channel(pp.r, pw, pp.v_local_screened, n=1,
                                        strict=False)
        assert channel.vanderbilt_asymmetry > 1e-6


class TestGhostSearch:
    """:func:`~mandacaru.pseudopotentials.oncv.ghost_free` on a stub generator:
    the decisions, without paying for a generation."""

    class _Channel:
        def __init__(self, r_cut, reference):
            self.r_cut = r_cut
            self.reference_energies = [reference, reference + 1.0]

    class _Dataset:
        def __init__(self, levels, r_cuts=(3.0, 1.0)):
            self.channels = {l: TestGhostSearch._Channel(rc, -0.2)
                             for l, rc in zip((0, 2), r_cuts)}
            self.levels = levels            # {l: (lowest, second)}
            self.atom = "atom"

    CLEAN = {0: (-0.2, 0.3), 2: (-0.2, 0.3)}
    GHOST = {0: (-1.8, -0.2), 2: (-0.2, 0.3)}

    @staticmethod
    def _levels(pp, l):
        return pp.levels[l]

    @pytest.fixture(autouse=True)
    def _scattering(self, monkeypatch):
        """Phase errors by dataset, default clean."""
        from mandacaru.pseudopotentials import oncv
        self.phases = {}
        monkeypatch.setattr(oncv, "scattering_errors",
                            lambda pp, _ld: self.phases.get(id(pp), {}))

    def _options(self, **overrides):
        options = {"r_cut": None, "r_cut_local": None, "local_shift": None,
                   "local_factor": 0.9, "atom": None}
        options.update(overrides)
        return options

    def _generator(self, outcomes):
        """Returns datasets from ``outcomes`` in order, recording each call."""
        calls = []

        def generate(symbol, *, ghosts, **options):
            assert ghosts == "keep"
            calls.append(options)
            return outcomes[len(calls) - 1]
        return generate, calls

    def _run(self, generate, mode="repair", overrides=None, **options):
        from mandacaru.pseudopotentials.oncv import ghost_free
        return ghost_free(generate, self._levels, None, "X",
                          self._options(**options), mode, overrides)

    def test_a_clean_dataset_is_returned_unchanged(self):
        clean = self._Dataset(self.CLEAN)
        generate, calls = self._generator([clean])
        assert self._run(generate) is clean
        assert len(calls) == 1

    def test_an_inaccurate_level_is_not_a_ghost(self):
        """One level 6e-4 Ha low with nothing displaced: accuracy, not a ghost."""
        from mandacaru.pseudopotentials.oncv import ghost_errors
        pp = self._Dataset({0: (-0.2006, 0.3), 2: (-0.2, 0.3)})
        assert ghost_errors(pp, self._levels) == {}
        assert ghost_errors(self._Dataset(self.GHOST), self._levels) == {
            0: pytest.approx(-1.6)}

    def test_overrides_are_tried_alone_first_then_kept_in_every_attempt(self):
        still = self._Dataset(self.GHOST)
        clean = self._Dataset(self.CLEAN)
        generate, calls = self._generator([self._Dataset(self.GHOST), still,
                                           clean])
        assert self._run(generate, overrides={"norm_deficit": 0.0}) is clean
        alone, balanced = calls[1], calls[2]
        assert alone["norm_deficit"] == 0.0
        assert alone["r_cut"] is None           # the cutoffs are untouched
        assert alone["atom"] == "atom"          # the SCF atom is reused
        assert balanced["norm_deficit"] == 0.0  # and kept with balanced ones
        assert balanced["r_cut"] == {0: 3.0, 2: 3.0}

    def test_a_ghost_is_repaired_with_balanced_cutoffs_and_a_raised_shift(self):
        from mandacaru.pseudopotentials.oncv import GHOST_REMEDY_SHIFTS
        still = self._Dataset({0: (-0.5, -0.2), 2: (-0.2, 0.3)})
        clean = self._Dataset(self.CLEAN)
        generate, calls = self._generator([self._Dataset(self.GHOST), still,
                                           clean])
        assert self._run(generate) is clean
        first, second = calls[1], calls[2]
        assert first["r_cut"] == {0: 3.0, 2: 3.0}
        assert first["r_cut_local"] == pytest.approx(2.7)
        assert first["local_shift"] == GHOST_REMEDY_SHIFTS[0]
        assert second["local_shift"] == GHOST_REMEDY_SHIFTS[1]

    def test_a_repair_that_breaks_the_scattering_is_rejected(self):
        """Iron at a 10 Ha raise: ghost-free and 0.74 rad wrong at +0.25 Ha."""
        wrong = self._Dataset(self.CLEAN)
        right = self._Dataset(self.CLEAN)
        self.phases[id(wrong)] = {0: (0.74, 0.74), 2: (0.005, 0.005)}
        self.phases[id(right)] = {0: (0.004, 0.025), 2: (0.007, 0.007)}
        generate, _calls = self._generator([self._Dataset(self.GHOST), wrong,
                                            right])
        assert self._run(generate) is right

    def test_a_resonance_just_outside_the_window_is_rejected(self):
        """Gallium at a 20 Ha raise: 0.027 rad near, 0.94 rad at +0.55 Ha."""
        resonant = self._Dataset(self.CLEAN)
        right = self._Dataset(self.CLEAN)
        self.phases[id(resonant)] = {0: (0.027, 0.936)}
        self.phases[id(right)] = {0: (0.001, 0.010), 2: (0.003, 0.062)}
        generate, _calls = self._generator([self._Dataset(self.GHOST),
                                            resonant, right])
        assert self._run(generate) is right

    def test_a_ghost_with_a_pinned_cutoff_is_refused_not_overridden(self):
        from mandacaru.pseudopotentials.oncv import GhostStateError
        generate, calls = self._generator([self._Dataset(self.GHOST)])
        with pytest.raises(GhostStateError, match="r_cut fixed by the caller"):
            self._run(generate, r_cut=2.0)
        assert len(calls) == 1

    def test_refuse_mode_refuses(self):
        from mandacaru.pseudopotentials.oncv import GhostStateError
        generate, _calls = self._generator([self._Dataset(self.GHOST)])
        with pytest.raises(GhostStateError, match="l=0"):
            self._run(generate, mode="refuse")

    def test_keep_mode_returns_the_ghost(self):
        ghosted = self._Dataset(self.GHOST)
        generate, _calls = self._generator([ghosted])
        assert self._run(generate, mode="keep") is ghosted

    def test_no_remedy_is_an_error_naming_every_attempt(self):
        from mandacaru.pseudopotentials.oncv import (GHOST_REMEDY_SHIFTS,
                                                     GhostStateError)
        generate, calls = self._generator(
            [self._Dataset(self.GHOST)] * (1 + len(GHOST_REMEDY_SHIFTS)))
        with pytest.raises(GhostStateError, match="no remedy removed it"):
            self._run(generate)
        assert len(calls) == 1 + len(GHOST_REMEDY_SHIFTS)

    def test_a_scattering_channel_is_not_judged(self):
        from mandacaru.pseudopotentials.oncv import ghost_errors
        pp = self._Dataset({0: (-0.2, 0.3), 2: (-5.0, 0.3)})
        pp.channels[2].reference_energies = [0.3, 1.3]
        assert ghost_errors(pp, self._levels) == {}

    def test_an_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="ghosts must be one of"):
            self._run(None, mode="ignore")
