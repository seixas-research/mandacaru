# -*- coding: utf-8 -*-
# file: test/test_ncpp_family.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The pseudopotential **family** interface.

The norm-conserving Troullier-Martins potentials in Kleinman-Bylander form
are the family ``"ncpp"`` (aliases ``"tm"`` / ``"ncpp-tm"``), registered in
``PSEUDO_FAMILIES`` and **selected as a basis name** (``basis="NCPP"``,
``basis={"name": "NCPP", "size": "DZP"}``); the driver dispatches on the
registry so ONCVPSP / PAW plug in without touching it.  The nonlocal term became the general separable form
``H_NL = C D C^dagger`` with a block-diagonal coupling ``D``, plus an optional
overlap correction ``S + C Q C^dagger``.

The pinned energies below were computed with the code *before* the
generalization; the TM path must reproduce them.  H and Li carry a single
valence channel (local), so the shipped potentials have **no** projectors on
H2 / LiH -- the general form is therefore also exercised with synthetic
projectors on H2, against the old rank-one formula written out explicitly.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from ase import Atoms

from carcara.algorithms import ADAPTVQE, Carcara, VQE
from carcara.algorithms._hamiltonian_from_atoms import (
    build_basis_hamiltonian, coherent_positions, grid_from_cell,
    pseudopotential_family, resolve_basis, resolve_pseudo_basis)
from carcara.algorithms.dry_run import estimate_qubits
from carcara.core import MolecularIntegrals
from carcara.core.hamiltonian import assemble_block_matrix, projector_blocks
from carcara.pseudopotentials import (
    DEFAULT_FAMILY, FORMAT_VERSION, LEGACY_FAMILY, PSEUDO_FAMILIES,
    FamilySpec, KBProjector, family_names, get_pseudopotential,
    canonical_family_name, kb_coupling_blocks, kb_projectors,
    load_pseudopotential, lookup_family, pseudo_basis, register_family,
    resolve_family, save_pseudopotential, unregister_family)
from carcara.pseudopotentials.io import (FILE_EXTENSIONS,
                                                       PSEUDO_FORMATS,
                                                       library_file)

# --------------------------------------------------------------------------- #
# Test systems (small on purpose: h >= 0.25 A, cell <= 8 A).
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


#: Energies (Hartree) measured with the pre-generalization code, 2026-09-14.
# Re-pinned on 2026-09-16, when the FFT Coulomb cell self-energy became the
# closed-form cell average (`poisson.cell_self_potential`) instead of the
# rounded cube constant 2.3800756.  Every value moved by ~1e-8 Ha, five orders
# of magnitude below this grid's own discretization error; the previous values
# are kept alongside so the shift stays visible.
PINNED = {
    "H2": {"h": H2_H, "rhf": -1.061096237049, "fci": -1.075333378284,
           "adapt": -1.075333378229,
           "before_exact_self_energy": {"rhf": -1.061096245397,
                                        "fci": -1.075333384728}},
    "LiH": {"h": LIH_H, "rhf": -0.729555400740, "fci": -0.740838978397,
            "adapt": -0.740838977840,
            "before_exact_self_energy": {"rhf": -0.729555411706,
                                         "fci": -0.740838986301}},
}
SYSTEMS = {"H2": h2, "LiH": lih}


def _fci(hamiltonian) -> float:
    m = hamiltonian.map_to_qubits("jordan_wigner").to_matrix()
    return float(np.linalg.eigvalsh(0.5 * (m + m.conj().T)).min())


# --------------------------------------------------------------------------- #
# Family names and the registry.
# --------------------------------------------------------------------------- #

class TestFamilyResolution:
    def test_default_is_troullier_martins(self):
        assert DEFAULT_FAMILY == "ncpp"
        assert resolve_family(None).name == "ncpp"

    @pytest.mark.parametrize("name", ["tm", "TM", "ncpp", "NCPP", "ncpp-tm",
                                      "NCPP-TM", "ncpp_tm", " Ncpp-Tm "])
    def test_aliases_are_case_insensitive(self, name):
        assert resolve_family(name) is PSEUDO_FAMILIES["ncpp"]
        assert lookup_family(name) is PSEUDO_FAMILIES["ncpp"]
        assert pseudopotential_family(name) is PSEUDO_FAMILIES["ncpp"]
        assert canonical_family_name(name) == "ncpp"

    def test_unknown_family_lists_the_registered_ones(self):
        with pytest.raises(ValueError, match="unknown pseudopotential family "
                                             "'gth'.*'ncpp'.*'tm'"):
            resolve_family("gth")
        assert lookup_family("gth") is None and lookup_family(3) is None
        assert canonical_family_name("gth") == "gth"

    def test_non_string_is_a_type_error(self):
        with pytest.raises(TypeError):
            resolve_family(3)

    def test_registry_entry_is_complete(self):
        spec = PSEUDO_FAMILIES["ncpp"]
        assert isinstance(spec, FamilySpec)
        assert spec.norm_conserving is True
        assert set(spec.aliases) == {"tm", "ncpp-tm"}
        assert spec.label == "NCPP"
        assert spec.options == ("size", "split_norm", "directory")
        assert callable(spec.generate) and callable(spec.get) \
            and callable(spec.build)
        assert spec.get("H").symbol == "H"
        assert family_names()[0] == "ncpp" and "ncpp-tm" in family_names()

    @pytest.mark.parametrize("basis, expected", [
        ("NCPP", {}),
        ("tm", {}),
        ({"name": "NCPP-TM", "size": "DZ"}, {"size": "DZ"}),
        ({"name": "ncpp", "directory": "/x"}, {"directory": "/x"}),
    ])
    def test_basis_name_selects_the_family(self, basis, expected):
        name, options = resolve_basis(basis)
        family, options = resolve_pseudo_basis(name, options, ["H"])
        assert family is PSEUDO_FAMILIES["ncpp"] and options == expected

    @pytest.mark.parametrize("basis", ["FAO", "6-31G(d)", "cc-pVDZ",
                                       {"name": "NAO", "size": "DZP"},
                                       {"name": "NAO-AE", "tier": 1}])
    def test_all_electron_names_are_not_families(self, basis):
        name, options = resolve_basis(basis)
        assert pseudopotential_family(name) is None
        assert resolve_pseudo_basis(name, options, ["H"])[0] is None

    @pytest.mark.parametrize("basis", ["PP", "pseudo", {"name": "PP",
                                                        "size": "DZ"}])
    def test_retired_names_are_refused_not_aliased(self, basis):
        with pytest.raises(ValueError, match="no longer a basis name.*'NCPP'"):
            resolve_basis(basis)

    def test_per_element_sizes(self):
        family, options = resolve_pseudo_basis(
            "per-element", {"O": {"name": "NCPP", "size": "DZP"}, "*": "tm"},
            ["O", "H", "H"])
        assert family is PSEUDO_FAMILIES["ncpp"]
        assert options == {"size": {"O": "DZP", "H": "SZ"}}
        with pytest.raises(ValueError, match="cannot mix a pseudopotential"):
            resolve_pseudo_basis("per-element", {"O": "NCPP", "H": "FAO"},
                                 ["O", "H"])
        with pytest.raises(ValueError, match="one pseudopotential family"):
            resolve_pseudo_basis("per-element", {"O": "NCPP", "H": "PAW"},
                                 ["O", "H"])
        with pytest.raises(ValueError, match="only 'size' and 'split_norm'"):
            resolve_pseudo_basis("per-element",
                                 {"O": {"name": "NCPP", "directory": "/x"},
                                  "H": "NCPP"}, ["O", "H"])

    def test_taken_names_cannot_be_reregistered(self):
        dummy = FamilySpec(name="tm2", description="", generate=None,
                           get=None, build=None, aliases=("ncpp",))
        with pytest.raises(ValueError, match="already taken"):
            register_family(dummy)
        assert "tm2" not in PSEUDO_FAMILIES
        with pytest.raises(ValueError, match="cannot be unregistered"):
            unregister_family("tm")

    @pytest.mark.parametrize("driver", [VQE, ADAPTVQE])
    def test_drivers_validate_the_basis_at_construction(self, driver):
        assert driver(basis="ncpp-tm").basis == "ncpp-tm"
        with pytest.raises(ValueError, match="unknown option.*'tier'.*NCPP"):
            driver(basis={"name": "NCPP", "tier": 1})
        with pytest.raises(ValueError, match="frozen_core is redundant"):
            driver(basis="NCPP", frozen_core=True)
        with pytest.raises(ValueError, match="no longer a basis name"):
            driver(basis="PP")


# --------------------------------------------------------------------------- #
# The ``family`` field on disk.
# --------------------------------------------------------------------------- #

class TestFamilyField:
    def test_format_version_was_bumped(self):
        assert FORMAT_VERSION == 2 and LEGACY_FAMILY == "ncpp"

    def test_library_entries_are_the_ncpp_family(self):
        for symbol in ("H", "Li", "O"):
            assert get_pseudopotential(symbol).family == "ncpp"

    def test_shipped_library_is_still_version_1(self):
        """The library was *not* regenerated: legacy files must keep loading."""
        pp = load_pseudopotential(library_file("H"))
        assert pp.family == "ncpp" and pp.symbol == "H"

    @pytest.mark.parametrize("fmt", PSEUDO_FORMATS)
    def test_round_trip_keeps_the_family(self, tmp_path, fmt):
        pp = copy.copy(get_pseudopotential("H"))
        pp.family = "ncpp"
        path = save_pseudopotential(pp, tmp_path / f"H{FILE_EXTENSIONS[fmt]}",
                                    format=fmt)
        assert load_pseudopotential(path).family == "ncpp"
        # The pre-rename spelling on disk canonicalizes on load.
        pp.family = "tm"
        path = save_pseudopotential(pp, tmp_path / f"T{FILE_EXTENSIONS[fmt]}",
                                    format=fmt)
        assert load_pseudopotential(path).family == "ncpp"
        pp.family = "some-future-family"
        path = save_pseudopotential(pp, tmp_path / f"X{FILE_EXTENSIONS[fmt]}",
                                    format=fmt)
        assert load_pseudopotential(path).family == "some-future-family"

    def test_json_carries_version_and_family(self, tmp_path):
        path = save_pseudopotential(get_pseudopotential("H"),
                                    tmp_path / "H.json")
        with open(path) as handle:
            payload = json.load(handle)
        assert payload["version"] == FORMAT_VERSION
        assert payload["family"] == "ncpp"

    def test_legacy_file_without_the_field_loads_as_ncpp(self, tmp_path):
        path = save_pseudopotential(get_pseudopotential("H"),
                                    tmp_path / "H.json")
        with open(path) as handle:
            payload = json.load(handle)
        del payload["family"]
        payload["version"] = 1
        with open(path, "w") as handle:
            json.dump(payload, handle)
        assert load_pseudopotential(path).family == "ncpp"

    def test_tm_loader_refuses_another_family(self, tmp_path):
        """A plain TM record merely *carrying* another family's name keeps
        the TM file layout (the table families are recognized by type)."""
        pp = copy.copy(get_pseudopotential("H"))
        pp.family = "paw"
        save_pseudopotential(pp, tmp_path / "H.json")
        with pytest.raises(ValueError, match="belongs to family 'paw'"):
            PSEUDO_FAMILIES["ncpp"].get("H", tmp_path)


# --------------------------------------------------------------------------- #
# The TM path reproduces the pre-generalization energies.
# --------------------------------------------------------------------------- #

class TestPinnedEnergies:
    @pytest.mark.parametrize("name", sorted(PINNED))
    @pytest.mark.parametrize("basis", ["NCPP", "ncpp-tm", {"name": "TM"}])
    def test_rhf_and_fci_are_unchanged(self, name, basis):
        atoms = SYSTEMS[name]()
        pinned = PINNED[name]
        H, particles, n_orb, _profile, context = build_basis_hamiltonian(
            atoms, basis, None, pinned["h"], 0, None)
        integrals = context["integrals"]
        assert context["family"] == "ncpp"
        assert n_orb == 2 and particles == (1, 1)
        assert integrals.kb_projectors == []          # s-only valence
        assert np.abs(integrals.kb_nonlocal()).max() == 0.0
        rhf = integrals.hartree_fock(context["n_electrons"])
        e_rhf = rhf.electronic_energy + integrals.nuclear_repulsion
        assert e_rhf == pytest.approx(pinned["rhf"], abs=1e-9)
        assert _fci(H) == pytest.approx(pinned["fci"], abs=1e-9)

    @pytest.mark.parametrize("name", sorted(PINNED))
    def test_adapt_vqe_is_unchanged(self, name):
        atoms = SYSTEMS[name]()
        atoms.calc = Carcara(method="adapt-vqe", basis="ncpp",
                             h=PINNED[name]["h"], pool="qeb",
                             max_iterations=4, verbose=False, profile=False)
        atoms.get_potential_energy()
        assert atoms.calc.n_qubits == 4
        # The pins are Hartree; the result is eV.
        assert atoms.calc.result.in_units("Ha") == pytest.approx(
            PINNED[name]["adapt"], abs=1e-6)


# --------------------------------------------------------------------------- #
# The general separable nonlocal form.
# --------------------------------------------------------------------------- #

def synthetic_h(kb_energy=0.35, n_radial=1):
    """H's potential with a synthetic *p* projector channel bolted on.

    H has a single (local) s channel, so to exercise the nonlocal machinery
    on H2 a p projector is invented: ``chi_i(r) = r^(1+i) exp(-r^2)``.  Its
    physics is irrelevant -- only the algebra is under test.
    """
    pp = copy.copy(get_pseudopotential("H"))
    r = pp.r
    pp.projectors = {1: r * np.exp(-r * r)}
    pp.kb_energies = {1: float(kb_energy)}
    return pp


def h2_setup(n_radial=1, kb_energy=0.35):
    """Basis, projectors and grid for H2 with the synthetic projectors."""
    atoms = h2()
    symbols = atoms.get_chemical_symbols()
    positions = coherent_positions(atoms)
    pp = synthetic_h(kb_energy)
    potentials = {"H": pp}
    basis_fns, _owners = pseudo_basis(symbols, positions, potentials)
    projectors = []
    for atom, position in enumerate(positions):
        for m in (-1, 0, 1):
            for i in range(n_radial):
                radial = pp.r ** (1 + i) * np.exp(-pp.r ** 2)
                projectors.append(KBProjector(
                    pp, 1, m, center=position, atom_index=atom, index=i,
                    radial=radial, kb_energy=kb_energy / (1 + i)))
    grid = grid_from_cell(atoms, H2_H, center=positions.mean(axis=0))
    nuclei = [(pp.valence_charge, p) for p in positions]
    return nuclei, basis_fns, projectors, grid, [pp] * 2


def integrals_for(projectors_kw=None, **kwargs):
    nuclei, basis_fns, projectors, grid, pps = h2_setup(**(projectors_kw or {}))
    return MolecularIntegrals(nuclei, basis_fns, grid, softening=0.0,
                              pseudos=pps, kb_projectors=projectors,
                              **kwargs), projectors


class TestGeneralizedNonlocal:
    def test_projector_labels(self):
        _ints, projectors = integrals_for()
        assert len(projectors) == 6                       # 2 atoms x 3 m
        for p in projectors:
            assert p.channel == (1, p.m) and p.index == 0
            assert p.block_key == (p.atom_index, 1, p.m)
        blocks = kb_coupling_blocks(projectors)
        assert set(blocks) == {(a, 1, m) for a in (0, 1) for m in (-1, 0, 1)}
        assert all(b.shape == (1, 1) for b in blocks.values())

    def test_shipped_projectors_carry_the_labels_too(self):
        pp = get_pseudopotential("O")
        projectors = kb_projectors(["O"], np.zeros((1, 3)), {"O": pp})
        assert [p.block_key for p in projectors] == [(0, 0, 0)]
        assert projectors[0].index == 0
        assert kb_coupling_blocks(projectors)[(0, 0, 0)][0, 0] == \
            pytest.approx(pp.kb_energies[0])

    def test_kb_form_matches_the_old_rank_one_formula(self):
        """``C D C^dagger`` with ``D = diag(E_KB)`` == ``(C * E) @ C^dagger``."""
        ints, projectors = integrals_for()
        C = ints.projections()
        assert C.shape == (2, 6)
        energies = np.array([p.kb_energy for p in projectors])
        old = (C * energies) @ C.conj().T
        new = ints.kb_nonlocal()
        assert np.abs(new).max() > 1e-4                   # actually nonzero
        assert np.abs(new - old).max() < 1e-12
        # The same through the family's explicit blocks.
        ints_blocks, _p = integrals_for(
            nonlocal_coupling=kb_coupling_blocks(projectors))
        assert np.abs(ints_blocks.kb_nonlocal() - old).max() < 1e-12
        assert np.allclose(new, new.conj().T)
        assert ints.nonlocal_matrix() is not None      # alias kept
        assert ints.kb_resolution_ratios.shape == (6,)

    def test_multi_projector_blocks(self):
        """Two radial projectors per channel with a full 2x2 coupling block."""
        ints, projectors = integrals_for({"n_radial": 2})
        assert len(projectors) == 12
        block = np.array([[0.4, 0.1], [0.1, -0.2]])
        blocks = {key: block for key in projector_blocks(projectors)}
        ints.nonlocal_coupling = blocks
        D = ints.nonlocal_coupling_matrix()
        assert D.shape == (12, 12)
        # Explicit sum over channels: sum_{ij} C_i D_ij C_j^dagger.
        C = ints.projections()
        expected = np.zeros((2, 2), dtype=complex)
        for key, positions in projector_blocks(projectors).items():
            assert len(positions) == 2
            assert [projectors[p].index for p in positions] == [0, 1]
            for a, pa in enumerate(positions):
                for b, pb in enumerate(positions):
                    expected += np.outer(C[:, pa], C[:, pb].conj()) * block[a, b]
        assert np.abs(ints.kb_nonlocal() - expected).max() < 1e-12

    def test_block_validation(self):
        ints, projectors = integrals_for()
        groups = projector_blocks(projectors)
        with pytest.raises(ValueError, match="missing"):
            assemble_block_matrix(projectors, {})
        with pytest.raises(ValueError, match="unmatched"):
            bad = {k: [[1.0]] for k in groups}
            bad[(5, 1, 0)] = [[1.0]]
            assemble_block_matrix(projectors, bad)
        with pytest.raises(ValueError, match="must be \\(1, 1\\)"):
            assemble_block_matrix(projectors, {k: np.eye(2) for k in groups})
        with pytest.raises(ValueError, match="duplicate radial indices"):
            projector_blocks(projectors + [projectors[0]])
        with pytest.raises(ValueError, match="diagonal"):
            assemble_block_matrix(projectors, None)
        with pytest.raises(ValueError, match="share the channel"):
            kb_coupling_blocks(projectors + [projectors[0]])

    def test_no_projectors_gives_zeros(self):
        nuclei, basis_fns, _p, grid, pps = h2_setup()
        ints = MolecularIntegrals(nuclei, basis_fns, grid, softening=0.0,
                                  pseudos=pps)
        assert np.abs(ints.kb_nonlocal()).max() == 0.0
        assert ints.projections().shape == (2, 0)
        assert ints.nonlocal_overlap_matrix() is None


# --------------------------------------------------------------------------- #
# The overlap-correction hook.
# --------------------------------------------------------------------------- #

class TestOverlapHook:
    def test_zero_blocks_are_the_plain_hamiltonian(self):
        plain, projectors = integrals_for()
        zero = {k: np.zeros((1, 1)) for k in projector_blocks(projectors)}
        with_q, _p = integrals_for(nonlocal_overlap=zero)
        assert np.abs(with_q.overlap() - plain.overlap()).max() < 1e-12
        assert np.abs(with_q.overlap() - with_q.bare_overlap()).max() == 0.0
        assert np.abs(with_q.one_body() - plain.one_body()).max() < 1e-12
        assert np.abs(with_q.two_body() - plain.two_body()).max() < 1e-12
        assert with_q.nonlocal_overlap_matrix().shape == (6, 6)

    def test_nonzero_q_augments_the_overlap_consistently(self):
        plain, projectors = integrals_for()
        q = {k: np.array([[0.3]]) for k in projector_blocks(projectors)}
        aug, _p = integrals_for(nonlocal_overlap=q)
        C = aug.projections()
        Q = aug.nonlocal_overlap_matrix()
        expected = plain.bare_overlap() + C @ Q @ C.conj().T
        assert np.abs(aug.overlap() - expected).max() < 1e-12
        assert np.abs(aug.bare_overlap() - plain.bare_overlap()).max() == 0.0
        assert np.abs(aug.overlap() - plain.overlap()).max() > 1e-6
        # The orthonormalizer is built from the augmented overlap:
        X = aug._lowdin_x()
        assert np.abs(X.conj().T @ aug.overlap() @ X - np.eye(2)).max() < 1e-12
        assert np.abs(X.conj().T @ plain.overlap() @ X - np.eye(2)).max() > 1e-6
        # and the transformed integrals differ from the plain ones.
        assert np.abs(aug.one_body() - plain.one_body()).max() > 1e-6

    def test_rhf_and_uhf_run_on_the_augmented_integrals(self):
        _plain, projectors = integrals_for()
        q = {k: np.array([[0.3]]) for k in projector_blocks(projectors)}
        aug, _p = integrals_for(nonlocal_overlap=q)
        rhf = aug.hartree_fock(2)
        assert rhf.converged and np.isfinite(rhf.electronic_energy)
        uhf = aug.open_shell_hartree_fock(1, 1)
        assert np.isfinite(uhf.reference_energy)
        H = aug.molecular_hamiltonian(mo_basis=True, n_electrons=2)
        assert np.isfinite(_fci(H))

    def test_overlap_hook_needs_projectors(self):
        nuclei, basis_fns, _p, grid, pps = h2_setup()
        with pytest.raises(ValueError, match="needs projectors"):
            MolecularIntegrals(nuclei, basis_fns, grid, softening=0.0,
                               pseudos=pps,
                               nonlocal_overlap={(0, 1, 0): [[0.1]]})

    def test_tm_family_passes_no_overlap_correction(self):
        _H, _p, _n, _pr, context = build_basis_hamiltonian(
            h2(), "TM", None, H2_H, 0, None)
        assert context["integrals"].nonlocal_overlap is None
        assert context["integrals"].nonlocal_coupling == {}


# --------------------------------------------------------------------------- #
# Dispatch and the dry run.
# --------------------------------------------------------------------------- #

class TestDispatch:
    def test_a_new_family_needs_no_driver_edit(self):
        calls = []
        sentinel = ("H", (1, 1), 2, {}, {"family": "dummy"})

        def build(atoms, grid, h, charge, spin, options, kinetic):
            calls.append((options, kinetic))
            return sentinel

        register_family(FamilySpec(
            name="dummy", description="test double", generate=None,
            get=lambda symbol, directory=None: get_pseudopotential(symbol),
            build=build, norm_conserving=False, aliases=("dmy",)))
        try:
            assert resolve_family("DMY").name == "dummy"
            assert pseudopotential_family("dummy").name == "dummy"
            out = build_basis_hamiltonian(
                h2(), {"name": "dmy", "size": "DZ", "directory": "/nowhere"},
                None, H2_H, 0, None, kinetic="spectral")
            assert out is sentinel
            assert calls == [({"directory": "/nowhere", "size": "DZ"},
                              "spectral")]
            with pytest.raises(ValueError, match="unknown option.*'tier'"):
                build_basis_hamiltonian(h2(), {"name": "dmy", "tier": 1},
                                        None, H2_H, 0, None)
        finally:
            unregister_family("dummy")
        assert "dummy" not in PSEUDO_FAMILIES and "dmy" not in family_names()

    @pytest.mark.parametrize("basis", ["NCPP", "ncpp", {"name": "ncpp-tm"}])
    def test_dry_run_estimates_through_the_family(self, basis):
        estimate = estimate_qubits(h2(), basis=basis)
        assert estimate.n_qubits == 4
        assert estimate.num_particles == (1, 1)
        assert estimate.basis.startswith("NCPP (SZ")
        assert any("NCPP family" in note for note in estimate.notes)
        estimate = estimate_qubits(lih(), basis=basis)
        assert estimate.n_qubits == 4                      # Li 2s + H 1s

    def test_dry_run_rejects_the_retired_name(self):
        with pytest.raises(ValueError, match="no longer a basis name"):
            estimate_qubits(h2(), basis="PP")

    def test_calculator_dry_run(self):
        atoms = h2()
        calc = Carcara(method="vqe", basis="ncpp-tm", h=H2_H, verbose=False)
        assert calc.dry_run(atoms).n_qubits == 4
        atoms.calc = Carcara(method="adapt-vqe", basis={"name": "TM"},
                             h=H2_H, dry_run=True, verbose=False)
        assert np.isnan(atoms.get_potential_energy())
        assert atoms.calc.dry_run_result.n_qubits == 4
