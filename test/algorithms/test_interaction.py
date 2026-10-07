# -*- coding: utf-8 -*-
# file: test/test_interaction.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Per-element basis mappings and interaction energies on a shared grid."""

import numpy as np
from mandacaru.units import HARTREE_TO_EV
import pytest
from ase import Atoms
from ase.build import molecule

from mandacaru.algorithms import Mandacaru, InteractionEnergy, interaction_energy
from mandacaru.algorithms._hamiltonian_from_atoms import (
    PER_ELEMENT, build_basis_hamiltonian, is_per_element_basis,
    per_element_basis, resolve_basis)
from mandacaru.algorithms.dry_run import estimate_qubits
from mandacaru.basis import BasisSet, PerElementBasisSet
from mandacaru.integrals import Grid
from mandacaru.optimizers import Optimizer

# The classical optimizers used below, with the iteration budget and
# the convergence tolerance written out rather than left to the
# library default: a test that pins an energy should say what it was
# optimized with.
LBFGS = Optimizer(method="L-BFGS", maxiter=2000, tol=1e-12)


# --------------------------------------------------------------------------- #
# Per-element basis.
# --------------------------------------------------------------------------- #

class TestPerElementBasis:
    def test_detection(self):
        assert is_per_element_basis({"O": "HAO", "H": "6-31G"})
        assert is_per_element_basis({"*": "HAO"})
        assert not is_per_element_basis({"name": "HAO"})
        assert not is_per_element_basis("HAO")
        assert not is_per_element_basis({})
        assert not is_per_element_basis({"O": "HAO", "size": "DZP"})

    def test_resolve_returns_the_mapping(self):
        name, options = resolve_basis({"O": "HAO", "h": "6-31G"})
        assert name == PER_ELEMENT and set(options) == {"O", "h"}
        table = per_element_basis(options, ["O", "H", "H"])
        assert table["O"] == ("HAO", {}) and table["H"] == ("6-31G", {})

    def test_default_entry_and_missing_element(self):
        table = per_element_basis({"O": "6-31G*", "*": "HAO"}, ["O", "H"])
        assert table["H"] == ("HAO", {})
        with pytest.raises(ValueError, match="no basis given for element"):
            per_element_basis({"O": "HAO"}, ["O", "H"])

    def test_plane_waves_and_nesting_are_refused(self):
        with pytest.raises(ValueError, match="plane-wave"):
            per_element_basis({"O": {"name": "PW"}, "*": "HAO"}, ["O"])
        with pytest.raises(ValueError, match="nested"):
            per_element_basis({"O": {"H": "HAO"}, "*": "HAO"}, ["O"])

    def test_factory_builds_each_element_with_its_family(self):
        bset = BasisSet.build({"O": {"name": "NAO", "size": "DZP"},
                               "H": "6-31G", "*": "HAO"})
        assert isinstance(bset, PerElementBasisSet)
        nao = BasisSet.build("NAO", size="DZP")
        assert len(bset.atom("O")) == len(nao.atom("O"))
        assert len(bset.atom("H")) == 2                    # 6-31G hydrogen
        assert len(bset.atom("Na")) == 6                   # HAO default
        assert "per-element" in bset.name and "NAO" in bset.name
        with pytest.raises(TypeError):
            BasisSet.build({"O": "HAO"}, size="DZP")

    def test_hamiltonian_and_dry_run_agree(self):
        water = molecule("H2O"); water.center(vacuum=2.5)
        basis = {"O": "HAO", "H": "6-31G"}
        est = estimate_qubits(water, basis=basis)
        assert est.per_atom == [("O", 5), ("H", 2), ("H", 2)]
        assert est.n_qubits == 18
        H, particles, n_orb, _p, ctx = build_basis_hamiltonian(
            water, basis, None, 0.35, 0, None)
        assert n_orb == 9 and H.n_modes() == 18 and particles == (5, 5)
        assert ctx["atom_of_orbital"] == [0] * 5 + [1] * 2 + [2] * 2

    def test_calculator_accepts_the_mapping(self):
        h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[6.0] * 3)
        h2.calc = Mandacaru(basis={"H": "STO-4G"}, dry_run=True)
        assert np.isnan(h2.get_potential_energy())
        assert h2.calc.dry_run_result.n_qubits == 4


# --------------------------------------------------------------------------- #
# Interaction energies.
# --------------------------------------------------------------------------- #

def _two_h2(separation):
    """Two H2 molecules side by side, ``separation`` Angstrom apart."""
    a = Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]])
    b = Atoms("H2", positions=[[separation, 0, -0.37], [separation, 0, 0.37]])
    atoms = a + b
    atoms.set_cell([separation + 6.0, 6.0, 6.0])
    atoms.center()
    return atoms


class TestInteractionEnergy:
    def test_fragment_validation(self):
        atoms = _two_h2(4.0)
        with pytest.raises(ValueError, match="partition"):
            interaction_energy(atoms, [[0, 1], [2]], method="rhf", h=0.4)
        with pytest.raises(ValueError, match="at least two"):
            interaction_energy(atoms, [[0, 1, 2, 3]], method="rhf", h=0.4)
        with pytest.raises(ValueError, match="sum to"):
            interaction_energy(atoms, [[0, 1], [2, 3]], charges=[1, 0],
                               method="rhf", h=0.4)
        with pytest.raises(ValueError, match="one charge per fragment"):
            interaction_energy(atoms, [[0, 1], [2, 3]], charges=[0],
                               method="rhf", h=0.4)

    def test_far_apart_fragments_do_not_interact(self):
        atoms = _two_h2(5.0)
        result = interaction_energy(atoms, [[0, 1], [2, 3]], method="rhf",
                                    h=0.35)
        assert isinstance(result, InteractionEnergy)
        assert result.energy_unit == "eV"
        assert abs(result.energy) < 0.02                  # eV
        assert result.in_units("eV") == pytest.approx(result.energy)
        assert result.in_units("Ha") == pytest.approx(result.energy / HARTREE_TO_EV)
        assert result.complex_energy == pytest.approx(
            result.energy + sum(result.fragment_energies))
        assert result.method == "rhf" and result.charges == [0, 0]

    def test_shared_grid_is_the_point(self):
        """The same fragments in their own re-centered boxes disagree."""
        atoms = _two_h2(5.0)
        shared = interaction_energy(atoms, [[0, 1], [2, 3]], method="rhf",
                                    h=0.35)
        assert isinstance(shared.grid, Grid)
        # Re-centered fragment: a different sampling of the same molecule.
        from mandacaru.algorithms._hamiltonian_from_atoms import \
            build_basis_hamiltonian
        frag = atoms[[0, 1]]
        frag.set_cell(atoms.get_cell()); frag.center()
        _H, _p, _n, _prof, ctx = build_basis_hamiltonian(frag, "HAO", None,
                                                          0.35, 0, None)
        own = (ctx["integrals"].hartree_fock(2).electronic_energy
               + ctx["integrals"].nuclear_repulsion) * HARTREE_TO_EV
        assert own != pytest.approx(shared.fragment_energies[0], abs=1e-9)
        # Same ballpark (H2 ~ -30 eV) -- it is the grid sampling that differs.
        assert abs(own - shared.fragment_energies[0]) < 1.0

    def test_charged_fragment_with_adapt_vqe(self):
        # H2 + H+ : two electrons in three orbitals, 6 qubits.
        atoms = Atoms("H3", positions=[[0, 0, -0.37], [0, 0, 0.37], [3.0, 0, 0]],
                      cell=[8.0, 6.0, 6.0])
        atoms.center()
        result = interaction_energy(atoms, [[0, 1], [2]], charges=[0, 1],
                                    charge=1, method="adapt-vqe", h=0.4,
                                    profile=False, optimizer=LBFGS,
                                    convergence={"gradient": 1e-5})
        assert result.charges == [0, 1]
        # A bare proton has no electrons: its "energy" is exactly zero.
        assert result.fragment_energies[1] == pytest.approx(0.0, abs=1e-12)
        assert result.complex_result is not None and len(result.results) == 2
        # The proton is attracted to the H2 charge cloud: bound.
        assert result.energy < 0.0

    def test_calculator_method_reuses_its_options(self):
        atoms = _two_h2(5.0)
        calc = Mandacaru(method="vqe", basis="HAO", h=0.4,
                         optimizer=LBFGS)
        result = calc.interaction_energy(atoms, [[0, 1], [2, 3]])
        assert result.method == "vqe" and abs(result.in_units("eV")) < 0.05


# --------------------------------------------------------------------------- #
# Counterpoise: ghost atoms.
# --------------------------------------------------------------------------- #

def _helium_pair(separation=3.0, vacuum=3.0):
    atoms = Atoms("He2", positions=[[0, 0, 0], [0, 0, separation]])
    atoms.center(vacuum=vacuum)
    return atoms


PAW_SZ = {"name": "PAW-LCAO", "size": "SZ"}


class TestCounterpoise:
    def test_a_ghost_lends_functions_and_nothing_else(self):
        """The ghost's basis functions are there; its nucleus, projectors
        and electrons are not."""
        atoms = _helium_pair()
        grid = Grid(center=list(atoms.get_cell().diagonal() / 2),
                    box_size=list(atoms.get_cell().diagonal()), h=0.3)
        full = build_basis_hamiltonian(atoms, PAW_SZ, grid, 0.3, 0, None,
                                       hamiltonian=False)[4]
        ghosted = build_basis_hamiltonian(atoms, PAW_SZ, grid, 0.3, 0, None,
                                          hamiltonian=False, ghosts=[1])[4]
        assert ghosted["n_electrons"] == full["n_electrons"] // 2 == 2
        assert ghosted["ghosts"] == (1,)
        ints, ref = ghosted["integrals"], full["integrals"]
        assert ints.n_orbitals == ref.n_orbitals
        assert len(ints._potentials.nuclei) == 1
        assert len(ints.kb_projectors) == len(ref.kb_projectors) // 2
        assert ints.nuclear_repulsion == pytest.approx(0.0, abs=1e-12)
        estimate = estimate_qubits(atoms, basis=PAW_SZ, ghosts=[1])
        assert estimate.n_electrons == 2
        assert any("ghost" in note for note in estimate.notes)

    def test_it_removes_the_superposition_error(self):
        """Each helium inside the pair borrows its partner's functions; in
        the pair's basis the isolated atom borrows them too, so the
        counterpoise energy is above the uncorrected one by that error."""
        atoms = _helium_pair()
        plain = interaction_energy(atoms, [[0], [1]], method="rhf", h=0.3,
                                   basis=PAW_SZ)
        corrected = interaction_energy(atoms, [[0], [1]], method="rhf",
                                       h=0.3, basis=PAW_SZ, counterpoise=True)
        assert corrected.counterpoise and not plain.counterpoise
        # The complex is the same run either way; only the fragments move.
        assert corrected.complex_energy == pytest.approx(plain.complex_energy,
                                                         abs=1e-9)
        assert all(cp < bare for cp, bare in zip(corrected.fragment_energies,
                                                 plain.fragment_energies))
        assert corrected.energy > plain.energy

    def test_ghosts_are_validated(self):
        atoms = _helium_pair()
        for bad in ([5], [0, 1], ["x"]):
            with pytest.raises(ValueError, match="ghost"):
                estimate_qubits(atoms, basis=PAW_SZ, ghosts=bad)

    def test_forces_are_refused_with_ghosts(self):
        atoms = _helium_pair()
        atoms.calc = Mandacaru(method="dft", h=0.3, basis=PAW_SZ,
                               ghosts=[1], trace=False)
        assert np.isfinite(atoms.get_potential_energy())
        with pytest.raises(NotImplementedError, match="ghost"):
            atoms.get_forces()

    def test_a_crystal_takes_ghosts_too(self):
        """A periodic pair (Gamma only): the counterpoise fragments carry
        their partner's functions and none of its electrons, and their
        energies drop by the superposition they borrow; the stress, like
        the forces, is refused."""
        atoms = _helium_pair(separation=2.6, vacuum=2.5)
        atoms.pbc = True
        options = dict(method="dft", h=0.3, basis=PAW_SZ, trace=False,
                       kpts={"size": (1, 1, 1), "gamma": True},
                       smearing={"method": "fermi-dirac", "width": 0.01})
        plain = interaction_energy(atoms, [[0], [1]], **options)
        corrected = interaction_energy(atoms, [[0], [1]], counterpoise=True,
                                       **options)
        assert corrected.complex_energy == pytest.approx(plain.complex_energy,
                                                         abs=1e-8)
        assert all(cp < bare for cp, bare in zip(corrected.fragment_energies,
                                                 plain.fragment_energies))
        ghosted = atoms.copy()
        ghosted.calc = Mandacaru(ghosts=[1], **options)
        ghosted.get_potential_energy()
        assert ghosted.calc.solver._gradient_context["n_electrons"] == 2.0
        with pytest.raises(NotImplementedError, match="ghost"):
            ghosted.calc.get_stress()

    def test_an_all_electron_ghost_has_no_nucleus(self):
        atoms = _helium_pair()
        grid = Grid(center=list(atoms.get_cell().diagonal() / 2),
                    box_size=list(atoms.get_cell().diagonal()), h=0.3)
        full = build_basis_hamiltonian(atoms, "HAO", grid, 0.3, 0, None,
                                       hamiltonian=False)[4]
        ghosted = build_basis_hamiltonian(atoms, "HAO", grid, 0.3, 0, None,
                                          hamiltonian=False, ghosts=[1])[4]
        assert ghosted["n_electrons"] == 2
        assert ghosted["integrals"].n_orbitals == full["integrals"].n_orbitals
        assert len(ghosted["integrals"]._potentials.nuclei) == 1

    def test_plane_waves_and_bloch_methods_refuse_ghosts(self):
        atoms = _helium_pair()
        with pytest.raises(NotImplementedError, match="plane-wave"):
            build_basis_hamiltonian(atoms, {"name": "PW", "energy_cutoff": 60},
                                    None, 0.3, 0, None, ghosts=[1])
        atoms.pbc = True
        with pytest.raises(NotImplementedError, match="Bloch"):
            atoms.calc = Mandacaru(method="bloch-vqe", h=0.3, basis=PAW_SZ,
                                   kpts={"size": (1, 1, 1), "gamma": True},
                                   ghosts=[1])
            atoms.get_potential_energy()

    def test_d4_counts_the_real_atoms_only(self):
        pytest.importorskip("dftd4")
        from mandacaru.algorithms.dft import d4_dispersion_energy
        atoms = _helium_pair()
        atoms.calc = Mandacaru(method="dft", xc="pbe", dispersion="d4",
                               h=0.3, basis=PAW_SZ, ghosts=[1], trace=False)
        atoms.get_potential_energy()
        assert atoms.calc.solver._scf.dispersion_energy == pytest.approx(
            d4_dispersion_energy(atoms[[0]], "pbe"), abs=1e-12)

    def test_per_atom_properties_are_refused_and_the_log_names_the_ghosts(
            self, tmp_path):
        atoms = _helium_pair()
        log = tmp_path / "run.log"
        atoms.calc = Mandacaru(method="dft", h=0.3, basis=PAW_SZ, ghosts=[1],
                               txt=str(log))
        atoms.get_potential_energy()
        with pytest.raises(NotImplementedError, match="ghost"):
            atoms.calc.get_charges()
        with pytest.raises(NotImplementedError, match="ghost"):
            atoms.calc.write_cube(tmp_path / "density.cube")
        ghost_lines = [line for line in log.read_text().splitlines()
                       if "ghost" in line]
        assert len(ghost_lines) == 1 and "atoms 1" in ghost_lines[0]

    def test_interaction_energy_sets_the_ghosts_itself(self):
        with pytest.raises(ValueError, match="counterpoise=True"):
            interaction_energy(_helium_pair(), [[0], [1]], method="rhf",
                               basis=PAW_SZ, ghosts=[1])
