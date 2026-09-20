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


# --------------------------------------------------------------------------- #
# Per-element basis.
# --------------------------------------------------------------------------- #

class TestPerElementBasis:
    def test_detection(self):
        assert is_per_element_basis({"O": "FAO", "H": "6-31G"})
        assert is_per_element_basis({"*": "FAO"})
        assert not is_per_element_basis({"name": "FAO"})
        assert not is_per_element_basis("FAO")
        assert not is_per_element_basis({})
        assert not is_per_element_basis({"O": "FAO", "size": "DZP"})

    def test_resolve_returns_the_mapping(self):
        name, options = resolve_basis({"O": "FAO", "h": "6-31G"})
        assert name == PER_ELEMENT and set(options) == {"O", "h"}
        table = per_element_basis(options, ["O", "H", "H"])
        assert table["O"] == ("FAO", {}) and table["H"] == ("6-31G", {})

    def test_default_entry_and_missing_element(self):
        table = per_element_basis({"O": "6-31G*", "*": "FAO"}, ["O", "H"])
        assert table["H"] == ("FAO", {})
        with pytest.raises(ValueError, match="no basis given for element"):
            per_element_basis({"O": "FAO"}, ["O", "H"])

    def test_plane_waves_and_nesting_are_refused(self):
        with pytest.raises(ValueError, match="plane-wave"):
            per_element_basis({"O": {"name": "PW"}, "*": "FAO"}, ["O"])
        with pytest.raises(ValueError, match="nested"):
            per_element_basis({"O": {"H": "FAO"}, "*": "FAO"}, ["O"])

    def test_factory_builds_each_element_with_its_family(self):
        bset = BasisSet.build({"O": {"name": "NAO", "size": "DZP"},
                               "H": "6-31G", "*": "FAO"})
        assert isinstance(bset, PerElementBasisSet)
        nao = BasisSet.build("NAO", size="DZP")
        assert len(bset.atom("O")) == len(nao.atom("O"))
        assert len(bset.atom("H")) == 2                    # 6-31G hydrogen
        assert len(bset.atom("Na")) == 6                   # FAO default
        assert "per-element" in bset.name and "NAO" in bset.name
        with pytest.raises(TypeError):
            BasisSet.build({"O": "FAO"}, size="DZP")

    def test_hamiltonian_and_dry_run_agree(self):
        water = molecule("H2O"); water.center(vacuum=2.5)
        basis = {"O": "FAO", "H": "6-31G"}
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
        _H, _p, _n, _prof, ctx = build_basis_hamiltonian(frag, "FAO", None,
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
                                    profile=False, optimizer="L-BFGS-B",
                                    gradient_tolerance=1e-5)
        assert result.charges == [0, 1]
        # A bare proton has no electrons: its "energy" is exactly zero.
        assert result.fragment_energies[1] == pytest.approx(0.0, abs=1e-12)
        assert result.complex_result is not None and len(result.results) == 2
        # The proton is attracted to the H2 charge cloud: bound.
        assert result.energy < 0.0

    def test_calculator_method_reuses_its_options(self):
        atoms = _two_h2(5.0)
        calc = Mandacaru(method="vqe", basis="FAO", h=0.4,
                         optimizer="L-BFGS-B")
        result = calc.interaction_energy(atoms, [[0, 1], [2, 3]])
        assert result.method == "vqe" and abs(result.in_units("eV")) < 0.05
