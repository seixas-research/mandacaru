# -*- coding: utf-8 -*-
# file: test/pseudopotentials/test_families.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The family registry as the basis argument sees it
(:mod:`mandacaru.pseudopotentials.families`)."""

import pytest
from ase import Atoms

from mandacaru.integrals import Grid


class TestBasisArgumentIsHonored:
    """The family is the basis name; its options are the size hierarchy."""

    def test_size_is_forwarded(self):
        from mandacaru.algorithms._hamiltonian_from_atoms import (
            resolve_basis, resolve_pseudo_basis)

        family, options = resolve_pseudo_basis(
            *resolve_basis({"name": "ONCVPSP", "size": "DZP"}), ["O"])
        assert family.label == "ONCVPSP" and options["size"] == "DZP"

    @pytest.mark.parametrize("name", ["ONCVPSP", "oncvpsp", "ONCV", "oncv"])
    def test_the_family_is_accepted_by_every_alias(self, name):
        from mandacaru.algorithms._hamiltonian_from_atoms import (
            resolve_basis, resolve_pseudo_basis)

        family, options = resolve_pseudo_basis(
            *resolve_basis({"name": name, "size": "DZ"}), ["O"])
        assert family.name == "oncvpsp" and options == {"size": "DZ"}

    def test_all_electron_names_stay_all_electron(self):
        from mandacaru.algorithms._hamiltonian_from_atoms import (
            resolve_basis, resolve_pseudo_basis)

        assert resolve_pseudo_basis(*resolve_basis("HAO"), ["O"]) == (None, {})

    @pytest.mark.parametrize("basis", ["PP", "PSEUDO", {"name": "pp",
                                                        "size": "DZ"}])
    def test_the_retired_names_are_refused(self, basis):
        """``basis="PP"`` was the old spelling; it must not alias silently."""
        from mandacaru.algorithms._hamiltonian_from_atoms import resolve_basis

        with pytest.raises(ValueError, match="no longer a basis name"):
            resolve_basis(basis)

    def test_options_of_other_families_are_refused(self):
        from mandacaru.algorithms._hamiltonian_from_atoms import (
            build_basis_hamiltonian)

        atoms = Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]])
        grid = Grid(center=[0, 0, 0], box_size=6.0, h=0.30)
        # `projector_basis` belongs to PAW-LCAO alone (the confinement
        # options are shared by every pseudopotential family).
        with pytest.raises(ValueError, match="unknown option.*projector_basis"):
            build_basis_hamiltonian(atoms, {"name": "ONCVPSP",
                                            "projector_basis": "raw"},
                                    grid, 0.30, 0, None)
