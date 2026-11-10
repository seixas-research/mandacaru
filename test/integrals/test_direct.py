# -*- coding: utf-8 -*-
# file: test/integrals/test_direct.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The native active-space Hamiltonian built without the full tensor, against the tensor.

``MeanField.active_model`` builds only the selected four-index block (the MP2
selection reads an ``oovv`` block from Coulomb factors, never a full
source-orbital tensor); ``MeanField.model`` is the dense Hamiltonian of the
whole basis. Both are reached from a ``Mandacaru`` calculation, and the
selected block must equal the same contraction of the dense one. The dense
tensor is also contracted into the Hartree-Fock energy the native SCF reports.
"""

import json

import numpy as np
import pytest
from ase.build import molecule

from mandacaru import Mandacaru

# Energy-ordered selections, whose active orbitals are canonical ones: the
# reduced dense model is the reference. The MP2 selection rotates the virtual
# space, so its reference is the explicit contraction in the rotated frame.
ENERGY_SPECS = {
    "paw": {"orbitals": 8, "symmetry": True},
    "hao": {"frozen": 1, "orbitals": 6},
}
MP2_SPEC = {"orbitals": 6, "method": "mp2"}
BASES = {"paw": {"name": "PAW-LCAO", "size": "DZP"}, "hao": "HAO"}


def _nh3():
    # Tilted off every grid direction: complex p functions off any symmetry
    # plane, where a wrong conjugation pattern shows.
    atoms = molecule("NH3")
    atoms.center(vacuum=3.0)
    atoms.rotate(41, (1, 2, 3), center="COM")
    return atoms


@pytest.fixture(scope="module", params=["paw", "hao"])
def system(request):
    """The converged reference, the selections built before any dense model
    exists, and whether the reference still held no dense model after them."""
    kind = request.param
    with Mandacaru(method="rhf", basis=BASES[kind], grid_spacing=0.3,
                   resources={"threads": 1}, atoms=_nh3()) as calc:
        calc.get_potential_energy()
        reference = calc.native_reference
        specs = {"energy": ENERGY_SPECS[kind], "mp2": MP2_SPEC}
        selected = {name: reference.active_model(json.dumps(spec))
                    for name, spec in specs.items()}
        no_tensor = not reference.has_model
        yield kind, reference, selected, no_tensor


class TestTheSelectedBlockEqualsTheTensor:
    def test_no_dense_model_was_formed_for_the_selections(self, system):
        _kind, _reference, selected, no_tensor = system
        assert selected
        assert no_tensor

    def test_an_energy_selection_is_the_reduced_dense_model(self, system):
        _kind, reference, selected, _no_tensor = system
        chosen = selected["energy"]
        assert chosen.orbital_rotation is None
        dense = reference.model.reduce(chosen.frozen_indices,
                                       chosen.orbital_indices)
        assert chosen.num_particles == dense.num_particles
        assert chosen.constant_energy == pytest.approx(dense.constant_energy,
                                                       abs=1e-9)
        assert np.abs(chosen.one_body - dense.one_body).max() < 1e-9
        assert np.abs(chosen.two_body - dense.two_body).max() < 1e-9

    def test_an_mp2_selection_is_the_dense_model_in_the_rotated_frame(
            self, system):
        _kind, reference, selected, _no_tensor = system
        chosen = selected["mp2"]
        dense = reference.model
        c = chosen.orbital_rotation[:, chosen.orbital_indices]
        h = c.conj().T @ dense.one_body @ c
        g = np.einsum("ap,bq,cr,ds,abcd->pqrs", c.conj(), c.conj(), c, c,
                      dense.two_body, optimize=True)
        assert chosen.frozen_indices == []
        assert np.abs(chosen.one_body - h).max() < 1e-9
        assert np.abs(chosen.two_body - g).max() < 1e-9 * np.abs(g).max()

    def test_the_tensor_contracts_to_the_native_hartree_fock_energy(
            self, system):
        _kind, reference, _selected, _no_tensor = system
        dense = reference.model
        n = reference.n_occupied
        h, g = dense.one_body, dense.two_body
        occupied = range(n)
        electronic = 2.0 * sum(h[i, i] for i in occupied).real + sum(
            2.0 * g[i, j, i, j] - g[i, j, j, i]
            for i in occupied for j in occupied).real
        assert electronic == pytest.approx(
            reference.snapshot()["electronic_energy"], abs=1e-9)


class TestRefusals:
    """What the native selection does not do raises, it never selects
    something else (``docs/source/guide/active_space.md``)."""

    @pytest.fixture(scope="class")
    def hydroxyl(self):
        atoms = molecule("OH")
        atoms.center(vacuum=3.0)
        with Mandacaru(method="uhf", basis="HAO", grid_spacing=0.3,
                       resources={"threads": 1}, atoms=atoms) as calc:
            calc.get_potential_energy()
            yield calc.native_reference

    def test_an_open_shell_mp2_selection_is_refused(self, hydroxyl):
        with pytest.raises(ValueError, match="closed-shell"):
            hydroxyl.active_model(json.dumps(MP2_SPEC))
        assert not hydroxyl.has_model

    def test_correlating_pairs_are_refused(self, system):
        _kind, reference, _selected, _no_tensor = system
        with pytest.raises(ValueError, match="correlating_pairs"):
            reference.active_model(json.dumps(
                {"orbitals": 8, "correlating_pairs": True}))
