# -*- coding: utf-8 -*-
# file: test/algorithms/test_kpoints.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""``kpts=``: the mesh is generated and exposed, then refused at run.

The integral engine is Gamma-point (molecular), so a denser Monkhorst-Pack
mesh is accepted, resolved through ASE and reported on ``.kpoints`` -- and
then raises at run time rather than quietly computing the Gamma-point answer
under a k-point name.
"""

import pytest

from mandacaru.algorithms import Mandacaru
import numpy as np
from mandacaru.core import MolecularIntegrals, minimal_hao_basis
from mandacaru.integrals import Grid


@pytest.fixture(scope="module")
def h2_hamiltonian():
    """A 4-qubit H2 Hamiltonian, built once for the whole module."""
    R = 0.74
    nuclei = [(1.0, np.array([0.0, 0.0, -R / 2])),
              (1.0, np.array([0.0, 0.0, +R / 2]))]
    grid = Grid(center=[0, 0, 0], box_size=5.0, h=0.35)
    integrals = MolecularIntegrals(nuclei, minimal_hao_basis(nuclei), grid)
    return integrals.molecular_hamiltonian(mo_basis=True, n_electrons=2)


class TestKPoints:
    def test_default_is_gamma(self):
        adapt = Mandacaru(method="adapt-vqe", pool="ceo", basis="HAO")
        assert adapt.kpts == (1, 1, 1)
        assert len(adapt.kpoints) == 1
        np.testing.assert_allclose(adapt.kpoints[0], [0.0, 0.0, 0.0])

    def test_mesh_generated_via_ase_monkhorst_pack(self):
        from ase.dft.kpoints import monkhorst_pack
        adapt = Mandacaru(method="adapt-vqe", pool="ceo", basis="HAO",
                          kpts=(2, 2, 1))
        np.testing.assert_allclose(adapt.kpoints, monkhorst_pack((2, 2, 1)))

    def test_non_gamma_rejected_at_run(self, h2_hamiltonian):
        adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                          pool="ceo", num_particles=(1, 1),
                          n_spatial_orbitals=2, kpts=(2, 1, 1))
        with pytest.raises(NotImplementedError, match="Monkhorst-Pack"):
            adapt.run()

    def test_invalid_kpts_rejected(self):
        with pytest.raises(ValueError):
            Mandacaru(method="adapt-vqe", pool="ceo", basis="HAO", kpts=(1, 1))       # not length-3

    def test_dict_spec_with_gamma_centering(self):
        # ASE dict form {"size": ..., "gamma": True}: Gamma-centered mesh.
        adapt = Mandacaru(method="adapt-vqe", pool="ceo", basis="HAO",
                          kpts={"size": (2, 2, 1), "gamma": True})
        assert adapt.kpts == (2, 2, 1)
        assert adapt.kpts_gamma is True
        # Gamma-centering shifts the even-axis mesh so it includes the Gamma point.
        assert any(np.allclose(k, [0, 0, 0]) for k in adapt.kpoints)

    def test_dict_gamma_only(self):
        adapt = Mandacaru(method="adapt-vqe", pool="ceo", basis="HAO",
                          kpts={"size": (1, 1, 1), "gamma": True})
        assert len(adapt.kpoints) == 1 and adapt.kpts_gamma is True
