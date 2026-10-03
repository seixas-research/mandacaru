# -*- coding: utf-8 -*-
# file: test/algorithms/test_dlpno_mp2.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""DLPNO-MP2: exact in its limit, local by default, integral-direct."""

import numpy as np
import pytest
from ase.build import molecule

from mandacaru.algorithms.dlpno_mp2 import dlpno_mp2
from mandacaru.algorithms.mp2 import mp2_natural_orbitals
from mandacaru.algorithms.orbital_integrals import (DirectOrbitalIntegrals,
                                                    TensorOrbitalIntegrals)
from mandacaru.core.hamiltonian import molecular_orbital_integrals


def _problem(name, size="DZP", h=0.3):
    from mandacaru.algorithms._hamiltonian_from_atoms import \
        build_basis_hamiltonian

    atoms = molecule(name)
    atoms.center(vacuum=3.0)
    _none, particles, _M, _profile, context = build_basis_hamiltonian(
        atoms, {"name": "PAW-LCAO", "size": size}, None, h, 0, None,
        hamiltonian=False)
    integrals = context["integrals"]
    n = sum(particles)
    h, g, C = molecular_orbital_integrals(integrals, n, particles, None)
    return integrals, n // 2, h, g, C


@pytest.fixture(scope="module")
def water():
    return _problem("H2O")


@pytest.fixture(scope="module")
def propane():
    # SZ keeps the tensor reference cheap (10 s; ethane DZP took 65 s) while
    # the domains still truncate: 9.4 PNOs per pair against 10 in full.
    return _problem("C3H8", "SZ", 0.35)


class TestTheCanonicalLimit:
    """Full domains and no PNO cutoff: the PNO space is the whole virtual
    space, and DLPNO-MP2 is canonical MP2 exactly."""

    def test_energy_density_and_natural_occupations(self, propane):
        integrals, o, h, g, C = propane
        reference = mp2_natural_orbitals(np.real(h), np.real(g), o)
        local = dlpno_mp2(TensorOrbitalIntegrals(h, g, integrals, C), o,
                          domains="full", pno_cutoff=0.0)
        assert local.correlation_energy == pytest.approx(
            reference.correlation_energy, abs=1e-9)
        assert np.allclose(local.occupied_density, reference.occupied_density,
                           atol=1e-8)
        assert np.allclose(local.virtual_occupations,
                           reference.virtual_occupations, atol=1e-9)


class TestLocalDomains:
    def test_the_defaults_recover_almost_all_the_correlation(self, propane):
        integrals, o, h, g, C = propane
        reference = mp2_natural_orbitals(np.real(h), np.real(g), o)
        local = dlpno_mp2(TensorOrbitalIntegrals(h, g, integrals, C), o)
        recovered = local.correlation_energy / reference.correlation_energy
        # Measured 99.999% here and 99.986% on ethane DZP (TCutDO 1e-2,
        # TCutPNO 1e-8); Loewdin-population domains alone gave 96.6% there.
        assert recovered > 0.999
        full = dlpno_mp2(TensorOrbitalIntegrals(h, g, integrals, C), o,
                         domains="full", pno_cutoff=0.0)
        assert (np.mean(list(local.pno_counts.values()))
                < np.mean(list(full.pno_counts.values())))

    def test_the_direct_provider_agrees_with_the_tensor(self, water):
        integrals, o, h, g, C = water
        tensor = dlpno_mp2(TensorOrbitalIntegrals(h, g, integrals, C), o)
        direct = dlpno_mp2(DirectOrbitalIntegrals(integrals, C), o)
        assert direct.correlation_energy == pytest.approx(
            tensor.correlation_energy, abs=1e-11)
        assert np.allclose(direct.virtual_occupations,
                           tensor.virtual_occupations, atol=1e-11)


class TestLocality:
    def test_local_boxes_reproduce_the_global_exchange_blocks(self, water):
        integrals, o, h, g, C = water
        global_ = dlpno_mp2(DirectOrbitalIntegrals(integrals, C), o)
        try:
            # A negligible threshold: the boxes cover every density, so the
            # local path must give the global numbers to round-off.
            integrals.direct_box_threshold = 1e-30
            integrals._direct = None
            local = dlpno_mp2(DirectOrbitalIntegrals(integrals, C), o)
        finally:
            integrals.direct_box_threshold = None
            integrals._direct = None
        assert local.correlation_energy == pytest.approx(
            global_.correlation_energy, abs=1e-10)

    def test_weak_pairs_are_dropped_and_their_estimate_kept(self, propane):
        integrals, o, h, g, C = propane
        provider = TensorOrbitalIntegrals(h, g, integrals, C)
        every = dlpno_mp2(provider, o, pair_cutoff=0.0)
        # Small domains and a coarse cutoff so that pairs are screened.
        screened = dlpno_mp2(provider, o, doi_cutoff=0.3, pair_cutoff=1e-4)
        reference = dlpno_mp2(provider, o, doi_cutoff=0.3, pair_cutoff=0.0)
        assert len(screened.pno_counts) < len(reference.pno_counts)
        assert len(every.pno_counts) == o * (o + 1) // 2
        # The dropped pairs' dipole estimate stands in for them: the
        # screened energy stays within the cutoff's order of the full one.
        dropped = len(reference.pno_counts) - len(screened.pno_counts)
        assert abs(screened.correlation_energy
                   - reference.correlation_energy) < dropped * 1e-4


class TestTheActiveSpaceMethod:
    def test_it_selects_without_forming_the_tensor(self):
        from mandacaru import Mandacaru

        spaces = {}
        for method in ("mp2", "dlpno-mp2"):
            atoms = molecule("H2O")
            atoms.center(vacuum=3.0)
            atoms.calc = Mandacaru(
                method="adapt-vqe", basis={"name": "PAW-LCAO", "size": "DZP"},
                h=0.3, active_space={"orbitals": 6, "method": method},
                max_iterations=1, trace=False)
            atoms.get_potential_energy()
            context = atoms.calc.solver._gradient_context
            spaces[method] = context["active_space"]
            if method == "dlpno-mp2":
                integrals = context["integrals"]
                assert integrals._eri is None and integrals._eri_ao is None
        assert spaces["dlpno-mp2"].active == spaces["mp2"].active
        assert spaces["dlpno-mp2"].correlation_energy == pytest.approx(
            spaces["mp2"].correlation_energy, rel=1e-4)

    def test_forces_match_the_mp2_active_space(self):
        # The gradient is the central difference of the rebuilt reduced
        # Hamiltonian, so it carries the response of the DLPNO selection; on
        # H2 it equals the canonical MP2 selection's force.  (Against a
        # total-energy difference along an internal direction of H2O DZP the
        # two agree to 3.5e-3 eV/Angstrom.)
        from mandacaru import Mandacaru

        forces = {}
        for method in ("mp2", "dlpno-mp2"):
            atoms = molecule("H2")
            atoms.center(vacuum=3.0)
            atoms.calc = Mandacaru(
                method="adapt-vqe",
                basis={"name": "PAW-LCAO", "size": "DZP"}, h=0.35,
                active_space={"orbitals": 2, "method": method},
                max_iterations=2, trace=False)
            forces[method] = atoms.get_forces()
            if method == "dlpno-mp2":
                result = atoms.calc.force_result
                assert result.hellmann_feynman is None
                context = atoms.calc.solver._gradient_context
                assert context["integrals"]._eri_ao is None
        assert np.allclose(forces["dlpno-mp2"], forces["mp2"], atol=1e-4)
        assert np.abs(forces["mp2"]).max() > 1e-2

    def test_an_open_shell_is_refused(self):
        from mandacaru.algorithms.active_space import (resolve_active_space,
                                                       resolve_active_space_spec)

        spec = resolve_active_space_spec({"orbitals": 3,
                                          "method": "dlpno-mp2"})
        with pytest.raises(NotImplementedError, match="closed-shell only"):
            resolve_active_space(n_orbitals=6, num_particles=(2, 1),
                                 spec=spec, open_shell=True)
