# -*- coding: utf-8 -*-
# file: test/algorithms/test_polarizability.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""A uniform electric field on a molecule, and the finite-field
polarizability built on it."""

import numpy as np
import pytest
from ase.build import molecule

from mandacaru import Mandacaru
from mandacaru.algorithms import Polarizability, polarizability
from mandacaru.units import HARTREE_TO_EV

PAW_DZP = {"name": "PAW-LCAO", "size": "DZP"}
PAW_SZP = {"name": "PAW-LCAO", "size": "SZP"}


def _water(vacuum=2.5):
    """Water turned off every symmetry plane: the Loewdin orbitals are then
    complex, which is where a transposed convention would show."""
    atoms = molecule("H2O")
    atoms.rotate(37, (1, 2, 0.5))
    atoms.center(vacuum=vacuum)
    return atoms


def _run(field=None, method="rhf", basis=PAW_SZP, **kwargs):
    atoms = _water()
    atoms.calc = Mandacaru(method=method, basis=basis, h=0.3, trace=False,
                           electric_field=field, **kwargs)
    energy = atoms.get_potential_energy() / HARTREE_TO_EV
    atoms.calc.get_dipole_moment()
    return energy, np.asarray(atoms.calc.dipole_result.total), atoms


class TestTheField:
    @pytest.mark.slow
    @pytest.mark.parametrize("method", ["rhf", "dft"])
    def test_the_energy_slope_is_minus_the_dipole(self, method):
        """Hellmann-Feynman: the field couples through the same dipole
        matrices the dipole is computed from, augmentation included."""
        _e0, mu0, _ = _run(method=method)
        # One general direction: the molecule is turned off every axis.
        unit = np.array([0.48, -0.6, 0.64])
        step = 1e-3
        e_plus, _, _ = _run(tuple(step * unit), method=method)
        e_minus, _, _ = _run(tuple(-step * unit), method=method)
        slope = (e_plus - e_minus) / (2 * step)
        assert slope == pytest.approx(-float(mu0 @ unit), abs=1e-5)

    def test_the_dipole_matrices_are_the_dipole(self):
        from mandacaru.algorithms.volumetric import (OrbitalExpansion,
                                                     _spinors,
                                                     spin_resolved_rdm)
        _e, _mu, atoms = _run(basis={"name": "PAW-LCAO", "size": "SZ"})
        calc = atoms.calc
        solver, integrals, frozen, active = calc._volumetric_context()
        gamma, _ = calc._state_rdms(solver, psi=calc._volumetric_state(
            solver, 0), two_body=False)
        Da, Db = spin_resolved_rdm(gamma, len(integrals.basis), frozen,
                                   active, spinors=_spinors(integrals))
        mo = OrbitalExpansion(integrals).mo
        D_ao = mo @ (Da + Db) @ mo.conj().T
        electrons = np.real(np.einsum("apq,qp->a",
                                      integrals.dipole_matrices(), D_ao))
        result = calc.dipole_result
        assert np.abs(electrons - (result.smooth + result.augmentation)
                      ).max() < 1e-10

    def test_the_log_names_the_field_once(self, tmp_path):
        atoms = _water()
        log = tmp_path / "run.log"
        atoms.calc = Mandacaru(method="rhf", basis=PAW_SZP, h=0.3,
                               electric_field=(0.0, 0.0, 1e-3), txt=str(log))
        atoms.get_potential_energy()
        lines = [l for l in log.read_text().splitlines() if "electric" in l]
        assert len(lines) == 1 and "Hartree/(e Bohr)" in lines[0]

    def test_what_a_field_cannot_do_is_refused(self):
        with pytest.raises(ValueError, match="three finite numbers"):
            Mandacaru(method="rhf", basis=PAW_DZP, electric_field=(0, 1e-3))
        with pytest.raises(ValueError, match="would ignore it"):
            Mandacaru(method="vqe", load_hamiltonian="h.json",
                      electric_field=(0, 0, 1e-3))
        atoms = _water()
        atoms.calc = Mandacaru(method="dft", basis=PAW_SZP, h=0.3,
                               electric_field=(0, 0, 1e-3), trace=False)
        with pytest.raises(NotImplementedError, match="electric field"):
            atoms.get_forces()
        # A crystal takes a field through the Berry phase (method="dft",
        # test_berry_phase.py); the Bloch supercell methods cannot.
        with pytest.raises(NotImplementedError, match="periodicity"):
            Mandacaru(method="bloch-vqe", basis=PAW_SZP, h=0.4,
                      kpts={"size": (1, 1, 1), "gamma": True},
                      electric_field=(0, 0, 1e-3))


class TestThePolarizability:
    @pytest.fixture(scope="class")
    def water(self):
        atoms = _water()
        return atoms, polarizability(atoms, method="dft", basis=PAW_SZP,
                                     h=0.3, trace=False)

    @pytest.mark.slow
    def test_the_dipoles_and_the_energies_agree(self, water):
        """The dipole derivative and the energy curvature are the same
        tensor for a variational method (Kohn-Sham here); the tensor is
        symmetric."""
        _atoms, result = water
        assert isinstance(result, Polarizability)
        alpha = result.tensor
        assert np.abs(alpha - alpha.T).max() < 1e-3 * np.abs(alpha).max()
        assert np.allclose(result.energy_diagonal, np.diag(alpha), rtol=2e-3)
        assert np.all(np.linalg.eigvalsh(0.5 * (alpha + alpha.T)) > 0)
        assert result.in_units("angstrom^3") == pytest.approx(
            alpha * 0.529177210903 ** 3)
        # The trace and the anisotropy are rotation invariants.
        assert result.isotropic == pytest.approx(np.trace(alpha) / 3)

    def test_the_calculator_reuses_its_options(self, monkeypatch):
        """``calc.polarizability`` is the function with the calculator's
        method, basis, spacing and solver options (overrides win)."""
        import sys
        # The package exports the function under the module's own name.
        module = sys.modules["mandacaru.algorithms.polarizability"]
        seen = {}
        monkeypatch.setattr(module, "polarizability",
                            lambda atoms, **kwargs: seen.update(kwargs))
        calc = Mandacaru(method="dft", xc="pbe", basis=PAW_SZP, h=0.3,
                         trace=False)
        calc.polarizability(_water(), field=1e-3)
        assert seen["method"] == "dft" and seen["basis"] == PAW_SZP
        assert seen["h"] == 0.3 and seen["xc"] == "pbe"
        assert seen["field"] == 1e-3

    def test_a_crystal_and_a_fixed_field_are_refused(self):
        atoms = _water()
        with pytest.raises(ValueError, match="applies the fields itself"):
            polarizability(atoms, method="rhf", basis=PAW_SZP,
                           electric_field=(0, 0, 1e-3))
        atoms.pbc = True
        with pytest.raises(NotImplementedError, match="Berry phase"):
            polarizability(atoms, method="rhf", basis=PAW_SZP)
