# -*- coding: utf-8 -*-
# file: test/test_fao_forces.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Forces in the all-electron FAO basis, and what they are worth.

Two separate claims, which this file keeps apart on purpose.

**The gradient is right.** For H2 and LiH, with and without the
``virtual_orbitals`` levels, the analytic force matches a central difference of
the *same* energy on the *same* frozen grid to ~1e-4 relative.  Whatever the
energy surface is, the force is its derivative.

**The surface underneath is not, once an atom has a core.** Translating a free
molecule cannot change its energy, so the forces must sum to zero.  On H2 they
do.  On LiH the Li 1s is far too sharp for any affordable grid spacing and the
net force reaches hundreds of eV/Angstrom -- larger than every real force in the
problem -- while the energy, the RDMs and the orbital-response residual all look
healthy.  ``Carcara`` warns about exactly that
(:data:`~carcara.algorithms.calculator.TRANSLATIONAL_RESIDUAL_TOLERANCE`), and
the same geometry in the PAW basis, which removes the core instead of sampling
it, gives a net force of ~0.03 eV/Angstrom and a bond force that matches VASP.
"""

import warnings

import numpy as np
import pytest
from ase import Atoms

from carcara import Carcara
from carcara.integrals import Grid
from test_all_electron_forces import analytic_and_numerical

# VASP 6 PBE PAW: bond-projected force on atom 0 (eV/A, + = toward atom 1).
VASP_LIH_2_19 = 1.35493
VASP_H2_1_00 = 4.22206

PAW = {"name": "PAW", "size": "DZP"}


def dimer(symbols, distance, cell=10.0):
    """The dimer along z, off the grid nodes."""
    atoms = Atoms(symbols, positions=[[0, 0, 0], [0, 0, distance]],
                  cell=[cell] * 3)
    atoms.center()
    atoms.positions += np.array([0.013, -0.021, 0.007])
    return atoms


def forces_of(symbols, distance, basis, cell=9.0, h=0.25, **options):
    atoms = dimer(symbols, distance, cell)
    atoms.calc = Carcara(method="adapt-vqe", basis=basis, h=h, pool="fermionic",
                         optimizer="L-BFGS-B", max_iterations=60,
                         gradient_tolerance=1e-6, profile=False,
                         **options)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atoms.get_forces()
    # The unprojected gradient: these tests measure the translational artifact
    # itself, which the default projection exists to remove from what is
    # reported (see ``ForceResult.unprojected``).
    return atoms.calc.force_result.unprojected, atoms.calc


def bond_force(forces):
    """Force on atom 0 along the bond (+ = toward atom 1)."""
    return float((forces[0, 2] - forces[1, 2]) / 2)


def net_force(forces):
    """``|sum_A F_A|`` -- zero for an exact free molecule."""
    return float(np.abs(np.asarray(forces).sum(axis=0)).max())


# --------------------------------------------------------------------------- #
# The gradient is the derivative of the energy.
# --------------------------------------------------------------------------- #

class TestGradientIsExact:
    @pytest.mark.parametrize("virtual_orbitals", [0, 1])
    def test_h2(self, virtual_orbitals):
        """A compact box on purpose: this compares the analytic gradient with a
        finite difference of the *same* energy, so it tests internal
        consistency, not basis convergence.  Clipping the diffuse 2s tail
        changes both sides identically and keeps the grid cheap."""
        grid = Grid(center=[3.0, 3.0, 3.0], box_size=6.0, h=0.35,
                    units="angstrom")
        atoms = Atoms("H2", positions=[[3.013, 2.979, 2.607],
                                       [3.013, 2.979, 3.407]], cell=[6.0] * 3)
        forces, numerical = analytic_and_numerical(
            atoms, {"name": "FAO", "virtual_orbitals": virtual_orbitals},
            grid, 0.35)
        scale = np.abs(numerical).max()
        assert scale > 1.0                              # a real force
        assert np.abs(forces - numerical).max() < 1e-3 * scale

    def test_lih(self):
        """Correct even where the surface it differentiates is not usable."""
        grid = Grid(center=[3.5, 3.5, 3.5], box_size=7.0, h=0.35,
                    units="angstrom")
        atoms = Atoms("LiH", positions=[[3.513, 3.479, 2.607],
                                        [3.513, 3.479, 4.407]], cell=[7.0] * 3)
        forces, numerical = analytic_and_numerical(atoms, "FAO", grid, 0.35)
        scale = np.abs(numerical).max()
        assert np.abs(forces - numerical).max() < 1e-3 * scale


# --------------------------------------------------------------------------- #
# H2: no core, so the forces are usable.
# --------------------------------------------------------------------------- #

class TestH2:
    @pytest.fixture(scope="class")
    def stretched(self):
        return forces_of("H2", 1.00, "FAO", cell=7.0)[0]

    def test_the_forces_are_physical(self, stretched):
        assert np.abs(stretched[:, :2]).max() < 1e-6      # along the bond only
        assert net_force(stretched) < 1e-6               # translation invariant
        assert stretched[0, 2] == pytest.approx(-stretched[1, 2], abs=1e-9)

    def test_a_stretched_bond_pulls_back(self, stretched):
        """Same sign as VASP, and the same order of magnitude."""
        assert bond_force(stretched) > 0
        assert bond_force(stretched) == pytest.approx(VASP_H2_1_00, rel=0.8)

    def test_a_compressed_bond_pushes_apart(self):
        forces, _ = forces_of("H2", 0.60, "FAO", cell=7.0)
        assert bond_force(forces) < 0

    def test_virtual_levels_keep_the_forces_clean(self):
        """The extra levels change the surface, not its symmetry."""
        forces, calc = forces_of("H2", 1.00,
                                 {"name": "FAO", "virtual_orbitals": 1},
                                 cell=7.0)
        assert calc.n_qubits == 8
        assert np.abs(forces[:, :2]).max() < 1e-6
        assert net_force(forces) < 1e-6
        assert bond_force(forces) > 0


# --------------------------------------------------------------------------- #
# LiH: the Li 1s core breaks the energy surface, and Carcara says so.
# --------------------------------------------------------------------------- #

class TestLiHCoreArtifact:
    def test_the_net_force_is_enormous(self):
        """No physical force here exceeds a few eV/Angstrom."""
        forces, calc = forces_of("LiH", 2.19265, "FAO")
        residual = calc.force_result.details["translational_residual"]
        assert residual == pytest.approx(net_force(forces), rel=1e-12)
        assert residual > 10.0
        assert residual > np.abs(forces).max() / 10      # it dominates

    def test_carcara_warns(self):
        atoms = dimer("LiH", 2.19265)
        atoms.calc = Carcara(method="adapt-vqe", basis="FAO", h=0.25,
                             pool="fermionic", optimizer="L-BFGS-B",
                             max_iterations=60, gradient_tolerance=1e-6, profile=False)
        with pytest.warns(RuntimeWarning, match="do not sum to zero"):
            atoms.get_forces()

    def test_freezing_the_core_does_not_cure_it(self):
        """The core leaves the active space, not the density or the grid."""
        forces, _ = forces_of("LiH", 2.19265, "FAO", frozen_core=True)
        assert net_force(forces) > 10.0

    @pytest.fixture(scope="class")
    def paw(self):
        """The PAW-DZP run of the same geometry, done once (it is the
        heaviest calculation in this file)."""
        atoms = dimer("LiH", 2.19265)
        atoms.calc = Carcara(method="adapt-vqe", basis=PAW, h=0.25,
                             pool="fermionic", optimizer="L-BFGS-B",
                             max_iterations=60, gradient_tolerance=1e-6, profile=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            forces = atoms.get_forces()
            messages = [str(w.message) for w in caught]
        return forces, dict(atoms.calc.force_result.details), messages

    def test_the_paw_basis_is_clean_on_the_same_geometry(self, paw):
        forces, details, _ = paw
        assert net_force(forces) < 0.2
        assert details["translational_residual"] < 0.2
        assert bond_force(forces) == pytest.approx(VASP_LIH_2_19, rel=0.3)

    def test_no_warning_for_the_pseudopotential_basis(self, paw):
        assert not [m for m in paw[2] if "do not sum to zero" in m]


# --------------------------------------------------------------------------- #
# The check itself.
# --------------------------------------------------------------------------- #

class TestTranslationalCheck:
    def test_a_single_atom_is_exempt(self):
        """The grid re-centers on it, so the residual carries no information."""
        from carcara.algorithms.calculator import Carcara as C
        from carcara.algorithms.forces import ForceResult

        one = ForceResult(forces=np.array([[3.0, 0.0, 0.0]]),
                          hellmann_feynman=np.zeros((1, 3)),
                          pulay=np.zeros((1, 3)), gradient=np.zeros((1, 3)))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert C._check_translational_invariance(one) == 0.0

    def test_a_small_residual_is_tolerated(self):
        from carcara.algorithms.calculator import Carcara as C
        from carcara.algorithms.forces import ForceResult

        forces = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, -4.99]])
        result = ForceResult(forces=forces, hellmann_feynman=np.zeros((2, 3)),
                             pulay=np.zeros((2, 3)), gradient=np.zeros((2, 3)))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            residual = C._check_translational_invariance(result)
        assert residual == pytest.approx(0.01)
        assert result.details["translational_residual"] == pytest.approx(0.01)


# --------------------------------------------------------------------------- #
# When the translational projection applies.
# --------------------------------------------------------------------------- #

class TestTranslationProjectionPolicy:
    """``project_translation="auto"`` projects exactly where the identity holds.

    A free molecule's exact forces sum to zero, so subtracting the mean is an
    orthogonal projection onto a subspace containing the true answer -- it cannot
    move the estimate away from it.  Under periodic boundary conditions that
    identity is not the one being enforced, and for a single atom the residual is
    zero by construction (the grid re-centers on it), so neither is projected.
    """

    @staticmethod
    def _run(**options):
        atoms = dimer("H2", 0.74, 8.0)
        atoms.calc = Carcara(method="adapt-vqe", basis="FAO", h=0.35,
                             pool="fermionic", max_iterations=4,
                             gradient_tolerance=1e-4, profile=False,
                             **options)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forces = atoms.get_forces()
        return forces, atoms.calc.force_result

    def test_auto_is_the_default_and_projects_a_free_molecule(self):
        forces, result = self._run()
        assert net_force(forces) < 1e-12
        assert result.details["translation_projected"] is True
        # Nothing is lost: the raw gradient and the residual are both kept.
        assert net_force(result.unprojected) == pytest.approx(
            result.details["translational_residual"], rel=1e-9)

    def test_false_reports_the_raw_gradient(self):
        projected, _ = self._run()
        raw, result = self._run(project_translation=False)
        assert "translation_projected" not in result.details
        assert result.unprojected is result.forces
        # The two differ by exactly the mean that was removed.
        assert np.allclose(projected, raw - raw.mean(axis=0), atol=1e-10)

    def test_auto_leaves_a_periodic_system_alone(self):
        atoms = dimer("H2", 0.74, 8.0)
        atoms.pbc = True
        atoms.calc = Carcara(method="adapt-vqe", basis="FAO", h=0.35,
                             pool="fermionic", max_iterations=4,
                             gradient_tolerance=1e-4, profile=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forces = atoms.get_forces()
        # The sum over a cell's atoms is not the free-molecule identity.
        assert "translation_projected" not in atoms.calc.force_result.details
        assert net_force(forces) > 0.0

    def test_explicit_true_on_a_periodic_system_says_it_is_ignored(self):
        atoms = dimer("H2", 0.74, 8.0)
        atoms.pbc = True
        atoms.calc = Carcara(method="adapt-vqe", basis="FAO", h=0.35,
                             pool="fermionic", max_iterations=4,
                             gradient_tolerance=1e-4, profile=False,
                             project_translation=True)
        with pytest.warns(RuntimeWarning, match="ignored for a periodic"):
            atoms.get_forces()

    @pytest.mark.parametrize("value", ["yes", 1, None, "AUTO"])
    def test_an_unknown_setting_is_refused(self, value):
        with pytest.raises(ValueError, match="project_translation must be"):
            Carcara(method="adapt-vqe", project_translation=value)
