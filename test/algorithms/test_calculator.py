# -*- coding: utf-8 -*-
# file: test/algorithms/test_calculator.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

""":class:`~mandacaru.algorithms.Mandacaru`: the options it accepts and refuses.

The single entry point validates its options in the constructor, so an option
a method would accept and never act on is refused there rather than producing
a run that reports nothing.  Two outputs that resolve to one path are refused
for the same reason: each writes a different document.
"""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.circuits import UCCSD
from mandacaru.core.mapping import PauliSum
from mandacaru.core.serialization import load_hamiltonian, save_hamiltonian
import os
from mandacaru.core import WavefunctionCheckpoint
from mandacaru.optimizers.optim import OptimizeResult, Optimizer


HAMILTONIAN = PauliSum({"ZIII": 1.0, "IXII": 0.4})
ANGLES = np.array([0.4, -0.7, 0.6])


class KeepInitial(Optimizer):
    """A zero-update optimizer: the result is the state at the given angles."""

    def minimize(self, cost, x0, callback=None):
        x = np.asarray(x0, dtype=float)
        energy = cost(x)
        return OptimizeResult(x, energy, 1, [energy], False, "zero budget")


#: A direct-mode (no geometry) problem: the smallest register that carries a
#: real ansatz, in Hartree so the numbers read as the solver stores them.
DIRECT = dict(num_particles=(1, 1), n_spatial_orbitals=2, atomic_units=True,
              trace=False, profile=False)


def h2(distance=0.74, cell=6.0):
    """H2 centered in a cubic cell (the grid is cut from it)."""
    middle = cell / 2
    return Atoms("H2", positions=[[middle, middle, middle - distance / 2],
                                  [middle, middle, middle + distance / 2]],
                 cell=[cell] * 3)


class TestReportingOptionsAreHonest:
    """An option that writes a file either writes it or is refused."""

    def test_a_subspace_method_refuses_a_log_it_would_ignore(self):
        # `txt=` is a driver-wide option now (every driver has to know where it
        # reports), so the refusal is about the *capability*, not the keyword.
        with pytest.raises(NotImplementedError, match="does not write 'txt'"):
            Mandacaru(method="subspace-vqe", basis="HAO", txt="x.txt")

    def test_vqe_accepts_the_log(self, tmp_path):
        Mandacaru(method="vqe", basis="HAO", txt=str(tmp_path / "out.txt"))

    def test_a_subspace_method_refuses_checkpoints(self):
        with pytest.raises(NotImplementedError, match="does not write"):
            Mandacaru(method="subspace-adapt-vqe", basis="HAO",
                      checkpoint="x.json")

    def test_an_unknown_option_is_not_kept_as_an_ase_parameter(self):
        with pytest.raises(TypeError, match="does not take 'nonsense'"):
            Mandacaru(method="adapt-vqe", basis="HAO", nonsense=1)

    def test_vqe_keeps_a_checkpoint_it_does_support(self, tmp_path):
        Mandacaru(method="vqe", basis="HAO", checkpoint=str(tmp_path / "c.json"))

    def test_a_run_without_a_file_keeps_its_trace(self, capsys, tmp_path):
        """The trap: no file *and* no trace is a run that reports nothing."""
        atoms = h2()
        atoms.calc = Mandacaru(method="rhf", basis="HAO", grid_spacing=0.45)
        atoms.get_potential_energy()
        assert "[SCF SUMMARY]" in capsys.readouterr().out


class TestOutputPathsAndSnapshots:
    def test_the_default_dump_and_cache_names_differ(self):
        from mandacaru.core.serialization import resolve_save_path
        from mandacaru.utils.dumps import HAMILTONIAN_FILE, resolve_dump_path

        assert resolve_dump_path(True, HAMILTONIAN_FILE) != \
            resolve_save_path(True, "json")

    def test_two_outputs_on_one_path_are_refused(self, tmp_path):
        path = str(tmp_path / "same.json")
        with pytest.raises(ValueError, match="both resolve to"):
            Mandacaru(method="adapt-vqe", hamiltonian=PauliSum({"ZI": 1.0}),
                      num_particles=(1, 1), n_spatial_orbitals=1,
                      save_hamiltonian=path, verbose_hamiltonian=path)

    def test_an_inspection_dump_is_not_loadable_as_a_cache(self, tmp_path):
        from mandacaru.utils.dumps import dump_hamiltonian

        path = str(tmp_path / "h.json")
        dump_hamiltonian(path, PauliSum({"ZI": 1.0}), n_qubits=2)
        # Used to fail with KeyError: 0.
        with pytest.raises(ValueError, match="inspection"):
            load_hamiltonian(path)

    def test_a_failed_write_keeps_the_previous_snapshot(self, tmp_path,
                                                       monkeypatch):
        path = str(tmp_path / "h.json")
        save_hamiltonian(path, PauliSum({"ZI": 1.0}), num_particles=(1, 1),
                         n_spatial_orbitals=1)
        good = open(path, encoding="utf-8").read()

        import mandacaru.core.serialization as serialization
        def explode(*args, **kwargs):
            raise OSError("disk full")
        monkeypatch.setattr(serialization, "_write_json", explode)
        with pytest.raises(OSError):
            save_hamiltonian(path, PauliSum({"ZI": 2.0}), num_particles=(1, 1),
                             n_spatial_orbitals=1)
        # The snapshot that was there is still there, and still loadable.
        assert open(path, encoding="utf-8").read() == good
        assert load_hamiltonian(path).hamiltonian.terms == {"ZI": 1.0 + 0j}
        assert not [name for name in os.listdir(tmp_path)
                    if name.endswith(".tmp")]


class TestOutputCollisions:
    @pytest.mark.parametrize("method, extra", [
        ("vqe", {"ansatz": UCCSD(2, (1, 1))}),
        ("adapt-vqe", {"num_particles": (1, 1), "n_spatial_orbitals": 2}),
    ])
    def test_checkpoint_may_not_share_the_cache_path(self, tmp_path, method,
                                                     extra):
        path = str(tmp_path / "same.json")
        with pytest.raises(ValueError, match="both resolve to"):
            Mandacaru(method=method, hamiltonian=HAMILTONIAN, trace=False,
                      save_hamiltonian=path, checkpoint=path,
                      hamiltonian_format="json", **extra)
        assert not os.path.exists(path)              # refused before any write

    def test_a_symlinked_directory_is_still_the_same_file(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        alias = tmp_path / "alias"
        alias.symlink_to(real, target_is_directory=True)
        with pytest.raises(ValueError, match="both resolve to"):
            Mandacaru(method="vqe", hamiltonian=HAMILTONIAN,
                      ansatz=UCCSD(2, (1, 1)), trace=False,
                      save_hamiltonian=str(real / "h.json"),
                      checkpoint=str(alias / "h.json"))

    def test_distinct_paths_are_fine(self, tmp_path):
        calc = Mandacaru(method="vqe", hamiltonian=HAMILTONIAN,
                         ansatz=UCCSD(2, (1, 1)), trace=False,
                         optimizer=KeepInitial(),
                         save_hamiltonian=str(tmp_path / "h.json"),
                         checkpoint=str(tmp_path / "state.json"))
        calc.run(initial_parameters=ANGLES)
        assert load_hamiltonian(str(tmp_path / "h.json")).num_qubits == 4
        assert WavefunctionCheckpoint.load(str(tmp_path / "state.json"))


class TestCholeskyTwoBody:
    """``two_body={"method": "cholesky", ...}``: the same energies as the
    dense tensor, without it; and the cases that need the tensor refused."""

    @staticmethod
    def _water():
        from ase.build import molecule

        atoms = molecule("H2O")
        atoms.center(vacuum=3.0)
        return atoms

    def test_energies_match_the_dense_tensor(self):
        """RHF, then ADAPT-VQE in a 4-orbital active space, PAW-LCAO water:
        a tight tolerance gives the dense energies to 1e-7 Ha, and the
        factored run never forms the tensor."""
        from mandacaru.units import to_hartree

        basis = {"name": "PAW-LCAO", "size": "SZP"}
        energies = {}
        for two_body in ("dense", {"method": "cholesky", "tolerance": 1e-9}):
            key = "dense" if two_body == "dense" else "cholesky"
            atoms = self._water()
            atoms.calc = Mandacaru(method="rhf", basis=basis, grid_spacing=0.35,
                                   trace=False, two_body=two_body)
            rhf = to_hartree(atoms.get_potential_energy(), "eV")
            integrals = atoms.calc.solver._gradient_context["integrals"]
            assert (integrals._eri_ao is None) == (key == "cholesky")
            atoms = self._water()
            atoms.calc = Mandacaru(method="adapt-vqe", basis=basis, grid_spacing=0.35,
                                   trace=False, two_body=two_body,
                                   active_space={"orbitals": 4},
                                   pool="fermionic",
                                   optimizer={"method": "SLSQP",
                                              "maxiter": 500, "tol": 1e-10})
            adapt = to_hartree(atoms.get_potential_energy(), "eV")
            energies[key] = (rhf, adapt)
        assert energies["cholesky"][0] == pytest.approx(energies["dense"][0],
                                                        abs=1e-7)
        assert energies["cholesky"][1] == pytest.approx(energies["dense"][1],
                                                        abs=1e-7)

    def test_integral_noise_stays_out_of_the_operator(self):
        """Integrals that vanish by symmetry come out of the factors as noise
        below the tolerance; the operator leaves them out, so the qubit
        Hamiltonian stays near the dense one's size with the same energy."""
        hamiltonians, energies = {}, {}
        for key, two_body in (("dense", "dense"), ("cholesky", "cholesky")):
            atoms = self._water()
            atoms.calc = Mandacaru(method="adapt-vqe",
                                   basis={"name": "PAW-LCAO", "size": "SZP"},
                                   grid_spacing=0.35, trace=False,
                                   two_body=two_body,
                                   active_space={"orbitals": 6,
                                                 "method": "mp2"},
                                   max_iterations=4)
            energies[key] = atoms.get_potential_energy()
            hamiltonians[key] = atoms.calc.solver.hamiltonian
        integrals = atoms.calc.solver._gradient_context["integrals"]
        floor = integrals.two_body_spec.tolerance
        # The reduced integrals the forces contract are kept whole.
        assert np.any((np.abs(integrals.active_two_body) > 0)
                      & (np.abs(integrals.active_two_body) < floor))
        assert (len(hamiltonians["cholesky"].terms)
                < 1.25 * len(hamiltonians["dense"].terms))
        assert energies["cholesky"] == pytest.approx(energies["dense"],
                                                     abs=1e-4)

    def test_the_option_is_validated(self):
        with pytest.raises(ValueError, match="unknown two_body method"):
            Mandacaru(method="rhf", two_body="cd", trace=False)
        with pytest.raises(ValueError, match="positive"):
            Mandacaru(method="rhf", trace=False,
                      two_body={"method": "cholesky", "tolerance": -1.0})
        with pytest.raises(ValueError, match="no tolerance"):
            Mandacaru(method="rhf", trace=False,
                      two_body={"method": "dense", "tolerance": 1e-6})

    @pytest.mark.parametrize("method", ["dft", "ghf"])
    def test_methods_that_need_the_tensor_are_refused(self, method):
        with pytest.raises(NotImplementedError, match="dense"):
            Mandacaru(method=method, trace=False, two_body="cholesky")

    def test_forces_without_an_active_space_are_refused(self):
        """Without an active space the force would be the analytic gradient,
        which differentiates the truncated factors themselves."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]],
                      cell=[6] * 3)
        atoms.calc = Mandacaru(method="adapt-vqe", grid_spacing=0.35, trace=False,
                               two_body="cholesky")
        with pytest.raises(NotImplementedError, match="need an active_space"):
            atoms.get_forces()

    def test_active_space_forces_match_the_dense_tensor(self, monkeypatch):
        """With deleted virtuals the force is the central difference of the
        reduced Hamiltonian, each displaced geometry factored on the
        reference's pivots: it equals the dense route's force, with no
        tensor formed and no Hellmann-Feynman/Pulay split."""
        import mandacaru.integrals.cholesky as cholesky

        fixed = []
        adaptive = cholesky.cholesky_two_body

        def spy(*args, **kwargs):
            fixed.append(kwargs.get("pivots") is not None)
            return adaptive(*args, **kwargs)

        monkeypatch.setattr(cholesky, "cholesky_two_body", spy)
        forces = {}
        for key, two_body in (("dense", "dense"),
                              ("cholesky", {"method": "cholesky",
                                            "tolerance": 1e-9})):
            atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]],
                          cell=[6] * 3)
            atoms.calc = Mandacaru(
                method="adapt-vqe", basis={"name": "PAW-LCAO", "size": "DZP"},
                grid_spacing=0.35, two_body=two_body,
                active_space={"orbitals": 2, "method": "mp2"},
                max_iterations=2, trace=False)
            forces[key] = atoms.get_forces()
        assert atoms.calc.force_result.hellmann_feynman is None
        assert atoms.calc.solver._gradient_context["integrals"]._eri_ao is None
        # The reference factors adaptively; both sides of H2's one internal
        # direction reuse its pivots.
        assert fixed == [False, True, True]
        assert np.allclose(forces["cholesky"], forces["dense"], atol=1e-5)
        assert np.abs(forces["dense"]).max() > 1e-2

    def test_a_crystal_is_refused(self):
        from mandacaru.algorithms.bloch import BLOCH_METHODS

        with pytest.raises(NotImplementedError, match="dense"):
            Mandacaru(method=BLOCH_METHODS[0], trace=False,
                      two_body="cholesky")

    def test_the_run_cites_the_factorization(self):
        calc = Mandacaru(method="rhf", trace=False, two_body="cholesky")
        assert {"BeebeLinderberg1977", "Koch2003",
                "Folkestad2019"} <= set(calc.solver.citation_keys())
        dense = Mandacaru(method="rhf", trace=False)
        assert "Koch2003" not in dense.solver.citation_keys()


def _kohn_sham_h2(**atoms_options):
    """H2 with the cheapest Kohn-Sham calculator that still has forces."""
    atoms = h2()
    if atoms_options:
        atoms.set_pbc(atoms_options["pbc"])
    atoms.calc = Mandacaru(method="dft", xc="lda",
                           basis={"name": "PAW-LCAO", "size": "SZ"},
                           grid_spacing=0.25, trace=False)
    return atoms


class TestTheStressFollowsAse:
    """A molecule has no stress, and says so the way ASE's calculators do.

    ASE's own calculators raise
    :class:`~ase.calculators.calculator.PropertyNotImplementedError` for the
    stress of a cell-less geometry, and the ASE machinery that probes for a
    stress (trajectory writers, optimizers) catches exactly that class.
    """

    def test_a_molecule_refuses_with_the_ase_exception(self):
        from ase.calculators.calculator import PropertyNotImplementedError

        atoms = _kohn_sham_h2()
        with pytest.raises(PropertyNotImplementedError,
                           match="not periodic.*calculate_numerical_forces"):
            atoms.get_stress()
        with pytest.raises(PropertyNotImplementedError, match="not periodic"):
            atoms.calc.get_property("stress", atoms)

    def test_only_a_periodic_calculation_lists_the_stress(self):
        from mandacaru.algorithms.bloch import BLOCH_METHODS

        assert "stress" not in _kohn_sham_h2().calc.implemented_properties
        crystal = _kohn_sham_h2(pbc=True)
        assert "stress" in crystal.calc.implemented_properties
        assert "stress" not in Mandacaru(
            method="adapt-vqe", basis="HAO",
            trace=False).implemented_properties
        assert "stress" in Mandacaru(
            method=BLOCH_METHODS[0], basis="HAO",
            kpts={"size": (1, 1, 1), "gamma": True},
            trace=False).implemented_properties

    def test_the_analytic_stress_wrapper_fails_as_for_ase_s_own(self):
        """``eps_strain=None`` asks the wrapped calculator for its stress."""
        from ase.calculators.calculator import PropertyNotImplementedError
        from ase.calculators.fd import FiniteDifferenceCalculator
        from ase.calculators.lj import LennardJones

        argon = Atoms("Ar2", positions=[[0, 0, 0], [0, 0, 3.8]])
        argon.calc = FiniteDifferenceCalculator(LennardJones(), eps_disp=None,
                                                eps_strain=None)
        with pytest.raises(PropertyNotImplementedError) as theirs:
            argon.get_forces()
        atoms = _kohn_sham_h2()
        atoms.calc = FiniteDifferenceCalculator(atoms.calc, eps_disp=None,
                                                eps_strain=None)
        with pytest.raises(PropertyNotImplementedError) as ours:
            atoms.get_forces()
        assert isinstance(ours.value, type(theirs.value))

    def test_the_numerical_wrapper_reproduces_the_analytic_forces(self):
        from ase.calculators.fd import FiniteDifferenceCalculator

        atoms = _kohn_sham_h2()
        analytic = atoms.get_forces()
        atoms.calc = FiniteDifferenceCalculator(atoms.calc, eps_disp=0.01)
        numerical = atoms.get_forces()
        assert np.abs(analytic).max() > 1.0
        assert np.allclose(numerical, analytic, atol=2e-2)

    def test_a_relaxation_never_asks_for_the_stress(self, tmp_path):
        """BFGS and its trajectory probe for properties; neither needs it."""
        from ase.io import read
        from ase.optimize import BFGS

        atoms = _kohn_sham_h2()
        path = tmp_path / "h2.traj"
        BFGS(atoms, trajectory=str(path), logfile=None).run(steps=1)
        assert "stress" not in atoms.calc.results
        written = read(path, index=-1)
        assert written.get_forces() == pytest.approx(atoms.get_forces())


class TestNoncollinearMoments:
    """(N, 3) initial magnetic moments start a two-component run, which only
    molecular Kohn-Sham has; every other method takes one moment per atom."""

    @pytest.mark.parametrize("method", ["rhf", "uhf", "adapt-vqe"])
    def test_a_method_without_spinors_refuses_them(self, method):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        atoms.set_initial_magnetic_moments([[0, 0, 1], [1, 0, 0]])
        atoms.calc = Mandacaru(method=method, basis="HAO", trace=False)
        with pytest.raises(NotImplementedError,
                           match=r"\(N, 3\) initial magnetic moments.*"
                                 rf"method={method!r}"):
            atoms.get_potential_energy()
