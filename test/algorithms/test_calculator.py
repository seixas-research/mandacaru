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

    def test_vqe_refuses_a_log_it_would_ignore(self):
        # `txt=` is a driver-wide option now (every driver has to know where it
        # reports), so the refusal is about the *capability*, not the keyword.
        with pytest.raises(NotImplementedError, match="does not write 'txt'"):
            Mandacaru(method="vqe", basis="HAO", txt="x.txt")

    def test_a_subspace_method_refuses_checkpoints(self):
        with pytest.raises(NotImplementedError, match="does not write"):
            Mandacaru(method="subspace-adapt-vqe", basis="HAO",
                      checkpoint="x.json")

    def test_an_unknown_option_is_not_kept_as_an_ase_parameter(self):
        with pytest.raises(TypeError, match="does not take 'nonsense'"):
            Mandacaru(method="adapt-vqe", basis="HAO", nonsense=1)

    def test_vqe_keeps_a_checkpoint_it_does_support(self, tmp_path):
        Mandacaru(method="vqe", basis="HAO", checkpoint=str(tmp_path / "c.json"))

    def test_a_method_without_a_log_keeps_its_trace(self, capsys, tmp_path):
        """The trap: no file *and* no trace is a run that reports nothing."""
        atoms = h2()
        atoms.calc = Mandacaru(method="vqe", basis="HAO", h=0.45)
        atoms.get_potential_energy()
        assert "VQE" in capsys.readouterr().out


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
