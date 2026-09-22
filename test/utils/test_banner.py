# -*- coding: utf-8 -*-
# file: test_banner.py

"""The start-up banner writes provenance to standard output (not via print)."""

import numpy as np
import pytest

from mandacaru.algorithms import Mandacaru
from mandacaru.circuits import UCCSD
from mandacaru.core import MolecularIntegrals, minimal_hao_basis
from mandacaru.integrals import Grid
from mandacaru.utils import banner


@pytest.fixture
def h2_hamiltonian():
    nuclei = [(1.0, np.array([0.0, 0.0, -0.37])),
              (1.0, np.array([0.0, 0.0, 0.37]))]
    grid = Grid(center=[0, 0, 0], box_size=5.0, h=0.35)
    return MolecularIntegrals(
        nuclei, minimal_hao_basis(nuclei), grid
    ).molecular_hamiltonian(mo_basis=True, n_electrons=2)


class TestBanner:
    def test_show_writes_to_stdout(self, capsys):
        banner.show()
        out = capsys.readouterr().out
        assert "version:" in out
        assert "System:" in out
        assert "Dependencies:" in out
        assert "ase version:" in out

    def test_uses_write_not_builtin_print(self):
        # The module must not shadow print with ase.parallel.parprint anymore.
        import inspect
        src = inspect.getsource(banner)
        assert "parprint" not in src
        assert "_write" in src


@pytest.fixture(autouse=True)
def fresh_banner():
    """The banner prints once per process; these tests each need a fresh one."""
    from mandacaru.algorithms import base
    base._BANNER_SHOWN = False
    yield
    base._BANNER_SHOWN = False


class TestBannerInRun:
    def test_verbose_run_shows_banner_before_header(self, h2_hamiltonian, capsys):
        Mandacaru(method="vqe", hamiltonian=h2_hamiltonian,
                  ansatz=UCCSD(2, (1, 1)), trace=True).run()
        out = capsys.readouterr().out
        assert "Dependencies:" in out               # the banner ran
        assert out.index("Dependencies:") < out.index("Qubit Hamiltonian")

    def test_banner_precedes_output_txt(self, h2_hamiltonian, tmp_path, capsys):
        out_file = str(tmp_path / "output.txt")
        Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian, pool="ceo",
                  num_particles=(1, 1), n_spatial_orbitals=2, profile=False,
                  trace=True, max_iterations=2, gradient_tolerance=1e-6,
                  txt=out_file).run()
        assert "Dependencies:" in capsys.readouterr().out

    def test_silent_when_not_verbose(self, h2_hamiltonian, capsys):
        Mandacaru(method="vqe", hamiltonian=h2_hamiltonian,
                  ansatz=UCCSD(2, (1, 1)), trace=False).run()
        assert capsys.readouterr().out == ""
