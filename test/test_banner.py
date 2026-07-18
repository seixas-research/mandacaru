# -*- coding: utf-8 -*-
# file: test_banner.py

"""The start-up banner writes provenance to standard output (not via print)."""

import numpy as np
import pytest
from ase import Atoms

from carcara.algorithms import ADAPTVQE, VQE
from carcara.circuits import UCCSD
from carcara.core import MolecularIntegrals, minimal_fao_basis
from carcara.integrals import Grid
from carcara.utils import banner


@pytest.fixture
def h2_hamiltonian():
    nuclei = [(1.0, np.array([0.0, 0.0, -0.37])),
              (1.0, np.array([0.0, 0.0, 0.37]))]
    grid = Grid(center=[0, 0, 0], box_size=5.0, h=0.35)
    return MolecularIntegrals(
        nuclei, minimal_fao_basis(nuclei), grid
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


class TestBannerInRun:
    def test_verbose_run_shows_banner_before_header(self, h2_hamiltonian, capsys):
        VQE(h2_hamiltonian, UCCSD(2, (1, 1)), verbose=True).run()
        out = capsys.readouterr().out
        assert "Dependencies:" in out               # the banner ran
        assert out.index("Dependencies:") < out.index("Qubit Hamiltonian")

    def test_banner_precedes_output_txt(self, h2_hamiltonian, tmp_path, capsys):
        out_file = str(tmp_path / "output.txt")
        ADAPTVQE(h2_hamiltonian, "ceo", num_particles=(1, 1),
                 n_spatial_orbitals=2, profile=False, verbose=True,
                 max_iterations=2, gradient_tolerance=1e-6,
                 output=out_file).run()
        assert "Dependencies:" in capsys.readouterr().out

    def test_silent_when_not_verbose(self, h2_hamiltonian, capsys):
        VQE(h2_hamiltonian, UCCSD(2, (1, 1)), verbose=False).run()
        assert capsys.readouterr().out == ""
