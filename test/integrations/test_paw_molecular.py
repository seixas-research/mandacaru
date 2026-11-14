"""The molecular PAW-LCAO adapter (``integrations/paw_molecular.py``)."""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru



def _oxygen(cell):
    return Atoms("O2", positions=[[0, 0, 0], [0, 0, 1.21]], cell=cell)


def test_a_molecule_without_a_cell_raises_as_on_every_molecular_route():
    """A zero cell used to become a 2 x 2 x 2 box on the PAW route, and the SCF
    converged there to an energy thousands of eV off (O2: -3902 eV against
    -860 eV in a real box). The all-electron route raised; both raise now."""
    for basis in ("HAO", {"name": "PAW-LCAO", "size": "SZ"}):
        atoms = _oxygen(None)
        atoms.calc = Mandacaru(method="rhf", basis=basis, grid_spacing=0.3)
        with pytest.raises(ValueError, match="nondegenerate cell or an explicit grid"):
            atoms.get_potential_energy()


def test_the_box_is_the_shared_molecular_box():
    """The PAW route samples the box every molecular route uses: centered on the
    mean atomic position, spanning the cell at the requested spacing."""
    atoms = _oxygen(np.diag([5.0, 5.0, 6.2]))
    atoms.center()
    atoms.calc = Mandacaru(method="rhf", basis={"name": "PAW-LCAO", "size": "SZ"},
                           grid_spacing=0.3)
    atoms.get_potential_energy()
    assert tuple(atoms.calc.integrals.grid.shape) == (18, 18, 22)


def test_a_multi_atom_molecule_gives_the_same_bits_on_every_thread_count():
    """The one-center local and electrostatic matrices sum one partial per atom in atom order on
    every thread count: the SCF energy and the one-body matrix of H2O did differ by 4.4e-11 eV and
    1.3e-13 Ha between one and four threads when the serial route used a single accumulator."""
    runs = []
    for threads in (1, 2, 4):
        atoms = Atoms("OH2", positions=[[0, 0, 0], [0.76, 0, 0.59], [-0.76, 0, 0.59]])
        atoms.center(vacuum=3.0)
        atoms.calc = Mandacaru(method="rhf", basis={"name": "PAW-LCAO", "size": "SZ"},
                               grid_spacing=0.4, parallelization={"threads": threads})
        runs.append((atoms.get_potential_energy(), atoms.calc.integrals.snapshot()))
    for energy, snapshot in runs[1:]:
        assert energy == runs[0][0]
        for name in ("overlap", "kinetic", "potential", "one_body"):
            assert np.array_equal(snapshot[name], runs[0][1][name]), name


@pytest.mark.parametrize("active_space", [None, {"orbitals": 2}, {"orbitals": 2, "method": "mp2"}])
def test_a_kohn_sham_reference_over_a_heavy_core_has_no_hamiltonian_without_core_valence_exchange(
        active_space):
    """Kohn-Sham DFT of NaH runs (its energy never reads a core-valence exchange), but the
    second-quantized Hamiltonian over its frozen core does: the dense model, an active space and
    the MP2 natural orbitals that select one are refused like Hartree-Fock (owner Q20)."""
    atoms = Atoms("NaH", positions=[[0, 0, 0], [0, 0, 1.89]], cell=[7, 7, 7])
    atoms.center()
    atoms.calc = Mandacaru(method="dft", xc="lda", basis={"name": "PAW-LCAO", "size": "SZ"},
                           grid_spacing=0.3, active_space=active_space)
    assert np.isfinite(atoms.get_potential_energy())
    with pytest.raises(ValueError, match="frozen-core exchange"):
        atoms.calc.result.hamiltonian


def test_a_hybrid_on_a_heavy_core_is_refused_by_name():
    """HSE06 takes a fraction of exact exchange, which includes the valence's exchange with the
    frozen core: on NaH (a sodium core beyond a closed 1s shell) it raises, naming the
    functional, while the semilocal functional of the same molecule runs."""
    def nah(xc):
        atoms = Atoms("NaH", positions=[[0, 0, 0], [0, 0, 1.89]], cell=[7, 7, 7])
        atoms.center()
        atoms.calc = Mandacaru(method="dft", xc=xc, basis={"name": "PAW-LCAO", "size": "SZ"},
                               grid_spacing=0.3)
        return atoms

    assert np.isfinite(nah("pbe").get_potential_energy())
    with pytest.raises(ValueError, match="hse06 takes 0.25 of exact exchange.*frozen-core exchange"):
        nah("hse06").get_potential_energy()
