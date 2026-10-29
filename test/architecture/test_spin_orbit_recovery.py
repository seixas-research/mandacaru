"""SOC retains its angular spectrum, radial mixing and projection derivatives."""
from types import SimpleNamespace

import numpy as np
import pytest

from mandacaru import _native
from mandacaru import Mandacaru
from ase.build import bulk
from mandacaru.core.spin_orbit import ls_matrix, spin_orbit_one_body, spin_orbit_pair


@pytest.mark.parametrize("real", [False, True])
@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_native_angular_operator_has_exact_total_angular_momentum_levels(l, real):
    values = np.linalg.eigvalsh(ls_matrix(l, real=real))
    expected = np.sort([l / 2.] * (2 * l + 2) + [-(l + 1) / 2.] * (2 * l))
    assert np.allclose(values, expected, atol=2e-14)


@pytest.mark.parametrize("real", [False, True])
def test_native_soc_radial_spectrum_and_projection_response(real):
    projectors = [SimpleNamespace(atom_index=0, l=1, m=m, index=i, real=real)
                  for m in range(-1, 2) for i in range(2)]
    radial = np.array([[.2, .07], [.07, -.1]], dtype=complex)
    blocks = {(0, 1): radial}
    with_context = _native.Context(threads=1)
    projections = np.eye(6, dtype=complex)
    matrix = spin_orbit_one_body(projections, projectors, blocks, context=with_context)
    angular_levels = np.array([-1., -1., .5, .5, .5, .5])
    expected = np.sort(np.multiply.outer(angular_levels, np.linalg.eigvalsh(radial)).ravel())
    assert np.allclose(np.linalg.eigvalsh(matrix), expected, atol=2e-14)
    rng = np.random.default_rng(7)
    projections = rng.normal(size=(4, 6)) + 1j * rng.normal(size=(4, 6))
    direction = rng.normal(size=(4, 6)) + 1j * rng.normal(size=(4, 6))
    h = 1e-5
    plus = spin_orbit_one_body(projections + h * direction, projectors, blocks, context=with_context)
    minus = spin_orbit_one_body(projections - h * direction, projectors, blocks, context=with_context)
    derivative = spin_orbit_pair(direction, projections, projectors, blocks, context=with_context)
    derivative += spin_orbit_pair(projections, direction, projectors, blocks, context=with_context)
    assert np.allclose((plus - minus) / (2 * h), derivative, atol=2e-10)
    assert with_context.memory_used_bytes == 0


def test_native_soc_rejects_invalid_angular_degree_before_allocation():
    context = _native.Context(threads=1)
    with pytest.raises(ValueError, match="degree"):
        context.spin_orbit_angular(17)
    assert context.memory_used_bytes == 0


def test_periodic_spinor_ase_bands_and_dos_keep_unit_state_degeneracy():
    atoms = bulk("Si", "diamond", a=5.43)
    with Mandacaru(method="dft", xc="pbe", paw_dataset="pbe-fr", atoms=atoms,
                   basis="PAW-LCAO", grid_spacing=.8,
                   resources={"threads": 1}) as calc:
        atoms.get_potential_energy()
        assert calc.result.success
        assert calc.periodic_problem.spinor
        assert calc.periodic_problem.time_reversal
        assert calc.get_number_of_bands() == 16
        assert calc.result.scf.spin_orbit_energy < 0.
        levels = calc.get_eigenvalues()
        assert np.max(np.abs(levels[::2] - levels[1::2])) < 1e-8
        assert np.sum(calc.get_occupation_numbers()) == pytest.approx(8., abs=1e-9)
        assert calc.band_structure(path="GX", npoints=5).energies.shape == (1, 5, 16)
        axis, values = calc.dos(width=.2, npoints=3001)
        assert np.trapezoid(values, axis) == pytest.approx(16., abs=1e-7)
        axis, partial = calc.pdos(width=.2, npoints=3001, energies=axis)
        assert np.allclose(sum(partial.values()), values, atol=1e-9)
        assert "density_matrices" not in calc.result.snapshot()
        explicit = calc.result.scf.snapshot()
        assert explicit["density_matrices"].shape == (1, 1, 16, 16)


def test_nonmagnetic_time_reversal_reduction_matches_full_vector_spinor_scf():
    atoms = bulk("Si", "diamond", a=5.43)
    values = []
    for unrestricted in (False, True):
        with Mandacaru(method="dft", xc="pbe", paw_dataset="pbe-fr", atoms=atoms,
                       spin=unrestricted, basis="PAW-LCAO", grid_spacing=.8,
                       kpts={"size": (3, 1, 1), "gamma": True},
                       resources={"threads": 1}) as calc:
            energy = atoms.get_potential_energy()
            assert calc.result.success
            assert calc.periodic_problem.time_reversal == (not unrestricted)
            values.append((energy, calc.result.snapshot(), calc.result.scf.spin_orbit_energy))
    reduced, full = values
    assert len(reduced[1]["weights"]) < len(full[1]["weights"])
    assert abs(reduced[0] - full[0]) < 1e-5
    assert np.allclose(reduced[1]["density"][0], full[1]["density"][0], atol=1e-8)
    assert abs(reduced[2] - full[2]) < 1e-9
