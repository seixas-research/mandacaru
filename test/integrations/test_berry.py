"""Berry phases, curvature, Chern numbers, polarization and Hall conductivity through the calculator.

The physics is gated in Rust (``crates/mandacaru-core/tests/berry_links.rs``, ``berry_crystal.rs``,
``berry_functionals.rs``: lattice models, the exact rigid-shift gate, time reversal, Stokes); here
the public surface is checked: the views, their units and shapes, the symmetry identities seen
from Python, and every refusal.
"""
import numpy as np
import pytest
from ase import Atoms
from mandacaru import Mandacaru
from mandacaru.analysis.berry import AnomalousHall, BerryCurvature, BerryPhase, Polarization

BOHR = 0.529177210903
CELL = np.array([6.0, 8.0, 8.0]) * BOHR
#: e/Bohr^2 in C/m^2.
E_PER_BOHR2 = 1.602176634e-19 / (BOHR * 1e-10) ** 2


def _chain(**options):
    """H2 (1.4 Bohr bond) along x, centered: a centrosymmetric, time-reversal insulator."""
    atoms = Atoms("H2", positions=[[2.3 * BOHR, 4.0 * BOHR, 4.0 * BOHR],
                                   [3.7 * BOHR, 4.0 * BOHR, 4.0 * BOHR]],
                  cell=CELL, pbc=True)
    settings = dict(method="dft", xc="lda", basis={"name": "PAW-LCAO", "size": "SZ"},
                    grid_spacing=0.4 * BOHR, kpts={"size": (1, 1, 1), "gamma": True},
                    smearing={"method": "fermi-dirac", "width": 0.01},
                    parallelization={"threads": 2}, trace=False)
    settings.update(options)
    atoms.calc = Mandacaru(**settings)
    atoms.get_potential_energy()
    return atoms


@pytest.fixture(scope="module")
def chain():
    return _chain()


MESH = (3, 2, 2)


def test_the_polarization_view_has_units_quanta_and_the_inversion_lattice(chain):
    p = chain.calc.get_polarization(kpts=MESH)
    assert isinstance(p, Polarization)
    assert p.strings == MESH and p.quantum == 1 and p.degeneracy == 2.0
    assert p.vector.shape == (3,) and p.quanta.shape == (3, 3)
    assert p.phases.shape == (1, 3)
    # A quantum is e a_j / Omega (g = gcd(2, 1, 1) = 1), in C/m^2.
    volume = float(np.prod(CELL / BOHR))
    np.testing.assert_allclose(np.diag(p.quanta), CELL / BOHR / volume * E_PER_BOHR2, rtol=1e-9)
    # Centrosymmetric: every coordinate on 0 or 1/2.
    assert all(min(abs(x), abs(abs(x) - 0.5)) < 1e-6 for x in p.fractional), p.fractional
    np.testing.assert_allclose(p.vector, p.fractional @ p.quanta, atol=1e-12)


def test_the_berry_phase_view_holds_phases_centers_and_the_parallel_transport_connection(chain):
    phase = chain.calc.get_berry_phase(0, kpts=MESH)
    assert isinstance(phase, BerryPhase)
    assert phase.transverse_axes == (1, 2) and phase.strings == MESH
    assert phase.phases.shape == (1, 2, 2)
    assert phase.bands == ((0, 1),)
    centers = phase.wannier_charge_centers[0]
    assert centers.shape == (2, 2, 1)
    assert np.all((centers >= 0.0) & (centers < 1.0))
    # One occupied band: its Wilson center is the string phase over 2 pi (mod 1), and the
    # parallel-transport connection is that center (on (-1/2, 1/2]) times |a_x| (Angstrom).
    residual = centers[..., 0] - phase.phases[0] / (2 * np.pi)
    assert np.all(np.abs(residual - np.round(residual)) < 1e-9)
    shifted = np.where(centers > 0.5, centers - 1.0, centers)
    np.testing.assert_allclose(phase.connection[0], shifted * CELL[0], atol=1e-9)
    assert phase.minimum_singular_value > 0.1


def test_the_curvature_of_a_time_reversal_crystal_is_odd_and_its_chern_number_zero(chain):
    plane = chain.calc.get_berry_curvature(2, kpts=(3, 3, 1))
    assert isinstance(plane, BerryCurvature)
    assert plane.axes == (0, 1) and plane.size == (3, 3)
    assert plane.flux.shape == (1, 3, 3) and plane.curvature.shape == (1, 3, 3)
    assert plane.chern_numbers == (0,) and plane.chern_number == 0
    flux = plane.flux[0]
    np.testing.assert_allclose(flux, -flux[::-1, ::-1], atol=1e-9)
    assert chain.calc.get_chern_number(2, kpts=(3, 3, 1)) == 0
    # A single isolated band: the band-resolved curvature.
    assert chain.calc.get_berry_curvature(2, kpts=(3, 3, 1), bands=0).bands == ((0, 1),)


def test_the_hall_conductivity_of_a_time_reversal_insulator_vanishes(chain):
    sigma = chain.calc.get_anomalous_hall_conductivity(kpts=(2, 2, 2))
    assert sigma.shape == (3, 3)
    assert np.all(sigma == 0.0)


def test_spin_polarized_equals_unpolarized_channel_by_channel(chain):
    polarized = _chain(spin=True)
    p = chain.calc.get_polarization(kpts=MESH)
    q = polarized.calc.get_polarization(kpts=MESH)
    assert q.degeneracy == 1.0 and q.phases.shape == (2, 3)
    np.testing.assert_allclose(q.vector, p.vector, atol=1e-6)


def test_ts_dispersion_leaves_the_states_and_so_the_berry_quantities(chain):
    dispersed = _chain(vdw="ts")
    np.testing.assert_allclose(dispersed.calc.get_polarization(kpts=MESH).raw,
                               chain.calc.get_polarization(kpts=MESH).raw, atol=1e-9)


@pytest.mark.parametrize("call", [
    lambda c: c.get_berry_phase(3, kpts=MESH),
    lambda c: c.get_berry_phase(True, kpts=MESH),
    lambda c: c.get_berry_phase(0, kpts=(1, 2, 2)),
    lambda c: c.get_berry_phase(0, kpts=(3, 2)),
    lambda c: c.get_berry_phase(0, kpts={"gamma": True}),
    lambda c: c.get_berry_phase(0, kpts=MESH, bands=(1, 1)),
    lambda c: c.get_berry_phase(0, kpts=MESH, bands=(0, 9)),
    lambda c: c.get_berry_phase(0, kpts=MESH, bands=-1),
    lambda c: c.get_berry_curvature(2, kpts=MESH, offset=1.0),
    lambda c: c.get_berry_curvature(2, kpts=(1, 3, 1)),
    lambda c: c.get_polarization(kpts=(2, 2, 1)),
])
def test_bad_requests_raise_value_error(chain, call):
    with pytest.raises(ValueError):
        call(chain.calc)


def test_the_default_mesh_is_the_calculation_kpts(chain):
    # The fixture's kpts is (1, 1, 1): a string of one point is refused, naming the fix.
    with pytest.raises(ValueError, match="at least 2"):
        chain.calc.get_polarization()


def test_a_molecule_has_no_brillouin_zone():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    atoms.calc = Mandacaru(method="dft", xc="lda", basis={"name": "PAW-LCAO", "size": "SZ"},
                           grid_spacing=0.25, trace=False)
    with pytest.raises(ValueError, match="Brillouin zone"):
        atoms.calc.get_polarization(kpts=(2, 2, 2))


def test_a_quantum_method_on_a_crystal_is_not_implemented():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]], cell=[3.0, 4.0, 4.0], pbc=True)
    atoms.calc = Mandacaru(method="adapt-vqe", crystal_orbitals="real-space", trace=False)
    with pytest.raises(NotImplementedError, match="Bloch bands"):
        atoms.calc.get_berry_phase(0, kpts=(2, 1, 1))


def _hydrogen_chain(**options):
    """One H per 3 Bohr along x: a half-filled band, a metal."""
    atoms = Atoms("H", positions=[[1.5 * BOHR, 3.5 * BOHR, 3.5 * BOHR]],
                  cell=np.array([3.0, 7.0, 7.0]) * BOHR, pbc=True)
    settings = dict(method="dft", xc="lda", basis={"name": "PAW-LCAO", "size": "SZ"},
                    grid_spacing=0.35 * BOHR, kpts={"size": (6, 1, 1), "gamma": True},
                    smearing={"method": "fermi-dirac", "width": 0.27},
                    parallelization={"threads": 2}, trace=False)
    settings.update(options)
    atoms.calc = Mandacaru(**settings)
    atoms.get_potential_energy()
    return atoms


@pytest.fixture(scope="module")
def metal():
    return _hydrogen_chain()


@pytest.mark.slow  # 46 s: one 8^3 metal mesh (the same wiring is gated in berry_metals.rs)
def test_a_metal_reports_its_hall_conductivity_with_the_stride_two_estimate(metal):
    # The chain is centrosymmetric and real, so its zero is trivial: this checks the plumbing
    # (the non-trivial zeros are in crates/mandacaru-core/tests/berry_metals.rs).
    import warnings as python_warnings

    with python_warnings.catch_warnings():
        python_warnings.simplefilter("error")  # an 8^3 mesh of a one-band chain warns nothing
        hall = metal.calc.get_anomalous_hall(kpts=(8, 8, 8))
    assert isinstance(hall, AnomalousHall)
    assert hall.metallic and hall.chern_numbers == ((), (), ())
    assert hall.sigma.shape == (3, 3) and hall.sigma_coarse.shape == (3, 3)
    assert np.all(np.abs(hall.sigma) < 1e-4)  # S/cm: zero to rounding
    assert len(hall.layers) == 3 and len(hall.layers[0]) == 1 and hall.cap >= 1
    assert hall.plaquettes == 3 * 8 * 64 and hall.merged == 0
    assert abs(hall.fermi - metal.calc.get_fermi_level()) < 1.0  # eV


@pytest.mark.parametrize("mesh", [(3, 2, 2), (4, 4, 4)])
def test_a_metal_below_eight_points_warns_that_its_convergence_is_unknown(metal, mesh):
    with pytest.warns(UserWarning, match="at least 8"):
        hall = metal.calc.get_anomalous_hall(kpts=mesh)
    assert hall.sigma_coarse is None
    with pytest.warns(UserWarning, match="at least 8"):
        sigma = metal.calc.get_anomalous_hall_conductivity(kpts=mesh)
    np.testing.assert_allclose(sigma, hall.sigma, atol=1e-12)


def test_a_metal_still_has_no_polarization(metal):
    with pytest.raises(ValueError, match="insulator"):
        metal.calc.get_polarization(kpts=(4, 2, 2))


def test_an_insulator_reports_its_chern_numbers_in_the_hall_view(chain):
    hall = chain.calc.get_anomalous_hall(kpts=(2, 2, 2))
    assert not hall.metallic and hall.sigma_coarse is None
    assert hall.chern_numbers == ((0,), (0,), (0,))
    assert np.all(hall.numbers == 0.0)
