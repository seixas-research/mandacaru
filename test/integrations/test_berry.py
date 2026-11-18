"""Berry phases, curvature, Chern numbers, polarization and Hall conductivity through the calculator.

The physics is gated in Rust (``crates/mandacaru-core/tests/berry_links.rs``, ``berry_crystal.rs``,
``berry_functionals.rs``: lattice models, the exact rigid-shift gate, time reversal, Stokes); here
the public surface is checked: the views, their units and shapes, the symmetry identities seen
from Python, and every refusal.
"""
import os
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from mandacaru import Mandacaru
from mandacaru.analysis.berry import (AnomalousHall, BerryCurvature, BerryPhase, BornCharges,
                                      PiezoelectricTensor, Polarization, WannierChargeCenterFlow,
                                      Z2Invariant)

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
MESH_3 = (3, 3, 3)


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


def test_a_spin_degenerate_crystal_has_the_z2_zero_of_its_doubled_manifold_by_flow_and_by_parity(chain):
    # Without spin-orbit coupling every Wannier center appears twice (once per spin): every
    # crossing is counted twice and every index is zero. The H2 chain is also centrosymmetric.
    flow = chain.calc.get_z2_invariant(kpts=MESH_3)
    assert isinstance(flow, Z2Invariant) and flow.method == "flow"
    assert flow.indices == (0, 0, 0, 0) and flow.strong == 0 and flow.weak == (0, 0, 0)
    assert len(flow.flows) == 6 and flow.spin_degenerate
    assert all(isinstance(f, WannierChargeCenterFlow) and f.half and f.z2 == 0 for f in flow.flows)
    assert all(f.k_perp[0] == 0.0 and f.k_perp[-1] == 0.5 for f in flow.flows)
    assert set(flow.planes) == {(c, h) for c in range(3) for h in range(2)}
    parity = chain.calc.get_z2_invariant(kpts=MESH_3, method="parity")
    assert parity.method == "parity" and parity.indices == (0, 0, 0, 0)
    assert parity.planes == flow.planes and parity.spin_degenerate
    assert len(parity.parity) == 8 and parity.flows == ()
    # H2 is centered in its cell: the inversion center is the cell center (modulo half vectors).
    center = np.asarray(parity.inversion_center)
    assert np.all(np.abs(center * 2 - np.round(center * 2)) < 1e-6)
    for point in parity.parity:
        assert point["residual"] < 1e-6 and point["occupied"] == 1
        assert point["delta"] in (-1, 1) and point["parities"].size >= 1


def test_a_plane_gives_its_own_z2_index_and_a_generic_plane_its_chern_number(chain):
    plane = chain.calc.get_z2_invariant(kpts=MESH_3, plane=2, offset=0.5)
    assert plane.indices == (0,) and plane.strong is None and plane.weak is None
    assert plane.planes == {(2, 1): 0} and len(plane.flows) == 1
    parity = chain.calc.get_z2_invariant(kpts=MESH_3, plane=1, offset=0.0, method="parity")
    assert parity.indices == (0,) and parity.planes == {(1, 0): 0}
    # A plane off the time-reversal-invariant lines is flown over the whole period: its winding
    # is the Chern number, zero here.
    flow = chain.calc.get_wannier_charge_center_flow(2, 0, kpts=MESH_3, offset=0.25)
    assert not flow.half and flow.winding == 0 and flow.z2 is None
    assert flow.k_perp[0] == 0.0 and flow.k_perp[-1] == 1.0
    np.testing.assert_allclose(flow.wannier_charge_centers[0], flow.wannier_charge_centers[-1])
    assert flow.transverse == 1 and flow.direction == 0 and flow.normal == 2
    assert chain.calc.get_chern_number(2, kpts=MESH_3, offset=0.25) == flow.winding
    half = chain.calc.get_wannier_charge_center_flow(2, 1, kpts=MESH_3, offset=0.0)
    assert half.half and half.z2 == 0 and half.winding is None and half.crossings % 2 == 0
    assert half.gap_midpoints.shape == half.k_perp.shape == half.gaps.shape
    # One center per spatial band (the Z2 count doubles it: spin up and down).
    assert half.wannier_charge_centers.shape == (half.k_perp.size, 1)


@pytest.mark.parametrize("call", [
    lambda c: c.get_z2_invariant(kpts=MESH_3, method="invariant"),
    lambda c: c.get_z2_invariant(kpts=MESH_3, plane=2, offset=0.3),
    lambda c: c.get_z2_invariant(kpts=MESH_3, plane=2, direction=2),
    lambda c: c.get_z2_invariant(kpts=MESH_3, plane=2, method="parity", direction=0),
    lambda c: c.get_z2_invariant(kpts=MESH_3, offset=0.5),
    lambda c: c.get_z2_invariant(kpts=(2, 3, 3)),
    lambda c: c.get_z2_invariant(kpts=(3, 1, 3), plane=2),
    lambda c: c.get_wannier_charge_center_flow(2, 2, kpts=MESH_3),
    lambda c: c.get_wannier_charge_center_flow(3, 0, kpts=MESH_3),
    lambda c: c.get_wannier_charge_center_flow(2, 0, kpts=MESH_3, offset=1.0),
])
def test_bad_z2_requests_raise_value_error(chain, call):
    with pytest.raises(ValueError):
        call(chain.calc)


def test_a_spin_polarized_crystal_has_no_z2_index_and_a_metal_none_either(metal):
    polarized = _chain(spin=True)
    for method in ("flow", "parity"):
        with pytest.raises(NotImplementedError, match="spin-polarized"):
            polarized.calc.get_z2_invariant(kpts=MESH_3, method=method)
    with pytest.raises(NotImplementedError, match="spin-polarized"):
        polarized.calc.get_wannier_charge_center_flow(2, 0, kpts=MESH_3)
    with pytest.raises(ValueError, match="insulator"):
        metal.calc.get_z2_invariant(kpts=MESH_3)
    with pytest.raises(ValueError, match="insulator"):
        metal.calc.get_z2_invariant(kpts=MESH_3, method="parity")


# ---------------------------------------------------------------------------------------------
# Bulk Bi2Se3 (PAW regeneration gates G12 and G13: the spin-orbit topology of the PAW-LCAO datasets).
#
# The owner's rhombohedral primitive cell (Angstrom, Cartesian; the PBE-expanded structure). Se(0) at the
# origin is an inversion center; the Se(1) and Bi layers are placed about it.
BI2SE3_CELL = [[2.0956832649999995, 1.2099432971839437, 9.9762290666666651],
               [-2.0956832649999999, 1.2099432971839437, 9.9762290666666651],
               [-0.0000000000000008, -2.4198865943678873, 9.9762290666666651]]
BI2SE3_POSITIONS = [[-0.0000000000000005, 0.0000000000000004, 11.9246364845319981],
                    [-0.0000000000000008, 0.0000000000000003, 18.0040507154679972],
                    [-0.0000000000000000, 0.0000000000000003, 0.0000000000000006],
                    [-0.0000000000000003, 0.0000000000000003, 6.4560667593479977],
                    [-0.0000000000000011, 0.0000000000000003, 23.4726204406520012]]


def _rhombohedral_bi2se3(a, c, u_bi, u_se):
    """The 5-atom rhombohedral primitive cell of R-3m Bi2Se3 (hexagonal ``a``, ``c``; Bi at +-u_bi c and
    Se at 0 and +-u_se c along the axis), in the atom order of the owner's cell (Bi, Bi, Se, Se, Se)."""
    r3, h = np.sqrt(3.0), c / 3.0
    cell = [[a / 2, a / (2 * r3), h], [-a / 2, a / (2 * r3), h], [0.0, -a / r3, h]]
    positions = [[0.0, 0.0, u_bi * c], [0.0, 0.0, (1 - u_bi) * c], [0.0, 0.0, 0.0],
                 [0.0, 0.0, u_se * c], [0.0, 0.0, (1 - u_se) * c]]
    return cell, positions


#: The experimental lattice (a, c in Angstrom, u(Bi), u(Se)) of the companion cell, as used for the
#: published PBE+SOC bulk gap (Zhang et al. 2009). Nakajima 1963 (the structure type) has a = 4.143, c = 28.636,
#: u(Bi) = 0.4008, u(Se) = 0.2117 (COD 9011965): 0.1 % and 0.006 away; the exact source of the set used here
#: was not found, and the independent plane-wave reference (-0.53 eV) was computed with it.
BI2SE3_EXPERIMENTAL = (4.138, 28.64, 0.399, 0.206)
#: The working point, fixed by the regeneration diagnosis (``dev_plan/DECISIONS.md``, 2026-11-13, "G12
#: Bi2Se3 diagnosis"): DZP at the minimum (SZ inverts only because its scalar gap is 0.35 eV too small),
#: 0.18 Angstrom, a 3x3x3 SCF mesh, and a narrow Fermi-Dirac smearing (eV: the occupied manifold is
#: accepted only when every occupation is 0 or 1). ``MANDACARU_G12_LIBRARY`` names a folder holding
#: ``pbe-fr/`` and ``pbe-sr/`` (how a regenerated set is gated), ``MANDACARU_G12_BASIS`` another size.
BI2SE3_GRID = 0.18
BI2SE3_BASIS = os.environ.get("MANDACARU_G12_BASIS", "DZP")
BI2SE3_SCF = (3, 3, 3)
BI2SE3_STRINGS = (6, 6, 6)
BI2SE3_SMEARING = {"method": "fermi-dirac", "width": 2e-4}


def _bi2se3_dataset(name):
    root = os.environ.get("MANDACARU_G12_LIBRARY")
    return name if not root else str(Path(root) / name)


def _bi2se3(cell, positions, dataset, scratch):
    atoms = Atoms("Bi2Se3", positions=positions, cell=cell, pbc=True)
    atoms.calc = Mandacaru(
        method="dft", xc="pbe", paw_dataset=_bi2se3_dataset(dataset),
        basis={"name": "PAW-LCAO", "size": BI2SE3_BASIS},
        kpts={"size": BI2SE3_SCF, "gamma": True}, grid_spacing=BI2SE3_GRID,
        smearing=BI2SE3_SMEARING, parallelization={"threads": 2}, trace=False,
        resources={"memory_limit_bytes": 2 * 2**30, "scratch_dir": str(scratch)},
        periodic_scf_options={"max_iterations": 120})
    atoms.get_potential_energy()
    return atoms


def _signed_gamma_gap(parity):
    """E(even frontier level) - E(odd frontier level) at Gamma, in eV: the highest occupied and the lowest
    empty level (Kramers pairs with spin-orbit coupling) are one even and one odd. Positive when the
    occupied one is odd (the normal order, a plain gap), negative when the occupied one is even (inverted).
    Gamma is looked up in the view (``get_eigenvalues(kpt=0)`` is not Gamma on a Gamma-centered 3x3x3 mesh)."""
    gamma = next(p for p in parity.parity if p["index"] == (0, 0, 0))
    n = gamma["occupied"]
    below, above = gamma["parities"][n - 1], gamma["parities"][n]
    assert below == -above, (below, above)
    return float((gamma["energies"][n] - gamma["energies"][n - 1]) * (-below))


def _assert_strong(parity, flow, signed_gap):
    assert parity.method == "parity" and parity.indices == (1, 0, 0, 0)
    assert not parity.spin_degenerate
    # The inversion center is Se(0), the origin: delta = -1 at Gamma and +1 at the other seven points.
    assert np.all(np.abs(np.asarray(parity.inversion_center)
                         - np.round(parity.inversion_center)) < 1e-6)
    deltas = {p["index"]: p["delta"] for p in parity.parity}
    assert deltas.pop((0, 0, 0)) == -1 and set(deltas.values()) == {1}
    assert all(p["residual"] < 1e-6 and p["occupied"] == 78 for p in parity.parity)
    # 78 occupied states in 39 Kramers pairs; the inversion has exchanged one pair across the gap.
    assert next(p for p in parity.parity if p["index"] == (0, 0, 0))["minus"] == 34
    assert signed_gap < 0.0, signed_gap
    if flow is not None:
        assert flow.method == "flow" and flow.indices == (1, 0, 0, 0)
        assert flow.planes == parity.planes
        assert len(flow.flows) == 6 and not flow.spin_degenerate
        assert all(f.half and f.z2 is not None and f.winding is None for f in flow.flows)


def _bulk_bi2se3_with_spin_orbit_coupling(cell, positions, scratch):
    atoms = _bi2se3(cell, positions, "pbe-fr", scratch)
    parity = atoms.calc.get_z2_invariant(kpts=BI2SE3_STRINGS, method="parity")
    gap = _signed_gamma_gap(parity)
    # Parity first: it fails cheaply; the six planes of the flow are most of the cost.
    _assert_strong(parity, None, gap)
    flow = atoms.calc.get_z2_invariant(kpts=BI2SE3_STRINGS)
    _assert_strong(parity, flow, gap)
    return gap


@pytest.mark.slow
def test_bulk_bi2se3_at_the_experimental_lattice_with_spin_orbit_coupling_is_inverted(tmp_path):
    """Z2 = (1; 000) and an inverted signed Gamma gap on the experimental lattice, where the independent
    reference inverts by 0.53 eV and the regenerated set gives -0.365 eV: the robust gate of the
    spin-orbit coupling of the ``pbe-fr`` datasets. Parity and the flow of the six planes agree plane by
    plane. Slow by design: 8286 s (2.3 h; SCF 3^3 DZP, parity, six planes of 66 or 24 frames),
    peak RSS 1.85 GB; measured 2026-11-13, 2 threads, 8-core Mac. Over the informational 180 s limit."""
    cell, positions = _rhombohedral_bi2se3(*BI2SE3_EXPERIMENTAL)
    _bulk_bi2se3_with_spin_orbit_coupling(cell, positions, tmp_path)


@pytest.mark.slow
def test_bulk_bi2se3_at_the_owner_cell_with_spin_orbit_coupling_is_inverted(tmp_path):
    """The owner's required quantitative gate: Z2 = (1; 000) with the signed Gamma gap negative on the
    PBE-expanded cell, by parity and by the flow of the six planes. With calibrated pbe-fr Bi 6p
    spin-orbit coupling (G13: +0.2% vs Dirac), parity gives Z2 = (1, 0, 0, 0), gamma_minus = 34, and
    signed_gamma_gap = -0.029273 eV (measured 2026-11-18)."""
    _bulk_bi2se3_with_spin_orbit_coupling(BI2SE3_CELL, BI2SE3_POSITIONS, tmp_path)


@pytest.mark.slow
def test_bulk_bi2se3_without_spin_orbit_coupling_is_a_trivial_insulator(tmp_path):
    """Z2 = (0; 000) on the scalar-relativistic set: every delta is +1, the spin-degenerate occupied
    manifold holds 18 odd spatial bands at Gamma, the flow counts every crossing twice, and the signed
    Gamma gap is the plain gap of an independent scalar-relativistic reference (+0.42 eV, mesh 3; this
    engine gives 0.4228 on the 3x3x3 mesh). 2543 s (SCF 309 s, parity 82 s, six planes), peak RSS 1.5 GB (measured 2026-11-13)."""
    atoms = _bi2se3(BI2SE3_CELL, BI2SE3_POSITIONS, "pbe-sr", tmp_path)
    parity = atoms.calc.get_z2_invariant(kpts=BI2SE3_STRINGS, method="parity")
    assert parity.indices == (0, 0, 0, 0) and parity.spin_degenerate
    assert all(p["delta"] == 1 and p["residual"] < 1e-6 and p["occupied"] == 39
               for p in parity.parity)
    gamma = next(p for p in parity.parity if p["index"] == (0, 0, 0))
    assert gamma["minus"] == 18
    # The normal order at Gamma: the highest occupied level is odd, the lowest empty one even.
    assert (gamma["parities"][38], gamma["parities"][39]) == (-1, 1)
    assert abs(_signed_gamma_gap(parity) - 0.42) < 0.05
    flow = atoms.calc.get_z2_invariant(kpts=BI2SE3_STRINGS)
    assert flow.indices == (0, 0, 0, 0) and flow.planes == parity.planes and flow.spin_degenerate


def _lih_polar(**options):
    """Polar LiH chain in a skew cell (no inversion center, nonzero P and Z*)."""
    atoms = Atoms("LiH", positions=[[0.0, 0.0, 0.0], [1.6, 0.25, 0.15]],
                  cell=[[3.5, 0.0, 0.0], [0.4, 3.8, 0.0], [0.2, 0.3, 4.0]], pbc=True)
    settings = dict(method="dft", xc="lda", basis={"name": "PAW-LCAO", "size": "SZ"},
                    grid_spacing=0.38, kpts={"size": (1, 1, 1), "gamma": True},
                    smearing={"method": "fermi-dirac", "width": 0.01},
                    parallelization={"threads": 2}, trace=False)
    settings.update(options)
    atoms.calc = Mandacaru(**settings)
    atoms.get_potential_energy()
    return atoms


def test_born_effective_charges_views_asr_and_spin_channels(chain):
    z = chain.calc.get_born_effective_charges(displacement=0.01, kpts=MESH)
    assert isinstance(z, BornCharges)
    assert z.charges.shape == (2, 3, 3)
    assert z.channels.shape == (1, 2, 3, 3)
    assert z.polarization_steps.shape == (2, 3, 3)
    assert abs(z.displacement - 0.01) < 1e-12
    assert isinstance(z.asr_residual, float) and z.corrected is None
    assert len(z.scf_iterations) > 0
    np.testing.assert_allclose(z.charges[0] + z.charges[1], z.asr_mean * 2.0, atol=1e-10)

    z_imposed = chain.calc.get_born_effective_charges(
        displacement=0.01, kpts=MESH, acoustic_sum_rule="impose"
    )
    assert z_imposed.corrected is not None and z_imposed.corrected.shape == (2, 3, 3)
    np.testing.assert_allclose(z_imposed.corrected.sum(axis=0), 0.0, atol=1e-12)

    polarized = _chain(spin=True)
    z_spin = polarized.calc.get_born_effective_charges(displacement=0.01, kpts=MESH)
    assert z_spin.channels.shape == (2, 2, 3, 3)
    np.testing.assert_allclose(z_spin.charges, z.charges, atol=1e-3)


def test_piezoelectric_tensor_clamped_and_units(chain):
    mesh = (2, 2, 2)
    e_si = chain.calc.get_piezoelectric_tensor(strain=0.005, kpts=mesh, ions="clamped")
    assert isinstance(e_si, PiezoelectricTensor)
    assert e_si.tensor.shape == (3, 6) and e_si.clamped.shape == (3, 6)
    np.testing.assert_allclose(e_si.tensor, e_si.clamped, atol=1e-12)
    assert e_si.relaxed is None and e_si.born_charges is None
    assert e_si.internal_strain is None and e_si.force_constants is None
    assert e_si.acoustic_frequencies_removed is None
    assert e_si.polarization_steps.shape == (6, 3)
    assert len(e_si.scf_iterations) == 12
    # Centrosymmetric H2 chain has zero clamped piezoelectric tensor.
    np.testing.assert_allclose(e_si.clamped, 0.0, atol=1e-3)

    chain_au = _chain(atomic_units=True)
    e_au = chain_au.calc.get_piezoelectric_tensor(strain=0.005, kpts=mesh, ions="clamped")
    np.testing.assert_allclose(e_au.tensor * E_PER_BOHR2, e_si.tensor, atol=1e-6)


def test_born_and_piezo_refusals(chain, metal):
    with pytest.raises(ValueError, match="positive"):
        chain.calc.get_born_effective_charges(displacement=0.0, kpts=MESH)
    with pytest.raises(ValueError, match="acoustic_sum_rule"):
        chain.calc.get_born_effective_charges(acoustic_sum_rule="drop", kpts=MESH)
    with pytest.raises(ValueError, match="positive"):
        chain.calc.get_piezoelectric_tensor(strain=-0.01, kpts=MESH)
    with pytest.raises(ValueError, match="ions"):
        chain.calc.get_piezoelectric_tensor(ions="frozen", kpts=MESH)
    with pytest.raises(ValueError, match="bool"):
        chain.calc.get_piezoelectric_tensor(richardson="yes", kpts=MESH)
    with pytest.raises(ValueError):
        chain.calc.get_born_effective_charges(displacement=0.2, kpts=MESH)
    with pytest.raises(ValueError):
        chain.calc.get_piezoelectric_tensor(strain=0.05, kpts=MESH)

    mol = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    mol.calc = Mandacaru(method="dft", xc="lda", basis={"name": "PAW-LCAO", "size": "SZ"},
                         grid_spacing=0.25, trace=False)
    with pytest.raises(ValueError, match="Brillouin zone"):
        mol.calc.get_born_effective_charges(kpts=MESH)
    with pytest.raises(ValueError, match="Brillouin zone"):
        mol.calc.get_piezoelectric_tensor(kpts=MESH)

    with pytest.raises(ValueError, match="insulator"):
        metal.calc.get_born_effective_charges(kpts=(4, 2, 2))
    with pytest.raises(ValueError, match="insulator"):
        metal.calc.get_piezoelectric_tensor(kpts=(4, 2, 2))

    qmol = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]], cell=[3.0, 4.0, 4.0], pbc=True)
    qmol.calc = Mandacaru(method="adapt-vqe", crystal_orbitals="real-space", trace=False)
    with pytest.raises(NotImplementedError, match="Bloch bands"):
        qmol.calc.get_born_effective_charges(kpts=(2, 1, 1))
    with pytest.raises(NotImplementedError, match="Bloch bands"):
        qmol.calc.get_piezoelectric_tensor(kpts=(2, 1, 1))

    f32 = _chain(precision="float32")
    with pytest.raises(ValueError, match="float64"):
        f32.calc.get_born_effective_charges(kpts=MESH)
    with pytest.raises(ValueError, match="float64"):
        f32.calc.get_piezoelectric_tensor(kpts=MESH)


def test_ts_dispersion_clamped_piezoelectric():
    chain_ts = _chain(vdw="ts")
    e_ts = chain_ts.calc.get_piezoelectric_tensor(strain=0.005, kpts=(2, 2, 2), ions="clamped")
    assert e_ts.tensor.shape == (3, 6)
    np.testing.assert_allclose(e_ts.clamped, 0.0, atol=1e-3)


@pytest.mark.slow  # 255 s: 18 SCF+force calls on P1 LiH skew cell; gated fast in berry_perturbation.rs
def test_polar_lih_soft_mode_refusal():
    lih = _lih_polar()
    with pytest.raises(ValueError, match="soft or unstable mode"):
        lih.calc.get_piezoelectric_tensor(strain=0.005, kpts=(2, 2, 2), ions="relaxed")


@pytest.mark.slow
def test_zincblende_bn_relaxed_piezoelectric_tensor():
    a = 3.615
    bn = Atoms("BN", scaled_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
               cell=[[0.0, 0.5 * a, 0.5 * a], [0.5 * a, 0.0, 0.5 * a], [0.5 * a, 0.5 * a, 0.0]],
               pbc=True)
    bn.calc = Mandacaru(
        method="dft", xc="lda", basis={"name": "PAW-LCAO", "size": "SZ"},
        grid_spacing=0.30, kpts={"size": (2, 2, 2), "gamma": True},
        parallelization={"threads": 2}, trace=False,
    )
    bn.get_potential_energy()
    e = bn.calc.get_piezoelectric_tensor(strain=0.005, displacement=0.01, kpts=(2, 2, 2),
                                         ions="relaxed")
    assert isinstance(e, PiezoelectricTensor) and e.relaxed is not None
    assert e.born_charges is not None and e.born_charges.charges.shape == (2, 3, 3)
    assert e.internal_strain is not None and e.internal_strain.shape == (6, 6)
    assert e.force_constants is not None and e.force_constants.shape == (6, 6)
    assert e.acoustic_frequencies_removed is not None and e.acoustic_frequencies_removed.shape == (3,)
    np.testing.assert_allclose(e.tensor, e.relaxed, atol=1e-12)
    np.testing.assert_allclose(e.force_constants, e.force_constants.T, atol=1e-10)
    assert abs(e.tensor[0, 3]) > 1e-3
    np.testing.assert_allclose(e.tensor[0, 3], e.tensor[1, 4], atol=1e-2)
    np.testing.assert_allclose(e.tensor[0, 3], e.tensor[2, 5], atol=1e-2)
