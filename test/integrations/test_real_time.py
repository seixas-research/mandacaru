"""``Mandacaru.get_real_time``: the request reaches the native propagator, the spectrum's
first line sits at the first bright Casida root of ``get_excitations``, a crystal runs in the
velocity gauge with its dielectric function, the refusals name what is missing, and each call
writes one ``[REAL TIME]`` block. ``Mandacaru.get_dielectric_function``: the frequency-domain
dielectric function of a crystal, its refusals and its ``[DIELECTRIC FUNCTION]`` block.

H2 on a PAW-LCAO DZP basis, and an H2 chain (SZ, Gamma). Needs ``MANDACARU_PAW_PATH``."""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.utils import parse_output


def hydrogen(tmp_path=None, **options):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[5, 5, 5])
    atoms.center()
    settings = dict(method="dft", xc="lda", basis={"name": "PAW-LCAO", "size": "DZP"},
                    grid_spacing=0.3)
    settings.update(options)
    if tmp_path is not None:
        settings["txt"] = str(tmp_path / "run.log")
    atoms.calc = Mandacaru(**settings)
    return atoms


def test_the_spectrum_peaks_at_the_bright_casida_root(tmp_path):
    atoms = hydrogen(tmp_path)
    rt = atoms.calc.get_real_time(duration=6.0, time_step=0.01, direction="z")
    assert rt.times.shape == (rt.steps + 1,) and rt.dipole.shape == (rt.steps + 1, 3)
    assert rt.propagator == "etrs" and rt.kick == "density" and rt.gauge == "length"
    assert rt.step_width <= rt.step_width_bound
    assert rt.worst_norm_error < 1e-10 and rt.energy_drift < 0.05
    # The kick along the bond excites the sigma_u roots only; the tallest line has the largest
    # f_z = 2 omega mu_z^2.
    casida = atoms.calc.get_excitations(9, approximation="casida")
    bright = int(np.argmax(casida.energies * casida.transition_dipoles[:, 2] ** 2))
    energies = np.linspace(5.0, 40.0, 3501)
    spectrum = rt.strength_function(energies, damping=0.5)
    assert np.all(np.isfinite(spectrum))
    peak = energies[int(np.argmax(spectrum))]
    assert abs(peak - casida.energies[bright]) < 0.15
    alpha = rt.polarizability(energies[:5], damping=0.5)
    assert alpha.dtype == np.complex128
    block = parse_output(str(tmp_path / "run.log"))["steps"][-1]["real_time"]
    assert block["propagator"] == "etrs" and block["kick"] == "density"
    assert int(block["steps"]) == rt.steps
    assert "fock_builds" in block and "step_width_bound" in block


def test_the_atomic_units_view_and_the_spin_kick():
    atoms = hydrogen(atomic_units=True)
    rt = atoms.calc.get_real_time(duration=2.0, time_step=0.2, direction="z", strength=1e-3,
                                  kick="spin")
    assert rt.time_unit == "au" and abs(rt.times[1] - 0.2) < 1e-12
    assert abs(rt.strength - 1e-3) < 1e-15 and rt.kick == "spin"


def test_refusals_name_what_is_missing():
    atoms = hydrogen()
    with pytest.raises(ValueError, match="etrs"):
        atoms.calc.get_real_time(duration=0.1, propagator="crank-nicolson")
    with pytest.raises(ValueError, match="time_step"):
        atoms.calc.get_real_time(duration=1.0, time_step=0.5, propagator="aetrs")
    with pytest.raises(ValueError, match="direction"):
        atoms.calc.get_real_time(duration=0.1, direction="w")
    with pytest.raises(NotImplementedError, match="rVV10"):
        hydrogen(xc="pbe+rvv10").calc.get_real_time(duration=0.1)
    with pytest.raises(ValueError, match="kpts"):
        atoms.calc.get_real_time(duration=0.1, kpts=(2, 2, 2))
    with pytest.raises(NotImplementedError, match="independent"):
        atoms.calc.get_real_time(duration=0.1, kernel="independent")


BOHR = 0.529177210903


def chain(tmp_path=None):
    """H2 (1.4 Bohr bond) along x in a 6 x 8 x 8 Bohr cell: a centrosymmetric insulator."""
    atoms = Atoms("H2", positions=[[2.3 * BOHR, 4.0 * BOHR, 4.0 * BOHR],
                                   [3.7 * BOHR, 4.0 * BOHR, 4.0 * BOHR]],
                  cell=[6.0 * BOHR, 8.0 * BOHR, 8.0 * BOHR], pbc=True)
    settings = dict(method="dft", xc="lda", basis={"name": "PAW-LCAO", "size": "SZ"},
                    grid_spacing=0.4 * BOHR, kpts={"size": (1, 1, 1), "gamma": True},
                    smearing={"method": "fermi-dirac", "width": 0.01}, trace=False)
    if tmp_path is not None:
        settings["txt"] = str(tmp_path / "run.log")
    atoms.calc = Mandacaru(**settings)
    return atoms


def test_a_crystal_propagates_in_the_velocity_gauge(tmp_path):
    atoms = chain(tmp_path)
    rt = atoms.calc.get_real_time(duration=1.0, time_step=0.01, direction="x")
    assert rt.periodic and rt.dipole is None and rt.current.shape == (rt.steps + 1, 3)
    assert rt.gauge == "velocity" and rt.kernel == "adiabatic" and rt.mesh == (1, 1, 1)
    assert rt.worst_norm_error < 1e-10 and rt.energy_drift < 0.05
    assert rt.electrons == 2 and len(rt.diamagnetic) == 1 and rt.diamagnetic[0] > 0
    assert rt.diamagnetic_source == "band sum"
    epsilon = rt.dielectric_function(np.linspace(2.0, 30.0, 8), damping=1.0)
    assert epsilon.dtype == np.complex128 and np.all(np.isfinite(epsilon))
    block = parse_output(str(tmp_path / "run.log"))["steps"][-1]["real_time"]
    assert block["kernel"] == "adiabatic" and block["gauge"] == "velocity"
    assert "diamagnetic_sum" in block and "mesh" in block
    with pytest.raises(ValueError, match="dielectric_function"):
        rt.strength_function(np.linspace(2.0, 30.0, 8))
    with pytest.raises(ValueError, match="dielectric_function"):
        hydrogen().calc.get_real_time(duration=0.2, time_step=0.01).dielectric_function([5.0])
    with pytest.raises(ValueError, match="exponential"):
        rt.dielectric_function(np.linspace(2.0, 30.0, 8), window="gaussian")


def test_direction_all_gives_the_tensor_of_three_runs(tmp_path):
    atoms = chain(tmp_path)
    rt = atoms.calc.get_real_time(duration=0.5, time_step=0.01, direction="all",
                                  kernel="independent")
    nt = rt.steps + 1
    assert rt.current.shape == (nt, 3, 3) and rt.corrected.shape == (nt, 3, 3)
    assert rt.along.shape == (nt, 3) and rt.energy.shape == (nt, 3)
    assert len(rt.diamagnetic) == 3 and len(rt.direction) == 9
    epsilon = rt.dielectric_function(np.linspace(2.0, 30.0, 8), damping=1.0)
    assert epsilon.shape == (8, 3, 3) and np.all(np.isfinite(epsilon))
    block = parse_output(str(tmp_path / "run.log"))["steps"][-1]["real_time"]
    assert len(str(block["diamagnetic_sum"]).split()) == 3
    with pytest.raises(NotImplementedError, match="all"):
        hydrogen().calc.get_real_time(duration=0.1, time_step=0.01, direction="all")


def test_a_crystal_refuses_the_length_gauge_and_the_spin_kick():
    atoms = chain()
    with pytest.raises(ValueError, match="velocity"):
        atoms.calc.get_real_time(duration=0.1, gauge="length")
    with pytest.raises(ValueError, match="density"):
        atoms.calc.get_real_time(duration=0.1, kick="spin")
    with pytest.raises(ValueError, match="hartree"):
        atoms.calc.get_real_time(duration=0.1, kernel="rpa")


def test_r2scan_propagates_with_the_gauge_invariant_kinetic_density():
    rt = hydrogen(xc="r2scan").calc.get_real_time(duration=0.5, time_step=0.01, direction="z")
    assert rt.functional == "r2scan" and rt.worst_norm_error < 1e-10 and rt.energy_drift < 0.05
    assert np.all(np.isfinite(rt.dipole))


def test_a_crystal_has_a_frequency_domain_dielectric_function(tmp_path):
    atoms = chain(tmp_path)
    view = atoms.calc.get_dielectric_function(np.linspace(0.0, 30.0, 7), damping=1.0)
    assert view.kernel == "hartree" and view.mesh == (1, 1, 1) and view.energy_unit == "eV"
    assert view.tensor.shape == (7, 3, 3) and view.epsilon.shape == (7,)
    assert view.loss.shape == view.absorption.shape == (7,) and view.static_tensor.shape == (3, 3)
    assert view.static_tensor[0, 0] > 1 and view.sum_rule[0, 0] > 0 and view.vectors > 0
    along = atoms.calc.get_dielectric_function([5.0], kernel="independent", direction="x")
    assert along.cutoff is None and along.vectors == 0 and along.direction == [1.0, 0.0, 0.0]
    assert np.isclose(along.epsilon[0], along.tensor[0, 0, 0])
    block = parse_output(str(tmp_path / "run.log"))["steps"][-1]["dielectric_function"]
    assert block["kernel"] == "independent" and "static_dielectric_diagonal" in block
    model = atoms.calc.get_dielectric_function([5.0], kernel="long-range", damping=1.0)
    assert model.kernel == "long-range" and model.alpha > 0 and view.alpha is None
    with pytest.raises(ValueError, match="long-range"):
        atoms.calc.get_dielectric_function([5.0], alpha=0.2)
    with pytest.raises(ValueError, match="hartree"):
        atoms.calc.get_dielectric_function([5.0], kernel="rpa")
    with pytest.raises(ValueError, match="tensor"):
        atoms.calc.get_dielectric_function([5.0], direction="all")
    with pytest.raises(ValueError, match="crystal"):
        hydrogen().calc.get_dielectric_function([5.0])
