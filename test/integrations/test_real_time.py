"""``Mandacaru.get_real_time``: the request reaches the native propagator, the spectrum's
first line sits at the first bright Casida root of ``get_excitations``, the refusals name what
is missing, and each call writes one ``[REAL TIME]`` block.

H2 on a PAW-LCAO DZP basis. Needs ``MANDACARU_PAW_PATH``."""

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
    with pytest.raises(NotImplementedError, match="r2scan"):
        hydrogen(xc="r2scan").calc.get_real_time(duration=0.1)
    crystal = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[3, 3, 3], pbc=True)
    crystal.calc = Mandacaru(method="dft", xc="lda", basis={"name": "PAW-LCAO", "size": "DZP"},
                             grid_spacing=0.3)
    with pytest.raises(NotImplementedError, match="crystal"):
        crystal.calc.get_real_time(duration=0.1)
