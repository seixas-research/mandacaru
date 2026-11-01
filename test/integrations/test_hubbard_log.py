"""The DFT+U facts of a real crystal run, as ``parse_output`` reads them from the run log.

The fields themselves are pinned with stand-ins in ``test_run_log_fields.py``; here one CuH box
runs end to end and the log is read back. Needs ``MANDACARU_PAW_PATH``.
"""

from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.utils import parse_output

OPTIONS = dict(
    method="dft", xc="lda", grid_spacing=0.3, symmetry=False,
    basis={"name": "PAW-LCAO", "size": "SZ"},
    smearing={"method": "fermi-dirac", "width": 0.01},
)


def logged(tmp_path, name, **option):
    atoms = Atoms("CuH", positions=[[0, 0, 0], [0, 0, 1.46]], cell=[7.0] * 3, pbc=True)
    atoms.center()
    log = tmp_path / f"{name}.log"
    atoms.calc = Mandacaru(txt=str(log), **OPTIONS, **option)
    atoms.get_potential_energy()
    return log, atoms.calc.result


def test_the_energy_and_the_definition_are_written_once_in_the_scf_summary(tmp_path):
    log, result = logged(tmp_path, "four", hubbard_u={"Cu": {"l": "d", "U": 4.0}})
    text = log.read_text()
    assert text.count("hubbard_energy_eV") == 1 and text.count("U_eff") == 1
    summary = parse_output(str(log))["summary"]
    assert abs(float(summary["hubbard_energy_eV"]) - result.hubbard_energy) < 1e-8
    assert summary["hubbard_Cu_d"].startswith("U 4.000000 eV, J 0.000000 eV, U_eff 4.000000 eV")


def test_a_run_without_the_correction_writes_no_hubbard_lines(tmp_path):
    log, _ = logged(tmp_path, "plain")
    assert "hubbard" not in log.read_text()
    assert "[SCF SUMMARY]" in log.read_text()
