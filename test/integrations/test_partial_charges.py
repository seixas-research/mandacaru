"""End-to-end partial charge tests (Bader, Hirshfeld, Voronoi) via ASE and Result."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from mandacaru import Mandacaru


def _h2():
    atoms = Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]], cell=[6.0, 6.0, 6.0])
    atoms.center()
    return atoms


def _lih():
    atoms = Atoms("LiH", positions=[[0, 0, 0], [0, 0, 1.6]], cell=[6.5, 6.5, 7.5])
    atoms.center()
    return atoms


@pytest.mark.parametrize("method_scf", ["rhf", "dft"])
def test_molecular_mean_field_partial_charges_conserve_charge_and_polarity(method_scf):
    kwargs = {"xc": "lda"} if method_scf == "dft" else {}
    h2 = _h2()
    h2.calc = Mandacaru(method=method_scf, basis="HAO", grid_spacing=0.35, **kwargs)
    h2.get_potential_energy()
    for pm in ("bader", "hirshfeld", "voronoi"):
        q = h2.calc.get_partial_charges(pm)
        assert q.shape == (2,)
        assert np.isclose(q.sum(), 0.0, atol=1e-6)
        assert abs(q[0] - q[1]) < 1e-4

    lih = _lih()
    lih.calc = Mandacaru(method=method_scf, basis="HAO", grid_spacing=0.35, **kwargs)
    lih.get_potential_energy()
    for pm in ("bader", "hirshfeld", "voronoi"):
        res = lih.calc.result.partial_charges(pm)
        assert res["interpolated"] is True
        q = lih.calc.get_partial_charges(pm)
        assert np.allclose(q, res["charges"])
        assert np.isclose(q.sum(), res["net_charge"], atol=1e-6)
        assert q[0] > 0.15 and q[1] < -0.15, f"{method_scf}/{pm}: {q}"

        res_raw = lih.calc.result.partial_charges(pm, interpolate=False)
        assert res_raw["interpolated"] is False
        q_raw = lih.calc.get_partial_charges(pm, interpolate=False)
        assert np.allclose(q_raw, res_raw["charges"])
        assert np.isclose(q_raw.sum(), res_raw["net_charge"], atol=1e-6)


@pytest.mark.parametrize("vq_method", ["vqe", "adapt-vqe", "vasqa", "valsqa"])
def test_variational_quantum_algorithms_support_all_partial_charge_methods(vq_method):
    extra = (
        {"max_steps": 3, "max_length": 2, "seed": 7}
        if vq_method in ("vasqa", "valsqa")
        else {}
    )
    atoms = _h2()
    atoms.calc = Mandacaru(
        method=vq_method,
        basis="HAO",
        grid_spacing=0.4,
        **extra,
    )
    atoms.get_potential_energy()
    for pm in ("bader", "hirshfeld", "voronoi"):
        res = atoms.calc.result.partial_charges(pm)
        q = atoms.calc.get_partial_charges(pm)
        assert q.shape == (2,)
        assert np.allclose(q, res["charges"])
        assert np.isclose(q.sum(), 0.0, atol=1e-5)
        assert abs(q[0] - q[1]) < 1e-3, f"{vq_method}/{pm}: {q}"


def test_spin_orbit_ghf_and_vqe_partial_charges_on_hbr():
    from mandacaru.pseudopotentials import get_paw

    try:
        get_paw("Br", "pbe-fr")
    except (FileNotFoundError, ValueError):
        pytest.skip("MANDACARU_PAW_PATH with pbe-fr is not configured")

    hbr = Atoms("HBr", positions=[[0, 0, 0], [0, 0, 1.41]], cell=[6.0, 6.0, 7.0])
    hbr.center()
    basis_fr = {"name": "PAW-LCAO", "size": "SZ", "paw_dataset": "pbe-fr"}
    for method in ("ghf", "vqe"):
        hbr.calc = Mandacaru(method=method, basis=basis_fr, grid_spacing=0.35)
        hbr.get_potential_energy()
        for pm in ("bader", "hirshfeld", "voronoi"):
            q = hbr.calc.get_partial_charges(pm)
            assert q.shape == (2,)
            assert np.isclose(q.sum(), 0.0, atol=1e-4)
            if pm != "voronoi":
                # Bader and Hirshfeld capture H -> Br polarity; bisecting Voronoi
                # assigns Br's diffuse 4p tail across R/2 to the small H atom.
                assert q[0] > 0.05 and q[1] < -0.05, f"{method}/{pm}: {q}"


def test_periodic_crystal_dft_partial_charges():
    from mandacaru.pseudopotentials import get_paw

    try:
        get_paw("Si")
    except (FileNotFoundError, ValueError):
        pytest.skip("MANDACARU_PAW_PATH is not configured")

    si = bulk("Si", "diamond", a=5.43)
    si.calc = Mandacaru(
        method="dft",
        xc="lda",
        basis={"name": "PAW-LCAO", "size": "SZ"},
        grid_spacing=0.4,
        kpts=(1, 1, 1),
    )
    si.get_potential_energy()
    for pm in ("bader", "hirshfeld", "voronoi"):
        q = si.calc.get_partial_charges(pm)
        q_raw = si.calc.get_partial_charges(pm, interpolate=False)
        assert q.shape == (2,)
        assert np.isclose(q.sum(), 0.0, atol=1e-5)
        assert np.isclose(q_raw.sum(), 0.0, atol=1e-5)
        assert abs(q[0] - q[1]) < 2e-2, f"Si {pm}: {q}"
        assert abs(q[0] - q[1]) <= abs(q_raw[0] - q_raw[1]) + 1e-6


def test_periodic_quantum_vqe_partial_charges():
    cell = Atoms("H2", positions=[[2.0, 2.0, 1.4], [2.0, 2.0, 2.6]], cell=[4.0, 4.0, 4.0], pbc=True)
    cell.calc = Mandacaru(method="vqe", basis="HAO", grid_spacing=0.35, kpts=(1, 1, 1))
    cell.get_potential_energy()
    for pm in ("bader", "hirshfeld", "voronoi"):
        q = cell.calc.get_partial_charges(pm)
        assert q.shape == (2,)
        assert np.isclose(q.sum(), 0.0, atol=1e-5)
        assert abs(q[0] - q[1]) < 1e-3, f"periodic VQE {pm}: {q}"
