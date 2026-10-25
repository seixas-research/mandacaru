# -*- coding: utf-8 -*-
# file: test/integrals/test_precision.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Tests for mixed-precision real-space integration and calculator options."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from mandacaru.algorithms import Mandacaru
from mandacaru.basis import HydrogenicAtomicOrbital
from mandacaru.integrals import Grid, IntegralEngine, _backend


def test_grid_precision_dtypes():
    """Grid dtypes match precision specification."""
    g_double = Grid(box_size=4.0, grid_spacing=1.0, center=(0.0, 0.0, 0.0), precision="double")
    assert g_double.precision == "double"
    assert g_double.real_dtype == np.float64
    assert g_double.complex_dtype == np.complex128

    g_single = Grid(box_size=4.0, grid_spacing=1.0, center=(0.0, 0.0, 0.0), precision="single")
    assert g_single.precision == "single"
    assert g_single.real_dtype == np.float32
    assert g_single.complex_dtype == np.complex64

    with pytest.raises(ValueError, match="precision must be 'double' or 'single'"):
        Grid(box_size=4.0, grid_spacing=1.0, center=(0.0, 0.0, 0.0), precision="quadruple")


def test_engine_one_body_single_vs_double():
    """One-body integrals in single precision match double precision to high accuracy."""
    basis = [HydrogenicAtomicOrbital(1, 0, 0)]
    g_double = Grid(box_size=6.0, grid_spacing=0.4, center=(0.0, 0.0, 0.0), precision="double")
    g_single = Grid(box_size=6.0, grid_spacing=0.4, center=(0.0, 0.0, 0.0), precision="single")

    eng_double = IntegralEngine(basis, g_double)
    eng_single = IntegralEngine(basis, g_single)

    def vext(x, y, z):
        r = np.sqrt(x * x + y * y + z * z)
        return -1.0 / np.maximum(r, 0.2)

    T_d, V_d = eng_double.one_body(vext)
    T_s, V_s = eng_single.one_body(vext)

    # Matrix dtypes reflect the engine precision
    assert T_d.dtype == np.float64
    assert T_s.dtype == np.float32

    # Agreement between single and double should be within 1e-4 Hartree
    np.testing.assert_allclose(T_s, T_d, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(V_s, V_d, atol=1e-4, rtol=1e-4)


def test_calculator_precision_rhf_h2():
    """RHF calculation of H2 agrees between single and double precision."""
    h2_d = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
                 cell=[6.0, 6.0, 6.0], pbc=False)
    calc_double = Mandacaru(method="rhf", basis="HAO",
                            grid_spacing=0.35, precision="double")
    h2_d.calc = calc_double
    e_double = h2_d.get_potential_energy()

    h2_s = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
                 cell=[6.0, 6.0, 6.0], pbc=False)
    calc_single = Mandacaru(method="rhf", basis="HAO",
                            grid_spacing=0.35, precision="single")
    h2_s.calc = calc_single
    e_single = h2_s.get_potential_energy()

    # Energy difference must be very small (< 1e-3 eV)
    assert abs(e_single - e_double) < 1e-3


def test_calculator_precision_periodic_al():
    """Periodic DFT calculation of FCC Al agrees between single and double precision."""
    al_d = bulk("Al", "fcc", a=4.05)
    calc_double = Mandacaru(method="dft", xc="lda", basis="PAW-LCAO",
                            kpts=(2, 2, 2), grid_spacing=0.4,
                            precision="double", trace=False)
    al_d.calc = calc_double
    e_double = al_d.get_potential_energy()

    al_s = bulk("Al", "fcc", a=4.05)
    calc_single = Mandacaru(method="dft", xc="lda", basis="PAW-LCAO",
                            kpts=(2, 2, 2), grid_spacing=0.4,
                            precision="single", trace=False)
    al_s.calc = calc_single
    e_single = al_s.get_potential_energy()

    assert abs(e_single - e_double) < 1e-4


def test_rust_f32_kernels_available():
    """When the native backend is Rust, single-precision kernels are exported."""
    status = _backend.check_backend()
    if not status.available or not getattr(status, "is_rust", False):
        pytest.skip("Rust backend not active")

    lib = _backend._LIB
    assert hasattr(lib, "mandacaru_real_shell_f32")
    assert hasattr(lib, "mandacaru_laplacian_real_f32")
    assert hasattr(lib, "mandacaru_kb_project_f32")
