# -*- coding: utf-8 -*-
# file: test/integrations/test_energy_error.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""``result.energy_error``: the shot-noise standard error of a sampled energy, computed in Rust
from the counts with the in-basis covariances (``quantum::energy_error``), and its two lines in
the run log.

The independent reference is the scatter of the energy over repeated sampled runs at the same
circuit (here the reference determinant, re-measured with a different simulator seed each
time); the Rust gate (``crates/mandacaru-core/tests/energy_error.rs``) also checks the estimate
against the exact sigma of the state.
"""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.utils.logging import parse_output

H2 = dict(symbols="H2", positions=[[3, 3, 2.63], [3, 3, 3.37]], cell=[6.0, 6.0, 6.0])
OPTIONS = dict(basis="HAO", grid_spacing=0.35, trace=False, atomic_units=True)


@pytest.fixture(scope="module")
def model():
    atoms = Atoms(**H2)
    atoms.calc = Mandacaru(method="rhf", **OPTIONS)
    atoms.get_potential_energy()
    return atoms.calc.native_reference.model


def measured(model, seed, **options):
    calc = Mandacaru(method="vqe", native_model=model, atomic_units=True, trace=False,
                     shots=2000, backend_options={"seed": seed},
                     optimizer={"method": "COBYLA", "maxiter": 0}, **options)
    return calc, calc.run()


def test_the_reported_error_is_the_scatter_of_repeated_runs(model):
    energies, errors = [], []
    for seed in range(24):
        _calc, result = measured(model, 500 + seed)
        energies.append(result.in_units("Ha"))
        errors.append(result.energy_error)
    observed, reported = np.std(energies, ddof=1), np.mean(errors)
    assert reported > 0
    # 24 runs estimate a standard deviation to about 15 percent.
    assert 0.6 < observed / reported < 1.5, (observed, reported)


def test_the_error_is_in_the_energy_unit_and_absent_without_shots(model):
    _calc, in_ha = measured(model, 7)
    from mandacaru import _native
    calc = Mandacaru(method="vqe", native_model=model, trace=False, shots=2000,
                     backend_options={"seed": 7}, optimizer={"method": "COBYLA", "maxiter": 0})
    result = calc.run()
    assert result.energy_error == pytest.approx(
        _native.to_user_units(in_ha.energy_error, "energy"), rel=1e-12)
    exact = Mandacaru(method="vqe", native_model=model, atomic_units=True, trace=False,
                      optimizer={"method": "COBYLA", "maxiter": 0}).run()
    assert exact.energy_error is None


def test_the_summary_owns_the_error_and_the_table_has_a_sigma_column(tmp_path):
    out = str(tmp_path / "output.txt")
    atoms = Atoms(**H2)
    atoms.calc = Mandacaru(method="adapt-vqe", pool="fermionic", shots=2000, txt=out,
                           backend_options={"seed": 3}, max_iterations=1,
                           optimizer={"method": "COBYLA", "maxiter": 5}, **OPTIONS)
    atoms.get_potential_energy()
    text = open(out).read()
    assert text.count("energy_standard_error_Ha") == 1
    summary = parse_output(out)["summary"]
    assert float(summary["energy_standard_error_Ha"]) == pytest.approx(
        atoms.calc.result.energy_error, rel=1e-6)
    assert "sigma (Ha)" in text


def strings_of(calc):
    return sum(1 for label, _re, _im in calc.problem.hamiltonian_terms if set(label) != {"I"})


def test_the_products_cost_no_budget_a_run_at_its_ceiling_still_runs(model):
    """Owner: "Exempt products from budget". A ceiling of exactly the Hamiltonian's strings
    admitted the run before the error existed and still does (the products would exceed it);
    one string less is refused as before."""
    calc, _ = measured(model, 11)
    ceiling = strings_of(calc)
    calc, result = measured(model, 11, measurement_budget={"observables": ceiling})
    assert result.energy_error is not None
    with pytest.raises(ValueError, match="observables budget exceeded"):
        measured(model, 11, measurement_budget={"observables": ceiling - 1})


def test_a_basis_past_the_product_limit_runs_and_states_why_there_is_no_error(tmp_path):
    """Every pair density on 25 orbitals puts 1,275 strings in the diagonal basis, whose
    products are far past the 20,000 limit: the run measures what it measured before and
    reports no error, with the reason stated once by the summary block."""
    from mandacaru.core.mapping import Fermion

    orbitals = 25
    terms = {}
    for i in range(2 * orbitals):
        terms[((i, True), (i, False))] = 0.01 * (i + 1)
        for j in range(i + 1, 2 * orbitals):
            terms[((i, True), (j, True), (j, False), (i, False))] = 0.001 * (i + j + 1)
    out = str(tmp_path / "output.txt")
    calc = Mandacaru(method="vqe", pool="fermionic", hamiltonian=Fermion(terms),
                     n_spatial_orbitals=orbitals, num_particles=(1, 1), shots=64,
                     backend_provider="qiskit", atomic_units=True, trace=False, txt=out,
                     optimizer={"method": "COBYLA", "maxiter": 1, "maxfev": 2},
                     measurement_budget={"observables": 500_000, "bases": 100_000})

    def landscape(n_qubits, occupied, requests, max_bases_per_job=None):
        return [({label: 0.1 for label in labels}, dict.fromkeys(labels, 0.01))
                for _generators, _thetas, labels in requests]

    calc.provider.expectation_batch = landscape
    result = calc.run()
    assert result.energy_error is None
    assert "not reported" in result.energy_error_reason and "20000" in result.energy_error_reason
    text = open(out).read()
    assert text.count("energy_standard_error_Ha") == 1
    assert "energy_standard_error_Ha: not reported" in text
