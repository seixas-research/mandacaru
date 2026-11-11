"""A test-side view of the native double factorization (``_native.double_factorization`` and
``_native.spatial_double_factorization``): its leaves, its measurement count, its one-norm and
the second-quantized operator it represents, as the oracle the factorization tests compare.

The package has no Python factorization: the double-factorized measurement is the calculator
option ``measurement_scheme="double-factorized"``, built in Rust. These definitions are the
checks' own (``L + 1`` bases; the one-norm of the one-body eigenvalues and of each leaf).
"""
from types import SimpleNamespace

import numpy as np

from mandacaru import _native
from mandacaru.core.mapping import Fermion


def _view(raw):
    n_modes, constant, one_body, one_body_orbitals, one_body_energies, leaves, truncation = raw
    leaves = [SimpleNamespace(weight=float(w), orbitals=u, coefficients=g) for w, u, g in leaves]
    one_norm = float(np.abs(one_body_energies).sum() + sum(
        0.5 * abs(leaf.weight) * float(np.abs(leaf.coefficients).sum()) ** 2 for leaf in leaves))

    def to_fermion():
        operator = Fermion.from_integrals(one_body)
        if constant:
            operator = operator + Fermion({(): complex(constant)}, n_modes=n_modes)
        for leaf in leaves:
            single = Fermion.from_integrals(leaf.orbitals @ np.diag(leaf.coefficients)
                                            @ leaf.orbitals.conj().T)
            operator = operator + (0.5 * leaf.weight) * (single * single)
        return operator

    return SimpleNamespace(
        n_modes=n_modes, constant=constant, one_body=one_body,
        one_body_orbitals=one_body_orbitals, one_body_energies=one_body_energies,
        leaves=leaves, truncation_error=truncation, measurement_bases=len(leaves) + 1,
        one_norm=one_norm, to_fermion=to_fermion,
        energy=lambda one_rdm, two_rdm: _native.factorized_energy_rdm(
            float(constant), np.ascontiguousarray(one_body, dtype=np.complex128),
            [leaf.weight for leaf in leaves],
            [np.ascontiguousarray(leaf.orbitals, dtype=np.float64) for leaf in leaves],
            [np.ascontiguousarray(leaf.coefficients, dtype=np.float64) for leaf in leaves],
            np.ascontiguousarray(one_rdm, dtype=np.float64),
            np.ascontiguousarray(two_rdm, dtype=np.float64)))


def double_factorization(h, g, *, constant=0.0):
    return _view(_native.double_factorization(
        np.ascontiguousarray(h, dtype=np.float64), np.ascontiguousarray(g, dtype=np.float64),
        constant=float(constant)))


def spatial_double_factorization(h, eri=None, *, factors=None, coupling=None, constant=0.0):
    as_array = (lambda a: None if a is None else np.ascontiguousarray(a, dtype=np.float64))
    return _view(_native.spatial_double_factorization(
        as_array(h), eri=as_array(eri), factors=as_array(factors), coupling=as_array(coupling),
        constant=float(constant)))


def givens_decomposition(unitary):
    return _native.givens_decomposition(np.ascontiguousarray(unitary, dtype=np.float64))
