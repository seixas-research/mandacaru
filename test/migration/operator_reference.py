"""Readers of ``reference/operator_algebra.json`` (written by ``freeze_operator_algebra.py``).

The file froze the Python operator algebra before it was deleted (MG-Q5): the
mappings, products, commutators, matrices, sectors and reference determinants
the native route must reproduce.
"""

import json
from pathlib import Path

import numpy as np

REFERENCE = json.loads((Path(__file__).parent / "reference" / "operator_algebra.json").read_text())


def complex_of(value):
    return complex(value[0], value[1])


def pauli_terms(frozen):
    """``{label: complex}`` of a frozen Pauli sum."""
    return {label: complex_of(value) for label, value in frozen.items()}


def matrix_of(frozen):
    return np.array([[complex_of(v) for v in row] for row in frozen], dtype=complex)


def vector_of(frozen):
    return np.array([complex_of(v) for v in frozen], dtype=complex)


def encoded_index(mapping, occupied, n_modes=6):
    """Basis-state index of the determinant with the ``occupied`` spin-orbitals
    in the encoding, from the frozen bit strings (qubit 0 the most significant)."""
    assert n_modes == 6
    mask = sum(1 << int(j) for j in occupied)
    return int(REFERENCE["reference_bits"][mapping][str(mask)], 2)


def frozen_sector(mapping, particles=(1, 1)):
    """Sorted basis-state indices of the particle sector of 6 spin-orbitals."""
    for case in REFERENCE["sectors"]:
        if (case["mapping"] == mapping and case["num_particles"] == list(particles)
                and case.get("spin_conserving", True)):
            return np.array(case["indices"], dtype=np.int64)
    raise KeyError((mapping, particles))
