"""Thin parameter and array adapters to native expressibility calculations."""

import numpy as np
from .. import _native


def active_space_dimension(n_qubits, num_particles=None):
    return _native.Context(threads=1).active_space_dimension(n_qubits, num_particles)


def sample_pqc_fidelities(ansatz, num_pairs=3000, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    seed = int(rng.integers(0, 2**63))
    return ansatz.problem.sample_fidelities(
        ansatz.indices, pairs=num_pairs, seed=seed, product_form=ansatz.product_form
    )


def calculate_kl_divergence(fidelities, n_qubits, num_bins=60, dimension=None):
    dimension = active_space_dimension(n_qubits) if dimension is None else dimension
    return _native.Context(threads=1).expressibility_kl(
        np.ascontiguousarray(fidelities, dtype=float), num_bins, dimension
    )
