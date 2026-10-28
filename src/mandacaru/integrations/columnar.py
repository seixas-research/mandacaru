"""Optional Parquet decoding into an owned native Hamiltonian document."""

import json


def hamiltonian_snapshot(context, path):
    from ..core.serialization import load_hamiltonian, FORMAT_TAG

    record = load_hamiltonian(path)
    payload = dict(
        format=FORMAT_TAG,
        version=1,
        num_qubits=record.num_qubits,
        mapping=record.mapping,
        num_particles=record.num_particles,
        n_spatial_orbitals=record.n_spatial_orbitals,
        metadata=record.metadata,
        terms=[
            (label, float(value.real), float(value.imag))
            for label, value in record.hamiltonian.terms.items()
        ],
    )
    return context.parse_snapshot(json.dumps(payload, allow_nan=False), FORMAT_TAG)
