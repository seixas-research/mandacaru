"""Public interfaces loaded on demand; optional integrations stay optional."""

from importlib import import_module

_EXPORTS = {'Fermion': ('.mapping', 'Fermion'), 'PauliSum': ('.mapping', 'PauliSum'), 'bravyi_kitaev': ('.mapping', 'bravyi_kitaev'), 'jordan_wigner': ('.mapping', 'jordan_wigner'), 'parity': ('.mapping', 'parity'), 'parity_reduced': ('.mapping', 'parity_reduced'), 'HAMILTONIAN_FORMATS': ('.serialization', 'HAMILTONIAN_FORMATS'), 'HamiltonianRecord': ('.serialization', 'HamiltonianRecord'), 'HamiltonianHeader': ('.serialization', 'HamiltonianHeader'), 'detect_format': ('.serialization', 'detect_format'), 'load_hamiltonian': ('.serialization', 'load_hamiltonian'), 'read_hamiltonian_header': ('.serialization', 'read_hamiltonian_header'), 'save_hamiltonian': ('.serialization', 'save_hamiltonian'), 'WavefunctionCheckpoint': ('.checkpoint', 'WavefunctionCheckpoint'), 'load_checkpoint': ('.checkpoint', 'load_checkpoint'), 'prepare_state': ('.checkpoint', 'prepare_state'), 'electric_dipole_potential': ('.dipole', 'electric_dipole_potential'), 'electric_field_vector': ('.dipole', 'electric_field_vector'), 'SymmetryLeakError': ('.tapering', 'SymmetryLeakError'), 'reduce_generators': ('.tapering', 'reduce_generators'), 'sector_signs': ('.tapering', 'sector_signs'), 'symmetry_generators': ('.tapering', 'symmetry_generators'), 'taper': ('.tapering', 'taper')}
__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = (
        import_module("." + attribute, __name__)
        if module == "."
        else getattr(import_module(module, __name__), attribute)
    )
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
