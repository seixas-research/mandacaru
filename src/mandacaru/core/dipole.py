"""Electric dipole coupling in a finite orthonormal orbital space."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .mapping import Fermion, PauliSum, resolve_mapping


def electric_field_vector(direction: ArrayLike, strength: float) -> NDArray[np.float64]:
    """Return ``strength * direction`` after validating a Cartesian unit vector.

    ``strength`` is a finite, nonnegative field magnitude in atomic units.
    ``direction`` must be real, finite, length three and unit length to 1e-8.
    Non-unit inputs are rejected rather than silently rescaled. Reverse the
    unit vector to reverse the field; a zero magnitude is allowed.
    """
    unit = np.asarray(direction)
    if (unit.shape != (3,) or not np.isrealobj(unit)
            or not np.all(np.isfinite(unit))):
        raise ValueError("field_direction must be a finite real Cartesian unit vector")
    unit = unit.astype(float)
    if not np.isclose(np.linalg.norm(unit), 1.0, rtol=0, atol=1e-8):
        raise ValueError("field_direction must have unit norm")
    if (np.ndim(strength) != 0 or not np.isrealobj(strength)
            or not np.isfinite(strength) or strength < 0):
        raise ValueError("field_strength must be finite and nonnegative")
    return float(strength) * unit


def electric_dipole_potential(
    position_integrals: ArrayLike,
    field: ArrayLike | None = None,
    *,
    field_direction: ArrayLike | None = None,
    field_strength: float | None = None,
    spatial_orbitals: bool = True,
    mapping: str = "jordan_wigner",
    num_particles: tuple[int, int] | None = None,
    nuclear_dipole: ArrayLike = (0.0, 0.0, 0.0),
) -> PauliSum:
    r"""Build the length-gauge potential ``V = -E . mu`` in Hartree.

    ``mu = mu_nuc - sum_pq <p|r|q> a†_p a_q`` in atomic units, hence the
    **electronic potential has a positive sign**, ``+E . r``. The pulse is
    ``exp(-i tau_p V)``; this function does not include ``tau_p``.

    Parameters
    ----------
    position_integrals
        Three Hermitian matrices of shape ``(3, m, m)`` in Bohr, in the same
        orthonormal orbital basis and active space as the Hamiltonian. For
        molecular orbitals transform AO integrals with ``C.conj().T @ r @ C``.
    field
        Real Cartesian electric field in atomic units; its magnitude is kept.
        Defaults to (0, 0, 1) when no directional parameters are supplied.
    field_direction, field_strength
        Alternative input: a user-defined Cartesian unit vector and a finite,
        nonnegative magnitude in atomic units. Supply both, without ``field``.
        The coupled field is ``field_strength * field_direction``. Non-unit
        directions are rejected, including the zero vector.
    spatial_orbitals
        If True (default), duplicate the spatial matrix into alpha and beta
        spin blocks, matching Mandacaru's orbital ordering. Otherwise the
        supplied matrices already act on spin orbitals.
    mapping, num_particles
        Fermion-to-qubit encoding and, for reduced parity, particle counts.
        Hamiltonian, perturbation, and state must use the same encoding.
        Any further symmetry tapering must also be applied consistently;
        dipole transitions can leave a symmetry sector used for the ground state.
    nuclear_dipole
        Fixed nuclear dipole (and optionally frozen-core contribution) in
        atomic units. Adds ``-E . mu_nuc`` times identity, a global phase.

    Notes
    -----
    Applicable to molecules, finite solid clusters, and explicitly supplied
    finite-space position matrices. The position operator is not periodic:
    this builder does not construct a bulk Bloch/Berry-phase or velocity-gauge
    coupling from periodic coordinates.
    """
    if field_direction is not None or field_strength is not None:
        if field is not None or field_direction is None or field_strength is None:
            raise ValueError("supply both field_direction and field_strength, without field")
        field = electric_field_vector(field_direction, field_strength)
    elif field is None:
        field = (0.0, 0.0, 1.0)
    positions = np.asarray(position_integrals, dtype=complex)
    if (positions.ndim != 3 or positions.shape[0] != 3
            or positions.shape[1] == 0 or positions.shape[1] != positions.shape[2]):
        raise ValueError("position_integrals must have shape (3, m, m), m > 0")
    if not np.all(np.isfinite(positions)) or not np.allclose(
            positions, positions.conj().swapaxes(1, 2), rtol=0, atol=1e-12):
        raise ValueError("position_integrals must be finite and Hermitian")
    vectors = []
    for name, value in (("field", field), ("nuclear_dipole", nuclear_dipole)):
        array = np.asarray(value)
        if array.shape != (3,) or not np.isrealobj(array) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be a finite real Cartesian vector")
        vectors.append(array.astype(float))
    electric_field, nuclear = vectors
    one_body = np.einsum("a,apq->pq", electric_field, positions)
    if spatial_orbitals:
        one_body = np.kron(np.eye(2), one_body)
    mapping = resolve_mapping(mapping)
    fermion = Fermion.from_integrals(one_body, tol=0)
    fermion = fermion + Fermion({(): -float(electric_field @ nuclear)})
    qubit = fermion.map_to_qubits(mapping, num_particles=num_particles)
    # Mapping an identically zero operator otherwise loses its register width.
    width = one_body.shape[0] - (2 if mapping == "parity_reduced" else 0)
    return PauliSum(qubit.terms, num_qubits=qubit.num_qubits or width)
