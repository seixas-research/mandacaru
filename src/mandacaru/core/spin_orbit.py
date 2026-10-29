"""Python projector labels and array views for Rust spin-orbit models.

The angular L.S operator and all radial-projector contractions execute in the
native scientific core. Spin-major ordering places alpha orbitals before beta.
"""
from __future__ import annotations
import numpy as np
from .. import _native

ALPHA, BETA = 0, 1


def ls_matrix(l: int, *, real: bool) -> np.ndarray:
    """Native L.S in real or complex spherical harmonics, ordered by m+l."""
    return _native.Context(threads=1).spin_orbit_angular(l, real=real)


def _channels(projectors, blocks):
    labels = {}
    for position, projector in enumerate(projectors):
        key = (int(projector.atom_index), int(projector.l))
        group = labels.setdefault(key, {})
        label = (int(projector.m), int(projector.index))
        if label in group:
            raise ValueError("duplicate spin-orbit projector label")
        group[label] = position
    channels = []
    for (atom, l), positions in labels.items():
        if (atom, l) not in blocks or l == 0:
            continue
        radial = sorted({i for m, i in positions})
        expected = [(m, i) for m in range(-l, l + 1) for i in radial]
        if set(expected) != positions.keys():
            raise ValueError("spin-orbit channel needs every angular and radial projector")
        indices = [positions[label] for label in expected]
        real = bool(projectors[indices[0]].real)
        if any(bool(projectors[i].real) != real for i in indices):
            raise ValueError("spin-orbit channel uses inconsistent harmonic conventions")
        coupling = np.ascontiguousarray(blocks[atom, l], dtype=complex)
        if coupling.shape != (len(radial), len(radial)):
            raise ValueError("spin-orbit block dimensions must match its radial projectors")
        channels.append((l, real, indices, coupling))
    return channels


def spin_orbit_pair(left, right, projectors, blocks, *, context=None) -> np.ndarray:
    """Native left D_SO (L.S) right^dagger, including unequal bra/ket sets."""
    context = _native.Context(threads=1) if context is None else context
    return context.spin_orbit_pair(
        np.ascontiguousarray(left, dtype=complex),
        np.ascontiguousarray(right, dtype=complex), _channels(projectors, blocks),
    )


def spin_orbit_one_body(projections, projectors, blocks, *, context=None) -> np.ndarray:
    """Native Hermitian SOC matrix in the spin-major Bloch or molecular basis."""
    context = _native.Context(threads=1) if context is None else context
    values = np.ascontiguousarray(projections, dtype=complex)
    return context.spin_orbit_pair(
        values, values, _channels(projectors, blocks), hermitian=True,
    )


def conserves_spin_projection(fermion, tolerance: float = 1e-12) -> bool:
    r"""Whether a spin-blocked :class:`~mandacaru.core.mapping.Fermion`
    operator commutes with :math:`S_z`: no term changes the number of alpha
    electrons (modes ``0..M-1`` of a ``2M``-mode register, the layout of
    :func:`~mandacaru.core.hamiltonian.spin_block_integrals`).

    Read off the operator itself, so it also holds for a Hamiltonian loaded
    from a file.
    """
    n_modes = fermion.n_modes()
    half = int(n_modes) // 2
    for term, coefficient in fermion.terms.items():
        if abs(coefficient) <= tolerance:
            continue
        change = sum((1 if dagger else -1) for mode, dagger in term
                     if mode < half)
        if change:
            return False
    return True


def breaks_spin_symmetry(h_spin_orbit, tolerance: float = 1e-12) -> bool:
    """Whether the term actually couples the two spin blocks.

    Zero for a molecule of scalar-relativistic atoms, and zero for s-only
    atoms (hydrogen), where the ``L . S`` operator vanishes identically -- so a
    calculation that asks for spin-orbit coupling and gets nothing is told so
    rather than paying for a broken :math:`S_z`.
    """
    h = np.asarray(h_spin_orbit)
    M = h.shape[0] // 2
    return bool(np.max(np.abs(h[:M, M:])) > tolerance)
