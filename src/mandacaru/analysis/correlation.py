"""Owned Python result views for the native closed-shell MP2/FNO implementation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
IMAGINARY_TOLERANCE=1e-8
def _real(array, what: str) -> np.ndarray:
    """``array`` as a real array, refusing a non-negligible imaginary part.

    Casting has to happen *before* an eigendecomposition, not after it: a
    Hermitian matrix with a degenerate pair (the two pi orbitals of a linear
    molecule) lets LAPACK return eigenvectors with any complex phase, e.g.
    ``1j * v``, and the real part of that is zero rather than ``v``.
    """
    array = np.asarray(array)
    if not np.iscomplexobj(array):
        return array
    imaginary = float(np.max(np.abs(array.imag))) if array.size else 0.0
    if imaginary > IMAGINARY_TOLERANCE:
        raise ValueError(
            f"the {what} has an imaginary part of {imaginary:.2e} Ha: the "
            f"orbitals are genuinely complex, and the MP2 natural orbitals are "
            f"built as a real rotation.  Use the 'energy' active_space method for "
            f"a complex basis.")
    return np.ascontiguousarray(array.real)

@dataclass
class MP2Result:
    r"""MP2 correlation energy and the natural orbitals of its density.

    Attributes
    ----------
    correlation_energy : float
        :math:`E^{(2)}` in Hartree.  Always negative.
    occupied_occupations : ndarray
        ``2 + diag(D_oo)`` per **canonical** occupied orbital, in canonical
        order -- how much charge second-order correlation takes out of each one.
        Reported, not acted on: the occupied block is deliberately left alone
        (see :func:`mp2_natural_orbitals`), so these are diagonal elements in
        the semicanonical basis rather than eigenvalues.
    virtual_occupations : ndarray
        Occupation numbers of the virtual natural orbitals, the eigenvalues of
        ``D_vv``, in **descending** order -- the most populated orbital first,
        which is the order an active space should keep them in.
    rotation : ndarray
        ``(M, M)`` orthogonal matrix taking the input MO basis to the frozen
        natural orbitals: the identity on the occupied block (up to
        semicanonicalization) and the virtual natural orbitals, ordered to match
        ``virtual_occupations``, on the virtual block.  Applying it to the MO
        coefficients gives those orbitals in the underlying basis.
    occupied_density : ndarray
        The spin-summed density's occupied block, ``2 + D_oo`` on its
        diagonal, in the basis ``rotation`` leaves the occupied orbitals in.
        Its diagonal is ``occupied_occupations``; a caller that wants occupied
        natural orbitals diagonalizes the part of it it is free to rotate
        (:func:`~mandacaru.algorithms.active_space.resolve_active_space`
        does, over the active doubly occupied orbitals only).
    """

    correlation_energy: float
    occupied_occupations: np.ndarray
    virtual_occupations: np.ndarray
    rotation: np.ndarray
    occupied_density: np.ndarray | None = None

    @property
    def occupations(self) -> np.ndarray:
        """Every occupation number, occupied block then virtual block."""
        return np.concatenate([self.occupied_occupations,
                               self.virtual_occupations])

def mp2_natural_orbitals(h_mo: np.ndarray, eri_mo: np.ndarray, n_occ: int, *, resources=None) -> MP2Result:
    """MP2 correlation energy and the **frozen** natural orbitals of its density.

    Semicanonicalizes the input basis, forms the second-order density and
    diagonalizes its **virtual-virtual block only**, ordering the result by
    descending occupation.  The returned ``rotation`` composes both steps, so it
    takes the input MO basis straight to those orbitals.

    The occupied block is left in canonical order on purpose, and this is a
    choice worth stating rather than an omission.  Two things depend on it.
    The reference determinant and the occupation the ansatz prepares are
    defined by *which* orbitals are occupied, and a rotation inside the
    occupied block leaves the determinant invariant but renames its orbitals --
    so anything that identifies a core orbital by its index, ``"frozen":
    "auto"`` above all (the noble-gas core is "the lowest so many MOs", which
    is only the chemical core while the orbitals are energy-ordered), would
    silently freeze the wrong ones.  And truncating the occupied space is a far
    coarser approximation than truncating the virtual space, so it belongs to
    the explicit frozen-core controls and not to an automatic selector.  This
    is the classic frozen-natural-orbital scheme, and it is deliberately the
    conservative half of it.
    """
    from .._native import Context
    return MP2Result(**Context(**dict(resources or {})).mp2(np.ascontiguousarray(h_mo, dtype=complex), np.ascontiguousarray(eri_mo, dtype=complex), n_occ))

def mp2_natural_orbitals_from_block(oovv: np.ndarray, energies: np.ndarray, n_occ: int, canonical: np.ndarray, *, resources=None) -> MP2Result:
    """:func:`mp2_natural_orbitals` from the only integrals it reads.

    ``oovv`` is :math:`\\langle ij|ab\\rangle` (``(o, o, v, v)``, i.e. the
    chemists' :math:`(ia|jb)`) in the semicanonical basis that ``canonical``
    takes the input orbitals to, and ``energies`` its orbital energies
    (:func:`semicanonical_rotation`).  :math:`o^2 v^2` numbers, which an
    integral-direct build forms without the :math:`M^4` tensor
    (:meth:`~mandacaru.algorithms.orbital_integrals.DirectOrbitalIntegrals.mp2_natural_orbitals`);
    the arithmetic is :func:`mp2_energy` and :func:`mp2_density` on it.
    """
    from .._native import Context
    return MP2Result(**Context(**dict(resources or {})).mp2_from_block(np.ascontiguousarray(_real(oovv, 'MP2 block')), np.ascontiguousarray(_real(energies, 'orbital energies')), n_occ, np.ascontiguousarray(_real(canonical, 'canonical rotation'))))
