"""The calculator's Berry-phase requests: argument validation and the hand-off to the native crystal.

The numbers are Rust's (``periodic::berry``); this module checks what the user asked for,
finds the converged crystal and wraps the native result in its view (:mod:`mandacaru.analysis.berry`).
"""

from __future__ import annotations

import operator

from ..analysis.berry import (AnomalousHall, BerryCurvature, BerryPhase, BornCharges,
                              PiezoelectricTensor, Polarization, WannierChargeCenterFlow,
                              Z2Invariant)


def _integer(value):
    """``value`` as an ``int`` when it is an integer (NumPy integers included, booleans not)."""
    if isinstance(value, bool):
        return None
    try:
        return operator.index(value)
    except TypeError:
        return None


def _axis(value, name):
    index = _integer(value)
    if index not in (0, 1, 2):
        raise ValueError(
            f"{name} names a reciprocal lattice vector G_j by its index 0, 1 or 2; got {value!r}"
        )
    return index


def _size(calculation, kpts):
    """The ``(n1, n2, n3)`` mesh of ``kpts`` (a size or ``{"size": ...}``), or the calculation's."""
    if kpts is None:
        spec = getattr(calculation, "_kpts_spec", None) or {}
        kpts = spec.get("size", (1, 1, 1))
    elif isinstance(kpts, dict):
        if "size" not in kpts:
            raise ValueError("a Berry k mesh given as a mapping needs its 'size'")
        kpts = kpts["size"]
    try:
        size = tuple(kpts)
    except TypeError:
        raise ValueError(f"a Berry k mesh is three positive integers; got {kpts!r}") from None
    size = tuple(_integer(n) for n in size)
    if len(size) != 3 or any(n is None or n < 1 for n in size):
        raise ValueError(f"a Berry k mesh is three positive integers; got {kpts!r}")
    return size


def _bands(bands):
    """``None`` (the occupied bands), a band index, or a half-open ``(lo, hi)`` pair."""
    if bands is None:
        return None
    index = _integer(bands)
    if index is not None:
        if index < 0:
            raise ValueError(f"a band index is nonnegative; got {bands!r}")
        return (index, index + 1)
    try:
        lo, hi = (_integer(b) for b in bands)
    except (TypeError, ValueError):
        raise ValueError(
            f"bands is None, a band index or a half-open (lo, hi) pair; got {bands!r}"
        ) from None
    if lo is None or hi is None or not 0 <= lo < hi:
        raise ValueError(f"a band group (lo, hi) needs integers 0 <= lo < hi; got {bands!r}")
    return (lo, hi)


def _offset(offset):
    if isinstance(offset, bool) or not isinstance(offset, (int, float)):
        raise ValueError(f"a plane offset is a fractional coordinate in [0, 1); got {offset!r}")
    offset = float(offset)
    if not 0.0 <= offset < 1.0:
        raise ValueError(f"a plane offset is a fractional coordinate in [0, 1); got {offset!r}")
    return offset


def _crystal(calculation):
    """The native periodic result of a converged DFT crystal, running the calculation if needed."""
    if calculation.capabilities.quantum:
        raise NotImplementedError(
            "Berry phases, curvature and polarization are those of Kohn-Sham Bloch states; a "
            f"quantum method on a crystal ({calculation.method!r}) solves its Born-von Karman "
            "supercell, whose correlated state has no Bloch bands: run method='dft'"
        )
    if calculation.result is None:
        atoms = calculation._attached_atoms
        if atoms is None:
            raise RuntimeError("run a periodic DFT calculation before asking for Berry phases")
        if not any(atoms.pbc) and calculation._load_crystal is None:
            raise ValueError(
                "Berry phases, curvature and polarization need a periodic crystal: a molecule "
                "has no Brillouin zone"
            )
        calculation.calculate(atoms, properties=("energy",))
    periodic = calculation.result.periodic
    if periodic is None:
        raise ValueError(
            "Berry phases, curvature and polarization need a periodic crystal: a molecule has no "
            "Brillouin zone"
        )
    return periodic


def berry_phase(calculation, direction, kpts=None, bands=None):
    direction = _axis(direction, "direction")
    size, band_range = _size(calculation, kpts), _bands(bands)
    raw = _crystal(calculation).berry_phase(direction, list(size), bands=band_range)
    return BerryPhase.from_native(raw, calculation._atomic_units)


def berry_curvature(calculation, plane, kpts=None, offset=0.0, bands=None):
    normal = _axis(plane, "plane")
    size, band_range, offset = _size(calculation, kpts), _bands(bands), _offset(offset)
    raw = _crystal(calculation).berry_curvature(normal, list(size), offset=offset, bands=band_range)
    return BerryCurvature.from_native(raw, calculation._atomic_units)


def polarization(calculation, kpts=None):
    size = _size(calculation, kpts)
    raw = _crystal(calculation).polarization(list(size))
    # The warning points at the user's get_polarization call.
    return Polarization.from_native(raw, calculation._atomic_units, stacklevel=3)


def anomalous_hall(calculation, kpts=None, stacklevel=3):
    size = _size(calculation, kpts)
    raw = _crystal(calculation).anomalous_hall(list(size))
    # The warnings point at the user's call.
    return AnomalousHall.from_native(raw, calculation._atomic_units, stacklevel=stacklevel)


def wannier_charge_center_flow(calculation, plane, direction, kpts=None, offset=0.0, bands=None):
    normal = _axis(plane, "plane")
    along = _axis(direction, "direction")
    if along == normal:
        raise ValueError(
            f"the strings of a flow run in the plane normal to G_{normal}: direction must differ "
            f"from plane; got {direction!r}"
        )
    size, band_range, offset = _size(calculation, kpts), _bands(bands), _offset(offset)
    raw = _crystal(calculation).wannier_flow(normal, along, list(size), offset=offset,
                                             bands=band_range)
    return WannierChargeCenterFlow.from_native(raw)


_Z2_METHODS = ("flow", "parity")


def z2_invariant(calculation, kpts=None, plane=None, offset=0.0, direction=None, method="flow"):
    if method not in _Z2_METHODS:
        raise ValueError(f"a Z2 method is one of {_Z2_METHODS}; got {method!r}")
    size = _size(calculation, kpts)
    if plane is None:
        if direction is not None or offset != 0.0:
            raise ValueError(
                "the Z2 indices of the whole crystal take no direction or offset: name a plane for "
                "the index of one plane"
            )
        raw = _crystal(calculation).z2_invariant(list(size), method=method)
        return Z2Invariant.from_native(raw, calculation._atomic_units)
    normal = _axis(plane, "plane")
    offset = _offset(offset)
    if offset not in (0.0, 0.5):
        raise ValueError(
            f"a Z2 index belongs to a time-reversal-invariant plane, offset 0 or 1/2; got {offset!r}"
        )
    half = int(offset == 0.5)
    if method == "parity":
        if direction is not None:
            raise ValueError("the parity of a plane has no string direction")
        whole = z2_invariant(calculation, kpts=kpts, method="parity")
        index = whole.planes[(normal, half)]
        return Z2Invariant(method="parity", indices=(index,), strong=None, weak=None,
                           planes={(normal, half): index}, parity=whole.parity,
                           inversion_center=whole.inversion_center,
                           spin_degenerate=whole.spin_degenerate, strings=whole.strings)
    along = (normal + 1) % 3 if direction is None else _axis(direction, "direction")
    flow = wannier_charge_center_flow(calculation, normal, along, kpts=kpts, offset=offset)
    if flow.z2 is None:
        raise NotImplementedError(
            "the Z2 index needs a time-reversal-invariant crystal; a collinear spin-polarized or "
            "magnetic spinor crystal has none"
        )
    return Z2Invariant(method="flow", indices=(flow.z2,), strong=None, weak=None,
                       planes={(normal, half): flow.z2}, flows=(flow,),
                       spin_degenerate=flow.spin_degenerate, strings=size)


_ASR_MODES = ("report", "impose")
_PIEZO_IONS = ("clamped", "relaxed")


def _positive_step(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive real number; got {value!r}")
    return float(value)


def born_effective_charges(calculation, displacement=0.01, kpts=None, richardson=False,
                           acoustic_sum_rule="report"):
    if not isinstance(richardson, bool):
        raise ValueError(f"richardson must be a bool; got {richardson!r}")
    if acoustic_sum_rule not in _ASR_MODES:
        raise ValueError(
            f"acoustic_sum_rule must be one of {_ASR_MODES}; got {acoustic_sum_rule!r}"
        )
    step = _positive_step(displacement, "displacement")
    size = _size(calculation, kpts)
    raw = _crystal(calculation).born_charges(
        list(size),
        displacement=step,
        richardson=richardson,
        impose_rule=(acoustic_sum_rule == "impose"),
        atomic_units=calculation._atomic_units,
    )
    return BornCharges.from_native(raw, calculation._atomic_units, stacklevel=3)


def piezoelectric_tensor(calculation, strain=0.005, kpts=None, ions="clamped",
                       richardson=False, displacement=0.01):
    if not isinstance(richardson, bool):
        raise ValueError(f"richardson must be a bool; got {richardson!r}")
    if ions not in _PIEZO_IONS:
        raise ValueError(f"ions must be one of {_PIEZO_IONS}; got {ions!r}")
    eta = _positive_step(strain, "strain")
    disp = _positive_step(displacement, "displacement")
    size = _size(calculation, kpts)
    dispersion_xc = calculation.xc if getattr(calculation, "_vdw", None) == "ts" else None
    raw = _crystal(calculation).piezoelectric(
        list(size),
        strain=eta,
        displacement=disp,
        richardson=richardson,
        impose_rule=False,
        relaxed_ions=(ions == "relaxed"),
        atomic_units=calculation._atomic_units,
        dispersion_xc=dispersion_xc,
    )
    return PiezoelectricTensor.from_native(raw, calculation._atomic_units, stacklevel=3)
