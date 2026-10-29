"""Canonical method capabilities and scientific setup specifications."""

from __future__ import annotations
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Capability:
    quantum: bool
    forces: bool
    periodic: bool = False


METHODS = {
    "rhf": Capability(False, True),
    "uhf": Capability(False, True),
    "ghf": Capability(False, False),
    "dft": Capability(False, True, True),
    "vqe": Capability(True, True),
    "adapt-vqe": Capability(True, True),
    "subspace-vqe": Capability(True, False),
    "subspace-adapt-vqe": Capability(True, False),
    "mcas-vqe": Capability(True, True),
    "valqa": Capability(True, True),
}
from ._native import FUNCTIONALS as _FUNCTIONALS

FUNCTIONALS = tuple(_FUNCTIONALS)
MAPPINGS = ("jordan_wigner", "parity", "bravyi_kitaev", "parity_reduced")


def _element_key(value):
    if value == "*":
        return value
    if isinstance(value, int) or str(value).isdigit():
        return str(int(value))
    from .models.elements import ELEMENTS

    try:
        return str(ELEMENTS.index(value))
    except ValueError:
        raise ValueError(f"unknown element {value!r}") from None


def basis_specification(value):
    spec = {"name": value} if isinstance(value, str) else dict(value)
    if "name" not in spec:
        return {
            "name": "per_element",
            "elements": {
                _element_key(key): basis_specification(val) for key, val in spec.items()
            },
        }
    name = spec.pop("name")
    if name == "HAO":
        if set(spec) - {"virtual_orbitals"}:
            raise TypeError("HAO accepts virtual_orbitals")
        return dict(name=name, **spec)
    if name == "NAO":
        from .models.radial import (
            DEFAULT_NAO_SIZE,
            DEFAULT_SPLIT_NORM,
            resolve_zeta,
            resolve_split_scheme,
        )

        if set(spec) - {"size", "r_c", "n_grid", "energy_shift", "zeta_split"}:
            raise TypeError(
                "NAO accepts size, r_c, n_grid, energy_shift and zeta_split"
            )
        zetas, polarization = resolve_zeta(spec.get("size", DEFAULT_NAO_SIZE))
        split, tails = resolve_split_scheme(spec.get("zeta_split"))
        radius = spec.get("r_c")
        if radius is None:
            from math import pi, sqrt
            from .units import EV_TO_HARTREE

            shift = float(spec.get("energy_shift", 0.03))
            if not shift > 0:
                raise ValueError("energy_shift must be positive")
            radius = pi / sqrt(2 * shift * EV_TO_HARTREE)
        return dict(
            name=name,
            r_c=radius,
            n_grid=spec.get("n_grid", 2000),
            n_zeta=zetas,
            n_polarization=polarization,
            split_norm=DEFAULT_SPLIT_NORM if split is None else split,
            tail_norms=tails,
        )
    if name == "NAO-AE":
        from .models.radial import NAO_AE_DEFAULTS
        from .units import ANGSTROM_TO_BOHR

        if set(spec) - NAO_AE_DEFAULTS.keys():
            raise TypeError(f"NAO-AE accepts {tuple(NAO_AE_DEFAULTS)}")
        result = dict(NAO_AE_DEFAULTS, **spec)
        result["onset"] *= ANGSTROM_TO_BOHR
        result["width"] *= ANGSTROM_TO_BOHR
        result["wall_radius"] = result["onset"] + result["width"]
        return dict(name=name, **result)
    if name in ("per_element", "gaussian"):
        return dict(name=name, **spec)
    if spec:
        raise TypeError("a named Gaussian recipe does not accept extra basis options")
    from .models.basis import parse_basis_name

    return {"name": "gaussian", "recipe": asdict(parse_basis_name(name))}


def available_methods():
    return tuple(METHODS)


@dataclass(frozen=True)
class QubitEstimate:
    n_qubits: int
    n_spatial_orbitals: int
    mapping: str


def estimate_qubits(atoms, *, basis="HAO", mapping="jordan_wigner", resources=None):
    """Count the native basis without constructing molecular integrals."""
    import json
    from . import _native
    from .structure import geometry_value

    if mapping not in MAPPINGS:
        raise ValueError(f"expected a canonical mapping in {MAPPINGS}")
    context = _native.Context(**dict(resources or {}))
    count = context.basis_size(
        geometry_value(atoms).native(),
        basis_spec=json.dumps(basis_specification(basis)),
    )
    return QubitEstimate(
        2 * count - (2 if mapping == "parity_reduced" else 0), count, mapping
    )
