"""Freeze item 4: the per-species DFT constant ``one_center_energy + core_correction_offset`` of the periodic route.

``integrations/paw.py::configure`` passes ``constant = sum(d.one_center_energy + core_correction_offset(d))`` to the crystal
solver; the Rust molecular route has the same quantity as ``Species::dft_constant`` (``paw/molecular/data.rs``).  No existing
binding reports that Rust field (``MolecularModel.constant_energy`` is 0 for a PAW molecule: measured, H lda-sr 0.0 against
0.0018738 Ha), so the Rust comparison belongs to the Rust test that reads this file; here the frozen value is the Python one,
with two checks that do not need Rust:

* ``terms["constant"]`` of a crystal snapshot equals the sum over its atoms (checked per case by the results driver);
* an independent recomputation of ``core_correction_offset`` from the stored radial arrays with libxc (``pyscf``) and a
  MacDonald-Vosko exchange factor written from its definition (every shipped dataset has ``relativistic_exchange``), using
  the same trapezoid rule, so only the functional and the potential differ.  GGA: the divergence in ``v_xc`` is integrated by
  parts, ``-2 int rho_v div(v_sigma grad rho_t) = 2 int v_sigma grad rho_v . grad rho_t``, which needs no second derivative.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

#: ``(folder, symbols)`` of every frozen species.
SPECIES = (("lda-sr", ("H", "Al", "Si")), ("pbe-sr", ("H", "Al", "Si")), ("lda-fr", ("Si",)))
LIBXC = {"lda": ("LDA_X", "LDA_C_PZ"), "pbe": ("GGA_X_PBE", "GGA_C_PBE")}


def macdonald_vosko(rho, c):
    """``Phi_E(rho)``, the relativistic factor of the exchange energy per electron (MacDonald and Vosko 1979), written
    here from its definition ``1 - 3/2 F^2`` with ``F = sqrt(1+b^2)/b - asinh(b)/b^2``, ``b = (3 pi^2 rho)^(1/3)/c``."""
    import numpy as np
    b = (3.0 * np.pi ** 2 * np.maximum(rho, 1e-30)) ** (1.0 / 3.0) / c
    b = np.maximum(b, 1e-3)               # below this Phi_E = 1 - 2/3 b^2 to 7e-13; the clip changes it by < 1e-6 b^2
    f = np.sqrt(1.0 + b * b) / b - np.arcsinh(b) / (b * b)
    return 1.0 - 1.5 * f * f


def independent_offset(dataset, relativistic=None) -> dict:
    """``bare - dressed`` of ``core_correction_offset`` with libxc and a relativistic exchange factor of its own;
    ``value`` is ``None`` when the dataset has no core correction (the offset is exactly 0)."""
    import numpy as np
    from pyscf.dft import libxc
    from mandacaru.basis.relativity import SPEED_OF_LIGHT
    from mandacaru.integrals.exchange_correlation import xc_core_density
    core = xc_core_density(dataset)
    if core is None or getattr(dataset, "valence_density", None) is None:
        return dict(value=None, reason="no core correction: the offset is exactly 0")
    r = np.asarray(dataset.r, dtype=float)
    valence = np.asarray(dataset.valence_density, dtype=float)
    functional = dataset.xc or "lda"
    if relativistic is None:
        relativistic = bool(getattr(dataset, "relativistic_exchange", False))
    names = LIBXC[functional]
    shell = 4.0 * np.pi * r * r

    def part(name, total, d_total):
        """``(f = rho e, df/drho, df/dsigma)`` of one libxc component."""
        floor = np.maximum(total, 1e-30)
        if functional == "lda":
            exc, vxc = libxc.eval_xc(name, floor, spin=0, deriv=1)[:2]
            return floor * exc, vxc[0], np.zeros_like(r)
        grid = np.vstack([floor, d_total, np.zeros_like(r), np.zeros_like(r)])
        exc, vxc = libxc.eval_xc(name, grid, spin=0, deriv=1)[:2]
        return floor * exc, vxc[0], vxc[1]

    def terms(total):
        d_total = np.gradient(total, r)
        f_x, dx_drho, dx_dsigma = part(names[0], total, d_total)
        f_c, dc_drho, dc_dsigma = part(names[1], total, d_total)
        if relativistic:
            phi = macdonald_vosko(total, SPEED_OF_LIGHT)
            step = 1e-5 * np.maximum(total, 1e-30)
            dphi = (macdonald_vosko(total + step, SPEED_OF_LIGHT) - macdonald_vosko(total - step, SPEED_OF_LIGHT)) / (2 * step)
            dx_drho = phi * dx_drho + f_x * dphi
            dx_dsigma = phi * dx_dsigma
            f_x = phi * f_x
        energy = np.trapezoid((f_x + f_c) * shell, r)
        d_valence = np.gradient(valence, r)
        # int rho_v v_xc, with the divergence of v_xc integrated by parts (gradient part: 2 v_sigma grad rho_v . grad rho_t)
        v_term = np.trapezoid((valence * (dx_drho + dc_drho) + 2.0 * (dx_dsigma + dc_dsigma) * d_valence * d_total) * shell, r)
        return float(energy - v_term)

    bare, dressed = terms(valence), terms(valence + core)
    return dict(value=bare - dressed, bare=bare, dressed=dressed, libxc=list(names), relativistic_exchange=relativistic)


def constants() -> dict:
    """``{"folder/Symbol": {...}}`` for every frozen species (Hartree)."""
    from freeze_stamp import sha256_file
    from mandacaru.integrals.exchange_correlation import core_correction_offset
    from mandacaru.pseudopotentials.paw import get_paw
    root = Path(os.environ["MANDACARU_PAW_PATH"])
    out = {}
    for folder, symbols in SPECIES:
        for symbol in symbols:
            d = get_paw(symbol, folder)
            offset = float(core_correction_offset(d))
            record = dict(symbol=symbol, folder=folder, file=str(root / folder / f"{symbol}.parquet"),
                          sha256=sha256_file(root / folder / f"{symbol}.parquet"), dataset_xc=d.xc,
                          relativistic_exchange=bool(getattr(d, "relativistic_exchange", False)),
                          nlcc_applied=bool((getattr(d, "nlcc", None) or {}).get("applied")),
                          valence_charge=float(d.valence_charge), one_center_energy=float(d.one_center_energy),
                          core_correction_offset=offset, constant=float(d.one_center_energy) + offset)
            check = independent_offset(d)
            record["independent_offset"] = check
            if check["value"] is not None:
                record["independent_offset_difference"] = check["value"] - offset
                # control: without the relativistic exchange factor the check must disagree, or it could not see it
                control = independent_offset(d, relativistic=False)["value"]
                record["control_nonrelativistic_difference"] = control - offset
            out[f"{folder}/{symbol}"] = record
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(constants(), indent=1, sort_keys=True))
