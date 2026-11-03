# -*- coding: utf-8 -*-
# file: pseudopotentials/record.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""The dataset record across the native boundary.

A PAW-LCAO / UPAW-LCAO record travels in the layout of its own file: a JSON payload (every
scalar, matrix and block) and the radial tables by column name (``r``, ``v_local``,
``ae_wave_l0_0`` ...).  ``mandacaru-core`` (``paw::record``) reads and writes the files, applies
the storage rule (stride, ten digits), and refuses what a file may not hold; this module
only turns that layout into the :class:`~.paw.PAWDataset` view the basis and the integral
engines read, and back.  No number is computed here.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .. import _native
from .defects import read_defects, warn_defects
from .io import DEFAULT_FORMAT, FORMAT_TAG, FORMAT_VERSION, _EXTENSION_FORMATS, resolve_format
from .native_context import context


def _channel(tables: dict, l: int, entry: dict, view, v_local, v_screened):
    waves = {name: [] for name in ("ae", "pseudo", "projector", "raw_projector")}
    i = 0
    while f"pseudo_wave_l{l}_{i}" in tables:
        waves["ae"].append(np.asarray(tables[f"ae_wave_l{l}_{i}"]))
        waves["pseudo"].append(np.asarray(tables[f"pseudo_wave_l{l}_{i}"]))
        waves["projector"].append(np.asarray(tables[f"projector_l{l}_{i}"]))
        waves["raw_projector"].append(np.asarray(tables[f"raw_projector_l{l}_{i}"]))
        i += 1
    q = np.asarray(entry["overlap_correction"], dtype=float)
    return view(
        l=l, n=int(entry["n"]), eigenvalue=float(entry["reference_energies"][0]),
        r_cut=float(entry["r_cut"]),
        coefficients=np.asarray(entry["wave_coefficients"][0], float),
        pseudo_radial=waves["pseudo"][0], v_screened=v_screened, v_ionic=v_local,
        occupation=float(entry["occupation"]),
        occupations=[float(o) for o in entry["occupations"]],
        norm_error=float(abs(q[0, 0])),
        reference_energies=[float(e) for e in entry["reference_energies"]],
        wavevectors=[np.asarray(v, float) for v in entry["wavevectors"]],
        wave_coefficients=[np.asarray(c, float) for c in entry["wave_coefficients"]],
        ae_waves=waves["ae"], pseudo_waves=waves["pseudo"],
        projectors=waves["projector"], raw_projectors=waves["raw_projector"],
        overlap_correction=q,
        norm_correction=np.asarray(entry["norm_correction"], dtype=float),
        kinetic_difference=np.asarray(entry["kinetic_difference"], dtype=float),
        potential_difference=np.asarray(entry["potential_difference"], dtype=float),
        coupling_screened=np.asarray(entry["coupling_screened"], dtype=float),
        coupling=np.asarray(entry["coupling"], dtype=float),
        vanderbilt=np.asarray(entry["vanderbilt"], dtype=float),
        duality_error=float(entry["duality_error"]),
        overlap_minimum=float(entry["overlap_minimum"]),
        asymmetry=float(entry["asymmetry"]),
        consistency_error=float(entry["consistency_error"]),
        residual_kinetic=[float(e) for e in entry["residual_kinetic"]],
        q_cut=float(entry["q_cut"]))


def dataset_from_record(payload: str, tables: dict, *, warn: bool = True):
    """The :class:`~.paw.PAWDataset` of a native record ``(payload, tables)``.

    The family is the record's own.  ``warn`` raises the
    :class:`~.defects.GhostStateWarning` of a record that carries defects.
    """
    from .paw import PAWChannel, PAWDataset

    meta = json.loads(payload)
    r = np.asarray(tables["r"])
    v_local = np.asarray(tables["v_local"])
    v_screened = np.asarray(tables["v_local_screened"])
    channels, projectors, coupling, coupling_screened = {}, {}, {}, {}
    overlap, kinetic, norm = {}, {}, {}
    for key, entry in meta["channels"].items():
        l = int(key)
        channel = _channel(tables, l, entry, PAWChannel, v_local, v_screened)
        channels[l] = channel
        projectors[l] = channel.projectors
        coupling[l], coupling_screened[l] = channel.coupling, channel.coupling_screened
        overlap[l] = channel.overlap_correction
        kinetic[l], norm[l] = channel.kinetic_difference, channel.norm_correction
    spin_orbit = {}
    for key, entry in meta["spin_orbit"].items():
        spin_orbit[int(key)] = {
            name: ([np.asarray(p, dtype=float) for p in value]
                   if name == "projectors" else np.asarray(value, dtype=float))
            for name, value in entry.items()}
    dataset = PAWDataset(
        symbol=meta["symbol"], atomic_number=int(meta["atomic_number"]),
        valence_charge=float(meta["valence_charge"]), r=r, channels=channels,
        v_local=v_local, projectors=projectors,
        valence_density=np.asarray(tables["valence_density"]),
        family=meta["family"], coupling=coupling,
        coupling_screened=coupling_screened, overlap_correction=overlap,
        norm_correction=norm, kinetic_difference=kinetic,
        v_local_screened=v_screened,
        core_density=np.asarray(tables["core_density"]),
        smooth_core_density=np.asarray(tables["smooth_core_density"]),
        r_cut_local=float(meta["r_cut_local"]), local_shift=float(meta["local_shift"]),
        compensation_radius=float(meta["compensation_radius"]),
        compensation_charge=float(meta["compensation_charge"]),
        hartree_screening=float(meta["hartree_screening"]),
        one_center_energy=float(meta["one_center_energy"]),
        energies={k: float(v) for k, v in meta["energies"].items()},
        q_cut=float(meta["q_cut"]), energy_offset=float(meta["energy_offset"]),
        norm_deficit=float(meta["norm_deficit"]), xc=str(meta["xc"]),
        relativity=str(meta["relativity"]),
        relativistic_exchange=bool(meta["relativistic_exchange"]),
        extra_l=int(meta["extra_l"]),
        frozen_subshells=tuple(tuple(int(x) for x in o)
                               for o in meta["frozen_subshells"]),
        semicore_subshells=tuple(tuple(int(x) for x in o)
                                 for o in meta["semicore_subshells"]),
        nlcc=dict(meta["nlcc"]), spin_orbit=spin_orbit,
        defects=read_defects(meta["defects"]))
    if warn:
        warn_defects(dataset, dataset.family)
    return dataset


def _matrix(value) -> list:
    return np.asarray(value, dtype=float).tolist()


def record_of(pp) -> tuple[str, dict]:
    """``(payload, tables)`` of a :class:`~.paw.PAWDataset`: the layout of its file.

    The native reader validates it as it would a file, so a view that was edited into
    something a file cannot hold raises there, not here.
    """
    tables = {name: np.ascontiguousarray(getattr(pp, name), dtype=np.float64)
              for name in ("r", "v_local", "v_local_screened", "valence_density",
                           "core_density", "smooth_core_density")}
    channels = {}
    for l, channel in sorted(pp.channels.items()):
        for i, (ae, ps, p, chi) in enumerate(zip(channel.ae_waves, channel.pseudo_waves,
                                                 channel.projectors,
                                                 channel.raw_projectors)):
            for name, values in (("ae_wave", ae), ("pseudo_wave", ps),
                                 ("projector", p), ("raw_projector", chi)):
                tables[f"{name}_l{l}_{i}"] = np.ascontiguousarray(values, dtype=np.float64)
        channels[str(l)] = {
            "n": int(channel.n), "r_cut": float(channel.r_cut),
            "occupation": float(channel.occupation),
            "occupations": [float(o) for o in channel.occupations],
            "reference_energies": [float(e) for e in channel.reference_energies],
            "wavevectors": [_matrix(q) for q in channel.wavevectors],
            "wave_coefficients": [_matrix(c) for c in channel.wave_coefficients],
            "coupling": _matrix(channel.coupling),
            "coupling_screened": _matrix(channel.coupling_screened),
            "overlap_correction": _matrix(channel.overlap_correction),
            "norm_correction": _matrix(channel.norm_correction),
            "kinetic_difference": _matrix(channel.kinetic_difference),
            "potential_difference": _matrix(channel.potential_difference),
            "vanderbilt": _matrix(channel.vanderbilt),
            "duality_error": float(channel.duality_error),
            "overlap_minimum": float(channel.overlap_minimum),
            "asymmetry": float(channel.asymmetry),
            "consistency_error": float(channel.consistency_error),
            "residual_kinetic": [float(e) for e in channel.residual_kinetic],
            "q_cut": float(channel.q_cut)}
    defects = pp.defects or {}
    record = {
        "format": FORMAT_TAG, "version": FORMAT_VERSION, "family": str(pp.family),
        "symbol": pp.symbol, "atomic_number": int(pp.atomic_number),
        "valence_charge": float(pp.valence_charge),
        "r_cut_local": float(pp.r_cut_local), "local_shift": float(pp.local_shift),
        "compensation_radius": float(pp.compensation_radius),
        "compensation_charge": float(pp.compensation_charge),
        "hartree_screening": float(pp.hartree_screening),
        "one_center_energy": float(pp.one_center_energy),
        "energies": {k: float(v) for k, v in pp.energies.items()},
        "q_cut": float(pp.q_cut), "energy_offset": float(pp.energy_offset),
        "norm_deficit": pp.norm_deficit, "xc": str(pp.xc),
        "relativity": str(pp.relativity),
        "relativistic_exchange": bool(pp.relativistic_exchange),
        "extra_l": int(pp.extra_l), "nlcc": dict(pp.nlcc or {}),
        "frozen_subshells": [list(o) for o in pp.frozen_subshells],
        "semicore_subshells": [list(o) for o in pp.semicore_subshells],
        "defects": {
            "ghosts": {str(l): float(e) for l, e in defects.get("ghosts", {}).items()},
            "phases": {str(l): [float(a), float(b)]
                       for l, (a, b) in defects.get("phases", {}).items()},
            **({"residuals": {str(l): float(e) for l, e in defects["residuals"].items()}}
               if defects.get("residuals") else {}),
            **({"s_miss": float(defects["s_miss"])}
               if defects.get("s_miss") is not None else {}),
            **({"unprojected": {str(l): float(e)
                                for l, e in defects["unprojected"].items()}}
               if defects.get("unprojected") else {}),
            **({"missing": {str(l): float(e) for l, e in defects["missing"].items()}}
               if defects.get("missing") else {}),
            **({"branch_ghosts": {str(k): float(e)
                                  for k, e in defects["branch_ghosts"].items()}}
               if defects.get("branch_ghosts") else {})},
        "spin_orbit": {
            str(l): {name: ([np.asarray(p, dtype=float).tolist() for p in value]
                            if name == "projectors" else _matrix(value))
                     for name, value in entry.items()}
            for l, entry in (pp.spin_orbit or {}).items()},
        "channels": channels}
    return json.dumps(record, allow_nan=False), tables


def read_dataset(path, *, warn: bool = True):
    """The dataset a file holds (Parquet or JSON, detected by content)."""
    path = os.fspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no such pseudopotential file: {path!r}")
    payload, tables = _native.paw_record_read(context(), path)
    return dataset_from_record(payload, tables, warn=warn)


def write_dataset(pp, path, *, stride: int = 1, format: str | None = None,
                  codec: str = "zstd") -> str:
    """Write ``pp`` to ``path`` and return it.

    The format is the one the extension names (else Parquet); ``stride`` keeps every
    ``stride``-th radial point and the tables are rounded to ten digits, whatever the
    stride (a library is written at :data:`~.io.STRIDE`).  The file is staged beside ``path``,
    read back, and moved into place only if the read-back is what was written.
    """
    path = os.fspath(path)
    if format is None:
        format = _EXTENSION_FORMATS.get(os.path.splitext(path)[1].lower(),
                                        DEFAULT_FORMAT)
    format = resolve_format(format)
    payload, tables = record_of(pp)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _native.paw_record_write(context(), path, payload, tables, stride=int(stride),
                             format=format, codec=codec)
    return path
