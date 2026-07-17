# -*- coding: utf-8 -*-
# file: utils/logging.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Structured runtime logging for ADAPT-VQE (the ``output.txt`` protocol).

:class:`AdaptOutputLogger` writes a human-readable, machine-parseable trace of an
ADAPT-VQE run to a plain text file (``output.txt`` by default) *as the loop
runs* -- every block is flushed immediately, so a reader can follow the
optimization live.  The file has three kinds of section:

* a **metadata / initialization** block with the initial geometry and the
  explicit unit-cell parameters;
* an **optimization setup** block naming the classical optimizer and the
  reference (Hartree-Fock) energy the VQE starts from;
* one **iteration** block per accepted ADAPT operator, listing the pool
  operators as explicit Pauli strings, the screening gradient magnitude of each,
  the operator selected to join the ansatz, and the parameterized circuit's
  expressivity score :math:`E` (KL divergence from the Haar distribution).

The format uses ``KEY: value`` lines and fixed section banners so it can be
parsed by simple line scanning (see :func:`parse_output` for a reference reader
used by the tests).
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

_BANNER = "=" * 72
_RULE = "-" * 72


def _format_pauli(generator, atol: float = 1e-9) -> str:
    """Explicit Pauli-string form of an anti-Hermitian generator ``A``.

    Renders ``sum_k c_k * P_k`` with the complex coefficients of the pool
    operator's :class:`~carcara.core.mapping.PauliSum`, e.g.
    ``+0.5j XYZI  -0.5j YXZI``.  Terms are sorted for a stable, diffable order.
    """
    terms = generator.simplify(atol).terms
    if not terms:
        return "0"
    parts = []
    for label, coeff in sorted(terms.items()):
        c = complex(coeff)
        if abs(c.imag) < atol:
            cstr = f"{c.real:+.6g}"
        elif abs(c.real) < atol:
            cstr = f"{c.imag:+.6g}j"
        else:
            cstr = f"({c.real:+.6g}{c.imag:+.6g}j)"
        parts.append(f"{cstr} {label}")
    return "  ".join(parts)


class AdaptOutputLogger:
    """Append-only writer for the ADAPT-VQE ``output.txt`` protocol.

    Parameters
    ----------
    path : str
        Destination file (overwritten at construction; ``"output.txt"`` by
        convention).
    """

    def __init__(self, path: str = "output.txt"):
        self.path = path
        # Truncate any previous run and keep the handle open for live appends.
        self._fh = open(path, "w", encoding="utf-8")

    # -- low-level helpers ------------------------------------------------- #

    def _emit(self, *lines: str) -> None:
        for line in lines:
            self._fh.write(line + "\n")
        self._fh.flush()

    # -- metadata / initialization block ----------------------------------- #

    def write_metadata(self, symbols: Sequence[str] | None = None,
                        positions=None, cell=None,
                        units: str = "Angstrom", title: str = "ADAPT-VQE run",
                        extra: dict | None = None) -> None:
        """Write the initial geometry and the explicit unit-cell parameters.

        Parameters
        ----------
        symbols : sequence of str, optional
            Chemical symbols, one per atom.
        positions : (N, 3) array_like, optional
            Cartesian coordinates in ``units``.
        cell : (3, 3) array_like, optional
            Lattice/cell tensor (rows are lattice vectors) in ``units``.  ``None``
            (or an all-zero matrix) means a non-periodic molecule.
        units : str
            Length unit label for the geometry and cell.
        title : str
            Header title.
        extra : dict, optional
            Additional ``KEY: value`` metadata lines.
        """
        self._emit(_BANNER, f" {title}", _BANNER, "", "[METADATA]")
        if extra:
            for key, value in extra.items():
                self._emit(f"{key}: {value}")

        # Initial geometry.
        self._emit(f"units: {units}")
        if symbols is not None and positions is not None:
            positions = np.asarray(positions, dtype=float)
            self._emit(f"n_atoms: {len(symbols)}", "geometry:")
            for sym, (x, y, z) in zip(symbols, positions):
                self._emit(f"  {sym:<3s} {x:16.10f} {y:16.10f} {z:16.10f}")
        else:
            self._emit("geometry: (not provided)")

        # Explicit unit-cell parameters.
        cell = None if cell is None else np.asarray(cell, dtype=float)
        if cell is not None and np.any(cell):
            self._emit("cell_present: True", "cell_vectors:")
            for i, vec in enumerate(cell):
                self._emit(f"  a{i + 1} = "
                           f"[{vec[0]:14.10f} {vec[1]:14.10f} {vec[2]:14.10f}]")
            a, b, c, alpha, beta, gamma = _cell_parameters(cell)
            self._emit(f"cell_lengths: a={a:.10f} b={b:.10f} c={c:.10f}",
                       f"cell_angles: alpha={alpha:.6f} beta={beta:.6f} "
                       f"gamma={gamma:.6f}")
        else:
            self._emit("cell_present: False", "cell_vectors: (non-periodic)")
        self._emit("")

    # -- optimization setup block ------------------------------------------ #

    def write_optimizer_setup(self, optimizer_method: str,
                              reference_energy: float,
                              gradient_tol: float | None = None,
                              max_iterations: int | None = None,
                              extra: dict | None = None) -> None:
        """Write the classical optimizer and the pre-loop (reference) results.

        The *final classical ansatz optimization result leading into the VQE
        runtime* is, before the first operator is added, the Hartree-Fock
        reference energy of the empty ansatz -- logged here as the loop's
        starting point.
        """
        self._emit("[OPTIMIZATION SETUP]",
                   f"classical_optimizer: {optimizer_method}")
        if max_iterations is not None:
            self._emit(f"max_iterations: {max_iterations}")
        if gradient_tol is not None:
            self._emit(f"gradient_tol: {gradient_tol:g}")
        self._emit(f"reference_energy_Ha: {reference_energy:.10f}",
                   "initial_ansatz: |HF> (0 parameters)")
        if extra:
            for key, value in extra.items():
                self._emit(f"{key}: {value}")
        self._emit("")

    # -- per-iteration block ----------------------------------------------- #

    def write_iteration(self, iteration: int, pool_operators: Sequence,
                        gradients: Iterable[float], selected_index: int,
                        expressivity: float | None, energy: float,
                        num_parameters: int, metrics: Any = None) -> None:
        """Append one ADAPT iteration's tracked metrics, in order.

        Parameters
        ----------
        iteration : int
            1-based macro-iteration index.
        pool_operators : sequence of PoolOperator
            The full operator pool at this step.
        gradients : iterable of float
            Screening gradient of each pool operator (same order as
            ``pool_operators``); their magnitudes are logged.
        selected_index : int
            Index (into ``pool_operators``) of the operator chosen for the
            ansatz.
        expressivity : float or None
            Expressivity score :math:`E` of the parameterized ansatz at this
            iteration (``None`` if not computed).
        energy : float
            Energy after the inner re-optimization (Hartree).
        num_parameters : int
            Number of parameters (operators) in the ansatz after this step.
        metrics : optional
            Object exposing ``cnot_count`` / ``depth`` (logged if present).
        """
        grads = [float(g) for g in gradients]
        selected = pool_operators[selected_index]
        self._emit(_RULE,
                   f"[ITERATION {iteration}]",
                   f"selected_operator: {selected.label}",
                   f"selected_operator_kind: {selected.kind}",
                   f"max_gradient: {abs(grads[selected_index]):.6e}")
        if expressivity is not None:
            self._emit(f"expressivity_E: {expressivity:.6f}")
        else:
            self._emit("expressivity_E: (not computed)")
        self._emit(f"energy_Ha: {energy:.10f}",
                   f"num_parameters: {num_parameters}")
        if metrics is not None and getattr(metrics, "cnot_count", None) is not None:
            self._emit(f"cnot_count: {metrics.cnot_count}",
                       f"circuit_depth: {metrics.depth}")

        # Pool operators (Pauli strings) with their gradient magnitudes.
        self._emit(f"pool_size: {len(pool_operators)}", "pool_operators:")
        for i, op in enumerate(pool_operators):
            flag = "  <== SELECTED" if i == selected_index else ""
            self._emit(f"  [{i:3d}] label={op.label} "
                       f"|grad|={abs(grads[i]):.6e}{flag}")
            self._emit(f"        pauli: {_format_pauli(op.generator)}")
        self._emit("")

    # -- footer / teardown ------------------------------------------------- #

    def write_summary(self, converged: bool, optimal_energy: float,
                      num_operators: int, extra: dict | None = None) -> None:
        """Write the closing summary block."""
        self._emit(_BANNER, "[SUMMARY]",
                   f"converged: {converged}",
                   f"optimal_energy_Ha: {optimal_energy:.10f}",
                   f"num_operators: {num_operators}")
        if extra:
            for key, value in extra.items():
                self._emit(f"{key}: {value}")
        self._emit(_BANNER)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "AdaptOutputLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _cell_parameters(cell: np.ndarray):
    """Return ``(a, b, c, alpha, beta, gamma)`` for a (3, 3) cell tensor.

    Lengths in the cell's own units; angles in degrees (alpha between b and c,
    beta between a and c, gamma between a and b -- crystallographic convention).
    """
    cell = np.asarray(cell, dtype=float)
    a_vec, b_vec, c_vec = cell[0], cell[1], cell[2]
    a, b, c = (np.linalg.norm(a_vec), np.linalg.norm(b_vec),
               np.linalg.norm(c_vec))

    def angle(u, v):
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu == 0 or nv == 0:
            return 0.0
        cosang = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
        return float(np.degrees(np.arccos(cosang)))

    return (float(a), float(b), float(c),
            angle(b_vec, c_vec), angle(a_vec, c_vec), angle(a_vec, b_vec))


def parse_output(path: str) -> dict:
    """Reference parser for an ADAPT ``output.txt`` (used by the tests).

    Scans the file line by line and returns a dict with the metadata, the
    optimization setup, and a list of per-iteration records -- demonstrating that
    the protocol is machine-parseable as it is written.
    """
    result: dict[str, Any] = {"metadata": {}, "setup": {}, "iterations": []}
    section = None
    current: dict[str, Any] | None = None

    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            stripped = raw.strip()
            if stripped == "[METADATA]":
                section = "metadata"
                continue
            if stripped == "[OPTIMIZATION SETUP]":
                section = "setup"
                continue
            if stripped.startswith("[ITERATION"):
                section = "iteration"
                current = {"index": int(stripped.split()[1].rstrip("]")),
                           "pool": []}
                result["iterations"].append(current)
                continue
            if stripped == "[SUMMARY]":
                section = "summary"
                result["summary"] = {}
                continue

            if section in ("metadata", "setup") and ":" in stripped \
                    and not stripped.startswith(("a1", "a2", "a3")):
                key, _, value = stripped.partition(":")
                result[section][key.strip()] = value.strip()
            elif section == "summary" and ":" in stripped:
                key, _, value = stripped.partition(":")
                result["summary"][key.strip()] = value.strip()
            elif section == "iteration" and current is not None:
                if stripped.startswith("expressivity_E:"):
                    current["expressivity_E"] = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("selected_operator:"):
                    current["selected_operator"] = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("energy_Ha:"):
                    current["energy_Ha"] = float(stripped.split(":", 1)[1])
                elif stripped.startswith("pauli:"):
                    current["pool"].append(stripped.split(":", 1)[1].strip())
    return result
