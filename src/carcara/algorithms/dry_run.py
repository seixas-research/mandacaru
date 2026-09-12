# -*- coding: utf-8 -*-
# file: algorithms/dry_run.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Dry run: the qubit budget of a calculation, without running anything.

A **dry run** answers "how many qubits would this calculation need?" *before*
any integral is computed, any Hamiltonian is mapped, or any circuit is executed.
The count is determined by the problem specification alone -- the geometry, the
basis family, the charge / spin, the frozen core or pseudopotential settings and
the fermion-to-qubit mapping -- so it can be evaluated in milliseconds:

.. math::

    N_\text{qubits} = 2\,M_\text{active}
    = 2\,\big(M_\text{basis} - M_\text{frozen}\big),

one qubit per **spin-orbital** of the active space, where :math:`M_\text{basis}`
is the number of spatial basis functions of the molecule (summed over its atoms)
and :math:`M_\text{frozen}` the number of doubly occupied core orbitals removed by
the frozen-core approximation.  The Jordan-Wigner, parity and Bravyi-Kitaev
mappings all use exactly that many qubits; the parity mapping *can* drop two
more (particle-number and spin-parity symmetries), which is reported separately
as :attr:`QubitEstimate.n_qubits_reduced`.

The estimate also says whether the register fits a **device** -- the registered
QPUs carry their qubit counts -- and, for a state-vector simulator, how much
memory the :math:`2^N` complex amplitudes take.

Entry points
------------
* :func:`estimate_qubits` -- the estimator (geometry or cached Hamiltonian in,
  :class:`QubitEstimate` out);
* every driver's ``dry_run=True`` constructor flag and
  :meth:`~carcara.algorithms.base.VariationalDriver.estimate_qubits` method;
* the command line: ``carcara geometry.xyz --dry-run``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

from ..backends.hardware import (device_qubits, get_device, is_simulator,
                                 normalize_device)

#: Bytes per complex128 amplitude of a state vector.
_BYTES_PER_AMPLITUDE = 16


def statevector_memory_bytes(n_qubits: int) -> int:
    """Memory of one ``complex128`` state vector on ``n_qubits`` (``16 * 2^n``)."""
    return _BYTES_PER_AMPLITUDE * (1 << int(n_qubits))


def _format_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if value < 1024.0 or unit == "PiB":
            return f"{value:.3g} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.3g} PiB"


# --------------------------------------------------------------------------- #
# The estimate.
# --------------------------------------------------------------------------- #

@dataclass
class QubitEstimate:
    """Qubit requirements of a calculation, as determined by a dry run.

    Attributes
    ----------
    n_qubits : int
        Qubits the run would allocate: one per active spin-orbital.
    n_qubits_reduced : int
        The count after the parity mapping's two-qubit reduction (``n_qubits -
        2``; only reachable with ``mapping="parity"`` and a reduction step the
        drivers do not apply by default).
    n_spatial_orbitals : int
        Active spatial orbitals (basis functions minus frozen core).
    n_basis_functions : int
        Spatial basis functions before freezing anything.
    n_frozen_orbitals : int
        Doubly occupied spatial orbitals removed by the frozen core.
    n_electrons : int
        Electrons in the active space.
    num_particles : (int, int)
        Reference occupation ``(n_alpha, n_beta)`` of the active space.
    per_atom : list of (symbol, count)
        Basis functions contributed by each atom (empty for a cached
        Hamiltonian or a plane-wave basis, which is not atom-centered).
    basis, mapping, method, device : str
        The specification the estimate was made for.
    device_qubits : int or None
        Register size of ``device`` when the name fixes one (the Braket QPUs).
    fits_device : bool or None
        ``n_qubits <= device_qubits`` when the capacity is known.
    statevector_bytes : int
        Memory of one exact state vector, the cost driver of a local simulator.
    source : str
        ``"geometry"``, ``"hamiltonian-file"`` or ``"hamiltonian"``.
    notes : list of str
        Human-readable caveats (e.g. an unrunnable device, a plane-wave basis).
    """

    n_qubits: int
    n_spatial_orbitals: int
    n_electrons: int
    num_particles: tuple[int, int]
    n_basis_functions: int = 0
    n_frozen_orbitals: int = 0
    per_atom: list[tuple[str, int]] = field(default_factory=list)
    basis: str = "FAO"
    mapping: str = "jordan_wigner"
    method: str = "adapt-vqe"
    device: str = "AER_simulator"
    device_qubits: int | None = None
    fits_device: bool | None = None
    source: str = "geometry"
    notes: list[str] = field(default_factory=list)

    # -- derived ---------------------------------------------------------- #

    @property
    def n_spin_orbitals(self) -> int:
        return 2 * int(self.n_spatial_orbitals)

    @property
    def n_qubits_reduced(self) -> int:
        """Qubits after the parity mapping's two-qubit symmetry reduction."""
        return max(int(self.n_qubits) - 2, 0)

    @property
    def statevector_bytes(self) -> int:
        return statevector_memory_bytes(self.n_qubits)

    @property
    def device_is_simulator(self) -> bool:
        return is_simulator(self.device)

    # -- presentation ----------------------------------------------------- #

    def as_dict(self) -> dict:
        """Plain-data view (JSON serializable)."""
        data = asdict(self)
        data["num_particles"] = list(self.num_particles)
        data["per_atom"] = [list(item) for item in self.per_atom]
        data["n_spin_orbitals"] = self.n_spin_orbitals
        data["n_qubits_reduced"] = self.n_qubits_reduced
        data["statevector_bytes"] = self.statevector_bytes
        data["device_is_simulator"] = self.device_is_simulator
        return data

    def to_json(self, **kwargs) -> str:
        kwargs.setdefault("indent", 2)
        return json.dumps(self.as_dict(), **kwargs)

    def summary(self) -> str:
        """Multi-line human-readable report."""
        lines = ["Dry run -- no integrals computed, no circuits executed.", ""]
        lines.append(f"  method            : {self.method}")
        lines.append(f"  basis             : {self.basis}")
        lines.append(f"  mapping           : {self.mapping}")
        lines.append(f"  source            : {self.source}")
        if self.per_atom:
            atoms = ", ".join(f"{sym}:{n}" for sym, n in self.per_atom)
            lines.append(f"  basis functions   : {self.n_basis_functions}  "
                         f"({atoms})")
        else:
            lines.append(f"  basis functions   : {self.n_basis_functions}")
        if self.n_frozen_orbitals:
            lines.append(f"  frozen core       : {self.n_frozen_orbitals} "
                         "spatial orbital(s)")
        lines.append(f"  active orbitals   : {self.n_spatial_orbitals} spatial "
                     f"/ {self.n_spin_orbitals} spin")
        lines.append(f"  active electrons  : {self.n_electrons}  "
                     f"(n_alpha, n_beta) = {tuple(self.num_particles)}")
        lines.append("")
        lines.append(f"  QUBITS REQUIRED   : {self.n_qubits}")
        lines.append(f"  with parity 2-qubit reduction : {self.n_qubits_reduced}")
        lines.append("")
        dev = get_device(self.device)
        lines.append(f"  device            : {dev.name}  ({dev.description})")
        if self.device_qubits is not None:
            verdict = "fits" if self.fits_device else "DOES NOT FIT"
            lines.append(f"  device capacity   : {self.device_qubits} qubits  "
                         f"-> {verdict}")
        if self.device_is_simulator:
            lines.append(f"  state vector      : 2^{self.n_qubits} amplitudes = "
                         f"{_format_bytes(self.statevector_bytes)}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"QubitEstimate(n_qubits={self.n_qubits}, "
                f"n_spatial_orbitals={self.n_spatial_orbitals}, "
                f"num_particles={tuple(self.num_particles)}, "
                f"basis={self.basis!r}, device={self.device!r})")


# --------------------------------------------------------------------------- #
# Counting basis functions without a grid.
# --------------------------------------------------------------------------- #

def _basis_key(name: str) -> str:
    return str(name).upper().replace("-", "").replace(" ", "")


def count_basis_functions(atoms, basis="FAO", pseudopotentials=False):
    """Spatial basis functions per atom, ``[(symbol, count), ...]``, plus a label.

    Instantiates the basis family exactly as a run would (so every option that
    changes the function count -- ``size``, polarization, ``n_gaussians`` -- is
    honored) but never samples anything on a grid.  The plane-wave family
    returns one entry ``("PW", n_plane_waves)`` since it is not atom-centered.
    Returns ``(per_atom, basis_label)``.
    """
    from ._hamiltonian_from_atoms import (_merge_pseudo_basis_options,
                                          coherent_positions, resolve_basis)

    symbols = list(atoms.get_chemical_symbols())
    if pseudopotentials:
        from ..experimental.pseudopotentials.io import get_pseudopotential
        from ..experimental.pseudopotentials.orbitals import pseudo_basis

        options = ({} if pseudopotentials is True else dict(pseudopotentials))
        options = _merge_pseudo_basis_options(basis, options)
        directory = options.get("directory")
        potentials = {s: get_pseudopotential(s, directory) for s in set(symbols)}
        positions = coherent_positions(atoms)
        _fns, atom_of_orbital = pseudo_basis(
            symbols, positions, potentials, size=options.get("size", "SZ"),
            split_norm=options.get("split_norm"))
        counts = np.bincount(np.asarray(atom_of_orbital, dtype=int),
                             minlength=len(symbols))
        size = str(options.get("size", "SZ"))
        return ([(s, int(c)) for s, c in zip(symbols, counts)],
                f"PP ({size}, pseudopotentials)")

    name, options = resolve_basis(basis)
    if name == "per-element":
        from ..basis import BasisSet
        from ._hamiltonian_from_atoms import per_element_basis
        per_element_basis(options, symbols)      # validate for these symbols
        bset = BasisSet.build(options)
        return [(s, len(bset.atom(s))) for s in symbols], bset.name
    if _basis_key(name) in ("PW", "PLANEWAVE"):
        from ..core.planewave import (DEFAULT_ENERGY_CUTOFF_EV,
                                      plane_wave_vectors)
        from ..units import EV_TO_HARTREE, to_bohr

        cell = np.asarray(atoms.get_cell(), dtype=float)
        if not np.any(cell):
            raise ValueError("the plane-wave (PW) basis requires a unit cell")
        cutoff = float(options.get("energy_cutoff", DEFAULT_ENERGY_CUTOFF_EV))
        G, _ = plane_wave_vectors(np.asarray(to_bohr(cell, "angstrom")),
                                  cutoff * EV_TO_HARTREE)
        return [("PW", int(len(G)))], f"PW (E_cut = {cutoff:g} eV)"

    from ..basis import BasisSet

    bset = BasisSet.build(name, **options)
    per_atom = [(s, len(bset.atom(s))) for s in symbols]
    label = getattr(bset, "name", None) or str(name)
    if options:
        label += " " + json.dumps(options, sort_keys=True)
    return per_atom, label


# --------------------------------------------------------------------------- #
# The estimator.
# --------------------------------------------------------------------------- #

def _device_fields(device, notes):
    canon = normalize_device(device)
    dev = get_device(canon)
    capacity = device_qubits(canon)
    if not dev.runnable:
        notes.append(f"device {dev.name!r} is reserved (not runnable in this "
                     "build); the estimate is what a run there would need")
    elif not dev.simulator:
        notes.append(f"device {dev.name!r} is real quantum hardware: a run "
                     "there needs shots > 0 (energies are measured, not read "
                     "off a state vector)")
    return canon, capacity


def estimate_qubits(atoms=None, *, basis="FAO", mapping: str = "jordan_wigner",
                    charge: int = 0, n_electrons=None, spin: bool = False,
                    frozen_core=False, frozen_orbitals=None,
                    pseudopotentials=False, load_hamiltonian=None,
                    hamiltonian=None, num_particles=None,
                    n_spatial_orbitals=None, method: str = "adapt-vqe",
                    device: str = "AER_simulator") -> QubitEstimate:
    """Estimate the qubit count of a calculation without running it.

    Three problem sources are understood, in this order of precedence:

    * ``hamiltonian`` -- an already-built :class:`~carcara.core.PauliSum` (or a
      :class:`~carcara.core.Fermion`, whose mode count is used), with
      ``num_particles`` / ``n_spatial_orbitals``;
    * ``load_hamiltonian`` -- a cached Hamiltonian file, whose header records
      the qubit count and the occupation (nothing else is read);
    * ``atoms`` -- an ASE geometry, from which the basis functions are
      *counted* (never integrated) and the electron / frozen-core bookkeeping is
      resolved exactly as :func:`~carcara.algorithms._hamiltonian_from_atoms.build_basis_hamiltonian`
      would.

    Every keyword mirrors the driver argument of the same name.
    """
    from ..core.mapping import Fermion, PauliSum

    notes: list[str] = []
    canon_device, capacity = _device_fields(device, notes)
    fits = None
    method = str(method)
    mapping = str(mapping)

    def finish(**kw):
        kw.setdefault("mapping", mapping)
        est = QubitEstimate(method=method, device=canon_device,
                            device_qubits=capacity, notes=notes, **kw)
        if capacity is not None:
            est.fits_device = est.n_qubits <= capacity
        return est

    # -- an explicit operator ------------------------------------------- #
    if hamiltonian is not None:
        if isinstance(hamiltonian, PauliSum):
            n_qubits = int(hamiltonian.num_qubits)
        elif isinstance(hamiltonian, Fermion):
            n_qubits = int(hamiltonian.n_modes())
        else:
            raise TypeError("hamiltonian must be a PauliSum or Fermion")
        n_orb = int(n_spatial_orbitals) if n_spatial_orbitals else n_qubits // 2
        if num_particles is None:
            raise ValueError("num_particles is required with an explicit "
                             "hamiltonian")
        na, nb = (int(v) for v in num_particles)
        return finish(n_qubits=n_qubits, n_spatial_orbitals=n_orb,
                      n_electrons=na + nb, num_particles=(na, nb),
                      n_basis_functions=n_orb, basis="(explicit Hamiltonian)",
                      source="hamiltonian")

    # -- the cache file -------------------------------------------------- #
    if load_hamiltonian is not None:
        from ..core.serialization import load_hamiltonian as _load

        record = _load(str(load_hamiltonian))
        n_qubits = int(record.hamiltonian.num_qubits)
        n_orb = (int(record.n_spatial_orbitals)
                 if record.n_spatial_orbitals else n_qubits // 2)
        if record.num_particles is None:
            raise ValueError(f"{load_hamiltonian!r} records no num_particles")
        na, nb = (int(v) for v in record.num_particles)
        meta = getattr(record, "metadata", None) or {}
        # The file fixes the mapping the operator was written in.
        return finish(n_qubits=n_qubits, n_spatial_orbitals=n_orb,
                      n_electrons=na + nb, num_particles=(na, nb),
                      n_basis_functions=n_orb, mapping=str(record.mapping),
                      basis=str(meta.get("basis", "(cached)")),
                      source="hamiltonian-file")

    # -- a geometry ------------------------------------------------------ #
    if atoms is None:
        raise ValueError("estimate_qubits needs atoms, load_hamiltonian or an "
                         "explicit hamiltonian")
    from ._hamiltonian_from_atoms import (_num_particles, resolve_frozen_core,
                                          resolve_num_unpaired)

    per_atom, label = count_basis_functions(atoms, basis, pseudopotentials)
    n_basis = int(sum(n for _s, n in per_atom))
    numbers = atoms.get_atomic_numbers()

    if pseudopotentials:
        if frozen_core or frozen_orbitals:
            raise ValueError("frozen_core is redundant with pseudopotentials")
        from ..experimental.pseudopotentials.io import get_pseudopotential
        from ..experimental.pseudopotentials.orbitals import valence_electrons

        options = ({} if pseudopotentials is True else dict(pseudopotentials))
        symbols = atoms.get_chemical_symbols()
        potentials = {s: get_pseudopotential(s, options.get("directory"))
                      for s in set(symbols)}
        n_el = int(round(valence_electrons(symbols, potentials))) - int(charge)
        frozen: list[int] = []
        notes.append("pseudopotentials: the core is absent from the valence "
                     "problem (counts are valence electrons / orbitals)")
    else:
        n_el = (int(n_electrons) if n_electrons is not None
                else int(sum(int(z) for z in numbers)) - int(charge))
        if per_atom and per_atom[0][0] == "PW":
            if frozen_core or frozen_orbitals:
                raise NotImplementedError(
                    "the frozen-core approximation is not supported for the "
                    "plane-wave (PW) basis")
            frozen = []
            notes.append("plane-wave basis: one qubit pair per plane wave; the "
                         "count grows steeply with the cutoff and cell")
            per_atom = []          # not atom-centered
        else:
            frozen = resolve_frozen_core(frozen_core, frozen_orbitals, numbers,
                                         n_el, n_basis)

    n_active_el = n_el - 2 * len(frozen)
    n_unpaired = resolve_num_unpaired(atoms, spin, n_el)
    particles = _num_particles(n_active_el, n_unpaired, label)
    n_active = n_basis - len(frozen)
    if n_active_el > 2 * n_active:
        raise ValueError(
            f"{n_active_el} active electrons cannot fit {n_active} spatial "
            "orbitals; the basis is too small for this charge state")

    return finish(n_qubits=2 * n_active, n_spatial_orbitals=n_active,
                  n_electrons=n_active_el, num_particles=particles,
                  n_basis_functions=n_basis, n_frozen_orbitals=len(frozen),
                  per_atom=per_atom, basis=label, source="geometry")
