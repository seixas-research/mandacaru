# -*- coding: utf-8 -*-
# file: algorithms/dry_run.py

# This code is part of Mandacaru.
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
mappings all use exactly that many qubits; ``parity_reduced`` uses two fewer by
tapering the fixed particle-number and spin-parity qubits.

The estimate also says whether the register fits a **device** -- the registered
QPUs carry their qubit counts -- and, for a state-vector simulator, how much
memory the :math:`2^N` complex amplitudes take.

Entry points
------------
* :func:`estimate_qubits` -- the estimator (geometry or cached Hamiltonian in,
  :class:`QubitEstimate` out);
* every driver's ``dry_run=True`` constructor flag and
  :meth:`~mandacaru.algorithms.base.VariationalDriver.estimate_qubits` method;
* the command line: ``mandacaru geometry.xyz --dry-run``.
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


#: Ansatz length against register width, measured on the LiH / PAW-LCAO series
#: (``docs/source/guide/measurement_cost.md``): qubits -> converged ADAPT
#: operators at ``pool="ceo-ovp"``.  One system, so it is an *order of magnitude*
#: for the depth estimate below and nothing finer -- which is still the
#: difference between "fits" and "fits and returns noise".
MEASURED_ANSATZ_LENGTH = ((4, 3), (8, 15), (20, 45), (24, 75))

#: Two-qubit gates per ADAPT operator, from the same series: 2,223 two-qubit
#: gates for 75 operators at 24 qubits, 881 for 45 at 20, 150 for 15 at 8.
#: Roughly 30 per operator once routing on a heavy-hex lattice dominates, which
#: it does above a handful of qubits.
GATES_PER_OPERATOR = 30


def estimate_ansatz_operators(n_qubits: int) -> int:
    """Converged ADAPT operator count expected at this register width.

    Log-log interpolation of :data:`MEASURED_ANSATZ_LENGTH`, extrapolated at the
    ends.  This is the one quantity in a dry run that is genuinely a *guess*:
    ADAPT decides its own length from gradients the dry run never computes.  It
    is here because the alternative -- reporting a register width and calling the
    job feasible -- is the mistake that
    ``docs/source/guide/measurement_cost.md`` exists to document.
    """
    width = max(int(n_qubits), 1)
    xs = np.log([q for q, _ in MEASURED_ANSATZ_LENGTH])
    ys = np.log([n for _, n in MEASURED_ANSATZ_LENGTH])
    return int(round(float(np.exp(np.interp(np.log(width), xs, ys)))))


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
        2``), or :attr:`n_qubits` itself when ``mapping="parity_reduced"``.
    n_spatial_orbitals : int
        Active spatial orbitals (basis functions minus frozen core).
    n_basis_functions : int
        Spatial basis functions before freezing anything.
    n_frozen_orbitals : int
        Doubly occupied spatial orbitals removed by the frozen core, plus any
        the active-space selector folded into the mean field on top of it.
    n_deleted_orbitals : int
        Virtual spatial orbitals dropped by ``active_orbitals``.
    active_selection : str
        How the virtuals were ranked.  Only the *count* is a dry-run quantity --
        which orbitals a selector picks needs the integrals this run does not
        compute -- so the register width below is exact and the identity of its
        orbitals is not yet decided.
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
        ``n_qubits <= device_qubits`` when the capacity is known.  **Width
        only** -- see :attr:`runnable_on_device`, which is the question a user
        actually has.
    expected_operators : int or None
        Converged ADAPT operator count expected at this width
        (:func:`estimate_ansatz_operators`), for a real device.
    expected_two_qubit_gates : int or None
        ``expected_operators * GATES_PER_OPERATOR``: the depth estimate.
    expected_fidelity : float or None
        ``(1 - e)^n2q`` at the nominal two-qubit error.  This, not the width, is
        what decides whether a hardware run returns signal.
    runnable_on_device : bool or None
        Both criteria together: the register fits **and** the expected fidelity
        clears the threshold below which an expectation value is noise.  A run
        can fit a 156-qubit processor with room to spare and still be
        unrunnable, which is the case this field exists to state.
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
    n_deleted_orbitals: int = 0
    active_selection: str = "energy"
    per_atom: list[tuple[str, int]] = field(default_factory=list)
    basis: str = "HAO"
    mapping: str = "jordan_wigner"
    method: str = "adapt-vqe"
    device: str = "AER_simulator"
    device_qubits: int | None = None
    fits_device: bool | None = None
    expected_operators: int | None = None
    expected_two_qubit_gates: int | None = None
    source: str = "geometry"
    notes: list[str] = field(default_factory=list)

    # -- derived ---------------------------------------------------------- #

    @property
    def n_spin_orbitals(self) -> int:
        return 2 * int(self.n_spatial_orbitals)

    @property
    def n_qubits_reduced(self) -> int:
        """Qubits after the parity mapping's two-qubit symmetry reduction."""
        return (int(self.n_qubits) if self.mapping == "parity_reduced"
                else max(int(self.n_qubits) - 2, 0))

    @property
    def statevector_bytes(self) -> int:
        return statevector_memory_bytes(self.n_qubits)

    @property
    def device_is_simulator(self) -> bool:
        return is_simulator(self.device)

    @property
    def expected_fidelity(self) -> float | None:
        """``(1 - e)^n2q`` at the nominal two-qubit error, or ``None``."""
        from ..backends.measurement import NOMINAL_2Q_ERROR

        if self.expected_two_qubit_gates is None:
            return None
        return float((1.0 - NOMINAL_2Q_ERROR) ** self.expected_two_qubit_gates)

    @property
    def runnable_on_device(self) -> bool | None:
        """Width **and** depth, or ``None`` when either is unknown.

        The width test alone says a 56-qubit problem fits a 156-qubit processor,
        which is true and beside the point: the binding constraint is the
        two-qubit gate count of the ansatz, and past a few hundred gates an
        unmitigated expectation value is noise.  A dry run that reported only
        the first would be answering a question nobody asked.
        """
        from ..backends.measurement import FIDELITY_WARNING

        fidelity = self.expected_fidelity
        if self.fits_device is None or fidelity is None:
            return None
        return bool(self.fits_device and fidelity >= FIDELITY_WARNING)

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
        data["expected_fidelity"] = self.expected_fidelity
        data["runnable_on_device"] = self.runnable_on_device
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
        if self.n_deleted_orbitals:
            lines.append(f"  deleted virtuals  : {self.n_deleted_orbitals} "
                         f"spatial orbital(s), ranked by "
                         f"{self.active_selection}")
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
                         f"-> {verdict} (width)")
        if self.expected_two_qubit_gates is not None:
            fidelity = self.expected_fidelity
            lines.append(f"  expected ansatz   : ~{self.expected_operators} "
                         f"operators, ~{self.expected_two_qubit_gates} "
                         f"two-qubit gates")
            lines.append(f"  expected fidelity : {fidelity:.2e}  "
                         f"-> {'usable' if self.runnable_on_device else 'NOISE'}")
            lines.append(f"  RUNNABLE HERE     : "
                         f"{'yes' if self.runnable_on_device else 'NO'}"
                         f"  (width and depth together)")
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


def _pseudo_basis_count(atoms, family, options):
    """Per-atom valence function counts, a label and the loaded potentials
    for a pseudopotential family (nothing sampled on a grid).

    The Fourier filter (``basis={..., "filter": ...}``) cannot change a
    *count* -- it reshapes each radial function and adds none -- so the basis
    is built unfiltered here, which also keeps the estimate free of the
    spherical Bessel transforms.  The option is still **validated** and named
    in the label: a dry run that quietly accepted a mistyped filter would
    report the qubit count of a calculation that will not start.

    The label names the filter **always**, filtered or not, and reads the
    family's own default (``filter`` is on for PAW-LCAO / UPAW-LCAO, off for the
    norm-conserving families).  Printing it only when it differs from the
    default would make the reader work out each family's default before they
    could tell what basis the estimate describes.
    """
    from ..basis.filtering import filter_label
    from ..pseudopotentials.orbitals import pseudo_basis
    from ._hamiltonian_from_atoms import coherent_positions

    options = family.resolved_options(options)
    symbols = list(atoms.get_chemical_symbols())
    directory = options.get("directory")
    potentials = {s: family.get(s, directory) for s in set(symbols)}
    positions = coherent_positions(atoms)
    # The run's own construction: a Gaussian polarization shell can have a
    # different angular momentum (and so a different function count) than the
    # default one.  It solves radial problems, never anything on a grid.
    from ..pseudopotentials.families import pseudo_basis_arguments
    _fns, atom_of_orbital = pseudo_basis(
        symbols, positions, potentials,
        **pseudo_basis_arguments(family.name, options))
    counts = np.bincount(np.asarray(atom_of_orbital, dtype=int),
                         minlength=len(symbols))
    size = options.get("size", "SZ")
    size_label = ("per-element sizes " + json.dumps(size, sort_keys=True)
                  if isinstance(size, dict) else str(size))
    parts = [size_label, filter_label(options.get("filter"))]
    if "energy_shift" in family.options:
        # Named always, like the filter: "unconfined" is a statement about the
        # basis too, and the one a GPAW comparison has to get right.
        from ..pseudopotentials.confinement import energy_shift_label
        parts.append(energy_shift_label(options.get("energy_shift")))
        from ..pseudopotentials.confinement import resolve_polarization
        parts.append(resolve_polarization(options.get("polarization"),
                                          options.get("energy_shift"))
                     + " polarization")
    parts.append("pseudopotentials")
    per_atom = [(s, int(c)) for s, c in zip(symbols, counts)]
    return per_atom, f"{family.label} ({', '.join(parts)})", potentials


def count_basis_functions(atoms, basis="HAO"):
    """Spatial basis functions per atom, ``[(symbol, count), ...]``, plus a label.

    Instantiates the basis family exactly as a run would (so every option that
    changes the function count -- ``size``, polarization, ``n_gaussians`` -- is
    honored) but never samples anything on a grid.  A pseudopotential family
    (``"NCPP"`` / ``"ONCVPSP"`` / ``"PAW-LCAO"``) counts its valence pseudo-atomic
    orbitals; the plane-wave family returns one entry ``("PW", n_plane_waves)``
    since it is not atom-centered.  Returns ``(per_atom, basis_label)``.
    """
    from ._hamiltonian_from_atoms import resolve_basis, resolve_pseudo_basis

    symbols = list(atoms.get_chemical_symbols())
    name, options = resolve_basis(basis)
    family, options = resolve_pseudo_basis(name, options, symbols)
    if family is not None:
        per_atom, label, _potentials = _pseudo_basis_count(atoms, family, options)
        return per_atom, label

    if name == "per-element":
        from ..basis import BasisSet
        bset = BasisSet.build(options)          # validated by resolve_pseudo_basis
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
    # A named Gaussian family is a native recipe, not the published table it is
    # named after -- the label has to say so wherever the basis is reported.
    label = (getattr(bset, "provenance", None) or getattr(bset, "name", None)
             or str(name))
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


def estimate_qubits(atoms=None, *, basis="HAO", mapping: str = "jordan_wigner",
                    charge: int = 0, n_electrons=None, spin: bool = False,
                    frozen_core=False, frozen_orbitals=None,
                    active_orbitals=None, active_selection: str = "energy",
                    active_threshold=None, taper: bool = False,
                    load_hamiltonian=None,
                    hamiltonian=None, num_particles=None,
                    n_spatial_orbitals=None, method: str = "adapt-vqe",
                    device: str = "AER_simulator") -> QubitEstimate:
    """Estimate the qubit count of a calculation without running it.

    Three problem sources are understood, in this order of precedence:

    * ``hamiltonian`` -- an already-built :class:`~mandacaru.core.PauliSum` (or a
      :class:`~mandacaru.core.Fermion`, whose mode count is used), with
      ``num_particles`` / ``n_spatial_orbitals``;
    * ``load_hamiltonian`` -- a cached Hamiltonian file, whose header records
      the qubit count and the occupation (nothing else is read);
    * ``atoms`` -- an ASE geometry, from which the basis functions are
      *counted* (never integrated) and the electron / frozen-core bookkeeping is
      resolved exactly as :func:`~mandacaru.algorithms._hamiltonian_from_atoms.build_basis_hamiltonian`
      would.

    Every keyword mirrors the driver argument of the same name.
    """
    from ..core.mapping import Fermion, PauliSum, _canonical_method

    notes: list[str] = []
    canon_device, capacity = _device_fields(device, notes)
    method = str(method)
    mapping = _canonical_method(str(mapping))

    def finish(**kw):
        kw.setdefault("mapping", mapping)
        kw["mapping"] = _canonical_method(kw["mapping"])
        if (kw["mapping"] == "parity_reduced"
                and kw.get("source") == "geometry"):
            kw["n_qubits"] = max(int(kw["n_qubits"]) - 2, 0)
            notes.append("parity_reduced mapping: two qubits fewer")
        est = QubitEstimate(method=method, device=canon_device,
                            device_qubits=capacity, notes=notes, **kw)
        if capacity is not None:
            est.fits_device = est.n_qubits <= capacity
        # The depth estimate is only meaningful for a processor: a simulator has
        # no two-qubit error, so quoting a fidelity for it would be noise of a
        # different kind.
        if not est.device_is_simulator:
            est.expected_operators = estimate_ansatz_operators(est.n_qubits)
            est.expected_two_qubit_gates = int(est.expected_operators
                                               * GATES_PER_OPERATOR)
            fidelity = est.expected_fidelity
            if fidelity is not None and est.runnable_on_device is False \
                    and est.fits_device:
                from ..backends.measurement import FIDELITY_WARNING

                notes.append(
                    f"the register fits, and the run would still return noise: "
                    f"~{est.expected_two_qubit_gates} two-qubit gates give an "
                    f"expected fidelity of {fidelity:.1e}, below the "
                    f"{FIDELITY_WARNING:g} at which an unmitigated expectation "
                    f"value stops being signal.  The ansatz length is estimated "
                    f"from a measured series, not computed here -- see "
                    f"docs/source/guide/measurement_cost.md")
        return est

    # -- an explicit operator ------------------------------------------- #
    if hamiltonian is not None:
        if isinstance(hamiltonian, PauliSum):
            n_qubits = int(hamiltonian.num_qubits)
        elif isinstance(hamiltonian, Fermion):
            n_qubits = int(hamiltonian.n_modes())
            if mapping == "parity_reduced":
                n_qubits = max(n_qubits - 2, 0)
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
        from ..core.serialization import read_hamiltonian_header

        # The header alone: a Parquet cache is never decoded (a JSON one has to
        # be parsed to reach it -- see `read_hamiltonian_header`).
        record = read_hamiltonian_header(str(load_hamiltonian))
        n_qubits = int(record.num_qubits)
        n_orb = (int(record.n_spatial_orbitals)
                 if record.n_spatial_orbitals else n_qubits // 2)
        if record.num_particles is None:
            raise ValueError(f"{load_hamiltonian!r} records no num_particles")
        na, nb = (int(v) for v in record.num_particles)
        meta = getattr(record, "metadata", None) or {}
        # The file's mapping fixes the register representation.
        if record.mapping == "parity_reduced":
            notes.append("cached Hamiltonian is already tapered: its "
                         f"{n_qubits} qubits are the reduced count")
        return finish(n_qubits=n_qubits, n_spatial_orbitals=n_orb,
                      n_electrons=na + nb, num_particles=(na, nb),
                      n_basis_functions=n_orb, mapping=str(record.mapping),
                      basis=str(meta.get("basis", "(cached)")),
                      source="hamiltonian-file")

    # -- a geometry ------------------------------------------------------ #
    if atoms is None:
        raise ValueError("estimate_qubits needs atoms, load_hamiltonian or an "
                         "explicit hamiltonian")
    from ._hamiltonian_from_atoms import (_num_particles, resolve_basis,
                                          resolve_frozen_core,
                                          resolve_num_unpaired,
                                          resolve_pseudo_basis)

    symbols = list(atoms.get_chemical_symbols())
    numbers = atoms.get_atomic_numbers()
    name, options = resolve_basis(basis)
    family, options = resolve_pseudo_basis(name, options, symbols)

    if family is not None:
        if frozen_core or frozen_orbitals:
            raise ValueError(
                f"frozen_core is redundant with the {family.label} basis -- "
                "the core is already absent from the valence-only pseudo basis")
        from ..pseudopotentials.orbitals import valence_electrons

        per_atom, label, potentials = _pseudo_basis_count(atoms, family, options)
        n_basis = int(sum(n for _s, n in per_atom))
        n_el = int(round(valence_electrons(symbols, potentials))) - int(charge)
        frozen: list[int] = []
        notes.append(f"pseudopotentials ({family.label} family): the core is "
                     "absent from the valence problem (counts are valence "
                     "electrons / orbitals)")
    else:
        per_atom, label = count_basis_functions(atoms, basis)
        n_basis = int(sum(n for _s, n in per_atom))
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

    n_unpaired = resolve_num_unpaired(atoms, spin, n_el)
    n_deleted = 0
    if taper:
        notes.append(
            "taper=True: the qubit count below is an UPPER BOUND.  How many Z2 "
            "symmetries a Hamiltonian has is a property of its Pauli terms, "
            "which a dry run does not build -- two is the minimum for a "
            "molecule (the two the parity reduction already knows) and a "
            "symmetric molecule usually has more.  Reduced density matrices, "
            "and so forces, densities and charges, are refused on a tapered "
            "register")
    if active_threshold is not None:
        # A count fixes the register width without any integrals; an occupation
        # threshold does not -- how many virtual orbitals clear it is a property
        # of the second-order density, which a dry run does not compute.  So the
        # width reported here is an *upper* bound, and saying so is the only
        # honest option: printing the untruncated number unqualified would have
        # a user size a job against a register the run will not use.
        from .active_space import resolve_threshold

        value = resolve_threshold(active_threshold)
        notes.append(
            f"active_threshold={value:g}: the qubit count below is an UPPER "
            f"BOUND.  How many virtual orbitals clear an occupation threshold "
            f"depends on the second-order density, which a dry run does not "
            f"compute -- run the geometry to learn the real width, or give "
            f"active_orbitals=<count> as well, which caps it in advance")
    if active_orbitals is not None:
        # Only the counts, never the choice: which orbitals a selector keeps
        # takes the integrals, and a dry run computes none.  The register width
        # does not depend on that choice, so it is still exact here.
        from .active_space import (_resolve_counts, normalize_active_orbitals,
                                   resolve_selection)

        active_selection = resolve_selection(active_selection)
        spec = normalize_active_orbitals(active_orbitals)
        n_alpha_full, n_beta_full = _num_particles(n_el, n_unpaired, label)
        n_doubly = min(n_alpha_full, n_beta_full)
        n_singly = abs(n_alpha_full - n_beta_full)
        n_virtual = n_basis - n_doubly - n_singly
        if isinstance(spec, tuple):
            kept = [p for p in spec if p not in set(frozen)]
            n_deleted = n_virtual - sum(1 for p in kept
                                        if p >= n_doubly + n_singly)
        else:
            n_occ_active, n_virt_active = _resolve_counts(
                spec, n_doubly, n_singly, n_virtual, len(frozen))
            frozen = list(frozen) + [
                p for p in range(n_doubly) if p not in set(frozen)
            ][:n_doubly - len(frozen) - n_occ_active]
            n_deleted = n_virtual - n_virt_active
        notes.append(
            f"active_orbitals: {n_deleted} virtual spatial orbital(s) dropped "
            f"(ranked by {active_selection} once the integrals exist). Nuclear "
            f"forces and the stress are refused with a truncated virtual "
            f"space, because the selection moves with the nuclei")

    n_active_el = n_el - 2 * len(frozen)
    particles = _num_particles(n_active_el, n_unpaired, label)
    n_active = n_basis - len(frozen) - n_deleted
    if n_active_el > 2 * n_active:
        raise ValueError(
            f"{n_active_el} active electrons cannot fit {n_active} spatial "
            "orbitals; the basis is too small for this charge state")

    return finish(n_qubits=2 * n_active, n_spatial_orbitals=n_active,
                  n_electrons=n_active_el, num_particles=particles,
                  n_basis_functions=n_basis, n_frozen_orbitals=len(frozen),
                  n_deleted_orbitals=n_deleted,
                  active_selection=active_selection,
                  per_atom=per_atom, basis=label, source="geometry")
