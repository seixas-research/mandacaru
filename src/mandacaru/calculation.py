"""ASE calculator and user interface for Rust-owned calculations."""

from __future__ import annotations

import atexit
import contextlib
import json
import time
import warnings
import weakref
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, PropertyNotImplementedError
from .integrations.ase_properties import ASEProperties

from . import _native, hubbard
from .resources import CalculationResources
from .results import CalculationResult
from .specifications import (
    METHODS,
    FUNCTIONALS,
    MAPPINGS,
    basis_specification,
)
from .structure import geometry_value
from .units import convert_energy, to_bohr

DEFAULT_METHOD = "adapt-vqe"
#: The ADAPT operator pools the native solver builds (``_native.pool_kind``).
OPERATOR_POOLS = ("fermionic", "qubit", "qeb", "ceo", "ceo-ovp", "spin-orbit")
#: The circuit templates of ``method="vqe"`` (``ansatz=``).
ANSATZ_NAMES = ("uccsd", "hva")
#: The ``two_body=`` factorization methods (the integrals are always Cholesky factors).
TWO_BODY_METHODS = ("cholesky",)
#: The default stopping threshold of the Cholesky factorization, in eV (owner, 2026-11-10).
DEFAULT_TWO_BODY_TOLERANCE_EV = 1e-8
#: Methods that difference the energy at tiny steps: refused on a measured energy.
NOISE_INTOLERANT = ("SLSQP", "BFGS", "L-BFGS", "CG")
STABLE_METHODS = tuple(METHODS)

_SEARCH_METHODS = ("vasqa", "valsqa")
#: Options that act only for some methods, and the methods they act for. A value
#: other than the default on any other method raises: nothing is silently ignored.
#: ``spin``/``unrestricted`` also suit ``uhf``, which is unrestricted by definition.
#: ``mapping`` and ``active_space`` are not listed: the classical methods use them
#: to export their Hamiltonian as a quantum problem.
_METHOD_SCOPED_OPTIONS = {
    **dict.fromkeys(
        (
            "seed",
            "min_length",
            "max_length",
            "move_weights",
            "temperature",
            "length_penalty",
            "warm_start",
            "replace_start",
            "proposal_temperature",
            "proposal",
        ),
        _SEARCH_METHODS,
    ),
    **dict.fromkeys(
        (
            "xc",
            "vdw",
            "smearing",
            "periodic_scf_options",
            "eigensolver",
            "hubbard_u",
            "exxdiv_treatment",
            "gygi_baldereschi",
        ),
        ("dft",),
    ),
    "taper": ("vqe", "adapt-vqe", "subspace-vqe", "subspace-adapt-vqe"),
    "measurement_scheme": ("vqe", "adapt-vqe"),
    "quenching": ("vqe", "adapt-vqe"),
    "checkpoint_every": ("vqe", "adapt-vqe"),
    **dict.fromkeys(("tetris", "prune"), ("adapt-vqe",)),
    "kpts": ("dft", "vqe", "adapt-vqe", "vasqa", "valsqa"),
    "crystal_orbitals": ("vqe", "adapt-vqe", "vasqa", "valsqa"),
    **dict.fromkeys(("spin", "unrestricted"), ("dft", "uhf")),
    "pool": tuple(name for name, capability in METHODS.items() if capability.quantum),
}


def _refuse_inapplicable(method, arguments):
    """Raise for an option given a non-default value that ``method`` would ignore."""
    import inspect

    parameters = inspect.signature(Mandacaru.__init__).parameters
    refused = []
    for name, methods in _METHOD_SCOPED_OPTIONS.items():
        if method in methods:
            continue
        value, default = arguments[name], parameters[name].default
        try:
            changed = bool(value is not default and value != default)
        except (TypeError, ValueError):
            changed = True
        if changed:
            refused.append(f"{name} (for {', '.join(methods)})")
    if refused:
        raise ValueError(
            f"method={method!r} does not use: {'; '.join(refused)}; "
            "remove the option or choose a method it applies to"
        )


_EXXDIV_TREATMENTS = ("none", "gygi-baldereschi")


def _exchange_divergence(exxdiv_treatment, gygi_baldereschi, periodic_scf_options):
    """The canonical divergence treatment of the exact exchange.

    ``exxdiv_treatment`` is ``None`` (left out), ``"none"`` or
    ``"gygi-baldereschi"``; ``gygi_baldereschi=True`` is the one shorthand for
    the second. Any other spelling, a non-boolean flag, a flag that contradicts
    an explicit ``exxdiv_treatment`` and the old route through
    ``periodic_scf_options`` raise.
    """
    for key in ("exxdiv_treatment", "gygi_baldereschi"):
        if key in (periodic_scf_options or {}):
            raise ValueError(
                f"{key} is a Mandacaru option, not a periodic_scf_options key; "
                f"pass Mandacaru({key}=...)"
            )
    if exxdiv_treatment is not None and exxdiv_treatment not in _EXXDIV_TREATMENTS:
        raise ValueError(
            f"unknown exxdiv_treatment {exxdiv_treatment!r}; "
            "use 'none' or 'gygi-baldereschi'"
        )
    if gygi_baldereschi is not None and not isinstance(gygi_baldereschi, bool):
        raise TypeError("gygi_baldereschi must be True, False or None")
    if gygi_baldereschi is None:
        return "none" if exxdiv_treatment is None else exxdiv_treatment
    wanted = "gygi-baldereschi" if gygi_baldereschi else "none"
    if exxdiv_treatment is not None and exxdiv_treatment != wanted:
        raise ValueError(
            f"gygi_baldereschi={gygi_baldereschi} contradicts "
            f"exxdiv_treatment={exxdiv_treatment!r}; give one of them"
        )
    return wanted


def _refuse_exchange_divergence_without_hybrid(treatment, functional, periodic):
    """Raise for a divergence treatment that no exact exchange would use."""
    if treatment == "none":
        return
    if not periodic:
        raise ValueError(
            f"exxdiv_treatment={treatment!r} treats the divergence of a periodic "
            "exact exchange; a molecule has none, so leave the option out"
        )
    if functional != "hse06":
        raise ValueError(
            f"exxdiv_treatment={treatment!r} applies to a hybrid functional "
            f"(xc='hse06'), not to xc={functional!r}"
        )


def _check_float32_floors(method, tolerance, convergence, *, adapt):
    """Raise for a setting a complex64 state cannot resolve, naming the floor.

    The floors are Rust constants (``FLOAT32_ENERGY_FLOOR_HA`` and
    ``FLOAT32_GRADIENT_FLOOR``); the solver checks them again as defense in depth.
    ``convergence["energy"]`` is in eV like every convergence energy of the
    calculator; the optimizer ``tol`` is in Hartree.
    """
    energy_floor = _native.FLOAT32_ENERGY_FLOOR_HA
    gradient_floor = _native.FLOAT32_GRADIENT_FLOOR
    if tolerance is None and method != "SPSA":
        raise ValueError(
            "optimizer tol=None resolves to the solver default, which is below the "
            f"float32 energy floor {energy_floor:g} Ha of the complex64 state; give "
            "a tol at or above the floor or use precision='float64'"
        )
    if tolerance is not None and tolerance < energy_floor:
        raise ValueError(
            f"optimizer tol {tolerance:g} Ha is below the float32 energy floor "
            f"{energy_floor:g} Ha of the complex64 state; raise it or use "
            "precision='float64'"
        )
    if not adapt:
        return
    energy = convergence.get("energy")
    if energy is not None and _native.energy_to_hartree(float(energy), "eV") < energy_floor:
        raise ValueError(
            f"convergence['energy'] {energy:g} eV is below the float32 energy floor "
            f"{energy_floor:g} Ha ({_native.energy_from_hartree(float(energy_floor), 'eV'):g} eV) of the "
            "complex64 state; raise it or use precision='float64'"
        )
    gradient = convergence.get("gradient")
    if gradient is not None and gradient < gradient_floor:
        raise ValueError(
            f"convergence['gradient'] {gradient:g} Ha/rad is below the float32 gradient "
            f"floor {gradient_floor:g} Ha/rad of the complex64 state; raise it or use "
            "precision='float64'"
        )


def _close_log_at_exit(reference):
    """The ``atexit`` body: it holds the calculator weakly, so it keeps nothing alive."""
    calculator = reference()
    if calculator is not None:
        calculator._write_summary_at_exit()


def _method_key(name):
    return str(name).lower().replace("_", "-")


def resolve_method(name):
    if isinstance(name, str) and name.startswith("bloch-") and name[6:] in METHODS:
        raise ValueError(
            f"{name!r} is not a method name: use method={name[6:]!r}, which solves a crystal "
            "(any periodic axis in atoms.pbc) as the Born-von Karman supercell of kpts"
        )
    if name not in METHODS:
        raise ValueError(
            f"unsupported native method {name!r}; available: {', '.join(METHODS)}"
        )
    return name, METHODS[name]


class Mandacaru(ASEProperties, Calculator):
    """ASE calculator with scientific state and numerical algorithms owned by Rust.

    Attach directly with ``atoms.calc = Mandacaru(...)``. ``run(Geometry(...))``
    also accepts standalone geometry values. Each calculator owns its resources.
    This docstring is the options reference: every constructor parameter is
    keyword-only (except ``method``) and is listed below in signature order.

    Lengths given to the calculator are in Angstrom and energies it reports
    through ASE are in eV; Hartree and Bohr appear only where an entry says so.
    Results report eV unless ``atomic_units=True``.

    Selecting the method. ``method`` picks the calculation and, with it, which
    options apply. The classical mean-field methods are ``"rhf"``, ``"uhf"``,
    ``"ghf"`` and ``"dft"``. The quantum methods are ``"vqe"`` (a fixed
    ansatz), ``"adapt-vqe"`` (the default; the ansatz grows from an operator
    pool), ``"subspace-vqe"`` and ``"subspace-adapt-vqe"`` (several states at
    once), and the architecture searches ``"vasqa"`` and ``"valsqa"``. The
    quantum methods start from a Hartree-Fock reference on the same geometry
    and basis, build the second-quantized Hamiltonian, map it to qubits and
    run the variational solver in Rust. On a structure with any periodic axis
    (``atoms.pbc``), ``"vqe"``, ``"adapt-vqe"``, ``"vasqa"`` and ``"valsqa"``
    solve the Born-von Karman supercell of the Gamma-centered ``kpts`` mesh
    (the Gamma point alone by default) and report the energy per primitive
    cell; the subspace methods refuse a crystal.

    The option groups are: geometry and grid (``atoms``, ``grid_spacing``,
    ``grid``, ``kinetic``, ``precision``, ``resources``); basis and
    pseudopotentials (``basis``, ``paw_dataset``); the electronic problem
    (``xc``, ``spin``, ``unrestricted``, ``charge``, ``n_electrons``,
    ``scf_options``, ``two_body``); the qubit problem
    (``mapping``, ``active_space``, ``sector``, ``sparse``, ``taper``); the ansatz
    (``pool``, ``ansatz``, ``trotter``, ``include_singles``); the optimization
    (``optimizer``, ``convergence``, ``max_iterations``, ``gradient``); several
    states (``num_states``, ``weights``); architecture search (``max_steps``,
    ``min_length``, ``max_length``, ``move_weights``, ``temperature``,
    ``length_penalty``, ``warm_start``, ``replace_start``,
    ``screen_insertions``, ``proposal_temperature``, ``proposal``,
    ``proposal_model``, ``record``, ``seed``); handover between geometries
    (``transfer``, ``transfer_threshold``, ``transfer_steps``,
    ``update_between_geometries``); circuits and hardware
    (``execute_circuits``, ``backend_provider``, ``device``, ``shots``,
    ``backend_options``, ``measurement_budget``); cached problems
    (``hamiltonian``, ``n_spatial_orbitals``, ``num_particles``,
    ``hamiltonian_format``, ``load_hamiltonian``, ``save_hamiltonian``,
    ``native_model``, ``checkpoint``, ``resume``, ``load_crystal``,
    ``save_crystal``); crystals (``kpts``, ``smearing``,
    ``periodic_scf_options``, ``symmetry``); forces
    (``project_translation``); and reporting (``atomic_units``, ``trace``,
    ``txt``, ``profile``).

    A bad value, or a combination the calculator cannot honor, raises: most
    checks run in the constructor, a few need the geometry or the SCF and run
    in ``run()``. An option that belongs to a different method is not always
    refused: where the entry below says "ignored" the value is accepted and has
    no effect. Check the method before relying on such an option.

    Parameters
    ----------
    method : str, default "adapt-vqe"
        One of ``"rhf"``, ``"uhf"``, ``"ghf"``, ``"dft"``, ``"vqe"``,
        ``"adapt-vqe"``, ``"subspace-vqe"``, ``"subspace-adapt-vqe"``,
        ``"vasqa"``, ``"valsqa"``. Anything else raises ``ValueError`` naming
        the available methods. ``"ghf"`` has no forces. ``"subspace-*"`` have
        no forces. Forces exist for the other methods on a molecule, with
        ``kinetic="spectral"``, no circuit execution, no ``taper=True`` and no
        periodic cell. A periodic geometry runs ``"dft"``, and ``"vqe"``,
        ``"adapt-vqe"``, ``"vasqa"`` and ``"valsqa"`` on its Born-von Karman
        supercell (see below); the other methods refuse it. The
        classical methods refuse ``shots``, ``execute_circuits=True``,
        ``backend_provider``, ``device``, ``backend_options`` and
        ``measurement_budget`` (``ValueError``, since they have no quantum
        state to measure), and they refuse ``hamiltonian``, ``load_hamiltonian``
        and ``native_model`` (``ValueError``: a mean-field calculation needs a
        geometry or a cached crystal).
    atoms : ase.Atoms, optional
        Attach the calculator to this object (``atoms.calc = self``); the same
        as assigning ``atoms.calc`` afterwards. Positions and cell are in
        Angstrom. A molecule needs a nonzero cell (for example
        ``atoms.center(vacuum=3.0)``) to define the real-space box, or an
        explicit ``grid``; otherwise ``run()`` raises ``ValueError`` ("molecular
        integration requires a nondegenerate cell or an explicit grid"). If any
        direction is periodic the periodic route is taken (see ``kpts``). The
        sum of the ASE initial magnetic moments sets the number of unpaired
        electrons (rounded to an integer); with zero moments an odd electron
        count gives one unpaired electron. Moments given as one 3-vector per
        atom (``(N, 3)``, a noncollinear start) run spinor Kohn-Sham
        (``method="dft"``; another method raises ``NotImplementedError``,
        and so does a crystal): the collinear state with the moments' lengths
        is turned per atom to its moment's direction.
    hamiltonian : Fermion, PauliSum or native PauliOperator, optional
        A Hamiltonian to minimize instead of one built from a geometry, for
        the quantum methods, run without a geometry. It needs
        ``n_spatial_orbitals`` and ``num_particles`` (``ValueError`` if either
        is missing). A ``Fermion`` is taken over ``2 * n_spatial_orbitals``
        spin-orbitals and mapped with ``mapping``; a ``PauliSum`` or a native
        ``PauliOperator`` is used as the qubit operator directly (Hartree).
        Mutually exclusive with ``load_hamiltonian``, ``native_model`` and
        ``load_crystal``. Any other type raises ``TypeError`` at ``run()``.
    n_spatial_orbitals : int, optional
        Number of spatial orbitals of the supplied ``hamiltonian`` (the register
        has ``2 * n_spatial_orbitals`` qubits, two fewer for
        ``mapping="parity_reduced"``). Used together with ``hamiltonian`` (and by
        the ``n_qubits`` property before a run); has no effect on a calculation
        that does not supply one.
    num_particles : tuple of int, optional
        ``(n_alpha, n_beta)`` electron counts of the supplied ``hamiltonian``;
        they fix the reference determinant and the particle sector. Used together
        with ``hamiltonian`` (and by the ``num_particles`` property before a
        run); has no effect on a calculation that does not supply one.
    hamiltonian_format : {None, "json", "parquet"}, default None
        File format of ``save_hamiltonian``. Any other value raises
        ``ValueError``. For a run from a geometry, ``None`` writes the native
        JSON snapshot unless the path ends in ``.parquet`` or ``.pq``; for a run from a
        supplied ``hamiltonian`` or ``load_hamiltonian`` alone, ``None`` follows
        the file extension and falls back to Parquet. Reading is by content,
        not by name: ``load_hamiltonian`` recognizes a Parquet file by its magic
        bytes.
    basis : str or dict, default "HAO"
        The basis. A name, a dict with a ``"name"`` key and options, or a
        mapping from element (symbol, atomic number or ``"*"``) to such a name
        or dict, to mix bases per element. Names: ``"HAO"`` (hydrogenic atomic
        orbitals; the only option is ``virtual_orbitals``), ``"NAO"`` (options
        ``size`` with default ``"DZP"``, ``r_c`` in Bohr, ``n_grid`` with
        default 2000, ``energy_shift`` in eV with default 0.03, which sets the
        default ``r_c`` as pi over the square root of twice the shift in
        Hartree, and ``zeta_split``), ``"NAO-AE"`` (all-electron numerical
        orbitals; options ``tier``, ``onset`` and ``width`` in Angstrom,
        ``scale``, ``extra``, ``linear_dependence_tol``, ``tail_norm``,
        ``points``, ``r_max``), a Gaussian recipe name such as ``"STO-3G"`` (no
        extra options), and the pseudopotential families ``"PAW-LCAO"`` and
        ``"UPAW-LCAO"``. An unknown name or option raises ``ValueError`` or
        ``TypeError`` at construction. The PAW-LCAO and UPAW-LCAO dict options
        are: ``size`` (one of SZ, SZP, DZ, DZP, DZ2P, TZ, TZP, TZ2P, QZ, QZP,
        QZ2P, or an ``(n_zeta, n_polarization)`` pair; default ``"SZ"``);
        ``zeta_split`` (``"first_zeta"``, the default, or ``"last_zeta"``; a
        number or list in (0, 1) sets custom tail norms for ``"first_zeta"``);
        ``paw_dataset`` (the dataset folder, see the parameter of the same
        name); ``filter_cutoff`` (default ``"auto"``: a Fourier filter at the
        Nyquist wave-vector of the grid; ``False`` or ``None`` switches it off;
        a positive number is a kinetic-energy cutoff in eV); ``energy_shift``
        (eV, default 0.1, confines each orbital; ``None`` keeps the free-atom
        orbital); ``confinement_potential`` (a pair: amplitude in Hartree and
        inner radius as a fraction of the cutoff radius, default (12.0, 0.6));
        ``polarization_method`` (``"gaussian"`` or ``"orbital"``);
        ``projector_basis`` (``"raw"``, the default, or ``"dual"``); and
        ``diffuse`` (diffuse shells; molecules only, a crystal raises).
        PAW-LCAO needs the pseudopotential library (``MANDACARU_PAW_PATH``);
        without it the calculation raises ``LibraryPathError``. A spin-orbit
        dataset (``lda-fr``, ``pbe-fr``) runs spinor (noncollinear) Kohn-Sham
        with the spin-orbit term, for a molecule and for a crystal, with
        ``method="dft"``; any other method on a molecule raises
        ``NotImplementedError``. A crystal accepts only
        PAW-LCAO or UPAW-LCAO (``NotImplementedError`` at ``run()``).
    grid_spacing : float, default 0.2
        Real-space grid spacing in Angstrom; must be positive and finite
        (``ValueError``). Used when no explicit ``grid`` is given. For the
        crystal route it sets the grid of the cell.
    precision : {"float32", "float64"}, default "float64"
        Storage precision of the large arrays; any other value raises
        ``ValueError``. ``"float64"`` stores float64 reals and complex128
        complex numbers. ``"float32"`` stores float32 reals and complex64
        complex numbers for: the stored sampled orbitals (with the r2SCAN
        orbital gradients), the Bloch orbitals of a crystal and the state
        vector of every quantum method. These stay float64 under both
        values: the Poisson/FFT workspace, the Cholesky factors and the
        two-body model, the one-body, overlap and every other small matrix, the
        SCF and DIIS iteration, the Hamiltonian coefficients, and every sum
        over many float32 terms.
        The Python assembly of the crystal Bloch matrices is float64 and
        dominates a crystal's process memory until its port to Rust. Savings
        are not stated here: measure the context peak and the process RSS of
        your own system. Float32 changes the numbers at the level the
        accuracy gates measure, and a setting it cannot resolve raises
        ``ValueError`` naming the floor instead of running: an optimizer
        ``tol`` or an ADAPT ``convergence["energy"]`` below the energy floor
        (``FLOAT32_ENERGY_FLOOR_HA``) and an ADAPT ``convergence["gradient"]``
        below the gradient floor (``FLOAT32_GRADIENT_FLOOR``) of the Rust
        extension; an optimizer ``tol`` left unset resolves to the energy floor
        (it is ``1e-12`` in float64) and is written in the optimization setup
        of the run log. Refused with
        ``ValueError``: forces of a quantum method (the 1e-3 Bohr
        finite-difference step turns float32 rounding into force noise; the
        analytic forces of the classical methods, all-electron and PAW-LCAO,
        are allowed), the forces and stress of a crystal (the
        analytic derivative engines read float64 data only), ``load_crystal`` with a precision other than
        the one stored in the file, and ``hamiltonian``,
        ``load_hamiltonian`` or ``native_model`` on a circuit device
        (``shots > 0`` or a hardware device), which holds neither orbital
        samples nor a state vector. The precision is written once, in the
        ``[BASIS]`` block of the run log.
    grid : mandacaru.integrals.grid.Grid or native SpatialGrid, optional
        An explicit real-space grid that replaces the one built from the cell
        and ``grid_spacing``. With it the atoms need no cell. The first force
        evaluation freezes the grid in use so that displaced geometries share
        it.
    resources : dict, optional
        Per-calculator allocation, with keys ``threads`` (default chosen by the
        native context), ``memory_limit_bytes`` (default 34359738368, that is
        32 GiB), ``disk_offload_bytes`` (the size from which a periodic
        calculation keeps its bulk arrays, such as the Bloch orbital stacks,
        in a scratch folder on disk instead of memory: default 8 MiB, ``0``
        offloads every one) and ``scratch_dir`` (parent directory for both
        Python scratch and native archived arrays). An unknown key raises
        ``TypeError``. A calculation that needs more memory than the limit
        raises ``MemoryError`` at ``run()``. Not allowed together with
        ``native_model``, which carries its own context (``ValueError``).
    xc : str, optional
        Exchange-correlation functional, one of ``"lda"``, ``"pbe"``,
        ``"r2scan"``, ``"hse06"``, ``"pbe+rvv10"``, ``"r2scan+rvv10"``; anything
        else raises ``ValueError``. ``None`` means ``"lda"``. Used by
        ``method="dft"`` only and ignored by every other method (an
        ``"rhf"`` run with ``xc="pbe"`` is the same Hartree-Fock). It also
        chooses the default PAW-LCAO dataset folder (``pbe-sr`` for PBE, HSE06
        and r2SCAN, ``lda-sr`` otherwise); a dataset generated with another
        functional triggers a warning. A crystal supports every one of them;
        ``"hse06"`` runs on the whole k mesh (see ``symmetry``), with or
        without spin-orbit coupling; its forces and stress are analytic (Rust).
        ``"r2scan"`` and ``"rvv10"`` crystals have analytic forces and stress,
        unpolarized, spin-polarized and with spin-orbit coupling. With
        ``load_crystal`` a different ``xc`` raises ``ValueError``.
    vdw : str or bool, optional
        Van der Waals dispersion correction, ``"rvv10"`` or ``"ts"``
        (Tkatchenko-Scheffler method). For ``"rvv10"``, it is
        appended to the exchange-correlation functional for supported
        functionals (e.g. ``xc="r2scan", vdw="rvv10"`` becomes
        ``"r2scan+rvv10"``). Used by ``method="dft"`` only.
    spin : bool, default False
        Request an unrestricted (spin-polarized) Kohn-Sham calculation. It is
        the same switch as ``unrestricted`` (the two are combined with "or")
        and has an effect only for ``method="dft"`` (molecule and crystal).
        The other methods choose open or closed shell from the electron count
        and the initial magnetic moments, and ignore it. A non-boolean raises
        ``TypeError``. A crystal with nonzero initial magnetic moments
        automatically calculates with ``spin=True`` (unrestricted Kohn-Sham)
        for ``method="dft"``, matching molecular behavior (a spin-orbit dataset
        takes its moments as the noncollinear start and needs neither).
    unrestricted : bool, default False
        Same as ``spin``; either one set to ``True`` selects unrestricted
        ``method="dft"``. A molecule with unequal alpha and beta counts or a
        crystal with nonzero initial magnetic moments is unrestricted
        regardless. Non-boolean raises ``TypeError``.
    charge : int, default 0
        Total charge. The electron count is the sum of the nuclear charges (the
        dataset valence charges for PAW-LCAO) minus ``charge``. A non-integer
        raises ``TypeError``. ``method="rhf"`` with an odd electron count
        raises ``ValueError`` at ``run()``; use ``"uhf"``.
    n_electrons : int, optional
        Total electron count, overriding the count above; ``charge`` is not
        subtracted when this is given. A count that does not fit the basis, or
        has the wrong parity for the initial magnetic moment, raises
        ``ValueError`` at ``run()``. The crystal route refuses it
        (``ValueError``): its electron count follows the datasets and
        ``charge``.
    scf_options : dict, optional
        Options of the molecular Hartree-Fock or Kohn-Sham SCF: ``max_iterations``
        (default 300), ``tolerance`` (default 1e-9), ``diis`` (default
        ``True``) and ``guesses`` (default 3, how many staged starting guesses
        the SCF may try). An unknown key raises ``TypeError`` at ``run()``. A crystal
        refuses it (``ValueError``); use ``periodic_scf_options``.
    optimizer : str or dict, optional
        The classical optimizer of the quantum methods; the mean-field methods
        ignore it. A string is the method name; a dict has the keys ``method``,
        ``maxiter``, ``maxfev``, ``tol`` and ``track_best``, and any other key
        raises ``ValueError``. The method is one of ``"SPSA"``, ``"COBYLA"``,
        ``"Nelder-Mead"``, ``"SLSQP"``, ``"BFGS"``, ``"L-BFGS"`` and ``"CG"``,
        and nothing else (``ValueError``). Defaults when a key is absent:
        method ``"SLSQP"``, ``maxiter`` 2000, ``tol`` 1e-12, ``maxfev`` 5000,
        ``track_best`` ``True``. With ``shots > 0`` the default is instead the
        strict two-evaluation SPSA with ``maxiter`` 100, ``tol`` ``None`` and
        ``track_best`` ``False``; it costs ``2 * maxiter + 1`` energy
        evaluations, each a job on a processor. On a shot-based energy
        ``"SLSQP"``, ``"BFGS"``, ``"L-BFGS"`` and ``"CG"`` are refused
        (``ValueError``) because they difference the energy at tiny steps;
        use ``"SPSA"`` or ``"COBYLA"``. On a shot-based fixed-ansatz run
        whose optimizer could exceed the provider's run budget, ``run()`` raises
        ``RunBudgetExceeded`` before any job is queued.
    mapping : {"jordan_wigner", "parity", "bravyi_kitaev", "parity_reduced"}, optional
        Fermion-to-qubit mapping of the quantum methods. ``None`` means
        ``"jordan_wigner"``. ``"parity_reduced"`` removes two qubits (the
        register has ``2 * M - 2`` qubits for ``M`` spatial orbitals) and is
        refused by the ``"spin-orbit"`` pool at ``run()``. Anything else raises
        ``ValueError``. With ``load_hamiltonian`` a mapping that disagrees with
        the cached one raises ``ValueError``; a crystal raises ``ValueError``
        for any explicit mapping. The classical methods ignore it.
    active_space : dict, optional
        Reduce the molecular Hamiltonian to an active space before mapping.
        A dict with the keys: ``method`` (``"energy"``, the default, orders
        the orbitals by SCF energy; ``"mp2"`` ranks by MP2 natural-orbital
        occupation on a Hartree-Fock reference, closed or open shell;
        ``"dlpno-mp2"`` does the same, closed shell only), ``orbitals`` (a count of spatial orbitals, a list of
        orbital indices, or ``{"occupied": n, "virtual": n}``), ``frozen`` (a
        count, a list of doubly occupied orbital indices, or ``"auto"`` for the
        element-based frozen core, which is zero for a PAW-LCAO basis),
        ``threshold`` (an occupation threshold in (0, 2), or ``True`` for 1e-3;
        MP2 ranking only), and ``symmetry`` (``True`` keeps degenerate orbital
        sets whole; closed shell only) and ``correlating_pairs`` (``True``
        keeps the correlating partner of every active doubly occupied orbital
        active; not together with ``symmetry``). A bad key or value raises ``ValueError`` at
        ``run()``. ``None`` keeps every orbital. A crystal refuses it, and
        the classical methods ignore it. Occupied orbitals must be frozen or
        active; the active space must be nonempty.
    load_hamiltonian : path, optional
        Read a cached qubit Hamiltonian (written by ``save_hamiltonian``, JSON
        or Parquet) instead of building one; run it without a geometry. The
        file fixes the mapping and the particle numbers. Quantum methods only.
        Mutually exclusive with ``hamiltonian``, ``native_model`` and
        ``load_crystal`` (``ValueError``).
    save_hamiltonian : path, optional
        Write the molecular Hamiltonian (after the active-space reduction) to
        this path when the run starts, in the format of ``hamiltonian_format``.
        Written by any method that has a geometry, including the classical
        ones. A crystal refuses it (``ValueError``); use ``save_crystal``.
    native_model : mandacaru._native.MolecularModel, optional
        A molecular model taken from an earlier result (``result.hamiltonian``)
        to minimize without a geometry. The calculator reuses the context of
        the model, so ``resources`` must not be given. The type is not
        checked at construction: anything else fails with ``AttributeError``.
        Quantum methods only; mutually exclusive with ``hamiltonian``,
        ``load_hamiltonian`` and ``load_crystal``.
    checkpoint : path, optional
        Write the final wavefunction checkpoint (ansatz indices, parameters,
        energy and termination status) to this path at the end of a quantum
        run, and during it with ``checkpoint_every``. A ``"subspace-*"``
        checkpoint also stores the state weights and energies. The classical
        methods ignore it.
    checkpoint_every : int, optional
        Also write ``checkpoint`` during the run (``vqe`` and ``adapt-vqe``).
        ``adapt-vqe`` writes it after every ``checkpoint_every`` accepted growth
        steps. ``vqe`` writes the best point so far after every
        ``checkpoint_every`` cost evaluations. A mid-run checkpoint's status
        says ``"complete": false``, and ``resume=`` continues from it.
        Each write is atomic, so an interrupted run leaves the last whole
        checkpoint. It needs ``checkpoint=`` and a positive integer
        (``ValueError``); ``None`` (default) writes only the final checkpoint.
    resume : path, optional
        Continue a fixed-ansatz or ADAPT run from a checkpoint file written
        by ``checkpoint``; a ``"subspace-*"`` run needs a checkpoint of the
        same method and weights (``ValueError``). The architecture searches
        raise ``NotImplementedError`` at ``run()``; ``transfer=True``
        together with ``resume`` raises ``ValueError``.
    transfer : bool, default False
        Carry the optimized ansatz from the previous geometry of this
        calculator to the next one (for relaxations and scans), accepted only if
        its overlap confidence reaches ``transfer_threshold``. Requires
        ``"adapt-vqe"``, ``"vasqa"`` or ``"valsqa"``, local native execution
        (no circuit execution) and a fresh run without ``resume``; otherwise
        ``ValueError``. A non-boolean raises ``TypeError``.
    transfer_threshold : float, default 0.9
        Confidence in [0, 1] a carried ansatz must reach to be accepted
        (``ValueError`` outside the range). Ignored without ``transfer``.
    transfer_steps : int, optional
        Number of search steps spent on a carried ansatz. Needs
        ``transfer=True`` and ``"vasqa"`` or ``"valsqa"``; a negative or
        non-integer value raises ``ValueError``.
    update_between_geometries : bool, default False
        ``"valsqa"`` only (``ValueError`` otherwise): record that the proposal
        model is meant to be updated between geometries. The graph model is
        carried unchanged; its training is offline.
    pool : {"fermionic", "qubit", "qeb", "ceo", "ceo-ovp", "spin-orbit"}, optional
        The operator pool: fermionic excitations, qubit excitations, qubit
        excitation-based operators, coupled exchange operators (``"ceo"`` and
        its ``"ceo-ovp"`` variant) or spin-orbit-mixing excitations.
        Anything else raises ``ValueError``. ``None`` (the default) chooses by
        method: ``"ceo"`` for the growing ADAPT methods (``"adapt-vqe"`` and
        ``"subspace-adapt-vqe"``) and ``"fermionic"``
        for every other quantum method; an explicit value always wins. The
        table is ``METHODS[method].default_pool`` in ``specifications.py``.
        The fermionic pool can stop above the exact energy where the ``"ceo"``
        pool does not (the 4-cell metallic H chain, 0.0196 eV). ``"ceo"`` is
        generalized over every spin-orbital quadruple, so its size grows as the
        fourth power of the register: at 50 qubits (one electron) it has 588,300 generators
        and a shot-based gradient step needs about 7.8 million Pauli
        observables (``measurement_budget`` refuses that by default, in
        seconds: the planning stops as soon as the distinct strings pass the
        ceiling), where ``pool="fermionic"`` or
        ``"qeb"`` stays within a budget sized for the run. For ``"vqe"``
        the pool supplies the generators of the fixed ansatz.
        ``ansatz="hva"`` requires ``"fermionic"``, which ``"vqe"`` already
        defaults to. The classical methods have no pool and refuse it
        (``ValueError``).
    ansatz : str or dict, default "uccsd"
        ``"uccsd"`` (the pool excitations) or ``"hva"`` (a Hamiltonian
        variational ansatz). Anything else raises ``ValueError``. A dict must
        carry ``"name"``; for ``"uccsd"`` it may carry nothing else. ``"hva"``
        accepts ``layers``, ``evolution`` (``"exact"`` or ``"trotter"``),
        ``order`` (1 or 2), ``steps`` and ``seed_angle``; another key raises
        ``ValueError``. ``"hva"`` requires ``method="vqe"`` and
        ``pool="fermionic"`` (``ValueError``), and turns ``sector`` off and
        ``trotter`` on by default.
    trotter : bool, default False
        Apply the ansatz as an ordered product of operator exponentials, the
        form a circuit implements, instead of the exponential of the sum of the
        generators. The calculator turns it on regardless of this value for
        ``ansatz="hva"``, circuit execution, ``"vasqa"`` and ``"valsqa"``;
        an ADAPT ansatz is always recorded (checkpoint, final state) in
        product form. A non-boolean raises
        ``TypeError``.
    include_singles : bool, default True
        Include single excitations in the pool. A non-boolean raises
        ``TypeError``.
    sector : bool, optional
        Restrict the simulation to the Hilbert-space sector with the target
        electron numbers. ``None`` chooses automatically: on unless the pool
        is ``"qubit"`` or ``"spin-orbit"``, or the ansatz is ``"hva"``, or
        ``taper=True``. Any
        value other than ``True``, ``False`` or ``None`` raises ``TypeError``.
    sparse : bool, default False
        Apply the Hamiltonian as a sparse matrix in the native solver. A
        non-boolean raises ``TypeError``.
    taper : bool, default False
        Remove the :math:`Z_2` symmetries of the qubit Hamiltonian from the
        register, one qubit per independent symmetry (``"vqe"``, ``"adapt-vqe"``
        and the ``"subspace-*"`` methods; other methods raise ``ValueError``). The
        symmetries are read off the Hamiltonian as the mapping wrote it, so the
        spatial parities of the molecule are found beyond the two particle-number
        parities that ``mapping="parity_reduced"`` removes. The sector is the one
        that holds the reference determinant; the energy equals the untapered one
        and the density matrices are read from the state lifted back to the
        fermionic register. Pool generators that do not commute with the
        symmetries are dropped (they cannot act in the reference's sector) and
        listed in the run log. A tapered calculation simulates the whole tapered
        register, so ``sector=True`` raises ``ValueError`` (``None`` resolves to
        ``False``), as do ``ansatz="hva"``, ``transfer=True`` and
        ``save_hamiltonian``; the ``"spin-orbit"`` and ``"qubit"`` pools, which
        need the untapered register, raise at ``run()``. The excited states of
        the ``"subspace-*"`` methods are the ones in the reference's symmetry
        sector. Density matrices need the ``2**n`` fermionic-register state: they
        are refused on a shot-based run. A Hamiltonian with no symmetry beyond the
        identity runs untapered, and the run log says so. A non-boolean raises
        ``TypeError``.
    convergence : dict, optional
        ADAPT stopping criteria, with keys ``gradient`` (largest pool gradient,
        in Hartree units, default 1e-3) and ``energy`` (energy tolerance of the growth
        loop, always in eV even with ``atomic_units=True``, default 1e-3); ``None`` as a value switches that test off. Any other key
        raises ``ValueError``. Ignored by the fixed-ansatz methods.
    max_iterations : int, default 50
        Maximum number of growth steps of an ADAPT run. A step adds one
        operator, or two for a coupled exchange step of the ``"ceo"`` pool, so
        ``num_operators`` can exceed it. This is the growth limit, not the
        optimizer's ``maxiter``.
    quenching : bool, default True
        Whether every parameter is re-optimized after each change (``vqe`` and
        ``adapt-vqe``; another method given ``False`` raises ``ValueError``).
        ``True`` is standard ADAPT-VQE and VQE: each growth step optimizes all
        parameters jointly, warm-started with the new angles at zero.
        ``False`` optimizes only what changed. ADAPT varies the newly appended
        parameters with every earlier one frozen at its optimum. VQE sweeps
        its parameters once, one at a time in order. Each step is cheaper, and
        the energy is an upper bound to the joint optimization's. Not a
        boolean: ``TypeError``.
    tetris : bool, default False
        TETRIS-ADAPT-VQE (``adapt-vqe`` only). Each growth step appends the
        operator of largest gradient, then every further one, in descending
        gradient order, whose qubit support is disjoint from those already
        taken. The step stays one circuit layer, and the ansatz grows in fewer
        steps. ``max_iterations`` still counts growth steps.
    prune : bool, default False
        Pruned-ADAPT-VQE (``adapt-vqe`` only). After each growth step's
        optimization, one operator is removed: the one ranked first by
        ``exp(-10 x) / theta**2`` (``x`` its relative position, the newest
        never), when ``|theta|`` is below a tenth of the mean amplitude of the
        last four. The remaining parameters are not re-optimized.
        ``result.pruned_operators`` lists ``(step, label)`` of each removal.
        Neither option applies to a resumed or transferred ansatz
        (``ValueError``); neither is a boolean: ``TypeError``.
    weights : sequence of float, optional
        Weights of the states in the subspace methods, one per state, positive
        and finite (``ValueError`` at construction) and strictly decreasing
        (``ValueError`` from the solver at ``run()``). Default
        ``num_states, ..., 2, 1``. Validated for every method but used only by
        ``"subspace-vqe"`` and ``"subspace-adapt-vqe"``.
    num_states : int, default 2
        Number of states of the subspace methods. A value below 1 or a
        non-integer raises ``ValueError``. Used only by ``"subspace-*"``.
    atomic_units : bool, default False
        Report results in Hartree and Bohr instead of eV and Angstrom. ASE
        properties (``get_potential_energy``) remain in eV. Also sets the unit
        of the search ``temperature`` and ``length_penalty``. A non-boolean
        raises ``TypeError``.
    trace : bool, default False
        Print the structured run log to standard output. A non-boolean raises
        ``TypeError``.
    txt : path, optional
        Write the structured run log to this file. Independent of ``trace``.
    gradient : str, default "analytic"
        Vestigial. The native solver supports only the analytic gradient, so the
        only accepted value is ``"analytic"``; any other value raises
        ``ValueError``.
    profile : bool, optional
        Whether the per-step ADAPT iteration records include circuit metrics
        (CNOT count and depth). ``None`` computes them for registers of at
        most 12 qubits only; ``True`` always, ``False`` never. Anything else
        raises ``TypeError``.
    execute_circuits : bool, optional
        Evaluate energies by building circuits and running them through the
        SDK provider instead of the native state-vector solver. ``None``
        resolves to ``True`` when ``backend_provider`` names a non-Qiskit
        provider and to ``False`` otherwise. Circuit execution is on whatever
        this says when ``shots > 0`` or ``device`` is not ``"AER_simulator"``,
        so ``False`` never overrides those. It removes forces; the
        ``"subspace-*"`` methods raise ``NotImplementedError`` at ``run()``,
        the architecture searches raise ``NotImplementedError`` at ``run()``,
        and ``transfer`` refuses it. A non-boolean, non-``None`` value raises
        ``TypeError``; ``True`` on a classical method raises ``ValueError``.
    backend_provider : {"qiskit", "braket", "cirq"}, optional
        The quantum SDK that builds and runs the circuits (aliases such as
        ``"ibm"``, ``"aws"`` and ``"google"`` are accepted). ``None`` uses
        the provider of ``device``, else ``"qiskit"``. An unknown name raises
        ``ValueError``, as does a provider that disagrees with the device.
        The Cirq provider runs exact simulation only: it refuses ``shots > 0``
        (``ValueError``) and takes no ``backend_options``. Classical methods
        refuse it.
    device : str, default "AER_simulator"
        Where the circuits run. ``"AER_simulator"`` is the local ideal
        simulator and keeps the native state-vector numerics. Other names
        are Braket devices (``"braket-local"``, ``"braket-sv1"``,
        ``"braket-dm1"``, ``"braket-tn1"``, ``"braket-ionq-aria"``,
        ``"braket-ionq-forte"``, ``"braket-iqm-garnet"``,
        ``"braket-rigetti-ankaa"``), ``"ibm-quantum"`` (the least busy IBM
        processor), ``"fake_<name>"`` (a local rehearsal of an IBM processor,
        for example ``"fake_kingston"``), ``"AER_mps"`` (Aer's matrix-product-state
        simulator for circuits too wide for a state vector; sampled, with an
        optional ``max_bond_dimension`` in ``backend_options``) and
        ``"ibm_<name>"`` (a real IBM processor). An unknown name raises
        ``ValueError``. A fake or real processor and ``"AER_mps"`` require
        ``shots > 0`` (``ValueError``). On an ``ibm_*``
        device the optimization itself runs on the hardware, one job per
        energy evaluation. Classical methods refuse any device other than the
        default.
    shots : int, default 0
        Measurement shots per energy. ``0`` is the exact local expectation
        value; ``> 0`` is a measured energy through the circuit provider and
        every QPU needs it. A non-integer positive value
        raises ``ValueError``. A negative value is refused (``ValueError``)
        only when a circuit provider or device is configured; otherwise it is
        accepted silently and behaves as 0. A shot-based run keeps classical memory polynomial in
        the qubit count: its problem is built register-free, from Pauli operators only (no
        state vector, basis or matrix, up to 50 qubits), so the result has no state and
        density matrices, orbitals, forces, ``sparse=True`` and the subspace methods refuse
        it with an error naming the operation. For ADAPT it measures the pool gradients on
        the device; a fixed-ansatz run with a derivative-free optimizer reports no final
        gradient (NaN). Classical methods refuse it. See ``optimizer`` for the
        default optimizer and the refused ones.
    backend_options : dict, optional
        Keyword options of the provider; ``device`` and ``shots`` are supplied
        by the constructor and must not be repeated. Qiskit takes ``instance``,
        ``token``, ``channel``, ``optimization_level`` (default 3),
        ``estimator_options``, ``physical_qubits``, ``seed``,
        ``max_bases_per_job``, ``enable_fractional_gates`` and ``run_budget``.
        Braket takes ``s3_folder``, ``poll_timeout_seconds`` and
        ``run_budget``. Cirq takes none. An unknown key raises ``TypeError``.
        ``run_budget`` is the run-wide limit on ``jobs`` and ``total_shots``;
        it is on by default for processors (``jobs`` 250, no shot ceiling) and
        ``False`` switches it off. Giving ``backend_options`` without a circuit
        provider, device or ``shots`` raises ``ValueError``.
    measurement_budget : bool or dict, optional
        The ceilings checked before a shot-based measurement is queued, with the keys
        ``observables`` (default 50000), ``bases`` (default 10000),
        ``circuit_instances`` (default 2000000) and ``total_shots`` (default
        none). ``None`` or ``True`` use the defaults, ``False`` switches the
        checks off, and a dict overrides the keys it names (a value of
        ``None`` disables that one check). An unknown key raises
        ``ValueError`` at construction; exceeding a ceiling raises
        ``ValueError`` at ``run()``. Without a circuit provider, device or
        ``shots`` it raises ``ValueError``.
    measurement_scheme : {"qwc", "double-factorized"}, default "qwc"
        How a circuit run measures the energy of ``method="vqe"`` or
        ``"adapt-vqe"``. ``"qwc"`` measures the Pauli terms of the qubit
        Hamiltonian in qubit-wise commuting groups (``O(M^3)`` bases).
        ``"double-factorized"`` measures it in the ``L + 1`` bases of the
        double factorization of the molecular integrals (``O(M)`` bases):
        after a Givens network of the ansatz register, one computational-basis
        measurement answers for a whole squared one-body operator. The energy
        equals the commuting-group energy up to the leaf truncation (below
        ``1e-11`` Ha for H2 and LiH in the ``HAO`` basis; the run reports the
        discarded tensor norm in ``problem.factorized_measurement``); each basis is its own submission, so a run
        costs ``L + 1`` jobs and ``(L + 1) x shots`` shots per energy, which
        ``measurement_budget`` and ``backend_options["run_budget"]`` check
        before the first job. The pool gradients of ``"adapt-vqe"`` stay on
        the commuting groups. It needs a circuit device (``shots > 0``, a
        hardware or fake device, or ``execute_circuits=True``) and molecular
        integrals: a supplied or loaded Hamiltonian, a mapping other than
        Jordan-Wigner, ``taper=True``, ``ansatz="hva"``, a spinor or complex
        model and a crystal raise ``ValueError``; for the other methods the
        option raises ``ValueError``. An unknown name raises ``ValueError``.
    max_steps : int, optional
        Length of the architecture-search chain, in edits. ``"vasqa"`` and
        ``"valsqa"`` only: setting it for any other method raises
        ``ValueError``. Those two methods default to 200.
    min_length : int, default 0
        Smallest ansatz length. The search starts from the empty sequence, so
        only ``0`` is accepted (``ValueError``) by the search methods; ignored
        by the others.
    max_length : int, default 20
        Largest number of operators in a searched ansatz; a positive integer
        (``ValueError``). Ignored outside the search methods.
    move_weights : dict, optional
        Relative frequency of the edit moves, with keys from ``insert``,
        ``delete`` and ``replace``; moves not named get weight 0. ``None``
        weights all three equally. Another key (``swap`` included: there is no
        swap move) raises ``ValueError``. A deletion or a replacement picks
        its position by importance: the operator whose re-insertion gradient
        is smallest is the most likely to go. Ignored outside the search
        methods.
    temperature : float or dict, optional
        Acceptance temperature of the search in the energy unit of the
        results (eV, or Hartree with ``atomic_units=True``). A number is
        constant; a dict with exactly ``initial`` and ``final`` anneals.
        ``None`` anneals from 0.1 eV to 0.001 eV. Ignored outside the search
        methods.
    length_penalty : float, default 0.0
        Energy penalty per operator in the search objective, in the energy unit
        of the results. Ignored outside the search methods.
    warm_start : bool, default True
        Start each re-optimization from the parameters already found. A
        non-boolean raises ``TypeError``. Ignored outside the search methods.
    replace_start : {"zero", "inherit"}, default "zero"
        Starting parameter of an operator introduced by a replace move: zero
        or the replaced operator's value. Anything else raises ``ValueError``.
        Ignored outside the search methods.
    screen_insertions : int, default 0
        Number of candidate insertions screened by gradient before one is
        tried; 0 disables screening. It records its measurements, so it needs
        a ``record`` directory (``ValueError`` without one). Used with the
        search methods; for any other method a nonzero value raises
        ``ValueError``.
    proposal_temperature : float, default 0.2
        Temperature of the gradient-weighted proposal distribution. Ignored
        outside the search methods.
    proposal : {"gradient", "uniform"}, default "gradient"
        How edits are proposed; anything else raises ``ValueError`` for the
        search methods. Ignored outside them.
    proposal_model : path or object, optional
        ``"valsqa"`` only (``ValueError`` for any other method, including
        ``"vasqa"``): a trained proposal model, or its file. ``None`` looks
        for the model file in the ``record`` directory and, without one, falls
        back to gradient proposals; ``False`` disables the model.
    record : path or bool, optional
        Directory where the search stores its problems and edit chains as
        training data. ``None`` uses the directory in ``MANDACARU_PROPOSAL_DATA``
        if set; ``False`` records nothing; ``True`` is refused (``ValueError``).
        Setting it for a method other than ``"vasqa"`` or ``"valsqa"``
        raises ``ValueError``.
    seed : int, optional
        Seed of the search chain, a nonnegative 64-bit integer (``ValueError``
        otherwise); ``None`` draws a fresh seed. Ignored outside the search
        methods.
    kinetic : {"spectral", "fd"}, default "spectral"
        How the kinetic energy is discretized on the grid: the spectral
        (FFT) Laplacian or finite differences. Anything else raises
        ``ValueError`` at ``run()``. Only ``"spectral"`` supports forces
        (``get_forces`` raises) and crystals (``ValueError``). The two
        discretizations give different energies at a given spacing; do not mix
        them in a comparison.
    two_body : str or dict, optional
        Representation of the two-electron integrals. They are always Cholesky
        factors: ``"cholesky"`` or ``{"method": "cholesky", "tolerance": t,
        "representation": r, "truncation": tau}``. ``t`` (eV, default
        ``1e-8``) stops the pivoted Cholesky factorization once the largest
        remaining diagonal integral is at or below it: it bounds each integral,
        not the energy. ``r`` is how the Hamiltonian of the active space holds
        its two-electron tensor: ``"dense"`` (the default, the ``n^4`` tensor)
        or ``"double-factorized"`` (the Cholesky vectors rotated into the active
        orbitals and the eigen decomposition of each, about ``L n^2`` numbers;
        the qubit Hamiltonian is built from them one vector at a time).
        ``tau`` (eV) is required with ``"double-factorized"`` and refused with
        ``"dense"``: eigenpairs are dropped, smallest first, while a bound on
        the change of the two-body operator stays at or below ``tau``, so no
        energy of the model moves by more than ``tau``, plus the roundoff of
        the Pauli coefficients at or below ``1e-12`` Ha that the operator
        drops, as the dense route does (``0`` drops only roundoff). It changes energies; the run log states it. A
        double-factorized model has no dense tensor: analytic forces, MP2
        natural orbitals, ``measurement_scheme="double-factorized"`` and the
        model's ``two_body`` raise ``NotImplementedError``. Any other method,
        representation or key raises ``ValueError``. A crystal builds no
        two-body tensor.
    paw_dataset : str, optional
        Name of the PAW-LCAO or UPAW-LCAO dataset folder in the pseudopotential
        library, for example ``"pbe-sr"``. Without it the folder follows the
        functional (see ``xc``). Valid only with a PAW-LCAO or UPAW-LCAO basis
        (``ValueError`` otherwise, and when the basis dict names another
        folder).
    project_translation : {"auto", True, False}, default "auto"
        How the forces treat rigid translation. ``False`` returns the raw
        negative energy gradient (the net force is then a measure of the grid
        error). ``True`` and ``"auto"`` behave identically, and for a
        single atom nothing is projected: otherwise the mean force is removed.
        For the variational methods with a PAW-LCAO or UPAW-LCAO basis the
        gradient is instead evaluated along the internal (non-rigid)
        directions of the molecule only. Any other value raises ``ValueError``.
    kpts : sequence of int or dict, optional
        Brillouin-zone mesh of a crystal (``"dft"``, and ``"vqe"``,
        ``"adapt-vqe"``, ``"vasqa"``, ``"valsqa"`` on a periodic structure). A
        sequence of three positive integers is a Monkhorst-Pack mesh; a dict
        ``{"size": (n1, n2, n3), "gamma": bool}`` chooses whether the mesh is
        centered on Gamma. The default is the Gamma point alone. Three positive
        integers are required (``ValueError``); a non-periodic direction with
        more than one k-point raises ``ValueError`` at ``run()``. A quantum
        method solves the mesh as its Born-von Karman supercell, which needs a
        Gamma-centered mesh (an even bare triple raises ``ValueError``).
        Ignored for molecules.
    crystal_orbitals : {"k-space", "real-space"}, default "k-space"
        The orbitals a quantum method on a crystal works in: ``"real-space"``,
        the real supercell orbitals, or ``"k-space"``, complex Bloch orbitals
        with momentum-conserving excitations. ``"k-space"`` is not implemented
        yet and raises ``NotImplementedError`` at ``run()``; pass
        ``crystal_orbitals="real-space"``. Any other value raises
        ``ValueError``; a value other than the default on a molecule raises
        ``ValueError`` at ``run()``.
    smearing : float, tuple or dict, optional
        Electronic smearing of a periodic calculation. A number is a Fermi-Dirac
        width in eV; ``(method, width)`` and ``{"method": ..., "width": ...}``
        choose the method, one of ``"fermi-dirac"`` (also ``"fd"``, ``"fermi"``),
        ``"gaussian"`` and ``"methfessel-paxton"`` (``"mp"``), with the width in
        eV. A dict defaults to Fermi-Dirac at 0.1 eV; any other dict key
        raises ``ValueError`` and an unknown method raises at ``run()``. When
        given it overrides the smearing in ``periodic_scf_options``; when not,
        the default is Fermi-Dirac at 0.1 eV. A molecular Kohn-Sham run takes
        Fermi-Dirac only (another method raises ``NotImplementedError``), one
        Fermi level per spin, default width 1e-4 Ha (2.7 meV): a gapped molecule
        is exactly aufbau, and a degenerate frontier shell is shared. A spinor
        Kohn-Sham molecule (spin-orbit dataset) given ``smearing`` raises
        ``NotImplementedError``; a method other than ``dft`` refuses it.
    periodic_scf_options : dict, optional
        Options of the periodic SCF: ``max_iterations`` (default 300),
        ``tolerance`` (default 1e-7), ``density_tolerance`` (default 1e-5),
        ``smearing``, ``width_ha`` (the width in Hartree), ``mixing`` (default
        0.25), ``history`` (default 7, the mixing history length) and ``eigensolver``.
        An unknown key raises ``TypeError`` at ``run()``. Ignored for molecules.
    eigensolver : {"davidson", "direct", "rmm-diis", "mixed"}, default "davidson"
        Eigensolver algorithm for periodic Kohn-Sham diagonalization. ``"direct"``
        performs full dense diagonalization. ``"davidson"`` uses block Davidson
        subspace iteration. ``"rmm-diis"`` uses residual minimization with DIIS.
        ``"mixed"`` begins SCF iterations using Davidson and switches to RMM-DIIS
        once the density residual passes 0.01. Periodic DFT only.
    symmetry : bool, default True
        Reduce the k-mesh with the space-group operations of the cell (time
        reversal is always used). Spinor (spin-orbit) calculations never use
        the space-group reduction. A collinear moment pattern keeps the
        operations that map every atom onto an atom of the same element and
        the same initial moment (an operation that would send a moment to its
        opposite is not kept); with ``symmetry=False`` nonzero initial
        magnetic moments keep the full mesh. A hybrid functional (``xc="hse06"``) always keeps the
        whole mesh, neither reduced by the space group nor halved by time
        reversal, because its exact exchange sums over the listed k-points.
        A non-boolean raises ``TypeError``. Ignored for molecules.
    hubbard_u : dict, optional
        The on-site DFT+U correction (fully localized double counting) of a
        ``method="dft"`` calculation: ``{"Ni": {"l": "d", "U": 6.0, "J": 0.0}}``
        names, per element, the correlated shell (``"d"`` or ``"f"``) and its
        ``U`` and ``J`` in eV (``J`` defaults to 0; ``U_eff = U - J``) and, optionally, the
        functional ``"form"``: ``"dudarev-fll"`` (the default, simplified rotationally
        invariant form of Dudarev et al.) or ``"liechtenstein-fll"`` (separate ``U`` and
        ``J`` through Slater integrals). The
        occupation matrix of the shell is the PAW partial-wave projection, so a
        filled shell has an occupation below 1 (Ni 3d 0.95). Refused with
        ``ValueError``: a malformed mapping, an unknown element, a shell other
        than ``d`` or ``f``, a negative or non-finite energy, ``J`` above
        ``U``, ``precision="float32"``, a molecule with a method other than
        ``"dft"``, a basis that is not PAW-LCAO or UPAW-LCAO (the shell is a projection on a
        PAW partial wave), an element missing from the geometry, and a dataset
        without the shell or whose reference atom holds no electron in it
        (Ba ``d``, Au ``f`` in ``lda-sr``). A k mesh reduced by symmetry is
        supported: the occupation matrices are completed over the kept
        operations and time reversal, so the reduced mesh gives the whole
        mesh's (a magnetic crystal keeps the operations that preserve its
        moment pattern, as without ``hubbard_u``); ``hubbard_response`` on such a mesh needs every
        ``hubbard_u`` atom fixed by every kept operation and raises
        ``ValueError`` naming ``symmetry=False`` otherwise.
        ``NotImplementedError``: a screened hybrid functional and a spin-orbit
        dataset. Other methods raise ``ValueError``. The refusals that need
        the geometry (element, dataset, spin-orbit) are raised by the
        constructor when ``atoms`` is given and at the first calculation
        otherwise, always before any SCF (a molecule's are always raised at the first
        calculation, ``ValueError`` for an element or shell it lacks and
        ``NotImplementedError`` for a spin-orbit dataset; its analytic forces include
        the +U term). With
        ``load_crystal`` the file owns the correction: leave the option out, or
        give the file's own values (a different mapping raises ``ValueError``
        at the first calculation, when the file is read). The result carries
        ``hubbard_energy`` and the per-shell ``hubbard`` occupation matrices.
    load_crystal : path, optional
        Load a crystal setup written by ``save_crystal`` instead of building
        it; run it without a geometry (``method="dft"`` only, otherwise
        ``NotImplementedError``). The functional is that of the file (a
        different explicit ``xc`` raises ``ValueError``). Mutually exclusive
        with ``hamiltonian``, ``native_model`` and ``load_hamiltonian``.
    save_crystal : path, optional
        Write the periodic setup (before the SCF) to this path for reuse with
        ``load_crystal``. Periodic calculations only.
    exxdiv_treatment : str, optional
        Treatment of the divergence of the exact exchange at the zero
        momentum transfer of a periodic hybrid (``xc="hse06"``), one of
        ``"none"`` (the default: the finite point limit ``pi / omega**2`` of
        the screened Coulomb kernel) and ``"gygi-baldereschi"`` (the kernel
        is averaged over the Voronoi sphere of the k-point, which speeds the
        convergence with the k mesh). These two are the only values; any
        other spelling raises ``ValueError``. It applies to a periodic hybrid
        only: with a non-hybrid functional (``ValueError`` at construction,
        or at the first calculation with ``load_crystal``), a molecule
        (``ValueError`` at the first calculation) any value but ``"none"``
        raises, and giving it to a method other than ``"dft"`` raises
        ``ValueError`` whatever the value. Giving it in
        ``periodic_scf_options`` raises ``ValueError``. Used by
        ``method="dft"`` only.
    gygi_baldereschi : bool, optional
        The one shorthand for ``exxdiv_treatment="gygi-baldereschi"``:
        ``True`` selects that treatment, ``False`` selects ``"none"``. Giving
        it together with an ``exxdiv_treatment`` that says otherwise raises
        ``ValueError``, a non-boolean raises ``TypeError``, and it is refused
        wherever ``exxdiv_treatment`` is. Used by ``method="dft"`` only.

    Notes
    -----
    A ``method="dft"`` crystal calculation (any periodic direction, or
    ``load_crystal``) needs a PAW-LCAO or UPAW-LCAO basis. It refuses
    ``active_space``, ``save_hamiltonian``, ``scf_options``, ``n_electrons``,
    ``mapping`` and ``kinetic="fd"`` with ``ValueError``.
    """

    def __init__(
        self,
        method=DEFAULT_METHOD,
        *,
        atoms=None,
        hamiltonian=None,
        n_spatial_orbitals=None,
        num_particles=None,
        hamiltonian_format=None,
        basis="HAO",
        grid_spacing=0.2,
        precision="float64",
        grid=None,
        resources=None,
        xc=None,
        vdw=None,
        spin=False,
        unrestricted=False,
        charge=0,
        n_electrons=None,
        scf_options=None,
        optimizer=None,
        mapping=None,
        active_space=None,
        load_hamiltonian=None,
        save_hamiltonian=None,
        native_model=None,
        checkpoint=None,
        checkpoint_every=None,
        resume=None,
        transfer=False,
        transfer_threshold=0.9,
        transfer_steps=None,
        update_between_geometries=False,
        pool=None,
        ansatz="uccsd",
        trotter=False,
        include_singles=True,
        sector=None,
        sparse=False,
        taper=False,
        convergence=None,
        max_iterations=50,
        quenching=True,
        tetris=False,
        prune=False,
        weights=None,
        num_states=2,
        atomic_units=False,
        trace=False,
        txt=None,
        gradient="analytic",
        profile=None,
        execute_circuits=None,
        backend_provider=None,
        device="AER_simulator",
        shots=0,
        backend_options=None,
        measurement_budget=None,
        measurement_scheme="qwc",
        max_steps=None,
        min_length=0,
        max_length=20,
        move_weights=None,
        temperature=None,
        length_penalty=0.0,
        warm_start=True,
        replace_start="zero",
        screen_insertions=0,
        proposal_temperature=0.2,
        proposal="gradient",
        proposal_model=None,
        record=None,
        seed=None,
        kinetic="spectral",
        two_body=None,
        paw_dataset=None,
        project_translation="auto",
        kpts=None,
        crystal_orbitals="k-space",
        smearing=None,
        periodic_scf_options=None,
        eigensolver="davidson",
        symmetry=True,
        hubbard_u=None,
        load_crystal=None,
        save_crystal=None,
        exxdiv_treatment=None,
        gygi_baldereschi=None,
    ):
        super().__init__()
        # Every constructor option: the finite-difference forces of a crystal rebuild the same
        # calculation at displaced geometries (integrations/bloch.py::finite_difference_forces).
        self._constructor_options = {
            name: value for name, value in locals().items() if name not in ("self", "__class__")
        }
        self._method, self._capabilities = resolve_method(method)
        _refuse_inapplicable(method, locals())
        if precision not in ("float32", "float64"):
            raise ValueError("precision must be 'float32' or 'float64'")
        self._precision = precision
        if gradient != "analytic":
            raise ValueError("the native state-vector solver uses analytic gradients")
        for name, flag in (("quenching", quenching), ("tetris", tetris), ("prune", prune)):
            if type(flag) is not bool:
                raise TypeError(f"{name} must be True or False")
        self._quenching = quenching
        self._tetris = tetris
        self._prune = prune
        if profile is not None and type(profile) is not bool:
            raise TypeError("profile must be True, False or None")
        if execute_circuits is not None and type(execute_circuits) is not bool:
            raise TypeError("execute_circuits must be a boolean or None")
        if not self._capabilities.quantum:
            requested = [
                name
                for name, given in (
                    ("shots", shots > 0),
                    ("execute_circuits", execute_circuits is True),
                    ("backend_provider", backend_provider is not None),
                    ("device", device != "AER_simulator"),
                    ("backend_options", backend_options is not None),
                    ("measurement_budget", measurement_budget is not None),
                )
                if given
            ]
            if requested:
                raise ValueError(
                    f"{', '.join(requested)} apply to quantum methods: method={method!r} "
                    "is a classical mean-field calculation with no quantum state to measure"
                )
        if execute_circuits is None:
            # Naming a non-Qiskit provider is a request to use it; the Qiskit
            # default keeps the fast state-vector numerics unless asked.
            from .backends.providers import normalize_provider

            execute_circuits = (
                backend_provider is not None
                and normalize_provider(backend_provider) != "qiskit"
            )
        self._execute_circuits = execute_circuits or shots > 0 or device != "AER_simulator"
        if precision == "float32" and self._capabilities.quantum:
            supplied = [
                name
                for name, given in (
                    ("hamiltonian", hamiltonian is not None),
                    ("load_hamiltonian", load_hamiltonian is not None),
                    ("native_model", native_model is not None),
                )
                if given
            ]
            if supplied and self._execute_circuits:
                raise ValueError(
                    f"precision='float32' acts on the sampled orbitals and the state "
                    f"vector; {', '.join(supplied)} on a circuit device (shots > 0, a "
                    "hardware device or execute_circuits=True) builds neither, so it "
                    "has nothing to store in float32: use precision='float64'"
                )
        self._device = device
        self._shots = shots
        from .integrations.transfer import configure as configure_transfer

        configure_transfer(self, transfer=transfer, transfer_threshold=transfer_threshold,
                           transfer_steps=transfer_steps, update_between_geometries=update_between_geometries,
                           resume=resume)
        self._profile = profile
        if xc is not None:
            if not isinstance(xc, str):
                raise TypeError(f"xc must be a string or None, not {type(xc).__name__}")

        vdw_norm = None
        if vdw is not None:
            if not isinstance(vdw, (str, bool)):
                raise TypeError("vdw must be a string, boolean, or None")
            if isinstance(vdw, bool):
                vdw_norm = "rvv10" if vdw else None
            else:
                vdw_norm = vdw.strip().lower()

            if vdw_norm is not None:
                if vdw_norm in ("rvv10", "ts"):
                    if vdw_norm == "rvv10":
                        base_xc = "pbe" if xc is None else xc.strip().lower()
                        if not base_xc.endswith("+rvv10"):
                            if base_xc in ("pbe", "r2scan"):
                                xc = f"{base_xc}+rvv10"
                            else:
                                raise ValueError(
                                    f"vdw='rvv10' can only be combined with 'pbe' or 'r2scan', not {xc!r}"
                                )
                    elif vdw_norm == "ts":
                        base_xc = "lda" if xc is None else xc.strip().lower()
                        if base_xc.endswith("+rvv10"):
                            raise ValueError("cannot combine vdw='ts' with an rvv10 functional")
                else:
                    raise ValueError(f"unknown vdw choice {vdw!r}; supported vdw options are 'rvv10' and 'ts'")
        elif xc is not None and xc.strip().lower().endswith("+rvv10"):
            vdw_norm = "rvv10"

        if xc is not None and xc not in FUNCTIONALS:
            raise ValueError(f"expected a canonical XC name in {FUNCTIONALS}")
        if mapping is not None and mapping not in MAPPINGS:
            raise ValueError(f"expected a canonical mapping in {MAPPINGS}")
        if pool is None:
            # The one owner of the default: ``METHODS[method].default_pool`` (``None`` for a
            # classical method, which has no pool and refuses an explicit one above).
            pool = self._capabilities.default_pool
        elif pool not in OPERATOR_POOLS:
            raise ValueError("unknown native operator pool")
        ansatz_spec = {"name": ansatz} if isinstance(ansatz, str) else dict(ansatz)
        if ansatz_spec.get("name") not in ANSATZ_NAMES:
            raise ValueError("native variational ansätze are uccsd and hva")
        self._hva = None
        if ansatz_spec["name"] == "hva":
            if method != "vqe" or pool != "fermionic":
                raise ValueError("HVA uses fixed VQE Hamiltonian groups")
            if set(ansatz_spec) - {
                "name",
                "layers",
                "evolution",
                "order",
                "steps",
                "seed_angle",
            }:
                raise ValueError("unknown HVA option")
            self._hva = {k: v for k, v in ansatz_spec.items() if k != "name"}
            if self._hva.get("layers", 2) == 1:
                warnings.warn(
                    "ansatz='hva' with layers=1 cannot lower the Hartree-Fock energy: each "
                    "Hamiltonian group acts on the real reference with an imaginary relative "
                    "amplitude, so the energy has no first-order term. The result is exactly "
                    "the Hartree-Fock energy; use layers >= 2.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        elif len(ansatz_spec) != 1:
            raise ValueError("uccsd accepts only its name")
        self._pool = pool
        if sector is not None and type(sector) is not bool:
            raise TypeError(
                "sector must be True, False or None for automatic selection"
            )
        if type(project_translation) is not bool and project_translation != "auto":
            raise ValueError("project_translation must be True, False or 'auto'")
        self._project_translation = project_translation
        if not np.isfinite(grid_spacing) or grid_spacing <= 0:
            raise ValueError("grid_spacing must be positive and finite")
        if type(charge) is not int:
            raise TypeError("charge must be an integer")
        for value in (
            spin,
            unrestricted,
            trotter,
            include_singles,
            sparse,
            atomic_units,
            trace,
            symmetry,
        ):
            if type(value) is not bool:
                raise TypeError("boolean calculation controls require booleans")
        if native_model is not None and resources:
            raise ValueError(
                "a supplied model retains its context; configure resources on its originating calculation"
            )
        self.resources = CalculationResources(
            **dict(resources or {}),
            context=None if native_model is None else native_model.context,
        )
        self.context = self.resources.context
        self.provider = None
        #: Why the run budget ended the last run early, or ``None``.
        self.run_budget_exhausted = None
        self._gradient_measurement = None
        self._energy_measurement = None
        if self._execute_circuits or backend_provider is not None:
            from .integrations.circuits import configure

            configure(self, provider=backend_provider, device=device, shots=shots,
                      options=backend_options, measurement_budget=measurement_budget)
        elif backend_options is not None or measurement_budget is not None:
            raise ValueError("backend options require a circuit provider or device")
        from copy import deepcopy

        basis = deepcopy(basis)
        self._basis = basis
        from .pseudopotentials.families import lookup_family

        family = lookup_family(basis if isinstance(basis, str) else basis.get("name"))
        self._basis_spec = None if family is not None else basis_specification(basis)
        if paw_dataset is not None:
            if self._basis_spec is not None:
                raise ValueError("paw_dataset applies to a PAW-LCAO or UPAW-LCAO basis")
            selected = {"name": basis} if isinstance(basis, str) else dict(basis)
            if "paw_dataset" in selected and selected["paw_dataset"] != paw_dataset:
                raise ValueError("basis and calculator specify different PAW datasets")
            selected["paw_dataset"] = paw_dataset
            basis = selected
            self._basis = basis
        if family is not None:
            family.validate_options(
                {} if isinstance(basis, str)
                else {key: value for key, value in basis.items() if key != "name"}
            )
        self._vdw = vdw_norm
        self._requested_xc = xc
        self._xc = "lda" if xc is None else xc
        self._unrestricted = spin or unrestricted
        self._charge = charge
        self._n_electrons = n_electrons
        self._scf_options = dict(scf_options or {})
        self._mapping = mapping or "jordan_wigner"
        self._active_spec = active_space
        self._grid_spacing = float(grid_spacing)
        self._explicit_grid = grid
        self._kinetic = kinetic
        tolerance_ev = DEFAULT_TWO_BODY_TOLERANCE_EV
        self._two_body_representation = "dense"
        self._two_body_truncation_ev = self._two_body_truncation_ha = None
        self._loaded_two_body = None
        if two_body is not None:
            representation = (
                {"method": two_body} if isinstance(two_body, str) else dict(two_body)
            )
            if (
                set(representation) - {"method", "tolerance", "representation", "truncation"}
                or representation.get("method", "cholesky") not in TWO_BODY_METHODS
            ):
                raise ValueError(
                    "two_body accepts Cholesky factors with their tolerance, and a "
                    "representation ('dense' or 'double-factorized') with its truncation"
                )
            truncation_ev = representation.get("truncation")
            if truncation_ev is not None and (
                    isinstance(truncation_ev, bool)
                    or not isinstance(truncation_ev, (int, float))):
                raise ValueError("two_body truncation must be an energy in eV")
            truncation_ha = (None if truncation_ev is None
                             else _native.energy_to_hartree(float(truncation_ev), "eV"))
            self._two_body_representation = _native.two_body_representation(
                representation.get("representation", "dense"), truncation_ha)
            self._two_body_truncation_ev = (
                None if truncation_ev is None else float(truncation_ev))
            self._two_body_truncation_ha = truncation_ha
            tolerance_ev = representation.get("tolerance", tolerance_ev)
            if (isinstance(tolerance_ev, bool) or not isinstance(tolerance_ev, (int, float))
                    or not np.isfinite(tolerance_ev) or tolerance_ev <= 0):
                raise ValueError("two_body tolerance must be a finite positive energy in eV")
        # The Cholesky stopping threshold: given in eV, held in Hartree for the engine.
        self._two_body_tolerance_ev = float(tolerance_ev)
        self._cholesky_tolerance = _native.energy_to_hartree(float(tolerance_ev), "eV")
        self._two_body_requested = two_body is not None
        self._load_hamiltonian = load_hamiltonian
        self._save_hamiltonian = save_hamiltonian
        if (
            sum(x is not None for x in (native_model, load_hamiltonian, load_crystal))
            > 1
        ):
            raise ValueError("supply only one native model or cached problem")
        self._supplied_model = native_model
        self._supplied_hamiltonian = hamiltonian
        self._hamiltonian_orbitals = n_spatial_orbitals
        self._hamiltonian_particles = num_particles
        if hamiltonian_format is not None:
            from .core.serialization import resolve_format

            hamiltonian_format = resolve_format(hamiltonian_format)
        self._hamiltonian_format = hamiltonian_format
        if hamiltonian is not None:
            if (
                load_hamiltonian is not None
                or native_model is not None
                or load_crystal is not None
            ):
                raise ValueError(
                    "supply one Hamiltonian, native model, or cached problem"
                )
            if n_spatial_orbitals is None or num_particles is None:
                raise ValueError(
                    "a supplied Hamiltonian needs n_spatial_orbitals and num_particles"
                )
        self._checkpoint_path = checkpoint
        if checkpoint_every is not None and (
            type(checkpoint_every) is not int or checkpoint_every < 1 or checkpoint is None
        ):
            raise ValueError(
                "checkpoint_every must be a positive integer, together with checkpoint="
            )
        self._checkpoint_every = checkpoint_every
        self._resume_path = resume
        self._trotter = trotter or self._hva is not None or self._execute_circuits or method in ("vasqa", "valsqa")
        self._include_singles = include_singles
        from .integrations import tapering

        self._taper = tapering.validate(
            taper,
            sector=sector,
            hva=self._hva is not None,
            transfer=self._transfer,
            save_hamiltonian=save_hamiltonian,
        )
        from .integrations import measurement_scheme as scheme_options

        self._measurement_scheme = scheme_options.validate(
            measurement_scheme,
            execute_circuits=self._execute_circuits,
            hamiltonian=hamiltonian,
            load_hamiltonian=load_hamiltonian,
            mapping=mapping,
            taper=self._taper,
            hva=self._hva is not None,
        )
        self._sector = (
            (pool not in ("qubit", "spin-orbit") and self._hva is None and not self._taper)
            if sector is None
            else sector
        )
        self._sparse = sparse
        self._max_growth = max_iterations
        if type(num_states) is not int or num_states < 1:
            raise ValueError("num_states must be a positive integer")
        self._weights = (
            np.arange(num_states, 0, -1, dtype=float)
            if weights is None
            else np.asarray(weights, dtype=float).copy()
        )
        if (
            self._weights.shape != (num_states,)
            or not np.isfinite(self._weights).all()
            or np.any(self._weights <= 0)
        ):
            raise ValueError(
                "weights must contain one positive finite value per requested state"
            )
        self._convergence = dict(
            {"gradient": 1e-3, "energy": 1e-3} if convergence is None else convergence
        )
        self._atomic_units = atomic_units
        self._search = None
        self._search_geometry = None
        if method in ("vasqa", "valsqa"):
            from .integrations.search import configure

            configure(self, max_steps=200 if max_steps is None else max_steps,
                      min_length=min_length, max_length=max_length,
                      move_weights=move_weights, temperature=temperature,
                      length_penalty=length_penalty, warm_start=warm_start,
                      replace_start=replace_start, screen_insertions=screen_insertions,
                      proposal_temperature=proposal_temperature, proposal=proposal,
                      record=record, seed=seed, proposal_model=proposal_model)
        elif max_steps is not None or record is not None or screen_insertions or proposal_model is not None:
            raise ValueError("architecture-search controls require vasqa or valsqa")
        self.trace = trace
        self.txt = txt
        requested = (
            dict(method=optimizer)
            if isinstance(optimizer, str)
            else (
                {
                    k: v
                    for k, v in {
                        "method": optimizer.method,
                        "maxiter": optimizer.maxiter,
                        "tol": optimizer.tol,
                        **getattr(optimizer, "options", {}),
                    }.items()
                    if k in {"method", "maxiter", "maxfev", "tol", "track_best"}
                    and v is not None
                }
                if hasattr(optimizer, "method") and not isinstance(optimizer, dict)
                else dict(optimizer or {})
            )
        )
        # A complex64 state is not resolved below the floors Rust owns: the
        # optimizer tolerance left unset resolves to the energy floor (printed in
        # the optimization setup), a value given below it raises.
        float32_state = (
            precision == "float32"
            and self._capabilities.quantum
            and not self._execute_circuits
        )
        defaults = {
            "method": "SLSQP",
            "maxiter": 2000,
            "tol": _native.FLOAT32_ENERGY_FLOOR_HA if float32_state else 1e-12,
        }
        if shots > 0 and requested.get("method", "SPSA") == "SPSA":
            # A measured energy: the strict two-evaluation SPSA, whose
            # 2 * maxiter + 1 evaluations are each one job on a processor.
            defaults = {"method": "SPSA", "maxiter": 100, "tol": None, "track_best": False}
        self._optimizer = {**defaults, **requested}
        if self._optimizer["method"] not in _native.OPTIMIZERS:
            raise ValueError(
                f"expected a native optimizer in {tuple(_native.OPTIMIZERS)}"
            )
        unknown = sorted(set(self._optimizer) - {"method", "maxiter", "maxfev", "tol", "track_best"})
        if unknown:
            raise ValueError(f"unknown optimizer option(s) {unknown}; optimizer options are method, maxiter, maxfev, tol and track_best")
        if shots > 0 and self._optimizer["method"] in NOISE_INTOLERANT:
            raise ValueError(
                f"optimizer {self._optimizer['method']!r} differences the energy at tiny "
                "steps, which turns shot noise into gradient noise; with "
                "shots > 0 use 'SPSA' (the default) or 'COBYLA'"
            )
        if set(self._convergence) - {"gradient", "energy"}:
            raise ValueError("convergence options are gradient and energy")
        if float32_state:
            _check_float32_floors(
                self._optimizer["method"],
                self._optimizer.get("tol"),
                self._convergence,
                adapt=method.endswith("adapt-vqe"),
            )
        if not self.capabilities.quantum and (
            load_hamiltonian is not None
            or native_model is not None
            or hamiltonian is not None
        ):
            raise ValueError(
                "mean-field calculations require a geometry or cached crystal"
            )
        self._geometry = None
        self._attached_atoms = None
        self._force_grid = None
        self.integrals = None
        self.native_reference = None
        self.problem = None
        self._solver_instance = None
        self.result = None
        self.results = {}
        self.checkpoint = None
        #: What the run log states about the run: the checkpoint it resumed
        #: from, the recipe of its PAW basis, the seconds of its stages.
        self._resume_lineage = None
        self._basis_report = None
        self._stages = {}
        self._exit_hook_armed = False
        self._load_crystal = load_crystal
        self._save_crystal = save_crystal
        self._kpts_spec = (
            {"size": (1, 1, 1), "gamma": True}
            if kpts is None
            else dict(kpts)
            if isinstance(kpts, dict)
            else {"size": tuple(kpts), "gamma": False}
        )
        size = self._kpts_spec.get("size", (1, 1, 1))
        if len(size) != 3 or any(type(n) is not int or n < 1 for n in size):
            raise ValueError("kpts needs three positive integer mesh sizes")
        if crystal_orbitals not in ("k-space", "real-space"):
            raise ValueError(
                f"crystal_orbitals must be 'k-space' or 'real-space', got {crystal_orbitals!r}"
            )
        self._crystal_orbitals = crystal_orbitals
        self._periodic_options = dict(periodic_scf_options or {})
        # The occupations of a molecular Kohn-Sham run (``smearing=``); empty: Rust's default.
        self._molecular_smearing = {}
        if eigensolver is not None:
            eigen_norm = str(eigensolver).strip().lower().replace("_", "-")
            if eigen_norm not in ("direct", "davidson", "rmm-diis", "mixed"):
                raise ValueError(
                    f"unknown eigensolver {eigensolver!r}; supported options are 'direct', 'davidson', 'rmm-diis', 'mixed'"
                )
            self._periodic_options.setdefault("eigensolver", eigen_norm)
        self._periodic_options.setdefault("switch_tolerance", 0.01)
        self._periodic_options["exxdiv_treatment"] = _exchange_divergence(
            exxdiv_treatment, gygi_baldereschi, periodic_scf_options
        )
        if load_crystal is None and self._periodic_options["exxdiv_treatment"] != "none":
            # A crystal file owns its functional: it is checked when the file is read.
            _refuse_exchange_divergence_without_hybrid(
                self._periodic_options["exxdiv_treatment"], self._xc, periodic=True
            )
        self._periodic_symmetry = symmetry
        self._hubbard_u = hubbard.validate(hubbard_u, xc=self._xc, precision=precision)
        if atoms is not None:
            hubbard.require_crystal(self._hubbard_u, atoms.pbc, method=self._method)
        # A crystal file owns its basis; otherwise the option needs a PAW dataset.
        paw = None if load_crystal is not None else hubbard.paw_basis(self._hubbard_u, basis)
        if atoms is not None:
            hubbard.check_atoms(self._hubbard_u, self.context, atoms, paw, self._xc)
            if self._method == "dft" and any(atoms.pbc):
                moments = atoms.get_initial_magnetic_moments()
                if moments is not None and np.ndim(moments) != 1:
                    raise NotImplementedError(
                        "noncollinear initial moments (one vector per atom) start a molecule's "
                        "spinor Kohn-Sham; a crystal takes collinear moments (with a spin-orbit "
                        "dataset they start its noncollinear magnetization)"
                    )
                if moments is not None and any(moments):
                    self._unrestricted = True
        self._periodic_basis = basis
        # Periodic DFT holds no two-body tensor to represent: a representation other than
        # the dense default is refused there (`integrations.paw.configure`).
        self._periodic_two_body_requested = self._two_body_representation != "dense"
        self._requested_mapping = mapping
        self._save_hamiltonian = save_hamiltonian
        self.options = {
            "threads": self.context.threads,
            "memory_limit_bytes": self.context.memory_limit_bytes,
        }
        if smearing is not None:
            from numbers import Real

            if isinstance(smearing, dict):
                if set(smearing) - {"method", "width"}:
                    raise ValueError("smearing accepts method and width")
                method, width = (
                    smearing.get("method", "fermi-dirac"),
                    smearing.get("width", 0.1),
                )
            else:
                method, width = (
                    ("fermi-dirac", smearing)
                    if isinstance(smearing, Real)
                    else smearing
                )
            width_ha = _native.energy_to_hartree(float(width), "eV")
            self._periodic_options.update(smearing=method, width_ha=width_ha)
            self._molecular_smearing = {"smearing": method, "width": width_ha}

        if atoms is not None:
            atoms.calc = self

    @property
    def solver(self):
        if getattr(self, "_solver_instance", None) is not None:
            return self._solver_instance
        if self.capabilities.quantum and hasattr(self, "_optimizer"):
            from types import SimpleNamespace
            from .optimizers import Optimizer
            opt = Optimizer(
                method=self._optimizer["method"],
                maxiter=self._optimizer.get("maxiter", 300),
                tol=self._optimizer.get("tol", 1e-9),
            )
            return SimpleNamespace(optimizer=opt)
        return None

    @solver.setter
    def solver(self, val):
        self._solver_instance = val

    @property
    def spin(self):
        return self._unrestricted

    @property
    def unrestricted(self):
        return self._unrestricted

    @property
    def method(self):
        return self._method

    @property
    def capabilities(self):
        return self._capabilities

    @property
    def xc(self):
        return self._xc

    @property
    def vdw(self):
        return self._vdw

    @property
    def basis(self):
        import copy

        return copy.deepcopy(self._basis)

    @property
    def implemented_properties(self):
        values = ["energy", "free_energy"]
        if self.method == "dft" or self.capabilities.quantum:
            values.append("stress")
        if self.method == "dft":
            values.append("magmom")
        # A circuit device's force is the device force (`integrations/device_forces.py`); a periodic
        # one is refused by `bloch.check_derivatives`.
        if (
            self.capabilities.forces
            and not self._taper
            and self._kinetic == "spectral"
        ):
            values.append("forces")
        return values

    @property
    def n_qubits(self):
        if self.problem is not None:
            return self.problem.num_qubits
        if self.result is None:
            m = self._hamiltonian_orbitals
            if m is None and self._supplied_model is not None:
                m = self._supplied_model.n_spatial_orbitals
            if m is not None:
                return 2 * m - (2 if self._mapping == "parity_reduced" else 0)
            raise RuntimeError("run a calculation before requesting the register size")
        if self._active_spec is not None:
            return self.result.qubit_hamiltonian().num_qubits
        return 2 * self.integrals.n_orbitals - (
            2 if self._mapping == "parity_reduced" else 0
        )

    @property
    def num_particles(self):
        if self.problem is not None:
            return tuple(self.problem.num_particles)
        if self.result is None:
            if self._hamiltonian_particles is not None:
                return tuple(self._hamiltonian_particles)
            if self._supplied_model is not None:
                return tuple(self._supplied_model.num_particles)
            raise RuntimeError("run a calculation before requesting electron counts")
        return self.result.num_particles

    def _grid(self, geometry):
        grid = self._force_grid if self._force_grid is not None else self._explicit_grid
        if grid is None:
            return None
        if isinstance(grid, _native.SpatialGrid):
            return grid
        return _native.SpatialGrid(
            grid.shape,
            [float(grid.X.flat[0]), float(grid.Y.flat[0]), float(grid.Z.flat[0])],
            np.ascontiguousarray(grid.step),
            periodic=bool(grid.periodic),
        )

    def natural_orbitals(self, *, state=0):
        """Return native natural occupations and atomic/MO orbital coefficients."""
        if self.result is None:
            raise RuntimeError("run a calculation before requesting natural orbitals")
        return self.result.natural_orbitals(state=state)

    def volumetric_field(
        self,
        quantity="density",
        index=0,
        *,
        state=0,
        component="auto",
        grid=None,
        grid_spacing=None,
    ):
        """Return a native sampled molecular field, optionally on a finer grid."""
        if self.result is None:
            raise RuntimeError("run a calculation before requesting a molecular field")
        if grid_spacing is not None:
            if grid is not None:
                raise ValueError("specify grid or grid_spacing, not both")
            if not np.isfinite(grid_spacing) or grid_spacing <= 0:
                raise ValueError("field grid spacing must be positive and finite")
            if self._geometry is None:
                raise ValueError("field resampling requires the molecular geometry")
            grid = _native.SpatialGrid.from_geometry(
                self._geometry.native(), [to_bohr(float(grid_spacing))] * 3
            )
        return self.result.volumetric_field(
            quantity, index, state=state, component=component, grid=grid
        )

    def write_cube(self, path, quantity="density", index=0, **options):
        """Sample a molecular field and write its cube file using Rust."""
        return self.volumetric_field(quantity, index, **options).write(
            path, format="cube"
        )

    def _converged_state(self, solver=None):
        if solver is None:
            solver = self.solver
        if hasattr(solver, "ansatz") and hasattr(solver, "result"):
            params = getattr(solver.result, "optimal_parameters", None)
            if params is None and isinstance(solver.result, dict):
                params = solver.result.get("parameters")
            if params is not None:
                return solver.ansatz.state(params)
        raise ValueError("solver has no converged state")

    def _state_rdms(self, solver=None, psi=None, two_body=True):
        if solver is None:
            solver = self.solver
        if psi is None:
            psi = self._converged_state(solver)
        problem = getattr(solver, "problem", None)
        if problem is not None and hasattr(problem, "rdms"):
            return problem.rdms(np.ascontiguousarray(psi, dtype=complex), two_body=two_body)
        raise ValueError("problem has no rdms method")

    def finite_size_correction(self, scheme="mpc", *, state=None, shells=None):
        r"""Diagnostic finite-size corrections for periodic supercell calculations.

        Parameters
        ----------
        scheme : {"mpc", "ccmh", "kzk"}, default "mpc"
            Which correction scheme to evaluate:
            - ``"mpc"``: Model Periodic Coulomb hole correction with Madelung double-counting fix.
            - ``"kzk"``: Kwee-Zhang-Krakauer size-dependent LDA functional.
            - ``"ccmh"``: Chiesa-Ceperley-Martin-Holzmann structure factor extrapolation.
        state : array_like, optional
            State vector to evaluate from (if supported by method).
        shells : int, optional
            Lattice or reciprocal shells to use.

        Returns
        -------
        KZKCorrection, FiniteSizeCorrection, or CCMHCorrection
        """
        scheme = str(scheme).strip().lower()
        if scheme not in ("mpc", "ccmh", "kzk"):
            raise ValueError(
                f"unknown finite-size scheme {scheme!r}; choose 'mpc' (model "
                "periodic Coulomb), 'ccmh' (Chiesa-Ceperley-Martin-Holzmann) "
                "or 'kzk' (Kwee-Zhang-Krakauer)."
            )
        geom = self._geometry
        is_periodic = False
        if geom is not None and any(geom.pbc):
            is_periodic = True
        elif hasattr(self, "_attached_atoms") and self._attached_atoms is not None and any(self._attached_atoms.pbc):
            is_periodic = True
        elif hasattr(self, "atoms") and self.atoms is not None and any(self.atoms.pbc):
            is_periodic = True

        if not is_periodic:
            raise ValueError(
                f"method {self._method!r} is not periodic, so its electrons "
                "have no images and there is no exchange-correlation hole "
                "self-interaction to remove. The correction is defined "
                "against the periodic Coulomb kernel; a molecule is already "
                "computed with the isolated one."
            )
        if self._method == "dft":
            raise ValueError(
                "finite-size corrections are diagnostics for many-body states and "
                "are not supported for method='dft' (whose supercell error is k-point sampling)."
            )

        solver = self.solver
        if solver is None:
            raise RuntimeError(
                "the calculation needs to be run before computing finite-size corrections; "
                "call get_potential_energy() first"
            )

        integrals = solver._gradient_context.get("integrals") if hasattr(solver, "_gradient_context") else None
        if integrals is None:
            raise RuntimeError("no integrals available in solver context")

        # Get RDMs
        gamma, gamma2 = self._state_rdms(solver, psi=state, two_body=True)
        from .core.mpc import spatial_rdms
        D, Gamma = spatial_rdms(gamma, gamma2, len(integrals.basis))

        n_cells = getattr(self, "n_supercells", 1)
        volume = _native.grid_cell(np.asarray(integrals.cell, dtype=float).tolist(), [1, 1, 1])[2]
        energy_per_cell = _native.energy_to_hartree(float(self.results.get("energy", 0.0)), "eV")

        if scheme == "mpc":
            from .core.mpc import (
                exchange_correlation_hole_energy,
                TruncatedCoulombSolver,
                MINIMUM_IMAGE_SHELLS,
            )
            sh = MINIMUM_IMAGE_SHELLS if shells is None else int(shells)
            eri_periodic = integrals.two_body_with_kernel(None, mo=True)
            truncated_solver = TruncatedCoulombSolver(
                integrals.grid.shape, step=integrals.grid.step, shells=sh
            )
            eri_truncated = integrals.two_body_with_kernel(truncated_solver, mo=True)
            return exchange_correlation_hole_energy(
                eri_periodic,
                eri_truncated,
                D,
                Gamma,
                volume=volume,
                n_cells=n_cells,
                energy_per_cell=energy_per_cell,
            )
        elif scheme == "ccmh":
            from .core.ccmh import structure_factor, ccmh_correction, DEFAULT_SHELLS
            sh = DEFAULT_SHELLS if shells is None else int(shells)
            structure = structure_factor(integrals, D, Gamma, limit=2)
            return ccmh_correction(
                structure,
                volume=volume,
                n_cells=n_cells,
                shells=sh,
                energy_per_cell=energy_per_cell,
            )
        elif scheme == "kzk":
            from .core.kzk import kzk_correction
            psi = np.ascontiguousarray(integrals._engine._psi, dtype=float)
            X = (np.ascontiguousarray(integrals._lowdin_x().real, dtype=float)
                 if integrals.orthogonalize else None)
            V = (None if integrals.mo_coefficients is None
                 else np.ascontiguousarray(integrals.mo_coefficients, dtype=complex))
            density = _native.rotated_density(
                self.context, psi, X, V, np.ascontiguousarray(np.real(D), dtype=float))
            return kzk_correction(
                density,
                integrals.grid.dV,
                integrals.cell,
                n_cells=n_cells,
                energy_per_cell=energy_per_cell,
            )

    @contextlib.contextmanager
    def _stage(self, name):
        """Add the seconds of the wrapped block to the run log's stage ``name``."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self._stages[name] = self._stages.get(name, 0.0) + time.perf_counter() - start

    def _mean_field(self, geometry):
        with self._stage("integration: molecular integrals"):
            self.integrals = self._molecular_integrals(geometry)
        particles = self.integrals.particles(
            charge=self._charge,
            n_electrons=self._n_electrons,
            magnetic_moments=[float(m) for m in geometry.magnetic_moments],
        )
        self.context.release_freed_memory()
        # The native integrals are lazy: the two-body integrals are built by the
        # first call that needs them, which is this one.
        with self._stage("mean-field SCF (builds the two-body integrals)"):
            if geometry.moment_vectors is not None and self.method != "dft":
                raise NotImplementedError(
                    "noncollinear initial moments (one vector per atom) start spinor Kohn-Sham: "
                    f"they need method='dft', not {self.method!r}"
                )
            if self._molecular_smearing and (
                self._has_spin_orbit_dataset(geometry) or geometry.moment_vectors is not None
            ):
                raise NotImplementedError(
                    "smearing sets the occupations of molecular Kohn-Sham on a "
                    "scalar-relativistic dataset and of a crystal; a spinor Kohn-Sham "
                    "molecule (spin-orbit dataset) has integer occupations"
                )
            if self.method == "dft" and geometry.moment_vectors is not None:
                # A noncollinear start: collinear Kohn-Sham with the moments' lengths, turned per
                # atom, then spinor Kohn-Sham (with the spin-orbit term of a spin-orbit dataset).
                self.native_reference = self.integrals.spinor_kohn_sham_noncollinear(
                    self.xc, geometry.moment_vectors.tolist(), charge=self._charge,
                    n_electrons=self._n_electrons, **self._scf_options
                )
            elif self.method == "dft" and self._has_spin_orbit_dataset(geometry):
                # Spinor Kohn-Sham: one set of 2m spinors with the spin-orbit term.
                self.native_reference = self.integrals.spinor_kohn_sham(
                    self.xc, int(sum(particles)), **self._scf_options
                )
            elif self.method == "dft":
                self.native_reference = self.integrals.kohn_sham(
                    self.xc,
                    particles,
                    unrestricted=self._unrestricted or particles[0] != particles[1],
                    **self._scf_options,
                    **self._molecular_smearing,
                )
            else:
                kind = (
                    self.method
                    if self.method in ("rhf", "uhf", "ghf")
                    else ("rhf" if particles[0] == particles[1] else "uhf")
                )
                self.native_reference = self.integrals.mean_field(
                    kind, particles, **self._scf_options
                )
        dispersion = (
            self.context.tkatchenko_scheffler(geometry.native(), self.xc)
            if self._vdw == "ts"
            else None
        )
        return CalculationResult(
            method=self.method,
            context=self.context,
            reference=self.native_reference,
            integrals=self.integrals,
            mapping=self._mapping,
            active_space=self._active_spec,
            two_body=(self._two_body_representation, self._two_body_truncation_ha),
            atomic_units=self._atomic_units,
            dispersion=dispersion,
        )

    def _molecular_integrals(self, geometry):
        if (self._basis_spec is None and self.integrals is not None
                and self._force_grid is not None and self._geometry is not None
                and geometry.numbers == self._geometry.numbers
                and np.array_equal(geometry.cell, self._geometry.cell)):
            old = self.integrals
            self.integrals = None
            rebuilt = old.at_geometry(geometry.native())
            del old
            self.context.release_freed_memory()
            return rebuilt
        if self._basis_spec is None:
            from .integrations.paw_molecular import build

            return build(self, geometry)
        return self.context.molecular_integrals(
            geometry.native(),
            basis_spec=json.dumps(self._basis_spec),
            spacing_bohr=to_bohr(float(self._grid_spacing)),
            grid=self._grid(geometry),
            kinetic=self._kinetic,
            cholesky_tolerance=self._cholesky_tolerance,
            precision=self._precision,
        )

    def _saves_over_its_source(self):
        """True when ``save_hamiltonian`` names the file ``load_hamiltonian`` read.

        The loaded file is the richer record (it can carry the native term
        groups a Trotterized run reads back); rewriting it from the problem
        would clobber that, so the run leaves it as it is.
        """
        import os

        if self._load_hamiltonian is None or not isinstance(
            self._save_hamiltonian, (str, os.PathLike)
        ):
            return False
        try:
            return os.path.samefile(self._load_hamiltonian, self._save_hamiltonian)
        except OSError:
            return False

    def _quantum(self, reference, initial_parameters, callback=None):
        if self._supplied_hamiltonian is not None:
            source = self._supplied_hamiltonian
            if isinstance(source, _native.PauliOperator):
                operator = source
            else:
                from .core.mapping import Fermion, PauliSum

                if isinstance(source, Fermion):
                    operator = self.context.fermion_operator(
                        2 * self._hamiltonian_orbitals,
                        self._hamiltonian_particles,
                        [
                            (list(term), float(value.real), float(value.imag))
                            for term, value in source.terms.items()
                        ],
                        mapping=self._mapping,
                    )
                elif isinstance(source, PauliSum):
                    operator = self.context.pauli_operator(
                        source.num_qubits,
                        [
                            (label, float(value.real), float(value.imag))
                            for label, value in source.terms.items()
                        ],
                    )
                else:
                    raise TypeError(
                        "hamiltonian must be a Fermion, PauliSum, or native PauliOperator"
                    )
            self.problem = self.context.quantum_problem(
                operator,
                self._hamiltonian_orbitals,
                self._hamiltonian_particles,
                mapping=self._mapping,
                sector=self._sector,
                include_singles=self._include_singles,
                sparse=self._sparse,
                pool=self._pool,
                precision=self._precision,
                taper=self._taper,
                register_free=self._shots > 0,
            )
        elif self._load_hamiltonian is not None:
            import os

            if not os.path.exists(self._load_hamiltonian):
                raise FileNotFoundError(
                    f"load_hamiltonian: no such file: {os.fspath(self._load_hamiltonian)!r}"
                )
            snapshot = self.context.read_snapshot(
                str(self._load_hamiltonian), "mandacaru-qubit-hamiltonian"
            )
            # A double-factorized Hamiltonian records its truncation; the run log states it.
            self._loaded_two_body = _native.snapshot_two_body(snapshot)
            self.problem = snapshot.problem(
                sector=self._sector,
                sparse=self._sparse,
                include_singles=self._include_singles,
                pool=self._pool,
                precision=self._precision,
                taper=self._taper,
                register_free=self._shots > 0,
            )
            if (
                self._requested_mapping is not None
                and self.problem.mapping != self._mapping
            ):
                raise ValueError("mapping disagrees with the cached Hamiltonian")
            self._mapping = self.problem.mapping
        else:
            model = (
                self._supplied_model
                if self._supplied_model is not None
                else reference.hamiltonian
            )
            measured = {}
            if self._measurement_scheme != "qwc":
                from .integrations import measurement_scheme as scheme_options

                scheme_options.check_model(model, _native)
                measured["measurement"] = self._measurement_scheme
            self.problem = model.problem(
                mapping=self._mapping,
                sector=self._sector,
                sparse=self._sparse,
                include_singles=self._include_singles,
                pool=self._pool,
                precision=self._precision,
                taper=self._taper,
                register_free=self._shots > 0,
                **measured,
            )
        if self._hva is not None:
            self.problem = self.problem.hva(**self._hva)
        if (
            reference is None
            and self._save_hamiltonian
            and not self._saves_over_its_source()
        ):
            from .core.mapping import PauliSum
            from .core.serialization import save_hamiltonian

            save_hamiltonian(
                self._save_hamiltonian,
                PauliSum(
                    {
                        label: complex(re, im)
                        for label, re, im in self.problem.hamiltonian_terms
                    },
                    num_qubits=self.problem.num_qubits,
                ),
                mapping=self.problem.mapping,
                num_particles=self.problem.num_particles,
                n_spatial_orbitals=self.problem.n_spatial_orbitals,
                format=self._hamiltonian_format,
            )
        opts = dict(
            adapt=self.method.endswith("adapt-vqe"),
            trotter=self._trotter,
            quenching=self._quenching,
            tetris=self._tetris,
            prune=self._prune,
            optimizer=self._optimizer["method"],
            max_iterations=self._optimizer.get("maxiter", 300),
            max_evaluations=self._optimizer.get("maxfev", 5000),
            tolerance=self._optimizer.get("tol", 1e-9),
            track_best=self._optimizer.get("track_best", True),
            max_growth=self._max_growth,
            gradient_tolerance=self._convergence.get("gradient"),
            energy_tolerance=(None if self._convergence.get("energy") is None else
                              _native.energy_to_hartree(float(self._convergence["energy"]), "eV")),
            initial_parameters=initial_parameters,
        )
        from .integrations.transfer import prepare as prepare_transfer

        carried = prepare_transfer(self, reference)
        if carried is not None and self._search is None:
            if initial_parameters is not None:
                raise ValueError("choose transferred or explicit initial parameters")
            opts.update(initial_parameters=carried["parameters"], initial_indices=carried["indices"])
            if carried.get("hessian") is not None and opts["optimizer"] in ("SLSQP", "BFGS"):
                opts["initial_hessian"] = carried["hessian"]
        executor = None
        if self.method.startswith("subspace-"):
            if self._execute_circuits:
                raise NotImplementedError("subspace SDK execution requires a multi-reference executor")
            weights = [2.0, 1.0] if self._weights is None else list(self._weights)
            resumed = (None if self._resume_path is None else self.context.read_snapshot(
                str(self._resume_path), "mandacaru-wavefunction-checkpoint"))
            with self._stage("parameter optimization"):
                # Always quenched, one operator per step: these options apply to vqe/adapt-vqe.
                subspace = {k: v for k, v in opts.items()
                            if k not in ("quenching", "tetris", "prune")}
                raw = self.problem.solve_subspace(weights, resume=resumed, **subspace)
            for note in raw.get("notes", ()):
                warnings.warn(note, RuntimeWarning, stacklevel=2)
            if self._checkpoint_path is not None:
                self.problem.subspace_checkpoint(
                    raw["indices"], raw["parameters"].tolist(), adapt=opts["adapt"],
                    product_form=opts["adapt"] or self._trotter, weights=weights,
                    energies_ha=raw["energies_ha"], hessian=raw.get("hessian"),
                    hessian_optimizer=self._optimizer["method"],
                    status_json=json.dumps({"complete": True, "converged": raw["success"],
                                            "termination": raw["message"]}),
                ).save(str(self._checkpoint_path))
        else:
            resumed = (
                None
                if self._resume_path is None
                else self.context.read_snapshot(
                    str(self._resume_path), "mandacaru-wavefunction-checkpoint"
                )
            )
            observer = None
            if callback is not None:
                from .analysis.states import AnsatzView

                def observe(info):
                    info = dict(info)
                    info["energy"] = _native.to_user_units(
                        float(info["energy_ha"]), "energy", atomic_units=self._atomic_units)
                    info["ansatz"] = AnsatzView(self.problem, info["indices"], True)
                    callback(info)

                observer = observe

            executor = None
            if self._execute_circuits:
                from .integrations.circuits import CircuitExecutor

                executor = CircuitExecutor(self)
                from .integrations.circuits import preflight_run_budget

                preflight_run_budget(self, opts)
            if self._search is not None:
                if callback is not None:
                    raise ValueError("growth callbacks apply to ADAPT, not architecture edits")
                from .integrations.search import run

                with self._stage("parameter optimization"):
                    raw = run(self, initial_parameters)
            else:
                self.run_budget_exhausted = None
                self._resume_lineage = (
                    None if resumed is None else _native.snapshot_lineage(resumed)
                )
                try:
                    with self._stage("parameter optimization"):
                        raw = self.problem.solve(
                            resume=resumed, callback=observer, executor=executor,
                            checkpoint_path=(None if self._checkpoint_every is None
                                             else str(self._checkpoint_path)),
                            checkpoint_every=self._checkpoint_every or 0, **opts)
                except RuntimeError as exc:
                    if executor is not None and executor.stopped is not None:
                        # The budget ended the run before any energy existed.
                        from .backends.measurement import RunBudgetExceeded

                        raise RunBudgetExceeded(executor.stopped) from exc
                    raise
                for note in raw.get("notes", ()):
                    warnings.warn(note, RuntimeWarning, stacklevel=2)
                if raw.get("budget_stopped"):
                    self.run_budget_exhausted = raw["message"]
                    warnings.warn(
                        "the run budget stopped the optimization at its best "
                        f"point so far: {raw['message']}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            if executor is not None:
                raw["execution"] = self.provider.name
                raw["state_kind"] = "none" if self._shots else "statevector"
                raw["execution_usage"] = self.provider.run_usage()
            from .integrations.transfer import record as record_transfer

            record_transfer(self, raw)
            self.checkpoint = self.problem.checkpoint(
                raw["indices"],
                raw["parameters"].tolist(),
                product_form=opts["adapt"] or self._trotter,
                energy_ha=raw["energy_ha"],
                method="adapt-vqe" if self.method.endswith("adapt-vqe") else ("vqe" if self.method.endswith("vqe") else self.method),
                hessian=raw.get("hessian"), hessian_optimizer=self._optimizer["method"],
                status_json=json.dumps(
                    {
                        "complete": True,
                        "converged": raw["success"],
                        "termination": raw["message"],
                    }
                ),
            )
            if self._checkpoint_path is not None:
                self.checkpoint.save(str(self._checkpoint_path))
        from .analysis.states import SolverView

        self.solver = SolverView(
            self.problem, raw, self.checkpoint, opts["adapt"] or self._trotter
        )
        from .optimizers import Optimizer
        self.solver.optimizer = Optimizer(
            method=self._optimizer["method"],
            maxiter=self._optimizer.get("maxiter", 300),
            tol=self._optimizer.get("tol", 1e-9),
        )
        result = CalculationResult(
            method=self.method,
            context=self.context,
            solver=self.solver,
            profile=self._profile,
            problem=self.problem,
            reference=None if reference is None else reference.scf,
            integrals=None if reference is None else self.integrals,
            active_space=self._active_spec,
            two_body=(self._two_body_representation, self._two_body_truncation_ha),
            model=None if reference is None else reference.hamiltonian,
            raw=raw,
            mapping=self._mapping,
            atomic_units=self._atomic_units,
        )
        if executor is not None:
            import weakref

            # The device force measures again on this calculator's provider and budgets.
            result._device = weakref.ref(self)
        if result.optimizer_failures:
            warnings.warn(
                f"the inner optimizer did not converge in "
                f"{len(result.optimizer_failures)} inner minimizations; last message: "
                f"{result.optimizer_failures[-1][1]!r}; "
                f"outer convergence: {result.success} ({raw['message']})",
                RuntimeWarning,
                stacklevel=2,
            )
        return result

    def _periodic(self, geometry):
        if self.method != "dft":
            raise NotImplementedError(
                "periodic geometry calculations currently provide native DFT"
            )
        if self.xc not in ("lda", "pbe", "r2scan", "pbe+rvv10", "r2scan+rvv10", "hse06"):
            raise NotImplementedError(f"periodic calculation does not support functional {self.xc!r}")
        if self._load_crystal is not None:
            self.periodic_problem = self.context.load_crystal(str(self._load_crystal))
            if self.periodic_problem.precision != self._precision:
                raise ValueError(
                    f"crystal file {str(self._load_crystal)!r} stores "
                    f"{self.periodic_problem.precision}; construct "
                    f"Mandacaru(precision={self.periodic_problem.precision!r}) "
                    f"(this calculator has precision={self._precision!r})"
                )
            hubbard.check_loaded(self._hubbard_u, self.periodic_problem)
        else:
            from .integrations.ase import atoms_from_geometry
            from .integrations.paw import configure

            if self.method == "dft" and geometry is not None and any(geometry.magnetic_moments):
                self._unrestricted = True
            configure(self, atoms_from_geometry(geometry))
        if (
            self._requested_xc is not None
            and self.xc != self.periodic_problem.functional
        ):
            raise ValueError("xc disagrees with the crystal functional")
        self._xc = self.periodic_problem.functional
        _refuse_exchange_divergence_without_hybrid(
            self._periodic_options["exxdiv_treatment"], self._xc, periodic=True
        )
        if self._save_crystal:
            self.periodic_problem.save(str(self._save_crystal))
        # The setup's freed arrays go back to the system before the SCF takes its own.
        self.context.release_freed_memory()
        periodic = self.periodic_problem.solve(**self._periodic_options)
        dispersion = (
            self.context.tkatchenko_scheffler(geometry.native(), self.xc)
            if self._vdw == "ts"
            else None
        )
        return CalculationResult(
            method="dft",
            context=self.context,
            raw=periodic.snapshot(include_matrices=False),
            periodic=periodic,
            crystal=self.periodic_problem,
            atomic_units=self._atomic_units,
            dispersion=dispersion,
        )

    def _periodic_quantum(self, geometry, initial_parameters, callback=None):
        """A quantum method on a crystal: the Born-von Karman supercell of the ``kpts`` mesh.

        The supercell Hamiltonian is built in Rust from the primitive cell (lattice-summed
        electrostatics, image-summed basis, commensurate periodic grid); the energy is
        reported per primitive cell by ``run()``.
        """
        from .integrations.bloch import check_geometry

        size = check_geometry(self, geometry)
        from .structure import geometry_value
        from .integrations.ase import atoms_from_geometry

        # The ASE supercell carries the initial moments and the cell to the views below; the
        # Hamiltonian itself is built from the primitive geometry in Rust.
        supercell_geom = geometry_value(atoms_from_geometry(geometry).repeat(size))

        with self._stage("integration: periodic supercell integrals"):
            self.integrals = _native.supercell_integrals(
                self.context,
                geometry.native(),
                list(size),
                basis_spec=json.dumps(self._basis_spec),
                spacing_bohr=_native.length_to_bohr(float(self._grid_spacing), "angstrom"),
                cholesky_tolerance=self._cholesky_tolerance,
                precision=self._precision,
            )
        self.n_supercells = int(_native.bloch_supercell(self.integrals)["n_cells"])

        particles = self.integrals.particles(
            charge=self._charge,
            n_electrons=self._n_electrons,
            magnetic_moments=[float(m) for m in supercell_geom.magnetic_moments],
        )

        with self._stage("mean-field SCF (builds the two-body integrals)"):
            kind = "rhf" if particles[0] == particles[1] else "uhf"
            self.native_reference = self.integrals.mean_field(
                kind, particles, **self._scf_options
            )

        reference = CalculationResult(
            method=self.method,
            context=self.context,
            reference=self.native_reference,
            integrals=self.integrals,
            mapping=self._mapping,
            active_space=self._active_spec,
            two_body=(self._two_body_representation, self._two_body_truncation_ha),
            atomic_units=self._atomic_units,
        )

        result = self._quantum(reference, initial_parameters, callback)
        # The reference whose model the problem was built from: the spectral function rebuilds
        # the N +- 1 sectors of that model.
        self._bloch_reference = reference

        from .core.mpc import PeriodicIntegralsAdapter
        adapter = PeriodicIntegralsAdapter(
            self.integrals,
            self.native_reference,
            supercell_geom,
            self.context,
        )
        self.solver._gradient_context = {"integrals": adapter}
        self.solver.mapping = self._mapping

        result._periodic_geometry = geometry
        result._calculator = self

        return result

    def _has_spin_orbit_dataset(self, value):
        from ase.data import chemical_symbols
        dataset_name = getattr(self, "_paw_dataset", None)
        if dataset_name is None and isinstance(self._basis, dict):
            dataset_name = self._basis.get("paw_dataset")
        if dataset_name is None and isinstance(self._basis_spec, dict):
            dataset_name = self._basis_spec.get("paw_dataset")
        if dataset_name is None:
            return False
        from .pseudopotentials import get_paw
        return any(getattr(get_paw(chemical_symbols[z], dataset_name, xc=self.xc),
                           "has_spin_orbit", False) for z in value.numbers)

    def run(self, geometry=None, *, initial_parameters=None, callback=None,
            log_expressivity=False):
        """Run the calculation and return its result.

        ``log_expressivity=True`` adds the ``expr`` column to the run log's
        iteration table: the Kullback-Leibler distance of the grown ansatz's
        fidelity distribution from the Haar one. It samples state fidelities at
        every row, so it is a diagnostic that is off unless asked for, and it
        applies to ``adapt-vqe`` without shots or tapering (a tapered register has
        no particle-number sector to take the Haar dimension from).
        """
        if type(log_expressivity) is not bool:
            raise ValueError(f"log_expressivity must be True or False, got {log_expressivity!r}")
        if log_expressivity and (self.method != "adapt-vqe" or self._shots):
            raise ValueError(
                "log_expressivity samples the state fidelities of the grown ansatz: it needs "
                "method='adapt-vqe' without shots"
            )
        if log_expressivity and self._taper:
            raise ValueError(
                "log_expressivity compares with the Haar dimension of the particle-number "
                "sector, which a tapered register does not have: use taper=False"
            )
        with self.resources.lock:
            if self.resources.closed:
                raise RuntimeError("calculator is closed")
            source = geometry if geometry is not None else self._attached_atoms
            value = geometry_value(source) if source is not None else self._geometry
            self._search_geometry = value
            from .integrations.transfer import capture as capture_transfer

            capture_transfer(self)
            self.result = None
            self.solver = None
            self.checkpoint = None
            self.native_reference = None
            self.results = {}
            self.problem = None
            self._resume_lineage = None
            self.n_supercells = 1
            if (value is None or not any(value.pbc)) and self._crystal_orbitals != "k-space":
                raise ValueError(
                    "crystal_orbitals applies to a crystal; this structure has no periodic "
                    "axis: remove the option"
                )
            if (
                ((value is not None and any(value.pbc)) or self._load_crystal is not None)
                and self.method == "dft"
            ):
                result = self._periodic(value)
            elif (value is not None and any(value.pbc)) and self.capabilities.quantum:
                result = self._periodic_quantum(value, initial_parameters, callback)
            elif (value is not None and any(value.pbc)) and not self.capabilities.quantum:
                result = self._periodic(value)
            elif value is not None:
                hubbard.require_crystal(self._hubbard_u, value.pbc, method=self._method)
                _refuse_exchange_divergence_without_hybrid(
                    self._periodic_options["exxdiv_treatment"], self._xc, periodic=False
                )
                reference = self._mean_field(value)
                if not reference.success:
                    raise RuntimeError("native SCF did not converge")
                if self._save_hamiltonian:
                    reference.save_hamiltonian(
                        self._save_hamiltonian, format=self._hamiltonian_format
                    )
                result = (
                    self._quantum(reference, initial_parameters, callback)
                    if self.capabilities.quantum
                    else reference
                )
            elif self.capabilities.quantum and (
                self._load_hamiltonian is not None
                or self._supplied_model is not None
                or self._supplied_hamiltonian is not None
            ):
                result = self._quantum(None, initial_parameters, callback)
            else:
                raise ValueError(
                    "provide a geometry, cached Hamiltonian, or cached crystal"
                )
            self._geometry = value
            if isinstance(source, Atoms):
                self.atoms = source.copy()
            self.result = result
            n_cells = getattr(self, "n_supercells", 1)
            is_periodic_quantum = self.capabilities.quantum and (value is not None and any(value.pbc))
            cells = n_cells if is_periodic_quantum else 1
            self.results = {
                "energy": _native.to_user_units(float(result.energy_hartree), "energy", cells=cells),
                "free_energy": _native.to_user_units(
                    float(result.free_energy_hartree), "energy", cells=cells),
            }
            if self.method == "dft":
                if result.periodic is not None:
                    self.results["magmom"] = result._raw["magnetic_moment"]
                else:
                    alpha, beta = self.native_reference.num_particles
                    self.results["magmom"] = float(alpha - beta)
            if self.trace or self.txt is not None:
                from .integrations.reporting import write_calculation

                write_calculation(self, result, value, log_expressivity=log_expressivity)
            return result

    def energy_levels(self, num_states=2, *, restarts=4, beta=None, seed=0):
        """Sequential variational deflation with native overlap penalties."""
        if self.method != "vqe":
            raise ValueError("sequential deflation requires a fixed VQE ansatz")
        if self.result is None:
            self.run()
        from .analysis.excitations import EnergyLevels

        raw = self.problem.energy_levels(
            num_states=num_states,
            restarts=restarts,
            beta=beta,
            seed=seed,
            trotter=self._trotter,
            optimizer=self._optimizer["method"],
            max_iterations=self._optimizer.get("maxiter", 300),
            max_evaluations=self._optimizer.get("maxfev", 5000),
            tolerance=self._optimizer.get("tol", 1e-9),
        )
        return EnergyLevels(
            energies=_native.to_user_units(np.ascontiguousarray(raw["energies_ha"], dtype=float),
                                           "energy", atomic_units=self._atomic_units),
            states=self.problem.expand_state(raw["states"]),
            reference_energy=_native.to_user_units(float(raw["reference_ha"]), "energy",
                                                   atomic_units=self._atomic_units),
            num_evaluations=raw["num_evaluations"],
            success=tuple(raw["success"]),
            messages=tuple(raw["messages"]),
            energy_unit="Ha" if self._atomic_units else "eV",
        )

    def calculate(self, atoms=None, properties=("energy",), system_changes=None):
        with self.resources.lock:
            if self.resources.closed:
                raise RuntimeError("calculator is closed")
            started = time.perf_counter()
            self._stages = {}
            source = (
                atoms
                if atoms is not None
                else (
                    self._attached_atoms
                    if self._attached_atoms is not None
                    else self._geometry
                )
            )
            geometry = geometry_value(source)
            if isinstance(source, Atoms):
                Calculator.calculate(self, source, properties, system_changes)
            if "forces" in properties and (
                not self.capabilities.forces
                or self._taper
                or (not any(geometry.pbc) and self._kinetic != "spectral")
            ):
                raise NotImplementedError(
                    "forces require a supported native calculation with spectral kinetic energy"
                )
            if "stress" in properties and not any(geometry.pbc):
                raise PropertyNotImplementedError(
                    "stress is only available for periodic structures"
                )
            periodic_geometry = geometry is not None and any(geometry.pbc)
            if periodic_geometry and self.capabilities.quantum:
                from .integrations.bloch import check_derivatives

                check_derivatives(self, properties)
            if (
                ("forces" in properties or "stress" in properties)
                and self._precision == "float32"
                and periodic_geometry
            ):
                # Refused before any SCF or cached-result lookup, like the molecular case.
                raise ValueError(
                    "precision='float32' does not support crystal forces or stress: the "
                    "analytic derivative engines read float64 data only (the float32 energy "
                    "is unaffected); use precision='float64'"
                )
            if (
                "forces" in properties
                and self._precision == "float32"
                and not periodic_geometry
                and self.capabilities.quantum
            ):
                # Refused before any SCF or cached-result lookup: the variational
                # forces difference the energy over a 1e-3 Bohr (Richardson) step,
                # which float32 rounding turns into force noise of several meV/A;
                # the analytic gradients of the mean-field methods, all-electron
                # and PAW-LCAO, difference nothing.
                raise ValueError(
                    "precision='float32' does not support quantum-method forces: they "
                    "difference the energy over a 1e-3 Bohr step (Richardson "
                    "extrapolation), which float32 rounding turns into force "
                    "noise of several meV/A; use precision='float64'"
                )
            # Cleared ASE results never serve a property: the result is rerun.
            reusable = (
                self.result is not None
                and "energy" in self.results
                and geometry.same_as(self._geometry)
            )
            if not reusable:
                self._drop_crystal()
                # `result` is kept here: `run()` reads it to carry the previous ansatz into the
                # new geometry (`transfer`), then clears it.
                self.solver = None
                self.problem = None
                self.native_reference = None
                self.results.clear()
                self.context.release_freed_memory()
            if "forces" in properties and self._force_grid is None and not periodic_geometry:
                self._force_grid = (
                    self.integrals.grid
                    if reusable and self.integrals is not None
                    else self._grid(geometry)
                ) or _native.SpatialGrid.from_geometry(
                    geometry.native(), [to_bohr(float(self._grid_spacing))] * 3
                )
            result = self.result if reusable else self.run(geometry)
            if "forces" in properties:
                if result.periodic is not None:
                    with self._stage("nuclear gradient (forces)"):
                        self.results["forces"] = result.forces()
                    from .integrations.reporting import write_forces
                    write_forces(self, result, geometry, internal=False, translated=False)
                elif periodic_geometry and self.capabilities.quantum:
                    from .integrations.bloch import finite_difference_forces

                    with self._stage("nuclear gradient (forces)"):
                        self.results["forces"] = finite_difference_forces(self, geometry, result)
                else:
                    project = self._project_translation is not False and len(geometry.numbers) > 1
                    internal = project and self.capabilities.quantum and self._basis_spec is None
                    with self._stage("nuclear gradient (forces)"):
                        self.results["forces"] = result.forces(
                            internal=internal,
                            project_translation=project)
                    from .integrations.reporting import write_forces

                    write_forces(self, result, geometry, internal=internal, translated=project)
                self._arm_exit_hook()
            if "stress" in properties:
                with self._stage("stress tensor"):
                    self.results["stress"] = result.stress()
            if self.trace or self.txt is not None:
                from .integrations.reporting import write_performance

                write_performance(self, result, time.perf_counter() - started,
                                  solver_ran=not reusable)
            self.context.release_freed_memory()
            return result

    def _arm_exit_hook(self):
        """Close the log of a trajectory at interpreter exit, once.

        ASE never tells a calculator that a relaxation is over: the optimizer
        just stops calling it. Exit is the one signal that always arrives, so
        the hook is armed by the first forces that are logged. It holds the
        calculator weakly.
        """
        if not self._exit_hook_armed and (self.trace or self.txt is not None):
            self._exit_hook_armed = True
            atexit.register(_close_log_at_exit, weakref.ref(self))

    def _write_summary_at_exit(self):
        """The ``atexit`` body: close the log **without** claiming success.

        Exit handlers also run after an unhandled exception, so reaching this
        point says only that the process ended, not that the relaxation
        finished or converged. The footer therefore states that completion was
        never signaled; only ``write_optimization_summary(optimizer=...)`` can
        say ``converged``, because only its caller has the optimizer's verdict.
        It never raises: a log that cannot be closed must not turn a finished
        calculation into a failed one.
        """
        try:
            steps = len(getattr(self, "_optimization_history", ()))
            self.write_optimization_summary(status=(
                f"completion not signaled: the process exited after {steps} geometry "
                "step(s), which does not establish that the optimization finished or "
                "converged (call write_optimization_summary(optimizer=...) to record "
                "the verdict)"))
        except Exception:
            pass

    def write_optimization_summary(self, atoms=None, fmax=None, optimizer=None, force=False, status=None):
        """Record the geometry history and ASE optimizer's completion verdict."""
        from .integrations.reporting import write_optimization_summary

        with self.resources.lock:
            return write_optimization_summary(self, atoms, fmax, optimizer, force, status)

    def set_atoms(self, atoms):
        self._attached_atoms = atoms

    def get_property(self, name, atoms=None, allow_calculation=True):
        with self.resources.lock:
            if self.resources.closed:
                raise RuntimeError("calculator is closed")
            if name not in self.implemented_properties:
                raise PropertyNotImplementedError(
                    f"{name} is not implemented for {self.method}"
                )
            geometry = None if atoms is None else geometry_value(atoms)
            changed = geometry is not None and not geometry.same_as(self._geometry)
            if self.result is None or changed or name not in self.results:
                if not allow_calculation:
                    return None
                self.calculate(
                    atoms if atoms is not None else self._attached_atoms,
                    properties=(name,),
                )
            value = self.results[name]
            return (
                np.array(value, copy=True) if isinstance(value, np.ndarray) else value
            )

    def get_potential_energy(self, atoms=None, force_consistent=False):
        return self.get_property("free_energy" if force_consistent else "energy", atoms)

    def measured_rdms(self):
        """The spin-summed spatial 1- and 2-RDMs of the optimized circuit, measured on the
        circuit device of a ``shots > 0`` or ``execute_circuits=True`` run.

        Returns a dict: ``one`` ``(m, m)`` with ``one[p, q] = sum_s <a+_ps a_qs>`` and ``two``
        ``(m, m, m, m)`` with ``two[p, q, r, s] = sum_st <a+_ps a+_qt a_st a_rs>`` over the
        ``m`` active spatial orbitals (real parts; dimensionless), ``one_error`` and
        ``two_error`` (standard errors from the device's, strings treated as independent, or
        ``None``), and the ``observables`` and ``bases`` measured. One job, inside
        ``measurement_budget`` and the run budget; no state vector is formed. A state-vector
        run raises ``ValueError`` (its RDMs come from the state); a tapered problem and the
        methods other than ``vqe`` and ``adapt-vqe`` raise ``NotImplementedError``.
        """
        from .integrations.circuits import measure_rdms

        return measure_rdms(self)

    def get_forces(self, atoms=None):
        return self.get_property("forces", atoms)

    def get_stress(self, atoms=None, voigt=True):
        val = self.get_property("stress", atoms)
        if not voigt and hasattr(val, "shape") and val.shape == (6,):
            from ase.stress import voigt_6_to_full_3x3_stress
            return voigt_6_to_full_3x3_stress(val)
        return val

    def hubbard_response(self, alphas=(0.05, 0.1)):
        """Calculate linear-response Hubbard U parameters.

        Parameters
        ----------
        alphas : tuple of float, optional
            Potential shift amplitudes in eV (or Hartree if atomic_units=True).
            Default is (0.05, 0.1) eV.

        Returns
        -------
        dict
            Dictionary containing 'u_matrix', 'u_charge', 'chi0', 'chi', 'alphas',
            and 'j_response' if collinear spin.
        """
        with self.resources.lock:
            if self.resources.closed:
                raise RuntimeError("calculator is closed")
            if self.result is None or "energy" not in self.results:
                if self._attached_atoms is None:
                    raise RuntimeError("calculator has no attached atoms and no prior calculation")
                self.calculate(self._attached_atoms, properties=("energy",))
            if self.result.periodic is None:
                raise ValueError("hubbard_response is only supported for periodic crystal calculations")

            atomic = self._atomic_units
            unit = "Ha" if atomic else "eV"
            alphas_ha = [convert_energy(float(a), unit, "Ha") for a in alphas]

            raw = self.result.periodic.hubbard_response(alphas=alphas_ha)

            def user(values, quantity="energy"):
                if isinstance(values, (int, float, np.floating)):
                    return _native.to_user_units(float(values), quantity, atomic)
                return _native.to_user_units(np.ascontiguousarray(values, dtype=float), quantity, atomic)

            res = {
                "u_matrix": user(raw["u_matrix"]),
                "u_charge": user(raw["u_charge"]),
                "chi0": user(raw["chi0"], "inverse_energy"),
                "chi": user(raw["chi"], "inverse_energy"),
                "alphas": [user(a) for a in raw["alphas"]],
            }
            if "j_response" in raw and raw["j_response"] is not None:
                res["j_response"] = [user(j) for j in raw["j_response"]]
            return res

    def reset(self):
        with self.resources.lock:
            self.atoms = None
            self.result = None
            self.results = {}
            self._geometry = None
            self.integrals = None
            self.native_reference = None
            self.problem = None
            self.solver = None
            self._force_grid = None
            self._drop_crystal()

    def _drop_crystal(self):
        """Release the periodic problem of the last geometry."""
        if hasattr(self, "periodic_problem"):
            del self.periodic_problem
        if hasattr(self, "context") and self.context is not None:
            self.context.release_freed_memory()

    def close(self):
        with self.resources.lock:
            self.reset()
            if hasattr(self, "_optimization_history"):
                hist = getattr(self, "_optimization_history", None)
                if hasattr(hist, "close"):
                    hist.close()
                delattr(self, "_optimization_history")
            # A closed calculator is unusable: it lets go of the atoms it was attached to, which
            # hold it in turn (``atoms.calc``), so the pair is freed without a collector pass.
            self._attached_atoms = None
            self.resources.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
