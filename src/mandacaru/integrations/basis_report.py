"""What the ``[BASIS]`` block of a run log says.

A basis is built at run time: the size hierarchy, the confinement and the
Fourier filter are options, not files, so what the user typed is not a record of
the basis that ran. The defaults left alone (a PAW-LCAO basis is filtered and
confined), the radii an ``energy_shift`` resolved to and the dataset each
element was read from are decided below the calculator. This module gathers
them into the two things a log block is made of, keyed ``fields`` and named
``tables``; it formats what the builder and the native integrals already hold
and computes nothing.

The PAW-LCAO recipe leaves what it resolved on ``driver._basis_report``
(:func:`mandacaru.integrations.paw_molecular.build`); an all-electron basis is
reported from the specification the native engine received.
"""

import os
from collections import Counter

from ..units import BOHR_TO_ANGSTROM, HARTREE_TO_EV

#: The ``source`` of a dataset generated in the session rather than read.
IN_MEMORY = "generated-in-memory"


def _option_text(value):
    if isinstance(value, dict):
        ordered = sorted(value.items())
        return "{" + ", ".join(f"{k}: {_option_text(v)}" for k, v in ordered) + "}"
    return str(value)


def _per_element(spec, symbol, default):
    """One element's value of an option that may be ``{symbol: value}``."""
    if isinstance(spec, dict):
        table = {(k if k == "*" else str(k).capitalize()): v for k, v in spec.items()}
        spec = table.get(symbol.capitalize(), table.get("*"))
    return default if spec is None else spec


def _functions(driver, symbols):
    """``basis_functions`` and the ``functions`` table, from the native report."""
    from .. import _native

    report = _native.integrals_report(driver.integrals)
    owners = report["atom_of_orbital"]
    per_atom = Counter(owners)
    per_element = {}
    for index, symbol in enumerate(symbols):
        per_element.setdefault(symbol, per_atom[index])
    rows = [("symbol", "atoms", "functions_per_atom")]
    rows += [(symbol, symbols.count(symbol), count) for symbol, count in per_element.items()]
    return len(owners), rows


def _zeta_text(options):
    from ..models.radial import resolve_split_scheme

    split_norm, tail_norms = resolve_split_scheme(options.get("zeta_split"))
    if tail_norms is not None:
        return ("first_zeta: tail norms " + ", ".join(f"{t:g}" for t in tail_norms)
                + " (norm of the tail, every zeta split from the first)")
    return (f"last_zeta: split norm {split_norm:g} (squared norm of the tail, each "
            "zeta split from the previous one, halved per zeta)")


def _confinement_fields(family, options, confined, fields):
    from ..pseudopotentials.confinement import (
        energy_shift_label, resolve_polarization_method, validate_confinement_potential)

    fields["polarization_method"] = (
        "gaussian (quasi-Gaussian: r^l [exp(-r^2/r_char^2) - (a - b r^2)], further "
        "shells split from it)"
        if resolve_polarization_method(options.get("polarization_method"),
                                       options.get("energy_shift")) == "gaussian"
        else "orbital (r^k R_outer(r), one shell per added l)")
    # The label reads "energy_shift 0.1 eV"; the key already says so.
    fields["energy_shift"] = energy_shift_label(
        options.get("energy_shift")).removeprefix("energy_shift ")
    if confined:
        amplitude, inner = validate_confinement_potential(options.get("confinement_potential"))
        fields["confinement_potential"] = (
            f"A exp(-(r_c - r_i)/(r - r_i)) / (r_c - r), A = {amplitude:g} Ha, "
            f"r_i = {inner:g} r_c")


def _dataset_fields(driver, species, fields):
    """The functional and the folder the datasets were read from."""
    functionals = sorted({str(getattr(species[s], "xc", "lda")).upper() for s in species})
    # The one-center terms differ by method (guide/pseudopotentials.md).
    if driver.method == "dft":
        terms = ("the dataset's frozen one-center Hartree and exchange-correlation terms")
    else:
        terms = "the exact one-center Coulomb tensor and the frozen core's exchange"
    fields["dataset_xc"] = f"{', '.join(functionals)} (atomic reference; {terms})"
    sources = [str(getattr(species[s], "source", IN_MEMORY)) for s in species]
    folders = list(dict.fromkeys(os.path.dirname(source) for source in sources
                                 if source != IN_MEMORY))
    fields["paw_dataset"] = ("; ".join(f"{os.path.basename(folder)} ({folder})"
                                       for folder in folders)
                             if folders else f"none ({IN_MEMORY})")


def _dataset_row(symbol, pp):
    r_cut = max(float(c.r_cut) for c in pp.channels.values())
    return (symbol, int(round(float(getattr(pp, "atomic_number", 0) or 0))),
            f"{float(pp.valence_charge):g}", max(pp.channels), f"{r_cut:.4f}",
            os.path.basename(str(getattr(pp, "source", IN_MEMORY))).replace(" ", "_"),
            getattr(pp, "source_sha256", None) or "none")


def _orbital_rows(symbol, pp, options, confined):
    """One row per basis state, and one per channel without one.

    A semicore channel has two states; an empty channel keeps its row,
    unconfined, because the dataset has it.
    """
    from ..models.radial import resolve_zeta
    from ..pseudopotentials.confinement import free_energy
    from ..pseudopotentials.orbitals import basis_states

    n_zeta, n_polarization = resolve_zeta(_per_element(options.get("size"), symbol, "SZ"))
    states = basis_states(pp)
    held = {state.l for state in states}
    channels = sorted(pp.channels.items())
    rows = [(state.n, state.l, state.index, state.semicore) for state in states] + [
        (int(channel.n), int(l), 0, False) for l, channel in channels if l not in held]
    valence = [state for state in states if not state.semicore]
    outer = max(valence, key=lambda state: state.l) if valence else None
    table = []
    for n, l, index, semicore in rows:
        orbital = confined.get(symbol, {}).get((int(l), int(index)))
        polarization = (n_polarization if outer is not None
                        and (l, index) == (outer.l, outer.index) else 0)
        if orbital is None:
            free = free_energy(pp, l, index) * HARTREE_TO_EV
            cells = ("unconfined", "unconfined", f"{free:.6f}", f"{free:.6f}", "0.000000")
        else:
            cells = (f"{orbital.r_c:.4f}", f"{orbital.r_c * BOHR_TO_ANGSTROM:.4f}",
                     f"{orbital.free_energy * HARTREE_TO_EV:.6f}",
                     f"{orbital.energy * HARTREE_TO_EV:.6f}",
                     f"{orbital.achieved_shift:.6f}")
        table.append((symbol, n, l, 1 if semicore else n_zeta, polarization) + cells)
    return table


def _tables(species, options, report):
    """The ``datasets``, ``orbitals`` and ``polarization_shells`` tables."""
    datasets = [("symbol", "Z", "Z_ion", "l_max", "r_cut_Bohr", "source", "sha256")]
    orbitals = [("symbol", "n", "l", "zetas", "polarization", "r_c_Bohr", "r_c_Angstrom",
                 "eps_free_eV", "eps_basis_eV", "shift_eV")]
    for symbol, pp in species.items():
        datasets.append(_dataset_row(symbol, pp))
        orbitals.extend(_orbital_rows(symbol, pp, options, report["confinement"]))
    tables = {"datasets": datasets, "orbitals": orbitals}
    if report["polarization"]:
        tables["polarization_shells"] = [
            ("symbol", "l", "based_on_l", "r_cut_Bohr", "r_char_Bohr", "r_c_at_0.3eV_Bohr")
        ] + [(symbol, shell.l, shell.base_l, f"{shell.r_cut:.4f}", f"{shell.r_char:.4f}",
              f"{shell.reference_radius:.4f}")
             for symbol, shell in report["polarization"].items()]
    return tables


def _paw(driver, symbols, fields):
    from ..basis.filtering import cutoff_energy_ev, filter_label
    from ..pseudopotentials.families import lookup_family

    report = driver._basis_report
    options, species = report["options"], report["species"]
    family = lookup_family(driver.basis if isinstance(driver.basis, str) else driver.basis["name"])
    fields["family"] = f"{family.label} ({family.description})"
    fields["size"] = _option_text(options.get("size", "SZ"))
    fields["zeta_split"] = _zeta_text(options)
    if "projector_basis" in family.options:
        fields["projector_basis"] = options.get("projector_basis", "raw")
    if "energy_shift" in family.options:
        _confinement_fields(family, options, report["confinement"], fields)
    if "diffuse" in family.options:
        from ..pseudopotentials.diffuse import diffuse_recipe_text

        fields["diffuse"] = diffuse_recipe_text(options.get("diffuse"))
    cutoff = report["cutoff"]
    fields["filter_cutoff"] = (
        "unfiltered" if cutoff is None else
        f"{cutoff:.4f} Bohr^-1 ({cutoff_energy_ev(cutoff):.2f} eV), "
        + filter_label(options.get("filter_cutoff")))
    fields["local_potential"] = ("range-separated: long range on the grid, short range "
                                 "on atom-centered quadrature")
    _dataset_fields(driver, species, fields)
    return _tables(species, options, report)


def basis_report(driver, geometry):
    """``(fields, tables)`` of the ``[BASIS]`` block.

    A calculation without a geometry built no basis, so only the storage
    precision of its state is stated. Each table is a list of rows whose first
    row is the column names; cells never contain spaces.
    """
    from ase.data import chemical_symbols

    fields, tables = {}, {}
    if geometry is not None and driver.integrals is not None:
        symbols = [chemical_symbols[z] for z in geometry.numbers]
        spec = driver._basis_spec
        name = driver.basis if isinstance(driver.basis, str) else driver.basis.get("name", "per_element")
        fields["name"] = "per-element" if name == "per_element" else str(name)
        if spec is None:
            tables.update(_paw(driver, symbols, fields))
        else:
            fields["family"] = "all-electron"
            options = {key: value for key, value in spec.items() if key != "name"}
            if options:
                fields["options"] = _option_text(options)
        fields["basis_functions"], rows = _functions(driver, symbols)
        tables["functions"] = rows
    # The one owner of the storage precision: no other block repeats it.
    fields["precision"] = driver._precision
    return fields, tables
