"""Z2 tapering as an option: its validation and its line of the run log.

The symmetries, the Clifford, the sector and the dropped pool generators are
all found and applied in Rust (``QuantumProblem`` built with ``taper=True``);
this module only decides which combinations of options the calculator accepts.
"""


def validate(taper, *, sector, hva, transfer, save_hamiltonian):
    """Check ``taper=`` against the options it cannot be combined with; return it.

    ``taper`` must be a ``bool`` (``TypeError``). The other refusals are
    ``ValueError`` and name what the tapered register cannot do:

    * ``sector=True``: the particle-number subspace does not apply to a tapered
      register, which the calculation simulates whole;
    * ``ansatz="hva"``: its groups are the untapered Hamiltonian's;
    * ``transfer=True``: a carried ansatz indexes the untapered pool;
    * ``save_hamiltonian``: the file would hold the tapered terms beside the
      untapered mapping, which no reader can load back.
    """
    if type(taper) is not bool:
        raise TypeError("taper must be True or False")
    if not taper:
        return False
    if sector is True:
        raise ValueError(
            "taper=True simulates the whole tapered register, so it does not combine with "
            "sector=True (the particle-number subspace); leave sector unset"
        )
    if hva:
        raise ValueError("taper=True does not combine with ansatz='hva': its groups are untapered")
    if transfer:
        raise ValueError("taper=True does not combine with transfer=True: the carried ansatz is untapered")
    if save_hamiltonian:
        raise ValueError(
            "taper=True does not combine with save_hamiltonian: the tapered terms would be "
            "written beside the untapered mapping and could not be loaded back"
        )
    return True


def text(driver, problem):
    """The ``Z2 tapering`` line of the ``[ELECTRONS]`` block.

    ``problem.taper`` (Rust) owns what was removed; the register width after it
    is the block's ``qubits`` line and is not restated here.
    """
    if not driver._taper:
        return "none"
    found = problem.taper
    if found is None:
        return "none found (no Z2 symmetry beyond the identity)"
    signs = " ".join("+" if sign > 0 else "-" for sign in found["signs"])
    line = (f"{found['removed']} qubit(s) removed from a {found['qubits_before']}-qubit "
            f"register, sector signs ({signs})")
    if found["dropped"]:
        line += f"; {len(found['dropped'])} pool generator(s) dropped as symmetry-changing"
    return line
