"""``measurement_scheme=``: how a circuit run measures the energy.

``"qwc"`` (the default) measures the Pauli terms of the qubit Hamiltonian in qubit-wise
commuting groups, ``O(M^3)`` bases. ``"double-factorized"`` measures the energy in the
``L + 1`` bases of the double factorization of the molecular integrals, ``O(M)`` of them
(Motta et al., npj Quantum Inf. 7, 83, 2021; Huggins et al., npj Quantum Inf. 7, 23, 2021).
The factorization, the Givens networks that rotate into each basis, the diagonal
observables and the assembly of the energy are Rust
(``quantum::factorized_measurement``, executed by ``quantum::execution``); this module
decides which combinations of options the calculator accepts and prices a submission.
"""

SCHEMES = ("qwc", "double-factorized")


def validate(scheme, *, execute_circuits, hamiltonian, load_hamiltonian, mapping, taper, hva):
    """Check ``measurement_scheme=`` against the options it cannot be combined with.

    An unknown name is a ``ValueError`` naming the choices. ``"qwc"`` needs nothing. The scheme
    applies to ``vqe`` and ``adapt-vqe`` (the constructor's per-method option check refuses any
    other); ``adapt-vqe`` factorizes its energies, and its pool gradients stay on the commuting
    groups, which is where they are measured.
    ``"double-factorized"`` refuses, with a ``ValueError`` that names the cause, whatever it
    cannot measure instead of selecting another calculation:

    * a run with no circuit device (nothing is measured: the state-vector solver reads the
      energy off the state);
    * a Hamiltonian that does not come from molecular integrals (``hamiltonian=`` or
      ``load_hamiltonian=``): there is no two-body tensor to factorize;
    * a mapping other than Jordan-Wigner: the scheme reads occupations off the computational
      basis, and only Jordan-Wigner has that picture;
    * ``taper=True`` and ``ansatz="hva"``: their registers and generators are not the untapered
      Jordan-Wigner ones the Givens networks act on.

    The spinor and complex-integral refusals, and a crystal, are raised where the model is
    known (``problem(measurement=...)``, in Rust, and the calculator's ``run``).
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown measurement_scheme {scheme!r}; choose one of {SCHEMES}")
    if scheme == "qwc":
        return scheme
    if not execute_circuits:
        raise ValueError(
            "measurement_scheme='double-factorized' changes what a circuit device measures; "
            "without shots, a hardware device or execute_circuits=True the state-vector solver "
            "measures nothing"
        )
    if hamiltonian is not None or load_hamiltonian is not None:
        raise ValueError(
            "measurement_scheme='double-factorized' factorizes the molecular integrals; a "
            "supplied or loaded qubit Hamiltonian carries none (use the qwc scheme)"
        )
    if mapping not in (None, "jordan_wigner"):
        raise ValueError(
            f"measurement_scheme='double-factorized' reads occupations off the computational "
            f"basis, the Jordan-Wigner picture; mapping={mapping!r} has none (use the qwc scheme)"
        )
    if taper:
        raise ValueError(
            "measurement_scheme='double-factorized' does not combine with taper=True: the "
            "Givens networks act on the untapered register"
        )
    if hva:
        raise ValueError(
            "measurement_scheme='double-factorized' does not combine with ansatz='hva': its "
            "groups are the qubit Hamiltonian's, not the factorization's"
        )
    return scheme


def check_model(model, native):
    """Refuse a reference that is not a molecular model (a crystal, say)."""
    if not isinstance(model, native.MolecularModel):
        raise ValueError(
            "measurement_scheme='double-factorized' factorizes molecular integrals; this "
            "calculation has none (a crystal's Hamiltonian is not a molecular model)"
        )


def energy_cost(driver):
    """``(jobs, shots)`` of one energy of a shot-based run, from the native cost arithmetic.

    The factorized scheme measures ``L + 1`` bases, each a commuting group of its own, in one
    batch of circuits. A Qiskit device takes the batch as one job (or ``ceil((L + 1) / limit)``
    under the provider's ``max_bases_per_job``); Braket submits one task per basis, so it is
    ``measurement_cost`` with a job per basis there. ``problem.factorized_measurement`` (Rust)
    says how many bases there are. An energy the cache of the previous point can answer costs
    less, never more. ``None`` for the commuting groups, whose cost is the provider's.
    """
    from ..backends.measurement import resilience_multipliers

    plan = driver.problem.factorized_measurement
    if plan is None:
        return None
    provider = driver.provider
    factors, twirls, per_twirl = resilience_multipliers(provider)
    return driver.context.measurement_cost(
        plan["bases"], max_bases_per_job=getattr(provider, "max_bases_per_job", None) or None,
        job_per_basis=provider.name == "braket", shots=int(driver._shots),
        noise_factors=factors, randomizations=twirls, shots_per_randomization=per_twirl)
