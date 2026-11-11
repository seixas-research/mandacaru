"""The seeded VASQA chains behind ``reference/search_chain.json``.

Every option is set explicitly, so a later change of a default does not move the pin. The cases
are the ones a same-chain change of ``crates/mandacaru-core/src/quantum/search/driver.rs`` must
keep: the default gradient chain with a warm start, a chain from an accepted transfer, a cold
chain (``warm_start=False``) with and without a transfer, and the uniform proposal with the
recording on and off. ``freeze_search_chain.py`` writes the reference and
``test_native_search_chain.py`` replays it; both go through ``Mandacaru(method="vasqa")`` and the
public result views only.
"""

import json
import tempfile
from pathlib import Path

from ase import Atoms

from mandacaru import Mandacaru

HARTREE_PER_EV = 1 / 27.211386245988
INITIAL_TEMPERATURE = 0.1 * HARTREE_PER_EV
FINAL_TEMPERATURE = 0.001 * HARTREE_PER_EV

#: Options every case sets, whatever its model: nothing here is left to a default.
OPTIONS = dict(
    method="vasqa", max_length=4, seed=7, atomic_units=True, profile=False,
    move_weights={"insert": 1.0, "delete": 1.0, "replace": 1.0},
    temperature={"initial": INITIAL_TEMPERATURE, "final": FINAL_TEMPERATURE},
    length_penalty=0.0, replace_start="zero", proposal_temperature=0.1,
    optimizer={"method": "SLSQP", "maxiter": 2000, "maxfev": 5000, "tol": 1e-12},
)

#: Every case searches H2 (HAO basis, grid spacing 0.5, the pool of three ``qeb`` operators). The
#: ``transfer`` cases relax H2 at two geometries and pin the second search, which starts from the
#: first one's ansatz; the others pin the search of the first geometry.
CASES = {
    "gradient_warm": dict(max_steps=40, proposal="gradient", warm_start=True,
                          screen_insertions=0, record=False, transfer=False),
    "gradient_cold": dict(max_steps=40, proposal="gradient", warm_start=False,
                          screen_insertions=0, record=False, transfer=False),
    "uniform_warm_record_off": dict(max_steps=40, proposal="uniform", warm_start=True,
                                    screen_insertions=0, record=False, transfer=False),
    "uniform_warm_record_on": dict(max_steps=40, proposal="uniform", warm_start=True,
                                   screen_insertions=0, record=True, transfer=False),
    "uniform_cold_screened": dict(max_steps=40, proposal="uniform", warm_start=False,
                                  screen_insertions=1, record=True, transfer=False),
    "transfer_warm": dict(max_steps=20, transfer_steps=10, proposal="gradient",
                          warm_start=True, screen_insertions=0, record=False, transfer=True),
    "transfer_cold": dict(max_steps=20, transfer_steps=10, proposal="gradient",
                          warm_start=False, screen_insertions=0, record=False, transfer=True),
}
#: Fields of a recorded row that the chain does not recompute from the step (the four vectors).
RECORDED_VECTORS = ("source_gradients", "source_probabilities", "source_importances",
                    "source_removal")


def summary(result):
    """What a chain exposes through the public result views."""
    steps = [dict(step=row.step, move=row.move, position=row.position, operator=row.operator,
                  source=list(row.source), proposed=list(row.proposed),
                  accepted=bool(row.accepted), screened=bool(row.screened),
                  optimizer_success=bool(row.optimizer_success),
                  energy=row.energy, source_energy=row.source_energy, cost=row.cost,
                  source_cost=row.source_cost, delta_energy=row.delta_energy,
                  temperature=row.temperature, log_q_forward=row.log_q_forward,
                  log_q_reverse=row.log_q_reverse, log_acceptance=row.log_acceptance)
             for row in result.steps]
    return dict(
        steps=steps,
        final_operators=list(result.final_operators),
        final_parameters=result.final_parameters.tolist(),
        optimal_parameters=result.optimal_parameters.tolist(),
        energy=result.energy_hartree, best_energy=result.best_energy,
        final_energy=result.final_energy, optimal_cost=result.optimal_cost,
        final_max_gradient=result.final_max_gradient,
        acceptance_by_move={k: list(v) for k, v in result.acceptance_by_move.items()},
        num_architectures=result.num_architectures, num_evaluations=result.num_evaluations,
    )


def run_h2(case, threads, store=None, view=None):
    """One search of H2, or the second of a two-geometry relaxation when the case transfers.

    ``view`` replaces ``summary`` as the function of the final result."""
    options = dict(CASES[case])
    options.update(OPTIONS)
    options["record"] = store if options["record"] else False
    atoms = Atoms("H2", positions=[[2.5, 2.5, 2.1], [2.5, 2.5, 2.9]], cell=[5.] * 3)
    with Mandacaru(atoms=atoms, basis="HAO", pool="qeb", grid_spacing=.5,
                   resources={"threads": threads}, **options) as calc:
        atoms.get_potential_energy()
        first = calc.result.energy_hartree
        if not options["transfer"]:
            return (view or summary)(calc.result)
        atoms.positions[1, 2] += .001
        atoms.get_potential_energy()
        result = calc.result
        if view:
            return view(result)
        out = summary(result)
        out["first_energy"] = first
        out["transfer"] = dict(accepted=bool(result.transfer["accepted"]),
                               confidence=result.transfer["confidence"])
        return out


def recorded(store):
    """The recorded rows (the edit store of ``record=``) of the only chain in ``store``."""
    paths = list((Path(store) / "chains").glob("*.jsonl"))
    assert len(paths) == 1
    return [json.loads(row) for row in paths[0].read_text().splitlines()]


def run(case, threads, store=None):
    """The summary of a case; a recording case adds its recorded rows under ``records``."""
    out = run_h2(case, threads, store)
    if CASES[case]["record"]:
        out["records"] = [{key: row[key] for key in
                           ("step", "move", "position", "operator", "source", "proposed",
                            "screened", "accepted", "source_energy", "energy",
                            *RECORDED_VECTORS)}
                          for row in recorded(store)]
    return out


def run_in_temporary_store(case, threads):
    with tempfile.TemporaryDirectory() as store:
        return run(case, threads, store)
