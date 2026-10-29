"""Search controls and recorded-data transport for native ansatz searches."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import os
import secrets
import uuid

from ..units import HARTREE_TO_EV


def configure(driver, *, max_steps, min_length, max_length, move_weights,
              temperature, length_penalty, warm_start, replace_start,
              screen_insertions, proposal_temperature, proposal, record, seed,
              proposal_model=None):
    if min_length != 0:
        raise ValueError("architecture search starts at the empty sequence; min_length must be zero")
    if proposal not in ("gradient", "uniform") or replace_start not in ("zero", "inherit"):
        raise ValueError("search proposals are gradient or uniform; replacement starts are zero or inherit")
    if type(warm_start) is not bool:
        raise TypeError("warm_start must be a boolean")
    for name, value in (("max_steps", max_steps), ("max_length", max_length),
                        ("screen_insertions", screen_insertions)):
        if type(value) is not int or value < (1 if name == "max_length" else 0):
            raise ValueError(f"{name} must be a {'positive' if name == 'max_length' else 'nonnegative'} integer")
    names = ("insert", "delete", "replace", "swap")
    if move_weights is None:
        weights = [1.] * 4
    else:
        if set(move_weights) - set(names):
            raise ValueError(f"move_weights accepts {names}")
        weights = [move_weights.get(name, 0.) for name in names]
    if temperature is None:
        temperatures = [.1 / HARTREE_TO_EV, .001 / HARTREE_TO_EV]
    else:
        factor = 1. if driver._atomic_units else 1. / HARTREE_TO_EV
        if isinstance(temperature, dict):
            if set(temperature) != {"initial", "final"}:
                raise ValueError("an annealed temperature needs initial and final values")
            temperatures = [temperature[name] * factor for name in ("initial", "final")]
        else:
            temperatures = [temperature * factor] * 2
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
        raise ValueError("seed must be a nonnegative 64-bit integer or None")
    destination = None if record is False else record or os.environ.get("MANDACARU_PROPOSAL_DATA")
    if destination is True:
        raise ValueError("record requires a directory or MANDACARU_PROPOSAL_DATA")
    if screen_insertions and destination is None:
        raise ValueError("screen_insertions records its measurements; supply a record directory")
    driver._search_record = None if destination is None else Path(destination)
    if proposal_model is not None and driver.method != "valqa":
        raise ValueError("proposal_model requires valqa")
    driver._proposal_model = proposal_model
    driver._search = dict(max_steps=max_steps, max_length=max_length,
                          move_weights=weights, temperatures=temperatures,
                          length_penalty=length_penalty * (1. if driver._atomic_units else 1. / HARTREE_TO_EV),
                          warm_start=warm_start, replace_start=replace_start,
                          screen_insertions=screen_insertions,
                          proposal_temperature=proposal_temperature,
                          proposal=proposal, record=destination is not None,
                          seed=secrets.randbits(64) if seed is None else seed)


def run(driver, initial_parameters):
    if driver._execute_circuits or driver._resume_path is not None:
        raise NotImplementedError("native architecture search currently uses local state vectors and fresh chains")
    model = None
    reason = None
    if driver.method == "valqa":
        from ..analysis.proposals import MODEL_FILE, ProposalModel
        candidate = driver._proposal_model
        if candidate is None:
            store = driver._search_record or os.environ.get("MANDACARU_PROPOSAL_DATA")
            if store is not None and (Path(store) / MODEL_FILE).exists():
                candidate = Path(store) / MODEL_FILE
        if candidate is not None and candidate is not False:
            model = candidate if isinstance(candidate, ProposalModel) else ProposalModel.load(candidate)
            reason = model.report["reason"]
        else:
            reason = "no trained proposal model supplied; using gradient proposals"
    carried = driver._transfer_info
    carry = carried is not None and carried["accepted"]
    if carry and initial_parameters is not None:
        raise ValueError("choose transferred or explicit initial parameters")
    raw = driver.problem.search(
        **driver._search, optimizer=driver._optimizer["method"],
        max_iterations=driver._optimizer.get("maxiter", 300),
        max_evaluations=driver._optimizer.get("maxfev", 5000),
        tolerance=driver._optimizer.get("tol", 1e-9),
        initial_indices=carried["indices"] if carry else None,
        initial_parameters=carried["parameters"] if carry else initial_parameters,
        transfer_steps=driver._transfer_steps,
        proposal_model=None if model is None else model.native_json)
    if driver.method == "valqa":
        raw.update(proposal_ready=model is not None and model.ready,
                   proposal_reason=reason)
        if driver._update_between_geometries:
            raw["model_update"] = (
                "none (no proposal model)" if model is None else
                "none (model is not ready)" if not model.ready else
                "graph model carried unchanged; graph training is offline")
    details = json.loads(raw.pop("search_json"))
    records = details.pop("records")
    details["chain_steps"] = details.pop("steps")
    details["acceptance_by_move"] = dict(zip(
        ("insert", "delete", "replace", "swap"), details["acceptance_by_move"]))
    raw.update(details)
    if driver._search_record is not None:
        save_records(driver, records)
    return raw


def save_records(driver, records):
    problem = driver.problem
    payload = dict(format="mandacaru-search-problem", version=1,
                   hamiltonian=problem.hamiltonian_terms,
                   pool=problem.pool_terms, labels=problem.pool_labels,
                   mapping=problem.mapping, num_particles=problem.num_particles,
                   spatial_orbitals=problem.n_spatial_orbitals,
                   num_qubits=problem.num_qubits)
    geometry = driver._search_geometry
    if geometry is not None:
        payload["composition"] = sorted(Counter(geometry.numbers).items())
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
    identity = hashlib.sha256(encoded.encode()).hexdigest()
    store = driver._search_record
    problems, chains = store / "problems", store / "chains"
    problems.mkdir(parents=True, exist_ok=True)
    chains.mkdir(parents=True, exist_ok=True)
    target = problems / f"{identity}.json"
    if not target.exists():
        target.write_text(encoded + "\n")
    trajectory = uuid.uuid4().hex
    with (chains / f"{trajectory}.jsonl").open("w") as output:
        for row in records:
            output.write(json.dumps(dict(row, format="mandacaru-search-edit",
                                         version=1, problem_id=identity,
                                         trajectory=trajectory), allow_nan=False) + "\n")
