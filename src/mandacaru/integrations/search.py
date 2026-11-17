"""Search controls and recorded-data transport for native ansatz searches."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import os
import secrets
import uuid

from .. import _native


def configure(calculation, *, max_steps, min_length, max_length, move_weights,
              temperature, length_penalty, warm_start, replace_start,
              screen_insertions, proposal_temperature, proposal, record, seed,
              proposal_model=None, replicas=1, replica_spacing=2.0):
    if min_length != 0:
        raise ValueError("architecture search starts at the empty sequence; min_length must be zero")
    if proposal not in ("gradient", "uniform") or replace_start not in ("zero", "inherit"):
        raise ValueError("search proposals are gradient or uniform; replacement starts are zero or inherit")
    if type(warm_start) is not bool:
        raise TypeError("warm_start must be a boolean")
    for name, value in (("max_steps", max_steps), ("max_length", max_length),
                        ("screen_insertions", screen_insertions), ("replicas", replicas)):
        positive = name in ("max_length", "replicas")
        if type(value) is not int or value < (1 if positive else 0):
            raise ValueError(f"{name} must be a {'positive' if positive else 'nonnegative'} integer")
    if (isinstance(replica_spacing, bool) or not isinstance(replica_spacing, (int, float))
            or not replica_spacing >= 1 or replica_spacing == float("inf")):
        raise ValueError("replica_spacing must be a finite number of at least 1")
    names = ("insert", "delete", "replace")
    if move_weights is None:
        weights = [1.] * 3
    else:
        if set(move_weights) - set(names):
            raise ValueError(f"move_weights accepts {names}")
        weights = [move_weights.get(name, 0.) for name in names]
    if temperature is None:
        temperatures = [_native.energy_to_hartree(.1, "eV"), _native.energy_to_hartree(.001, "eV")]
    else:
        unit = "Ha" if calculation._atomic_units else "eV"
        if isinstance(temperature, dict):
            if set(temperature) != {"initial", "final"}:
                raise ValueError("an annealed temperature needs initial and final values")
            temperatures = [_native.convert_energy(float(temperature[name]), unit, "Ha")
                            for name in ("initial", "final")]
        else:
            temperatures = [_native.convert_energy(float(temperature), unit, "Ha")] * 2
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
        raise ValueError("seed must be a nonnegative 64-bit integer or None")
    destination = None if record is False else record or os.environ.get("MANDACARU_PROPOSAL_DATA")
    if destination is True:
        raise ValueError("record requires a directory or MANDACARU_PROPOSAL_DATA")
    if screen_insertions and destination is None:
        raise ValueError("screen_insertions records its measurements; supply a record directory")
    if replicas > 1 and (screen_insertions or 0. in temperatures):
        raise ValueError("replica exchange needs a nonzero temperature and no screen_insertions")
    if replicas == 1 and replica_spacing != 2.0:
        raise ValueError("replica_spacing requires replicas > 1")
    calculation._search_record = None if destination is None else Path(destination)
    if proposal_model is not None and calculation.method not in ("valsqa", "subspace-valsqa"):
        raise ValueError("proposal_model requires valsqa or subspace-valsqa")
    calculation._proposal_model = proposal_model
    calculation._search = dict(
        max_steps=max_steps, max_length=max_length, move_weights=weights, temperatures=temperatures,
        length_penalty=_native.convert_energy(float(length_penalty), "Ha" if calculation._atomic_units else "eV", "Ha"),
        warm_start=warm_start, replace_start=replace_start, screen_insertions=screen_insertions,
        proposal_temperature=proposal_temperature, proposal=proposal, record=destination is not None,
        seed=secrets.randbits(64) if seed is None else seed, replicas=replicas, replica_spacing=float(replica_spacing))


def run(calculation, initial_parameters, *, executor=None, resume=None):
    is_subspace = calculation.method.startswith("subspace-")
    if is_subspace and (calculation._execute_circuits or executor is not None):
        raise NotImplementedError("subspace architecture search requires local state vectors")
    resume_json = None
    if resume is not None:
        chain_state = (json.loads(resume.json).get("status") or {}).get("search_chain")
        if chain_state is None:
            raise ValueError("checkpoint does not contain a saved VASQA chain state")
        resume_json = json.dumps(chain_state)
    model, reason = None, None
    if calculation.method in ("valsqa", "subspace-valsqa"):
        from ..analysis.proposals import MODEL_FILE, ProposalModel
        candidate = calculation._proposal_model
        if candidate is None:
            store = calculation._search_record or os.environ.get("MANDACARU_PROPOSAL_DATA")
            if store is not None and (Path(store) / MODEL_FILE).exists():
                candidate = Path(store) / MODEL_FILE
        if candidate is not None and candidate is not False:
            model = candidate if isinstance(candidate, ProposalModel) else ProposalModel.load(candidate)
            reason = model.report["reason"]
        else:
            reason = "no trained proposal model supplied; using gradient proposals"
    carried = calculation._transfer_info
    carry = carried is not None and carried["accepted"]
    if carry and initial_parameters is not None:
        raise ValueError("choose transferred or explicit initial parameters")
    common = dict(**calculation._search, optimizer=calculation._optimizer["method"],
                  max_iterations=calculation._optimizer.get("maxiter", 300), max_evaluations=calculation._optimizer.get("maxfev", 5000),
                  tolerance=calculation._optimizer.get("tol", 1e-9), initial_indices=carried["indices"] if carry else None,
                  initial_parameters=carried["parameters"] if carry else initial_parameters, transfer_steps=calculation._transfer_steps,
                  proposal_model=None if model is None else model.native_json, resume_json=resume_json)
    if is_subspace:
        weights = [2.0, 1.0] if calculation._weights is None else list(calculation._weights)
        raw = calculation.problem.search_subspace(weights, **common)
    else:
        raw = calculation.problem.search(**common, executor=executor, shots=calculation._shots or 0)
    if calculation.method in ("valsqa", "subspace-valsqa"):
        raw.update(proposal_ready=model is not None and model.ready, proposal_reason=reason)
        if calculation._update_between_geometries:
            raw["model_update"] = ("none (no proposal model)" if model is None else
                "none (model is not ready)" if not model.ready else
                "graph model carried unchanged; graph training is offline")
    details = json.loads(raw.pop("search_json"))
    records = details.pop("records")
    details["chain_steps"] = details.pop("steps")
    details["acceptance_by_move"] = dict(zip(("insert", "delete", "replace"), details["acceptance_by_move"]))
    raw.update(details)
    if calculation._search_record is not None:
        save_records(calculation, records)
    return raw


def save_records(calculation, records):
    problem = calculation.problem
    payload = dict(format="mandacaru-search-problem", version=1, hamiltonian=problem.hamiltonian_terms,
                   pool=problem.pool_terms, labels=problem.pool_labels, mapping=problem.mapping,
                   num_particles=problem.num_particles, spatial_orbitals=problem.n_spatial_orbitals, num_qubits=problem.num_qubits)
    if calculation._search_geometry is not None:
        payload["composition"] = sorted(Counter(calculation._search_geometry.numbers).items())
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
    identity = hashlib.sha256(encoded.encode()).hexdigest()
    store = calculation._search_record
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
