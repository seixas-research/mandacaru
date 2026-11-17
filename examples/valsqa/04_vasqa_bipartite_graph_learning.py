"""VASQA simulation set generation and bipartite factor-graph learning example.

Workflow:
1. Run recorded VASQA searches (`screen_insertions=4`, `warm_start=False`) over
   multiple molecular species and bond lengths to assemble a training simulation set
   (`problems/<sha256>.json` + `chains/*.jsonl`).
2. Inspect the bipartite Hamiltonian factor-graph structure recorded in the store
   (qubit vertices V_Q, Pauli-term vertices V_P, typed X/Y/Z bipartite incidence, and
   pool operator readout vertices V_A).
3. Fit the native Rust bipartite message-passing graph neural network (`kinds=("graph",)`,
   `group_by="molecule"`) on the VASQA simulation set and inspect cross-validated
   within-state Spearman correlations against the physics gradient baseline.
4. Evaluate the saved bipartite-graph model (`proposal_model.npz`) with
   `Mandacaru(method="valsqa")` vs. baseline `Mandacaru(method="vasqa")` on a held-out
   molecule.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from ase import Atoms
from mandacaru import Mandacaru
from mandacaru.analysis.proposals import MODEL_FILE, fit

DATA = Path(__file__).resolve().parent / "data"
STORE = DATA / "bipartite_store"


def linear_chain(symbols: str, spacings: list[float]) -> Atoms:
    """Build a linear molecule centered inside a periodic/isolated simulation cell."""
    z = np.concatenate([[0.0], np.cumsum(spacings)])
    z -= z.mean()
    return Atoms(
        symbols,
        positions=[[5.0, 5.0, 5.0 + float(zi)] for zi in z],
        cell=[10.0, 10.0, 10.0],
    )


def run_vasqa_simulation_set(store: Path) -> None:
    """Collect recorded VASQA chains across four molecule groups."""
    store.mkdir(parents=True, exist_ok=True)
    # Reuse existing records when re-running; delete `store` to regenerate from scratch.
    if list((store / "chains").glob("*.jsonl")):
        print(f"Using existing VASQA simulation store at {store}")
        return

    training_specs = (
        ("H2", (0.65, 0.74, 0.90), 1),
        ("LiH", (1.45, 1.60, 1.75), 1),
        ("BeH2", (1.25, 1.35, 1.50), 2),
        ("H4", (0.85, 0.95, 1.10), 3),
    )
    print("== 1. Running VASQA simulation set ==")
    for symbols, distances, n_bonds in training_specs:
        for r in distances:
            atoms = linear_chain(symbols, [r] * n_bonds)
            atoms.calc = Mandacaru(
                method="vasqa",
                basis="HAO",
                grid_spacing=0.35,
                pool="qeb",
                max_steps=16,
                max_length=8,
                warm_start=False,
                screen_insertions=4,
                length_penalty=1e-3,
                record=str(store),
                seed=1,
                trace=False,
            )
            energy = atoms.get_potential_energy()
            evals = atoms.calc.result.num_evaluations
            print(f"  {symbols:5s} r={r:.2f} A  E={energy:11.6f} eV  ({evals:4d} evals)")


def summarize_bipartite_graphs(store: Path) -> None:
    """Summarize vertices and X/Y/Z incidence counts of the recorded bipartite graphs."""
    problems = sorted((store / "problems").glob("*.json"))
    chains = sorted((store / "chains").glob("*.jsonl"))
    n_insert = 0
    for path in chains:
        for line in path.read_text().splitlines():
            if line.strip() and json.loads(line).get("move") == "insert":
                n_insert += 1

    print("\n== 2. Bipartite factor-graph dataset summary ==")
    print(f"  Problems: {len(problems)}   Chain files: {len(chains)}   Insert rows: {n_insert}")
    for path in problems[:4]:
        meta = json.loads(path.read_text())
        n_q = meta["num_qubits"]
        terms = meta["hamiltonian"]
        pool_ops = meta["pool"]
        edge_counts = [0, 0, 0]  # X, Y, Z incidence edges between V_P and V_Q
        for label, _, _ in terms:
            for idx, axis in enumerate(("X", "Y", "Z")):
                edge_counts[idx] += label.count(axis)
        print(
            f"  {path.stem[:12]}: |V_Q|={n_q:2d} qubits, "
            f"|V_P|={len(terms):4d} Pauli terms, |V_A|={len(pool_ops):3d} pool ops, "
            f"|E_(X,Y,Z)|={tuple(edge_counts)}"
        )


def train_bipartite_graph_model(store: Path):
    """Fit the native Rust bipartite message-passing neural network on the VASQA store."""
    print("\n== 3. Training bipartite message-passing proposal network ==")
    model = fit(
        store,
        kinds=("graph",),
        group_by="molecule",
        fractions=(0.5, 1.0),
        seed=0,
        gnn_epochs=40,
        mixing=0.3,
        temperature=1.0,
    )
    rep = model.report
    print(
        f"  kind={model.kind}  ready={model.ready}  "
        f"groups={rep['groups']}  states={rep['states']}  insertions={rep['insertions']}"
    )
    print(f"  verdict: {rep['reason']}")
    for entry in rep.get("curve", []):
        within = entry["within_state"]
        g_sp = within.get("graph", {}).get("spearman")
        grad_sp = within.get("gradient", {}).get("spearman")
        print(
            f"  fraction={entry['fraction']:.2f}  within-state Spearman: "
            f"graph={g_sp}  gradient={grad_sp}"
        )
    payload = json.loads(model.native_json)
    weight_keys = sorted(payload["weights"].keys())
    print(f"  bipartite MPNN weight tensors ({len(weight_keys)}): {weight_keys[:6]} ...")
    return model


def compare_held_out_search(store: Path) -> None:
    """Compare VALSQA (loaded bipartite graph model) against baseline VASQA."""
    from ase.build import molecule

    print("\n== 4. Held-out comparison: VALSQA (bipartite graph) vs VASQA ==")
    atoms = molecule("H2O")
    atoms.center(vacuum=3.0)
    model_path = str(store / MODEL_FILE)
    for method, extra in (
        ("vasqa", {}),
        ("valsqa", {"proposal_model": model_path}),
    ):
        atoms.calc = Mandacaru(
            method=method,
            basis="HAO",
            grid_spacing=0.35,
            pool="qeb",
            active_space={"orbitals": {"occupied": 2, "virtual": 2}},
            max_steps=15,
            max_length=8,
            length_penalty=1e-3,
            record=False,
            seed=2,
            trace=False,
            **extra,
        )
        energy = atoms.get_potential_energy()
        evals = atoms.calc.result.num_evaluations
        print(f"  {method:7s}  E={energy:11.6f} eV  ({evals:4d} evaluations)")


def main() -> None:
    store = Path(os.environ.get("MANDACARU_EXAMPLE_STORE", str(STORE)))
    run_vasqa_simulation_set(store)
    summarize_bipartite_graphs(store)
    train_bipartite_graph_model(store)
    compare_held_out_search(store)


if __name__ == "__main__":
    main()
