"""Freeze the pairwise ranker of VALSQA's decision A before it is restored in Rust (MG-Q8).

The Python ranker (``PairwiseRanker`` / ``fit_pairwise_ranker``) was deleted in the port, in
commit 241d688. Its three pure functions are read verbatim from that commit's parent and run on a
seeded synthetic set of states: candidate descriptor rows (the nine descriptors) and their targets
(the compressed energy change, lower is better). Run once from the repository root:
``python test/migration/freeze_ranker.py``. It writes ``test/migration/reference/ranker.json``,
which the Rust ``tests/proposal_ranker.rs`` reads.
"""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np

OUT = Path(__file__).parent / "reference" / "ranker.json"
SOURCE = "241d688^:src/mandacaru/algorithms/proposal_model.py"
NAMES = ("_ranking_pairs", "_fit_logistic", "_logistic_curvature", "fit_pairwise_ranker")


def deleted_functions():
    text = subprocess.run(["git", "show", SOURCE], check=True, capture_output=True,
                          text=True).stdout
    lines = text.split("\n")
    namespace = {"np": np, "DESCRIPTORS": tuple(range(9)),
                 "PairwiseRanker": lambda **kw: SimpleNamespace(**kw)}
    for name in NAMES:
        start = next(i for i, line in enumerate(lines) if line.startswith(f"def {name}("))
        end = next(i for i in range(start + 1, len(lines))
                   if lines[i] and not lines[i].startswith((" ", "\t", ")")))
        exec("\n".join(lines[start:end]), namespace)
    return namespace


def states(seed=7, count=8):
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=9)
    out = []
    for s in range(count):
        rows = rng.normal(loc=rng.normal(size=9), scale=1.0 + rng.random(9), size=(4 + s % 4, 9))
        rows[:, 4] = 3.0  # a constant column: its standard deviation is replaced by one
        target = -(rows @ truth) + 0.5 * rng.normal(size=len(rows))
        target[1] = target[0]  # a tie: the pair is skipped
        out.append({"descriptors": rows.tolist(), "targets": target.tolist()})
    return out


def main():
    f = deleted_functions()
    data = states()
    examples = [SimpleNamespace(rows=SimpleNamespace(descriptors=np.asarray(s["descriptors"])),
                                target=np.asarray(s["targets"]),
                                state=[k] * len(s["targets"]))
                for k, s in enumerate(data)]
    ranker = f["fit_pairwise_ranker"](examples, l2=1e-2)
    OUT.write_text(json.dumps({
        "l2": 1e-2, "states": data,
        "weights": ranker.weights.tolist(), "mean": ranker.mean.tolist(),
        "std": ranker.std.tolist(), "pairs": int(ranker.pairs),
    }))
    print("pairs", ranker.pairs, "weights", np.round(ranker.weights, 6))


if __name__ == "__main__":
    main()
