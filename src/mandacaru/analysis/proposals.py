"""Recorded-edit I/O and model archives for Rust proposal learning."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .. import _native

MODEL_FILE = "proposal_model.npz"


def data_directory(directory=None):
    value = directory if directory is not None else os.environ.get("MANDACARU_PROPOSAL_DATA")
    if value is None:
        raise ValueError("supply an edit directory or set MANDACARU_PROPOSAL_DATA")
    return Path(value).expanduser()


@dataclass(frozen=True)
class ProposalModel:
    """An immutable native model payload; all scoring and training run in Rust."""
    native_json: str

    @property
    def ready(self):
        return json.loads(self.native_json)["ready"]

    @property
    def report(self):
        return json.loads(self.native_json)["report"]

    @property
    def kind(self):
        return json.loads(self.native_json)["kind"]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                np.savez_compressed(stream, native_model=np.frombuffer(self.native_json.encode(), dtype=np.uint8))
                stream.close()
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"native_model"}:
                raise ValueError("unsupported proposal archive layout")
            value = archive["native_model"]
            if value.dtype != np.uint8 or value.ndim != 1:
                raise ValueError("proposal archive must contain a UTF-8 native model")
            source = value.tobytes().decode()
        parsed = json.loads(source)
        if parsed.get("format") != "mandacaru-proposal-model" or parsed.get("version") != 1:
            raise ValueError("unsupported proposal archive format")
        return cls(source)


def fit(directory=None, path=None, *, group_by="molecule", fractions=(.25,.5,1.),
        seed=0, folds=None, gnn_epochs=100, mixing=.3, temperature=1.,
        kinds=("graph",), resources=None):
    """Train native models, assess disjoint held-out groups, and save the report.

    Missing or insufficient observations produce an unready model with an
    explicit reason. They are never replaced by generated observations.
    """
    if tuple(kinds) != ("graph",):
        raise ValueError("the native proposal trainer currently accepts kinds=('graph',)")
    store = data_directory(directory)
    problems = {}
    for source in sorted((store / "problems").glob("*.json")):
        problems[source.stem] = json.loads(source.read_text())
    records = []
    for source in sorted((store / "chains").glob("*.jsonl")):
        for line in source.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    settings = dict(resources or {})
    if set(settings) - {"threads", "memory_limit_bytes"}:
        raise ValueError("proposal resources accept threads and memory_limit_bytes")
    context = _native.Context(**settings)
    source = context.fit_proposal(
        json.dumps(dict(problems=problems, records=records), allow_nan=False),
        group_by=group_by, fractions=list(fractions), seed=seed, folds=folds,
        gnn_epochs=gnn_epochs, mixing=mixing, temperature=temperature)
    model = ProposalModel(source)
    model.save(store / MODEL_FILE if path is None else path)
    return model
