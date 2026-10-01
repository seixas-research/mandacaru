# -*- coding: utf-8 -*-
# file: algorithms/proposal_model.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""The learned operator proposal of VALQA: a Hamiltonian graph network with a
Gaussian-process head (Learned-Proposal Markov Chain Ansatz Search).

VALQA draws the operator an ``insert`` or ``replace`` places from

.. math::

    P(\mu \mid C, H) = (1 - \varepsilon)\, p_{\mathrm{ML}}(\mu \mid C, H)
                     + \varepsilon\, p_{\nabla}(\mu \mid C),

the mixture of a learned distribution and the gradient softmax MCAS-VQE draws
from (:func:`~mandacaru.algorithms.mcas.gradient_softmax`).  Until a model has
been trained on enough recorded edits *and* passed the readiness check below,
:math:`\varepsilon = 1` exactly: the proposal is MCAS-VQE's, draw for draw.  A
ready model keeps :math:`0 < \varepsilon`, so every operator keeps a positive
probability and every move its reverse.

Representation
--------------
The qubit Hamiltonian :math:`H = c_0 + \sum_\alpha c_\alpha P_\alpha` is a
bipartite **factor graph**: one node per qubit, one per non-identity Pauli
term, an edge labeled ``X``, ``Y`` or ``Z`` wherever a term acts on a qubit.
Qubit features are the reference occupation, the normalized index (the
Jordan-Wigner order), the coefficient of the one-qubit ``Z`` term (an
orbital-energy proxy) and the node's degree; term features are the absolute
and logarithmic coefficient over :math:`E_{\mathrm{scale}} =
\max_\alpha |c_\alpha|` and the term's weight and ``X``/``Y``/``Z`` counts.
The sign of a coefficient is left out: a term with ``X`` or ``Y`` letters
changes sign with the arbitrary sign of a molecular orbital, so a signed
feature would make two gauges of one molecule look different (the one-qubit
``Z`` coefficients, which do not change, keep theirs).
Two message-passing layers (terms gather from their qubits, qubits from their
terms, one weight matrix per Pauli letter, residual updates; a qubit's messages
are averaged with weights :math:`|c_\alpha|`) give qubit embeddings
:math:`\mathbf h_i`.  A pool generator :math:`A_\mu = \sum_s a_s P_s` -- a
sum of strings -- is embedded as the :math:`|a_s|`-weighted mean, over its
strings, of the mean of :math:`R_\sigma \mathbf h_i` over each string's
support.

The Hamiltonian's embedding :math:`\mathbf z_H` pools the qubit embeddings by
mean and sum and the term embeddings by their :math:`|c_\alpha|`-weighted
mean and sum.  The spec's plain sum over terms is replaced by the
coefficient-weighted one: a molecule has thousands of terms, and an
unweighted sum would dwarf every other feature in the kernel.

**Two geometries.**  Along a trajectory, the proposal at geometry
:math:`X_{n+1}` is conditioned on the Hamiltonian the transferred ansatz was
built for as well: :math:`q(C' \mid C_n, H_n, H_{n+1})`.  The candidate
vector carries :math:`\mathbf z_{n+1}`, :math:`\mathbf z_n` and
:math:`\Delta\mathbf z = \mathbf z_{n+1} - \mathbf z_n`.  A single
geometry -- and every recorded row without a ``previous_problem`` -- is the
case :math:`H_n = H_{n+1}`, :math:`\Delta\mathbf z = 0`.

A candidate (insert :math:`\mu` into the circuit :math:`C`) is described by
these embeddings, the circuit's mean and
position-weighted mean operator embedding, the candidate's embedding and
descriptors known **before** it is evaluated: its pool gradient at ``C``
(absolute and relative to the largest), the source energy above the
reference, its multiplicity in ``C`` and the circuit length.  The score does
not depend on the insertion slot -- positions stay uniform, as in MCAS-VQE.

Gaussian process
----------------
A linear map projects the candidate vector to 8 dimensions, where a squared
exponential kernel compares candidates.  The prior mean is linear in the
descriptors, so a candidate far from every training point -- a molecule
unlike the recorded ones -- falls back to the learned trend in its gradient
rather than to a constant.  The target is the energy change of
recorded ``insert`` proposals, compressed as
:math:`t(\Delta E) = \mathrm{sign}(\Delta E)\log(1 + |\Delta E|/\delta)` with
:math:`\delta = 10^{-5}` Hartree (changes span five decades) and
standardized; the length penalty is left out because it is the same for
every insertion.  The network and the kernel are trained together by
maximizing the marginal likelihood (a deep kernel) with Adam, on at most
:data:`MAX_TRAINING_ROWS` rows.  The score of a candidate is
:math:`b = -\mu + \kappa\sigma_f` (predicted improvement plus latent
uncertainty) and :math:`p_{\mathrm{ML}} = \mathrm{softmax}(b/\tau)`.

Between geometries
------------------
With ``update_between_geometries=True`` VALQA replaces the offline model
with an online one before each chain (:meth:`ProposalModel.condition`): a
ranker model updates its weights on the previous geometry's within-state
pairs, with the offline fit as the prior (:meth:`PairwiseRanker.update`); a
graph model conditions its Gaussian process, the network and the kernel's
hyperparameters frozen.  The online model takes the offline model's place in
the mixture, and the gradient proposal keeps its share.  The new rows are
scored first, by the model that has not seen them (test-then-train).  Each chain still runs with one frozen model, so its
Metropolis-Hastings ratio stays exact.

Readiness: how much data
------------------------
:func:`assess_data_volume` answers whether the recorded edits are enough.  It
holds out one **group** at a time (a molecule, or a Hamiltonian), trains on
the rest at growing fractions of the data, and ranks the held-out edits with
three predictors: this model, the same Gaussian process on the descriptors
alone (no graph), and the gradient heuristic :math:`-|g_\mu|`.  A model is
**ready** when, over at least :data:`MIN_GROUPS` held-out groups, its mean
**within-state** Spearman correlation (:func:`within_state`) exceeds the
gradient's by :data:`READINESS_MARGIN` and beats it in most groups: the
proposal decides among the insertions of one state, so that is where a model
has to be better.  Two models compete: the :class:`PairwiseRanker` comes
first, and the graph model is chosen only when it also beats the ranker.
The chosen kind is saved with the model.  The across-state correlations are
reported beside the rule.

That test compares insertions **across** states, where a large gradient means
a large gain almost by construction (:math:`\Delta E \approx -g^2/2k`), while
the proposal only ever chooses **within** a state.  Each fold therefore also
reports :func:`within_state`: over the held-out states with several recorded
insertions (``screen_insertions`` records them), each predictor's mean
Spearman correlation and its regret -- the energy its first choice gave up
against the best one recorded.  The readiness rule above is the
within-state correlation (decided 2026-10-01: a model that orders the
candidates of a state better shifts the whole softmax toward the better
ones, even when its first choice is no better).

Training runs in JAX; using a trained model needs only NumPy.
"""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .proposal_data import (MODEL_FILE, ProblemRecord, data_directory,
                            load_edits, save_npz_atomically)

#: Width of the node embeddings.
HIDDEN = 32
#: Message-passing layers (term <- qubit <- term cycles).
LAYERS = 2
#: Dimension the kernel compares candidates in.
PROJECTION = 8
#: :math:`\delta` of the target compression, Hartree.
TARGET_SCALE = 1e-5
#: Largest training set the Gaussian process is fitted on (its cost is cubic).
MAX_TRAINING_ROWS = 400
#: Adam steps of a fit.
TRAINING_STEPS = 300
#: Held-out groups required before a model can be ready.
MIN_GROUPS = 3
#: Mean held-out Spearman gain over the better baseline a ready model needs.
READINESS_MARGIN = 0.05
#: Weight of the gradient proposal once a model is ready (:math:`\varepsilon`).
DEFAULT_MIXING = 0.3
#: Exploration weight :math:`\kappa` of the latent standard deviation.
DEFAULT_KAPPA = 1.0
#: Softmax temperature :math:`\tau` of the scores (standardized units).
DEFAULT_SCORE_TEMPERATURE = 1.0
#: Descriptor columns (the "no graph" model uses only these).
DESCRIPTORS = ("support", "x_fraction", "y_fraction", "z_fraction",
               "relative_gradient", "gradient", "source_energy",
               "multiplicity", "length")
#: How held-out groups are formed: by molecule (formula), by Hamiltonian, or
#: by trajectory -- successive geometries are nearly identical, so holding
#: out single frames would leak.
GROUPINGS = ("molecule", "hamiltonian", "trajectory")
#: Version of the saved model layout (2: two-geometry features, mean and sum
#: pooling, no signed coefficient; 3: the model kind, graph or pairwise
#: ranker, chosen by the within-state readiness check).
MODEL_SCHEMA = 3


def compress(delta):
    r""":math:`t(x) = \mathrm{sign}(x)\log(1 + |x|/\delta)` (``x`` in Hartree)."""
    delta = np.asarray(delta, dtype=float)
    return np.sign(delta) * np.log1p(np.abs(delta) / TARGET_SCALE)


# --------------------------------------------------------------------------- #
# The factor graph.
# --------------------------------------------------------------------------- #

@dataclass
class HamiltonianGraph:
    """The factor graph of a qubit Hamiltonian and its pool, as dense arrays.

    ``incidence[s, a, i]`` is one where term ``a`` acts on qubit ``i`` with
    Pauli letter ``s`` (``X``, ``Y``, ``Z``); ``pool_incidence[s, mu, i]`` is
    the weight qubit ``i`` has in generator ``mu``'s embedding.
    """

    qubit_features: np.ndarray        # (n, 4)
    term_features: np.ndarray         # (M_H, 6)
    incidence: np.ndarray             # (3, M_H, n)
    weighted_incidence: np.ndarray    # incidence * |c~|
    term_weights: np.ndarray          # (M_H,)
    global_features: np.ndarray       # (3,)
    pool_incidence: np.ndarray        # (3, M, n)
    pool_descriptors: np.ndarray      # (M, 4)
    scale: float

    @classmethod
    def from_problem(cls, problem: ProblemRecord) -> "HamiltonianGraph":
        n = problem.n_qubits
        terms = problem.terms
        coefficients = problem.coefficients
        n_terms = len(coefficients)
        scale = float(np.max(np.abs(coefficients))) if n_terms else 1.0
        scale = scale if scale > 0.0 else 1.0
        scaled = coefficients / scale

        incidence = np.stack([(terms == code).astype(float)
                              for code in (1, 2, 3)])
        weight = incidence.sum(axis=(0, 2))                      # k_alpha
        counts = incidence.sum(axis=2).T                          # (M_H, 3)
        term_features = np.column_stack([
            np.abs(scaled), np.log(np.abs(scaled) + 1e-6),
            weight / n, counts / n]) if n_terms else np.zeros((0, 6))

        degree = incidence.sum(axis=(0, 1))
        z_only = np.zeros(n)
        for a in np.flatnonzero(weight == 1):
            qubit = int(np.flatnonzero(terms[a])[0])
            if terms[a, qubit] == 3:
                z_only[qubit] += scaled[a]
        qubit_features = np.column_stack([
            problem.reference.astype(float),
            np.arange(n) / max(1, n - 1),
            z_only,
            np.log1p(degree) / math.log1p(max(n_terms, 1))])

        pool_incidence = np.zeros((3, problem.pool_size, n))
        pool_descriptors = np.zeros((problem.pool_size, 4))
        for mu in range(problem.pool_size):
            lo, hi = problem.pool_offsets[mu], problem.pool_offsets[mu + 1]
            strings = problem.pool_strings[lo:hi]
            weights = problem.pool_weights[lo:hi]
            total = weights.sum()
            if total <= 0.0:
                continue
            for string, w in zip(strings, weights / total):
                support = np.flatnonzero(string)
                size = max(len(support), 1)
                for qubit in support:
                    pool_incidence[string[qubit] - 1, mu, qubit] += w / size
                letters = [np.count_nonzero(string == code) / size
                           for code in (1, 2, 3)]
                pool_descriptors[mu] += w * np.array(
                    [len(support) / n, *letters])
        return cls(
            qubit_features=qubit_features, term_features=term_features,
            incidence=incidence,
            weighted_incidence=incidence * np.abs(scaled)[None, :, None],
            term_weights=np.abs(scaled),
            global_features=np.array([math.log(scale), math.log(n),
                                      math.log1p(n_terms)]),
            pool_incidence=pool_incidence,
            pool_descriptors=pool_descriptors, scale=scale)

    def arrays(self) -> dict:
        """The arrays the network reads."""
        return {name: getattr(self, name) for name in (
            "qubit_features", "term_features", "incidence",
            "weighted_incidence", "term_weights", "global_features",
            "pool_incidence")}


# --------------------------------------------------------------------------- #
# Candidates: what is known before an insertion is evaluated.
# --------------------------------------------------------------------------- #

@dataclass
class Candidates:
    """Insertions to score on one problem, one row each.

    ``occupancy[r]`` is the circuit's operator counts over its length and
    ``positional[r]`` the same weighted by position (later occurrences
    weigh more), so ``occupancy @ zA`` is the circuit's mean operator
    embedding; ``operator[r]`` is the candidate and ``descriptors[r]`` the
    :data:`DESCRIPTORS`.
    """

    occupancy: np.ndarray
    positional: np.ndarray
    operator: np.ndarray
    descriptors: np.ndarray

    def __len__(self) -> int:
        return len(self.operator)


def _circuit_rows(architecture, pool_size: int):
    occupancy = np.zeros(pool_size)
    positional = np.zeros(pool_size)
    length = len(architecture)
    if length:
        ramp = np.arange(1, length + 1, dtype=float)
        ramp /= ramp.sum()
        for j, mu in enumerate(architecture):
            occupancy[mu] += 1.0 / length
            positional[mu] += ramp[j]
    return occupancy, positional


def candidate_rows(graph: HamiltonianGraph, architectures, operators,
                   gradients, source_energies) -> Candidates:
    """The :class:`Candidates` for inserting ``operators[r]`` into
    ``architectures[r]``, whose pool gradients are ``gradients[r]`` and whose
    energy above the reference is ``source_energies[r]`` (Hartree)."""
    pool_size = graph.pool_descriptors.shape[0]
    occupancy, positional, descriptors = [], [], []
    for architecture, mu, g, energy in zip(architectures, operators,
                                           gradients, source_energies):
        occ, pos = _circuit_rows(architecture, pool_size)
        occupancy.append(occ)
        positional.append(pos)
        g = np.abs(np.asarray(g, dtype=float))
        largest = float(g.max()) if g.size else 0.0
        descriptors.append([
            *graph.pool_descriptors[mu],
            g[mu] / largest if largest > 0.0 else 0.0,
            float(compress(g[mu])),
            float(compress(energy)),
            float(sum(1 for index in architecture if index == mu)),
            float(len(architecture))])
    rows = len(operators)
    return Candidates(
        occupancy=np.asarray(occupancy).reshape(rows, pool_size),
        positional=np.asarray(positional).reshape(rows, pool_size),
        operator=np.asarray(operators, dtype=np.int64),
        descriptors=np.asarray(descriptors, dtype=float).reshape(
            rows, len(DESCRIPTORS)))


# --------------------------------------------------------------------------- #
# The network, written once for NumPy (inference) and JAX (training).
# --------------------------------------------------------------------------- #

#: Width of the Hamiltonian embedding :math:`\mathbf z_H`.
EMBEDDING = 4 * HIDDEN + 3


def _feature_width(use_graph: bool) -> int:
    return (3 * EMBEDDING + 3 * HIDDEN + len(DESCRIPTORS)) if use_graph \
        else len(DESCRIPTORS)


def init_parameters(seed: int = 0, use_graph: bool = True) -> dict:
    """Small random weights; the kernel starts at unit signal, 0.1 noise."""
    rng = np.random.default_rng(seed)

    def dense(*shape):
        return rng.normal(0.0, 1.0 / math.sqrt(shape[-2]), size=shape)

    params = {
        "projection": dense(_feature_width(use_graph), PROJECTION),
        "log_signal": np.array(0.0), "log_noise": np.array(math.log(0.1)),
        "mean": np.array(0.0), "mean_weights": np.zeros(len(DESCRIPTORS))}
    if use_graph:
        d = HIDDEN
        params.update({
            "qubit_in": dense(4, d), "qubit_bias": np.zeros(d),
            "term_in": dense(6, d), "term_bias": np.zeros(d),
            "to_term": dense(LAYERS, 3, d, d),
            "to_qubit": dense(LAYERS, 3, d, d),
            "term_update": dense(LAYERS, 2 * d, d),
            "term_update_bias": np.zeros((LAYERS, d)),
            "qubit_update": dense(LAYERS, 2 * d, d),
            "qubit_update_bias": np.zeros((LAYERS, d)),
            "readout": dense(3, d, d)})
    return params


def embed(xp, params: dict, graph: dict):
    """``(z_H, z_A)``: the Hamiltonian's pooled embedding and every pool
    generator's, for array namespace ``xp`` (``numpy`` or ``jax.numpy``)."""
    hq = xp.tanh(graph["qubit_features"] @ params["qubit_in"]
                 + params["qubit_bias"])
    hp = xp.tanh(graph["term_features"] @ params["term_in"]
                 + params["term_bias"])
    incidence = graph["incidence"]
    weighted = graph["weighted_incidence"]
    support = incidence.sum(axis=(0, 2))[:, None] + 1e-12
    reach = weighted.sum(axis=(0, 1))[:, None] + 1e-12
    for layer in range(LAYERS):
        to_term = xp.einsum("id,sde->sie", hq, params["to_term"][layer])
        message = xp.einsum("sai,sie->ae", incidence, to_term) / support
        hp = hp + xp.tanh(xp.concatenate([hp, message], axis=1)
                          @ params["term_update"][layer]
                          + params["term_update_bias"][layer])
        to_qubit = xp.einsum("ad,sde->sae", hp, params["to_qubit"][layer])
        message = xp.einsum("sai,sae->ie", weighted, to_qubit) / reach
        hq = hq + xp.tanh(xp.concatenate([hq, message], axis=1)
                          @ params["qubit_update"][layer]
                          + params["qubit_update_bias"][layer])
    read = xp.einsum("id,sde->sie", hq, params["readout"])
    z_a = xp.einsum("smi,sie->me", graph["pool_incidence"], read)
    weights = graph["term_weights"]
    term_sum = weights @ hp
    z_h = xp.concatenate([hq.mean(axis=0), hq.sum(axis=0),
                          term_sum / (weights.sum() + 1e-12), term_sum,
                          graph["global_features"]])
    return z_h, z_a


def features(xp, params: dict, graph: dict | None, rows: Candidates,
             descriptors, use_graph: bool, previous: dict | None = None):
    """The candidate vectors the kernel compares (``descriptors`` already
    standardized).  ``previous`` is the graph of the previous geometry's
    Hamiltonian, ``None`` for the same Hamiltonian."""
    if not use_graph:
        return descriptors
    z_h, z_a = embed(xp, params, graph)
    z_prev = z_h if previous is None else embed(xp, params, previous)[0]
    z = xp.concatenate([z_h, z_prev, z_h - z_prev])
    n = len(rows)
    return xp.concatenate([
        xp.broadcast_to(z, (n, z.shape[0])),
        xp.asarray(rows.occupancy) @ z_a, xp.asarray(rows.positional) @ z_a,
        z_a[xp.asarray(rows.operator)], descriptors], axis=1)


def _kernel(xp, params, a, b):
    squared = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)
    return xp.exp(2.0 * params["log_signal"]) * xp.exp(-0.5 * squared)


def _noise(xp, params):
    return xp.exp(2.0 * params["log_noise"]) + 1e-6


def _prior_mean(params, descriptors):
    """The linear prior mean over the (standardized) descriptors."""
    return params["mean"] + descriptors @ params["mean_weights"]


# --------------------------------------------------------------------------- #
# The trained model.
# --------------------------------------------------------------------------- #

@dataclass
class ProposalModel:
    """A trained (or untrained) GNN-GP proposal and its readiness report.

    ``ready`` is what VALQA acts on: an unready model leaves the gradient
    proposal in effect (:math:`\\varepsilon = 1`).
    """

    params: dict
    use_graph: bool
    descriptor_mean: np.ndarray
    descriptor_std: np.ndarray
    target_mean: float
    target_std: float
    train_phi: np.ndarray
    alpha: np.ndarray
    cholesky: np.ndarray
    ready: bool = False
    mixing: float = DEFAULT_MIXING
    kappa: float = DEFAULT_KAPPA
    temperature: float = DEFAULT_SCORE_TEMPERATURE
    report: dict = field(default_factory=dict)
    version: str = ""
    #: Which predictor draws the proposals: ``"graph"`` (the Gaussian
    #: process) or ``"ranker"`` (:class:`PairwiseRanker`), as the readiness
    #: check chose.
    kind: str = "graph"
    ranker: "PairwiseRanker | None" = None

    # -- prediction -------------------------------------------------------- #

    def _phi(self, graph, rows, previous=None):
        descriptors = (rows.descriptors - self.descriptor_mean) \
            / self.descriptor_std
        return descriptors, features(
            np, self.params, graph, rows, descriptors, self.use_graph,
            previous) @ self.params["projection"]

    def predict(self, graph: dict | None, rows: Candidates,
                previous: dict | None = None):
        """Posterior mean and latent standard deviation of the standardized
        compressed energy change of each candidate (``previous``: the graph
        arrays of the previous geometry's Hamiltonian)."""
        descriptors, phi = self._phi(graph, rows, previous)
        cross = _kernel(np, self.params, phi, self.train_phi)
        mean = _prior_mean(self.params, descriptors) + cross @ self.alpha
        solved = np.linalg.solve(self.cholesky, cross.T) \
            if self.cholesky.size else np.zeros((0, len(rows)))
        variance = math.exp(2.0 * float(self.params["log_signal"])) \
            - (solved ** 2).sum(axis=0)
        return mean, np.sqrt(np.maximum(variance, 0.0))

    def bind(self, problem: ProblemRecord,
             previous: ProblemRecord | None = None) -> "BoundProposal":
        """The proposal for one run's problem, conditioned on the previous
        geometry's (``None``: the same Hamiltonian); the Hamiltonians and
        every pool generator are embedded once, here."""
        return BoundProposal(
            self, HamiltonianGraph.from_problem(problem),
            None if previous is None
            else HamiltonianGraph.from_problem(previous))

    def condition(self, examples) -> "ProposalModel":
        """This model updated with ``examples`` (:func:`training_examples`),
        the online model that takes this one's place.

        A ranker model updates its ranker (:meth:`PairwiseRanker.update`).
        A graph model conditions its Gaussian process; the network, the
        kernel and the standardization stay as trained.

        The old training residuals are recovered from the posterior,
        :math:`r = K\alpha = L L^\top \alpha`, the new ones are standardized
        with the stored statistics, and the factorization is rebuilt over
        both -- the most recent :data:`MAX_TRAINING_ROWS` rows when there are
        more.  Returns a new model; this one is unchanged.
        """
        if self.kind == "ranker":
            model = replace(self, ranker=self.ranker.update(examples))
            model.version = _version(model)
            return model
        phis, residuals = [self.train_phi], [
            self.cholesky @ (self.cholesky.T @ self.alpha)]
        for example in examples:
            descriptors, phi = self._phi(
                example.graph.arrays() if self.use_graph else None,
                example.rows,
                example.previous.arrays()
                if self.use_graph and example.previous is not None else None)
            target = (example.target - self.target_mean) / self.target_std
            phis.append(phi)
            residuals.append(target - _prior_mean(self.params, descriptors))
        phi = np.concatenate(phis)[-MAX_TRAINING_ROWS:]
        residual = np.concatenate(residuals)[-MAX_TRAINING_ROWS:]
        k = _kernel(np, self.params, phi, phi) \
            + float(_noise(np, self.params)) * np.eye(len(phi))
        chol = np.linalg.cholesky(k) if len(phi) else np.zeros((0, 0))
        alpha = (np.linalg.solve(chol.T, np.linalg.solve(chol, residual))
                 if len(phi) else np.zeros(0))
        model = ProposalModel(
            params=self.params, use_graph=self.use_graph,
            descriptor_mean=self.descriptor_mean,
            descriptor_std=self.descriptor_std,
            target_mean=self.target_mean, target_std=self.target_std,
            train_phi=phi, alpha=alpha, cholesky=chol, ready=self.ready,
            mixing=self.mixing, kappa=self.kappa,
            temperature=self.temperature, report=self.report,
            kind=self.kind, ranker=self.ranker)
        model.version = _version(model)
        return model

    # -- storage ----------------------------------------------------------- #

    def save(self, path) -> Path:
        path = Path(path).expanduser()
        arrays = {f"param_{name}": np.asarray(value)
                  for name, value in self.params.items()}
        if self.ranker is not None:
            if self.ranker.precision is not None:
                arrays["ranker_precision"] = self.ranker.precision
            arrays.update(ranker_weights=self.ranker.weights,
                          ranker_mean=self.ranker.mean,
                          ranker_std=self.ranker.std)
        meta = {"schema": MODEL_SCHEMA, "use_graph": self.use_graph,
                "kind": self.kind,
                "ranker_pairs": (None if self.ranker is None
                                 else self.ranker.pairs),
                "target_mean": self.target_mean,
                "target_std": self.target_std, "ready": self.ready,
                "mixing": self.mixing, "kappa": self.kappa,
                "temperature": self.temperature, "report": self.report,
                "version": self.version}
        return save_npz_atomically(
            path, meta=np.array(json.dumps(meta)),
            descriptor_mean=self.descriptor_mean,
            descriptor_std=self.descriptor_std,
            train_phi=self.train_phi, alpha=self.alpha,
            cholesky=self.cholesky, **arrays)

    @classmethod
    def load(cls, path) -> "ProposalModel":
        path = Path(path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"no proposal model at {str(path)!r}")
        try:
            archive = np.load(path, allow_pickle=False)
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            raise ValueError(f"{path}: not a proposal model file "
                             f"({error})") from error
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ValueError(f"{path}: not a proposal model file")
        if "meta" not in archive.files:
            archive.close()
            raise ValueError(f"{path}: not a proposal model file")
        with archive as data:
            meta = json.loads(str(data["meta"]))
            if meta.get("schema") != MODEL_SCHEMA:
                raise ValueError(
                    f"{path}: model schema {meta.get('schema')!r}, expected "
                    f"{MODEL_SCHEMA}; retrain it with train_proposal_model(), "
                    f"which rewrites the file from the recorded edits")
            params = {name[len("param_"):]: np.array(data[name])
                      for name in data.files if name.startswith("param_")}
            ranker = (PairwiseRanker(
                weights=data["ranker_weights"], mean=data["ranker_mean"],
                std=data["ranker_std"], pairs=int(meta["ranker_pairs"]),
                precision=(data["ranker_precision"]
                           if "ranker_precision" in data.files else None))
                if "ranker_weights" in data.files else None)
            return cls(kind=str(meta["kind"]), ranker=ranker,params=params, use_graph=bool(meta["use_graph"]),
                       descriptor_mean=data["descriptor_mean"],
                       descriptor_std=data["descriptor_std"],
                       target_mean=float(meta["target_mean"]),
                       target_std=float(meta["target_std"]),
                       train_phi=data["train_phi"], alpha=data["alpha"],
                       cholesky=data["cholesky"], ready=bool(meta["ready"]),
                       mixing=float(meta["mixing"]),
                       kappa=float(meta["kappa"]),
                       temperature=float(meta["temperature"]),
                       report=meta["report"], version=str(meta["version"]))


class BoundProposal:
    """A model bound to one problem: operator distributions at chain states."""

    def __init__(self, model: ProposalModel, graph: HamiltonianGraph,
                 previous: HamiltonianGraph | None = None):
        self.model = model
        self.graph = graph
        self.previous = previous
        self._arrays = graph.arrays() if model.use_graph else None
        self._previous = (previous.arrays()
                          if model.use_graph and previous is not None
                          else None)

    def learned(self, architecture, gradients, source_energy) -> np.ndarray:
        r""":math:`p_{\mathrm{ML}}(\mu|C) = \mathrm{softmax}(b/\tau)`,
        :math:`b = -\mu + \kappa\sigma_f`, over the whole pool."""
        pool = self.graph.pool_descriptors.shape[0]
        rows = candidate_rows(self.graph, [architecture] * pool, range(pool),
                              [gradients] * pool, [source_energy] * pool)
        if self.model.kind == "ranker":
            score = self.model.ranker.scores(rows) / self.model.temperature
        else:
            mean, std = self.model.predict(self._arrays, rows,
                                           self._previous)
            score = (-mean + self.model.kappa * std) / self.model.temperature
        score -= score.max()
        weights = np.exp(score)
        return weights / weights.sum()

    def probabilities(self, architecture, gradients, source_energy,
                      baseline: np.ndarray) -> np.ndarray:
        r""":math:`(1-\varepsilon) p_{\mathrm{ML}} + \varepsilon\,p_\nabla`;
        ``baseline`` itself when the model is not ready."""
        if not self.model.ready:
            return baseline
        eps = self.model.mixing
        mixed = (1.0 - eps) * self.learned(architecture, gradients,
                                           source_energy) + eps * baseline
        return mixed / mixed.sum()


# --------------------------------------------------------------------------- #
# Training (JAX).
# --------------------------------------------------------------------------- #

def _jax():
    import jax
    import jax.numpy as jnp
    # Double precision for the Cholesky factor, as the JAX force path does.
    jax.config.update("jax_enable_x64", True)
    return jax, jnp


@dataclass
class _Example:
    """The insert rows of one problem, ready for the network."""

    graph: HamiltonianGraph
    rows: Candidates
    target: np.ndarray            # compressed energy change
    gradient: np.ndarray          # |g_mu| of each candidate (baseline)
    group: list
    #: The previous geometry's Hamiltonian, ``None`` for the same one.
    previous: HamiltonianGraph | None = None
    #: The chain state each row inserted into (rows of one state compete).
    state: list | None = None


def training_examples(rows: list[dict], problems: dict,
                      group_by: str = "molecule") -> list[_Example]:
    """The recorded ``insert`` proposals, one :class:`_Example` per problem.

    Only inserts are learned from: their energy change is a function of the
    source state and the placed operator, which is what the score predicts;
    a ``replace`` also depends on what it removed.
    """
    if group_by not in GROUPINGS:
        raise ValueError(f"group_by must be one of {GROUPINGS}, got "
                         f"{group_by!r}")
    by_problem: dict[tuple, list[dict]] = {}
    for row in rows:
        if (row.get("move") == "insert" and row.get("source_gradients")
                and row.get("delta_energy") is not None
                and math.isfinite(row["delta_energy"])):
            pair = (row["problem"], row.get("previous_problem"))
            by_problem.setdefault(pair, []).append(row)
    graphs: dict[str, HamiltonianGraph] = {}

    def graph_of(key):
        if key not in graphs:
            if key not in problems:
                raise FileNotFoundError(
                    f"rows refer to the previous problem {key}, whose "
                    f"problem file is missing")
            graphs[key] = HamiltonianGraph.from_problem(problems[key])
        return graphs[key]

    def group(row, key):
        if group_by == "molecule":
            return row.get("formula") or key
        if group_by == "trajectory":
            return row.get("trajectory") or row.get("run") or key
        return key

    examples = []
    for (key, previous), chosen in sorted(by_problem.items(),
                                          key=lambda item: (item[0][0],
                                                            item[0][1] or "")):
        graph = graph_of(key)
        candidates = candidate_rows(
            graph, [row["source"] for row in chosen],
            [row["operator"] for row in chosen],
            [row["source_gradients"] for row in chosen],
            [row["source_energy"] - row["reference_energy"]
             for row in chosen])
        examples.append(_Example(
            graph=graph, rows=candidates,
            target=compress([row["delta_energy"] for row in chosen]),
            gradient=np.array([abs(row["source_gradients"][row["operator"]])
                               for row in chosen]),
            group=[group(row, key) for row in chosen],
            state=[(key, previous, tuple(row["source"]),
                    round(float(row["source_energy"]), 12))
                   for row in chosen],
            previous=(None if previous is None or previous == key
                      else graph_of(previous))))
    return examples


def _subset(example: _Example, index) -> _Example:
    index = np.asarray(index, dtype=np.int64)
    rows = example.rows
    return _Example(example.graph, Candidates(
        rows.occupancy[index], rows.positional[index], rows.operator[index],
        rows.descriptors[index]), example.target[index],
        example.gradient[index], [example.group[i] for i in index],
        example.previous,
        None if example.state is None else [example.state[i] for i in index])


def fit(examples: list[_Example], *, use_graph: bool = True, seed: int = 0,
        steps: int = TRAINING_STEPS, max_rows: int = MAX_TRAINING_ROWS,
        learning_rate: float = 0.01, weight_decay: float = 1e-3
        ) -> ProposalModel:
    """Train the network and the kernel on ``examples`` (marginal likelihood).

    The returned model is **not** ready: readiness is decided by
    :func:`assess_data_volume`, which :func:`train_proposal_model` runs.
    """
    jax, jnp = _jax()
    rng = np.random.default_rng(seed)
    total = sum(len(e.target) for e in examples)
    if total < 2:
        raise ValueError(f"{total} recorded insertion(s): nothing to fit")
    if total > max_rows:
        # A uniform subsample across problems, reproducible from the seed.
        keep = np.sort(rng.choice(total, size=max_rows, replace=False))
        offsets = np.cumsum([0] + [len(e.target) for e in examples])
        examples = [_subset(e, keep[(keep >= lo) & (keep < hi)] - lo)
                    for e, lo, hi in zip(examples, offsets[:-1], offsets[1:])]
        examples = [e for e in examples if len(e.target)]
    descriptors = np.concatenate([e.rows.descriptors for e in examples])
    d_mean = descriptors.mean(axis=0)
    d_std = descriptors.std(axis=0)
    d_std = np.where(d_std > 1e-12, d_std, 1.0)
    target = np.concatenate([e.target for e in examples])
    y_mean, y_std = float(target.mean()), float(target.std() or 1.0)
    y = jnp.asarray((target - y_mean) / y_std)
    all_scaled = (descriptors - d_mean) / d_std

    graphs = [{k: jnp.asarray(v) for k, v in e.graph.arrays().items()}
              for e in examples]
    previous = [None if e.previous is None else
                {k: jnp.asarray(v) for k, v in e.previous.arrays().items()}
                for e in examples]
    scaled = [jnp.asarray((e.rows.descriptors - d_mean) / d_std)
              for e in examples]
    decayed = [name for name in init_parameters(seed, use_graph)
               if name not in ("log_signal", "log_noise", "mean")]

    def phi_of(params):
        return jnp.concatenate([
            features(jnp, params, graph, example.rows, desc, use_graph, prev)
            for graph, example, desc, prev
            in zip(graphs, examples, scaled, previous)]) \
            @ params["projection"]

    def loss(params):
        phi = phi_of(params)
        n = phi.shape[0]
        k = _kernel(jnp, params, phi, phi) + _noise(jnp, params) * jnp.eye(n)
        chol = jnp.linalg.cholesky(k)
        residual = y - _prior_mean(params, jnp.asarray(all_scaled))
        solved = jax.scipy.linalg.cho_solve((chol, True), residual)
        nll = 0.5 * residual @ solved + jnp.log(jnp.diag(chol)).sum() \
            + 0.5 * n * math.log(2 * math.pi)
        penalty = sum((params[name] ** 2).sum() for name in decayed)
        return nll / n + weight_decay * penalty

    params = {k: jnp.asarray(v) for k, v in
              init_parameters(seed, use_graph).items()}
    step_fn = jax.jit(jax.value_and_grad(loss))
    first = {k: jnp.zeros_like(v) for k, v in params.items()}
    second = {k: jnp.zeros_like(v) for k, v in params.items()}
    best, best_params = math.inf, params
    for t in range(1, steps + 1):
        value, grads = step_fn(params)
        value = float(value)
        if not math.isfinite(value):
            break
        if value < best:
            best, best_params = value, dict(params)
        for name in params:
            first[name] = 0.9 * first[name] + 0.1 * grads[name]
            second[name] = 0.999 * second[name] + 0.001 * grads[name] ** 2
            update = (first[name] / (1 - 0.9 ** t)) / (
                jnp.sqrt(second[name] / (1 - 0.999 ** t)) + 1e-8)
            params[name] = params[name] - learning_rate * update
    params = {k: np.asarray(v, dtype=float) for k, v in best_params.items()}

    phi = np.asarray(phi_of({k: jnp.asarray(v) for k, v in params.items()}))
    k = _kernel(np, params, phi, phi) + float(_noise(np, params)) \
        * np.eye(len(phi))
    chol = np.linalg.cholesky(k)
    residual = np.asarray(y) - _prior_mean(params, all_scaled)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, residual))
    model = ProposalModel(params=params, use_graph=use_graph,
                          descriptor_mean=d_mean, descriptor_std=d_std,
                          target_mean=y_mean, target_std=y_std,
                          train_phi=phi, alpha=alpha, cholesky=chol)
    model.version = _version(model)
    return model


def _version(model: ProposalModel) -> str:
    digest = hashlib.sha256()
    for name in sorted(model.params):
        digest.update(np.ascontiguousarray(model.params[name]).tobytes())
    digest.update(model.alpha.tobytes())
    digest.update(model.kind.encode())
    if model.ranker is not None:
        digest.update(np.ascontiguousarray(model.ranker.weights).tobytes())
    return digest.hexdigest()[:12]


# --------------------------------------------------------------------------- #
# How much data is enough.
# --------------------------------------------------------------------------- #

def _spearman(prediction, observed) -> float:
    from scipy.stats import spearmanr
    prediction = np.asarray(prediction, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if len(observed) < 3 or np.ptp(observed) == 0 or np.ptp(prediction) == 0:
        return math.nan
    return float(spearmanr(prediction, observed).statistic)


def _split(examples, held_out):
    train, test = [], []
    for example in examples:
        inside = np.array([g == held_out for g in example.group])
        if (~inside).any():
            train.append(_subset(example, np.flatnonzero(~inside)))
        if inside.any():
            test.append(_subset(example, np.flatnonzero(inside)))
    return train, test


def _thin(examples, fraction, rng):
    if fraction >= 1.0:
        return examples
    out = []
    for example in examples:
        n = len(example.target)
        keep = max(1, int(round(fraction * n)))
        out.append(_subset(example, np.sort(rng.choice(n, keep,
                                                       replace=False))))
    return out


@dataclass
class PairwiseRanker:
    r"""The simplest within-state model: a pairwise logistic ranker on the
    descriptors (Bradley-Terry).

    For two insertions :math:`\mu, \nu` measured at the same state,
    :math:`P(\mu \text{ gains more than } \nu) =
    \sigma(\mathbf w \cdot (\mathbf x_\mu - \mathbf x_\nu))`, with
    :math:`\mathbf x` the standardized :data:`DESCRIPTORS`.  A candidate's
    score is :math:`\mathbf w \cdot \mathbf x`; within a state it ranks the
    candidates, which is the proposal's decision, and the scores feed a
    softmax like the Gaussian process's.  Fitted only on states with two or
    more recorded insertions (``screen_insertions`` records them).

    ``precision`` is the curvature of the fitted objective at ``weights``
    (a Laplace approximation of the posterior), which :meth:`update` uses as
    the prior when new pairs arrive.
    """

    weights: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    pairs: int = 0
    precision: np.ndarray | None = None

    def scores(self, rows: Candidates) -> np.ndarray:
        """Higher means a larger predicted energy decrease."""
        return ((rows.descriptors - self.mean) / self.std) @ self.weights

    def update(self, examples) -> "PairwiseRanker":
        r"""This ranker after the pairs in ``examples`` -- a sequential
        Bayesian update in the Laplace approximation.

        The new weights minimize the new pairs' logistic loss plus
        :math:`\tfrac12 (\mathbf w - \mathbf w_0)^\top P
        (\mathbf w - \mathbf w_0)`, with :math:`\mathbf w_0` the current
        weights and :math:`P` their precision, so what the offline data
        taught is kept rather than overwritten; the precision then grows by
        the new pairs' curvature.  The standardization stays as fitted.
        Returns this ranker unchanged when ``examples`` hold no pair.
        """
        diff = _ranking_pairs(examples, self.mean, self.std)
        if not len(diff):
            return self
        w0 = self.weights
        prior = (self.precision if self.precision is not None
                 else np.eye(len(w0)))
        w = _fit_logistic(diff, lambda w: (
            0.5 * (w - w0) @ prior @ (w - w0), prior @ (w - w0)), w0)
        return PairwiseRanker(weights=w, mean=self.mean, std=self.std,
                              pairs=self.pairs + len(diff),
                              precision=prior + _logistic_curvature(diff, w))


def _ranking_pairs(examples, mean, std) -> np.ndarray:
    """``x_better - x_worse`` for every pair of insertions recorded at one
    state, with ``x`` the descriptors standardized by ``mean`` and ``std``."""
    if not examples:
        return np.zeros((0, len(DESCRIPTORS)))
    x = (np.concatenate([e.rows.descriptors for e in examples]) - mean) / std
    target = np.concatenate([e.target for e in examples])
    states = [s for e in examples for s in (e.state or [None] * len(e.target))]
    groups: dict = {}
    for row, state in enumerate(states):
        if state is not None:
            groups.setdefault(state, []).append(row)
    better, worse = [], []
    for rows in groups.values():
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                i, j = rows[a], rows[b]
                if target[i] == target[j]:
                    continue
                # Lower compressed change = larger energy decrease.
                first, second = (i, j) if target[i] < target[j] else (j, i)
                better.append(first)
                worse.append(second)
    return x[better] - x[worse]


def _fit_logistic(diff, penalty, start) -> np.ndarray:
    """Minimize the summed logistic loss of ``diff @ w`` plus ``penalty(w)``
    (which returns its value and gradient)."""
    from scipy.optimize import minimize

    def loss(w):
        margin = diff @ w
        extra, extra_grad = penalty(w)
        value = np.logaddexp(0.0, -margin).sum() + extra
        grad = -(diff * (1.0 / (1.0 + np.exp(margin)))[:, None]).sum(axis=0)
        return value, grad + extra_grad

    return minimize(loss, start, jac=True, method="L-BFGS-B").x


def _logistic_curvature(diff, w) -> np.ndarray:
    """Hessian of the summed logistic loss at ``w``."""
    p = 1.0 / (1.0 + np.exp(-(diff @ w)))
    return (diff * (p * (1.0 - p))[:, None]).T @ diff


def fit_pairwise_ranker(examples, l2: float = 1e-2) -> PairwiseRanker | None:
    """Fit :class:`PairwiseRanker` on every pair of insertions measured at
    the same state, ``None`` when no state has two.

    The objective is the mean logistic loss plus ``l2 |w|^2``, i.e. the
    summed loss with a Gaussian prior of precision ``2 l2`` per pair."""
    if not examples:
        return None
    descriptors = np.concatenate([e.rows.descriptors for e in examples])
    mean = descriptors.mean(axis=0)
    std = np.where(descriptors.std(axis=0) > 1e-12, descriptors.std(axis=0),
                   1.0)
    diff = _ranking_pairs(examples, mean, std)
    if not len(diff):
        return None
    ridge = 2.0 * l2 * len(diff)
    w = _fit_logistic(diff, lambda w: (0.5 * ridge * w @ w, ridge * w),
                      np.zeros(diff.shape[1]))
    return PairwiseRanker(
        weights=w, mean=mean, std=std, pairs=len(diff),
        precision=_logistic_curvature(diff, w) + ridge * np.eye(len(w)))


def _score(model, examples):
    """Held-out predictions of ``model`` on ``examples``, concatenated."""
    predicted = []
    for example in examples:
        use = model.use_graph
        mean, _ = model.predict(
            example.graph.arrays() if use else None, example.rows,
            example.previous.arrays()
            if use and example.previous is not None else None)
        predicted.append(mean)
    return np.concatenate(predicted)


def uncompress(t):
    """The inverse of :func:`compress`: Hartree from the compressed scale."""
    t = np.asarray(t, dtype=float)
    return np.sign(t) * TARGET_SCALE * np.expm1(np.abs(t))


def within_state(predictions: dict, examples) -> dict:
    """How well each predictor chooses among the insertions of one state.

    ``predictions`` maps a predictor's name to its scores over the rows of
    ``examples`` concatenated (lower = predicted to lower the energy more).
    A predictor with a non-finite score is reported as ``nan``.  Over the
    states with at least two recorded insertions, returns the
    number of states, each predictor's mean Spearman correlation with the
    observed change (states with three or more), and its mean **regret**:
    the observed energy change of the insertion it ranks first minus the
    best one recorded there, in Hartree (0 for a perfect choice).
    """
    states = [s for e in examples for s in e.state]
    observed = np.concatenate([e.target for e in examples])
    groups: dict = {}
    for row, state in enumerate(states):
        groups.setdefault(state, []).append(row)
    groups = {k: np.asarray(v) for k, v in groups.items() if len(v) >= 2}
    out = {"states": len(groups)}
    for name, score in predictions.items():
        score = np.asarray(score, dtype=float)
        if not np.all(np.isfinite(score)):
            # A predictor that could not be fitted (no state to learn from).
            out[name] = {"spearman": math.nan, "regret": math.nan}
            continue
        ranks, regrets = [], []
        for rows in groups.values():
            truth = uncompress(observed[rows])
            regrets.append(float(truth[np.argmin(score[rows])] - truth.min()))
            if len(rows) >= 3:
                value = _spearman(score[rows], observed[rows])
                if math.isfinite(value):
                    ranks.append(value)
        out[name] = {
            "spearman": float(np.mean(ranks)) if ranks else math.nan,
            "regret": float(np.mean(regrets)) if regrets else math.nan}
    return out


def _pooled_within_state(folds) -> dict:
    """The folds' within-state results, weighted by their states."""
    total = sum(f["within_state"]["states"] for f in folds)
    out = {"states": total}
    for name in ("graph", "descriptors", "gradient", "ranker"):
        out[name] = {}
        for metric in ("spearman", "regret"):
            pairs = [(f["within_state"][name][metric],
                      f["within_state"]["states"]) for f in folds
                     if math.isfinite(f["within_state"][name][metric])]
            weight = sum(w for _, w in pairs)
            out[name][metric] = (sum(v * w for v, w in pairs) / weight
                                 if weight else math.nan)
    return out


def _graph_won(fold) -> bool:
    """Whether the graph model out-ranked both baselines on one fold."""
    baselines = [v for v in (fold["descriptors"], fold["gradient"])
                 if math.isfinite(v)]
    return math.isfinite(fold["graph"]) and (
        not baselines or fold["graph"] > max(baselines))


def _readiness(point) -> tuple[bool, str | None, str]:
    """``(ready, kind, reason)`` from the full-data point of the curve.

    A learned proposal decides among the insertions of one state, so a model
    is judged there: its mean within-state Spearman correlation must exceed
    the gradient's by :data:`READINESS_MARGIN`, and it must beat the gradient
    in most held-out groups.  The pairwise ranker is the first candidate; the
    graph model is preferred only when it also beats the ranker.
    """
    def within(fold, name):
        return fold["within_state"][name]["spearman"]

    folds = [f for f in point["folds"]
             if math.isfinite(within(f, "gradient"))]
    if len(folds) < MIN_GROUPS:
        return False, None, (
            f"only {len(folds)} held-out group(s) have states with three or "
            f"more recorded insertions; record chains with screen_insertions")
    pooled = {name: point["within_state"][name]["spearman"]
              for name in ("graph", "ranker", "gradient")}
    gradient = pooled["gradient"]
    verdicts = []
    for name, rivals in (("graph", ("gradient", "ranker")),
                         ("ranker", ("gradient",))):
        value = pooled[name]
        bar = max(pooled[r] for r in rivals if math.isfinite(pooled[r]))
        wins = sum(1 for f in folds if math.isfinite(within(f, name)) and all(
            not math.isfinite(within(f, r)) or within(f, name) > within(f, r)
            for r in rivals))
        if math.isfinite(value) and value >= gradient + READINESS_MARGIN \
                and value > bar - 1e-12 and wins * 2 > len(folds):
            return True, name, (
                f"within-state Spearman {value:.3f} against the gradient's "
                f"{gradient:.3f}, better in {wins} of {len(folds)} groups")
        verdicts.append(f"{name} {value:.3f} ({wins} of {len(folds)} groups)")
    return False, None, (
        f"within-state Spearman: gradient {gradient:.3f}, "
        + ", ".join(verdicts) + f"; a model needs the gradient's plus "
        f"{READINESS_MARGIN} and most groups")


def assess_data_volume(directory=None, *, group_by: str = "molecule",
                       fractions=(0.25, 0.5, 1.0), seed: int = 0,
                       steps: int = TRAINING_STEPS,
                       max_rows: int = MAX_TRAINING_ROWS) -> dict:
    """The learning curve of the recorded edits in ``directory`` (default:
    the shared store, ``MANDACARU_PROPOSAL_DATA``).

    Leave-one-group-out: each group (a molecule's formula, or a Hamiltonian
    with ``group_by="hamiltonian"``) is held out in turn, the model and the
    descriptor-only baseline are trained on a ``fraction`` of the other
    groups' insertions, and all three predictors rank the held-out ones.
    Returns ``{"curve": [...], "ready": bool, "reason": str, ...}``; each
    curve point has the training labels used, the mean held-out Spearman
    correlation of ``"graph"``, ``"descriptors"`` and ``"gradient"``, and the
    folds the graph model won.
    """
    rows, problems = load_edits(directory)
    examples = training_examples(rows, problems, group_by)
    groups = sorted({g for e in examples for g in e.group})
    labels = sum(len(e.target) for e in examples)
    report = {"group_by": group_by, "groups": len(groups),
              "insertions": labels, "problems": len(examples),
              "rows": len(rows), "curve": []}
    if len(groups) < MIN_GROUPS:
        report.update(ready=False, reason=(
            f"{len(groups)} {group_by} group(s); at least {MIN_GROUPS} are "
            f"needed to hold one out and still train on two"))
        return report
    rng = np.random.default_rng(seed)
    for fraction in fractions:
        folds = []
        for held_out in groups:
            train, test = _split(examples, held_out)
            train = _thin(train, fraction, rng)
            observed = np.concatenate([e.target for e in test])
            n_train = sum(len(e.target) for e in train)
            if len(observed) < 3 or n_train < 2:
                continue
            fold = {"held_out": held_out, "train": min(n_train, max_rows),
                    "test": len(observed)}
            scores = {}
            for name, use_graph in (("graph", True), ("descriptors", False)):
                model = fit(train, use_graph=use_graph, seed=seed,
                            steps=steps, max_rows=max_rows)
                scores[name] = _score(model, test)
                fold[name] = _spearman(scores[name], observed)
            scores["gradient"] = -np.concatenate([e.gradient for e in test])
            fold["gradient"] = _spearman(scores["gradient"], observed)
            ranker = fit_pairwise_ranker(train)
            scores["ranker"] = (
                np.full(len(observed), math.nan) if ranker is None else
                -np.concatenate([ranker.scores(e.rows) for e in test]))
            fold["ranker"] = (math.nan if ranker is None
                              else _spearman(scores["ranker"], observed))
            fold["within_state"] = within_state(scores, test)
            folds.append(fold)
        point = {"fraction": fraction, "folds": folds}
        for name in ("graph", "descriptors", "gradient", "ranker"):
            values = [f[name] for f in folds if math.isfinite(f[name])]
            point[name] = float(np.mean(values)) if values else math.nan
        point["train"] = (int(np.mean([f["train"] for f in folds]))
                          if folds else 0)
        point["graph_wins"] = sum(1 for f in folds if _graph_won(f))
        point["within_state"] = _pooled_within_state(folds)
        report["curve"].append(point)
    ready, kind, reason = _readiness(report["curve"][-1])
    report.update(ready=ready, kind=kind, reason=reason)
    return report


def train_proposal_model(directory=None, path=None, *,
                         group_by: str = "molecule",
                         fractions=(0.25, 0.5, 1.0), seed: int = 0,
                         steps: int = TRAINING_STEPS,
                         max_rows: int = MAX_TRAINING_ROWS,
                         mixing: float = DEFAULT_MIXING,
                         kappa: float = DEFAULT_KAPPA,
                         temperature: float = DEFAULT_SCORE_TEMPERATURE
                         ) -> ProposalModel:
    """Assess the recorded edits, train on all of them and save the model.

    ``directory`` defaults to the shared store (``MANDACARU_PROPOSAL_DATA``)
    and ``path`` to :data:`~mandacaru.algorithms.proposal_data.MODEL_FILE`
    inside it, where VALQA finds the model without being told.  The model is
    saved whether or not it is ready -- with its readiness report -- so a
    retrained file is picked up by the next run; VALQA uses the learned
    proposal only when ``model.ready`` is true.
    """
    if not 0.0 < mixing <= 1.0:
        raise ValueError(f"mixing (epsilon) must be in (0, 1], got {mixing!r}:"
                         f" at 0 an operator the model rules out loses its "
                         f"reverse move")
    if not (kappa >= 0.0 and temperature > 0.0):
        raise ValueError("need kappa >= 0 and temperature > 0")
    directory = data_directory(directory)
    path = directory / MODEL_FILE if path is None else path
    report = assess_data_volume(directory, group_by=group_by,
                                fractions=fractions, seed=seed, steps=steps,
                                max_rows=max_rows)
    rows, problems = load_edits(directory)
    examples = training_examples(rows, problems, group_by)
    model = fit(examples, seed=seed, steps=steps, max_rows=max_rows)
    model.ranker = fit_pairwise_ranker(examples)
    model.kind = report.get("kind") or "graph"
    model.version = _version(model)
    model.ready = bool(report["ready"])
    model.report = report
    model.mixing, model.kappa, model.temperature = mixing, kappa, temperature
    model.save(path)
    return model
