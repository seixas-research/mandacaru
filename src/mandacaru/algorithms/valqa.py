# -*- coding: utf-8 -*-
# file: algorithms/valqa.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""VALQA: the Variational Adaptive Learnable Quantum Algorithm.

VALQA is MCAS-VQE's Markov Chain Ansatz Search with a **learned** proposal:
the operator an ``insert`` or ``replace`` places is
drawn from

.. math::

    P(\mu \mid C, H) = (1 - \varepsilon)\, p_{\mathrm{ML}}(\mu \mid C, H)
                     + \varepsilon\, p_{\nabla}(\mu \mid C),

where :math:`p_{\mathrm{ML}}` comes from a graph neural network over the
qubit Hamiltonian (:mod:`~mandacaru.algorithms.proposal_model`) and :math:`p_\nabla` is MCAS-VQE's
gradient softmax.  Moves, positions, relaxation and the Metropolis-Hastings
test are MCAS-VQE's: the model is frozen for the run, and the distribution at a
state is a fixed function of that state, evaluated at the current state for
the forward probability and at the proposed one for the reverse, so the
ratio stays exact.

The model starts where MCAS-VQE is.  Without ``proposal_model``, or with a model
whose readiness check failed (too few recorded edits, or no measurable gain
over the gradient heuristic on held-out molecules), :math:`\varepsilon = 1`
and VALQA draws exactly what MCAS-VQE's gradient proposal draws -- the same chain
for the same seed.  The data comes from the chains of either method, which
record into the shared store ``MANDACARU_PROPOSAL_DATA`` by default;
:func:`~mandacaru.algorithms.proposal_model.fit` assesses
it, trains the model, records whether it is ready and saves it in the store,
where the next VALQA run picks it up.

Along a trajectory (an ASE relaxation, scan or dynamics) the proposal at a
geometry is conditioned on the previous geometry's Hamiltonian as well,
:math:`q(C' \mid C_n, H_n, H_{n+1})`, and with
``update_between_geometries=True`` the offline model is replaced by an
online one, updated with the insertions each geometry evaluated before the
next chain starts -- scored first, test-then-train.  The online model takes
the offline model's place in the mixture; the gradient proposal keeps its
share.  Every chain runs with one frozen model either way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .proposal_data import MODEL_FILE, shared_store
from .proposal_model import (ProposalModel, _spearman,
                             training_examples, within_state)
from .mcas_vqe import MarkovChainSearch, MCASVQEResult


@dataclass(repr=False)
class VALQAResult(MCASVQEResult):
    """A MCAS-VQE result plus the proposal model that drew the operators.

    ``proposal_model`` is the model's version (``None`` without a model) and
    ``model_ready`` whether it was used: ``False`` means the gradient
    proposal was in effect throughout.
    """

    proposal_model: str | None = None
    model_ready: bool = False
    #: What the update between geometries did before this chain (``None``
    #: without ``update_between_geometries``).
    model_update: str | None = None


class VALQA(MarkovChainSearch):
    """VALQA: the Markov chain with a learned operator proposal.

    Takes every option of
    :class:`~mandacaru.algorithms.mcas_vqe.MarkovChainSearch` and

    Parameters
    ----------
    proposal_model : str, path or False, optional
        A model saved by
        :func:`~mandacaru.algorithms.proposal_model.fit`.
        ``None`` (default) takes the shared store's model,
        ``$MANDACARU_PROPOSAL_DATA/proposal_model.npz``, when there is one;
        ``False`` uses none.  Without a model, or with one that is not ready,
        the gradient softmax is in effect.  Loaded at construction, frozen
        for the run.
    update_between_geometries : bool
        Along an ASE trajectory, replace the offline model with an online
        one, updated with the insertions the previous geometry's chain
        evaluated before this chain starts (default ``False``): a ranker
        model takes a sequential Bayesian step on the new within-state pairs
        (:meth:`~mandacaru.algorithms.proposal_model.PairwiseRanker.update`),
        a ``"gp"`` model conditions its Gaussian process with the feature
        weights and kernel as trained, and a ``"graph"`` model is carried
        over unchanged.  The online model takes the offline model's place
        in the mixture; the gradient proposal keeps its share
        (:math:`\varepsilon`).  The model is still frozen for each chain.  The previous
        geometry's insertions are scored first by the model that has not
        seen them (the ``model_update`` line and result field), so the
        update is test-then-train.  The updated model lives for the
        trajectory; the store's model file is not changed.
    """

    citation_method = "valqa"
    #: The name in the ``[SYSTEM]`` block's title.
    log_title = "VALQA"
    _result_class = VALQAResult
    #: The model is bound to the run's Hamiltonian and pool.
    _needs_problem = True
    _draws_from_gradients = True

    def __init__(self, hamiltonian=None, proposal_model=None,
                 update_between_geometries: bool = False,
                 **chain_options):
        if not isinstance(update_between_geometries, bool):
            raise ValueError(f"update_between_geometries must be True or "
                             f"False, got {update_between_geometries!r}")
        self.update_between_geometries = update_between_geometries
        #: The previous geometry's model and insertions, for the update.
        self._handover = None
        self._update = None
        if proposal_model is None:
            store = shared_store()
            if store is not None and (store / MODEL_FILE).is_file():
                proposal_model = store / MODEL_FILE
        elif proposal_model is False:
            proposal_model = None
        self.proposal_model = proposal_model
        self._model = (None if proposal_model is None
                       else ProposalModel.load(proposal_model))
        self._bound = None
        super().__init__(hamiltonian, **chain_options)

    @property
    def _model_in_use(self) -> bool:
        return self._model is not None and self._model.ready

    @property
    def _collects_insertions(self) -> bool:
        return self.update_between_geometries and self._model_in_use

    def inherit_ansatz(self, previous) -> None:
        super().inherit_ansatz(previous)
        if (self.update_between_geometries and self._inherited is not None
                and getattr(previous, "_model_in_use", False)):
            problems = {p.key: p for p in (previous._problem,
                                           previous._previous_problem())
                        if p is not None}
            self._handover = (previous._model, list(previous._insertions),
                              problems)

    def _update_model(self) -> str:
        """Score, then condition on, the previous geometry's insertions."""
        if self._handover is None or not self._linked:
            return "none (no previous geometry with a ready model)"
        model, rows, problems = self._handover
        self._handover = None
        if model.kind == "graph":
            self._model = model
            return (f"none (a graph neural network is not updated between "
                    f"geometries "
                    f"yet; model {model.version} carried over)")
        examples = training_examples(rows, problems, "hamiltonian")
        n = sum(len(e.target) for e in examples)
        if not n:
            self._model = model
            return (f"none (the previous geometry evaluated no insertion); "
                    f"model {model.version} carried over")
        if model.kind == "ranker":
            # Scored on its decision: the order of the insertions of a state.
            score = -np.concatenate([model.ranker.scores(e.rows)
                                     for e in examples])
            within = within_state({"model": score}, examples)
            spearman = within["model"]["spearman"]
            scored = (f"{within['states']} state(s) with several insertions"
                      + ("" if math.isnan(spearman)
                         else f", within-state Spearman {spearman:.3f}"))
        else:
            predicted, observed = [], []
            for example in examples:
                mean, _ = model.predict(
                    example.graph.arrays() if model.use_graph else None,
                    example.rows,
                    example.previous.arrays()
                    if model.use_graph and example.previous is not None
                    else None)
                predicted.append(mean)
                observed.append((example.target - model.target_mean)
                                / model.target_std)
            predicted = np.concatenate(predicted)
            observed = np.concatenate(observed)
            rmse = float(np.sqrt(np.mean((predicted - observed) ** 2)))
            spearman = _spearman(predicted, observed)
            scored = (f"RMSE {rmse:.3f} standardized"
                      + ("" if math.isnan(spearman)
                         else f", Spearman {spearman:.3f}"))
        self._model = model.condition(examples)
        return (f"{n} insertion(s) of geometry {self._step - 1} scored before "
                f"the update ({scored}); model {model.version} -> "
                f"{self._model.version}")

    def _prepare_proposal(self, problem) -> None:
        if self.update_between_geometries:
            if self._model_in_use:
                self._update = self._update_model()
            elif self._model is None:
                self._update = "none (no proposal model)"
            else:
                self._update = (f"none (model {self._model.version} is not "
                                f"ready)")
        self._bound = (self._model.bind(problem, self._previous_problem())
                       if self._model_in_use and problem is not None
                       else None)

    def _operator_distribution(self, evaluated):
        baseline = self._gradient_distribution(evaluated)
        if self._bound is None:
            return baseline
        return self._bound.probabilities(
            evaluated.architecture, evaluated.gradients,
            evaluated.energy - self._reference_ha, baseline)

    def _model_version(self) -> str | None:
        return None if self._model is None else self._model.version

    def _proposal_setup(self) -> dict:
        gradient = (f"gradient softmax (tau {self.proposal_temperature:g} of "
                    f"max |grad|)")
        if self._model is None:
            text = f"{gradient}; no proposal model yet"
        elif not self._model.ready:
            reason = self._model.report.get("reason", "readiness unknown")
            text = f"{gradient}; model {self._model.version} not ready: {reason}"
        else:
            name = {"graph": "graph neural network (GNN)",
                    "gp": "Gaussian process", "ranker": "pairwise ranker"}.get(
                self._model.kind, self._model.kind)
            text = (f"learned {name} (model {self._model.version}) mixed "
                    f"with {gradient}, epsilon {self._model.mixing:g}")
        if self.update_between_geometries:
            text += f"; updated between geometries: {self._update or 'none'}"
        return {"proposal": f"{text}, Metropolis-Hastings",
                "proposal_model": (str(self.proposal_model)
                                   if self.proposal_model is not None
                                   else "none")}

    def _result_extras(self) -> dict:
        return {"proposal_model": self._model_version(),
                "model_ready": self._model_in_use,
                "model_update": (self._update if self.update_between_geometries
                                 else None)}
