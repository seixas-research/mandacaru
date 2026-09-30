# -*- coding: utf-8 -*-
# file: algorithms/valqa.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""VALQA: the Variational Adaptive Learnable Quantum Algorithm.

VALQA is VASQA's Markov Chain Ansatz Search with a **learned** proposal
(Learned-Proposal MCAS): the operator an ``insert`` or ``replace`` places is
drawn from

.. math::

    P(\mu \mid C, H) = (1 - \varepsilon)\, p_{\mathrm{ML}}(\mu \mid C, H)
                     + \varepsilon\, p_{\nabla}(\mu \mid C),

where :math:`p_{\mathrm{ML}}` comes from a graph network over the qubit
Hamiltonian with a Gaussian-process head
(:mod:`~mandacaru.algorithms.proposal_model`) and :math:`p_\nabla` is VASQA's
gradient softmax.  Moves, positions, relaxation and the Metropolis-Hastings
test are VASQA's: the model is frozen for the run, and the distribution at a
state is a fixed function of that state, evaluated at the current state for
the forward probability and at the proposed one for the reverse, so the
ratio stays exact.

The model starts where VASQA is.  Without ``proposal_model``, or with a model
whose readiness check failed (too few recorded edits, or no measurable gain
over the gradient heuristic on held-out molecules), :math:`\varepsilon = 1`
and VALQA draws exactly what VASQA's gradient proposal draws -- the same chain
for the same seed.  The data comes from the chains of either method, which
record into the shared store ``MANDACARU_PROPOSAL_DATA`` by default;
:func:`~mandacaru.algorithms.proposal_model.train_proposal_model` assesses
it, trains the model, records whether it is ready and saves it in the store,
where the next VALQA run picks it up.
"""

from __future__ import annotations

from dataclasses import dataclass

from .proposal_data import MODEL_FILE, shared_store
from .proposal_model import ProposalModel
from .vasqa import MarkovChainSearch, VASQAResult


@dataclass(repr=False)
class VALQAResult(VASQAResult):
    """A VASQA result plus the proposal model that drew the operators.

    ``proposal_model`` is the model's version (``None`` without a model) and
    ``model_ready`` whether it was used: ``False`` means the gradient
    proposal was in effect throughout.
    """

    proposal_model: str | None = None
    model_ready: bool = False


class VALQA(MarkovChainSearch):
    """VALQA: the Markov chain with a learned operator proposal.

    Takes every option of
    :class:`~mandacaru.algorithms.vasqa.MarkovChainSearch` and

    Parameters
    ----------
    proposal_model : str, path or False, optional
        A model saved by
        :func:`~mandacaru.algorithms.proposal_model.train_proposal_model`.
        ``None`` (default) takes the shared store's model,
        ``$MANDACARU_PROPOSAL_DATA/proposal_model.npz``, when there is one;
        ``False`` uses none.  Without a model, or with one that is not ready,
        the gradient softmax is in effect.  Loaded at construction, frozen
        for the run.
    """

    citation_method = "valqa"
    #: The name in the ``[SYSTEM]`` block's title.
    log_title = "VALQA"
    _result_class = VALQAResult
    #: The model is bound to the run's Hamiltonian and pool.
    _needs_problem = True
    _draws_from_gradients = True

    def __init__(self, hamiltonian=None, proposal_model=None,
                 **chain_options):
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

    def _prepare_proposal(self, problem) -> None:
        self._bound = (self._model.bind(problem)
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
            text = (f"learned GNN-GP (model {self._model.version}) mixed with "
                    f"{gradient}, epsilon {self._model.mixing:g}")
        return {"proposal": f"{text}, Metropolis-Hastings",
                "proposal_model": (str(self.proposal_model)
                                   if self.proposal_model is not None
                                   else "none")}

    def _result_extras(self) -> dict:
        return {"proposal_model": self._model_version(),
                "model_ready": self._model_in_use}
