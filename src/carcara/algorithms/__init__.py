# -*- coding: utf-8 -*-
# file: __init__.py

# This code is part of Carcará. 
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Variational quantum algorithms."""

from .adapt_vqe import (
    AdaptAnsatz,
    AdaptIteration,
    AdaptVQE,
    AdaptVQEResult,
    CircuitMetrics,
    profile_ansatz,
)
from .expressivity import (
    ADAPTExpressivityTracker,
    ExpressibilityResult,
    ExpressibilityStep,
    active_space_dimension,
    calculate_haar_distribution,
    calculate_kl_divergence,
    compute_expressibility,
    estimate_effective_dimension,
    plot_expressivity_growth,
    plot_fidelity_distribution,
    sample_pqc_fidelities,
    track_adapt_expressivity,
)
from .hartree_fock import RHF, RHFResult, UHF, transform_integrals
from .vqe import VQE, VQEResult

__all__ = [
    "VQE",
    "VQEResult",
    "RHF",
    "RHFResult",
    "UHF",
    "transform_integrals",
    "AdaptVQE",
    "AdaptVQEResult",
    "AdaptAnsatz",
    "AdaptIteration",
    "CircuitMetrics",
    "profile_ansatz",
    "compute_expressibility",
    "ExpressibilityResult",
    "ExpressibilityStep",
    "sample_pqc_fidelities",
    "calculate_kl_divergence",
    "calculate_haar_distribution",
    "active_space_dimension",
    "estimate_effective_dimension",
    "ADAPTExpressivityTracker",
    "track_adapt_expressivity",
    "plot_fidelity_distribution",
    "plot_expressivity_growth",
]
