# -*- coding: utf-8 -*-
# file: __init__.py

# This code is part of Carcará. 
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Variational quantum algorithms."""

from .adapt_vqe import ADAPTVQE, ADAPTVQEResult, AdaptIteration
from .deflation import DeflationMixin, EnergyLevels
from .bloch import BandStructure, BlochCalculator
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
from .subspace import (
    SubspaceADAPTVQE,
    SubspaceADAPTVQEResult,
    SubspaceVQE,
    SubspaceVQEResult,
)
from .vasqe import (
    SubspaceVASQE,
    VASQE,
    VASQEResult,
    annealed_temperature,
    softmax_selection_probabilities,
)
from .base import format_pauli_sum
from .calculator import METHODS, QuantumCalculator, resolve_method
from .forces import ForceResult, hellmann_feynman_gradient, nuclear_gradient
from .rdm import electronic_energy, one_rdm, particle_number, two_rdm
from .vqe import VQE, VQEResult

__all__ = [
    "VQE",
    "VQEResult",
    "RHF",
    "RHFResult",
    "UHF",
    "transform_integrals",
    "ADAPTVQE",
    "ADAPTVQEResult",
    "AdaptIteration",
    "BlochCalculator",
    "BandStructure",
    "EnergyLevels",
    "SubspaceVQE",
    "SubspaceVQEResult",
    "SubspaceADAPTVQE",
    "SubspaceADAPTVQEResult",
    "VASQE",
    "VASQEResult",
    "SubspaceVASQE",
    "softmax_selection_probabilities",
    "annealed_temperature",
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
    "QuantumCalculator",
    "METHODS",
    "resolve_method",
    "format_pauli_sum",
    "nuclear_gradient",
    "hellmann_feynman_gradient",
    "ForceResult",
    "one_rdm",
    "two_rdm",
    "electronic_energy",
    "particle_number",
]
