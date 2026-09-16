# -*- coding: utf-8 -*-
# file: __init__.py

# This code is part of Carcará. 
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Variational quantum algorithms.

The stable solvers: VQE, **ADAPT-VQE** (the default method everywhere), the
excited-state extensions and the periodic :class:`BlochCalculator`, all reached
through :class:`Carcara`.  Solvers outside the stable API plug in through
:func:`register_method` and are not exported from here.
"""

from .adapt_vqe import ADAPTVQE, ADAPTVQEResult, AdaptIteration
from .deflation import DeflationMixin, EnergyLevels
from .dry_run import QubitEstimate, count_basis_functions, estimate_qubits
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
from .hartree_fock import (RHF, RHFResult, UHF, UHFResult, natural_orbitals,
                           transform_integrals)
from .subspace import (
    SubspaceADAPTVQE,
    SubspaceADAPTVQEResult,
    SubspaceVQE,
    SubspaceVQEResult,
)
from .base import format_pauli_sum
from .calculator import (DEFAULT_METHOD, METHODS, STABLE_METHODS, Carcara,
                         available_methods, experimental_methods,
                         register_method, resolve_method)
from .forces import ForceResult, hellmann_feynman_gradient, nuclear_gradient
from .interaction import InteractionEnergy, interaction_energy
from .rdm import electronic_energy, one_rdm, particle_number, two_rdm
from .vqe import VQE, VQEResult
from .qpe import (QPEMemoryEstimate, QPEResult, QuantumPhaseEstimation,
                  phase_estimation, qpe_memory_estimate)

__all__ = [
    "VQE",
    "VQEResult",
    "QuantumPhaseEstimation",
    "QPEResult",
    "QPEMemoryEstimate",
    "phase_estimation",
    "qpe_memory_estimate",
    "RHF",
    "RHFResult",
    "UHF",
    "UHFResult",
    "natural_orbitals",
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
    "Carcara",
    "QubitEstimate",
    "estimate_qubits",
    "count_basis_functions",
    "METHODS",
    "STABLE_METHODS",
    "DEFAULT_METHOD",
    "available_methods",
    "experimental_methods",
    "register_method",
    "resolve_method",
    "format_pauli_sum",
    "nuclear_gradient",
    "interaction_energy",
    "InteractionEnergy",
    "hellmann_feynman_gradient",
    "ForceResult",
    "one_rdm",
    "two_rdm",
    "electronic_energy",
    "particle_number",
]
