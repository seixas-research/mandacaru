# -*- coding: utf-8 -*-
# file: __init__.py

# This code is part of Mandacaru. 
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Variational quantum algorithms.

The stable solvers -- classical RHF/UHF, VQE, fixed-layer HVA,
**ADAPT-VQE** (the default method everywhere), and the subspace-search
variants -- are reached **only** through :class:`Mandacaru` with their
``method`` names.  The solver classes themselves are
the internal layer and are deliberately **not exported**; what is exported is
the calculator and the result types the solvers return.  The periodic
methods ``"bloch-vqe"`` and ``"bloch-adapt-vqe"`` are reached the same way, with
``kpts``.  Solvers outside the stable API plug in through
:func:`register_method`.
"""

from .adapt_vqe import (ADAPTVQEResult, AdaptIteration, GRADIENT_METHODS,
                        resolve_gradient_method)
from .deflation import EnergyLevels
from .dry_run import QubitEstimate, count_basis_functions, estimate_qubits
from .bloch import BLOCH_METHODS, SpectralFunction
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
from .active_space import (ACTIVE_SELECTIONS, ActiveSpace,
                           resolve_active_space)
from .mp2 import MP2Result, mp2_energy, mp2_natural_orbitals
from .hartree_fock import (RHF, RHFResult, UHF, UHFResult, natural_orbitals,
                           transform_integrals)
from .mean_field import MeanFieldResult
from .subspace import SubspaceADAPTVQEResult, SubspaceVQEResult
from .base import format_pauli_sum
from .calculator import (DEFAULT_METHOD, STABLE_METHODS, Mandacaru,
                         available_methods, experimental_methods,
                         register_method, resolve_method)
from .forces import ForceResult, hellmann_feynman_gradient, nuclear_gradient
from .interaction import InteractionEnergy, interaction_energy
from .rdm import electronic_energy, one_rdm, particle_number, two_rdm
from .volumetric import (NaturalOrbitals, QUANTITIES, VolumetricField,
                         state_natural_orbitals, volumetric_field)
from .vqe import VQEResult
from .qpe import (QPEMemoryEstimate, QPEResult, QuantumPhaseEstimation,
                  phase_estimation, qpe_memory_estimate)
from .time_evolution import SuzukiTrotter, time_evolve
from .quantum_echoes import (EchoSpectrum, QuantumEchoes, QuantumEchoesResult,
                             fourier_spectrum, quantum_echoes)
from .nested_otoc import NestedOTOC, NestedOTOCResult, OTOCSpectrum

__all__ = [
    "NestedOTOC",
    "NestedOTOCResult",
    "OTOCSpectrum",
    "SuzukiTrotter",
    "time_evolve",
    "QuantumEchoes",
    "QuantumEchoesResult",
    "quantum_echoes",
    "EchoSpectrum",
    "fourier_spectrum",
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
    "MeanFieldResult",
    "natural_orbitals",
    "ACTIVE_SELECTIONS",
    "ActiveSpace",
    "resolve_active_space",
    "MP2Result",
    "mp2_energy",
    "mp2_natural_orbitals",
    "transform_integrals",
    "ADAPTVQEResult",
    "AdaptIteration",
    "GRADIENT_METHODS",
    "resolve_gradient_method",
    "BLOCH_METHODS",
    "SpectralFunction",
    "EnergyLevels",
    "SubspaceVQEResult",
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
    "Mandacaru",
    "QubitEstimate",
    "estimate_qubits",
    "count_basis_functions",
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
    "VolumetricField",
    "NaturalOrbitals",
    "QUANTITIES",
    "volumetric_field",
    "state_natural_orbitals",
]


def __getattr__(name):
    """Refuse the retired public name instead of letting it look missing."""
    if name == "BlochCalculator":
        from .bloch import RETIRED_BLOCH_CLASS
        raise AttributeError(RETIRED_BLOCH_CLASS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
