"""Public interfaces loaded on demand; optional integrations stay optional."""

from importlib import import_module

_EXPORTS = {
    "AtomicResult": (".atomic_solver", "AtomicResult"),
    "solve_atom": (".atomic_solver", "solve_atom"),
    "solve_radial": (".atomic_solver", "solve_radial"),
    "BasisFunction": (".base", "BasisFunction"),
    "FILTER_METHODS": (".filtering", "FILTER_METHODS"),
    "FILTER_NYQUIST_FRACTION": (".filtering", "FILTER_NYQUIST_FRACTION"),
    "cutoff_energy_ev": (".filtering", "cutoff_energy_ev"),
    "filter_label": (".filtering", "filter_label"),
    "filter_radial": (".filtering", "filter_radial"),
    "filter_table": (".filtering", "filter_table"),
    "residual_weight": (".filtering", "residual_weight"),
    "resolve_filter_cutoff": (".filtering", "resolve_filter_cutoff"),
    "validate_filter_cutoff": (".filtering", "validate_filter_cutoff"),
    "BasisSet": (".factory", "BasisSet"),
    "HAOBasisSet": (".factory", "HAOBasisSet"),
    "GaussianBasisSet": (".factory", "GaussianBasisSet"),
    "GTOBasisSet": (".factory", "GTOBasisSet"),
    "NAOAEBasisSet": (".factory", "NAOAEBasisSet"),
    "NAOBasisSet": (".factory", "NAOBasisSet"),
    "PerElementBasisSet": (".factory", "PerElementBasisSet"),
    "Pople631GBasisSet": (".factory", "Pople631GBasisSet"),
    "NAMED_BASIS_SETS": (".gaussian_families", "NAMED_BASIS_SETS"),
    "GaussianRecipe": (".gaussian_families", "GaussianRecipe"),
    "available_basis_names": (".gaussian_families", "available_basis_names"),
    "count_functions": (".gaussian_families", "count_functions"),
    "gaussian_shells": (".gaussian_families", "gaussian_shells"),
    "parse_basis_name": (".gaussian_families", "parse_basis_name"),
    "shell_notation": (".gaussian_families", "shell_notation"),
    "GaussianOrbital": (".gaussian", "GaussianOrbital"),
    "HydrogenicAtomicOrbital": (".hao", "HydrogenicAtomicOrbital"),
    "DEFAULT_ENERGY_SHIFT": (".nao", "DEFAULT_ENERGY_SHIFT"),
    "NumericalAtomicOrbital": (".nao", "NumericalAtomicOrbital"),
    "energy_shift_to_rc": (".nao", "energy_shift_to_rc"),
    "RadialFunction": (".nao_ae", "RadialFunction"),
    "build_species": (".nao_ae", "build_species"),
    "confinement_potential": (".nao_ae", "confinement_potential"),
    "effective_charge_for_radius": (".nao_ae", "effective_charge_for_radius"),
    "hydrogenic_function": (".nao_ae", "hydrogenic_function"),
    "tier_specification": (".nao_ae", "tier_specification"),
    "pople_631g_shells": (".pople", "pople_631g_shells"),
    "occupied_subshells": (".sto_ng", "occupied_subshells"),
    "slater_exponent": (".sto_ng", "slater_exponent"),
    "sto_ng_contraction": (".sto_ng", "sto_ng_contraction"),
    "sto_ng_shells": (".sto_ng", "sto_ng_shells"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = (
        import_module("." + attribute, __name__)
        if module == "."
        else getattr(import_module(module, __name__), attribute)
    )
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
