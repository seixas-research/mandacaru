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
