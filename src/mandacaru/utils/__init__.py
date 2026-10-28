"""Public interfaces loaded on demand; optional integrations stay optional."""

from importlib import import_module

_EXPORTS = {
    "import_module": ("importlib", "import_module"),
    "REFERENCES": (".bibliography", "REFERENCES"),
    "bibtex": (".bibliography", "bibtex"),
    "Logger": (".logging", "Logger"),
    "append_block": (".logging", "append_block"),
    "append_forces": (".logging", "append_forces"),
    "append_optimization_summary": (".logging", "append_optimization_summary"),
    "append_performance": (".logging", "append_performance"),
    "append_quantum_echoes": (".logging", "append_quantum_echoes"),
    "append_nested_otoc": (".logging", "append_nested_otoc"),
    "log_steps": (".logging", "log_steps"),
    "parse_output": (".logging", "parse_output"),
    "reset_log": (".logging", "reset_log"),
    "Timings": (".profiling", "Timings"),
    "backend_cores": (".profiling", "backend_cores"),
    "cpu_count": (".profiling", "cpu_count"),
    "current_memory_mb": (".profiling", "current_memory_mb"),
    "peak_memory_mb": (".profiling", "peak_memory_mb"),
    "Camera": (".viewer", "Camera"),
    "Viewer3D": (".viewer", "Viewer3D"),
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
