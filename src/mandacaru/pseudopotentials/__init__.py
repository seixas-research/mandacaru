"""Optional pseudopotential setup; modules are imported only when requested."""

from importlib import import_module

_EXPORTS = {'COMMON_OPTIONS': ('.families', 'COMMON_OPTIONS'), 'DEFAULT_FAMILY': ('.families', 'DEFAULT_FAMILY'), 'PSEUDO_FAMILIES': ('.families', 'PSEUDO_FAMILIES'), 'FamilySpec': ('.families', 'FamilySpec'), 'canonical_family_name': ('.families', 'canonical_family_name'), 'family_listing': ('.families', 'family_listing'), 'family_names': ('.families', 'family_names'), 'lookup_family': ('.families', 'lookup_family'), 'register_family': ('.families', 'register_family'), 'resolve_family': ('.families', 'resolve_family'), 'unregister_family': ('.families', 'unregister_family'), 'FAMILY_VARIABLES': ('.environment', 'FAMILY_VARIABLES'), 'LibraryPathError': ('.environment', 'LibraryPathError'), 'paw_dataset_folder': ('.environment', 'paw_dataset_folder'), 'Channel': ('.dataset', 'Channel'), 'PseudoPotential': ('.dataset', 'PseudoPotential'), 'FORMAT_VERSION': ('.io', 'FORMAT_VERSION'), 'LIBRARY_ELEMENTS': ('.io', 'LIBRARY_ELEMENTS'), 'LIBRARY_Z_MAX': ('.io', 'LIBRARY_Z_MAX'), 'available_elements': ('.io', 'available_elements'), 'library_file': ('.io', 'library_file'), 'load_pseudopotential': ('.io', 'load_pseudopotential'), 'save_pseudopotential': ('.io', 'save_pseudopotential'), 'KBProjector': ('.orbitals', 'KBProjector'), 'PseudoAtomicOrbital': ('.orbitals', 'PseudoAtomicOrbital'), 'pseudo_basis': ('.orbitals', 'pseudo_basis'), 'valence_electrons': ('.orbitals', 'valence_electrons'), 'log_derivative_ae': ('.partial_waves', 'log_derivative_ae'), 'PAW_FAMILY': ('.paw', 'PAW_FAMILY'), 'PAWChannel': ('.paw', 'PAWChannel'), 'PAWDataset': ('.paw', 'PAWDataset'), 'build_paw_library': ('.paw', 'build_paw_library'), 'check_paw_channel': ('.paw', 'check_paw_channel'), 'compensation_coulomb': ('.paw', 'compensation_coulomb'), 'compensation_potential': ('.paw', 'compensation_potential'), 'compensation_shape': ('.paw', 'compensation_shape'), 'generate_paw': ('.paw', 'generate_paw'), 'get_paw': ('.paw', 'get_paw'), 'log_derivative_paw': ('.paw', 'log_derivative_paw'), 'paw_coupling_blocks': ('.paw', 'paw_coupling_blocks'), 'paw_eigenstate': ('.paw', 'paw_eigenstate'), 'paw_library_path': ('.paw', 'paw_library_path'), 'paw_overlap_blocks': ('.paw', 'paw_overlap_blocks'), 'paw_projectors': ('.paw', 'paw_projectors'), 'paw_spectrum': ('.paw', 'paw_spectrum'), 'reconstruct_ae': ('.paw', 'reconstruct_ae'), 'report_paw': ('.paw', 'report_paw'), 'smooth_partial_waves': ('.paw', 'smooth_partial_waves'), 'UPAW_FAMILY': ('.paw', 'UPAW_FAMILY'), 'build_upaw_library': ('.paw', 'build_upaw_library'), 'generate_upaw': ('.paw', 'generate_upaw'), 'get_upaw': ('.paw', 'get_upaw'), 'upaw_library_path': ('.paw', 'upaw_library_path')}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = (
        import_module("." + attribute, __name__)
        if module == "."
        else getattr(import_module(module, __name__), attribute)
    )
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
