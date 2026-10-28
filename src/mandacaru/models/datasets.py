"""Consistency checks for dataset metadata."""
import warnings

def _datasets_relativistic(datasets) -> bool:
    """The datasets' common relativistic-exchange flag (they must agree)."""
    flags = {bool(getattr(d, "relativistic_exchange", False))
             for d in datasets or ()}
    if len(flags) > 1:
        raise ValueError("the datasets mix relativistic and non-relativistic "
                         "exchange; the functional cannot match both")
    return flags.pop() if flags else False

def _warn_functional_mismatch(datasets, functional: str) -> None:
    """Warn when the datasets were not generated with the functional
    ``functional`` reads (:data:`.environment.DATASET_FUNCTIONALS`), and
    name the library folder of those datasets when the PAW-LCAO library
    ships one (``paw_dataset="pbe-sr"`` for PBE, HSE06 and r2SCAN)."""
    from ..pseudopotentials.environment import (DATASET_FUNCTIONALS,
                                                LIBRARY_FOLDERS)

    expected = DATASET_FUNCTIONALS.get(functional, functional)
    generated = sorted({str(getattr(d, "xc", "lda") or "lda").lower()
                        for d in datasets or ()})
    if generated and generated != [expected]:
        families = {str(getattr(d, "family", "")).lower()
                    for d in datasets or ()}
        relativity = {str(getattr(d, "relativity", "scalar")).lower()
                      for d in datasets or ()}
        folder = (LIBRARY_FOLDERS.get(("paw-lcao", expected,
                                       relativity.pop()))
                  if families == {"paw-lcao"} and len(relativity) == 1
                  else None)
        hint = (f"; Mandacaru(paw_dataset={folder!r}) reads the PAW-LCAO "
                f"datasets generated with {expected.upper()}"
                if folder is not None else "")
        warnings.warn(
            f"xc={functional!r} on datasets generated with "
            f"{'/'.join(g.upper() for g in generated)}, where it reads "
            f"datasets generated with {expected.upper()}: the molecule and "
            f"its datasets use different functionals{hint}", RuntimeWarning,
            stacklevel=3)
