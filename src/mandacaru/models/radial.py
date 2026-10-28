"""Radial basis recipe metadata, independent of numerical basis generators."""

#: The split-valence schemes ``zeta_split`` names: which function every extra
#: zeta is split from -- the first zeta, or the previous (last) one.
ZETA_SPLITS = ("first_zeta", "last_zeta")
DEFAULT_ZETA_SPLIT = "first_zeta"

#: ``zeta_split="last_zeta"``: the fraction of the orbital's **squared** norm
#: left outside the split radius, each higher zeta splitting the previous one
#: with the fraction halved.
DEFAULT_SPLIT_NORM = 0.15

#: ``zeta_split="first_zeta"``: the **norm** -- not the squared norm -- of the
#: tail left outside the split radius of the second, third and fourth zeta,
#: every one of them split from the *first* zeta.  0.15 is a squared-norm
#: fraction of 0.0225, so it is not comparable with
#: :data:`DEFAULT_SPLIT_NORM` digit for digit.
DEFAULT_TAIL_NORMS = (0.15, 0.31, 0.54)


def validate_zeta_split(spec):
    """Normalize a ``zeta_split`` option to one of :data:`ZETA_SPLITS`, or
    a numeric tail norm (or sequence of tail norms) for first-zeta mode, or
    ``None`` when it was not written."""
    if spec is None:
        return None
    if isinstance(spec, (int, float)):
        val = float(spec)
        if not (0.0 < val < 1.0):
            raise ValueError(f"zeta_split tail norm must be in (0, 1), got {spec!r}")
        return val
    if isinstance(spec, (tuple, list)):
        vals = tuple(float(x) for x in spec)
        if not vals or any(not (0.0 < x < 1.0) for x in vals):
            raise ValueError(
                f"zeta_split tail norms must be numbers in (0, 1), got {spec!r}"
            )
        return vals
    key = str(spec).strip().lower().replace("-", "_")
    if key not in ZETA_SPLITS:
        raise ValueError(f"zeta_split must be one of {list(ZETA_SPLITS)}, got {spec!r}")
    return key


def resolve_split_scheme(zeta_split=None):
    """``(split_norm, tail_norms)`` of the split-valence scheme ``zeta_split``.

    The scheme fixes its own parameter:
    ``"first_zeta"`` (the default) returns ``(None, DEFAULT_TAIL_NORMS)`` --
    every extra zeta split from the first by the tail's **norm** -- and
    ``"last_zeta"`` returns ``(DEFAULT_SPLIT_NORM, None)`` -- each zeta split
    from the previous one by the tail's **squared** norm, halved per zeta.
    A float or tuple of floats specifies custom tail norms for ``"first_zeta"`` mode.
    """
    scheme = validate_zeta_split(zeta_split)
    if scheme is None or scheme == "first_zeta":
        return None, DEFAULT_TAIL_NORMS
    if scheme == "last_zeta":
        return DEFAULT_SPLIT_NORM, None
    if isinstance(scheme, tuple):
        return None, scheme
    if isinstance(scheme, float):
        return None, (scheme, *DEFAULT_TAIL_NORMS[1:])
    return DEFAULT_SPLIT_NORM, None


#: Default NAO size.  Double zeta plus polarization: single zeta gives a shell
#: no radial freedom (it cannot contract or expand) and no angular freedom (it
#: cannot bend into a bond), and both matter for chemistry.  DZP is also where
#: the real-space grid stops being the limiting approximation -- see the module
#: docstring on why TZ and beyond need a finer mesh to be worth their cost.
DEFAULT_NAO_SIZE = "DZP"

#: Recognized basis-size names -> (number of zetas, polarization shells).
ZETA_NAMES = {
    "SZ": (1, 0),
    "SZP": (1, 1),
    "DZ": (2, 0),
    "DZP": (2, 1),
    "DZ2P": (2, 2),
    "TZ": (3, 0),
    "TZP": (3, 1),
    "TZ2P": (3, 2),
    "QZ": (4, 0),
    "QZP": (4, 1),
    "QZ2P": (4, 2),
}


def resolve_zeta(name):
    """Resolve a basis-size spec to ``(n_zeta, n_polarization)``.

    Accepts a name from :data:`ZETA_NAMES` (case-insensitive) or an explicit
    ``(n_zeta, n_polarization)`` pair.
    """
    if isinstance(name, (tuple, list)) and len(name) == 2:
        return int(name[0]), int(name[1])
    key = str(name).strip().upper().replace("-", "").replace(" ", "")
    if key not in ZETA_NAMES:
        raise ValueError(
            f"unknown basis size {name!r}; use one of "
            f"{sorted(ZETA_NAMES)} or an (n_zeta, n_polarization) pair"
        )
    return ZETA_NAMES[key]


NAO_AE_DEFAULTS = dict(
    tier=1,
    onset=3.0,
    width=1.0,
    scale=1.0,
    extra=None,
    linear_dependence_tol=1e-4,
    tail_norm=1e-4,
    points=4000,
    r_max=25.0,
)
