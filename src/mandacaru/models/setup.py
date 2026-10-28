"""Geometry/basis metadata normalized before native computation."""
from .. import _native
DEFAULT_ELEMENT_KEY="*"
PER_ELEMENT="per-element"

def is_per_element_basis(value):
    return isinstance(value,dict) and "name" not in value and bool(value)

def resolve_basis(value):
    if isinstance(value,str):return value,{}
    if is_per_element_basis(value):return PER_ELEMENT,dict(value)
    if isinstance(value,dict) and "name" in value:
        options=dict(value);return options.pop("name"),options
    raise TypeError("basis must be a canonical name or a basis specification")

def pseudopotential_family(name):
    from ..pseudopotentials.families import lookup_family
    return lookup_family(name)

def monkhorst_pack_kpts(value):
    options=value if isinstance(value,dict) else {"size":(1,1,1) if value is None else value}
    size=tuple(options.get("size",(1,1,1)));gamma=options.get("gamma",False)
    mesh=_native.Context(threads=1).kmesh(size,gamma=gamma,time_reversal=False)
    return size,gamma,mesh["full"]

def validate_ghosts(value,count):
    if value is not None and len(value):raise NotImplementedError("native ghost-basis geometry assembly is unavailable")
    return ()
