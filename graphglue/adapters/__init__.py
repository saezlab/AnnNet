from importlib import import_module, util

__all__ = ["available_backends", "load_adapter"]

# name -> (pip_import_name, submodule, class_name)
_BACKENDS = {
    "networkx": ("networkx", ".networkx", "NetworkXAdapter"),
    "igraph": ("igraph", ".igraph", "IGraphAdapter"),  # pip pkg is python-igraph; import is igraph
}


def _is_installed(name: str, modname: str) -> bool:
    return util.find_spec(modname) is not None


def available_backends() -> dict:
    return {name: _is_installed(name, mod) for name, (mod, _, _) in _BACKENDS.items()}


def load_adapter(name: str, *args, **kwargs):
    if name not in _BACKENDS:
        raise ValueError(f"Unknown adapter '{name}'")
    modname, submod, cls = _BACKENDS[name]
    if not _is_installed(name, modname):
        raise ModuleNotFoundError(
            f"Optional backend '{name}' is not installed. "
            f"Install with `pip install annnet[{name}]`."
        )
    mod = import_module(__name__ + submod)
    return getattr(mod, cls)(*args, **kwargs)


"""
Standardize via adapter pattern with shared interface + fallback to edge list + attributes.
Support common exchange formats (e.g., via GraphML) for indirect conversion.
Use reflection/meta-inspection to auto-map foreign structures to internal schema.
"""

"""
Add caching for repeated conversions if structure unchanged.
Track deltas for selective sync back (esp. for subgraphs).
Enable partial proxies: wrap only relevant substructures instead of whole graph.
For read-only ops: avoid sync entirely, document immutability contract.
"""
