from __future__ import annotations

from importlib import import_module, util
from typing import TYPE_CHECKING

from . import available_backends, load_adapter
from ._proxy import BackendProxy

if TYPE_CHECKING:
    from ..core._graph import Graph

def get_adapter(name: str):
    """Return a new adapter instance for the optional backend."""
    return load_adapter(name)

def get_proxy(backend_name: str, graph: "Graph") -> BackendProxy:
    """Return a lazy proxy so users can write `G.nx.<algo>()` etc."""
    if backend_name not in ("networkx", "igraph"):
        raise ValueError(f"No backend '{backend_name}' registered")
    return BackendProxy(graph, backend_name)

def _backend_import_name(name: str) -> str:
    # import name differs from pip extra only for igraph (pip: python-igraph, import: igraph)
    return "igraph" if name == "igraph" else "networkx"

def ensure_materialized(backend_name: str, graph: "Graph") -> dict:
    """
    Convert (or re-convert) *graph* into the requested backend object and
    cache the result on the graph’s private state object. Returns:
      {"module": <backend module>, "graph": <backend graph>, "version": int}
    """
    modname = _backend_import_name(backend_name)
    if util.find_spec(modname) is None:
        raise ModuleNotFoundError(
            f"Optional backend '{backend_name}' is not installed. "
            f"Install with `pip install graphglue[{backend_name}]`."
        )

    cache = graph._state._backend_cache
    entry = cache.get(backend_name)

    if entry is None or graph._state.dirty_since(entry["version"]):
        backend_module = import_module(modname)  # e.g. 'networkx' or 'igraph'
        # import adapter module lazily and call its to_backend()
        adapter_mod = import_module(f"{__package__}.{backend_name}")
        converted = getattr(adapter_mod, "to_backend")(graph)

        entry = cache[backend_name] = {
            "module": backend_module,
            "graph": converted,
            "version": graph._state.version,
        }

    return entry
