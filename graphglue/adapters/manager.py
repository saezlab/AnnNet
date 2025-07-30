from __future__ import annotations

import importlib
from typing import Dict, Type, Callable

from ._base import GraphAdapter                      # base class
from .networkx import to_backend as nx_to_backend, NetworkXAdapter
# from .igraph import to_backend as ig_to_backend, IGraphAdapter
# ...

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .._core._graph import Graph

from ._proxy import BackendProxy

__all__ = [
    'ensure_materialized',
    'get_adapter',
    'get_proxy',
]

# ---------------------------------------------------------------------------
# 1. Central registry --------------------------------------------------------
# ---------------------------------------------------------------------------
# Map backend name -> callable that converts gg.Graph -> backend graph
_REGISTRY = {
    "networkx": nx_to_backend,
    # "igraph": ig_to_backend,
}

# ---------------------------------------------------------------------------
# 2. Public helpers ----------------------------------------------------------
# ---------------------------------------------------------------------------
def get_adapter(name: str) -> GraphAdapter:
    """Return a *new* adapter instance of the requested backend."""
    try:
        return {
            "networkx": NetworkXAdapter,
            # "igraph": IGraphAdapter,
        }[name.lower()]()
    except KeyError:
        raise ValueError(f"No adapter registered for '{name}'") from None


def get_proxy(backend_name: str, graph: "Graph") -> BackendProxy:
    """Return a lazy proxy so users can write `G.nx.<algo>()`."""
    if backend_name not in _REGISTRY:
        raise ValueError(f"No backend '{backend_name}' registered")
    return BackendProxy(graph, backend_name)


def ensure_materialized(backend_name: str, graph: "Graph") -> dict:
    """
    Convert (or re-convert) *graph* into the requested backend object and
    cache the result on the graph’s private state object.  Returns the cache
    entry: {"module": nx, "graph": nx.Graph, "version": int}
    """
    cache = graph._state._backend_cache               # per-instance cache
    entry = cache.get(backend_name)

    if entry is None or graph._state.dirty_since(entry["version"]):
        # 1. import backend library lazily
        backend_module = importlib.import_module(backend_name)   # e.g. 'networkx'

        # 2. convert gg.Graph -> backend graph using the registered callable
        converted = _REGISTRY[backend_name](graph)

        # 3. stash result together with current version counter
        entry = cache[backend_name] = {
            "module":  backend_module,
            "graph":   converted,
            "version": graph._state.version,
        }

    return entry
