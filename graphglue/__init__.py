# graphglue/__init__.py
"""graphglue: single import, full API."""
from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any

# Lazily exposed submodules (imported on first attribute access)
_lazy_submodules = {
    # namespaces
    "adapters": "graphglue.adapters",
    "io": "graphglue.io",
    "core": "graphglue.core",
    "algorithms": "graphglue.algorithms",
    "utils": "graphglue.utils",
    # adapter modules (direct convenience)
    "graphml": "graphglue.adapters.GraphML_adapter",
    "sif": "graphglue.adapters.SIF_adapter",
    "sbml": "graphglue.adapters.sbml_adapter",
    "parquet": "graphglue.adapters.GraphDir_Parquet_adapter",
    "networkx": "graphglue.adapters.networkx_adapter",
    "igraph": "graphglue.adapters.igraph_adapter",
    "jsonio": "graphglue.adapters.json_adapter",
    "dataframe": "graphglue.adapters.dataframe_adapter",
    # io modules
    "csvio": "graphglue.io.csv",
    "excelio": "graphglue.io.excel",
    "annnet" : "graphglue.io.io_annnet"
}

# Curated top-level symbols (lazy). name -> (module, attribute)
_lazy_symbols: dict[str, tuple[str, str]] = {
    # Core
    "Graph": ("graphglue.core.graph", "Graph"),

    # Stdlib JSON I/O
    "to_json": ("graphglue.adapters.json_adapter", "to_json"),
    "from_json": ("graphglue.adapters.json_adapter", "from_json"),

    # NetworkX adapter (optional dependency)
    "to_nx": ("graphglue.adapters.networkx_adapter", "to_nx"),
    "from_nx": ("graphglue.adapters.networkx_adapter", "from_nx"),
    "from_nx_only": ("graphglue.adapters.networkx_adapter", "from_nx_only"),

    # GraphML
    "to_graphml": ("graphglue.adapters.GraphML_adapter", "to_graphml"),
    "from_graphml": ("graphglue.adapters.GraphML_adapter", "from_graphml"),

    # SIF
    "to_sif": ("graphglue.adapters.SIF_adapter", "to_sif"),
    "from_sif": ("graphglue.adapters.SIF_adapter", "from_sif"),

    # SBML (common direction)
    "from_sbml": ("graphglue.adapters.sbml_adapter", "from_sbml"),
    # If you add export later: "to_sbml": ("graphglue.adapters.sbml_adapter", "to_sbml"),

    # Parquet GraphDir
    "write_parquet_graphdir": ("graphglue.adapters.GraphDir_Parquet_adapter", "write_parquet_graphdir"),
    "read_parquet_graphdir": ("graphglue.adapters.GraphDir_Parquet_adapter", "read_parquet_graphdir"),
}

__all__ = sorted(set(list(_lazy_submodules) + list(_lazy_symbols)))


def __getattr__(name: str) -> Any:  # PEP 562: lazy attribute resolution
    if name in _lazy_submodules:
        return import_module(_lazy_submodules[name])
    if name in _lazy_symbols:
        mod, attr = _lazy_symbols[name]
        return getattr(import_module(mod), attr)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))


# Version: prefer internal, then fall back to distribution metadata
try:
    from ._version import __version__  # type: ignore
except Exception:
    try:
        __version__ = _pkg_version("graphglue")
    except PackageNotFoundError:
        __version__ = "0.0.0"
