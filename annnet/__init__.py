# annnet/__init__.py
"""annnet: single import, full API."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any

# Lazily exposed submodules (imported on first attribute access)
_lazy_submodules = {
    # namespaces
    "adapters": "annnet.adapters",
    "io": "annnet.io",
    "core": "annnet.core",
    "algorithms": "annnet.algorithms",
    "utils": "annnet.utils",
    # adapter modules (direct convenience)
    "graphml": "annnet.adapters.GraphML_adapter",
    "sif": "annnet.adapters.SIF_adapter",
    "sbml": "annnet.adapters.sbml_adapter",
    "parquet": "annnet.adapters.GraphDir_Parquet_adapter",
    "networkx": "annnet.adapters.networkx_adapter",
    "igraph": "annnet.adapters.igraph_adapter",
    "jsonio": "annnet.adapters.json_adapter",
    "dataframe": "annnet.adapters.dataframe_adapter",
    # io modules
    "csvio": "annnet.io.csv",
    "excelio": "annnet.io.excel",
    "annnet": "annnet.io.io_annnet",
}

# Curated top-level symbols (lazy). name -> (module, attribute)
_lazy_symbols: dict[str, tuple[str, str]] = {
    # Core
    "Graph": ("annnet.core.graph", "Graph"),
    # Stdlib JSON I/O
    "to_json": ("annnet.adapters.json_adapter", "to_json"),
    "from_json": ("annnet.adapters.json_adapter", "from_json"),
    # NetworkX adapter (optional dependency)
    "to_nx": ("annnet.adapters.networkx_adapter", "to_nx"),
    "from_nx": ("annnet.adapters.networkx_adapter", "from_nx"),
    "from_nx_only": ("annnet.adapters.networkx_adapter", "from_nx_only"),
    # GraphML
    "to_graphml": ("annnet.adapters.GraphML_adapter", "to_graphml"),
    "from_graphml": ("annnet.adapters.GraphML_adapter", "from_graphml"),
    # SIF
    "to_sif": ("annnet.adapters.SIF_adapter", "to_sif"),
    "from_sif": ("annnet.adapters.SIF_adapter", "from_sif"),
    # SBML (common direction)
    "from_sbml": ("annnet.adapters.sbml_adapter", "from_sbml"),
    # If you add export later: "to_sbml": ("annnet.adapters.sbml_adapter", "to_sbml"),
    # Parquet GraphDir
    "write_parquet_graphdir": (
        "annnet.adapters.GraphDir_Parquet_adapter",
        "write_parquet_graphdir",
    ),
    "read_parquet_graphdir": (
        "annnet.adapters.GraphDir_Parquet_adapter",
        "read_parquet_graphdir",
    ),
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
        __version__ = _pkg_version("annnet")
    except PackageNotFoundError:
        __version__ = "0.0.0"
