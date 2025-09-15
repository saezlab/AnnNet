import warnings
import networkx as nx
from typing import Any, Iterable
from enum import Enum

from ._base import GraphAdapter
from ..core.structure import EdgeType

def _is_mapping(x: Any) -> bool:
    return hasattr(x, "items") and callable(x.items)


def _serialize_value(v: Any) -> Any:
    # Convert Enums to their name, Attributes/mappings to dicts, keep scalars as-is
    if isinstance(v, Enum):
        return v.name
    if _is_mapping(v):
        return {kk: _serialize_value(vv) for kk, vv in v.items()}
    return v


def _serialize_attrs(attrs: Any) -> dict:
    """Turn Attributes (mapping-like) into a plain dict of JSON-ish values."""
    if not _is_mapping(attrs):
        return {}
    return {k: _serialize_value(v) for k, v in attrs.items()}


class NetworkXAdapter:
    """
    Export a BaseGraph/IncidenceAdapter to NetworkX.

    - Uses MultiDiGraph by default (pass directed=False for MultiGraph).
    - Skips hyper-edges by default (set skip_hyperedges=False to expand S×T).
    - Preserves vertex/edge attributes, serializing Enums/Attributes to dicts.
    """

    def export(
        self,
        graph,
        *,
        directed: bool | None = None,
        skip_hyperedges: bool = True,
    ) -> nx.Graph:
        # Decide graph type
        if directed is None:
            # default: directed; callers can override
            directed = True
        G: nx.Graph = nx.MultiDiGraph() if directed else nx.MultiGraph()

        # Add nodes with attributes
        for v in graph.V:
            v_attr = _serialize_attrs(graph.get_attr_vertex(v))
            G.add_node(v, **v_attr)

        # Add edges with attributes
        for eidx, (src_set, dst_set) in graph.edges():
            e_attr = _serialize_attrs(graph.get_attr_edge(eidx))

            # Binary edge
            if len(src_set) == 1 and len(dst_set) == 1:
                u = next(iter(src_set))
                w = next(iter(dst_set))
                # use eidx as the key so parallel edges are distinct/stable
                G.add_edge(u, w, key=eidx, **e_attr)
                continue

            # Hyper-edge
            if skip_hyperedges:
                # just ignore; user can opt-in to expansion
                continue
            for u in src_set:
                for w in dst_set:
                    G.add_edge(u, w, key=eidx, **e_attr)

        return G

# ───────────────────────────────────────────────────────────────────────
#  Helper functions
# ───────────────────────────────────────────────────────────────────────
def _iter(maybe_iterable: Any) -> Iterable[Any]:
    """Treat a single object like a 1-element iterable."""
    return maybe_iterable if isinstance(maybe_iterable, (set, list, tuple)) else (maybe_iterable,)


# ───────────────────────────────────────────────────────────────────────
#  Convenience shims (mirroring the old API)
# ───────────────────────────────────────────────────────────────────────
def to_backend(graph, **kwargs) -> nx.Graph:
    """
    Convert BaseGraph/IncidenceAdapter -> NetworkX graph.

    Example:
        nxG = to_backend(adapter_graph, directed=True)
    """
    return NetworkXAdapter().export(graph, **kwargs)


def sync_back(nx_graph: nx.Graph, gg_graph: "Graph") -> None:  # pragma: no cover
    """
    Optional reverse-adapter.  Not implemented yet – by design, the
    current workflow treats NetworkX as a read-only view.
    """
    raise NotImplementedError("Syncing back from NetworkX is not supported yet.")
