import warnings
import networkx as nx
from typing import Any, Iterable

from ._base import GraphAdapter
from ..core.structure import EdgeType

__all__ = [
    'NetworkXAdapter',
    'sync_back',
    'to_backend',
]


class NetworkXAdapter(GraphAdapter):
    """
    Export a `corneto.Graph` to NetworkX.

    * Uses **MultiDiGraph** / **MultiGraph** so we keep parallel edges.
    * Skips   hyper-edges by default (can be toggled).
    * Preserves vertex and edge attributes held in
      `graph._vertex_attr` and `graph._edge_attr`.
    """

    def export(
        self,
        graph: "Graph",
        *,
        directed: bool | None = None,
        skip_hyperedges: bool = True,
    ) -> nx.Graph:

        # ──────────────────────────────────────────────────────────────
        #  1. Which NetworkX class?
        # ──────────────────────────────────────────────────────────────
        if directed is None:
            directed = graph._default_edge_type is EdgeType.DIRECTED

        create_using = nx.MultiDiGraph if directed else nx.MultiGraph
        G: nx.Graph = create_using()

        # ──────────────────────────────────────────────────────────────
        #  2. Add vertices (with attributes, if any)
        # ──────────────────────────────────────────────────────────────
        for vid in graph._vertices:                              # OrderedDict keys
            attrs = dict(graph._vertex_attr.get(vid, {}))
            G.add_node(vid, **attrs)

        # ──────────────────────────────────────────────────────────────
        #  3. Add edges   (binary only ⇢ N× Graph)                   .
        # ──────────────────────────────────────────────────────────────
        for idx, (src_set, dst_set) in enumerate(graph._edges):
            attrs = dict(graph._edge_attr[idx]) if idx < len(graph._edge_attr) else {}

            # ---- binary edge? --------------------------------------
            if len(src_set) == 1 and len(dst_set) == 1:
                src = next(iter(src_set))
                dst = next(iter(dst_set))
                G.add_edge(src, dst, **attrs)
                continue

            # ---- hyper-edge handling -------------------------------
            if skip_hyperedges:
                warnings.warn(
                    f"Skipping hyper-edge #{idx}: {src_set} → {dst_set}",
                    stacklevel=2,
                )
                continue

            # fallback: fully connect every source with every target
            for s in _iter(src_set):
                for t in _iter(dst_set):
                    G.add_edge(s, t, **attrs)

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
def to_backend(graph: "Graph", **kwargs) -> nx.Graph:
    """
    `graph` → NetworkX graph   (keeps attributes & parallel edges).

    Example
    -------
    ```python
    nx_g = to_backend(corneto_graph, directed=True)
    ```
    """
    return NetworkXAdapter().export(graph, **kwargs)


def sync_back(nx_graph: nx.Graph, gg_graph: "Graph") -> None:  # pragma: no cover
    """
    Optional reverse-adapter.  Not implemented yet – by design, the
    current workflow treats NetworkX as a read-only view.
    """
    raise NotImplementedError("Syncing back from NetworkX is not supported yet.")
