# adapters/networkx.py
from typing import Any
from enum import Enum
import json
import networkx as nx

from ..core._base import BaseGraph, EdgeType, Attr, Attributes
from ..core.adapter import IncidenceAdapter   # your backend


# ────────────────────────────────
# Utilities
# ────────────────────────────────
def _is_mapping(v: Any) -> bool:
    return hasattr(v, "items")

def _serialize_value(v: Any) -> Any:
    if isinstance(v, Enum):
        return v.name
    if _is_mapping(v):
        return dict(v)
    return v

def _serialize_attrs(attrs: Any, public_only=False) -> dict:
    if isinstance(attrs, Attributes):
        attrs = dict(attrs)   # instead of attrs.to_dict()
    if not _is_mapping(attrs):
        return {}
    if public_only:
        return {k: _serialize_value(v) for k, v in attrs.items() if not str(k).startswith("__")}
    return {k: _serialize_value(v) for k, v in attrs.items()}



# ────────────────────────────────
# Core legacy exporter (no manifest)
# ────────────────────────────────
def _export_legacy(graph: BaseGraph, directed=True, skip_hyperedges=True, public_only=False, **kwargs):
    G = nx.MultiDiGraph() if directed else nx.MultiGraph()

    # Add vertices
    for v in graph.V:
        v_attr = _serialize_attrs(graph.get_attr_vertex(v), public_only=public_only)
        G.add_node(v, **v_attr)

    # Add edges
    for eidx, (src, dst) in graph.edges():
        e_attr = _serialize_attrs(graph.get_attr_edge(eidx), public_only=public_only)

        if len(src) == 1 and len(dst) == 1:
            u = next(iter(src)); v = next(iter(dst))
            G.add_edge(u, v, key=eidx, **e_attr)
        else:
            if skip_hyperedges:
                continue
            # explode hyperedge
            for u in src:
                for v in dst:
                    G.add_edge(u, v, key=eidx, **e_attr)

    return G


# ────────────────────────────────
# Manifest-aware exporter
# ────────────────────────────────
def to_nx(adapter_graph: IncidenceAdapter, directed=True,
          hyperedge_mode="skip", layer=None, layers=None,
          public_only=False):
    """
    Export IncidenceAdapter → (networkx.Graph, manifest).
    Manifest preserves hyperedges, layers, per-endpoint attributes, stable edge IDs.
    """
    nxG = _export_legacy(
        adapter_graph,
        directed=directed,
        skip_hyperedges=(hyperedge_mode == "skip"),
        public_only=public_only
    )

    deep = getattr(adapter_graph, "deep", None)
    manifest = {}
    if deep:
        manifest = {
            "edges": deep.edge_definitions.copy(),
            "weights": deep.edge_weights.copy(),
            "layers": {lid: list(deep.get_layer_edges(lid)) for lid in deep.layers(include_default=True)},
        }

    return nxG, manifest


# ────────────────────────────────
# Manifest save/load
# ────────────────────────────────
def save_manifest(manifest: dict, path: str):
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)

def load_manifest(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ────────────────────────────────
# Importer
# ────────────────────────────────
def from_nx(nxG: nx.Graph, manifest: dict) -> IncidenceAdapter:
    H = IncidenceAdapter()

    # Add vertices
    for v, attrs in nxG.nodes(data=True):
        H.add_vertex(v, **attrs)

    # Add edges from manifest
    for eid, defn in manifest.get("edges", {}).items():
        kind = defn[-1]
        if kind == "regular":
            u, v = defn[0], defn[1]
            H.add_edge({u}, {v})
        elif kind == "hyper":
            head, tail = set(defn[0]), set(defn[1])
            H.deep.add_hyperedge(head=head, tail=tail)

    # Reapply weights
    for eid, w in manifest.get("weights", {}).items():
        if eid in H.deep.edge_weights:
            H.deep.edge_weights[eid] = w

    # Reapply layers
    for lid, eids in manifest.get("layers", {}).items():
        if lid not in H.deep.layers(include_default=True):
            H.deep.add_layer(lid)
        for eid in eids:
            if eid in H.deep.edge_definitions:
                H.deep.add_edge_to_layer(lid, eid)

    return H


# ────────────────────────────────
# Compatibility shims
# ────────────────────────────────
def to_backend(graph, **kwargs):
    """Legacy wrapper: returns only the networkx.Graph, no manifest."""
    return _export_legacy(graph, **kwargs)

class NetworkXAdapter:
    def export(self, graph, **kwargs):
        return _export_legacy(graph, **kwargs)
