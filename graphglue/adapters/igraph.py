# Placeholder for igraph.py
from typing import Any
from enum import Enum
import json

try:
    import igraph as ig  # python-igraph
except Exception as e:  # pragma: no cover
    raise RuntimeError("python-igraph is required: pip install igraph") from e

from ..core._base import BaseGraph  # types
from ..core.adapter import IncidenceAdapter  # backend

# ────────────────────────────────
# Utilities (mirrored from the NX adapter where relevant)
# ────────────────────────────────

def _is_mapping(v: Any) -> bool:
    return hasattr(v, "items")


def _serialize_attrs(attrs, public_only: bool = False) -> dict:
    try:
        attrs = dict(attrs)  # Attributes is mapping-like
    except Exception:
        return {}
    if public_only:
        return {k: v.name if isinstance(v, Enum) else (dict(v) if _is_mapping(v) else v)
                for k, v in attrs.items() if not str(k).startswith("__")}
    return {k: v.name if isinstance(v, Enum) else (dict(v) if _is_mapping(v) else v)
            for k, v in attrs.items()}


def _attrs_to_dict(a):
    try:
        items = dict(a)
    except Exception:
        return {}
    out = {}
    for k, v in items.items():
        if isinstance(v, Enum):
            out[k] = v.name
        elif hasattr(v, "items"):
            out[k] = {kk: (vv.name if isinstance(vv, Enum) else vv) for kk, vv in dict(v).items()}
        else:
            out[k] = v
    return out


def _safe_set_node_attrs(deep, vid, attrs: dict):
    if hasattr(deep, "set_node_attrs"):
        deep.set_node_attrs(vid, **attrs)
        return
    df = deep.vertex_attributes
    if vid not in getattr(df, "index", []):
        if hasattr(deep, "add_node"):
            deep.add_node(vid)
    for k in attrs.keys():
        if k not in df.columns:
            df[k] = None
        df.at[vid, k] = attrs[k]


def _safe_set_edge_attrs(deep, eid, attrs: dict):
    if hasattr(deep, "set_edge_attrs"):
        deep.set_edge_attrs(eid, **attrs)
        return
    df = deep.edge_attributes
    if eid not in getattr(df, "index", []):
        df.loc[eid] = {c: None for c in df.columns}
    for k in attrs.keys():
        if k not in df.columns:
            df[k] = None
        df.at[eid, k] = attrs[k]


# ────────────────────────────────
# Core legacy exporter (no manifest)
# ────────────────────────────────

def _export_legacy(graph: BaseGraph, directed: bool = True,
                   skip_hyperedges: bool = True, public_only: bool = False, **kwargs) -> ig.Graph:
    """Export to python-igraph.Graph.

    Note: igraph requires integer vertex indices; we preserve external vertex IDs
    in the vertex attribute `name`. Edge IDs are stored in edge attribute `eid`.
    If `public_only=True`, internal attrs starting with `__` are dropped and the
    authoritative `weight` is exposed as `weight` (not `__weight`).
    """

    # 1) materialize vertex ordering and index map
    vertices = list(graph.V)
    vidx = {v: i for i, v in enumerate(vertices)}

    G = ig.Graph(directed=directed)
    G.add_vertices(len(vertices))
    G.vs["name"] = vertices  # preserve external IDs

    # 2) vertex attributes
    for v in vertices:
        v_attr = _serialize_attrs(graph.get_attr_vertex(v), public_only=public_only)
        # assign per-vertex to avoid alignment pitfalls
        for k, val in v_attr.items():
            # ensure attribute vector exists
            if k not in G.vs.attributes():
                G.vs[k] = [None] * G.vcount()
            G.vs[vidx[v]][k] = val

    # 3) edges + attributes
    for eidx, (src, dst) in graph.edges():
        # serialize edge attrs
        e_attr = _serialize_attrs(graph.get_attr_edge(eidx), public_only=public_only)

        # authoritative weight from backend (if available)
        deep = getattr(graph, "deep", None)
        weight = None
        if deep is not None:
            eid = None
            if hasattr(deep, "idx_to_edge"):
                try:
                    eid = deep.idx_to_edge[eidx]
                except Exception:
                    pass
            key = eid if (eid in getattr(deep, "edge_weights", {})) else eidx
            if hasattr(deep, "edge_weights") and key in deep.edge_weights:
                weight = deep.edge_weights[key]

        if public_only:
            if weight is not None:
                e_attr["weight"] = weight
            e_attr.pop("__weight", None)
        else:
            if weight is not None:
                e_attr["__weight"] = weight

        # emit edges
        if len(src) == 1 and len(dst) == 1:
            u = next(iter(src)); v = next(iter(dst))
            G.add_edge(vidx[u], vidx[v])
            e = G.es[-1]
            # stable key as attribute
            e["eid"] = (deep.idx_to_edge[eidx] if (deep is not None and hasattr(deep, "idx_to_edge") and eidx in deep.idx_to_edge)
                         else f"edge_{eidx}")
            for k, val in e_attr.items():
                e[k] = val
        else:
            if skip_hyperedges:
                continue
            for u in src:
                for v in dst:
                    G.add_edge(vidx[u], vidx[v])
                    e = G.es[-1]
                    e["eid"] = (deep.idx_to_edge[eidx] if (deep is not None and hasattr(deep, "idx_to_edge") and eidx in deep.idx_to_edge)
                                 else f"edge_{eidx}")
                    for k, val in e_attr.items():
                        e[k] = val

    return G


# ────────────────────────────────
# Manifest-aware exporter
# ────────────────────────────────

def to_igraph(adapter_graph: IncidenceAdapter, directed: bool = True,
              hyperedge_mode: str = "skip", layer=None, layers=None,
              public_only: bool = False):
    """
    Export IncidenceAdapter → (igraph.Graph, manifest).
    Manifest preserves hyperedges (with per-endpoint coefficients), layers,
    vertex/edge attrs, and stable edge IDs.
    """

    igG = _export_legacy(
        adapter_graph,
        directed=directed,
        skip_hyperedges=(hyperedge_mode == "skip"),
        public_only=public_only,
    )

    deep = getattr(adapter_graph, "deep", None)

    def _coeff_from_obj(obj) -> float:
        if isinstance(obj, (int, float)):
            return float(obj)
        if hasattr(obj, "items"):
            v = obj.get("__value", 1)
            if hasattr(v, "items"):
                v = v.get("__value", 1)
            try:
                return float(v)
            except Exception:
                return 1.0
        return 1.0

    def _endpoint_coeff_map(edge_attrs: dict, key: str, vertices: set) -> dict:
        out = {}
        side = edge_attrs.get(key, {})
        for v in vertices:
            val = side.get(v, {})
            out[v] = _coeff_from_obj(val)
        return out

    vertex_attrs = {v: _attrs_to_dict(adapter_graph.get_attr_vertex(v)) for v in adapter_graph.V}

    edge_attrs = {}
    for eidx, _ in adapter_graph.edges():
        eid = None
        if deep and hasattr(deep, "idx_to_edge"):
            try:
                eid = deep.idx_to_edge[eidx]
            except Exception:
                pass
        if eid is None:
            eid = f"edge_{eidx}"
        edge_attrs[eid] = _attrs_to_dict(adapter_graph.get_attr_edge(eidx))

    manifest = {}
    if deep:
        manifest_edges = {}
        for eidx, (S, T) in adapter_graph.edges():
            eid = None
            if hasattr(deep, "idx_to_edge"):
                try:
                    eid = deep.idx_to_edge[eidx]
                except Exception:
                    pass
            if eid is None:
                eid = f"edge_{eidx}"

            if len(S) == 1 and len(T) == 1:
                u, v = next(iter(S)), next(iter(T))
                manifest_edges[eid] = (u, v, "regular")
            else:
                eattr = edge_attrs.get(eid, {})
                head_map = _endpoint_coeff_map(eattr, "__source_attr", S)
                tail_map = _endpoint_coeff_map(eattr, "__target_attr", T)
                manifest_edges[eid] = (head_map, tail_map, "hyper")

        manifest = {
            "edges": manifest_edges,
            "weights": deep.edge_weights.copy(),
            "layers": {
                lid: list(deep.get_layer_edges(lid))
                for lid in deep.layers(include_default=True)
            },
            "vertex_attrs": vertex_attrs,
            "edge_attrs": edge_attrs,
        }
    else:
        manifest = {
            "vertex_attrs": vertex_attrs,
            "edge_attrs": edge_attrs,
        }

    return igG, manifest


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

def from_igraph(igG: ig.Graph, manifest: dict) -> IncidenceAdapter:
    H = IncidenceAdapter()

    # 1) vertices — prefer names if present, else index
    names = igG.vs["name"] if "name" in igG.vs.attributes() else list(range(igG.vcount()))
    for v in names:
        try:
            H.add_vertex(v)
        except Exception:
            pass

    # 2) edges — use manifest as the authoritative source (like NX adapter)
    edges_def = manifest.get("edges", {})
    if edges_def:
        for eid, defn in edges_def.items():
            kind = defn[-1]
            if kind == "regular":
                u, v = defn[0], defn[1]
                H.deep.add_edge(u, v, edge_id=eid, edge_directed=True)
            elif kind == "hyper":
                head, tail = tuple(defn[0]), tuple(defn[1])
                H.deep.add_hyperedge(head=head, tail=tail, edge_id=eid, edge_directed=True)
            else:
                try:
                    u, v = defn[0], defn[1]
                    H.deep.add_edge(u, v, edge_id=eid, edge_directed=True)
                except Exception:
                    pass
    else:
        # best-effort fallback if manifest lacks edges: read from igraph
        for e in igG.es:
            src = names[e.source]
            dst = names[e.target]
            eid = e.get("eid", f"edge_{e.index}")
            H.deep.add_edge(src, dst, edge_id=eid, edge_directed=igG.is_directed())

    # 3) weights — ONLY from manifest
    for eid, w in manifest.get("weights", {}).items():
        try:
            H.deep.edge_weights[eid] = w
        except Exception:
            pass

    # 4) layers — ensure present and reattach
    for lid, eids in manifest.get("layers", {}).items():
        if lid not in H.deep.layers(include_default=True):
            H.deep.add_layer(lid)
        for eid in eids:
            try:
                H.deep.add_edge_to_layer(lid, eid)
            except Exception:
                pass

    # 5) reapply vertex/edge attrs from manifest
    for vid, attrs in manifest.get("vertex_attrs", {}).items():
        _safe_set_node_attrs(H.deep, vid, attrs)
    for eid, attrs in manifest.get("edge_attrs", {}).items():
        _safe_set_edge_attrs(H.deep, eid, attrs)

    return H


# ────────────────────────────────
# Compatibility shims
# ────────────────────────────────

def to_backend(graph: BaseGraph, **kwargs):
    """Legacy wrapper: returns only the igraph.Graph, no manifest."""
    return _export_legacy(graph, **kwargs)


class IGraphAdapter:
    def export(self, graph: BaseGraph, **kwargs):
        return _export_legacy(graph, **kwargs)
