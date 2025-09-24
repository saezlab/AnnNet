try:
    import networkx as nx
except ModuleNotFoundError as e:  # only triggered if this module is imported without the extra
    raise ModuleNotFoundError(
        "Optional dependency 'networkx' is not installed. "
        "Install with: pip install graphglue[networkx]"
    ) from e

from typing import Any
from enum import Enum
import json

from ..core._base import BaseGraph, EdgeType, Attr, Attributes
from ..core.adapter import IncidenceAdapter   # backend


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

def _serialize_attrs(attrs, public_only=False) -> dict:
    try:
        attrs = dict(attrs)  # Attributes is mapping-like
    except Exception:
        return {}
    if public_only:
        return {k: v for k, v in attrs.items() if not str(k).startswith("__")}
    return dict(attrs)

from enum import Enum

def _attrs_to_dict(a):
    # Attributes is mapping-like; make it JSON-safe
    try:
        items = dict(a)
    except Exception:
        return {}
    out = {}
    for k, v in items.items():
        if isinstance(v, Enum):
            out[k] = v.name
        elif hasattr(v, "items"):  # nested Attributes/dicts
            out[k] = {kk: (vv.name if isinstance(vv, Enum) else vv) for kk, vv in dict(v).items()}
        else:
            out[k] = v
    return out

def _safe_set_node_attrs(deep, vid, attrs: dict):
    """Set vertex attrs without triggering pandas alignment quirks."""
    # Prefer public API if you have it
    if hasattr(deep, "set_node_attrs"):
        deep.set_node_attrs(vid, **attrs); return
    df = deep.vertex_attributes
    # ensure row exists
    if vid not in getattr(df, "index", []):
        # backend usually has add_node; fall back to creating row
        if hasattr(deep, "add_node"):
            deep.add_node(vid)
    # ensure columns exist as object dtype, and write scalars via .at
    for k in attrs.keys():
        if k not in df.columns:
            df[k] = None
        df.at[vid, k] = attrs[k]

def _safe_set_edge_attrs(deep, eid, attrs: dict):
    """Use public API if present; else upsert directly into edge_attributes."""
    if hasattr(deep, "set_edge_attrs"):
        deep.set_edge_attrs(eid, **attrs); return
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
def _export_legacy(graph: BaseGraph, directed=True, skip_hyperedges=True, public_only=False, **kwargs):
    G = nx.MultiDiGraph() if directed else nx.MultiGraph()

    # Add vertices
    for v in graph.V:
        v_attr = _serialize_attrs(graph.get_attr_vertex(v), public_only=public_only)
        G.add_node(v, **v_attr)

    # Add edges
    for eidx, (src, dst) in graph.edges():
        # 1) serialize edge attrs (drops __-keys if public_only=True)
        e_attr = _serialize_attrs(graph.get_attr_edge(eidx), public_only=public_only)

        # 2) fetch authoritative weight from deep (map index -> eid first if needed)
        weight = None
        deep = getattr(graph, "deep", None)
        if deep is not None:
            # many backends expose a mapping index->edge id
            eid = None
            if hasattr(deep, "idx_to_edge"):
                try:
                    eid = deep.idx_to_edge[eidx]
                except Exception:
                    pass
            # fall back to using the numeric index if your edge_weights uses indices
            key = eid if (eid in getattr(deep, "edge_weights", {})) else eidx
            if hasattr(deep, "edge_weights") and key in deep.edge_weights:
                weight = deep.edge_weights[key]

        # 3) normalize weight into the emitted attrs
        if public_only:
            # keep only user-facing `weight`
            if weight is not None:
                e_attr["weight"] = weight
            # do NOT keep __weight (it’s internal)
            e_attr.pop("__weight", None)
        else:
            # keep internal __weight (if present), but prefer the authoritative one
            if weight is not None:
                e_attr["__weight"] = weight

        # 4) emit edges as before
        if len(src) == 1 and len(dst) == 1:
            u = next(iter(src)); v = next(iter(dst))
            G.add_edge(u, v, key=eidx, **e_attr)
        else:
            if skip_hyperedges:
                continue
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
    Manifest preserves hyperedges (with per-endpoint coefficients), layers,
    vertex/edge attrs, and stable edge IDs.
    """
    nxG = _export_legacy(
        adapter_graph,
        directed=directed,
        skip_hyperedges=(hyperedge_mode == "skip"),
        public_only=public_only
    )

    deep = getattr(adapter_graph, "deep", None)

    # ---- helpers to pull per-endpoint coefficients from edge attrs
    def _coeff_from_obj(obj) -> float:
        # Accepts: 3, {"__value": 3}, {"__value": {"__value": 3}}
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
        """
        Build {vertex: coeff} for SOURCE/TARGET endpoints.
        Falls back to 1.0 if missing.
        """
        out = {}
        side = edge_attrs.get(key, {})
        for v in vertices:
            val = side.get(v, {})
            out[v] = _coeff_from_obj(val)
        return out

    # ---- collect vertex & edge attrs
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

    # ---- build manifest with correct hyperedge structure
    manifest = {}
    if deep:
        # assemble edges with real head/tail for hyperedges
        manifest_edges = {}
        for eidx, (S, T) in adapter_graph.edges():
            # resolve stable eid
            eid = None
            if hasattr(deep, "idx_to_edge"):
                try:
                    eid = deep.idx_to_edge[eidx]
                except Exception:
                    pass
            if eid is None:
                eid = f"edge_{eidx}"

            # figure kind and store properly
            if len(S) == 1 and len(T) == 1:
                u, v = next(iter(S)), next(iter(T))
                manifest_edges[eid] = (u, v, "regular")
            else:
                # hyperedge: include per-endpoint coefficients if present
                eattr = edge_attrs.get(eid, {})
                head_map = _endpoint_coeff_map(eattr, "__source_attr", S)
                tail_map = _endpoint_coeff_map(eattr, "__target_attr", T)
                # store as JSON-serializable dicts (lossless)
                manifest_edges[eid] = (head_map, tail_map, "hyper")

        manifest = {
            "edges": manifest_edges,                                   # structural
            "weights": deep.edge_weights.copy(),                       # global weights
            "layers": {
                lid: list(deep.get_layer_edges(lid))                   # presence per layer
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
def from_nx(nxG, manifest) -> IncidenceAdapter:
    H = IncidenceAdapter()

    # 1) vertices — build nodes; ignore NX node attrs (we’ll reapply from manifest if needed)
    for v in nxG.nodes():
        try:
            H.add_vertex(v)
        except Exception:
            pass

    # 2) edges — use manifest as the single source of truth
    edges_def = manifest.get("edges", {})
    for eid, defn in edges_def.items():
        kind = defn[-1]
        if kind == "regular":
            u, v = defn[0], defn[1]
            H.deep.add_edge(u, v, edge_id=eid, edge_directed=True)
        elif kind == "hyper":
            head, tail = tuple(defn[0]), tuple(defn[1])
            # if you encode undirected hyperedges differently, branch here
            H.deep.add_hyperedge(head=head, tail=tail, edge_id=eid, edge_directed=True)
        else:
            # best-effort fallback
            try:
                u, v = defn[0], defn[1]
                H.deep.add_edge(u, v, edge_id=eid, edge_directed=True)
            except Exception:
                pass

    # 3) weights — ONLY from manifest (ignore NX)
    for eid, w in manifest.get("weights", {}).items():
        try:
            H.deep.edge_weights[eid] = w
        except Exception:
            pass

    # 4) layers — ensure present and reattach using the public API
    for lid, eids in manifest.get("layers", {}).items():
        if lid not in H.deep.layers(include_default=True):
            H.deep.add_layer(lid)
        for eid in eids:
            try:
                H.deep.add_edge_to_layer(lid, eid)
            except Exception:
                pass

    # 5) (optional) reapply vertex/edge attrs from manifest if you export them:
    for vid, attrs in manifest.get("vertex_attrs", {}).items():
        _safe_set_node_attrs(H.deep, vid, attrs)
    for eid, attrs in manifest.get("edge_attrs", {}).items():
        _safe_set_edge_attrs(H.deep, eid, attrs)

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
