try:
    import networkx as nx
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional dependency 'networkx' is not installed. "
        "Install with: pip install graphglue[networkx]"
    ) from e

from typing import Any
from enum import Enum
import json


def _serialize_value(v: Any) -> Any:
    if isinstance(v, Enum):
        return v.name
    if hasattr(v, "items"):
        return dict(v)
    return v


def _attrs_to_dict(attrs_dict: dict) -> dict:
    out = {}
    for k, v in attrs_dict.items():
        if isinstance(v, Enum):
            out[k] = v.name
        elif hasattr(v, "items"):
            out[k] = {kk: (vv.name if isinstance(vv, Enum) else vv) for kk, vv in dict(v).items()}
        else:
            out[k] = v
    return out


def _is_directed_eid(graph: "Graph", eid: str) -> bool:
    kind = graph.edge_kind.get(eid)
    if kind == "hyper":
        return bool(graph.hyperedge_definitions[eid].get("directed", False))
    return bool(graph.edge_directed.get(eid, graph.directed))


def _export_legacy(graph: "Graph", *, directed: bool = True,
                   skip_hyperedges: bool = True, public_only: bool = False):
    """
    Export Graph to NetworkX Multi(Di)Graph without manifest.
    
    Parameters
    ----------
    graph : Graph
        Source graph instance.
    directed : bool
        If True, export as MultiDiGraph; else MultiGraph.
        Undirected edges in directed export are emitted bidirectionally.
    skip_hyperedges : bool
        If True, drop hyperedges. If False:
          - directed hyperedges expand head×tail (cartesian product)
          - undirected hyperedges expand to clique
    public_only : bool
        If True, strip private attrs starting with "__".
    
    Returns
    -------
    networkx.MultiGraph | networkx.MultiDiGraph
    """
    G = nx.MultiDiGraph() if directed else nx.MultiGraph()

    for v in graph.vertices():
        v_attrs = graph.vertex_attributes.filter(
            graph.vertex_attributes["vertex_id"] == v
        ).to_dicts()
        v_attr = v_attrs[0] if v_attrs else {}
        v_attr.pop("vertex_id", None)
        
        if public_only:
            v_attr = {k: _serialize_value(val) for k, val in v_attr.items() 
                     if not str(k).startswith("__")}
        else:
            v_attr = {k: _serialize_value(val) for k, val in v_attr.items()}
        
        G.add_node(v, **v_attr)

    for eidx in range(graph.number_of_edges()):
        eid = graph.idx_to_edge[eidx]
        S, T = graph.get_edge(eidx)
        
        e_attrs = graph.edge_attributes.filter(
            graph.edge_attributes["edge_id"] == eid
        ).to_dicts()
        e_attr = e_attrs[0] if e_attrs else {}
        e_attr.pop("edge_id", None)
        
        if public_only:
            e_attr = {k: _serialize_value(val) for k, val in e_attr.items() 
                     if not str(k).startswith("__")}
        else:
            e_attr = {k: _serialize_value(val) for k, val in e_attr.items()}

        weight = graph.edge_weights.get(eid, 1.0)
        if public_only:
            e_attr["weight"] = weight
        else:
            e_attr["__weight"] = weight

        is_hyper = (graph.edge_kind.get(eid) == "hyper")
        is_dir = _is_directed_eid(graph, eid)
        members = S | T

        if not is_hyper and len(members) <= 2:
            if len(members) == 1:
                u = next(iter(members))
                G.add_edge(u, u, key=eid, **e_attr)
            else:
                if is_dir:
                    uu = next(iter(S))
                    vv = next(iter(T))
                    G.add_edge(uu, vv, key=eid, **e_attr)
                else:
                    u, v = tuple(members)
                    if directed:
                        G.add_edge(u, v, key=eid, **e_attr)
                        G.add_edge(v, u, key=eid, **e_attr)
                    else:
                        G.add_edge(u, v, key=eid, **e_attr)
            continue

        if skip_hyperedges:
            continue

        if is_dir:
            for u in S:
                for v in T:
                    if directed:
                        G.add_edge(u, v, key=eid, **e_attr)
                    else:
                        G.add_edge(u, v, key=eid, directed=True, **e_attr)
        else:
            mem = list(members)
            n = len(mem)
            if directed:
                for a in range(n):
                    for b in range(n):
                        if a == b:
                            continue
                        G.add_edge(mem[a], mem[b], key=eid, **e_attr)
            else:
                for a in range(n):
                    for b in range(a + 1, n):
                        G.add_edge(mem[a], mem[b], key=eid, **e_attr)

    return G


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


def to_nx(graph: "Graph", directed=True, hyperedge_mode="skip",
          layer=None, layers=None, public_only=False):
    """
    Export Graph → (networkx.Graph, manifest).
    Manifest preserves hyperedges with per-endpoint coefficients, layers,
    vertex/edge attrs, and stable edge IDs.

    Parameters
    ----------
    graph : Graph
    directed : bool
    hyperedge_mode : {"skip", "expand"}
    layer : str, optional
        Export single layer only.
    layers : list[str], optional
        Export union of specified layers.
    public_only : bool

    Returns
    -------
    tuple[networkx.Graph, dict]
        (nxG, manifest)
    """
    nxG = _export_legacy(
        graph,
        directed=directed,
        skip_hyperedges=(hyperedge_mode == "skip"),
        public_only=public_only
    )

    # -------- vertex & edge attributes (respect public_only) ----------
    vertex_attrs = {}
    for v in graph.vertices():
        v_rows = graph.vertex_attributes.filter(
            graph.vertex_attributes["vertex_id"] == v
        ).to_dicts()
        attrs = dict(v_rows[0]) if v_rows else {}
        attrs.pop("vertex_id", None)
        if public_only:
            attrs = {k: v for k, v in attrs.items() if not str(k).startswith("__")}
        vertex_attrs[v] = _attrs_to_dict(attrs)

    edge_attrs = {}
    for eidx in range(graph.number_of_edges()):
        eid = graph.idx_to_edge[eidx]
        e_rows = graph.edge_attributes.filter(
            graph.edge_attributes["edge_id"] == eid
        ).to_dicts()
        attrs = dict(e_rows[0]) if e_rows else {}
        attrs.pop("edge_id", None)
        if public_only:
            attrs = {k: v for k, v in attrs.items() if not str(k).startswith("__")}
        edge_attrs[eid] = _attrs_to_dict(attrs)

    # -------- edge topology snapshot (regular vs hyper) ----------
    manifest_edges = {}
    for eidx in range(graph.number_of_edges()):
        S, T = graph.get_edge(eidx)
        eid = graph.idx_to_edge[eidx]
        is_hyper = (graph.edge_kind.get(eid) == "hyper")

        if not is_hyper:
            members = S | T
            if len(members) == 1:
                u = next(iter(members))
                manifest_edges[eid] = (u, u, "regular")
            elif len(members) == 2:
                u, v = sorted(members)
                manifest_edges[eid] = (u, v, "regular")
            else:
                eattr = edge_attrs.get(eid, {})
                head_map = _endpoint_coeff_map(eattr, "__source_attr", S)
                tail_map = _endpoint_coeff_map(eattr, "__target_attr", T)
                manifest_edges[eid] = (head_map, tail_map, "hyper")
        else:
            eattr = edge_attrs.get(eid, {})
            head_map = _endpoint_coeff_map(eattr, "__source_attr", S)
            tail_map = _endpoint_coeff_map(eattr, "__target_attr", T)
            manifest_edges[eid] = (head_map, tail_map, "hyper")

    # Baseline (global) edge weights
    try:
        weights = {eid: float(w) for eid, w in getattr(graph, "edge_weights", {}).items()}
    except Exception:
        weights = {}

    # -------- robust LAYER discovery + per-layer weights ----------
    def _rows_from_table(t):
        """Return list[dict] from common table backends: Polars, pandas, Arrow, duckdb, list-of-dicts, dict-of-lists."""
        if t is None:
            return []
        # Polars-like
        if hasattr(t, "to_dicts"):
            try:
                return list(t.to_dicts())
            except Exception:
                pass
        # pandas DataFrame
        if hasattr(t, "to_dict"):
            try:
                recs = t.to_dict(orient="records")
                if isinstance(recs, list):
                    return recs
            except Exception:
                pass
        # pyarrow Table
        if hasattr(t, "to_pylist"):
            try:
                return list(t.to_pylist())
            except Exception:
                pass
        # duckdb Relation
        if hasattr(t, "fetchall") and hasattr(t, "columns"):
            try:
                cols = list(t.columns)
                return [dict(zip(cols, row)) for row in t.fetchall()]
            except Exception:
                pass
        # dict-of-lists
        if isinstance(t, dict):
            keys = list(t.keys())
            if keys and isinstance(t[keys[0]], list):
                n = len(t[keys[0]])
                return [{k: t[k][i] for k in keys} for i in range(n)]
        # list-of-dicts
        if isinstance(t, list) and t and isinstance(t[0], dict):
            return list(t)
        return []

    # All exported edge ids (for probes)
    all_eids = list(manifest_edges.keys())

    # 1) candidate layer IDs from every source
    lids = set()
    try:
        lids.update(list(graph.list_layers(include_default=True)))
    except Exception:
        try:
            lids.update(list(graph.list_layers()))
        except Exception:
            pass

    t = getattr(graph, "edge_layer_attributes", None)
    if isinstance(t, dict):
        lids.update(t.keys())
    for r in _rows_from_table(t):
        lid = r.get("layer") or r.get("layer_id") or r.get("lid")
        if lid is not None:
            lids.add(lid)

    le = getattr(graph, "layer_edges", None)  # {layer: [eid,...]}
    if isinstance(le, dict):
        lids.update(le.keys())

    etl = getattr(graph, "edge_to_layers", None)  # {eid: [layer,...]}
    if isinstance(etl, dict):
        for arr in etl.values():
            for lid in (arr or []):
                lids.add(lid)

    # 2) build layer -> edges from all sources
    layers_section = {lid: [] for lid in lids}

    # native API (best signal)
    for lid in list(lids):
        try:
            eids = list(graph.get_layer_edges(lid))
        except Exception:
            eids = []
        if eids:
            seen = set(layers_section[lid])
            for e in eids:
                if e not in seen:
                    layers_section[lid].append(e); seen.add(e)

    # table-backed
    if isinstance(t, dict):
        for lid, mapping in t.items():
            if isinstance(mapping, dict):
                arr = layers_section.setdefault(lid, [])
                seen = set(arr)
                for eid in list(mapping.keys()):
                    if eid not in seen:
                        arr.append(eid); seen.add(eid)
    for r in _rows_from_table(t):
        lid = r.get("layer") or r.get("layer_id") or r.get("lid")
        if lid is None:
            continue
        eid = r.get("edge_id", r.get("edge"))
        if eid is not None:
            arr = layers_section.setdefault(lid, [])
            if eid not in arr:
                arr.append(eid)

    # internal maps
    if isinstance(le, dict):
        for lid, eids in le.items():
            arr = layers_section.setdefault(lid, [])
            seen = set(arr)
            for eid in list(eids):
                if eid not in seen:
                    arr.append(eid); seen.add(eid)
    if isinstance(etl, dict):
        for eid, arr_lids in etl.items():
            for lid in (arr_lids or []):
                arr = layers_section.setdefault(lid, [])
                if eid not in arr:
                    arr.append(eid)

    # 3) per-edge probe + per-layer weights (also infers membership)
    layer_weights = {}
    candidate_lids = set(layers_section.keys()) or lids
    if hasattr(graph, "get_edge_layer_attr"):
        for lid in candidate_lids:
            arr = layers_section.setdefault(lid, [])
            seen = set(arr)
            for eid in all_eids:
                w = None
                try:
                    w = graph.get_edge_layer_attr(lid, eid, "weight", default=None)
                except Exception:
                    try:
                        w = graph.get_edge_layer_attr(lid, eid, "weight")
                    except Exception:
                        w = None
                if w is not None:
                    if eid not in seen:
                        arr.append(eid); seen.add(eid)
                    layer_weights.setdefault(lid, {})[eid] = float(w)

    # Drop empties
    layers_section = {lid: eids for lid, eids in layers_section.items() if eids}

    # Respect layer/layers filters strictly
    if layer is not None or layers is not None:
        req = set()
        if layer is not None:
            req.update([layer] if isinstance(layer, str) else list(layer))
        if layers is not None:
            req.update(list(layers))
        req_norm = {str(x) for x in req}
        layers_section = {lid: eids for lid, eids in layers_section.items() if str(lid) in req_norm}
        layer_weights  = {lid: m    for lid, m    in layer_weights.items()    if str(lid) in req_norm}

    # -------- manifest ----------
    manifest = {
        "edges": manifest_edges,
        "weights": weights,
        "layers": layers_section,
        "vertex_attrs": vertex_attrs,
        "edge_attrs": edge_attrs,
        "layer_weights": layer_weights,  # always present (may be {})
    }

    return nxG, manifest


def save_manifest(manifest: dict, path: str):
    """
    Write manifest to JSON file.
    
    Parameters
    ----------
    manifest : dict
        Manifest dictionary from to_nx().
    path : str
        Output file path (typically .json extension).
    
    Returns
    -------
    None
    
    Raises
    ------
    OSError
        If file cannot be written.
    """
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def load_manifest(path: str) -> dict:
    """
    Load manifest from JSON file.
    
    Parameters
    ----------
    path : str
        Path to manifest JSON file created by save_manifest().
    
    Returns
    -------
    dict
        Manifest dictionary suitable for from_nx().
    
    Raises
    ------
    OSError
        If file cannot be read.
    json.JSONDecodeError
        If file contains invalid JSON.
    """
    with open(path) as f:
        return json.load(f)


def from_nx(nxG, manifest) -> "Graph":
    """
    Reconstruct a Graph from NetworkX graph + manifest.

    Parameters
    ----------
    nxG : networkx.Graph | networkx.MultiGraph | networkx.DiGraph | networkx.MultiDiGraph
        NetworkX graph (not authoritative for structure; manifest is the SSOT).
    manifest : dict
        Dictionary produced by to_nx(). Expected keys:
          - "edges": {edge_id: (u, v, "regular") | (head_map, tail_map, "hyper")}
          - "weights": {edge_id: float}
          - "layers": {layer_id: [edge_id, ...]}
          - "vertex_attrs": {vertex_id: {attr: value}}
          - "edge_attrs": {edge_id: {attr: value}}
          - "layer_weights": {layer_id: {edge_id: float}}  (optional)

    Returns
    -------
    Graph
        Fully reconstructed Graph with hyperedges, layers, weights, and attributes.
    """
    from ..core.graph import Graph

    H = Graph()

    # --- vertices (best-effort from nxG nodes; edges added below will ensure presence too)
    try:
        for v in nxG.nodes():
            try:
                H.add_vertex(v)
            except Exception:
                pass
    except Exception:
        pass

    # --- edges (use manifest, not nxG — manifest is the SSOT)
    edges_def = manifest.get("edges", {})
    for eid, defn in edges_def.items():
        kind = defn[-1]

        if kind == "regular":
            u, v = defn[0], defn[1]
            # ensure vertices exist
            try:
                H.add_vertex(u)
            except Exception:
                pass
            try:
                H.add_vertex(v)
            except Exception:
                pass
            # add edge
            try:
                H.add_edge(u, v, edge_id=eid, edge_directed=True)
            except Exception:
                pass

        elif kind == "hyper":
            head_map, tail_map = defn[0], defn[1]
            if isinstance(head_map, dict) and isinstance(tail_map, dict):
                head = list(head_map.keys())
                tail = list(tail_map.keys())
                # ensure vertices exist
                for u in head:
                    try:
                        H.add_vertex(u)
                    except Exception:
                        pass
                for v in tail:
                    try:
                        H.add_vertex(v)
                    except Exception:
                        pass
                # add hyperedge
                try:
                    H.add_hyperedge(head=head, tail=tail, edge_id=eid, edge_directed=True)
                except Exception:
                    # last-resort: degrade to binary if backend lacks hyperedge support
                    if len(head) == 1 and len(tail) == 1:
                        try:
                            H.add_edge(head[0], tail[0], edge_id=eid, edge_directed=True)
                        except Exception:
                            pass
                # restore endpoint coefficients if possible
                for u, coeff in (head_map or {}).items():
                    try:
                        H.set_edge_endpoint_attr(eid, u, "__source_attr", {"__value": float(coeff)})
                    except Exception:
                        pass
                for v, coeff in (tail_map or {}).items():
                    try:
                        H.set_edge_endpoint_attr(eid, v, "__target_attr", {"__value": float(coeff)})
                    except Exception:
                        pass
            else:
                # malformed hyper spec → try to treat as regular
                try:
                    u, v = defn[0], defn[1]
                    H.add_edge(u, v, edge_id=eid, edge_directed=True)
                except Exception:
                    pass
        else:
            # unknown kind → attempt regular edge
            try:
                u, v = defn[0], defn[1]
                H.add_edge(u, v, edge_id=eid, edge_directed=True)
            except Exception:
                pass

    # --- global (baseline) edge weights
    for eid, w in (manifest.get("weights", {}) or {}).items():
        try:
            H.edge_weights[eid] = float(w)
        except Exception:
            pass

    # --- layer memberships (from manifest["layers"])
    for lid, eids in (manifest.get("layers", {}) or {}).items():
        try:
            existing_layers = set(H.list_layers(include_default=True))
        except Exception:
            existing_layers = set()
        if lid not in existing_layers:
            try:
                H.add_layer(lid)
            except Exception:
                pass
        for eid in eids or []:
            try:
                H.add_edge_to_layer(lid, eid)
            except Exception:
                pass

    # --- per-layer weight overrides (from manifest["layer_weights"])
    lw = manifest.get("layer_weights", {})
    if isinstance(lw, dict):
        for lid, per_edge in (lw or {}).items():
            # ensure the layer exists
            try:
                existing_layers = set(H.list_layers(include_default=True))
            except Exception:
                existing_layers = set()
            if lid not in existing_layers:
                try:
                    H.add_layer(lid)
                except Exception:
                    pass
            for eid, w in (per_edge or {}).items():
                # ensure membership before setting attrs
                try:
                    H.add_edge_to_layer(lid, eid)
                except Exception:
                    pass
                # primary API: bulk setter
                try:
                    H.set_edge_layer_attrs(lid, eid, weight=float(w))
                except Exception:
                    # fallback API: singular setter
                    try:
                        H.set_edge_layer_attr(lid, eid, "weight", float(w))
                    except Exception:
                        pass

    # --- restore vertex/edge attributes (public-only filtering was already applied on export)
    for vid, attrs in (manifest.get("vertex_attrs", {}) or {}).items():
        if attrs:
            try:
                H.set_vertex_attrs(vid, **attrs)
            except Exception:
                pass

    for eid, attrs in (manifest.get("edge_attrs", {}) or {}).items():
        if attrs:
            try:
                H.set_edge_attrs(eid, **attrs)
            except Exception:
                pass

    return H


def to_backend(graph, **kwargs):
    """
    Export Graph to NetworkX without manifest (legacy compatibility).
    
    Parameters
    ----------
    graph : Graph
        Source Graph instance to export.
    **kwargs
        Forwarded to _export_legacy(). Supported:
        - directed : bool, default True
            Export as MultiDiGraph (True) or MultiGraph (False).
        - skip_hyperedges : bool, default True
            If True, drop hyperedges. If False, expand them
            (cartesian product for directed, clique for undirected).
        - public_only : bool, default False
            Strip attributes starting with "__" if True.
    
    Returns
    -------
    networkx.MultiGraph | networkx.MultiDiGraph
        NetworkX graph containing binary edges only. Hyperedges are
        either dropped or expanded. No manifest is returned, so
        round-tripping will lose hyperedge structure, layers, and
        precise edge IDs.
    
    Notes
    -----
    This is a lossy export. Use to_nx() with manifest for full fidelity.
    """
    return _export_legacy(graph, **kwargs)


class NetworkXAdapter:
    def export(self, graph, **kwargs):
        return _export_legacy(graph, **kwargs)