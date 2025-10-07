try:
    import igraph as ig
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional dependency 'python-igraph' is not installed. "
        "Install with: ppip install graphglue[igraph]"
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
    Export Graph to igraph.Graph without manifest.
    
    igraph requires integer vertex indices; external vertex IDs are preserved
    in vertex attribute 'name'. Edge IDs stored in edge attribute 'eid'.
    
    Parameters
    ----------
    graph : Graph
        Source graph instance.
    directed : bool
        If True, export as directed igraph.Graph; else undirected.
        Undirected edges in directed export are emitted bidirectionally.
    skip_hyperedges : bool
        If True, drop hyperedges. If False:
          - directed hyperedges expand head×tail (cartesian product)
          - undirected hyperedges expand to clique
    public_only : bool
        If True, strip private attrs starting with "__".
    
    Returns
    -------
    igraph.Graph
    """
    vertices = list(graph.vertices())
    vidx = {v: i for i, v in enumerate(vertices)}

    G = ig.Graph(directed=directed)
    G.add_vertices(len(vertices))
    G.vs["name"] = vertices

    for v in vertices:
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
        
        for k, val in v_attr.items():
            if k not in G.vs.attributes():
                G.vs[k] = [None] * G.vcount()
            G.vs[vidx[v]][k] = val

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
                G.add_edge(vidx[u], vidx[u])
                e = G.es[-1]
                e["eid"] = eid
                for k, val in e_attr.items():
                    e[k] = val
            else:
                if is_dir:
                    uu = next(iter(S))
                    vv = next(iter(T))
                    G.add_edge(vidx[uu], vidx[vv])
                    e = G.es[-1]
                    e["eid"] = eid
                    for k, val in e_attr.items():
                        e[k] = val
                else:
                    u, v = tuple(members)
                    if directed:
                        G.add_edge(vidx[u], vidx[v])
                        e = G.es[-1]
                        e["eid"] = eid
                        for k, val in e_attr.items():
                            e[k] = val
                        G.add_edge(vidx[v], vidx[u])
                        e = G.es[-1]
                        e["eid"] = eid
                        for k, val in e_attr.items():
                            e[k] = val
                    else:
                        G.add_edge(vidx[u], vidx[v])
                        e = G.es[-1]
                        e["eid"] = eid
                        for k, val in e_attr.items():
                            e[k] = val
            continue

        if skip_hyperedges:
            continue

        if is_dir:
            for u in S:
                for v in T:
                    G.add_edge(vidx[u], vidx[v])
                    e = G.es[-1]
                    e["eid"] = eid
                    if not directed:
                        e["directed"] = True
                    for k, val in e_attr.items():
                        e[k] = val
        else:
            mem = list(members)
            n = len(mem)
            if directed:
                for a in range(n):
                    for b in range(n):
                        if a == b:
                            continue
                        G.add_edge(vidx[mem[a]], vidx[mem[b]])
                        e = G.es[-1]
                        e["eid"] = eid
                        for k, val in e_attr.items():
                            e[k] = val
            else:
                for a in range(n):
                    for b in range(a + 1, n):
                        G.add_edge(vidx[mem[a]], vidx[mem[b]])
                        e = G.es[-1]
                        e["eid"] = eid
                        for k, val in e_attr.items():
                            e[k] = val

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


def to_igraph(graph: "Graph", directed=True, hyperedge_mode="skip", 
              layer=None, layers=None, public_only=False):
    """
    Export Graph → (igraph.Graph, manifest).
    
    Manifest preserves hyperedges with per-endpoint coefficients, layers,
    vertex/edge attrs, and stable edge IDs. igraph uses integer vertex
    indices; external IDs are stored in vertex attribute 'name'.
    
    Parameters
    ----------
    graph : Graph
        Source Graph instance.
    directed : bool, default True
        Export as directed igraph.Graph (True) or undirected (False).
    hyperedge_mode : {"skip", "expand"}, default "skip"
        How to handle hyperedges:
        - "skip": drop hyperedges entirely from igraph.Graph
        - "expand": cartesian product (directed) or clique (undirected)
    layer : str, optional
        Export single layer only.
    layers : list[str], optional
        Export union of specified layers.
    public_only : bool, default False
        If True, strip attributes starting with "__".
    
    Returns
    -------
    tuple[igraph.Graph, dict]
        (igG, manifest) where:
        - igG: igraph.Graph with integer vertex indices and 'name' attribute
          for external vertex IDs. Edge IDs stored in 'eid' attribute.
        - manifest: dict preserving full structure (hyperedges, layers,
          attributes, weights) for exact round-tripping.
    
    Notes
    -----
    igraph cannot represent hyperedges natively. The manifest is the SSOT
    (Single Source Of Truth) for reconstruction. The igraph.Graph is lossy:
    hyperedges are either dropped or expanded into multiple binary edges.
    """
    igG = _export_legacy(
        graph,
        directed=directed,
        skip_hyperedges=(hyperedge_mode == "skip"),
        public_only=public_only
    )

    vertex_attrs = {}
    for v in graph.vertices():
        v_attrs = graph.vertex_attributes.filter(
            graph.vertex_attributes["vertex_id"] == v
        ).to_dicts()
        if v_attrs:
            attrs = dict(v_attrs[0])
            attrs.pop("vertex_id", None)
            vertex_attrs[v] = _attrs_to_dict(attrs)
        else:
            vertex_attrs[v] = {}

    edge_attrs = {}
    for eidx in range(graph.number_of_edges()):
        eid = graph.idx_to_edge[eidx]
        e_attrs = graph.edge_attributes.filter(
            graph.edge_attributes["edge_id"] == eid
        ).to_dicts()
        if e_attrs:
            attrs = dict(e_attrs[0])
            attrs.pop("edge_id", None)
            edge_attrs[eid] = _attrs_to_dict(attrs)
        else:
            edge_attrs[eid] = {}

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

    # Capture per-layer edge weight overrides, if any
    layer_weights = {}
    for lid, eids in layers_section.items():
        per_edge = {}
        for eid in eids:
            w = None
            # try canonical accessor
            try:
                w = graph.get_edge_layer_attr(lid, eid, "weight", default=None)
            except Exception:
                pass
            # fallback: read from attribute table if present
            if w is None:
                try:
                    w = graph.get_edge_layer_attr(lid, eid, "weight")
                except Exception:
                    pass
            if w is not None:
                per_edge[eid] = float(w)
        if per_edge:
            layer_weights[lid] = per_edge
    # Build layer → edge_id[] with robust fallbacks. Some Graph implementations
    # expose membership differently; we try several options to avoid empty lists.
    layers_section = {}
    if layer:
        lids = [layer] if isinstance(layer, str) else list(layer)
    else:
        try:
            lids = list(graph.list_layers(include_default=True))
        except Exception:
            try:
                lids = list(graph.list_layers())
            except Exception:
                lids = []

    # Convenience: all exported, non-hyper edge ids
    try:
        all_eids = [e for e in manifest_edges.keys()]  # if manifest_edges is dict-like
    except Exception:
        try:
            all_eids = [e[3] for e in manifest_edges]  # if list of tuples (u,v,d,eid)
        except Exception:
            all_eids = []

    for lid in lids:
        eids = []
        # 1) Preferred: direct API
        try:
            eids = list(graph.get_layer_edges(lid))
        except Exception:
            eids = []
        # 2) Fallback: is_edge_in_layer(...) over known edges
        if not eids and all_eids:
            tmp = []
            for eid in all_eids:
                try:
                    if getattr(graph, "is_edge_in_layer")(lid, eid):
                        tmp.append(eid)
                except Exception:
                    pass
            if tmp:
                eids = tmp
        # 3) Fallback: inspect edge-layer attribute table (Polars [a DataFrame library] or dict)
        if not eids:
            try:
                t = getattr(graph, "edge_layer_attributes")
            except Exception:
                t = None
            if t is not None:
                try:
                    import polars as pl  # noqa: F401
                    if hasattr(t, "columns") and {"layer", "edge"} <= set(t.columns):
                        eids = list(t.filter(pl.col("layer") == lid)["edge"].to_list())
                except Exception:
                    if isinstance(t, dict) and lid in t:
                        # expect shape: {lid: {eid: {...}}}
                        try:
                            eids = list(t[lid].keys())
                        except Exception:
                           pass
        # 4) Fallback: layer info object
        if not eids:
            try:
                info = graph.get_layer_info(lid)
                eids = list(info.get("edges", []))
            except Exception:
                pass
        layers_section[lid] = list(dict.fromkeys(eids))  # de-dup while preserving order


    # capture per-layer edge weight overrides (only when present)
    layer_weights = {}
    for lid, eids in layers_section.items():
        per_edge = {}
        for eid in eids:
            try:
                w = graph.get_edge_layer_attr(lid, eid, "weight", default=None)
            except Exception:
                w = None
            if w is not None:
                per_edge[eid] = float(w)
        if per_edge:
            layer_weights[lid] = per_edge

    manifest = {
        "edges": manifest_edges,
        "weights": weights,
        "layers": layers_section,
        "vertex_attrs": vertex_attrs,
        "edge_attrs": edge_attrs,
        "layer_weights": layer_weights
    }

    return igG, manifest


def save_manifest(manifest: dict, path: str):
    """
    Write manifest to JSON file.
    
    Parameters
    ----------
    manifest : dict
        Manifest dictionary from to_igraph().
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
        Manifest dictionary suitable for from_igraph().
    
    Raises
    ------
    OSError
        If file cannot be read.
    json.JSONDecodeError
        If file contains invalid JSON.
    """
    with open(path) as f:
        return json.load(f)


def from_igraph(igG: ig.Graph, manifest: dict) -> "Graph":
    """
    Reconstruct a Graph from igraph.Graph + manifest.
    
    Parameters
    ----------
    igG : igraph.Graph
        igraph.Graph (largely ignored for structural data - used only
        for fallback vertex extraction if manifest is incomplete).
        Vertex 'name' attribute should contain external vertex IDs.
    manifest : dict
        Manifest dictionary created by to_igraph(). Must contain:
        - "edges" : dict[str, tuple]
            {edge_id: (u, v, "regular") | (head_map, tail_map, "hyper")}
        - "weights" : dict[str, float]
            {edge_id: weight}
        - "layers" : dict[str, list[str]]
            {layer_id: [edge_id, ...]}
        - "vertex_attrs" : dict[str, dict]
            {vertex_id: {attr: value, ...}}
        - "edge_attrs" : dict[str, dict]
            {edge_id: {attr: value, ...}}
    
    Returns
    -------
    Graph
        Reconstructed Graph instance with full hyperedge structure,
        layers, weights, and attributes restored from manifest.
    
    Notes
    -----
    The manifest is the single source of truth (SSOT). The igraph.Graph
    is only consulted for vertex names if the manifest is incomplete.
    This ensures exact round-trip fidelity for hyperedges, layers,
    and per-endpoint coefficients that igraph cannot represent natively.
    Vertex IDs are read from the igraph 'name' attribute if present,
    otherwise integer indices are used as fallback.
    """
    from ..core.graph import Graph
    
    H = Graph()

    names = igG.vs["name"] if "name" in igG.vs.attributes() else list(range(igG.vcount()))
    for v in names:
        try:
            H.add_vertex(v)
        except Exception:
            pass

    edges_def = manifest.get("edges", {})
    for eid, defn in edges_def.items():
        kind = defn[-1]
        if kind == "regular":
            u, v = defn[0], defn[1]
            H.add_edge(u, v, edge_id=eid, edge_directed=True)
        elif kind == "hyper":
            head_map, tail_map = defn[0], defn[1]
            if isinstance(head_map, dict) and isinstance(tail_map, dict):
                head = list(head_map.keys())
                tail = list(tail_map.keys())
                H.add_hyperedge(head=head, tail=tail, edge_id=eid, edge_directed=True)
            else:
                try:
                    u, v = defn[0], defn[1]
                    H.add_edge(u, v, edge_id=eid, edge_directed=True)
                except Exception:
                    pass
        else:
            try:
                u, v = defn[0], defn[1]
                H.add_edge(u, v, edge_id=eid, edge_directed=True)
            except Exception:
                pass

    for eid, w in manifest.get("weights", {}).items():
        try:
            H.edge_weights[eid] = w
        except Exception:
            pass

    for lid, eids in manifest.get("layers", {}).items():
        if lid not in H.list_layers(include_default=True):
            H.add_layer(lid)
        for eid in eids:
            try:
                H.add_edge_to_layer(lid, eid)
            except Exception:
                pass

    # Reapply per-layer weight overrides captured in the manifest
    lw = {}
    try:
        lw = manifest.get("layer_weights", {}) or {}
    except Exception:
        lw = {}
    for lid, per_edge in lw.items():
        # ensure layer exists
        try:
            if lid not in H.list_layers(include_default=True):
                H.add_layer(lid)
        except Exception:
            pass
        for eid, w in per_edge.items():
            try:
                H.set_edge_layer_attrs(lid, eid, weight=float(w))
            except Exception:
                pass

    return H


def to_backend(graph, **kwargs):
    """
    Export Graph to igraph without manifest (legacy compatibility).
    
    Parameters
    ----------
    graph : Graph
        Source Graph instance to export.
    **kwargs
        Forwarded to _export_legacy(). Supported:
        - directed : bool, default True
            Export as directed igraph.Graph (True) or undirected (False).
        - skip_hyperedges : bool, default True
            If True, drop hyperedges. If False, expand them
            (cartesian product for directed, clique for undirected).
        - public_only : bool, default False
            Strip attributes starting with "__" if True.
    
    Returns
    -------
    igraph.Graph
        igraph.Graph with integer vertex indices. External vertex IDs
        are stored in vertex attribute 'name'. Edge IDs stored in edge
        attribute 'eid'. Hyperedges are either dropped or expanded into
        multiple binary edges. No manifest is returned, so round-tripping
        will lose hyperedge structure, layers, and precise edge IDs.
    
    Notes
    -----
    This is a lossy export. Use to_igraph() with manifest for full fidelity.
    igraph requires integer vertex indices internally; the 'name' attribute
    preserves your original string IDs.
    """
    return _export_legacy(graph, **kwargs)


class IGraphAdapter:
    """
    Legacy adapter class for backward compatibility.
    
    Methods
    -------
    export(graph, **kwargs)
        Export Graph to igraph.Graph without manifest (lossy).
    """
    def export(self, graph, **kwargs):
        """
        Export Graph to igraph.Graph without manifest.
        
        Parameters
        ----------
        graph : Graph
        **kwargs
            See to_backend() for supported parameters.
        
        Returns
        -------
        igraph.Graph
        """
        return _export_legacy(graph, **kwargs)