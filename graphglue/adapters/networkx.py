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

    weights = dict(graph.edge_weights)

    layers_section = {}
    if layer:
        if graph.has_layer(layer):
            layers_section[layer] = list(graph.get_layer_edges(layer))
    elif layers:
        for lid in layers:
            if graph.has_layer(lid):
                layers_section[lid] = list(graph.get_layer_edges(lid))
    else:
        for lid in graph.list_layers(include_default=True):
            layers_section[lid] = list(graph.get_layer_edges(lid))

    manifest = {
        "edges": manifest_edges,
        "weights": weights,
        "layers": layers_section,
        "vertex_attrs": vertex_attrs,
        "edge_attrs": edge_attrs,
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
        NetworkX graph (ignored for structural data - only used for
        validation/compatibility).
    manifest : dict
        Manifest dictionary created by to_nx(). Must contain:
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
    The manifest is the single source of truth (SSOT). The NetworkX
    graph is ignored for edge definitions - only the manifest is used.
    This ensures exact round-trip fidelity for hyperedges, layers,
    and per-endpoint coefficients that NetworkX cannot represent.
    """
    from ..core.graph import Graph
    
    H = Graph()

    for v in nxG.nodes():
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

    for vid, attrs in manifest.get("vertex_attrs", {}).items():
        if attrs:
            try:
                H.set_vertex_attrs(vid, **attrs)
            except Exception:
                pass
    
    for eid, attrs in manifest.get("edge_attrs", {}).items():
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