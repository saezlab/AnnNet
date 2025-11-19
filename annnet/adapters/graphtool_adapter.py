"""
Graph-tool adapter for AnnNet Graph.

Provides:
    to_graphtool(G)      -> (gt.Graph, manifest_dict)
    from_graphtool(gtG, manifest=None) -> Graph

graph-tool only gets what it can natively represent:
    - vertices (type 'vertex')
    - simple binary edges with a global directedness + a 'weight' edge property
Everything else (hyperedges, per-edge directedness, multilayer, slices,
all attribute tables, etc.) is preserved in `manifest`.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import polars as pl
try:
    import graph_tool.all as gt
except ImportError:
    gt = None
from ..core.graph import Graph 

# Serialization helpers 

def _serialize_edge_layers(edge_layers: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert edge_layers[eid] (aa or (aa, bb)) into JSON-safe form.

    - intra:  aa -> {"kind": "single", "layers": [list(aa)]}
    - inter/coupling: (aa, bb) -> {"kind": "pair", "layers": [list(aa), list(bb)]}
    """
    out = {}
    for eid, L in edge_layers.items():
        if L is None:
            continue
        # e.g. intra: L == aa (tuple[str,...])
        if isinstance(L, tuple) and (len(L) == 0 or isinstance(L[0], str)):
            out[eid] = {"kind": "single", "layers": [list(L)]}
        # inter/coupling: L == (aa, bb)
        elif (
            isinstance(L, tuple)
            and len(L) == 2
            and isinstance(L[0], tuple)
            and isinstance(L[1], tuple)
        ):
            out[eid] = {"kind": "pair", "layers": [list(L[0]), list(L[1])]}
        else:
            # fallback: just repr it
            out[eid] = {"kind": "raw", "value": repr(L)}
    return out


def _deserialize_edge_layers(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inverse of _serialize_edge_layers.

    Returns eid -> aa or (aa, bb) (tuples).
    """
    out = {}
    for eid, rec in data.items():
        kind = rec.get("kind")
        if kind == "single":
            aa = tuple(rec["layers"][0])
            out[eid] = aa
        elif kind == "pair":
            La = tuple(rec["layers"][0])
            Lb = tuple(rec["layers"][1])
            out[eid] = (La, Lb)
        else:
            # unknown / raw -> ignore or store as is
            # here we just skip, user can handle it manually if needed
            continue
    return out


def _serialize_VM(VM: set[tuple[str, tuple[str, ...]]]) -> list[dict]:
    """
    Serialize V_M = {(u, aa)} to JSON-safe list of dicts.
    """
    return [{"node": u, "layer": list(aa)} for (u, aa) in VM]


def _deserialize_VM(data: list[dict]) -> set[tuple[str, tuple[str, ...]]]:
    """
    Inverse of _serialize_VM.
    """
    return {(rec["node"], tuple(rec["layer"])) for rec in data}


def _serialize_node_layer_attrs(nl_attrs: Dict[tuple[str, tuple[str, ...]], dict]) -> list[dict]:
    """
    (u, aa) -> {attrs}  ->  [{"node": u, "layer": list(aa), "attrs": {...}}, ...]
    """
    out = []
    for (u, aa), attrs in nl_attrs.items():
        out.append(
            {
                "node": u,
                "layer": list(aa),
                "attrs": dict(attrs),
            }
        )
    return out


def _deserialize_node_layer_attrs(data: list[dict]) -> Dict[tuple[str, tuple[str, ...]], dict]:
    """
    Inverse of _serialize_node_layer_attrs.
    """
    out: Dict[tuple[str, tuple[str, ...]], dict] = {}
    for rec in data:
        key = (rec["node"], tuple(rec["layer"]))
        out[key] = dict(rec.get("attrs", {}))
    return out


def _serialize_slices(slices: Dict[str, dict]) -> Dict[str, dict]:
    """
    _slices is {slice_id: {"vertices": set, "edges": set, "attributes": dict}}
    Convert sets to lists for JSON.
    """
    out = {}
    for sid, rec in slices.items():
        out[sid] = {
            "vertices": list(rec.get("vertices", [])),
            "edges": list(rec.get("edges", [])),
            "attributes": dict(rec.get("attributes", {})),
        }
    return out


def _deserialize_slices(data: Dict[str, dict]) -> Dict[str, dict]:
    """
    Inverse of _serialize_slices.
    """
    out = {}
    for sid, rec in data.items():
        out[sid] = {
            "vertices": set(rec.get("vertices", [])),
            "edges": set(rec.get("edges", [])),
            "attributes": dict(rec.get("attributes", {})),
        }
    return out


def _df_to_rows(df: pl.DataFrame) -> list[dict]:
    """
    Convert a Polars DataFrame to list-of-dicts in a stable way.
    """
    if df is None or df.height == 0:
        return []
    return df.to_dicts()


def _rows_to_df(rows: list[dict]) -> pl.DataFrame:
    """
    Build a Polars DataFrame from list-of-dicts. Empty -> empty DF.
    """
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)

def _serialize_layer_tuple_attrs(layer_attrs: dict[tuple[str, ...], dict]) -> list[dict]:
    """
    _layer_attrs: {aa_tuple -> {attr_name: value}}
    -> JSON-safe: [{"layer": list(aa), "attrs": {...}}, ...]
    """
    out = []
    for aa, attrs in layer_attrs.items():
        out.append({"layer": list(aa), "attrs": dict(attrs)})
    return out


def _deserialize_layer_tuple_attrs(data: list[dict]) -> dict[tuple[str, ...], dict]:
    """
    Inverse of _serialize_layer_tuple_attrs.
    """
    out: dict[tuple[str, ...], dict] = {}
    for rec in data:
        aa = tuple(rec["layer"])
        out[aa] = dict(rec.get("attrs", {}))
    return out

# Core adapter: to_graphtool

def to_graphtool(
    G: Graph,
    *,
    vertex_id_property: str = "id",
    edge_id_property: str = "id",
    weight_property: str = "weight",
) -> Tuple["gt.Graph", dict]:
    """
    Convert an AnnNet Graph -> (graph_tool.Graph, manifest).

    graph-tool graph:
      - vertices: only entities with entity_types[u] == "vertex"
      - edges: only binary edges whose endpoints are such vertices
      - vertex property vp[vertex_id_property] = AnnNet vertex id
      - edge property   ep[edge_id_property]   = AnnNet edge id
      - edge property   ep[weight_property]    = edge weight (float)

    manifest:
      - preserves everything graph-tool cannot: hyperedges, slices,
        multilayer, and ALL attribute tables.
    """
    if gt is None:
        raise RuntimeError("graph-tool is not installed; cannot call to_graphtool")

    # 1) graph-tool Graph (directed flag from AnnNet)
    directed = bool(G.directed) if G.directed is not None else True
    gtG = gt.Graph(directed=directed)

    # 2) vertices (only type 'vertex')
    vmap = {}  # annnet_id -> gt.Vertex
    vp_id = gtG.new_vertex_property("string")

    for u, t in G.entity_types.items():
        if t != "vertex":
            continue
        v = gtG.add_vertex()
        vmap[u] = v
        vp_id[v] = str(u)

    gtG.vp[vertex_id_property] = vp_id

    # 3) edges (only binary edges between such vertices)
    ep_id = gtG.new_edge_property("string")
    ep_w = gtG.new_edge_property("double")

    for eid, defn in G.edge_definitions.items():
        try:
            u, v, etype = defn
        except ValueError:
            # weird or malformed definition; skip
            continue

        if u not in vmap or v not in vmap:
            # not a pure vertex-vertex edge; hyperedge/hybrid -> only in manifest
            continue

        e = gtG.add_edge(vmap[u], vmap[v])
        ep_id[e] = str(eid)
        ep_w[e] = float(G.edge_weights.get(eid, 1.0))

    gtG.ep[edge_id_property] = ep_id
    gtG.ep[weight_property] = ep_w

    # 4) attribute tables as rows (DF [DataFrame] -> list[dict])

    vert_rows = _df_to_rows(getattr(G, "vertex_attributes", pl.DataFrame()))
    edge_rows = _df_to_rows(getattr(G, "edge_attributes", pl.DataFrame()))
    slice_rows = _df_to_rows(getattr(G, "slice_attributes", pl.DataFrame()))
    edge_slice_rows = _df_to_rows(getattr(G, "edge_slice_attributes", pl.DataFrame()))
    layer_attr_rows = _df_to_rows(getattr(G, "layer_attributes", pl.DataFrame()))

    # 5) slices internal structure (vertex/edge sets + attributes)
    slices_data = _serialize_slices(getattr(G, "_slices", {}))

    # 6) hyperedges and direction info
    hyperedges = dict(getattr(G, "hyperedge_definitions", {}))
    edge_directed = dict(getattr(G, "edge_directed", {}))
    edge_direction_policy = dict(getattr(G, "edge_direction_policy", {}))

    # 7) multilayer / Kivela metadata
    aspects = list(getattr(G, "aspects", []))
    elem_layers = dict(getattr(G, "elem_layers", {}))
    VM_serialized = _serialize_VM(getattr(G, "_VM", set()))
    edge_kind = dict(getattr(G, "edge_kind", {}))
    edge_layers_ser = _serialize_edge_layers(getattr(G, "edge_layers", {}))
    node_layer_attrs_ser = _serialize_node_layer_attrs(
        getattr(G, "_node_layer_attrs", {})
    )

    # aspect and layer-tuple level attributes (dicts)
    aspect_attrs = dict(getattr(G, "_aspect_attrs", {}))
    layer_tuple_attrs_ser = _serialize_layer_tuple_attrs(
        getattr(G, "_layer_attrs", {})
    )

    # 8) build manifest
    manifest = {
        "version": 1,
        "graph": {
            "directed": directed,
            "attributes": dict(getattr(G, "graph_attributes", {})),
        },
        "vertices": {
            "types": dict(G.entity_types),
            "attributes": vert_rows,
        },
        "edges": {
            "definitions": dict(G.edge_definitions),
            "weights": dict(G.edge_weights),
            "directed": edge_directed,
            "direction_policy": edge_direction_policy,
            "hyperedges": hyperedges,
            "attributes": edge_rows,
            "kivela": {
                "edge_kind": edge_kind,
                "edge_layers": edge_layers_ser,
            },
        },
        "slices": {
            "data": slices_data,
            "slice_attributes": slice_rows,
            "edge_slice_attributes": edge_slice_rows,
        },
        "multilayer": {
            "aspects": aspects,
            "aspect_attrs": aspect_attrs,
            "elem_layers": elem_layers,
            "VM": VM_serialized,
            "edge_kind": edge_kind,  # redundant but convenient
            "edge_layers": edge_layers_ser,
            "node_layer_attrs": node_layer_attrs_ser,
            "layer_tuple_attrs": layer_tuple_attrs_ser,
            "layer_attributes": layer_attr_rows,  # elementary 'aspect_layer' DF
        },
        "tables": {
            "vertex_attributes": vert_rows,
            "edge_attributes": edge_rows,
            "slice_attributes": slice_rows,
            "edge_slice_attributes": edge_slice_rows,
            "layer_attributes": layer_attr_rows,
        },
    }

    return gtG, manifest

# Core adapter: from_graphtool

def from_graphtool(
    gtG: "gt.Graph",
    manifest: Optional[dict] = None,
    *,
    vertex_id_property: str = "id",
    edge_id_property: str = "id",
    weight_property: str = "weight",
) -> Graph:
    """
    Convert graph_tool.Graph (+ optional manifest) back into AnnNet Graph.

    - Vertices: from vertex property `vertex_id_property` if present, else numeric index.
    - Edges:    from edges in gtG; edge_id from edge property `edge_id_property` if present,
                else auto; weight from edge property `weight_property` if present, else 1.0.

    If `manifest` is provided, rehydrates:
      - all attribute tables (vertex/edge/slice/edge_slice/layer),
      - _slices internal structure,
      - hyperedges,
      - edge_directed and edge_direction_policy,
      - multilayer (aspects, elem_layers, VM, aspect attrs, layer-tuple attrs,
        edge_kind, edge_layers, node-layer attrs),
      - graph_attributes.
    """
    if gt is None:
        raise RuntimeError("graph-tool is not installed; cannot call from_graphtool")

    directed = bool(gtG.is_directed())
    G = Graph(directed=directed)

    # 1) vertices
    vp = gtG.vp.get(vertex_id_property, None)
    v_to_id: Dict[Any, str] = {}

    for v in gtG.vertices():
        if vp is not None:
            vid = str(vp[v])
        else:
            vid = str(int(v))  # fallback: numeric id
        G.add_vertex(vid)
        v_to_id[v] = vid

    # 2) edges
    ep_id = gtG.ep.get(edge_id_property, None)
    ep_w = gtG.ep.get(weight_property, None)

    for e in gtG.edges():
        u = v_to_id[e.source()]
        v = v_to_id[e.target()]
        eid = str(ep_id[e]) if ep_id is not None else None
        w = float(ep_w[e]) if ep_w is not None else 1.0
        G.add_edge(u, v, edge_id=eid, weight=w)

    # 3) no manifest -> projected graph only
    if manifest is None:
        return G

    # ----- graph-level attributes -----
    gmeta = manifest.get("graph", {})
    G.graph_attributes = dict(gmeta.get("attributes", {}))

    # ----- vertices -----
    vmeta = manifest.get("vertices", {})
    v_rows = vmeta.get("attributes", [])
    if v_rows:
        G.vertex_attributes = _rows_to_df(v_rows)
    v_types = vmeta.get("types", {})
    if v_types:
        G.entity_types.update(v_types)

    # ----- edges -----
    emeta = manifest.get("edges", {})
    e_rows = emeta.get("attributes", [])
    if e_rows:
        G.edge_attributes = _rows_to_df(e_rows)

    weights = emeta.get("weights", {})
    if weights:
        G.edge_weights.update(weights)

    e_directed = emeta.get("directed", {})
    if e_directed:
        G.edge_directed.update(e_directed)

    e_dir_policy = emeta.get("direction_policy", {})
    if e_dir_policy:
        G.edge_direction_policy.update(e_dir_policy)

    hyperedges = emeta.get("hyperedges", {})
    if hyperedges:
        G.hyperedge_definitions = dict(hyperedges)

    kivela_edge = emeta.get("kivela", {})
    if kivela_edge:
        ek = kivela_edge.get("edge_kind", {})
        el_ser = kivela_edge.get("edge_layers", {})
        if ek:
            G.edge_kind.update(ek)
        if el_ser:
            G.edge_layers.update(_deserialize_edge_layers(el_ser))

    # ----- slices -----
    smeta = manifest.get("slices", {})
    slices_data = smeta.get("data", {})
    if slices_data:
        G._slices.update(_deserialize_slices(slices_data))

    slice_rows = smeta.get("slice_attributes", [])
    if slice_rows:
        G.slice_attributes = _rows_to_df(slice_rows)

    edge_slice_rows = smeta.get("edge_slice_attributes", [])
    if edge_slice_rows:
        G.edge_slice_attributes = _rows_to_df(edge_slice_rows)

    # ----- multilayer / Kivela -----
    mm = manifest.get("multilayer", {})
    aspects = mm.get("aspects", [])
    elem_layers = mm.get("elem_layers", {})

    if aspects:
        G.aspects = list(aspects)
        G.elem_layers = dict(elem_layers or {})
        G._rebuild_all_layers_cache()

    aspect_attrs = mm.get("aspect_attrs", {})
    if aspect_attrs:
        G._aspect_attrs.update(aspect_attrs)

    VM_data = mm.get("VM", [])
    if VM_data:
        G._VM = _deserialize_VM(VM_data)

    # edge_kind / edge_layers again (if present under multilayer)
    ek2 = mm.get("edge_kind", {})
    el2_ser = mm.get("edge_layers", {})
    if ek2:
        G.edge_kind.update(ek2)
    if el2_ser:
        G.edge_layers.update(_deserialize_edge_layers(el2_ser))

    nl_attrs_ser = mm.get("node_layer_attrs", [])
    if nl_attrs_ser:
        G._node_layer_attrs = _deserialize_node_layer_attrs(nl_attrs_ser)

    layer_tuple_attrs_ser = mm.get("layer_tuple_attrs", [])
    if layer_tuple_attrs_ser:
        G._layer_attrs = _deserialize_layer_tuple_attrs(layer_tuple_attrs_ser)

    layer_attr_rows = mm.get("layer_attributes", [])
    if layer_attr_rows:
        G.layer_attributes = _rows_to_df(layer_attr_rows)

    return G
