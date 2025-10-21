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
from ._utils import _is_directed_eid, _endpoint_coeff_map


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


def to_nx(graph: "Graph", directed=True, hyperedge_mode="skip",
          layer=None, layers=None, public_only=False,
          reify_prefix="he::"):
    """
    Export Graph → (networkx.Graph, manifest).
    Manifest preserves hyperedges with per-endpoint coefficients, layers,
    vertex/edge attrs, and stable edge IDs.

    Parameters
    ----------
    graph : Graph
    directed : bool
    hyperedge_mode : {"skip", "expand", "reify"}
    layer : str, optional
        Export single layer only (affects which hyperedges are reified).
    layers : list[str], optional
        Export union of specified layers (affects which hyperedges are reified).
    public_only : bool

    Returns
    -------
    tuple[networkx.Graph, dict]
        (nxG, manifest)
    """
    # ----------------- helpers -----------------
    def _public(d):
        if not d: return {}
        return {k: v for k, v in d.items() if not str(k).startswith("__")}

    def _edge_payload(eid):
        data = _public(edge_attrs.get(eid, {}))
        w = weights_map.get(eid)
        if w is not None:
            # keep the usual 'weight' for NX algorithms
            data["weight"] = float(w)
        # stable ID for MultiGraph keys is already EID; keeping attribute too helps debugging
        data.setdefault("eid", eid)
        return data

    # Figure out which hyperedges should be included if user filters by layer(s)
    requested_lids = set()
    if layer is not None:
        requested_lids.update([layer] if isinstance(layer, str) else list(layer))
    if layers is not None:
        requested_lids.update(list(layers))

    selected_eids = None
    if requested_lids:
        selected_eids = set()
        for lid in requested_lids:
            try:
                for eid in graph.get_layer_edges(lid):
                    selected_eids.add(eid)
            except Exception:
                pass

    # ----------------- base NX graph (binary edges only) -----------------
    # For "reify", we want to add membership edges ourselves, so start with hyperedges skipped.
    nxG = _export_legacy(
        graph,
        directed=directed,
        skip_hyperedges=(hyperedge_mode in ("skip", "reify")),
        public_only=public_only,
    )

    # ----------------- vertex & edge attributes for manifest -----------------
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

    # ----------------- edge topology snapshot (regular vs hyper) -----------------
    manifest_edges = {}
    for eidx in range(graph.number_of_edges()):
        S, T = graph.get_edge(eidx)  # endpoint sets
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
        weights_map = {eid: float(w) for eid, w in getattr(graph, "edge_weights", {}).items()}
    except Exception:
        weights_map = {}

    # ----------------- robust LAYER discovery + per-layer weights -----------------
    def _rows_from_table(t):
        if t is None:
            return []
        if hasattr(t, "to_dicts"):
            try: return list(t.to_dicts())
            except Exception: pass
        if hasattr(t, "to_dict"):
            try:
                recs = t.to_dict(orient="records")
                if isinstance(recs, list): return recs
            except Exception: pass
        if hasattr(t, "to_pylist"):
            try: return list(t.to_pylist())
            except Exception: pass
        if hasattr(t, "fetchall") and hasattr(t, "columns"):
            try:
                cols = list(t.columns)
                return [dict(zip(cols, row)) for row in t.fetchall()]
            except Exception: pass
        if isinstance(t, dict):
            keys = list(t.keys())
            if keys and isinstance(t[keys[0]], list):
                n = len(t[keys[0]])
                return [{k: t[k][i] for k in keys} for i in range(n)]
        if isinstance(t, list) and t and isinstance(t[0], dict):
            return list(t)
        return []

    all_eids = list(manifest_edges.keys())

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

    le = getattr(graph, "layer_edges", None)
    if isinstance(le, dict):
        lids.update(le.keys())

    etl = getattr(graph, "edge_to_layers", None)
    if isinstance(etl, dict):
        for arr in etl.values():
            for lid in (arr or []):
                lids.add(lid)

    layers_section = {lid: [] for lid in lids}

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

    #  per-edge probe for layer weights
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

    # Respect layer/layers filters for the manifest sections
    if requested_lids:
        req_norm = {str(x) for x in requested_lids}
        layers_section = {lid: eids for lid, eids in layers_section.items() if str(lid) in req_norm}
        layer_weights  = {lid: m    for lid, m    in layer_weights.items()    if str(lid) in req_norm}

    # ----------------- REIFY: add HE nodes + membership edges to nxG -----------------
    if hyperedge_mode == "reify":
        import networkx as nx
        # Build fast lookup of allowed hyperedge IDs if filtering by layers
        allowed = None
        if requested_lids:
            allowed = set()
            for lid, eids in layers_section.items():
                for eid in eids:
                    allowed.add(eid)

        # choose container semantics
        is_multi_di = isinstance(nxG, nx.MultiDiGraph)

        for eid, spec in manifest_edges.items():
            if spec[-1] != "hyper":
                continue
            if allowed is not None and eid not in allowed:
                continue

            head_map, tail_map = spec[0], spec[1]
            he_id = f"{reify_prefix}{eid}"
            # add HE node with public attrs
            he_attrs = _public(edge_attrs.get(eid, {}))
            he_attrs.update({
                "is_hyperedge": True,
                "eid": eid,
                "directed": bool(_is_directed_eid(graph, eid)),
                "hyper_weight": float(weights_map.get(eid, 1.0)),
            })
            if he_id not in nxG:
                nxG.add_node(he_id, **he_attrs)

            if _is_directed_eid(graph, eid):
                # tail -> HE
                for u, coeff in (tail_map or {}).items():
                    nxG.add_edge(u, he_id, key=f"m::{eid}::{u}::tail",
                                 role="tail", coeff=float(coeff), membership_of=eid)
                # HE -> head
                for v, coeff in (head_map or {}).items():
                    nxG.add_edge(he_id, v, key=f"m::{eid}::{v}::head",
                                 role="head", coeff=float(coeff), membership_of=eid)
            else:
                members = {}
                members.update(tail_map or {}); members.update(head_map or {})
                if is_multi_di:
                    # add both directions to simulate undirected membership
                    for u, coeff in members.items():
                        base = f"m::{eid}::{u}::m"
                        nxG.add_edge(u, he_id, key=base+"::fwd",
                                     role="member", coeff=float(coeff), membership_of=eid)
                        nxG.add_edge(he_id, u, key=base+"::rev",
                                     role="member", coeff=float(coeff), membership_of=eid)
                else:
                    for u, coeff in members.items():
                        nxG.add_edge(u, he_id, key=f"m::{eid}::{u}::m",
                                     role="member", coeff=float(coeff), membership_of=eid)

    # ----------------- manifest (unchanged) -----------------
    manifest = {
        "edges": manifest_edges,
        "weights": weights_map,
        "layers": layers_section,
        "vertex_attrs": vertex_attrs,
        "edge_attrs": edge_attrs,
        "layer_weights": layer_weights,  # always present (may be {})
        "edge_directed": {eid: bool(_is_directed_eid(graph, eid)) for eid in all_eids},
        "manifest_version": 1,
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

def _nx_collect_reified(nxG,
                        he_node_flag="is_hyperedge",
                        he_id_attr="eid",
                        role_attr="role",
                        coeff_attr="coeff",
                        membership_attr="membership_of"):
    """
    Scan nxG for reified hyperedges and return:
      - hyperdefs: list of (eid, directed:bool, head_map:dict, tail_map:dict, he_node_attrs:dict, he_node_id)
      - membership_edges: set of (u,v,key) tuples to skip when also importing binary edges
    """
    import networkx as nx

    if nxG.is_multigraph():
        edge_iter = nxG.edges(keys=True, data=True)
        def EK(u, v, k): return (u, v, k)
    else:
        edge_iter = ((u, v, None, d) for u, v, d in nxG.edges(data=True))
        def EK(u, v, k): return (u, v, None)

    # find HE nodes
    he_nodes = {n for n, d in nxG.nodes(data=True) if (d or {}).get(he_node_flag, False)}
    if not he_nodes:
        return [], set()

    hyperdefs = []
    membership_edges = set()

    for he in he_nodes:
        nd = dict(nxG.nodes[he])
        eid = nd.get(he_id_attr, f"he::{he}")

        head_map, tail_map = {}, {}
        saw_head = False; saw_tail = False; saw_member = False

        # collect all incident edges around the HE node
        for u, v, key, d in edge_iter:
            if u != he and v != he:
                continue
            membership_edges.add(EK(u, v, key))
            other = v if u == he else u
            role = (d or {}).get(role_attr, None)
            coeff = (d or {}).get(coeff_attr, d.get("__value", 1.0))
            try:
                coeff = float(coeff)
            except Exception:
                coeff = 1.0

            if role == "head":
                head_map[other] = coeff; saw_head = True
            elif role == "tail":
                tail_map[other] = coeff; saw_tail = True
            else:
                # treat as undirected membership
                head_map[other] = coeff
                tail_map[other] = coeff
                saw_member = True

        # decide directedness
        if saw_head or saw_tail:
            directed = True
        else:
            directed = False  # only 'member' edges

        hyperdefs.append((eid, directed, head_map, tail_map, nd, he))

    return hyperdefs, membership_edges

def from_nx(nxG, manifest, *,
            hyperedge="none",
            he_node_flag="is_hyperedge",
            he_id_attr="eid",
            role_attr="role",
            coeff_attr="coeff",
            membership_attr="membership_of",
            reify_prefix="he::") -> "Graph":
    """
    Reconstruct a Graph from NetworkX graph + manifest.

    hyperedge: "none" (default) | "reified"
      When "reified", also detect hyperedge nodes in nxG and rebuild true hyperedges
      (in addition to those specified in the manifest).
    """
    from ..core.graph import Graph
    H = Graph()

    # --- vertices from nxG (best-effort; edges will ensure presence too)
    try:
        for v, d in nxG.nodes(data=True):
            # When importing a reified file, do NOT add hyperedge nodes as real vertices
            if hyperedge == "reified":
                if bool((d or {}).get(he_node_flag, False)):
                    continue
                if isinstance(v, str) and v.startswith(reify_prefix):
                    continue
            try:
                H.add_vertex(v)
            except Exception:
                pass
    except Exception:
        pass

    # --- edges/hyperedges from manifest (SSOT)
    edges_def = manifest.get("edges", {}) or {}
    for eid, defn in edges_def.items():
        kind = defn[-1]
        if kind == "regular":
            u, v = defn[0], defn[1]
            for x in (u, v):
                try: H.add_vertex(x)
                except Exception: pass
            is_dir = bool(manifest.get("edge_directed", {}).get(eid, True))
            try: H.add_edge(u, v, edge_id=eid, edge_directed=is_dir)
            except Exception: pass

        elif kind == "hyper":
            head_map, tail_map = defn[0], defn[1]
            if isinstance(head_map, dict) and isinstance(tail_map, dict):
                head = list(head_map.keys())
                tail = list(tail_map.keys())
                is_dir = bool(manifest.get("edge_directed", {}).get(eid, True))

                try:
                    if is_dir:
                        H.add_hyperedge(head=head, tail=tail, edge_id=eid, edge_directed=True)
                    else:
                        members = list(set(head) | set(tail))
                        H.add_hyperedge(members=members, edge_id=eid, edge_directed=False)
                except Exception:
                    # last-resort: degrade to binary only when truly 1→1
                    if len(head) == 1 and len(tail) == 1:
                        try:
                            H.add_edge(head[0], tail[0], edge_id=eid, edge_directed=True)
                        except Exception:
                            pass
                    # otherwise, DO NOT drop the edge silently
                    # (optional) you can log/stash a note here

                # restore endpoint coefficients (works for both directed & undirected)
                try:
                    existing_src = H.get_edge_attribute(eid, "__source_attr") or {}
                except Exception:
                    existing_src = {}
                try:
                    existing_tgt = H.get_edge_attribute(eid, "__target_attr") or {}
                except Exception:
                    existing_tgt = {}

                src_map = {u: {"__value": float(c)} for u, c in (head_map or {}).items()}
                tgt_map = {v: {"__value": float(c)} for v, c in (tail_map or {}).items()}
                merged_src = {**existing_src, **src_map}
                merged_tgt = {**existing_tgt, **tgt_map}
                H.set_edge_attrs(eid, __source_attr=merged_src, __target_attr=merged_tgt)
        else:
            # unknown -> try (u,v)
            try:
                u, v = defn[0], defn[1]
                is_dir = bool(manifest.get("edge_directed", {}).get(eid, True))
                H.add_edge(u, v, edge_id=eid, edge_directed=is_dir)
            except Exception:
                pass

    # --- baseline weights
    for eid, w in (manifest.get("weights", {}) or {}).items():
        try: H.edge_weights[eid] = float(w)
        except Exception: pass

    # --- layers and per-layer overrides
    for lid, eids in (manifest.get("layers", {}) or {}).items():
        try:
            if lid not in set(H.list_layers(include_default=True)):
                H.add_layer(lid)
        except Exception:
            pass
        for eid in eids or []:
            try: H.add_edge_to_layer(lid, eid)
            except Exception: pass

    for lid, per_edge in (manifest.get("layer_weights", {}) or {}).items():
        try:
            if lid not in set(H.list_layers(include_default=True)):
                H.add_layer(lid)
        except Exception:
            pass
        for eid, w in (per_edge or {}).items():
            try: H.add_edge_to_layer(lid, eid)
            except Exception: pass
            try: H.set_edge_layer_attrs(lid, eid, weight=float(w))
            except Exception:
                try: H.set_edge_layer_attr(lid, eid, "weight", float(w))
                except Exception: pass

    # --- restore vertex/edge attrs
    for vid, attrs in (manifest.get("vertex_attrs", {}) or {}).items():
        if attrs:
            try: H.set_vertex_attrs(vid, **attrs)
            except Exception: pass
    for eid, attrs in (manifest.get("edge_attrs", {}) or {}).items():
        if attrs:
            try: H.set_edge_attrs(eid, **attrs)
            except Exception: pass

    # --- OPTIONAL: also import reified hyperedges present in nxG (not in manifest)
    if hyperedge == "reified":
        hyperdefs, membership_edges = _nx_collect_reified(
            nxG,
            he_node_flag=he_node_flag,
            he_id_attr=he_id_attr,
            role_attr=role_attr,
            coeff_attr=coeff_attr,
            membership_attr=membership_attr,
        )
        existing_eids = set(H.edges()) if hasattr(H, "edges") else set()
        for eid, directed, head_map, tail_map, he_attrs, he_node in hyperdefs:
            if eid in existing_eids:
                continue
            # ensure vertices
            for x in set(head_map) | set(tail_map):
                try: H.add_vertex(x)
                except Exception: pass
            # add hyperedge
            if directed:
                try:
                    H.add_hyperedge(head=list(head_map), tail=list(tail_map), edge_id=eid, edge_directed=True)
                except Exception:
                    pass
                H.set_edge_attrs(eid,
                                 __source_attr={u: {"__value": c} for u, c in head_map.items()},
                                 __target_attr={v: {"__value": c} for v, c in tail_map.items()})
            else:
                members = list(set(head_map) | set(tail_map))
                try:
                    H.add_hyperedge(members=members, edge_id=eid, edge_directed=False)
                except Exception:
                    pass
                # keep symmetric coeffs on both sides for compatibility
                H.set_edge_attrs(eid,
                                 __source_attr={u: {"__value": head_map[u]} for u in members if u in head_map},
                                 __target_attr={v: {"__value": tail_map[v]} for v in members if v in tail_map})
            # copy HE node attrs (minus reification markers)
            clean_attrs = {k: v for k, v in (he_attrs or {}).items() if k not in {he_node_flag, he_id_attr}}
            if clean_attrs:
                try: H.set_edge_attrs(eid, **clean_attrs)
                except Exception: pass

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

def from_nx_only(nxG, *,
                 hyperedge="none",
                 he_node_flag="is_hyperedge",
                 he_id_attr="eid",
                 role_attr="role",
                 coeff_attr="coeff",
                 membership_attr="membership_of",
                 reify_prefix="he::"):
    """
    Best-effort import from a bare NetworkX graph (no manifest).
    hyperedge: "none" (default) | "reified"
      When "reified", detect hyperedge nodes + membership edges and rebuild true hyperedges.
    """
    from ..core.graph import Graph
    import networkx as nx

    H = Graph()

    # 1) Nodes + node attributes (verbatim, but skip HE nodes if reified)
    for v, d in nxG.nodes(data=True):
        if hyperedge == "reified":
            if bool((d or {}).get(he_node_flag, False)):
                continue
            if isinstance(v, str) and str(v).startswith(reify_prefix):
                continue
        try:
            H.add_vertex(v)
        except Exception:
            pass
        if d:
            try:
                H.set_vertex_attrs(v, **dict(d))
            except Exception:
                pass

    # 2) Optionally collect reified hyperedges
    membership_edges = set()
    if hyperedge == "reified":
        hyperdefs, membership_edges = _nx_collect_reified(
            nxG,
            he_node_flag=he_node_flag,
            he_id_attr=he_id_attr,
            role_attr=role_attr,
            coeff_attr=coeff_attr,
            membership_attr=membership_attr,
        )
        for eid, directed, head_map, tail_map, he_attrs, he_node in hyperdefs:
            for x in set(head_map) | set(tail_map):
                try: H.add_vertex(x)
                except Exception: pass
            if directed:
                try:
                    H.add_hyperedge(head=list(head_map), tail=list(tail_map), edge_id=eid, edge_directed=True)
                except Exception: pass
                H.set_edge_attrs(eid,
                                 __source_attr={u: {"__value": c} for u, c in head_map.items()},
                                 __target_attr={v: {"__value": c} for v, c in tail_map.items()})
            else:
                members = list(set(head_map) | set(tail_map))
                try:
                    H.add_hyperedge(members=members, edge_id=eid, edge_directed=False)
                except Exception: pass
                H.set_edge_attrs(eid,
                                 __source_attr={u: {"__value": head_map.get(u, 1.0)} for u in members},
                                 __target_attr={v: {"__value": tail_map.get(v, 1.0)} for v in members})
            # copy HE node attrs (minus markers)
            clean_attrs = {k: v for k, v in (he_attrs or {}).items() if k not in {he_node_flag, he_id_attr}}
            if clean_attrs:
                try: H.set_edge_attrs(eid, **clean_attrs)
                except Exception: pass

    # 3) Binary edges (skip membership edges if we consumed them above)
    is_multi = nxG.is_multigraph()
    is_dir   = nxG.is_directed()
    if is_multi:
        iterator = nxG.edges(keys=True, data=True)
        def EK(u,v,k): return (u,v,k)
    else:
        iterator = ((u, v, None, d) for u, v, d in nxG.edges(data=True))
        def EK(u,v,k): return (u,v,None)

    seen_auto = 0
    for u, v, key, d in iterator:
        if hyperedge == "reified" and EK(u, v, key) in membership_edges:
            continue  # this edge was a membership edge; skip importing as binary

        eid = (d.get("eid") if isinstance(d, dict) else None) or (key if key is not None else None)
        if eid is None:
            seen_auto += 1
            eid = f"nx::e#{seen_auto}"

        e_directed = bool(d.get("directed", is_dir))
        w = d.get("weight", d.get("__weight", 1.0))

        try: H.add_vertex(u); H.add_vertex(v)
        except Exception: pass
        try: H.add_edge(u, v, edge_id=eid, edge_directed=e_directed)
        except Exception:
            H.add_edge(u, v, edge_id=eid, edge_directed=True)

        try: H.edge_weights[eid] = float(w)
        except Exception: pass

        if isinstance(d, dict) and d:
            try: H.set_edge_attrs(eid, **dict(d))
            except Exception: pass

    return H


class NetworkXAdapter:
    def export(self, graph, **kwargs):
        return _export_legacy(graph, **kwargs)