from __future__ import annotations

import json
from enum import Enum
from typing import TYPE_CHECKING, Any

from ._utils import _endpoint_coeff_map, _is_directed_eid

if TYPE_CHECKING:
    from ..core.graph import Graph

def _collect_layers_and_weights(graph) -> tuple[dict, dict]:
    """Returns:
      layers_section: {layer_id: [edge_id, ...]}
      layer_weights:  {layer_id: {edge_id: weight}}
    Backends supported: Polars-like, .to_dicts()-like, dict.

    """
    layers_section: dict = {}
    layer_weights: dict = {}

    # --- Source A: edge_layer_attributes table (Polars-like)
    t = getattr(graph, "edge_layer_attributes", None)
    if t is not None and hasattr(t, "filter"):
        try:
            # Attempt Polars path without hard dependency

            # Get all rows then group by layer in Python (keeps us backend-agnostic)
            if hasattr(t, "to_dicts"):
                rows = t.to_dicts()
            else:
                # last-ditch: try turning the entire table into a list of dicts
                # many DataFrame-likes support .rows(named=True)
                rows = getattr(t, "rows", lambda named=False: [])(named=True)  # type: ignore
            for r in rows:
                lid = r.get("layer")
                if lid is None:
                    continue
                eid = r.get("edge_id", r.get("edge"))
                if eid is None:
                    continue
                layers_section.setdefault(lid, []).append(eid)
                w = r.get("weight")
                if w is not None:
                    layer_weights.setdefault(lid, {})[eid] = float(w)
        except Exception:
            pass  # fall through to other sources

    # --- Source B: edge_layer_attributes with .to_dicts() but no Polars
    if not layers_section and t is not None and hasattr(t, "to_dicts"):
        try:
            for r in t.to_dicts():
                lid = r.get("layer")
                if lid is None:
                    continue
                eid = r.get("edge_id", r.get("edge"))
                if eid is None:
                    continue
                layers_section.setdefault(lid, []).append(eid)
                w = r.get("weight")
                if w is not None:
                    layer_weights.setdefault(lid, {})[eid] = float(w)
        except Exception:
            pass

    # --- Source C: dict mapping layer -> {edge_id: attrs}
    if not layers_section and isinstance(t, dict):
        for lid, ed in t.items():
            if isinstance(ed, dict):
                eids = list(ed.keys())
                layers_section.setdefault(lid, []).extend(eids)
                # pick weights if present
                for eid, attrs in ed.items():
                    if (
                        isinstance(attrs, dict)
                        and "weight" in attrs
                        and attrs["weight"] is not None
                    ):
                        layer_weights.setdefault(lid, {})[eid] = float(attrs["weight"])

    # --- Fallback D: per-edge scan (if graph exposes edge iteration + get_edge_layers)
    if not layers_section:
        edges_iter = None
        for attr in ("edges", "iter_edges", "edge_ids"):
            if hasattr(graph, attr):
                try:
                    edges_iter = list(getattr(graph, attr)())
                    break
                except Exception:
                    pass
        if edges_iter:
            for eid in edges_iter:
                lids = []
                for probe in ("get_edge_layers", "edge_layers"):
                    if hasattr(graph, probe):
                        try:
                            lids = list(getattr(graph, probe)(eid))
                            break
                        except Exception:
                            pass
                for lid in lids or []:
                    layers_section.setdefault(lid, []).append(eid)

    # --- Collect per-layer weight overrides using canonical accessor
    if hasattr(graph, "get_edge_layer_attr"):
        for lid, eids in list(layers_section.items()):
            for eid in eids:
                w = None
                try:
                    w = graph.get_edge_layer_attr(lid, eid, "weight", default=None)
                except Exception:
                    try:
                        # some implementations don't support default=
                        w = graph.get_edge_layer_attr(lid, eid, "weight")
                    except Exception:
                        w = None
                if w is not None:
                    layer_weights.setdefault(lid, {})[eid] = float(w)

    # Ensure deterministic and non-empty lists in manifest
    for lid, eids in layers_section.items():
        # unique, stable order
        seen = set()
        uniq = []
        for e in eids:
            if e not in seen:
                seen.add(e)
                uniq.append(e)
        layers_section[lid] = uniq

    return layers_section, layer_weights


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


def _export_legacy(
    graph: Graph,
    *,
    directed: bool = True,
    skip_hyperedges: bool = True,
    public_only: bool = False,
):
    """Export Graph to igraph.Graph without manifest.

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
    import igraph as ig

    # Build the vertex universe robustly
    # Start with declared vertices
    base_vertices = set(graph.vertices())

    # Ensure endpoints that appear in edges are also included
    endpoints = set()
    for eidx in range(graph.number_of_edges()):
        eid = graph.idx_to_edge[eidx]
        # graph.get_edge(eidx) returns (S, T) as sets for both binary and hyper encodings
        try:
            S, T = graph.get_edge(eidx)
        except Exception:
            S, T = set(), set()
        endpoints.update(S)
        endpoints.update(T)

    # If we are going to expand hyperedges, include their members/head/tail too
    if not skip_hyperedges:
        for eid, hdef in getattr(graph, "hyperedge_definitions", {}).items():
            if isinstance(hdef, dict):
                if hdef.get("members") is not None:
                    endpoints.update(hdef.get("members", []))
                else:
                    endpoints.update(hdef.get("head", []))
                    endpoints.update(hdef.get("tail", []))

    vertices = list(dict.fromkeys(list(base_vertices) + [v for v in endpoints]))  # stable order
    vidx = {v: i for i, v in enumerate(vertices)}

    # Create igraph graph and set vertex 'name'
    G = ig.Graph(directed=bool(directed))
    G.add_vertices(len(vertices))
    G.vs["name"] = vertices

    # Attach vertex attributes (works for both vertices and edge-entities)
    # Polars-safe extraction; ignore if table missing/empty.
    vtab = getattr(graph, "vertex_attributes", None)
    # Pre-scan to a dict for O(1) lookup
    vattr_map = {}
    try:
        if (
            vtab is not None
            and hasattr(vtab, "to_dicts")
            and vtab.height > 0
            and "vertex_id" in vtab.columns
        ):
            for row in vtab.to_dicts():
                d = dict(row)
                vid = d.pop("vertex_id", None)
                if vid is not None:
                    vattr_map[vid] = d
    except Exception:
        vattr_map = {}

    def _serialize_value(x):
        # reuse your serializer if it exists; otherwise passthrough
        try:
            return globals()["_serialize_value"](x)
        except KeyError:
            return x

    for v in vertices:
        v_attr = dict(vattr_map.get(v, {}))
        if public_only:
            v_attr = {
                k: _serialize_value(val) for k, val in v_attr.items() if not str(k).startswith("__")
            }
        else:
            v_attr = {k: _serialize_value(val) for k, val in v_attr.items()}

        # Ensure attribute columns exist and then write row value
        for k, val in v_attr.items():
            if k not in G.vs.attributes():
                G.vs[k] = [None] * G.vcount()
            G.vs[vidx[v]][k] = val

    # Helper: directedness per edge-id (fallback if helper missing)
    def _is_dir_eid(g, eid):
        try:
            return _is_directed_eid(g, eid)  # use existing helper if present
        except NameError:
            return bool(getattr(g, "edge_directed", {}).get(eid, getattr(g, "directed", True)))

    # Add edges (binary & vertex-edge). Hyperedges: skip or expand
    etab = getattr(graph, "edge_attributes", None)
    eattr_map = {}
    try:
        if (
            etab is not None
            and hasattr(etab, "to_dicts")
            and etab.height > 0
            and "edge_id" in etab.columns
        ):
            for row in etab.to_dicts():
                d = dict(row)
                eid = d.pop("edge_id", None)
                if eid is not None:
                    eattr_map[eid] = d
    except Exception:
        eattr_map = {}

    for eidx in range(graph.number_of_edges()):
        eid = graph.idx_to_edge[eidx]
        S, T = graph.get_edge(eidx)

        e_attr = dict(eattr_map.get(eid, {}))
        if public_only:
            e_attr = {
                k: _serialize_value(val) for k, val in e_attr.items() if not str(k).startswith("__")
            }
        else:
            e_attr = {k: _serialize_value(val) for k, val in e_attr.items()}

        weight = graph.edge_weights.get(eid, 1.0)
        if public_only:
            e_attr["weight"] = weight
        else:
            e_attr["__weight"] = weight

        is_hyper = graph.edge_kind.get(eid) == "hyper"
        is_dir = _is_dir_eid(graph, eid)
        members = S | T

        # Binary / vertex-edge path ----------
        if not is_hyper and len(members) <= 2:
            if len(members) == 1:
                # self-loop
                u = next(iter(members))
                # Guard: endpoint must be indexed (should be by construction)
                if u not in vidx:
                    continue
                G.add_edge(vidx[u], vidx[u])
                e = G.es[-1]
                e["eid"] = eid
                for k, val in e_attr.items():
                    e[k] = val
            else:
                if is_dir:
                    # directed: source in S, target in T
                    uu = next(iter(S))
                    vv = next(iter(T))
                    if uu in vidx and vv in vidx:
                        G.add_edge(vidx[uu], vidx[vv])
                        e = G.es[-1]
                        e["eid"] = eid
                        for k, val in e_attr.items():
                            e[k] = val
                else:
                    # undirected edge; emit bidirectionally if `directed=True`
                    u, v = tuple(members)
                    if (u in vidx) and (v in vidx):
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
            continue  # done with this edge

        # ---------- Hyperedge path ----------
        if skip_hyperedges:
            continue

        if is_dir:
            # expand head × tail
            for u in S:
                for v in T:
                    if u not in vidx or v not in vidx:
                        continue
                    G.add_edge(vidx[u], vidx[v])
                    e = G.es[-1]
                    e["eid"] = eid
                    if not directed:
                        e["directed"] = True  # mark orientation in an undirected export
                    for k, val in e_attr.items():
                        e[k] = val
        else:
            # undirected hyperedge: clique (or bidir clique if directed=True)
            mem = [m for m in members if m in vidx]
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


def to_igraph(
    graph: Graph,
    directed=True,
    hyperedge_mode="skip",
    layer=None,
    layers=None,
    public_only=False,
    reify_prefix="he::",
):
    """Export Graph → (igraph.Graph, manifest).

    hyperedge_mode: {"skip","expand","reify"}
      - "skip": drop HE edges from igG (manifest keeps them)
      - "expand": cartesian product (directed) / clique (undirected)
      - "reify": add a node per HE and membership edges V↔HE carrying roles/coeffs
    """
    # -------------- base igraph build (binary edges only) --------------
    # For "reify" we start with hyperedges skipped, then add them as nodes+membership edges.
    igG = _export_legacy(
        graph,
        directed=directed,
        skip_hyperedges=(hyperedge_mode in ("skip", "reify")),
        public_only=public_only,
    )

    # -------------- collect vertex/edge attrs for manifest --------------
    vertex_attrs = {}
    for v in graph.vertices():
        v_rows = graph.vertex_attributes.filter(
            graph.vertex_attributes["vertex_id"] == v
        ).to_dicts()
        attrs = dict(v_rows[0]) if v_rows else {}
        attrs.pop("vertex_id", None)
        if public_only:
            attrs = {k: val for k, val in attrs.items() if not str(k).startswith("__")}
        vertex_attrs[v] = _attrs_to_dict(attrs)

    edge_attrs = {}
    for eidx in range(graph.number_of_edges()):
        eid = graph.idx_to_edge[eidx]
        e_rows = graph.edge_attributes.filter(graph.edge_attributes["edge_id"] == eid).to_dicts()
        attrs = dict(e_rows[0]) if e_rows else {}
        attrs.pop("edge_id", None)
        if public_only:
            attrs = {k: val for k, val in attrs.items() if not str(k).startswith("__")}
        edge_attrs[eid] = _attrs_to_dict(attrs)

    # -------------- topology snapshot (regular vs hyper) --------------
    manifest_edges = {}
    for eidx in range(graph.number_of_edges()):
        S, T = graph.get_edge(eidx)
        eid = graph.idx_to_edge[eidx]
        is_hyper = graph.edge_kind.get(eid) == "hyper"
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

    # ---------- layers + per-layer weights for manifest ----------
    def _rows_from_table(t):
        if t is None:
            return []
        if hasattr(t, "to_dicts"):
            try:
                return list(t.to_dicts())
            except Exception:
                pass
        return []

    all_eids = list(manifest_edges.keys())

    # discover layers
    lids = set()
    try:
        lids.update(list(graph.list_layers(include_default=True)))
    except Exception:
        try:
            lids.update(list(graph.list_layers()))
        except Exception:
            pass

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
                    layers_section[lid].append(e)
                    seen.add(e)

    t = getattr(graph, "edge_layer_attributes", None)
    if isinstance(t, dict):
        for lid, mapping in t.items():
            if isinstance(mapping, dict):
                arr = layers_section.setdefault(lid, [])
                seen = set(arr)
                for eid in list(mapping.keys()):
                    if eid not in seen:
                        arr.append(eid)
                        seen.add(eid)
    for r in _rows_from_table(t):
        lid = r.get("layer") or r.get("layer_id") or r.get("lid")
        if lid is None:
            continue
        eid = r.get("edge_id", r.get("edge"))
        if eid is not None:
            arr = layers_section.setdefault(lid, [])
            if eid not in arr:
                arr.append(eid)

    etl = getattr(graph, "edge_to_layers", None)
    if isinstance(etl, dict):
        for eid, arr_lids in etl.items():
            for lid in arr_lids or []:
                arr = layers_section.setdefault(lid, [])
                if eid not in arr:
                    arr.append(eid)

    le = getattr(graph, "layer_edges", None)
    if isinstance(le, dict):
        for lid, eids in le.items():
            arr = layers_section.setdefault(lid, [])
            seen = set(arr)
            for eid in list(eids):
                if eid not in seen:
                    arr.append(eid)
                    seen.add(eid)

    # remove empties
    layers_section = {lid: eids for lid, eids in layers_section.items() if eids}

    # filter layers if requested
    requested_lids = set()
    if layer is not None:
        requested_lids.update([layer] if isinstance(layer, str) else list(layer))
    if layers is not None:
        requested_lids.update(list(layers))
    if requested_lids:
        req_norm = {str(x) for x in requested_lids}
        layers_section = {lid: eids for lid, eids in layers_section.items() if str(lid) in req_norm}

    # per-layer weights
    layer_weights = {}
    if hasattr(graph, "get_edge_layer_attr"):
        for lid, eids in layers_section.items():
            for eid in eids:
                w = None
                try:
                    w = graph.get_edge_layer_attr(lid, eid, "weight", default=None)
                except Exception:
                    try:
                        w = graph.get_edge_layer_attr(lid, eid, "weight")
                    except Exception:
                        w = None
                if w is not None:
                    layer_weights.setdefault(lid, {})[eid] = float(w)

    # baseline weights
    try:
        base_weights = {eid: float(w) for eid, w in getattr(graph, "edge_weights", {}).items()}
    except Exception:
        base_weights = {}

    # -------------- REIFY: add HE nodes + membership edges to igG --------------
    if hyperedge_mode == "reify":

        # allowed HE set under layer filter (None => all)
        allowed = None
        if requested_lids:
            allowed = set()
            for lid, eids in layers_section.items():
                for eid in eids:
                    allowed.add(eid)

        # ensure we have a 'name' attribute for vertices
        if "name" not in igG.vs.attributes():
            igG.vs["name"] = list(range(igG.vcount()))
        names = list(igG.vs["name"])
        name_to_idx = {str(n): i for i, n in enumerate(names)}

        def ensure_vertex(name: str) -> int:
            sname = str(name)
            if sname in name_to_idx:
                return name_to_idx[sname]
            # add with name set
            igG.add_vertices([sname])
            idx = igG.vcount() - 1
            # igraph sets 'name' from the list given to add_vertices
            name_to_idx[sname] = idx
            return idx

        new_edges = []
        payloads = []  # dict per new edge

        for eid, spec in manifest_edges.items():
            if spec[-1] != "hyper":
                continue
            if allowed is not None and eid not in allowed:
                continue

            head_map, tail_map = spec[0], spec[1]
            he_name = f"{reify_prefix}{eid}"
            he_idx = ensure_vertex(he_name)

            # copy selected edge attrs to HE node (public only)
            he_attrs = edge_attrs.get(eid, {}) or {}
            if public_only:
                he_attrs = {k: v for k, v in he_attrs.items() if not str(k).startswith("__")}
            igG.vs[he_idx]["is_hyperedge"] = True
            igG.vs[he_idx]["eid"] = eid
            igG.vs[he_idx]["directed"] = bool(_is_directed_eid(graph, eid))
            igG.vs[he_idx]["hyper_weight"] = float(base_weights.get(eid, 1.0))
            # also copy public user attrs
            for k, v in he_attrs.items():
                igG.vs[he_idx][k] = v

            if _is_directed_eid(graph, eid):
                # tail -> HE
                for u, coeff in (tail_map or {}).items():
                    ui = ensure_vertex(u)
                    new_edges.append((ui, he_idx))
                    payloads.append({"role": "tail", "coeff": float(coeff), "membership_of": eid})
                # HE -> head
                for v, coeff in (head_map or {}).items():
                    vi = ensure_vertex(v)
                    new_edges.append((he_idx, vi))
                    payloads.append({"role": "head", "coeff": float(coeff), "membership_of": eid})
            else:
                members = {}
                members.update(tail_map or {})
                members.update(head_map or {})
                if directed:  # directed container: add both directions
                    for u, coeff in members.items():
                        ui = ensure_vertex(u)
                        new_edges.append((ui, he_idx))
                        payloads.append(
                            {"role": "member", "coeff": float(coeff), "membership_of": eid}
                        )
                        new_edges.append((he_idx, ui))
                        payloads.append(
                            {"role": "member", "coeff": float(coeff), "membership_of": eid}
                        )
                else:  # undirected container: one edge is enough
                    for u, coeff in members.items():
                        ui = ensure_vertex(u)
                        new_edges.append((ui, he_idx))
                        payloads.append(
                            {"role": "member", "coeff": float(coeff), "membership_of": eid}
                        )

        if new_edges:
            start = igG.ecount()
            igG.add_edges(new_edges)
            # set attributes for the newly added membership edges
            keys = set().union(*(d.keys() for d in payloads))
            for k in keys:
                igG.es[start:][k] = [d.get(k) for d in payloads]

    # -------------- manifest (unchanged semantics) --------------
    manifest = {
        "edges": manifest_edges,
        "weights": base_weights,
        "layers": layers_section,
        "vertex_attrs": vertex_attrs,
        "edge_attrs": edge_attrs,
        "layer_weights": layer_weights,
        "edge_directed": {eid: bool(_is_directed_eid(graph, eid)) for eid in all_eids},
        "manifest_version": 1,
    }

    return igG, manifest


def save_manifest(manifest: dict, path: str):
    """Write manifest to JSON file.

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
    """Load manifest from JSON file.

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


def _ig_collect_reified(
    igG,
    he_node_flag="is_hyperedge",
    he_id_attr="eid",
    role_attr="role",
    coeff_attr="coeff",
    membership_attr="membership_of",
):
    """Scan igG for reified hyperedges.

    Returns:
      - hyperdefs: list of (eid, directed:bool, head_map:dict, tail_map:dict, he_node_attrs:dict, he_vertex_index)
      - membership_edge_idx: set of edge indices that are membership edges (to skip for binary import)

    """
    import math

    vattrs = set(igG.vs.attributes())
    if he_node_flag not in vattrs:
        return [], set()

    he_idxs = [i for i, flag in enumerate(igG.vs[he_node_flag]) if bool(flag)]
    if not he_idxs:
        return [], set()

    names = igG.vs["name"] if "name" in vattrs else list(range(igG.vcount()))
    membership_edge_idx = set()
    hyperdefs = []

    for hi in he_idxs:
        nd = {k: igG.vs[hi][k] for k in vattrs}  # HE node attrs
        eid = nd.get(he_id_attr, f"he::{names[hi]}")
        head_map, tail_map = {}, {}
        saw_head = saw_tail = saw_member = False

        for eidx in igG.incident(hi, mode="ALL"):
            membership_edge_idx.add(eidx)
            e = igG.es[eidx]
            u, v = e.tuple
            other_i = v if u == hi else u
            other = names[other_i]

            role = e[role_attr] if role_attr in igG.es.attributes() else None
            coeff = e[coeff_attr] if coeff_attr in igG.es.attributes() else (e.get("__value", 1.0))
            try:
                coeff = float(coeff)
                if math.isnan(coeff):
                    coeff = 1.0
            except Exception:
                coeff = 1.0

            if role == "head":
                head_map[other] = coeff
                saw_head = True
            elif role == "tail":
                tail_map[other] = coeff
                saw_tail = True
            else:
                head_map[other] = coeff
                tail_map[other] = coeff
                saw_member = True

        directed = bool(saw_head or saw_tail) and not (saw_member and not (saw_head or saw_tail))
        hyperdefs.append((eid, directed, head_map, tail_map, nd, hi))

    return hyperdefs, membership_edge_idx


def from_igraph(
    igG,
    manifest,
    *,
    hyperedge: str = "none",
    he_node_flag: str = "is_hyperedge",
    he_id_attr: str = "eid",
    reify_prefix: str = "he::",
) -> Graph:
    """Reconstruct a Graph from igraph.Graph + manifest.

    hyperedge: "none" (default) | "reified"
      When "reified", also detect hyperedge nodes in igG and rebuild true hyperedges
      that are NOT present in the manifest.
    """
    from ..core.graph import Graph

    H = Graph()

    # -------- helper: scan reified HE nodes in igG (used only if hyperedge == "reified") --------
    def _ig_collect_reified(ig):
        """Return list of tuples:
          (eid, directed, head_map, tail_map, he_attrs, he_index)
        where head_map/tail_map are {vertex_id: coeff}.
        """
        out = []
        # names for external vertex IDs
        names = ig.vs["name"] if "name" in ig.vs.attributes() else list(range(ig.vcount()))
        name_of = lambda idx: names[idx]

        # identify HE nodes
        he_indices = []
        for i in range(ig.vcount()):
            is_he = False
            try:
                is_he = bool(ig.vs[i][he_node_flag])
            except Exception:
                is_he = False
            if not is_he:
                nm = name_of(i)
                if isinstance(nm, str) and nm.startswith(reify_prefix):
                    is_he = True
            if is_he:
                he_indices.append(i)

        # membership edge attrs
        edge_attr_names = set(ig.es.attributes())
        role_attr = "role"
        coeff_attr = "coeff"
        membership_attr = "membership_of"

        for hi in he_indices:
            # hyperedge id from node attr or fallback from name sans prefix
            he_name = name_of(hi)
            eid = None
            try:
                eid = ig.vs[hi][he_id_attr]
            except Exception:
                eid = None
            if not eid and isinstance(he_name, str) and he_name.startswith(reify_prefix):
                eid = he_name[len(reify_prefix) :]
            if not eid:
                eid = f"he::{hi}"

            head_map, tail_map = {}, {}
            saw_head = saw_tail = False

            # all incident edges to this HE node
            try:
                inc = ig.incident(hi, mode="ALL")
            except Exception:
                inc = []
            for eidx in inc:
                e = ig.es[eidx]
                s, t = e.tuple
                other = t if s == hi else s
                v = name_of(other)

                # role / coeff
                role = None
                if "role" in edge_attr_names:
                    try:
                        role = e["role"]
                    except Exception:
                        role = None
                coeff = 1.0
                if "coeff" in edge_attr_names:
                    try:
                        coeff = float(e["coeff"])
                    except Exception:
                        coeff = 1.0

                if role == "head":
                    head_map[v] = coeff
                    saw_head = True
                elif role == "tail":
                    tail_map[v] = coeff
                    saw_tail = True
                else:
                    # undirected membership → symmetric
                    head_map[v] = head_map.get(v, coeff)
                    tail_map[v] = tail_map.get(v, coeff)

            directed = bool(saw_head or saw_tail)

            # copy HE-node attrs (minus markers)
            he_attrs = {}
            for k in ig.vs.attributes():
                if k in {he_node_flag, he_id_attr}:
                    continue
                try:
                    he_attrs[k] = ig.vs[hi][k]
                except Exception:
                    pass

            out.append((eid, directed, head_map, tail_map, he_attrs, hi))

        return out

    # -------- vertices (from manifest = SSOT) --------
    # Collect all vertex IDs referenced by manifest (attrs + edges)
    vertex_ids = set()

    for vid in (manifest.get("vertex_attrs", {}) or {}).keys():
        vertex_ids.add(vid)

    edges_def = manifest.get("edges", {}) or {}
    for eid, defn in edges_def.items():
        kind = defn[-1]
        if kind == "regular":
            u, v = defn[0], defn[1]
            vertex_ids.add(u)
            vertex_ids.add(v)
        elif kind == "hyper":
            head_map, tail_map = defn[0], defn[1]
            if isinstance(head_map, dict):
                for u in head_map.keys():
                    vertex_ids.add(u)
            if isinstance(tail_map, dict):
                for v in tail_map.keys():
                    vertex_ids.add(v)

    # Add vertices now (no he:: nodes will be included since they aren't in the manifest)
    for v in vertex_ids:
        try:
            H.add_vertex(v)
        except Exception:
            pass

    # -------- edges/hyperedges (from manifest = SSOT) --------
    for eid, defn in edges_def.items():
        kind = defn[-1]
        if kind == "regular":
            u, v = defn[0], defn[1]
            is_dir = bool(manifest.get("edge_directed", {}).get(eid, True))
            try:
                H.add_edge(u, v, edge_id=eid, edge_directed=is_dir)
            except Exception:
                pass

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
                    # degrade only if truly 1→1
                    if len(head) == 1 and len(tail) == 1:
                        try:
                            H.add_edge(head[0], tail[0], edge_id=eid, edge_directed=True)
                        except Exception:
                            pass

                # restore endpoint coeffs
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
                try:
                    H.set_edge_attrs(
                        eid,
                        __source_attr={**existing_src, **src_map},
                        __target_attr={**existing_tgt, **tgt_map},
                    )
                except Exception:
                    pass
        else:
            # unknown → best-effort binary
            try:
                u, v = defn[0], defn[1]
                is_dir = bool(manifest.get("edge_directed", {}).get(eid, True))
                H.add_edge(u, v, edge_id=eid, edge_directed=is_dir)
            except Exception:
                pass

    # -------- baseline weights --------
    for eid, w in (manifest.get("weights", {}) or {}).items():
        try:
            H.edge_weights[eid] = float(w)
        except Exception:
            pass

    # -------- layers + per-layer overrides --------
    for lid, eids in (manifest.get("layers", {}) or {}).items():
        try:
            if lid not in set(H.list_layers(include_default=True)):
                H.add_layer(lid)
        except Exception:
            pass
        for eid in eids or []:
            try:
                H.add_edge_to_layer(lid, eid)
            except Exception:
                pass

    for lid, per_edge in (manifest.get("layer_weights", {}) or {}).items():
        try:
            if lid not in set(H.list_layers(include_default=True)):
                H.add_layer(lid)
        except Exception:
            pass
        for eid, w in (per_edge or {}).items():
            try:
                H.add_edge_to_layer(lid, eid)
            except Exception:
                pass
            try:
                H.set_edge_layer_attrs(lid, eid, weight=float(w))
            except Exception:
                try:
                    H.set_edge_layer_attr(lid, eid, "weight", float(w))
                except Exception:
                    pass

    # -------- restore vertex/edge attrs --------
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

    # -------- OPTIONAL: pull in reified HEs from igG not present in manifest --------
    if hyperedge == "reified":
        try:
            hyperdefs = _ig_collect_reified(igG)
        except Exception:
            hyperdefs = []
        existing_eids = set(edges_def.keys())

        for eid, directed, head_map, tail_map, he_attrs, hi in hyperdefs:
            if eid in existing_eids:
                continue
            # ensure vertices
            for x in set(head_map) | set(tail_map):
                try:
                    H.add_vertex(x)
                except Exception:
                    pass

            if directed:
                try:
                    H.add_hyperedge(
                        head=list(head_map), tail=list(tail_map), edge_id=eid, edge_directed=True
                    )
                except Exception:
                    pass
                try:
                    H.set_edge_attrs(
                        eid,
                        __source_attr={u: {"__value": c} for u, c in head_map.items()},
                        __target_attr={v: {"__value": c} for v, c in tail_map.items()},
                    )
                except Exception:
                    pass
            else:
                members = list(set(head_map) | set(tail_map))
                try:
                    H.add_hyperedge(members=members, edge_id=eid, edge_directed=False)
                except Exception:
                    pass
                try:
                    H.set_edge_attrs(
                        eid,
                        __source_attr={u: {"__value": head_map.get(u, 1.0)} for u in members},
                        __target_attr={v: {"__value": tail_map.get(v, 1.0)} for v in members},
                    )
                except Exception:
                    pass

            # copy HE-node attrs minus markers
            if he_attrs:
                try:
                    H.set_edge_attrs(eid, **he_attrs)
                except Exception:
                    pass

    return H


def to_backend(graph, **kwargs):
    """Export Graph to igraph without manifest (legacy compatibility).

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


def from_ig_only(
    igG,
    *,
    hyperedge="none",
    he_node_flag="is_hyperedge",
    he_id_attr="eid",
    role_attr="role",
    coeff_attr="coeff",
    membership_attr="membership_of",
):
    """Best-effort import from a *plain* igraph.Graph (no manifest).
    Preserves all vertex/edge attributes.
    hyperedge: "none" | "reified"
      When "reified", rebuild true hyperedges and skip membership edges from binary import.
    """
    from ..core.graph import Graph

    H = Graph()

    # vertices
    names = igG.vs["name"] if "name" in igG.vs.attributes() else list(range(igG.vcount()))
    for i, vid in enumerate(names):
        try:
            H.add_vertex(vid)
        except Exception:
            pass
        vattrs = {k: igG.vs[i][k] for k in igG.vs.attributes()}
        if vattrs:
            try:
                H.set_vertex_attrs(vid, **vattrs)
            except Exception:
                pass

    membership_idx = set()
    if hyperedge == "reified":
        hyperdefs, membership_idx = _ig_collect_reified(
            igG,
            he_node_flag=he_node_flag,
            he_id_attr=he_id_attr,
            role_attr=role_attr,
            coeff_attr=coeff_attr,
            membership_attr=membership_attr,
        )
        for eid, directed, head_map, tail_map, he_attrs, hi in hyperdefs:
            for x in set(head_map) | set(tail_map):
                try:
                    H.add_vertex(x)
                except Exception:
                    pass
            if directed:
                try:
                    H.add_hyperedge(
                        head=list(head_map), tail=list(tail_map), edge_id=eid, edge_directed=True
                    )
                except Exception:
                    pass
                try:
                    H.set_edge_attrs(
                        eid,
                        __source_attr={u: {"__value": c} for u, c in head_map.items()},
                        __target_attr={v: {"__value": c} for v, c in tail_map.items()},
                    )
                except Exception:
                    pass
            else:
                members = list(set(head_map) | set(tail_map))
                try:
                    H.add_hyperedge(members=members, edge_id=eid, edge_directed=False)
                except Exception:
                    pass
                try:
                    H.set_edge_attrs(
                        eid,
                        __source_attr={u: {"__value": head_map.get(u, 1.0)} for u in members},
                        __target_attr={v: {"__value": tail_map.get(v, 1.0)} for v in members},
                    )
                except Exception:
                    pass
            # copy HE-node attrs (minus markers)
            he_node_attrs = {
                k: igG.vs[hi][k] for k in igG.vs.attributes() if k not in {he_node_flag, he_id_attr}
            }
            if he_node_attrs:
                try:
                    H.set_edge_attrs(eid, **he_node_attrs)
                except Exception:
                    pass

    # binary edges (skip membership edges if reified)
    is_dir = igG.is_directed()
    seen_auto = 0
    for e in igG.es:
        if e.index in membership_idx:
            continue
        src = names[e.source]
        dst = names[e.target]
        d = {k: e[k] for k in igG.es.attributes()}

        eid = d.get("eid")
        if eid is None:
            seen_auto += 1
            eid = f"ig::e#{seen_auto}"

        e_directed = bool(d.get("directed", is_dir))
        w = d.get("weight", d.get("__weight", 1.0))

        try:
            H.add_vertex(src)
            H.add_vertex(dst)
        except Exception:
            pass
        try:
            H.add_edge(src, dst, edge_id=eid, edge_directed=e_directed)
        except Exception:
            H.add_edge(src, dst, edge_id=eid, edge_directed=True)

        try:
            H.edge_weights[eid] = float(w)
        except Exception:
            pass

        if d:
            try:
                H.set_edge_attrs(eid, **d)
            except Exception:
                pass

    return H


class IGraphAdapter:
    """Legacy adapter class for backward compatibility.

    Methods
    -------
    export(graph, **kwargs)
        Export Graph to igraph.Graph without manifest (lossy).

    """

    def export(self, graph, **kwargs):
        """Export Graph to igraph.Graph without manifest.

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
