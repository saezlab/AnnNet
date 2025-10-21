from __future__ import annotations
import json
from typing import Dict, Any, List, Tuple


def to_cx2(graph: "Graph", path, *, hyperedge_mode="reify", public_only=True):
    """
    Minimal CX2 exporter.
    - nodes, edges, nodeAttributes, edgeAttributes
    - hyperedges: reify as HE node + membership edges (role, coeff, membership_of)
    - layers: encode as edgeAttributes 'layer' (multi-valued) and opaque 'edgeLayers' aspect with weights
    """
    nodes = []
    nodeAttrs = []
    edges = []
    edgeAttrs = []
    opaque = {}

    # collect vertices
    for v in graph.vertices():
        nodes.append({"@id": v})
        try:
            attrs = graph.vertex_attributes.filter(
                graph.vertex_attributes["vertex_id"] == v
            ).to_dicts()
            if attrs:
                d = dict(attrs[0]); d.pop("vertex_id", None)
                if public_only:
                    d = {k: val for k, val in d.items() if not str(k).startswith("__")}
                for k, val in d.items():
                    nodeAttrs.append({"po": v, "n": k, "v": val})
        except Exception:
            pass

    # layers membership + weights
    edge_layers = []
    lids = []
    try:
        lids = list(graph.list_layers(include_default=True))
    except Exception:
        pass

    # edges + hyperedges (reify)
    for eidx in range(graph.number_of_edges()):
        eid = graph.idx_to_edge[eidx]
        S, T = graph.get_edge(eidx)
        is_hyper = (graph.edge_kind.get(eid) == "hyper")
        if not is_hyper:
            members = (S | T)
            u, v = (next(iter(members)), next(iter(members))) if len(members)==1 else sorted(members)
            edges.append({"@id": eid, "s": u, "t": v, "i": True})
        else:
            he_id = f"he::{eid}"
            nodes.append({"@id": he_id})
            nodeAttrs.append({"po": he_id, "n": "is_hyperedge", "v": True})
            nodeAttrs.append({"po": he_id, "n": "eid", "v": eid})
            # membership edges
            head_map = _endpoint_coeff_map({}, "__source_attr", S) or {u: 1.0 for u in (S or [])}
            tail_map = _endpoint_coeff_map({}, "__target_attr", T) or {v: 1.0 for v in (T or [])}
            for u, c in (tail_map or {}).items():
                edges.append({"@id": f"m::{eid}::{u}::tail", "s": u, "t": he_id, "i": True})
                edgeAttrs.extend([{"po": f"m::{eid}::{u}::tail", "n": "role", "v": "tail"},
                                  {"po": f"m::{eid}::{u}::tail", "n": "coeff", "v": float(c)},
                                  {"po": f"m::{eid}::{u}::tail", "n": "membership_of", "v": eid}])
            for v, c in (head_map or {}).items():
                edges.append({"@id": f"m::{eid}::{v}::head", "s": he_id, "t": v, "i": True})
                edgeAttrs.extend([{"po": f"m::{eid}::{v}::head", "n": "role", "v": "head"},
                                  {"po": f"m::{eid}::{v}::head", "n": "coeff", "v": float(c)},
                                  {"po": f"m::{eid}::{v}::head", "n": "membership_of", "v": eid}])

        # public edge attrs onto edgeAttrs
        try:
            attrs = graph.edge_attributes.filter(
                graph.edge_attributes["edge_id"] == eid
            ).to_dicts()
            if attrs:
                d = dict(attrs[0]); d.pop("edge_id", None)
                if public_only:
                    d = {k: val for k, val in d.items() if not str(k).startswith("__")}
                for k, val in d.items():
                    edgeAttrs.append({"po": eid, "n": k, "v": val})
        except Exception:
            pass

        # layers + weights
        for lid in lids:
            try:
                if graph.is_edge_in_layer(lid, eid):
                    edgeAttrs.append({"po": eid, "n": "layer", "v": lid})
                    try:
                        w = graph.get_edge_layer_attr(lid, eid, "weight", default=None)
                    except Exception:
                        try: w = graph.get_edge_layer_attr(lid, eid, "weight")
                        except Exception: w = None
                    if w is not None:
                        edge_layers.append({"layer_id": lid, "edge_id": eid, "weight": float(w)})
            except Exception:
                continue

    cx2 = [
        {"CXVersion": "2.0"},
        {"nodes": nodes},
        {"edges": edges},
        {"nodeAttributes": nodeAttrs} if nodeAttrs else {},
        {"edgeAttributes": edgeAttrs} if edgeAttrs else {},
        {"opaqueAspects": {"edgeLayers": edge_layers}} if edge_layers else {},
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump([obj for obj in cx2 if obj], f, indent=2)


def from_cx2(path: str,
             *,
             he_node_flag: str = "is_hyperedge",
             he_id_attr: str = "eid",
             role_attr: str = "role",
             coeff_attr: str = "coeff",
             membership_attr: str = "membership_of",
             layer_attr: str = "layer",
             encoding: str = "utf-8") -> "Graph":
    """
    Load a CX2 JSON file exported with hyperedge reification.
    Rebuilds true hyperedges and restores layers + per-layer weights.

    - Nodes with nodeAttribute {is_hyperedge: True} are hyperedge nodes.
    - Membership edges are those incident to a hyperedge node and carrying
      edgeAttributes {role=head|tail|member, coeff, membership_of=<EID>}.
    - Binary edges are edges whose endpoints are both *non*-hyperedge nodes.

    Everything else (attrs) is copied verbatim. Baseline weights:
    - Binary edge: edge attribute "weight" if present.
    - Hyperedge: HE-node attribute "hyper_weight" if present.
    """
    from ..core.graph import Graph

    # ---- read & index aspects ----
    with open(path, "r", encoding=encoding) as f:
        doc = json.load(f)

    # CX2 is a list of aspects; build a name->payload map
    aspects: Dict[str, Any] = {}
    opaque: Dict[str, Any] = {}
    if isinstance(doc, list):
        for item in doc:
            if not isinstance(item, dict):
                continue
            # capture top-level aspects
            for k, v in item.items():
                if k == "opaqueAspects" and isinstance(v, dict):
                    opaque.update(v)
                else:
                    aspects[k] = v
    elif isinstance(doc, dict):
        aspects = {k: v for k, v in doc.items()}
        if "opaqueAspects" in aspects and isinstance(aspects["opaqueAspects"], dict):
            opaque.update(aspects["opaqueAspects"])

    nodes = aspects.get("nodes", []) or []
    edges = aspects.get("edges", []) or []
    nodeAttrs = aspects.get("nodeAttributes", []) or []
    edgeAttrs = aspects.get("edgeAttributes", []) or []
    edgeLayers = opaque.get("edgeLayers", []) or []  # [{"layer_id","edge_id","weight"}...]

    # Build attribute maps
    node_attr_map: Dict[Any, Dict[str, Any]] = {}
    for a in nodeAttrs:
        po = a.get("po")
        if po is None:
            continue
        node_attr_map.setdefault(po, {})
        n = a.get("n"); v = a.get("v")
        if n is not None:
            # multi-valued attributes: accumulate into list
            if n in node_attr_map[po]:
                prev = node_attr_map[po][n]
                if isinstance(prev, list):
                    prev.append(v)
                else:
                    node_attr_map[po][n] = [prev, v]
            else:
                node_attr_map[po][n] = v

    edge_attr_map: Dict[Any, Dict[str, Any]] = {}
    # keep *all* values for multi-valued attributes (e.g., repeated "layer")
    edge_attr_multimap: Dict[Any, Dict[str, List[Any]]] = {}
    for a in edgeAttrs:
        po = a.get("po")
        if po is None:
            continue
        n = a.get("n"); v = a.get("v")
        if n is None:
            continue
        edge_attr_map.setdefault(po, {})
        if n in edge_attr_map[po]:
            # promote to multimap
            edge_attr_multimap.setdefault(po, {}).setdefault(n, [])
            if n not in edge_attr_multimap[po]:
                edge_attr_multimap[po][n] = [edge_attr_map[po][n]]
            edge_attr_multimap[po][n].append(v)
        else:
            edge_attr_map[po][n] = v

    # After the pass above, ensure multimap values override singletons
    for eid, mm in edge_attr_multimap.items():
        edge_attr_map.setdefault(eid, {})
        for k, arr in mm.items():
            edge_attr_map[eid][k] = list(arr)

    # Identify hyperedge nodes
    he_nodes = set()
    for n in nodes:
        nid = n.get("@id")
        if nid is None:
            continue
        attrs = node_attr_map.get(nid, {})
        if bool(attrs.get(he_node_flag, False)):
            he_nodes.add(nid)

    # Index edges by endpoints & ids for quick lookup
    # edge record expected keys: "@id", "s", "t", maybe "i"
    edge_by_id: Dict[Any, Dict[str, Any]] = {}
    incident_to: Dict[Any, List[Any]] = {}  # node_id -> [edge_id,...]
    for e in edges:
        eid = e.get("@id")
        if eid is None:
            continue
        edge_by_id[eid] = e
        s = e.get("s"); t = e.get("t")
        if s is not None:
            incident_to.setdefault(s, []).append(eid)
        if t is not None:
            incident_to.setdefault(t, []).append(eid)

    # ---- Build the graph ----
    H = Graph()

    # 1) Add *real* vertices (skip HE nodes)
    for n in nodes:
        nid = n.get("@id")
        if nid is None or nid in he_nodes:
            continue
        H.add_vertex(nid)
        attrs = dict(node_attr_map.get(nid, {}))
        # nothing to strip for real vertices
        if attrs:
            try:
                H.set_vertex_attrs(nid, **attrs)
            except Exception:
                pass

    # 2) Reconstruct hyperedges from HE nodes
    created_he_eids = set()
    for he in he_nodes:
        nd_attrs = dict(node_attr_map.get(he, {}))
        eid = nd_attrs.get(he_id_attr, f"he::{he}")

        # Collect membership edges around this HE node
        head_map: Dict[Any, float] = {}
        tail_map: Dict[Any, float] = {}
        saw_head = saw_tail = saw_member = False

        for eidx in incident_to.get(he, []):
            # attributes for this membership edge
            eattrs = edge_attr_map.get(eidx, {})
            role = eattrs.get(role_attr, None)
            coeff = eattrs.get(coeff_attr, eattrs.get("__value", 1.0))
            try:
                coeff = float(coeff)
            except Exception:
                coeff = 1.0

            rec = edge_by_id.get(eidx, {})
            s, t = rec.get("s"), rec.get("t")
            other = t if s == he else s
            if other is None:
                continue

            # ensure vertex exists
            try:
                H.add_vertex(other)
            except Exception:
                pass

            if role == "head":
                head_map[other] = coeff; saw_head = True
            elif role == "tail":
                tail_map[other] = coeff; saw_tail = True
            else:
                # treat as undirected membership
                head_map[other] = coeff; tail_map[other] = coeff; saw_member = True

        # Decide directedness
        directed = True if (saw_head or saw_tail) else False

        # Create hyperedge
        if directed:
            try:
                H.add_hyperedge(head=list(head_map), tail=list(tail_map), edge_id=eid, edge_directed=True)
            except Exception:
                # degrade if needed
                if len(head_map) == 1 and len(tail_map) == 1:
                    hu = next(iter(tail_map)); hv = next(iter(head_map))
                    try:
                        H.add_edge(hu, hv, edge_id=eid, edge_directed=True)
                    except Exception:
                        pass
        else:
            members = list(set(head_map) | set(tail_map))
            try:
                H.add_hyperedge(members=members, edge_id=eid, edge_directed=False)
            except Exception:
                pass

        created_he_eids.add(eid)

        # Store endpoint coefficients in private maps (what your exporters read)
        try:
            H.set_edge_attrs(eid,
                __source_attr={u: {"__value": c} for u, c in head_map.items()},
                __target_attr={v: {"__value": c} for v, c in tail_map.items()}
            )
        except Exception:
            pass

        # Copy HE-node attrs onto the edge (minus the HE markers)
        he_edge_attrs = {k: v for k, v in nd_attrs.items() if k not in {he_node_flag, he_id_attr}}
        if he_edge_attrs:
            try:
                H.set_edge_attrs(eid, **he_edge_attrs)
            except Exception:
                pass

        # Baseline hyperedge weight from HE node, if present
        hw = nd_attrs.get("hyper_weight", None)
        if hw is not None:
            try:
                H.edge_weights[eid] = float(hw)
            except Exception:
                pass

    # 3) Import binary edges (skip membership edges that touch HE nodes)
    for eid, rec in edge_by_id.items():
        u = rec.get("s"); v = rec.get("t")
        if u in he_nodes or v in he_nodes:
            continue  # membership edge
        if u is None or v is None:
            continue

        # ensure vertices
        try:
            H.add_vertex(u); H.add_vertex(v)
        except Exception:
            pass

        # directedness: edge attr 'directed' wins; else assume True
        eattrs = dict(edge_attr_map.get(eid, {}))
        e_directed = bool(eattrs.get("directed", True))
        try:
            H.add_edge(u, v, edge_id=eid, edge_directed=e_directed)
        except Exception:
            H.add_edge(u, v, edge_id=eid, edge_directed=True)

        # baseline weight from edge attr 'weight' if present
        if "weight" in eattrs:
            try:
                H.edge_weights[eid] = float(eattrs["weight"])
            except Exception:
                pass

        # Attach edge attrs (drop membership-only keys if present)
        for k in (role_attr, coeff_attr, membership_attr):
            if k in eattrs:
                eattrs.pop(k, None)
        try:
            if eattrs:
                H.set_edge_attrs(eid, **eattrs)
        except Exception:
            pass

    # 4) Layers: from repeated edgeAttribute 'layer' and from opaque edgeLayers
    # 4a) create all layers mentioned
    layers_seen = set()
    # collect from edgeAttributes layer(s)
    for eid, attrs in edge_attr_map.items():
        vals = attrs.get(layer_attr)
        if isinstance(vals, list):
            layers_seen.update(str(v) for v in vals)
        elif vals is not None:
            layers_seen.add(str(vals))
    # collect from opaque edgeLayers
    for rec in edgeLayers:
        lid = rec.get("layer_id")
        if lid is not None:
            layers_seen.add(str(lid))

    for lid in layers_seen:
        try:
            if lid not in set(H.list_layers(include_default=True)):
                H.add_layer(lid)
        except Exception:
            pass

    # 4b) memberships via 'layer' attributes
    for eid, attrs in edge_attr_map.items():
        lids = attrs.get(layer_attr)
        if lids is None:
            continue
        lids_iter = lids if isinstance(lids, list) else [lids]
        for lid in lids_iter:
            try:
                H.add_edge_to_layer(str(lid), eid)
            except Exception:
                pass

    # 4c) per-layer weights via opaque edgeLayers
    for rec in edgeLayers:
        lid = rec.get("layer_id"); eid = rec.get("edge_id")
        if lid is None or eid is None:
            continue
        try:
            H.add_edge_to_layer(str(lid), eid)
        except Exception:
            pass
        if "weight" in rec:
            try:
                H.set_edge_layer_attrs(str(lid), eid, weight=float(rec["weight"]))
            except Exception:
                try:
                    H.set_edge_layer_attr(str(lid), eid, "weight", float(rec["weight"]))
                except Exception:
                    pass

    return H
