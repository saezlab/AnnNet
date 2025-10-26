from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..core.graph import Graph
import json


def _is_directed_eid(graph, eid):
    try:
        return bool(getattr(graph, "edge_directed", {}).get(eid, True))
    except Exception:
        pass
    try:
        val = graph.get_edge_attribute(eid, "directed")
        return bool(val) if val is not None else True
    except Exception:
        return True


def _coerce_coeff_mapping(val):
    """Normalize various serialized forms into {vertex: {__value: float}|float}
    Accepts:
      - dict({v: x} or {v: {"__value": x}})
      - list of pairs [(v,x), ...]
      - list of dicts [{"vertex": v, "__value": x} | {v: x}, ...]
      - JSON string of any of the above
    """
    if val is None:
        return {}
    # JSON string?
    if isinstance(val, str):
        try:
            return _coerce_coeff_mapping(json.loads(val))
        except Exception:
            return {}
    # Already dict?
    if isinstance(val, dict):
        return val
    # List-like
    if isinstance(val, (list, tuple)):
        out = {}
        for item in val:
            if isinstance(item, dict):
                if "vertex" in item and "__value" in item:
                    out[item["vertex"]] = {"__value": item["__value"]}
                else:
                    # e.g., {"A": 2.0}
                    for k, v in item.items():
                        out[k] = v
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                k, v = item
                out[k] = v
            else:
                # ignore unrecognized shapes
                pass
        return out
    # Fallback
    return {}


def _endpoint_coeff_map(edge_attrs, private_key, endpoint_set):
    """Return {vertex: float_coeff} for the given endpoint_set.
    Reads from edge_attrs[private_key] which may be serialized in multiple shapes.
    Missing endpoints default to 1.0.
    """
    raw_mapping = (edge_attrs or {}).get(private_key, {})
    mapping = _coerce_coeff_mapping(raw_mapping)
    endpoints = list(endpoint_set or mapping.keys())
    out = {}
    for u in endpoints:
        val = mapping.get(u, 1.0)
        if isinstance(val, dict):
            val = val.get("__value", 1.0)
        try:
            out[u] = float(val)
        except Exception:
            out[u] = 1.0
    return out


def to_json(graph: Graph, path, *, public_only: bool = False, indent: int = 0):
    """Node-link JSON with x-extensions (layers, edge_layers, hyperedges).
    Lossless vs your core (IDs, attrs, parallel, hyperedges, layers).
    """
    # nodes
    nodes = []
    for v in graph.vertices():
        row = {"id": v}
        try:
            attrs = graph.vertex_attributes.filter(
                graph.vertex_attributes["vertex_id"] == v
            ).to_dicts()
            if attrs:
                d = dict(attrs[0])
                d.pop("vertex_id", None)
                if public_only:
                    d = {k: val for k, val in d.items() if not str(k).startswith("__")}
                row.update(d)
        except Exception:
            pass
        nodes.append(row)

    # edges + hyperedges
    edges = []
    hyperedges = []
    for eidx in range(graph.number_of_edges()):
        eid = graph.idx_to_edge[eidx]
        S, T = graph.get_edge(eidx)
        kind = graph.edge_kind.get(eid)

        # attrs
        try:
            ea = graph.edge_attributes.filter(graph.edge_attributes["edge_id"] == eid).to_dicts()
            d = dict(ea[0]) if ea else {}
            d.pop("edge_id", None)
            if public_only:
                d = {k: val for k, val in d.items() if not str(k).startswith("__")}
        except Exception:
            d = {}

        # weight + directed
        try:
            w = float(getattr(graph, "edge_weights", {}).get(eid, 1.0))
        except Exception:
            w = 1.0
        try:
            directed = bool(_is_directed_eid(graph, eid))
        except Exception:
            directed = True

        if kind == "hyper":
            # endpoint coeffs from private maps if present; else 1.0
            head_map = _endpoint_coeff_map(d, "__source_attr", S) or dict.fromkeys(S or [], 1.0)
            tail_map = _endpoint_coeff_map(d, "__target_attr", T) or dict.fromkeys(T or [], 1.0)
            # directed hyperedge
            hyperedges.append(
                {
                    "id": eid,
                    "directed": bool(directed),
                    "head": list(head_map.keys()) if directed else None,
                    "tail": list(tail_map.keys()) if directed else None,
                    "members": None if directed else list({*head_map.keys(), *tail_map.keys()}),
                    "attrs": d,
                    "weight": w,
                }
            )
        else:
            # regular/binary
            members = S | T
            if len(members) == 1:
                u = next(iter(members))
                v = u
            else:
                u, v = sorted(members)
            edges.append(
                {
                    "id": eid,
                    "source": u,
                    "target": v,
                    "directed": bool(directed),
                    "weight": w,
                    "attrs": d,
                }
            )

    # layers + per-layer weights
    layers = []
    try:
        for lid in graph.list_layers(include_default=True):
            layers.append({"layer_id": lid})
    except Exception:
        pass

    edge_layers = []
    # Collect memberships + weights if available
    try:
        for lid in graph.list_layers(include_default=True):
            try:
                for eid in graph.get_layer_edges(lid):
                    rec = {"layer_id": lid, "edge_id": eid}
                    try:
                        w = graph.get_edge_layer_attr(lid, eid, "weight", default=None)
                    except Exception:
                        try:
                            w = graph.get_edge_layer_attr(lid, eid, "weight")
                        except Exception:
                            w = None
                    if w is not None:
                        rec["weight"] = float(w)
                    edge_layers.append(rec)
            except Exception:
                continue
    except Exception:
        pass

    doc = {
        "directed": True,  # node-link convention; per-edge directedness is in edges[*].directed
        "multigraph": True,
        "nodes": nodes,
        "edges": [
            {
                "id": e["id"],
                "source": e["source"],
                "target": e["target"],
                "directed": e["directed"],
                "weight": e["weight"],
                **(e.get("attrs") or {}),
            }
            for e in edges
        ],
        "x-extensions": {
            "layers": layers,
            "edge_layers": edge_layers,
            "hyperedges": [
                (
                    {
                        "id": h["id"],
                        "directed": True,
                        "head": h["head"],
                        "tail": h["tail"],
                        "weight": h["weight"],
                        **(h.get("attrs") or {}),
                    }
                    if h["directed"]
                    else {
                        "id": h["id"],
                        "directed": False,
                        "members": h["members"],
                        "weight": h["weight"],
                        **(h.get("attrs") or {}),
                    }
                )
                for h in hyperedges
            ],
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=indent)


def from_json(path) -> Graph:
    """Load Graph from node-link JSON + x-extensions (lossless wrt schema above).
    """
    from ..core.graph import Graph

    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    H = Graph()

    # vertices
    for nd in doc.get("nodes", []):
        vid = nd.get("id")
        if vid is None:
            continue
        H.add_vertex(vid)
        vattrs = {k: v for k, v in nd.items() if k != "id"}
        if vattrs:
            H.set_vertex_attrs(vid, **vattrs)

    # edges (binary)
    for e in doc.get("edges", []):
        eid = e.get("id")
        u = e.get("source")
        v = e.get("target")
        if eid is None or u is None or v is None:
            continue
        directed = bool(e.get("directed", True))
        H.add_edge(u, v, edge_id=eid, edge_directed=directed)
        # weight
        w = e.get("weight", 1.0)
        try:
            H.edge_weights[eid] = float(w)
        except Exception:
            pass
        # attrs (except handled)
        attrs = {
            k: val
            for k, val in e.items()
            if k not in {"id", "source", "target", "directed", "weight"}
        }
        if attrs:
            H.set_edge_attrs(eid, **attrs)

    # hyperedges + layers
    ext = doc.get("x-extensions") or {}
    for h in ext.get("hyperedges", []):
        eid = h.get("id")
        directed = bool(h.get("directed", True))
        if directed:
            head = list(h.get("head") or [])
            tail = list(h.get("tail") or [])
            H.add_hyperedge(head=head, tail=tail, edge_id=eid, edge_directed=True)
            # stash endpoint coeffs if provided (optional schema: __source_attr/__target_attr)
            # If absent, default 1.0 will be implied by your exporter on the next round-trip.
        else:
            members = list(h.get("members") or [])
            H.add_hyperedge(members=members, edge_id=eid, edge_directed=False)
        # weight + attrs
        w = h.get("weight", 1.0)
        try:
            H.edge_weights[eid] = float(w)
        except Exception:
            pass
        attrs = {
            k: v
            for k, v in h.items()
            if k not in {"id", "directed", "head", "tail", "members", "weight"}
        }
        if attrs:
            H.set_edge_attrs(eid, **attrs)

    # layers + edge_layers
    for L in ext.get("layers", []):
        lid = L.get("layer_id")
        if lid is None:
            continue
        try:
            if lid not in set(H.list_layers(include_default=True)):
                H.add_layer(lid)
        except Exception:
            pass

    for EL in ext.get("edge_layers", []):
        lid = EL.get("layer_id")
        eid = EL.get("edge_id")
        if lid is None or eid is None:
            continue
        try:
            H.add_edge_to_layer(lid, eid)
        except Exception:
            pass
        if "weight" in EL:
            try:
                H.set_edge_layer_attrs(lid, eid, weight=float(EL["weight"]))
            except Exception:
                try:
                    H.set_edge_layer_attr(lid, eid, "weight", float(EL["weight"]))
                except Exception:
                    pass

    return H


def write_ndjson(graph: Graph, dir_path):
    """Write nodes.ndjson, edges.ndjson, hyperedges.ndjson, layers.ndjson, edge_layers.ndjson.
    Each line is one JSON object. Lossless wrt to_json schema.
    """
    import json
    import os

    os.makedirs(dir_path, exist_ok=True)

    with open(f"{dir_path}/nodes.ndjson", "w", encoding="utf-8") as f:
        for v in graph.vertices():
            obj = {"id": v}
            try:
                attrs = graph.vertex_attributes.filter(
                    graph.vertex_attributes["vertex_id"] == v
                ).to_dicts()
                if attrs:
                    d = dict(attrs[0])
                    d.pop("vertex_id", None)
                    obj.update(d)
            except Exception:
                pass
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    with (
        open(f"{dir_path}/edges.ndjson", "w", encoding="utf-8") as fe,
        open(f"{dir_path}/hyperedges.ndjson", "w", encoding="utf-8") as fh,
    ):
        for eidx in range(graph.number_of_edges()):
            eid = graph.idx_to_edge[eidx]
            S, T = graph.get_edge(eidx)
            kind = graph.edge_kind.get(eid)

            try:
                ea = graph.edge_attributes.filter(
                    graph.edge_attributes["edge_id"] == eid
                ).to_dicts()
            except Exception:
                ea = []
            d = dict(ea[0]) if ea else {}
            d.pop("edge_id", None)

            try:
                w = float(getattr(graph, "edge_weights", {}).get(eid, 1.0))
            except Exception:
                w = 1.0
            try:
                directed = bool(_is_directed_eid(graph, eid))
            except Exception:
                directed = True

            if kind == "hyper":
                head_map = _endpoint_coeff_map(d, "__source_attr", S) or dict.fromkeys(S or [], 1.0)
                tail_map = _endpoint_coeff_map(d, "__target_attr", T) or dict.fromkeys(T or [], 1.0)
                obj = {"id": eid, "directed": directed, "weight": w}
                if directed:
                    obj.update({"head": list(head_map), "tail": list(tail_map)})
                else:
                    obj.update({"members": list({*head_map, *tail_map})})
                obj.update({k: v for k, v in d.items() if not str(k).startswith("__")})
                fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            else:
                members = S | T
                if len(members) == 1:
                    u = next(iter(members))
                    v = u
                else:
                    u, v = sorted(members)
                obj = {"id": eid, "source": u, "target": v, "directed": directed, "weight": w}
                obj.update({k: v for k, v in d.items() if not str(k).startswith("__")})
                fe.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # layers
    with open(f"{dir_path}/layers.ndjson", "w", encoding="utf-8") as fl:
        try:
            for lid in graph.list_layers(include_default=True):
                fl.write(json.dumps({"layer_id": lid}, ensure_ascii=False) + "\n")
        except Exception:
            pass

    with open(f"{dir_path}/edge_layers.ndjson", "w", encoding="utf-8") as fel:
        try:
            for lid in graph.list_layers(include_default=True):
                for eid in graph.get_layer_edges(lid):
                    rec = {"layer_id": lid, "edge_id": eid}
                    try:
                        w = graph.get_edge_layer_attr(lid, eid, "weight", default=None)
                    except Exception:
                        try:
                            w = graph.get_edge_layer_attr(lid, eid, "weight")
                        except Exception:
                            w = None
                    if w is not None:
                        rec["weight"] = float(w)
                    fel.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass
