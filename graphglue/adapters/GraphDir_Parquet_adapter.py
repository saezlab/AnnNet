from __future__ import annotations
from pathlib import Path
import json
import polars as pl

def write_parquet_graphdir(graph: "Graph", path):
    """
    Write lossless GraphDir:
      vertices.parquet, edges.parquet, layers.parquet, edge_layers.parquet, manifest.json
    Wide tables (attrs as columns). Hyperedges stored with 'kind' and head/tail/members lists.
    """
    path = Path(path); path.mkdir(parents=True, exist_ok=True)

    # vertices
    v_rows = []
    for v in graph.vertices():
        row = {"vertex_id": v}
        try:
            attrs = graph.vertex_attributes.filter(
                graph.vertex_attributes["vertex_id"] == v
            ).to_dicts()
            if attrs:
                d = dict(attrs[0]); d.pop("vertex_id", None); row.update(d)
        except Exception: pass
        v_rows.append(row)
    pl.DataFrame(v_rows).write_parquet(path/"vertices.parquet", compression="zstd")

    # edges
    e_rows = []
    for eidx in range(graph.number_of_edges()):
        eid = graph.idx_to_edge[eidx]
        S, T = graph.get_edge(eidx)
        kind = graph.edge_kind.get(eid)
        row = {"edge_id": eid,
               "kind": ("hyper" if kind == "hyper" else "binary"),
               "directed": bool(_is_directed_eid(graph, eid)),
               "weight": float(getattr(graph, "edge_weights", {}).get(eid, 1.0))}
        try:
            attrs = graph.edge_attributes.filter(
                graph.edge_attributes["edge_id"] == eid
            ).to_dicts()
            if attrs:
                d = dict(attrs[0]); d.pop("edge_id", None); row.update(d)
        except Exception: pass

        if row["kind"] == "binary":
            members = (S | T)
            if len(members) == 1:
                u = next(iter(members)); v = u
            else:
                u, v = sorted(members)
            row.update({"source": u, "target": v})
        else:
            head_map = _endpoint_coeff_map(row, "__source_attr", S) or {u: 1.0 for u in (S or [])}
            tail_map = _endpoint_coeff_map(row, "__target_attr", T) or {v: 1.0 for v in (T or [])}
            row.update({
                "head": list(head_map.keys()),
                "tail": list(tail_map.keys()),
                "members": list({*head_map.keys(), *tail_map.keys()}) if not row["directed"] else None
            })
        e_rows.append(row)
    pl.DataFrame(e_rows).write_parquet(path/"edges.parquet", compression="zstd")

    # layers
    L = []
    try:
        for lid in graph.list_layers(include_default=True):
            L.append({"layer_id": lid})
    except Exception:
        pass
    pl.DataFrame(L).write_parquet(path/"layers.parquet", compression="zstd")

    # edge_layers
    EL = []
    try:
        for lid in graph.list_layers(include_default=True):
            for eid in graph.get_layer_edges(lid):
                rec = {"layer_id": lid, "edge_id": eid}
                try:
                    w = graph.get_edge_layer_attr(lid, eid, "weight", default=None)
                except Exception:
                    try: w = graph.get_edge_layer_attr(lid, eid, "weight")
                    except Exception: w = None
                if w is not None:
                    rec["weight"] = float(w)
                EL.append(rec)
    except Exception:
        pass
    pl.DataFrame(EL).write_parquet(path/"edge_layers.parquet", compression="zstd")

    # manifest.json (tiny)
    manifest = {
        "format_version": 1,
        "counts": {"V": len(v_rows), "E": len(e_rows), "layers": len(L)},
        "schema": {"edges.kind": ["binary","hyper"]},
        "provenance": {"package": "graphglue"}
    }
    (path/"manifest.json").write_text(json.dumps(manifest, indent=2))


def read_parquet_graphdir(path) -> "Graph":
    """
    Read GraphDir (lossless vs write_parquet_graphdir()).
    """
    from ..core.graph import Graph
    path = Path(path)
    V = pl.read_parquet(path/"vertices.parquet")
    E = pl.read_parquet(path/"edges.parquet")
    L = pl.read_parquet(path/"layers.parquet", use_pyarrow=True) if (path/"layers.parquet").exists() else pl.DataFrame([])
    EL= pl.read_parquet(path/"edge_layers.parquet", use_pyarrow=True) if (path/"edge_layers.parquet").exists() else pl.DataFrame([])

    H = Graph()
    # vertices
    for rec in V.to_dicts():
        vid = rec.pop("vertex_id")
        H.add_vertex(vid)
        if rec: H.set_vertex_attrs(vid, **rec)

    # edges
    for rec in E.to_dicts():
        eid = rec.pop("edge_id")
        kind = rec.pop("kind")
        directed = bool(rec.pop("directed", True))
        w = float(rec.pop("weight", 1.0))
        if kind == "binary":
            u = rec.pop("source"); v = rec.pop("target")
            H.add_edge(u, v, edge_id=eid, edge_directed=directed)
        else:
            head = list(rec.pop("head", []) or [])
            tail = list(rec.pop("tail", []) or [])
            members = list(rec.pop("members", []) or [])
            if directed:
                H.add_hyperedge(head=head, tail=tail, edge_id=eid, edge_directed=True)
            else:
                if not members: members = list(set(head) | set(tail))
                H.add_hyperedge(members=members, edge_id=eid, edge_directed=False)
        H.edge_weights[eid] = w
        if rec: H.set_edge_attrs(eid, **rec)

    # layers
    for rec in L.to_dicts():
        lid = rec.get("layer_id")
        try:
            if lid not in set(H.list_layers(include_default=True)):
                H.add_layer(lid)
        except Exception:
            pass

    # edge_layers
    for rec in EL.to_dicts():
        lid = rec.get("layer_id"); eid = rec.get("edge_id")
        if lid is None or eid is None: continue
        try: H.add_edge_to_layer(lid, eid)
        except Exception: pass
        if "weight" in rec:
            try: H.set_edge_layer_attrs(lid, eid, weight=float(rec["weight"]))
            except Exception:
                try: H.set_edge_layer_attr(lid, eid, "weight", float(rec["weight"]))
                except Exception: pass

    return H
