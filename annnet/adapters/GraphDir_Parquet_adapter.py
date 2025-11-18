from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from ..core.graph import Graph


def _strip_nulls(d: dict):
    # remove keys whose value is None or NaN
    clean = {}
    for k, v in list(d.items()):
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        clean[k] = v
    return clean


def _is_directed_eid(graph, eid):
    """Best-effort directedness probe; default True."""
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
    """Normalize various serialized forms into {vertex: {__value: float}|float}.
    Accepts dict | list | list-of-dicts | list-of-pairs | JSON string.
    """
    if val is None:
        return {}
    if isinstance(val, str):
        try:
            return _coerce_coeff_mapping(json.loads(val))
        except Exception:
            return {}
    if isinstance(val, dict):
        return val
    if isinstance(val, (list, tuple)):
        out = {}
        for item in val:
            if isinstance(item, dict):
                if "vertex" in item and "__value" in item:
                    out[item["vertex"]] = {"__value": item["__value"]}
                else:
                    for k, v in item.items():
                        out[k] = v
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                k, v = item
                out[k] = v
        return out
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


def write_parquet_graphdir(graph: Graph, path):
    """Write lossless GraphDir:
      vertices.parquet, edges.parquet, slices.parquet, edge_slices.parquet, manifest.json
    Wide tables (attrs as columns). Hyperedges stored with 'kind' and head/tail/members lists.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # vertices
    v_rows = []
    for v in graph.vertices():
        row = {"vertex_id": v}
        try:
            attrs = graph.vertex_attributes.filter(
                graph.vertex_attributes["vertex_id"] == v
            ).to_dicts()
            if attrs:
                d = dict(attrs[0])
                d.pop("vertex_id", None)
                row.update(d)
        except Exception:
            pass
        v_rows.append(row)
    pl.DataFrame(v_rows).write_parquet(path / "vertices.parquet", compression="zstd")

    # edges
    e_rows = []
    for eidx in range(graph.number_of_edges()):
        eid = graph.idx_to_edge[eidx]
        S, T = graph.get_edge(eidx)
        kind = graph.edge_kind.get(eid)
        row = {
            "edge_id": eid,
            "kind": ("hyper" if kind == "hyper" else "binary"),
            "directed": bool(_is_directed_eid(graph, eid)),
            "weight": float(getattr(graph, "edge_weights", {}).get(eid, 1.0)),
        }
        try:
            attrs = graph.edge_attributes.filter(graph.edge_attributes["edge_id"] == eid).to_dicts()
            if attrs:
                d = dict(attrs[0])
                d.pop("edge_id", None)
                row.update(d)
        except Exception:
            pass

        if row["kind"] == "binary":
            members = S | T
            if len(members) == 1:
                u = next(iter(members))
                v = u
            else:
                u, v = sorted(members)
            row.update({"source": u, "target": v})
        else:
            head_map = _endpoint_coeff_map(row, "__source_attr", S) or dict.fromkeys(S or [], 1.0)
            tail_map = _endpoint_coeff_map(row, "__target_attr", T) or dict.fromkeys(T or [], 1.0)
            row.update(
                {
                    "head": list(head_map.keys()),
                    "tail": list(tail_map.keys()),
                    "members": list({*head_map.keys(), *tail_map.keys()})
                    if not row["directed"]
                    else None,
                }
            )
        e_rows.append(row)
    pl.DataFrame(e_rows).write_parquet(path / "edges.parquet", compression="zstd")

    # slices
    L = []
    try:
        for lid in graph.list_slices(include_default=True):
            L.append({"slice_id": lid})
    except Exception:
        pass
    pl.DataFrame(L).write_parquet(path / "slices.parquet", compression="zstd")

    # edge_slices
    EL = []
    try:
        for lid in graph.list_slices(include_default=True):
            for eid in graph.get_slice_edges(lid):
                rec = {"slice_id": lid, "edge_id": eid}
                try:
                    w = graph.get_edge_slice_attr(lid, eid, "weight", default=None)
                except Exception:
                    try:
                        w = graph.get_edge_slice_attr(lid, eid, "weight")
                    except Exception:
                        w = None
                if w is not None:
                    rec["weight"] = float(w)
                EL.append(rec)
    except Exception:
        pass
    pl.DataFrame(EL).write_parquet(path / "edge_slices.parquet", compression="zstd")

    # manifest.json (tiny)
    manifest = {
        "format_version": 1,
        "counts": {"V": len(v_rows), "E": len(e_rows), "slices": len(L)},
        "schema": {"edges.kind": ["binary", "hyper"]},
        "provenance": {"package": "annnet"},
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2))


def read_parquet_graphdir(path) -> Graph:
    """Read GraphDir (lossless vs write_parquet_graphdir())."""
    from ..core.graph import Graph

    path = Path(path)
    V = pl.read_parquet(path / "vertices.parquet")
    E = pl.read_parquet(path / "edges.parquet")
    L = (
        pl.read_parquet(path / "slices.parquet", use_pyarrow=True)
        if (path / "slices.parquet").exists()
        else pl.DataFrame([])
    )
    EL = (
        pl.read_parquet(path / "edge_slices.parquet", use_pyarrow=True)
        if (path / "edge_slices.parquet").exists()
        else pl.DataFrame([])
    )

    H = Graph()
    # vertices
    for rec in V.to_dicts():
        vid = rec.pop("vertex_id")
        H.add_vertex(vid)
        if rec:
            H.set_vertex_attrs(vid, **rec)

    # edges
    for rec in E.to_dicts():
        eid = rec.pop("edge_id")
        kind = rec.pop("kind")
        directed = bool(rec.pop("directed", True))
        w = float(rec.pop("weight", 1.0))

        if kind == "binary":
            # take endpoints and drop hyper-only columns if present
            u = rec.pop("source", None)
            v = rec.pop("target", None)
            # these can exist as NULL because the DF is wide
            rec.pop("head", None)
            rec.pop("tail", None)
            rec.pop("members", None)

            if u is None or v is None:
                # defensive: reconstruct from any leftover endpoint list (rare)
                # if nothing found, skip cleanly
                continue

            H.add_edge(u, v, edge_id=eid, edge_directed=directed)

        else:  # hyper
            head = rec.pop("head", None) or []
            tail = rec.pop("tail", None) or []
            members = rec.pop("members", None) or []
            if directed:
                H.add_hyperedge(head=list(head), tail=list(tail), edge_id=eid, edge_directed=True)
            else:
                if not members:
                    members = list(set(head) | set(tail))
                H.add_hyperedge(members=list(members), edge_id=eid, edge_directed=False)

        # weight
        H.edge_weights[eid] = w

        # drop schema-nulls before attaching attrs (avoids head=None, etc.)
        rec = _strip_nulls(rec)
        if rec:
            H.set_edge_attrs(eid, **rec)

    # slices
    for rec in L.to_dicts():
        lid = rec.get("slice_id")
        try:
            if lid not in set(H.list_slices(include_default=True)):
                H.add_slice(lid)
        except Exception:
            pass

    # edge_slices
    for rec in EL.to_dicts():
        lid = rec.get("slice_id")
        eid = rec.get("edge_id")
        if lid is None or eid is None:
            continue
        try:
            H.add_edge_to_slice(lid, eid)
        except Exception:
            pass
        if "weight" in rec:
            try:
                H.set_edge_slice_attrs(lid, eid, weight=float(rec["weight"]))
            except Exception:
                try:
                    H.set_edge_slice_attr(lid, eid, "weight", float(rec["weight"]))
                except Exception:
                    pass

    return H
