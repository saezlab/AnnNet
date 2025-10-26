import json
import math
import os
import re

import networkx as nx

from .networkx_adapter import from_nx, from_nx_only, to_nx

_BOOL = {"true": True, "false": False}
_NUM_RE = re.compile(r"^[+-]?(?:\d+|\d*\.\d+)(?:[eE][+-]?\d+)?$")


def _sanitize_graphml_inplace(G):
    """Remove values GraphML can't encode and normalize types before write."""

    def clean_dict(d: dict):
        for k in list(d.keys()):
            v = d[k]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                del d[k]
                continue
            if isinstance(v, (str, int, float, bool)):
                continue
            try:
                d[k] = json.dumps(v, ensure_ascii=False)
            except Exception:
                d[k] = str(v)

    clean_dict(G.graph)
    for _, data in G.nodes(data=True):
        clean_dict(data)
    for _, _, data in G.edges(data=True):
        clean_dict(data)


def _restore_types_graphml_inplace(G):
    """Heuristically restore types after read_graphml."""

    def coerce(s):
        if not isinstance(s, str):
            return s
        t = s.strip()
        low = t.lower()
        if low in _BOOL:
            return _BOOL[low]
        if (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]")):
            try:
                return json.loads(t)
            except Exception:
                return t
        if _NUM_RE.match(t):
            try:
                if "." not in t and "e" not in low:
                    return int(t)
                return float(t)
            except Exception:
                return t
        return t

    def fix_dict(d: dict):
        for k in list(d.keys()):
            d[k] = coerce(d[k])

    fix_dict(G.graph)
    for _, data in G.nodes(data=True):
        fix_dict(data)
    for _, _, data in G.edges(data=True):
        fix_dict(data)


def to_graphml(graph, path, *, directed=True, hyperedge_mode="reify", public_only=False):
    """Export via NetworkX with reified hyperedges; sanitize attrs for GraphML.
    Also writes a sidecar manifest for lossless re-import.
    """
    G, manifest = to_nx(
        graph, directed=directed, hyperedge_mode=hyperedge_mode, public_only=public_only
    )
    _sanitize_graphml_inplace(G)
    nx.write_graphml(G, path)
    # sidecar manifest
    mpath = str(path) + ".manifest.json"
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)


def from_graphml(path, *, hyperedge="reified"):
    """Import via NetworkX; if a sidecar manifest is present, use it as SSOT.
    Otherwise, fall back to from_nx_only with type restoration.
    """
    G = nx.read_graphml(path)
    _restore_types_graphml_inplace(G)
    mpath = str(path) + ".manifest.json"
    if os.path.exists(mpath):
        with open(mpath, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        # Rebuild exactly from the manifest (lossless), ignoring GraphML-added noise
        return from_nx(G, manifest, hyperedge=("reified" if hyperedge == "reified" else "none"))
    # Fallback (no manifest available)
    return from_nx_only(G, hyperedge=("reified" if hyperedge == "reified" else "none"))


def to_gexf(graph: "Graph", path, *, directed=True, hyperedge_mode="reify", public_only=False):
    G, _m = to_nx(graph, directed=directed, hyperedge_mode=hyperedge_mode, public_only=public_only)
    nx.write_gexf(G, path)


def from_gexf(path, *, hyperedge="reified") -> "Graph":
    G = nx.read_gexf(path)
    return from_nx_only(G, hyperedge=("reified" if hyperedge == "reified" else "none"))
