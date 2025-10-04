"""
This module purposefully avoids importing stdlib `csv` and uses Polars for IO.

It ingests a CSV into IncidenceGraph by auto-detecting common schemas:
- Edge list (including DOK/COO triples and variations)
- Hyperedge table (members column or head/tail sets)
- Incidence matrix (rows=entities, cols=edges, ±w orientation)
- Adjacency matrix (square matrix, weighted/unweighted)
- LIL-style neighbor lists (single column of neighbors)

If auto-detection fails or you want control, pass schema=... explicitly.

Dependencies: polars, numpy, scipy (only if you use sparse helpers), IncidenceGraph

Design notes:
- We treat unknown columns as attributes ("pure" non-structural) and write them via
  the corresponding set_*_attrs APIs when applicable.
- Layers: if a `layer` column exists it can contain a single layer or multiple
  (separated by `|`, `;`, or `,`). Per-layer weight overrides support columns of the
  form `weight:<layer_name>`.
- Directedness: we honor an explicit `directed` column when present (truthy), else
  infer for incidence (presence of negative values) and adjacency (symmetry check).
- We try not to guess too hard. If the heuristics get it wrong, supply
  schema="edge_list" / "hyperedge" / "incidence" / "adjacency" / "lil".

Public entry points:
- load_csv_to_graph(path, graph=None, schema="auto", **options) -> IncidenceGraph
- from_dataframe(df, graph=None, schema="auto", **options) -> IncidenceGraph

Both will create and return an IncidenceGraph (or mutate the provided one).
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple, Dict, Any, List, Set
import math
import json
import re

import polars as pl
import numpy as np

from ..core.incgraph import IncidenceGraph

# ---------------------------
# Helpers / parsing utilities
# ---------------------------

_STR_TRUE = {"1", "true", "t", "yes", "y", "on"}
_STR_FALSE = {"0", "false", "f", "no", "n", "off"}
_LAYER_SEP = re.compile(r"[|;,]")
_SET_SEP = re.compile(r"[|;,]\s*")

SRC_COLS = ["source", "src", "from", "u"]
DST_COLS = ["target", "dst", "to", "v"]
WGT_COLS = ["weight", "w"]
DIR_COLS = ["directed", "is_directed", "dir", "orientation"]
LAYER_COLS = ["layer", "layers"]
EDGE_ID_COLS = ["edge", "edge_id", "id"]
NODE_ID_COLS = ["node", "node_id", "id", "name", "label"]
NEIGH_COLS = ["neighbors", "nbrs", "adj", "adjacency", "neighbors_out", "neighbors_in"]
MEMBERS_COLS = ["members", "verts", "participants"]
HEAD_COLS = ["head", "heads"]
TAIL_COLS = ["tail", "tails"]
ROW_COLS = ["row", "i", "r"]
COL_COLS = ["col", "column", "j", "c"]
VAL_COLS = ["val", "value", "w", "weight"]

RESERVED = set(SRC_COLS + DST_COLS + WGT_COLS + DIR_COLS + LAYER_COLS + EDGE_ID_COLS + NODE_ID_COLS + NEIGH_COLS + MEMBERS_COLS + HEAD_COLS + TAIL_COLS + ROW_COLS + COL_COLS + VAL_COLS)


def _norm(s: Any) -> str:
    if s is None:
        return ""
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        return str(s)
    return str(s).strip()


def _truthy(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return bool(int(x))
    s = str(x).strip().lower()
    if s in _STR_TRUE:
        return True
    if s in _STR_FALSE:
        return False
    return None


def _split_layers(cell: Any) -> List[str]:
    if cell is None:
        return []
    if isinstance(cell, str):
        cell = cell.strip()
        if not cell:
            return []
        # Try JSON array first
        if (cell.startswith("[") and cell.endswith("]")) or (cell.startswith("{") and cell.endswith("}")):
            try:
                val = json.loads(cell)
                if isinstance(val, (list, set, tuple)):
                    return [_norm(v) for v in val]
                if isinstance(val, dict):
                    return list(val.keys())
            except Exception:
                pass
        return [p.strip() for p in _LAYER_SEP.split(cell) if p.strip()]
    if isinstance(cell, (list, tuple, set)):
        return [_norm(v) for v in cell]
    return [str(cell)]


def _split_set(cell: Any) -> Set[str]:
    if cell is None:
        return set()
    if isinstance(cell, str):
        s = cell.strip()
        if not s:
            return set()
        # JSON array
        if (s.startswith("[") and s.endswith("]")):
            try:
                return { _norm(v) for v in json.loads(s) }
            except Exception:
                return {p.strip() for p in _SET_SEP.split(s) if p.strip()}
        return {p.strip() for p in _SET_SEP.split(s) if p.strip()}
    if isinstance(cell, (list, tuple, set)):
        return {_norm(v) for v in cell}
    return {str(cell)}


def _pick_first(df: pl.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k in cols_lower:
            return cols_lower[k]
    return None


def _is_numeric_series(s: pl.Series) -> bool:
    return s.dtype.is_numeric() or s.dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8)


def _attr_columns(df: pl.DataFrame, exclude: Iterable[str]) -> List[str]:
    excl = {c.lower() for c in exclude}
    return [c for c in df.columns if c.lower() not in excl]


# ---------------------------
# Schema detection
# ---------------------------

def _detect_schema(df: pl.DataFrame) -> str:
    cols = [c.lower() for c in df.columns]

    # Hyperedge table if we have 'members' OR (head and tail)
    if any(c in cols for c in MEMBERS_COLS) or (any(c in cols for c in HEAD_COLS) and any(c in cols for c in TAIL_COLS)):
        return "hyperedge"

    # LIL: neighbors column
    if any(c in cols for c in NEIGH_COLS):
        return "lil"

    # COO/DOK triple variants
    if any(c in cols for c in ROW_COLS) and any(c in cols for c in COL_COLS) and any(c in cols for c in VAL_COLS):
        return "edge_list"

    # Classic edge list (src/dst present)
    if any(c in cols for c in SRC_COLS) and any(c in cols for c in DST_COLS):
        return "edge_list"

    # Heuristic: if first column is a node id and remaining many numeric -> incidence
    if df.width >= 3:
        first = df.get_column(df.columns[0])
        rest_numeric = all(_is_numeric_series(df.get_column(c)) for c in df.columns[1:])
        if not _is_numeric_series(first) and rest_numeric:
            # Could be incidence OR adjacency with labels on rows
            # If square shape (n rows == n numeric columns) -> adjacency
            if df.height == (df.width - 1):
                return "adjacency"
            return "incidence"

    # Square, mostly numeric -> adjacency (no explicit row label)
    if df.height == df.width and all(_is_numeric_series(df.get_column(c)) for c in df.columns):
        return "adjacency"

    # Fallback
    return "edge_list"


# ---------------------------
# Public API
# ---------------------------

def load_csv_to_graph(
    path: str,
    *,
    graph: Optional["IncidenceGraph"] = None,
    schema: str = "auto",
    default_layer: Optional[str] = None,
    default_directed: Optional[bool] = None,
    default_weight: float = 1.0,
    infer_schema_length: int = 10000,
    encoding: Optional[str] = None,
    null_values: Optional[List[str]] = None,
    low_memory: bool = True,
    **kwargs: Any,
):
    """
    Load a CSV and construct/augment an IncidenceGraph.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    graph : IncidenceGraph or None, optional
        If provided, mutate this graph; otherwise create a new IncidenceGraph using
        `IncidenceGraph(**kwargs)`.
    schema : {'auto','edge_list','hyperedge','incidence','adjacency','lil'}, default 'auto'
        Parsing mode. 'auto' tries to infer the schema from columns and types.
    default_layer : str or None, optional
        Layer to register nodes/edges when none is specified in the data.
    default_directed : bool or None, optional
        Default directedness for binary edges when not implied by data.
    default_weight : float, default 1.0
        Default weight when not specified.
    infer_schema_length : int, default 10000
        Row count Polars uses to infer column types.
    encoding : str or None, optional
        File encoding override.
    null_values : list[str] or None, optional
        Additional strings to interpret as nulls.
    low_memory : bool, default True
        Pass to Polars read_csv for balanced memory usage.
    **kwargs : Any
        Passed to IncidenceGraph constructor if `graph` is None.

    Returns
    -------
    IncidenceGraph
        The populated graph instance.

    Raises
    ------
    RuntimeError
        If no IncidenceGraph can be constructed or imported.
    ValueError
        If schema is unknown or parsing fails.
    """
    df = pl.read_csv(
        path,
        infer_schema_length=infer_schema_length,
        encoding=encoding,
        null_values=null_values,
        low_memory=low_memory,
    )
    return from_dataframe(
        df,
        graph=graph,
        schema=schema,
        default_layer=default_layer,
        default_directed=default_directed,
        default_weight=default_weight,
        **kwargs,
    )


def from_dataframe(
    df: pl.DataFrame,
    *,
    graph: Optional["IncidenceGraph"] = None,
    schema: str = "auto",
    default_layer: Optional[str] = None,
    default_directed: Optional[bool] = None,
    default_weight: float = 1.0,
    **kwargs: Any,
):
    """
    Build/augment an IncidenceGraph from a Polars DataFrame.

    Parameters
    ----------
    df : polars.DataFrame
        Input table parsed from CSV.
    graph : IncidenceGraph or None, optional
        If provided, mutate this graph; otherwise create a new IncidenceGraph using
        `IncidenceGraph(**kwargs)`.
    schema : {'auto','edge_list','hyperedge','incidence','adjacency','lil'}, default 'auto'
        Parsing mode. 'auto' tries to infer the schema.
    default_layer : str or None, optional
        Fallback layer if no layer is specified in the data.
    default_directed : bool or None, optional
        Default directedness for binary edges when not implied by data.
    default_weight : float, default 1.0
        Weight to use when no explicit weight is present.

    Returns
    -------
    IncidenceGraph
        The populated graph instance.
    """
    G = graph
    if G is None:
        if IncidenceGraph is None:
            raise RuntimeError("IncidenceGraph class not importable; pass an instance via `graph=`.")
        G = IncidenceGraph(**kwargs)  # type: ignore

    mode = schema.lower().strip()
    if mode == "auto":
        mode = _detect_schema(df)

    if mode == "edge_list":
        _ingest_edge_list(df, G, default_layer, default_directed, default_weight)
    elif mode == "hyperedge":
        _ingest_hyperedge(df, G, default_layer, default_weight)
    elif mode == "incidence":
        _ingest_incidence(df, G, default_layer, default_weight)
    elif mode == "adjacency":
        _ingest_adjacency(df, G, default_layer, default_directed, default_weight)
    elif mode == "lil":
        _ingest_lil(df, G, default_layer, default_directed, default_weight)
    else:
        raise ValueError(f"Unknown schema: {schema}")

    return G


# ---------------------------
# Ingestors
# ---------------------------

def _ingest_edge_list(
    df: pl.DataFrame,
    G,
    default_layer: Optional[str],
    default_directed: Optional[bool],
    default_weight: float,
):
    """Parse edge-list-like tables (incl. COO/DOK)."""
    src = _pick_first(df, SRC_COLS)
    dst = _pick_first(df, DST_COLS)

    # COO/DOK triples: map row/col->src/dst, value->weight
    if src is None or dst is None:
        rcol = _pick_first(df, ROW_COLS)
        ccol = _pick_first(df, COL_COLS)
        vcol = _pick_first(df, VAL_COLS)
        if rcol and ccol:
            src, dst = rcol, ccol
            # if weight exists, use it; else default
            wcol = vcol
        else:
            raise ValueError("Edge list ingest: could not find source/target columns.")
    else:
        wcol = _pick_first(df, WGT_COLS)

    dcol = _pick_first(df, DIR_COLS)
    lcol = _pick_first(df, LAYER_COLS)
    ecol = _pick_first(df, EDGE_ID_COLS)

    reserved_now = {src, dst, wcol, dcol, lcol, ecol}
    attrs_cols = _attr_columns(df, [c for c in reserved_now if c])

    for row in df.iter_rows(named=True):
        u = _norm(row[src])
        v = _norm(row[dst])
        if not u or not v:
            continue
        if dcol:
            directed = _truthy(row[dcol])
        else:
            directed = default_directed
        w = float(row[wcol]) if (wcol and row[wcol] is not None and str(row[wcol]).strip() != "") else default_weight
        layers = _split_layers(row[lcol]) if lcol else ([] if default_layer is None else [default_layer])

        # attributes for the edge (pure)
        pure_attrs = {k: row[k] for k in attrs_cols if row[k] is not None}

        # ensure nodes
        G.add_node(u)
        G.add_node(v)

        # create edge per layer (or default)
        if not layers:
            eid = G.add_edge(u, v, directed=directed, weight=w, layer=default_layer, propagate="none", **pure_attrs)
        else:
            eid = None
            for L in layers:
                eid = G.add_edge(u, v, directed=directed, weight=w, layer=L, propagate="none", **pure_attrs)
                # per-layer override columns like weight:Layer
                for c in df.columns:
                    if c.lower().startswith("weight:"):
                        _, _, suffix = c.partition(":")
                        if suffix == L and row[c] is not None:
                            try:
                                G.set_edge_layer_attrs(L, eid, weight=float(row[c]))  # type: ignore[arg-type]
                            except Exception:
                                pass
        # explicit edge id mapping if present
        if ecol and eid is not None and row[ecol]:
            # no-op for now (edge ids are internal); could add alias map here if your graph supports it
            pass


def _ingest_hyperedge(
    df: pl.DataFrame,
    G,
    default_layer: Optional[str],
    default_weight: float,
):
    """Parse hyperedge tables (members OR head/tail)."""
    mcol = _pick_first(df, MEMBERS_COLS)
    hcol = _pick_first(df, HEAD_COLS)
    tcol = _pick_first(df, TAIL_COLS)
    wcol = _pick_first(df, WGT_COLS)
    lcol = _pick_first(df, LAYER_COLS)

    attrs_cols = _attr_columns(df, [c for c in [mcol, hcol, tcol, wcol, lcol] if c])

    for row in df.iter_rows(named=True):
        weight = float(row[wcol]) if (wcol and row[wcol] is not None and str(row[wcol]).strip() != "") else default_weight
        layer = _split_layers(row[lcol]) if lcol else ([] if default_layer is None else [default_layer])
        if not layer:
            layer = [default_layer] if default_layer else [None]

        pure_attrs = {k: row[k] for k in attrs_cols if row[k] is not None}

        if mcol:
            members = _split_set(row[mcol])
            for ent in members:
                G.add_node(ent)
            for L in layer:
                G.add_hyperedge(members=members, layer=L, directed=False, weight=weight, **pure_attrs)
        else:
            head = _split_set(row[hcol]) if hcol else set()
            tail = _split_set(row[tcol]) if tcol else set()
            for ent in head | tail:
                G.add_node(ent)
            for L in layer:
                G.add_hyperedge(head=head, tail=tail, layer=L, directed=True, weight=weight, **pure_attrs)


def _ingest_incidence(
    df: pl.DataFrame,
    G,
    default_layer: Optional[str],
    default_weight: float,
):
    """Parse incidence matrices (first col = entity id, remaining numeric edge columns)."""
    idcol = _pick_first(df, NODE_ID_COLS) or df.columns[0]
    if idcol != df.columns[0]:
        df = df.rename({idcol: df.columns[0]})
        idcol = df.columns[0]

    # Create / ensure all nodes
    for nid in df.get_column(idcol).to_list():
        nid_s = _norm(nid)
        if nid_s:
            G.add_node(nid_s)

    # Each remaining column is an edge column; determine kind per column
    for edge_col in df.columns[1:]:
        col = df.get_column(edge_col)
        if not _is_numeric_series(col):
            # skip non-numeric columns (attribute table?)
            continue
        values = col.fill_null(0)
        # collect nonzero indices
        nz_idx: List[int] = [i for i, v in enumerate(values) if float(v or 0) != 0.0]
        if not nz_idx:
            continue
        # map row index -> entity id
        ents = [ _norm(df.get_column(idcol)[i]) for i in nz_idx ]
        vals = [ float(values[i]) for i in nz_idx ]

        pos = [ents[i] for i, x in enumerate(vals) if x > 0]
        neg = [ents[i] for i, x in enumerate(vals) if x < 0]

        # Determine kind
        if len(pos) == 1 and len(neg) == 1:
            # directed binary
            G.add_edge(pos[0], neg[0], directed=True, weight=abs(vals[0]) if len(vals) >= 1 else default_weight, layer=default_layer)
        elif len(pos) == 2 and len(neg) == 0:
            # undirected binary (two + entries)
            G.add_edge(pos[0], pos[1], directed=False, weight=abs(vals[0]) if len(vals) >= 1 else default_weight, layer=default_layer)
        else:
            # hyperedge
            if neg and pos:
                G.add_hyperedge(head=set(pos), tail=set(neg), directed=True, weight=1.0, layer=default_layer)
            else:
                G.add_hyperedge(members=set(pos or neg), directed=False, weight=1.0, layer=default_layer)


def _ingest_adjacency(
    df: pl.DataFrame,
    G,
    default_layer: Optional[str],
    default_directed: Optional[bool],
    default_weight: float,
):
    """Parse adjacency matrices (square). If first column is non-numeric, treat as row labels."""
    # Determine if first column holds row labels
    row_labels: List[str]
    mat_cols: List[str]

    if df.width >= 2 and not _is_numeric_series(df.get_column(df.columns[0])):
        row_labels = [_norm(x) for x in df.get_column(df.columns[0]).to_list()]
        mat_cols = df.columns[1:]
    else:
        row_labels = [str(i) for i in range(df.height)]
        mat_cols = df.columns

    # Ensure all nodes exist
    for nid in row_labels:
        G.add_node(nid)
    for c in mat_cols:
        if not _is_numeric_series(df.get_column(c)):
            raise ValueError("Adjacency ingest: non-numeric column detected in matrix region.")

    # Directedness inference: if symmetric within tolerance and default_directed is None -> undirected
    A = np.asarray(df.select(mat_cols).to_numpy(), dtype=float)
    if len(row_labels) != len(mat_cols):
        raise ValueError("Adjacency ingest: number of rows must equal number of columns.")

    # Map col index -> node id
    col_ids = [ _norm(c) for c in mat_cols ]

    directed = default_directed
    if directed is None:
        sym = np.allclose(A, A.T, atol=1e-12, equal_nan=True)
        directed = not sym

    n = len(row_labels)
    for i in range(n):
        for j in range(n):
            w = A[i, j]
            if not w or (isinstance(w, float) and math.isclose(w, 0.0)):
                continue
            u = row_labels[i]
            v = col_ids[j]
            if not directed:
                # Only use one triangle to avoid duplicates
                if j < i:
                    continue
                if i == j:
                    continue  # ignore self-loops from diagonal in undirected mode
                G.add_edge(u, v, directed=False, weight=float(w), layer=default_layer)
            else:
                if i == j:
                    continue  # ignore self-loops by default; adjust if desired
                G.add_edge(u, v, directed=True, weight=float(w), layer=default_layer)


def _ingest_lil(
    df: pl.DataFrame,
    G,
    default_layer: Optional[str],
    default_directed: Optional[bool],
    default_weight: float,
):
    """Parse LIL-style neighbor tables: one row per node with a neighbors column."""
    idcol = _pick_first(df, NODE_ID_COLS) or df.columns[0]
    ncol = _pick_first(df, NEIGH_COLS)
    wcol = _pick_first(df, WGT_COLS)
    dcol = _pick_first(df, DIR_COLS)
    lcol = _pick_first(df, LAYER_COLS)

    if not ncol:
        raise ValueError("LIL ingest: no neighbors column found.")

    attrs_cols = _attr_columns(df, [idcol, ncol, wcol, dcol, lcol])

    for row in df.iter_rows(named=True):
        u = _norm(row[idcol])
        if not u:
            continue
        G.add_node(u)
        nbrs = _split_set(row[ncol])
        w_default = float(row[wcol]) if (wcol and row[wcol] is not None and str(row[wcol]).strip() != "") else default_weight
        directed = _truthy(row[dcol]) if dcol else default_directed
        layers = _split_layers(row[lcol]) if lcol else ([] if default_layer is None else [default_layer])

        pure_attrs = {k: row[k] for k in attrs_cols if row[k] is not None}

        for v in nbrs:
            if not v:
                continue
            G.add_node(v)
            if not layers:
                G.add_edge(u, v, directed=directed, weight=w_default, layer=default_layer, **pure_attrs)
            else:
                for L in layers:
                    G.add_edge(u, v, directed=directed, weight=w_default, layer=L, **pure_attrs)


# ---------------------------
# Minimal CLI for quick use
# ---------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="CSV -> IncidenceGraph loader (auto-detecting)")
    p.add_argument("path", help="Path to CSV file")
    p.add_argument("--schema", default="auto", choices=["auto","edge_list","hyperedge","incidence","adjacency","lil"], help="Force a schema")
    p.add_argument("--layer", default=None, help="Default layer")
    p.add_argument("--directed", default=None, choices=["true","false"], help="Default directedness for binary edges")
    p.add_argument("--weight", type=float, default=1.0, help="Default weight")
    args = p.parse_args()

    dflt_dir = None if args.directed is None else (args.directed.lower() == "true")

    if IncidenceGraph is None:
        raise SystemExit("ERROR: IncidenceGraph not importable; run this inside your project or pass a graph=")
    G = load_csv_to_graph(
        args.path,
        schema=args.schema,
        default_layer=args.layer,
        default_directed=dflt_dir,
        default_weight=args.weight,
    )
    # Print a tiny summary
    try:
        edges_df = G.edges_view(layer=args.layer, include_directed=True, resolved_weight=True)
        print(edges_df.head().to_string())
        print(f"Loaded graph with {len(G.entity_index)} entities and {len(G.edge_index)} edges")
    except Exception:
        print("Loaded graph.")
