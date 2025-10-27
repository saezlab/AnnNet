"""This module purposefully avoids importing stdlib `csv` and uses Polars for IO.

It ingests a CSV into Graph by auto-detecting common schemas:
- Edge list (including DOK/COO triples and variations)
- Hyperedge table (members column or head/tail sets)
- Incidence matrix (rows=entities, cols=edges, ±w orientation)
- Adjacency matrix (square matrix, weighted/unweighted)
- LIL-style neighbor lists (single column of neighbors)

If auto-detection fails or you want control, pass schema=... explicitly.

Dependencies: polars, numpy, scipy (only if you use sparse helpers), Graph

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
- load_csv_to_graph(path, graph=None, schema="auto", **options) -> Graph
- from_dataframe(df, graph=None, schema="auto", **options) -> Graph

Both will create and return an Graph (or mutate the provided one).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from typing import Any

import numpy as np
import polars as pl

from ..core.graph import Graph

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
vertex_ID_COLS = ["vertex", "vertex_id", "id", "name", "label"]
NEIGH_COLS = ["neighbors", "nbrs", "adj", "adjacency", "neighbors_out", "neighbors_in"]
MEMBERS_COLS = ["members", "verts", "participants"]
HEAD_COLS = ["head", "heads"]
TAIL_COLS = ["tail", "tails"]
ROW_COLS = ["row", "i", "r"]
COL_COLS = ["col", "column", "j", "c"]
VAL_COLS = ["val", "value", "w", "weight"]

RESERVED = set(
    SRC_COLS
    + DST_COLS
    + WGT_COLS
    + DIR_COLS
    + LAYER_COLS
    + EDGE_ID_COLS
    + vertex_ID_COLS
    + NEIGH_COLS
    + MEMBERS_COLS
    + HEAD_COLS
    + TAIL_COLS
    + ROW_COLS
    + COL_COLS
    + VAL_COLS
)


def _norm(s: Any) -> str:
    if s is None:
        return ""
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        return str(s)
    return str(s).strip()


def _truthy(x: Any) -> bool | None:
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


def _split_layers(cell: Any) -> list[str]:
    if cell is None:
        return []
    if isinstance(cell, str):
        cell = cell.strip()
        if not cell:
            return []
        # Try JSON array first
        if (cell.startswith("[") and cell.endswith("]")) or (
            cell.startswith("{") and cell.endswith("}")
        ):
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


def _split_set(cell: Any) -> set[str]:
    if cell is None:
        return set()
    if isinstance(cell, str):
        s = cell.strip()
        if not s:
            return set()
        # JSON array
        if s.startswith("[") and s.endswith("]"):
            try:
                return {_norm(v) for v in json.loads(s)}
            except Exception:
                return {p.strip() for p in _SET_SEP.split(s) if p.strip()}
        return {p.strip() for p in _SET_SEP.split(s) if p.strip()}
    if isinstance(cell, (list, tuple, set)):
        return {_norm(v) for v in cell}
    return {str(cell)}


def _pick_first(df: pl.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k in cols_lower:
            return cols_lower[k]
    return None


def _is_numeric_series(s: pl.Series) -> bool:
    return s.dtype.is_numeric() or s.dtype in (
        pl.Float64,
        pl.Float32,
        pl.Int64,
        pl.Int32,
        pl.Int16,
        pl.Int8,
        pl.UInt64,
        pl.UInt32,
        pl.UInt16,
        pl.UInt8,
    )


def _attr_columns(df: pl.DataFrame, exclude: Iterable[str]) -> list[str]:
    excl = {c.lower() for c in exclude}
    return [c for c in df.columns if c.lower() not in excl]


# ---------------------------
# Schema detection
# ---------------------------


def _detect_schema(df: pl.DataFrame) -> str:
    cols = [c.lower() for c in df.columns]

    # Hyperedge table if we have 'members' OR (head and tail)
    if any(c in cols for c in MEMBERS_COLS) or (
        any(c in cols for c in HEAD_COLS) and any(c in cols for c in TAIL_COLS)
    ):
        return "hyperedge"

    # LIL: neighbors column
    if any(c in cols for c in NEIGH_COLS):
        return "lil"

    # COO/DOK triple variants
    if (
        any(c in cols for c in ROW_COLS)
        and any(c in cols for c in COL_COLS)
        and any(c in cols for c in VAL_COLS)
    ):
        return "edge_list"

    # Classic edge list (src/dst present)
    if any(c in cols for c in SRC_COLS) and any(c in cols for c in DST_COLS):
        return "edge_list"

    # Heuristic: if first column is a vertex id and remaining many numeric -> incidence
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
    graph: Graph | None = None,
    schema: str = "auto",
    default_layer: str | None = None,
    default_directed: bool | None = None,
    default_weight: float = 1.0,
    infer_schema_length: int = 10000,
    encoding: str | None = None,
    null_values: list[str] | None = None,
    low_memory: bool = True,
    **kwargs: Any,
):
    """Load a CSV and construct/augment an Graph.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    graph : Graph or None, optional
        If provided, mutate this graph; otherwise create a new Graph using
        `Graph(**kwargs)`.
    schema : {'auto','edge_list','hyperedge','incidence','adjacency','lil'}, default 'auto'
        Parsing mode. 'auto' tries to infer the schema from columns and types.
    default_layer : str or None, optional
        Layer to register vertices/edges when none is specified in the data.
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
        Passed to Graph constructor if `graph` is None.

    Returns
    -------
    Graph
        The populated graph instance.

    Raises
    ------
    RuntimeError
        If no Graph can be constructed or imported.
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
    graph: Graph | None = None,
    schema: str = "auto",
    default_layer: str | None = None,
    default_directed: bool | None = None,
    default_weight: float = 1.0,
    **kwargs: Any,
):
    """Build/augment an Graph from a Polars DataFrame.

    Parameters
    ----------
    df : polars.DataFrame
        Input table parsed from CSV.
    graph : Graph or None, optional
        If provided, mutate this graph; otherwise create a new Graph using
        `Graph(**kwargs)`.
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
    Graph
        The populated graph instance.

    """
    G = graph
    if G is None:
        if Graph is None:
            raise RuntimeError("Graph class not importable; pass an instance via `graph=`.")
        G = Graph(**kwargs)  # type: ignore

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


def export_edge_list_csv(G, path, layer=None):
    """Export the binary edge subgraph to a CSV [Comma-Separated Values] file.

    Parameters
    ----------
    G : Graph
        Graph instance to export. Must support ``edges_view`` with columns
        compatible with binary endpoints (e.g., 'source', 'target').
    path : str or pathlib.Path
        Output path for the CSV file.
    layer : str, optional
        Restrict the export to a specific layer. If None, all layers are exported.

    Returns
    -------
    None

    Notes
    -----
    - Only binary edges are exported. Hyperedges (edges connecting more than two
      entities) are ignored.
    - Output columns include: 'source', 'target', 'weight', 'directed', and 'layer'.
    - If a weight column does not exist, a default weight of 1.0 is written.
    - If a directedness column is absent, it will be written as ``None``.
    - This format is compatible with ``load_csv_to_graph(schema="edge_list")``.

    """
    df = G.edges_view(layer=layer) if layer is not None else G.edges_view()

    if not isinstance(df, pl.DataFrame):
        df = pl.DataFrame(df)

    cols = {c.lower(): c for c in df.columns}

    # If a 'kind' column exists, drop hyper rows so we only export binary edges.
    kindcol = cols.get("kind")
    if kindcol:
        df = df.filter(pl.col(kindcol).cast(str).str.to_lowercase() != "hyper")

    # Find mandatory source/target columns
    src = next((cols[k] for k in ("source", "src", "u", "from") if k in cols), None)
    dst = next((cols[k] for k in ("target", "dst", "v", "to") if k in cols), None)
    if not (src and dst):
        raise ValueError(
            "No binary endpoints in edges_view; likely hyperedge-only. Use export_hyperedge_csv."
        )

    # Filter out rows lacking endpoints (safety if view is mixed)
    df = df.filter(pl.col(src).is_not_null() & pl.col(dst).is_not_null())

    # Optional columns
    dircol = cols.get("directed") or cols.get("edge_directed")
    # Prefer resolved/global weights if present
    wcol = (
        cols.get("effective_weight")
        or cols.get("global_weight")
        or cols.get("weight")
        or cols.get("w")
    )
    layercol = cols.get("layer")

    n = df.height

    out = pl.DataFrame(
        {
            "source": df[src],
            "target": df[dst],
            "weight": (df[wcol] if wcol else pl.Series("weight", [1.0] * n)),
            "directed": (df[dircol] if dircol else pl.Series("directed", [None] * n)),
            "layer": (
                pl.Series("layer", [layer] * n)
                if layer is not None
                else (df[layercol] if layercol else pl.Series("layer", [None] * n))
            ),
        }
    )

    out.write_csv(path)


def export_hyperedge_csv(G, path, layer=None, directed=None):
    """Export hyperedges from the graph to a CSV [Comma-Separated Values] file.

    Parameters
    ----------
    G : Graph
        Graph instance to export. Must support ``edges_view`` exposing either
        'members' (for undirected hyperedges) or 'head'/'tail' (for directed hyperedges).
    path : str or pathlib.Path
        Output path for the CSV file.
    layer : str, optional
        Restrict the export to a specific layer. If None, all layers are exported.
    directed : bool, optional
        Force treatment of hyperedges as directed or undirected. If None, the function
        attempts to infer directedness from the graph.

    Returns
    -------
    None

    Notes
    -----
    - If the graph exposes a 'members' column, the output will contain one row per
      undirected hyperedge.
    - If 'head' and 'tail' columns are present, the output will contain one row per
      directed hyperedge. If ``directed=False`` is passed, 'head' and 'tail' are merged
      into a 'members' column.
    - A 'weight' column is included if available; otherwise, all weights default to 1.0.
    - A 'layer' column is included if present or if ``layer`` is specified.
    - This format is compatible with ``load_csv_to_graph(schema="hyperedge")``.
    - If the graph does not expose hyperedge columns, a ``ValueError`` is raised.

    """
    df = G.edges_view(layer=layer) if layer is not None else G.edges_view()
    if not isinstance(df, pl.DataFrame):
        df = pl.DataFrame(df)

    cols = {c.lower(): c for c in df.columns}

    # Keep only hyperedges if 'kind' is available.
    kindcol = cols.get("kind")
    if kindcol:
        df = df.filter(pl.col(kindcol).cast(str).str.to_lowercase() == "hyper")

    members = cols.get("members")
    head = cols.get("head")
    tail = cols.get("tail")
    dircol = cols.get("directed") or cols.get("edge_directed")
    # Prefer resolved/global weights if present
    wcol = (
        cols.get("effective_weight")
        or cols.get("global_weight")
        or cols.get("weight")
        or cols.get("w")
    )
    layercol = cols.get("layer")
    n = df.height

    # Helper to coerce values/lists/tuples to pipe-joined strings; leaves strings as-is.
    def _to_pipe_joined(series: pl.Series, name: str) -> pl.Series:
        vals = series.to_list()
        out = []
        for v in vals:
            if v is None:
                out.append(None)
            elif isinstance(v, (list, tuple, set)):
                out.append("|".join(sorted(str(x) for x in v)))
            else:
                s = str(v).strip()
                # Fast path: JSON array like ["a","b"]
                if s.startswith("[") and s.endswith("]"):
                    try:
                        arr = json.loads(s)
                        out.append("|".join(sorted(str(x) for x in arr)))
                        continue
                    except Exception:
                        pass
                # Split on common separators if present
                parts = [p.strip() for p in _SET_SEP.split(s)] if _SET_SEP.search(s) else [s]
                out.append("|".join(sorted(p for p in parts if p)))
        return pl.Series(name, out)

    if members:
        m = _to_pipe_joined(df[members], "members")
        # drop null/empty rows (mixed views might include non-hyper rows)
        mask = m.is_not_null() & (m.str.len_chars() > 0)
        out = pl.DataFrame(
            {
                "members": m.filter(mask),
                "weight": (
                    df[wcol].filter(mask) if wcol else pl.Series("weight", [1.0] * int(mask.sum()))
                ),
                "layer": (
                    pl.Series("layer", [layer] * int(mask.sum()))
                    if layer is not None
                    else (
                        df[layercol].filter(mask)
                        if layercol
                        else pl.Series("layer", [None] * int(mask.sum()))
                    )
                ),
            }
        )
        out.write_csv(path)
        return

    if head and tail:
        is_dir = directed
        if is_dir is None:
            # If head/tail columns exist, default to directed unless explicitly overridden
            is_dir = True

        h = _to_pipe_joined(df[head], "head")
        t = _to_pipe_joined(df[tail], "tail")
        # drop rows where either side is missing
        mask = h.is_not_null() & (h.str.len_chars() > 0) & t.is_not_null() & (t.str.len_chars() > 0)

        if is_dir:
            out = pl.DataFrame(
                {
                    "head": h.filter(mask),
                    "tail": t.filter(mask),
                    "weight": (
                        df[wcol].filter(mask)
                        if wcol
                        else pl.Series("weight", [1.0] * int(mask.sum()))
                    ),
                    "layer": (
                        pl.Series("layer", [layer] * int(mask.sum()))
                        if layer is not None
                        else (
                            df[layercol].filter(mask)
                            if layercol
                            else pl.Series("layer", [None] * int(mask.sum()))
                        )
                    ),
                }
            )
        else:
            # Merge head/tail into undirected 'members'
            merged = pl.Series(
                "members",
                [
                    "|".join(sorted([p for p in (hval.split("|") + tval.split("|")) if p]))
                    for hval, tval in zip(h.filter(mask).to_list(), t.filter(mask).to_list())
                ],
            )
            out = pl.DataFrame(
                {
                    "members": merged,
                    "weight": (
                        df[wcol].filter(mask)
                        if wcol
                        else pl.Series("weight", [1.0] * int(mask.sum()))
                    ),
                    "layer": (
                        pl.Series("layer", [layer] * int(mask.sum()))
                        if layer is not None
                        else (
                            df[layercol].filter(mask)
                            if layercol
                            else pl.Series("layer", [None] * int(mask.sum()))
                        )
                    ),
                }
            )
        out.write_csv(path)
        return

    raise ValueError(
        "edges_view does not expose hyperedge columns; export via incidence or adjust edges_view."
    )


# ---------------------------
# Ingestors
# ---------------------------


def _ingest_edge_list(
    df: pl.DataFrame,
    G,
    default_layer: str | None,
    default_directed: bool | None,
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
        w = (
            float(row[wcol])
            if (wcol and row[wcol] is not None and str(row[wcol]).strip() != "")
            else default_weight
        )
        layers = (
            _split_layers(row[lcol]) if lcol else ([] if default_layer is None else [default_layer])
        )

        # attributes for the edge (pure)
        pure_attrs = {k: row[k] for k in attrs_cols if row[k] is not None}

        # ensure vertices
        G.add_vertex(u)
        G.add_vertex(v)

        # create edge per layer (or default)
        if not layers:
            eid = G.add_edge(
                u,
                v,
                directed=directed,
                weight=w,
                layer=default_layer,
                propagate="none",
                **pure_attrs,
            )
        else:
            eid = None
            for L in layers:
                eid = G.add_edge(
                    u, v, directed=directed, weight=w, layer=L, propagate="none", **pure_attrs
                )
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
    default_layer: str | None,
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
        weight = (
            float(row[wcol])
            if (wcol and row[wcol] is not None and str(row[wcol]).strip() != "")
            else default_weight
        )
        layer = (
            _split_layers(row[lcol]) if lcol else ([] if default_layer is None else [default_layer])
        )
        if not layer:
            layer = [default_layer] if default_layer else [None]

        pure_attrs = {k: row[k] for k in attrs_cols if row[k] is not None}

        if mcol:
            members = _split_set(row[mcol])
            for ent in members:
                G.add_vertex(ent)
            for L in layer:
                G.add_hyperedge(
                    members=members, layer=L, directed=False, weight=weight, **pure_attrs
                )
        else:
            head = _split_set(row[hcol]) if hcol else set()
            tail = _split_set(row[tcol]) if tcol else set()
            for ent in head | tail:
                G.add_vertex(ent)
            for L in layer:
                G.add_hyperedge(
                    head=head, tail=tail, layer=L, directed=True, weight=weight, **pure_attrs
                )


def _ingest_incidence(
    df: pl.DataFrame,
    G,
    default_layer: str | None,
    default_weight: float,
):
    """Parse incidence matrices (first col = entity id, remaining numeric edge columns)."""
    idcol = _pick_first(df, vertex_ID_COLS) or df.columns[0]
    if idcol != df.columns[0]:
        df = df.rename({idcol: df.columns[0]})
        idcol = df.columns[0]

    # Create / ensure all vertices
    for nid in df.get_column(idcol).to_list():
        nid_s = _norm(nid)
        if nid_s:
            G.add_vertex(nid_s)

    # Each remaining column is an edge column; determine kind per column
    for edge_col in df.columns[1:]:
        col = df.get_column(edge_col)
        if not _is_numeric_series(col):
            # skip non-numeric columns (attribute table?)
            continue
        values = col.fill_null(0)
        # collect nonzero indices
        nz_idx: list[int] = [i for i, v in enumerate(values) if float(v or 0) != 0.0]
        if not nz_idx:
            continue
        # map row index -> entity id
        ents = [_norm(df.get_column(idcol)[i]) for i in nz_idx]
        vals = [float(values[i]) for i in nz_idx]

        pos = [ents[i] for i, x in enumerate(vals) if x > 0]
        neg = [ents[i] for i, x in enumerate(vals) if x < 0]

        # Determine kind
        if len(pos) == 1 and len(neg) == 1:
            # directed binary
            G.add_edge(
                pos[0],
                neg[0],
                directed=True,
                weight=abs(vals[0]) if len(vals) >= 1 else default_weight,
                layer=default_layer,
            )
        elif len(pos) == 2 and len(neg) == 0:
            # undirected binary (two + entries)
            G.add_edge(
                pos[0],
                pos[1],
                directed=False,
                weight=abs(vals[0]) if len(vals) >= 1 else default_weight,
                layer=default_layer,
            )
        else:
            # hyperedge
            if neg and pos:
                G.add_hyperedge(
                    head=set(pos), tail=set(neg), directed=True, weight=1.0, layer=default_layer
                )
            else:
                G.add_hyperedge(
                    members=set(pos or neg), directed=False, weight=1.0, layer=default_layer
                )


def _ingest_adjacency(
    df: pl.DataFrame,
    G,
    default_layer: str | None,
    default_directed: bool | None,
    default_weight: float,
):
    """Parse adjacency matrices (square). If first column is non-numeric, treat as row labels."""
    # Determine if first column holds row labels
    row_labels: list[str]
    mat_cols: list[str]

    if df.width >= 2 and not _is_numeric_series(df.get_column(df.columns[0])):
        row_labels = [_norm(x) for x in df.get_column(df.columns[0]).to_list()]
        mat_cols = df.columns[1:]
    else:
        row_labels = [str(i) for i in range(df.height)]
        mat_cols = df.columns

    # Ensure all vertices exist
    for nid in row_labels:
        G.add_vertex(nid)
    for c in mat_cols:
        if not _is_numeric_series(df.get_column(c)):
            raise ValueError("Adjacency ingest: non-numeric column detected in matrix region.")

    # Directedness inference: if symmetric within tolerance and default_directed is None -> undirected
    A = np.asarray(df.select(mat_cols).to_numpy(), dtype=float)
    if len(row_labels) != len(mat_cols):
        raise ValueError("Adjacency ingest: number of rows must equal number of columns.")

    # Map col index -> vertex id
    col_ids = [_norm(c) for c in mat_cols]

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
    default_layer: str | None,
    default_directed: bool | None,
    default_weight: float,
):
    """Parse LIL-style neighbor tables: one row per vertex with a neighbors column."""
    idcol = _pick_first(df, vertex_ID_COLS) or df.columns[0]
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
        G.add_vertex(u)
        nbrs = _split_set(row[ncol])
        w_default = (
            float(row[wcol])
            if (wcol and row[wcol] is not None and str(row[wcol]).strip() != "")
            else default_weight
        )
        directed = _truthy(row[dcol]) if dcol else default_directed
        layers = (
            _split_layers(row[lcol]) if lcol else ([] if default_layer is None else [default_layer])
        )

        pure_attrs = {k: row[k] for k in attrs_cols if row[k] is not None}

        for v in nbrs:
            if not v:
                continue
            G.add_vertex(v)
            if not layers:
                G.add_edge(
                    u, v, directed=directed, weight=w_default, layer=default_layer, **pure_attrs
                )
            else:
                for L in layers:
                    G.add_edge(u, v, directed=directed, weight=w_default, layer=L, **pure_attrs)
