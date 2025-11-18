from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy as scipy
import scipy.sparse as sp

if TYPE_CHECKING:
    from ..core.graph import Graph

try:
    _write_cache  # type: ignore[name-defined]
except NameError:

    def _write_cache(*args, **kwargs):  # type: ignore[no-redef]
        raise NotImplementedError(
            "_write_cache() was referenced but is not defined. "
            "Replace this call with the actual writer function for AnnNet IO."
        )


ANNNET_EXT = "graph.annnet"


def write(graph, path: str | Path, *, compression="zstd", overwrite=False):
    """Write graph to disk with zero topology loss.

    Parameters
    ----------
    path : str | Path
        Target directory (e.g., "my_graph.annnet")
    compression : str, default "zstd"
        Compression codec for Zarr/Parquet
    overwrite : bool, default False
        Allow overwriting existing directory

    Notes
    -----
    Creates a self-contained directory with:
    - Zarr arrays for sparse matrices
    - Parquet tables for attributes/metadata
    - JSON for unstructured data

    """
    import json
    from pathlib import Path

    root = Path(path)
    if root.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Set overwrite=True.")

    root.mkdir(parents=True, exist_ok=overwrite)

    # 1. Write manifest
    manifest = {
        "format": "annnet",
        "version": "1.0.0",
        "created": datetime.now(UTC).isoformat(),
        "annnet_version": "0.1.0",
        "graph_version": graph._version,
        "directed": graph.directed,
        "counts": {
            "vertices": sum(1 for t in graph.entity_types.values() if t == "vertex"),
            "edges": graph._num_edges,
            "entities": graph._num_entities,
            "slices": len(graph._slices),
            "hyperedges": sum(1 for k in graph.edge_kind.values() if k == "hyper"),
        },
        "slices": list(graph._slices.keys()),
        "active_slice": graph._current_slice,
        "default_slice": graph._default_slice,
        "compression": compression,
        # make encoding explicit for tests/docs
        "encoding": {"zarr": "v3", "parquet": "2.0"},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # 2. Write structure/ (topology)
    _write_structure(graph, root / "structure", compression)

    # 3. Write tables/ (Polars > Parquet)
    _write_tables(graph, root / "tables", compression)

    # 4. Write slices/
    _write_slices(graph, root / "slices", compression)

    # 5. Write audit/
    _write_audit(graph, root / "audit", compression)

    # 6. Write uns/
    _write_uns(graph, root / "uns")

    # 7. Optional: Write cache/
    if hasattr(graph, "_cached_csr") or hasattr(graph, "_cached_csc"):
        _write_cache(graph, root / "cache", compression)


def _write_structure(graph, path: Path, compression: str):
    """Write sparse incidence matrix + all index mappings."""
    import polars as pl
    import zarr

    path.mkdir(parents=True, exist_ok=True)

    # Convert DOK > COO for efficient storage
    coo = graph._matrix.tocoo()

    # Write incidence matrix as Zarr (chunked, compressed)
    inc_path = path / "incidence.zarr"

    # Zarr v3 compatibility
    import numpy as np

    root = zarr.open_group(str(inc_path), mode="w")

    from zarr.codecs import BloscCname, BloscCodec, BloscShuffle

    codec = BloscCodec(cname=BloscCname.zstd, clevel=5, shuffle=BloscShuffle.shuffle)

    row = np.asarray(coo.row, dtype=np.int32)
    col = np.asarray(coo.col, dtype=np.int32)
    dat = np.asarray(coo.data, dtype=np.float32)

    root.create_array("row", data=row, chunks=(10000,), compressors=[codec])
    root.create_array("col", data=col, chunks=(10000,), compressors=[codec])
    root.create_array("data", data=dat, chunks=(10000,), compressors=[codec])
    root.attrs["shape"] = coo.shape

    # Write all index mappings as Parquet
    def dict_to_parquet(d: dict, filepath: Path, id_name: str, val_name: str):
        import polars as pl

        df = pl.DataFrame({id_name: list(d.keys()), val_name: list(d.values())})
        df.write_parquet(filepath, compression=compression)

    dict_to_parquet(graph.entity_to_idx, path / "entity_to_idx.parquet", "entity_id", "idx")
    dict_to_parquet(graph.idx_to_entity, path / "idx_to_entity.parquet", "idx", "entity_id")
    dict_to_parquet(graph.entity_types, path / "entity_types.parquet", "entity_id", "type")
    dict_to_parquet(graph.edge_to_idx, path / "edge_to_idx.parquet", "edge_id", "idx")
    dict_to_parquet(graph.idx_to_edge, path / "idx_to_edge.parquet", "idx", "edge_id")
    dict_to_parquet(graph.edge_weights, path / "edge_weights.parquet", "edge_id", "weight")
    dict_to_parquet(graph.edge_directed, path / "edge_directed.parquet", "edge_id", "directed")
    dict_to_parquet(graph.edge_kind, path / "edge_kind.parquet", "edge_id", "kind")

    # Edge definitions (tuples > struct column)

    edge_def_df = pl.DataFrame(
        {
            "edge_id": list(graph.edge_definitions.keys()),
            "source": [v[0] for v in graph.edge_definitions.values()],
            "target": [v[1] for v in graph.edge_definitions.values()],
            "edge_type": [v[2] for v in graph.edge_definitions.values()],
        }
    )
    edge_def_df.write_parquet(path / "edge_definitions.parquet", compression=compression)

    # Hyperedge definitions (lists > list column)
    if graph.hyperedge_definitions:
        eids, dirs, mems, heads, tails = [], [], [], [], []

        for eid, h in graph.hyperedge_definitions.items():
            eids.append(eid)
            is_dir = bool(h.get("directed", False))
            dirs.append(is_dir)

            if is_dir:
                # directed hyperedge: store head/tail lists; no members
                heads.append(sorted(map(str, h.get("head", ()))))
                tails.append(sorted(map(str, h.get("tail", ()))))
                mems.append(None)
            else:
                # undirected hyperedge: store members list; no head/tail
                heads.append(None)
                tails.append(None)
                mems.append(sorted(map(str, h.get("members", ()))))

        hyper_df = pl.DataFrame({"edge_id": eids, "directed": dirs}).with_columns(
            pl.Series("members", mems, dtype=pl.List(pl.Utf8)),
            pl.Series("head", heads, dtype=pl.List(pl.Utf8)),
            pl.Series("tail", tails, dtype=pl.List(pl.Utf8)),
        )
        hyper_df.write_parquet(path / "hyperedge_definitions.parquet", compression=compression)


def _write_tables(graph, path: Path, compression: str):
    """Write all Polars DataFrames directly to Parquet."""

    path.mkdir(parents=True, exist_ok=True)

    # Direct write - Polars handles schema preservation
    graph.vertex_attributes.write_parquet(
        path / "vertex_attributes.parquet", compression=compression
    )
    graph.edge_attributes.write_parquet(path / "edge_attributes.parquet", compression=compression)
    graph.slice_attributes.write_parquet(path / "slice_attributes.parquet", compression=compression)
    graph.edge_slice_attributes.write_parquet(
        path / "edge_slice_attributes.parquet", compression=compression
    )


def _write_slices(graph, path: Path, compression: str):
    """Write slice registry and memberships."""
    import json

    import polars as pl

    path.mkdir(parents=True, exist_ok=True)

    # Registry: slice_id > attributes
    registry_data = []
    for slice_id, slice_data in graph._slices.items():
        registry_data.append(
            {"slice_id": slice_id, "attributes": json.dumps(slice_data.get("attributes", {}))}
        )
    pl.DataFrame(registry_data).write_parquet(path / "registry.parquet", compression=compression)

    # Vertex memberships: long format
    vertex_members = []
    for slice_id, slice_data in graph._slices.items():
        for vertex_id in slice_data["vertices"]:
            vertex_members.append({"slice_id": slice_id, "vertex_id": vertex_id})
    pl.DataFrame(vertex_members).write_parquet(
        path / "vertex_memberships.parquet", compression=compression
    )

    # Edge memberships with weights
    edge_members: list[dict] = []
    # Primary: explicit per-slice weights (if present)
    for slice_id, edge_weights in getattr(graph, "slice_edge_weights", {}).items():
        for edge_id, weight in edge_weights.items():
            edge_members.append({"slice_id": slice_id, "edge_id": edge_id, "weight": weight})
    # Fallback: derive from registered slice edges if no explicit weights
    if not edge_members:
        for slice_id, slice_data in graph._slices.items():
            for edge_id in slice_data.get("edges", ()):
                edge_members.append({"slice_id": slice_id, "edge_id": edge_id, "weight": None})
    # Ensure a stable schema even if there are zero rows
    if edge_members:
        em_df = pl.DataFrame(edge_members)
    else:
        em_df = pl.DataFrame(
            {
                "slice_id": pl.Series([], dtype=pl.Utf8),
                "edge_id": pl.Series([], dtype=pl.Utf8),
                "weight": pl.Series([], dtype=pl.Float64),
            }
        )
    em_df.write_parquet(path / "edge_memberships.parquet", compression=compression)


def _write_audit(graph, path: Path, compression: str):
    """Write history, snapshots, provenance."""
    import json
    import sys
    from datetime import datetime
    try:
        from datetime import UTC  # Py3.11+
    except Exception:  # fallback for <3.11
        from datetime import timezone as _tz
        UTC = UTC

    import numpy as np
    import polars as pl
    import scipy

    path.mkdir(parents=True, exist_ok=True)

    # History log
    if graph._history:
        rows = list(graph._history)

        # normalize per-cell to avoid mixed dtypes at DataFrame construction
        def _norm(v):
            if v is None:
                return None
            # numpy scalar -> python scalar
            if isinstance(v, np.generic):
                v = v.item()
            # pl.Series/arrow-ish -> list
            if hasattr(v, "to_list") and callable(getattr(v, "to_list", None)):
                try:
                    v = v.to_list()
                except Exception:
                    return str(v)
            # numpy arrays -> list
            if isinstance(v, np.ndarray):
                v = v.tolist()
            # sets/tuples -> list (stable order for sets)
            if isinstance(v, set):
                try:
                    v = sorted(v)
                except Exception:
                    v = list(v)
            elif isinstance(v, tuple):
                v = list(v)
            # datetimes/dates -> ISO 8601
            try:
                from datetime import date as _dt_date
                from datetime import datetime as _dt_datetime
                if isinstance(v, (_dt_datetime, _dt_date)):
                    return v.isoformat()
            except Exception:
                pass
            # basic scalars pass through; everything else -> JSON string
            if isinstance(v, (int, float, bool, str)):
                return v
            try:
                return json.dumps(v, default=str)
            except Exception:
                return str(v)

        keys = sorted({k for r in rows for k in r.keys()})
        cols = {k: [_norm(r.get(k)) for r in rows] for k in keys}

        # choose stable dtypes per column
        def _dtype(vals):
            nonnull = [x for x in vals if x is not None]
            if nonnull and all(isinstance(x, bool) for x in nonnull):
                return pl.Boolean
            if nonnull and all(isinstance(x, int) for x in nonnull):
                return pl.Int64
            if nonnull and all(isinstance(x, (int, float)) for x in nonnull):
                return pl.Float64
            return pl.Utf8  # mixed or textual

        schema_overrides = {k: _dtype(vs) for k, vs in cols.items()}
        history_df = pl.DataFrame(cols, schema_overrides=schema_overrides, strict=False)

        # detect columns that can hold nested containers/objects (should be rare after _norm)
        def is_nested(col: pl.Series) -> bool:
            sample = col.head(32).to_list()
            for v in sample:
                if isinstance(v, (dict, list, set, tuple)):
                    return True
                if isinstance(v, np.ndarray):
                    return True
                try:
                    if hasattr(v, "to_list") and callable(v.to_list):
                        return True
                except Exception:
                    pass
            return False

        nested_cols = [c for c in history_df.columns if is_nested(history_df[c])]

        # robust JSON encoder for any remaining nested cell values
        def _jsonify_cell(v):
            if v is None:
                return None
            if hasattr(v, "to_list") and callable(getattr(v, "to_list", None)):
                try:
                    v = v.to_list()
                except Exception:
                    v = str(v)
            if isinstance(v, np.ndarray):
                v = v.tolist()
            if isinstance(v, set):
                try:
                    v = sorted(v)
                except Exception:
                    v = list(v)
            elif isinstance(v, tuple):
                v = list(v)
            try:
                return json.dumps(v, default=str)
            except Exception:
                return json.dumps(str(v))

        if nested_cols:
            history_df = history_df.with_columns(
                *[
                    pl.col(c).map_elements(_jsonify_cell, return_dtype=pl.Utf8).alias(c)
                    for c in nested_cols
                ]
            )

        history_df.write_parquet(path / "history.parquet", compression=compression)

    # Provenance
    provenance = {
        "created": datetime.now(UTC).isoformat(),
        "annnet_version": "0.1.0",
        "python_version": sys.version,
        "dependencies": {
            "scipy": scipy.__version__,
            "numpy": np.__version__,
            "polars": pl.__version__,
        },
    }
    (path / "provenance.json").write_text(json.dumps(provenance, indent=2))

    # Snapshots directory (if any)
    (path / "snapshots").mkdir(exist_ok=True)

def _write_uns(graph, path: Path):
    """Write unstructured metadata and results."""
    import json

    path.mkdir(parents=True, exist_ok=True)

    # Graph attributes
    (path / "graph_attributes.json").write_text(
        json.dumps(graph.graph_attributes, indent=2, default=str)
    )

    # Results directory for algorithm outputs
    (path / "results").mkdir(exist_ok=True)


def read(path: str | Path, *, lazy: bool = False) -> Graph:
    """Load graph from disk with zero loss.

    Parameters
    ----------
    path : str | Path
        Path to .annnet directory
    lazy : bool, default False
        If True, delay loading large arrays until accessed

    Returns
    -------
    Graph
        Reconstructed graph with all topology and metadata

    """
    import json
    from pathlib import Path

    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"{path} not found")

    # 1. Read manifest
    manifest = json.loads((root / "manifest.json").read_text())

    # 2. Create empty graph
    from ..core.graph import Graph

    G = Graph(directed=manifest["directed"])
    G._version = manifest["graph_version"]

    # 3. Load structure
    _load_structure(G, root / "structure", lazy=lazy)

    # 4. Load tables
    _load_tables(G, root / "tables")

    # 5. Load slices
    _load_slices(G, root / "slices")

    # 6. Load audit
    _load_audit(G, root / "audit")

    # 7. Load uns
    _load_uns(G, root / "uns")

    # 8. Set active slice
    _current_slice = manifest["active_slice"]
    _default_slice = manifest["default_slice"]

    return G


def _load_structure(graph, path: Path, lazy: bool):
    """Load sparse matrix and index mappings."""
    import polars as pl
    import zarr

    # Load incidence matrix
    try:
        # Zarr v2
        inc_store = zarr.DirectoryStore(str(path / "incidence.zarr"))
        inc_root = zarr.group(store=inc_store)
    except AttributeError:
        # Zarr v3
        inc_root = zarr.open_group(str(path / "incidence.zarr"), mode="r")

    row = inc_root["row"][:]
    col = inc_root["col"][:]
    data = inc_root["data"][:]
    shape = tuple(inc_root.attrs["shape"])

    # Reconstruct as DOK for mutability
    coo = sp.coo_matrix((data, (row, col)), shape=shape, dtype=np.float32)
    graph._matrix = coo.todok()

    # Load index mappings
    def parquet_to_dict(filepath: Path, key_col: str, val_col: str) -> dict:
        df = pl.read_parquet(filepath)
        return dict(zip(df[key_col], df[val_col]))

    graph.entity_to_idx = parquet_to_dict(path / "entity_to_idx.parquet", "entity_id", "idx")
    graph.idx_to_entity = parquet_to_dict(path / "idx_to_entity.parquet", "idx", "entity_id")
    graph.entity_types = parquet_to_dict(path / "entity_types.parquet", "entity_id", "type")
    graph.edge_to_idx = parquet_to_dict(path / "edge_to_idx.parquet", "edge_id", "idx")
    graph.idx_to_edge = parquet_to_dict(path / "idx_to_edge.parquet", "idx", "edge_id")
    graph.edge_weights = parquet_to_dict(path / "edge_weights.parquet", "edge_id", "weight")
    graph.edge_directed = parquet_to_dict(path / "edge_directed.parquet", "edge_id", "directed")
    graph.edge_kind = parquet_to_dict(path / "edge_kind.parquet", "edge_id", "kind")

    # Edge definitions
    edge_def_df = pl.read_parquet(path / "edge_definitions.parquet")
    graph.edge_definitions = {
        row["edge_id"]: (row["source"], row["target"], row["edge_type"])
        for row in edge_def_df.to_dicts()
    }

    # Hyperedges
    hyper_path = path / "hyperedge_definitions.parquet"
    if hyper_path.exists():
        hyper_df = pl.read_parquet(hyper_path)
        graph.hyperedge_definitions = {}
        # Expect columns: edge_id, directed, members, head, tail
        for row in hyper_df.to_dicts():
            eid = row["edge_id"]
            if row.get("directed", False):
                head = row.get("head") or []
                tail = row.get("tail") or []
                graph.hyperedge_definitions[eid] = {
                    "directed": True,
                    "head": head,
                    "tail": tail,
                }
            else:
                members = row.get("members") or []
                graph.hyperedge_definitions[eid] = {
                    "directed": False,
                    "members": members,
                }

    # Update counts
    graph._num_entities = len(graph.entity_to_idx)
    graph._num_edges = len(graph.edge_to_idx)


def _load_tables(graph, path: Path):
    """Load Polars DataFrames."""
    import polars as pl

    graph.vertex_attributes = pl.read_parquet(path / "vertex_attributes.parquet")
    graph.edge_attributes = pl.read_parquet(path / "edge_attributes.parquet")
    graph.slice_attributes = pl.read_parquet(path / "slice_attributes.parquet")
    graph.edge_slice_attributes = pl.read_parquet(path / "edge_slice_attributes.parquet")


def _load_slices(graph, path: Path):
    """Reconstruct slice registry and memberships."""
    import json

    import polars as pl

    # Registry
    registry_df = pl.read_parquet(path / "registry.parquet")
    for row in registry_df.to_dicts():
        slice_id = row["slice_id"]
        attrs = json.loads(row["attributes"])
        graph._slices[slice_id] = {"vertices": set(), "edges": set(), "attributes": attrs}

    # Vertex memberships
    vertex_df = pl.read_parquet(path / "vertex_memberships.parquet")
    for row in vertex_df.to_dicts():
        graph._slices[row["slice_id"]]["vertices"].add(row["vertex_id"])

    # Edge memberships
    edge_df = pl.read_parquet(path / "edge_memberships.parquet")
    for row in edge_df.to_dicts():
        lid = row["slice_id"]
        eid = row["edge_id"]
        graph._slices[lid]["edges"].add(eid)
        # Only set a per-slice weight if it was explicitly stored (not None).
        w = row.get("weight", None)
        if w is not None:
            graph.slice_edge_weights.setdefault(lid, {})[eid] = w


def _load_audit(graph, path: Path):
    """Load history and provenance."""
    import polars as pl

    history_path = path / "history.parquet"
    if history_path.exists():
        history_df = pl.read_parquet(history_path)
        graph._history = history_df.to_dicts()


def _load_uns(graph, path: Path):
    """Load unstructured metadata."""
    import json

    attrs_path = path / "graph_attributes.json"
    if attrs_path.exists():
        graph.graph_attributes = json.loads(attrs_path.read_text())
