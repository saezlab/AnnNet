# AnnNet Zero-Loss Serialization: Zarr + Parquet

## Design Goals
1. **Zero topology loss**: Preserve exact incidence matrix, all edge types, hyperedges, parallel edges
2. **Complete metadata**: All attributes, layers, history, provenance
3. **Cross-platform**: Works on Windows/Linux/Mac, Python/R/Julia
4. **Incremental updates**: Can append without full rewrite
5. **Cloud support**: S3/GCS/Azure compatible via Zarr
6. **Fast random access**: Chunked storage for large graphs

---

## File Structure

```
graph.annnet/
├── manifest.json                 # root descriptor (format, counts, compression, etc.)
├── structure/
│   ├── incidence.zarr/           # Zarr v3 group holding COO [coordinate list] arrays
│   │   ├── zarr.json             # Zarr v3 group metadata (includes group attributes)
│   │   ├── row/                  # Zarr array (int32) of entity indices (COO row)
│   │   ├── col/                  # Zarr array (int32) of edge indices   (COO col)
│   │   └── data/                 # Zarr array (float32) of weights       (COO data)
│   │   # group attributes include: {"shape": [n_entities, n_edges]}
│   ├── entity_to_idx.parquet     # entity_id → row index
│   ├── idx_to_entity.parquet     # row index → entity_id
│   ├── entity_types.parquet      # entity_id → "vertex" | "edge"
│   ├── edge_to_idx.parquet       # edge_id → column index
│   ├── idx_to_edge.parquet       # column index → edge_id
│   ├── edge_definitions.parquet  # edge_id → (source, target, edge_type) for simple edges
│   ├── edge_weights.parquet      # edge_id → weight
│   ├── edge_directed.parquet     # edge_id → bool | null
│   ├── edge_kind.parquet         # edge_id → "binary" | "hyper"
│   └── hyperedge_definitions.parquet
│       # columns: edge_id, directed(bool), members(List[Utf8]) OR head(List[Utf8]), tail(List[Utf8])
│
├── tables/
│   ├── vertex_attributes.parquet     # vertex-level DF [dataframe]
│   ├── edge_attributes.parquet       # edge-level DF
│   ├── layer_attributes.parquet      # layer metadata
│   └── edge_layer_attributes.parquet # (layer_id, edge_id, weight)
│
├── layers/
│   ├── registry.parquet              # layer_id, name, metadata…
│   ├── vertex_memberships.parquet    # (layer_id, vertex_id)
│   └── edge_memberships.parquet      # (layer_id, edge_id, weight)
│
├── cache/                            # optional materialized views
│   ├── csr.zarr/                     # CSR [compressed sparse row] cache
│   └── csc.zarr/                     # CSC [compressed sparse column] cache
│
├── audit/
│   ├── history.parquet               # operation log (nested payloads stringified to JSON [JavaScript Object Notation])
│   ├── snapshots/                    # optional labeled snapshots
│   └── provenance.json               # creation time, software versions, etc.
│
└── uns/                              # unstructured metadata & results
    ├── graph_attributes.json
    └── results/
```

---

## Manifest Schema (`manifest.json`)

```json
{
  "format": "annnet",
  "version": "1.0.0",
  "created": "2025-10-23T10:30:00Z",
  "annnet_version": "0.1.0",
  "graph_version": 42,
  "directed": true,
  "counts": {
    "vertices": 1000,
    "edges": 5000,
    "entities": 1050,
    "layers": 3,
    "hyperedges": 50
  },
  "layers": ["default", "temporal_2023", "temporal_2024"],
  "active_layer": "default",
  "default_layer": "default",
  "schema_version": "1.0",
  "checksum": "sha256:abcdef...",
  "compression": "zstd",
  "encoding": {
    "zarr": "v3",
    "parquet": "2.0"
  }
}
```

## Advantages

1. **Zero loss**: topology + metadata round-trip exactly
2. **Portable**: Parquet/Zarr are first-class in Python/R/Julia
3. **Incremental**: replace just the parts you touched
4. **Cloud-native**: Zarr stores are compatible with S3/GCS/Azure
5. **Interoperable**: PaParquet works with Pandas/DuckDB/Arrow ecosystems
6. **Compressed**: zstd/lz4 where supported
7. **Chunked**: fast random access on large graphs
8. **Schema evolution**: add new tables without breaking old readers

---


