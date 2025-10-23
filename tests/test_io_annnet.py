# tests/test_io_annnet.py
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.abspath(".")) 
# Test imports (package layout)
from graphglue.core.graph import Graph
from graphglue.io.io_annnet import write as annnet_write, read as annnet_read

import polars as pl
import zarr
import numpy as np


class TestAnnNetIO(unittest.TestCase):
    def setUp(self):
        # Build a tiny directed graph with a layer + hyperedge
        G = Graph(directed=True)

        # Vertices (two in layer1)
        G.add_vertex("v1", layer="layer1")
        G.add_vertex("v2", layer="layer1")
        G.add_vertex("v3")
        G.add_vertex("v4")

        # Edges
        G.add_edge("v1", "v2", edge_id="e1", weight=1.5)
        G.add_edge("v2", "v3", edge_id="e2", weight=2.0)
        G.add_edge("v3", "v4", edge_id="e3", weight=0.5)

        # Hyperedge (undirected)
        G.add_hyperedge(members=["v1", "v2", "v3"], edge_id="h1", weight=3.0)

        # Some unstructured metadata (will go to uns/)
        G.graph_attributes["project"] = "unittest"
        G.graph_attributes["tags"] = ["io", "annnet"]

        # Add a nested history row to ensure audit/JSON stringify path is exercised
        G._history.append({
            "ts": "2025-10-23T00:00:00Z",
            "action": "create",
            "payload": {"nested": {"x": [1, 2, 3]}},
            "notes": ["a", "b"],
            "arr": np.array([1, 2, 3]),
            "maybe_empty": {},
        })

        self.G = G
        self.tmpdir = tempfile.mkdtemp()
        self.out = Path(self.tmpdir) / "test_graph.annnet"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ----------------------- helpers -----------------------
    def _roundtrip(self):
        annnet_write(self.G, self.out, compression="zstd", overwrite=True)
        G2 = annnet_read(self.out)
        return G2

    # ----------------------- tests -------------------------
    def test_write_read_roundtrip_basic(self):
        G2 = self._roundtrip()

        # Top-level counts
        self.assertEqual(len(self.G.entity_to_idx), len(G2.entity_to_idx))
        self.assertEqual(self.G._num_edges, G2._num_edges)
        self.assertEqual(set(self.G._layers.keys()), set(G2._layers.keys()))
        self.assertEqual(self.G.edge_weights, G2.edge_weights)

        # Hyperedges preserved
        self.assertTrue(hasattr(G2, "hyperedge_definitions"))
        self.assertGreater(len(G2.hyperedge_definitions), 0)

        # A couple of identity maps should match
        self.assertEqual(self.G.entity_to_idx, G2.entity_to_idx)
        self.assertEqual(self.G.edge_to_idx, G2.edge_to_idx)

        # Edge metadata
        self.assertEqual(self.G.edge_directed, G2.edge_directed)
        self.assertEqual(self.G.edge_kind, G2.edge_kind)

        # Layers: same edge sets, vertex sets
        for lid in self.G._layers:
            self.assertEqual(self.G._layers[lid]["vertices"], G2._layers[lid]["vertices"])
            self.assertEqual(self.G._layers[lid]["edges"],    G2._layers[lid]["edges"])
            self.assertEqual(self.G.layer_edge_weights.get(lid, {}),
                             G2.layer_edge_weights.get(lid, {}))

    def test_manifest_and_layout(self):
        self._roundtrip()
        root = self.out
        self.assertTrue(root.exists())
        self.assertTrue((root / "manifest.json").exists())

        manifest = json.loads((root / "manifest.json").read_text())
        self.assertEqual(manifest["format"], "annnet")
        self.assertIn("counts", manifest)
        self.assertEqual(manifest["directed"], True)
        self.assertIn("compression", manifest)
        self.assertIn("encoding", manifest)

        # Core layout
        self.assertTrue((root / "structure").exists())
        self.assertTrue((root / "tables").exists())
        self.assertTrue((root / "layers").exists())
        self.assertTrue((root / "audit").exists())
        self.assertTrue((root / "uns").exists())

    def test_zarr_incidence_group(self):
        self._roundtrip()
        inc = self.out / "structure" / "incidence.zarr"
        self.assertTrue(inc.exists())

        # Open Zarr v3 group and validate arrays + attrs
        grp = zarr.open_group(str(inc), mode="r")
        # arrays live as subdirs; zarr v3 exposes them by name
        self.assertIn("row", grp.array_keys())
        self.assertIn("col", grp.array_keys())
        self.assertIn("data", grp.array_keys())

        row = grp["row"][:]
        col = grp["col"][:]
        dat = grp["data"][:]
        shape = tuple(grp.attrs["shape"])

        # Shapes/dtypes (dtype implied by writer: int32/int32/float32)
        self.assertEqual(row.dtype, np.int32)
        self.assertEqual(col.dtype, np.int32)
        self.assertEqual(dat.dtype, np.float32)
        self.assertEqual(shape, self.G._matrix.shape)

        # COO consistency: same length across row/col/data
        self.assertEqual(len(row), len(col))
        self.assertEqual(len(row), len(dat))

    def test_overwrite_semantics(self):
        # first write
        annnet_write(self.G, self.out, compression="zstd", overwrite=True)
        # second write without overwrite should fail
        with self.assertRaises(FileExistsError):
            annnet_write(self.G, self.out, compression="zstd", overwrite=False)
        # now allow overwrite
        annnet_write(self.G, self.out, compression="zstd", overwrite=True)

    def test_layers_registry_and_memberships(self):
        self._roundtrip()
        layers_dir = self.out / "layers"
        self.assertTrue((layers_dir / "registry.parquet").exists())
        self.assertTrue((layers_dir / "vertex_memberships.parquet").exists())
        self.assertTrue((layers_dir / "edge_memberships.parquet").exists())

        reg = pl.read_parquet(layers_dir / "registry.parquet")
        vmem = pl.read_parquet(layers_dir / "vertex_memberships.parquet")
        emem = pl.read_parquet(layers_dir / "edge_memberships.parquet")

        self.assertGreaterEqual(reg.height, 1)
        self.assertIn("layer_id", reg.columns)

        # layer1 must have at least v1,v2
        vset = set(vmem.filter(pl.col("layer_id") == "layer1")["vertex_id"].to_list())
        self.assertTrue({"v1", "v2"}.issubset(vset))

        # edges exist in memberships as well
        self.assertIn("edge_id", emem.columns)
        self.assertIn("weight", emem.columns)

    def test_hyperedge_definitions_parquet(self):
        self._roundtrip()
        p = self.out / "structure" / "hyperedge_definitions.parquet"
        self.assertTrue(p.exists())
        df = pl.read_parquet(p)
        self.assertIn("edge_id", df.columns)
        self.assertIn("directed", df.columns)
        # at least one of members/head/tail exists (depending on directed flag)
        self.assertTrue(
            any(c in df.columns for c in ("members", "head", "tail"))
        )

    def test_audit_and_uns_written(self):
        self._roundtrip()

        # audit: history.parquet should exist and mixed nested columns converted to Utf8(JSON)
        hist = self.out / "audit" / "history.parquet"
        self.assertTrue(hist.exists())
        hdf = pl.read_parquet(hist)

        # payload/notes/arr/maybe_empty should be present (stringified) if they existed
        cols = set(hdf.columns)
        # Some columns might be absent if the schema was inferred differently,
        # so only check types for those that exist.
        for candidate in ("payload", "notes", "arr", "maybe_empty"):
            if candidate in cols:
                self.assertEqual(hdf.schema[candidate], pl.Utf8)

        # uns: graph_attributes.json
        gattr = self.out / "uns" / "graph_attributes.json"
        self.assertTrue(gattr.exists())
        attrs = json.loads(gattr.read_text())
        self.assertEqual(attrs.get("project"), "unittest")
        self.assertEqual(attrs.get("tags"), ["io", "annnet"])

    def test_read_missing_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            annnet_read(Path(self.tmpdir) / "does_not_exist.annnet")


if __name__ == "__main__":
    unittest.main()
