import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
import json
import re
import unittest

import polars as pl

from annnet.core.graph import Graph
from annnet.io import csv as csv_io


def _colmap(df: pl.DataFrame):
    return {c.lower(): c for c in df.columns}


_SEP = re.compile(r"[|,; ]+")


def _explode_cell(x):
    """Return a list of strings from a cell that may be a Series/list/JSON/string."""
    if x is None:
        return []
    if isinstance(x, pl.Series):
        x = x.to_list()
    if isinstance(x, (list, tuple, set)):
        return [str(y) for y in x]
    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            return [str(y) for y in (arr if isinstance(arr, (list, tuple)) else [arr])]
        except Exception:
            pass
    # generic split on pipes/commas/semicolons/whitespace
    return [p for p in _SEP.split(s) if p]


class TestCSVIO(unittest.TestCase):
    def setUp(self):
        # Build a small test graph entirely in memory
        self.G = Graph(directed=True)
        self.G.add_vertices(["A", "B", "C"])
        # Binary edges: A->B (directed), B--C (undirected)
        self.e_ab = self.G.add_edge("A", "B", directed=True, weight=1.0, slice="L1", color="red")
        self.e_bc = self.G.add_edge("B", "C", directed=False, weight=2.0, slice="L1", tag="x")
        # Hyperedge on a different slice
        self.h_abc = self.G.add_hyperedge(members={"A", "B", "C"}, weight=3.0, slice="L2", note="h")

    def test_edge_list_roundtrip(self):
        # ---- Export binary edge list to CSV (in-memory) ----
        e_buf = io.StringIO()
        csv_io.export_edge_list_csv(self.G, e_buf, slice=None)  # all slices
        e_buf.seek(0)

        # Read back with Polars and import into a fresh graph
        df_edges = pl.read_csv(e_buf)
        G2 = Graph(directed=True)
        csv_io.from_dataframe(df_edges, graph=G2, schema="edge_list")

        # Validate: only binary edges came through (2); ignore directedness (importer may default True)
        ev = G2.edges_view()
        self.assertEqual(ev.shape[0], 2)

        cols = _colmap(ev)
        src = cols.get("source") or cols.get("src") or cols.get("u") or cols.get("from")
        tgt = cols.get("target") or cols.get("dst") or cols.get("v") or cols.get("to")
        wcol = (
            cols.get("effective_weight")
            or cols.get("global_weight")
            or cols.get("weight")
            or cols.get("w")
            or cols.get("edge_weight")
        )
        self.assertIsNotNone(src)
        self.assertIsNotNone(tgt)
        self.assertIsNotNone(wcol)

        rows3 = [tuple(r) for r in ev.select([src, tgt, wcol]).iter_rows()]
        self.assertIn(("A", "B", 1.0), rows3)
        self.assertIn(("B", "C", 2.0), rows3)

    def test_hyperedge_roundtrip(self):
        # ---- Export hyperedges to CSV (in-memory) ----
        h_buf = io.StringIO()
        csv_io.export_hyperedge_csv(self.G, h_buf)  # includes slice + weight
        h_buf.seek(0)

        # Read back and import into a fresh graph as hyperedge schema
        df_h = pl.read_csv(h_buf)
        G3 = Graph(directed=True)
        csv_io.from_dataframe(df_h, graph=G3, schema="hyperedge")

        # Validate: hyperedge exists, members survived
        ev = G3.edges_view()
        self.assertGreaterEqual(ev.shape[0], 1)
        cols = _colmap(ev)

        kind = cols.get("kind")
        self.assertIsNotNone(kind)
        kind_val = (ev.select(kind).to_series()[0] or "").lower()
        self.assertIn(kind_val, ("hyper", "hyperedge"))

        members = cols.get("members")
        head = cols.get("head")
        tail = cols.get("tail")
        wcol = (
            cols.get("effective_weight")
            or cols.get("global_weight")
            or cols.get("weight")
            or cols.get("w")
        )
        self.assertIsNotNone(wcol)

        if members:
            raw = ev.select(members).to_series()[0]
            parts = set(_explode_cell(raw))
        elif head and tail:
            h_raw = ev.select(head).to_series()[0]
            t_raw = ev.select(tail).to_series()[0]
            parts = set(_explode_cell(h_raw)) | set(_explode_cell(t_raw))
        else:
            self.fail("No hyperedge columns ('members' or 'head'/'tail') found in edges_view().")

        self.assertEqual(parts, {"A", "B", "C"})

    def test_auto_schema_with_load_csv(self):
        # Show that load_csv_to_graph can ingest from a file-like buffer too
        e_buf = io.StringIO()
        csv_io.export_edge_list_csv(self.G, e_buf)
        e_buf.seek(0)

        G4 = Graph(directed=True)
        csv_io.load_csv_to_graph(e_buf, graph=G4, schema="auto")  # auto-detects edge_list

        ev = G4.edges_view()
        self.assertEqual(ev.shape[0], 2)

    def test_bad_schema_rejection(self):
        # Build a bogus CSV lacking source/target/members to ensure a hard failure
        bad_buf = io.StringIO()
        bad_buf.write("foo,bar\n1,2\n3,4\n")
        bad_buf.seek(0)
        df_bad = pl.read_csv(bad_buf)
        with self.assertRaises(Exception):
            csv_io.from_dataframe(df_bad, graph=Graph(directed=True), schema="edge_list")
        with self.assertRaises(Exception):
            csv_io.from_dataframe(df_bad, graph=Graph(directed=True), schema="hyperedge")


if __name__ == "__main__":
    unittest.main()
