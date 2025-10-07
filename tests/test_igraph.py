# tests/test_adapters.py
import unittest
import warnings
import re

# Silence noisy NumPy longdouble warning seen on some builds
warnings.filterwarnings(
    "ignore",
    message=r"Signature .*numpy\.longdouble.*",
    category=UserWarning,
    module=r"numpy\._core\.getlimits",
)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from graphglue.core.graph import Graph

# Optional deps presence
HAS_IG = True
try:
    import igraph as ig  # noqa: F401
except Exception:
    HAS_IG = False


def _build_graph() -> Graph:
    """Build a realistic test graph using the real Graph class."""
    g = Graph(directed=True)

    # vertices with some attributes
    g.add_vertex("A", label="alpha", kind="src")
    g.add_vertex("B", label="beta")
    g.add_vertex("C", label="gamma", kind="sink")

    # edges: directed, undirected, and a hyperedge
    e1 = g.add_edge("A", "B", weight=2.0, interaction=+1, tag="ab")
    e2 = g.add_edge("B", "C", weight=1.0, edge_directed=False, interaction=-1)
    e3 = g.add_hyperedge(head=["A", "B"], tail=["C"], weight=0.5, interaction=+1)

    # layer + per-layer override for e1
    g.add_layer("Lw", region="EMEA")
    g.set_edge_layer_attrs("Lw", e1, weight=5.0)

    # a second layer with no overrides to test fallback
    g.add_layer("L0")

    # basic sanity
    assert g.number_of_edges() >= 3
    return g


class TestIgraphAdapter(unittest.TestCase):
    @unittest.skipUnless(HAS_IG, "python-igraph not installed")
    def test_to_igraph_export_and_roundtrip(self):
        from graphglue.adapters.igraph import to_igraph, from_igraph  # adapter under test

        g = _build_graph()

        igG, manifest = to_igraph(
            g,
            directed=True,
            hyperedge_mode="skip",   # if supported similarly
            public_only=True
        )

        # --- Export checks
        self.assertGreaterEqual(igG.vcount(), 3)
        self.assertGreaterEqual(igG.ecount(), 2)  # hyperedge skipped
        self.assertIn("weights", manifest)
        self.assertIn("layers", manifest)
        self.assertIn("Lw", manifest["layers"])

        # --- Round-trip back to Graph
        g2 = from_igraph(igG, manifest)
        self.assertEqual(set(g2.vertices()), set(g.vertices()))
        for eid in g.edge_weights:
            self.assertIn(eid, g2.edge_weights)
        self.assertAlmostEqual(
            g2.get_effective_edge_weight(list(manifest["layers"]["Lw"])[0], layer="Lw"),
            5.0, places=7
        )

    @unittest.skipUnless(HAS_IG, "python-igraph not installed")
    def test_to_igraph_labels_and_attrs(self):
        from graphglue.adapters.igraph import to_igraph

        g = _build_graph()
        igG, manifest = to_igraph(g, directed=True, hyperedge_mode="skip", public_only=True)

        # Vertex names present
        self.assertTrue(set(igG.vs["name"]) >= {"A", "B", "C"})
        # Edge count >= 2 (hyperedge skipped)
        self.assertGreaterEqual(igG.ecount(), 2)
        self.assertIn("weights", manifest)


if __name__ == "__main__":
    unittest.main()
