import unittest
import networkx as nx

from graphglue.core import IncidenceAdapter
from graphglue.adapters.networkx import to_nx, from_nx


class TestNetworkXAdapter(unittest.TestCase):

    def setUp(self):
        # Build a test graph with vertices, edges, weights, and layers
        H = IncidenceAdapter()
        H.add_vertices(["A", "B", "C"])
        e1 = H.add_edge({"A"}, {"B"}, weight=2.0)
        e2 = H.add_edge({"B"}, {"C"}, weight=3.0)

        # add a layer and attach edges
        H.deep.add_layer("L1")
        H.deep.add_layer("L2")
        H.deep.add_edge("A", "B", layer="L1", edge_id=H.deep.idx_to_edge[e1], edge_directed=True)
        H.deep.set_layer_edge_weight("L1", H.deep.idx_to_edge[e1], 5.0)

        H.deep.add_edge("B", "C", layer="L2", edge_id=H.deep.idx_to_edge[e2], edge_directed=True)

        self.H = H

    def test_roundtrip(self):
        # Export to NX + manifest
        nxG, manifest = to_nx(self.H, directed=True, hyperedge_mode="skip")

        # NX assertions
        self.assertTrue(isinstance(nxG, nx.DiGraph))
        self.assertIn("A", nxG.nodes)
        self.assertIn(("A", "B"), nxG.edges)

        # Reimport
        H2 = from_nx(nxG, manifest)

        # Compare vertices
        self.assertEqual(set(self.H.V), set(H2.V))

        # Compare number of edges
        self.assertEqual(self.H.num_edges, H2.num_edges)

        # Compare edge weights
        for eid, w in self.H.deep.edge_weights.items():
            self.assertAlmostEqual(w, H2.deep.edge_weights[eid])

        # Compare layers
        for lid in self.H.deep.layers(include_default=True):
            self.assertEqual(
                set(self.H.deep.get_layer_edges(lid)),
                set(H2.deep.get_layer_edges(lid))
            )
    def test_hyperedge_manifest_preserved(self):
        # Build a tiny graph with a hyperedge only
        H = IncidenceAdapter()
        H.add_vertices(["U", "V", "W", "X"])
        # hyperedge: {U:2, V:1} -> {W:4, X:1}
        H.add_edge({"U": 2, "V": 1}, {"W": 4, "X": 1}, label="hv")

        nxG, manifest = to_nx(H, directed=True, hyperedge_mode="skip", public_only=False)

        # NX view may have 0 edges (we skipped hyperedges), but manifest must preserve them
        self.assertTrue(any(v for v in manifest.get("edges", {}).values() if v[-1] == "hyper"))

        # Reimport and ensure we still have at least one edge column (backend represents hyperedges internally)
        H2 = from_nx(nxG, manifest)
        self.assertGreaterEqual(H2.num_edges, 1)

    def test_public_only_filter_strips_internal_attrs(self):
        H = IncidenceAdapter()
        H.add_vertices(["A", "B"])
        H.add_edge({"A"}, {"B"}, weight=1.23, label="x")

        nxG, _ = to_nx(H, directed=True, hyperedge_mode="skip", public_only=True)

        # Internal keys (like __edge_type, __source_attr, __target_attr) should be stripped
        for _, _, d in nxG.edges(data=True):
            self.assertIn("weight", d)
            self.assertIn("label", d)
            self.assertFalse(any(str(k).startswith("__") for k in d.keys()))


if __name__ == "__main__":
    unittest.main()
