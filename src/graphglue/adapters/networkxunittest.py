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

        # Basic NX assertions
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


if __name__ == "__main__":
    unittest.main()
