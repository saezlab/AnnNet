# test_graph.py
import os
import sys
import unittest

import polars as pl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import warnings

from graphglue.core.graph import Graph

warnings.filterwarnings(
    "ignore",
    message=r"Signature .*numpy\.longdouble.*",
    category=UserWarning,
    module=r"numpy\._core\.getlimits",
)


class TestGraphBasics(unittest.TestCase):
    def setUp(self):
        self.g = Graph(directed=True)  # default directed

    def test_add_vertex_and_attributes(self):
        self.g.add_vertex("v1", color="red", value=3)
        self.g.add_vertex("v2")  # no attrs
        self.assertEqual(self.g.number_of_vertices(), 2)
        # row exists even if no attrs were passed
        self.assertIn("v2", self.g.vertex_attributes.select("vertex_id").to_series().to_list())
        # attribute accessible
        self.assertEqual(self.g.get_attr_vertex("v1", "color"), "red")
        self.assertEqual(self.g.get_vertex_attribute("v1", "value"), 3)

    def test_add_edge_directed_default_and_matrix_signs(self):
        eid = self.g.add_edge("a", "b", weight=2.5, label="eab")
        self.assertTrue(self.g._is_directed_edge(eid))
        self.assertIn(eid, self.g.get_directed_edges())
        # get_edge canonical form
        S, T = self.g.get_edge(self.g.edge_to_idx[eid])
        self.assertEqual(S, frozenset({"a"}))
        self.assertEqual(T, frozenset({"b"}))
        # matrix signs: +w at source, -w at target (directed)
        ai = self.g.entity_to_idx["a"]
        bi = self.g.entity_to_idx["b"]
        col = self.g.edge_to_idx[eid]
        self.assertAlmostEqual(self.g._matrix[ai, col], 2.5, places=7)
        self.assertAlmostEqual(self.g._matrix[bi, col], -2.5, places=7)
        # attribute purity for edges (structural keys stripped)
        self.assertEqual(self.g.get_attr_edge(eid, "label"), "eab")
        self.assertIsNone(self.g.get_attr_edge(eid, "source"))  # structural, not stored

    def test_add_edge_undirected_override(self):
        eid = self.g.add_edge("c", "d", weight=1.0, edge_directed=False)
        self.assertIn(eid, self.g.get_undirected_edges())
        S, T = self.g.get_edge(self.g.edge_to_idx[eid])
        self.assertEqual(S, T)
        self.assertEqual(S, frozenset({"c", "d"}))
        # matrix signs: +w on both endpoints (undirected)
        ci = self.g.entity_to_idx["c"]
        di = self.g.entity_to_idx["d"]
        col = self.g.edge_to_idx[eid]
        self.assertAlmostEqual(self.g._matrix[ci, col], 1.0, places=7)
        self.assertAlmostEqual(self.g._matrix[di, col], 1.0, places=7)

    def test_parallel_edges_and_lookup(self):
        e1 = self.g.add_edge("p", "q", weight=1.0)
        e2 = self.g.add_parallel_edge("p", "q", weight=3.0)
        self.assertNotEqual(e1, e2)
        ids = self.g.get_edge_ids("p", "q")
        self.assertCountEqual(ids, [e1, e2])
        self.assertTrue(self.g.has_edge("p", "q"))
        self.assertTrue(self.g.has_edge("p", "q", edge_id=e1))

    def test_edge_entity_and_vertex_edge_mode(self):
        # Explicitly add an edge-entity, then connect vertex->edgeEntity
        self.g.add_edge_entity("edge_ghost", layer="Lx", kind="meta")
        e = self.g.add_edge("x", "edge_ghost", edge_type="vertex_edge", weight=1.2)
        self.assertIn("edge_ghost", self.g.entity_types)
        self.assertEqual(self.g.entity_types["edge_ghost"], "edge")
        self.assertAlmostEqual(self.g.edge_weights[e], 1.2, places=7)
        # also ensure attributes for edge entity were stored like vertex attrs
        self.assertEqual(self.g.get_attr_vertex("edge_ghost", "kind"), "meta")

    def test_hyperedge_undirected(self):
        hid = self.g.add_hyperedge(members=["h1", "h2", "h3"], weight=2.0, tag="tri")
        self.assertEqual(self.g.edge_kind[hid], "hyper")
        S, T = self.g.get_edge(self.g.edge_to_idx[hid])
        self.assertEqual(S, T)
        self.assertEqual(S, frozenset({"h1", "h2", "h3"}))
        # matrix entries are +2.0 on all three members
        col = self.g.edge_to_idx[hid]
        for v in ["h1", "h2", "h3"]:
            self.assertAlmostEqual(self.g._matrix[self.g.entity_to_idx[v], col], 2.0, places=7)
        # attribute present
        self.assertEqual(self.g.get_attr_edge(hid, "tag"), "tri")

    def test_hyperedge_directed(self):
        hid = self.g.add_hyperedge(head=["s1", "s2"], tail=["t1"], weight=4.0, category="flow")
        self.assertTrue(self.g.edge_directed[hid])
        S, T = self.g.get_edge(self.g.edge_to_idx[hid])
        self.assertEqual(S, frozenset({"s1", "s2"}))
        self.assertEqual(T, frozenset({"t1"}))
        col = self.g.edge_to_idx[hid]
        for v in ["s1", "s2"]:
            self.assertAlmostEqual(self.g._matrix[self.g.entity_to_idx[v], col], 4.0, places=7)
        self.assertAlmostEqual(self.g._matrix[self.g.entity_to_idx["t1"], col], -4.0, places=7)
        self.assertEqual(self.g.get_attr_edge(hid, "category"), "flow")

    def test_layers_and_activation_and_propagation(self):
        # add layers
        self.g.add_layer("L1", purpose="left")
        self.g.add_layer("L2")
        self.g.set_active_layer("L1")
        self.assertEqual(self.g.get_active_layer(), "L1")
        # add some vertices into current layer
        self.g.add_vertex("A")
        self.g.add_vertex("B")
        # switch layer and add C
        self.g.set_active_layer("L2")
        self.g.add_vertex("C")
        # add edge with propagate=shared (only layers that have both endpoints A,B -> L1)
        e1 = self.g.add_edge(
            "A", "B", layer="L2", propagate="shared"
        )  # placed in L2, but should propagate to L1?
        # L1 has both A,B so edge should be present in L1 as well
        self.assertIn(e1, self.g._layers["L1"]["edges"])
        self.assertIn(e1, self.g._layers["L2"]["edges"])
        # add edge with propagate=all for A-C (A in L1, C in L2) -> should appear in both and pull missing endpoint
        e2 = self.g.add_edge("A", "C", layer="L2", propagate="all")
        self.assertIn(e2, self.g._layers["L1"]["edges"])
        self.assertIn(e2, self.g._layers["L2"]["edges"])
        self.assertIn("C", self.g._layers["L1"]["vertices"])  # pulled across
        self.assertIn("A", self.g._layers["L2"]["vertices"])  # pulled across

    def test_set_and_get_layer_attrs(self):
        self.g.add_layer("Geo", region="EMEA")
        self.assertEqual(self.g.get_layer_attr("Geo", "region"), "EMEA")
        # upsert to new dtype
        self.g.set_layer_attrs("Geo", region="APAC")
        self.assertEqual(self.g.get_layer_attr("Geo", "region"), "APAC")

    def test_per_layer_weight_and_effective_weight(self):
        # ensure the layer exists first
        self.g.add_layer("Lw")
        # create the edge inside layer "Lw" so per-layer attrs apply
        eid = self.g.add_edge("u", "v", weight=5.0, layer="Lw")
        # override via edge_layer_attributes table using the EDGE ID (string)
        self.g.set_edge_layer_attrs("Lw", eid, weight=1.25, note="downweighted")
        # effective weight in Lw reflects the override
        self.assertAlmostEqual(self.g.get_effective_edge_weight(eid, layer="Lw"), 1.25, places=7)
        # asking for a non-existent layer should fall back to the global weight
        self.assertAlmostEqual(
            self.g.get_effective_edge_weight(eid, layer="NonExistent"), 5.0, places=7
        )

    def test_incident_edges(self):
        e1 = self.g.add_edge("i1", "i2", weight=1)
        e2 = self.g.add_edge("i2", "i3", weight=1)
        # undirected also counts on both sides
        e3 = self.g.add_edge("i2", "i4", weight=1, edge_directed=False)
        inc = self.g.incident_edges("i2")
        ids = {self.g.idx_to_edge[j] for j in inc}
        self.assertSetEqual(ids, {e1, e2, e3})

    def test_remove_edge_then_vertex(self):
        e = self.g.add_edge("r1", "r2", weight=1.0, tag="tmp")
        self.g.remove_edge(e)
        self.assertNotIn(e, self.g.edge_to_idx)
        # removing a vertex also removes incident edges
        e2 = self.g.add_edge("r1", "r3", weight=2.0)
        self.g.remove_vertex("r1")
        self.assertNotIn("r1", self.g.entity_to_idx)
        self.assertNotIn(e2, self.g.edge_to_idx)

    def test_remove_layer_and_default_layer_guard(self):
        self.g.add_layer("Z")
        self.g.remove_layer("Z")
        self.assertFalse(self.g.has_layer("Z"))
        with self.assertRaises(ValueError):
            self.g.remove_layer("default")

    def test_audit_attributes(self):
        # create mismatch intentionally
        self.g.add_vertex("a1")
        self.g.add_edge("a1", "a2", weight=1.0)
        # add stray row in vertex_attributes (keep schema identical: only 'vertex_id')
        self.g.vertex_attributes = pl.concat(
            [
                self.g.vertex_attributes,
                pl.DataFrame({"vertex_id": ["ghost"]}),
            ],
            how="vertical",
        )
        audit = self.g.audit_attributes()
        self.assertIn("ghost", audit["extra_vertex_rows"])
        self.assertIsInstance(audit["missing_edge_rows"], list)
        self.assertIsInstance(audit["invalid_edge_layer_rows"], list)

    def test_edges_views_and_counts(self):
        e = self.g.add_edge("x1", "x2", weight=7.0, edge_directed=False)
        self.assertEqual(self.g.number_of_edges(), len(self.g.edges()))
        elist = self.g.edge_list()
        found = [row for row in elist if row[2] == e]
        self.assertEqual(len(found), 1)
        src, tgt, eid, w = found[0]
        self.assertEqual((src, tgt, eid, w), ("x1", "x2", e, 7.0))
        # degree uses non-zero entries count
        d = self.g.degree("x1")
        self.assertGreaterEqual(d, 1)

    def test_update_existing_edge(self):
        e = self.g.add_edge("u1", "u2", weight=2.0, edge_directed=True)
        # Update same edge_id: flip direction flag and endpoints
        self.g.add_edge("u2", "u3", weight=3.5, edge_id=e, edge_directed=False)
        # Now undirected, between u2 and u3, weight 3.5
        S, T = self.g.get_edge(self.g.edge_to_idx[e])
        self.assertEqual(S, T)
        self.assertEqual(S, frozenset({"u2", "u3"}))
        col = self.g.edge_to_idx[e]
        u2i = self.g.entity_to_idx["u2"]
        u3i = self.g.entity_to_idx["u3"]
        self.assertAlmostEqual(self.g._matrix[u2i, col], 3.5, places=7)
        self.assertAlmostEqual(self.g._matrix[u3i, col], 3.5, places=7)


if __name__ == "__main__":
    unittest.main()
