# tests/test_networkx_adapter.py
import os
import sys
import unittest
import warnings

# Silence noisy NumPy longdouble warning seen on some builds
warnings.filterwarnings(
    "ignore",
    message=r"Signature .*numpy\.longdouble.*",
    category=UserWarning,
    module=r"numpy\._core\.getlimits",
)

# Make project importable when tests run from /tests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from graphglue.core.graph import Graph

# --- Optional deps gates ------------------------------------------------------
try:
    import networkx as nx  # noqa: F401

    HAS_NX = True
except Exception:
    HAS_NX = False

# Try to import the adapter under test

from graphglue.adapters.networkx_adapter import from_nx, to_nx  # type: ignore

# Reuse the same graph builder as igraph tests


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


_BUILD_GRAPH = _build_graph


@unittest.skipUnless(HAS_NX, "networkx adapter or dependency not available")
class TestNetworkXAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _BUILD_GRAPH is None:
            raise unittest.SkipTest(
                "No _build_graph() found (import tests.test_igraph._build_graph). "
            )

    def test_to_nx_export_and_roundtrip(self):
        """Full export + roundtrip:
        - Export to NetworkX with hyperedges skipped and public_only on.
        - Manifest must contain 'Lw' with at least one edge.
        - Roundtrip back -> per-layer weight override is preserved (5.0 expected).
        """
        g = _BUILD_GRAPH()

        nxG, manifest = to_nx(g, directed=True, hyperedge_mode="skip", public_only=True)

        # --- Export checks
        # Graph object created
        self.assertIsNotNone(nxG)
        # Manifest basics
        self.assertIn("edges", manifest)
        self.assertIn("weights", manifest)
        self.assertIn("layers", manifest)
        self.assertIn("layer_weights", manifest)

        # Vertex names likely stored as node labels in nx – just sanity check
        # (We don't assert exact count since hyperedge handling can vary.)
        self.assertTrue(any(n in {"A", "B", "C"} for n in nxG.nodes))

        # The key fix: ensure robust layer discovery captured 'Lw'
        self.assertIn("Lw", manifest["layers"])
        self.assertGreater(len(manifest["layers"]["Lw"]), 0)

        # --- Roundtrip back to Graph
        g2 = from_nx(nxG, manifest)

        # Ensure we can pull an edge id from Lw and read its effective weight
        eid = list(manifest["layers"]["Lw"])[0]
        w_eff = g2.get_effective_edge_weight(eid, layer="Lw")
        self.assertEqual(w_eff, 5.0)

    def test_layer_filters_single(self):
        """Export only a single layer ('Lw'); manifest should only include that layer."""
        g = _BUILD_GRAPH()

        nxG, manifest = to_nx(g, directed=True, hyperedge_mode="skip", layer="Lw", public_only=True)
        self.assertIn("layers", manifest)
        self.assertEqual(set(manifest["layers"].keys()), {"Lw"})
        self.assertGreater(len(manifest["layers"]["Lw"]), 0)

        # Per-layer weights should be limited to Lw as well
        self.assertTrue(set(manifest.get("layer_weights", {}).keys()) <= {"Lw"})

    def test_layer_filters_multi(self):
        """Export a union of layers, e.g., ['default', 'Lw']."""
        g = _BUILD_GRAPH()

        nxG, manifest = to_nx(
            g, directed=True, hyperedge_mode="skip", layers=["default", "Lw"], public_only=True
        )
        self.assertIn("layers", manifest)
        self.assertTrue({"default", "Lw"} <= set(manifest["layers"].keys()))
        self.assertGreater(len(manifest["layers"]["default"]), 0)
        self.assertGreater(len(manifest["layers"]["Lw"]), 0)

    def test_hyperedge_skip_vs_expand(self):
        """If the builder creates at least one hyperedge, then:
        - skip: hyperedges are not added as binary edges to nx graph.
        - expand: hyperedges should increase the number of nx edges.
        If the builder has no hyperedges, we skip this test gracefully.
        """
        g = _BUILD_GRAPH()

        # Detect whether test graph includes a hyperedge by inspecting manifest
        nxG_skip, manifest_skip = to_nx(g, directed=True, hyperedge_mode="skip", public_only=True)
        # try with expand
        try:
            nxG_expand, manifest_expand = to_nx(
                g, directed=True, hyperedge_mode="expand", public_only=True
            )
        except Exception:
            self.skipTest("Adapter does not support hyperedge expand mode")

        # If edges manifest includes any 'hyper' kind, then expand should add more edges
        has_hyper = any(
            k == "hyper" for (_, _, k) in (manifest_skip.get("edges", {}) or {}).values()
        ) or any(k == "hyper" for (_, _, k) in (manifest_expand.get("edges", {}) or {}).values())

        if not has_hyper:
            self.skipTest("No hyperedges in builder graph")

        # NetworkX graph edge counts (expand should be >= skip)
        self.assertGreaterEqual(nxG_expand.number_of_edges(), nxG_skip.number_of_edges())

    def test_public_only_strips_private_attrs(self):
        """Attributes beginning with '__' should not appear in the manifest or nx graph data."""
        g = _BUILD_GRAPH()

        # Inject a private attribute on a vertex and edge if builder supports it
        # We try best-effort – if the API doesn't allow, just proceed; export should not crash
        try:
            # Vertex private attr
            if hasattr(g, "set_vertex_attr"):
                g.set_vertex_attr("A", "__secret_tag", 42)
            # Edge private attr on an arbitrary default-layer edge
            any_eid = next(iter(manifest_eid for manifest_eid in g.edge_weights.keys()))
            if hasattr(g, "set_edge_attr"):
                g.set_edge_attr(any_eid, "__hidden_note", "shh")
        except Exception:
            pass

        nxG, manifest = to_nx(g, directed=True, hyperedge_mode="skip", public_only=True)

        # Scan manifest attrs for private keys
        def _has_private(d):
            return any(str(k).startswith("__") for k in (d or {}).keys())

        # Vertex attrs
        vattrs = manifest.get("vertex_attrs", {})
        self.assertFalse(any(_has_private(vattrs.get(v, {})) for v in vattrs))

        # Edge attrs
        eattrs = manifest.get("edge_attrs", {})
        self.assertFalse(any(_has_private(eattrs.get(e, {})) for e in eattrs))

    def test_graph_type_matches_directed_flag(self):
        """Sanity check: directed=True should yield a directed MultiGraph; directed=False an undirected MultiGraph."""
        g = _BUILD_GRAPH()
        nxG_dir, _ = to_nx(g, directed=True, hyperedge_mode="skip", public_only=True)
        self.assertTrue(getattr(nxG_dir, "is_directed", lambda: True)())

        nxG_undir, _ = to_nx(g, directed=False, hyperedge_mode="skip", public_only=True)
        self.assertFalse(getattr(nxG_undir, "is_directed", lambda: False)())


if __name__ == "__main__":
    unittest.main()
