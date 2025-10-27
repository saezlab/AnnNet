import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]  # project root
sys.path.insert(0, str(ROOT))
from annnet.adapters.GraphML_adapter import (
    from_graphml,
    to_graphml,
)  # GraphML (Graph Markup Language)

from .helpers import assert_graphs_equal


class TestGraphMLAdapter:
    """Tests for GraphML adapter (NetworkX wrapper)."""

    def test_simple_round_trip(self, simple_graph, tmpdir_fixture):
        G = simple_graph
        to_graphml(G, tmpdir_fixture / "graph.graphml", hyperedge_mode="skip")
        G2 = from_graphml(tmpdir_fixture / "graph.graphml", hyperedge="none")
        assert_graphs_equal(G, G2, check_layers=False, check_hyperedges=False)

    def test_reified_hyperedges(self, complex_graph, tmpdir_fixture):
        G = complex_graph
        to_graphml(G, tmpdir_fixture / "graph.graphml", hyperedge_mode="reify")
        G2 = from_graphml(tmpdir_fixture / "graph.graphml", hyperedge="reified")
        assert "h1" in G2.hyperedge_definitions
        assert "h2" in G2.hyperedge_definitions

    def test_manifest_sidecar(self, complex_graph, tmpdir_fixture):
        G = complex_graph
        to_graphml(G, tmpdir_fixture / "graph.graphml", hyperedge_mode="reify")
        assert (tmpdir_fixture / "graph.graphml.manifest.json").exists()
        G2 = from_graphml(tmpdir_fixture / "graph.graphml", hyperedge="reified")
        assert_graphs_equal(G, G2, check_layers=True, check_hyperedges=True)

    def test_attribute_type_preservation(self, tmpdir_fixture):
        from annnet.core.graph import Graph

        G = Graph()
        G.add_vertex("A")
        G.set_vertex_attrs("A", string_val="text", int_val=42, float_val=3.14, bool_val=True)
        to_graphml(G, tmpdir_fixture / "graph.graphml")
        G2 = from_graphml(tmpdir_fixture / "graph.graphml")
        attrs = G2.get_vertex_attrs("A")
        assert isinstance(attrs.get("string_val"), str)
        assert isinstance(attrs.get("int_val"), int)
        assert isinstance(attrs.get("float_val"), float)
        assert isinstance(attrs.get("bool_val"), bool)
