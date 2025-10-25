import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  # project root
sys.path.insert(0, str(ROOT))
from graphglue.adapters.json_adapter import to_json, from_json, write_ndjson  # JSON (JavaScript Object Notation), NDJSON (Newline-Delimited JSON)
from .helpers import assert_graphs_equal, assert_vertex_attrs_equal, assert_edge_attrs_equal

class TestJSONAdapter:
    """Tests for JSON adapter."""

    def test_simple_round_trip(self, simple_graph, tmpdir_fixture):
        G = simple_graph
        to_json(G, tmpdir_fixture / "graph.json", indent=2)
        G2 = from_json(tmpdir_fixture / "graph.json")
        assert_graphs_equal(G, G2, check_layers=False, check_hyperedges=False)

    def test_complex_round_trip(self, complex_graph, tmpdir_fixture):
        G = complex_graph
        to_json(G, tmpdir_fixture / "graph.json", public_only=False, indent=2)
        G2 = from_json(tmpdir_fixture / "graph.json")
        assert_graphs_equal(G, G2, check_layers=True, check_hyperedges=True)
        assert_vertex_attrs_equal(G, G2, "A")
        assert_edge_attrs_equal(G, G2, "e1", ignore_private=False)

    def test_hyperedges_preservation(self, complex_graph, tmpdir_fixture):
        G = complex_graph
        to_json(G, tmpdir_fixture / "graph.json")
        G2 = from_json(tmpdir_fixture / "graph.json")
        assert "h1" in G2.hyperedge_definitions
        assert "h2" in G2.hyperedge_definitions
        h1 = G2.hyperedge_definitions["h1"]
        assert h1["directed"] is True
        assert set(h1["head"]) == {"B", "C"}
        assert set(h1["tail"]) == {"A"}
        h2 = G2.hyperedge_definitions["h2"]
        assert h2["directed"] is False
        assert set(h2["members"]) == {"A", "D", "E"}

    def test_layers_preservation(self, complex_graph, tmpdir_fixture):
        G = complex_graph
        to_json(G, tmpdir_fixture / "graph.json")
        G2 = from_json(tmpdir_fixture / "graph.json")
        layers = set(G2.list_layers(include_default=False))
        assert "core" in layers and "signaling" in layers and "regulatory" in layers

    def test_public_only_filter(self, complex_graph, tmpdir_fixture):
        G = complex_graph
        G.set_vertex_attrs("A", __secret="hidden")
        G.set_edge_attrs("e1", __internal="private")
        to_json(G, tmpdir_fixture / "graph.json", public_only=True)
        G2 = from_json(tmpdir_fixture / "graph.json")
        attrs_a = G2.get_vertex_attrs("A") or {}
        assert "__secret" not in attrs_a

    def test_ndjson_format(self, complex_graph, tmpdir_fixture):
        G = complex_graph
        write_ndjson(G, tmpdir_fixture / "ndjson_dir")
        assert (tmpdir_fixture / "ndjson_dir" / "nodes.ndjson").exists()
        assert (tmpdir_fixture / "ndjson_dir" / "edges.ndjson").exists()
        assert (tmpdir_fixture / "ndjson_dir" / "hyperedges.ndjson").exists()
        assert (tmpdir_fixture / "ndjson_dir" / "layers.ndjson").exists()
        import json
        with open(tmpdir_fixture / "ndjson_dir" / "nodes.ndjson") as f:
            lines = f.readlines()
            assert len(lines) > 0
            for line in lines:
                obj = json.loads(line)
                assert "id" in obj
