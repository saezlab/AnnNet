import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  # project root
sys.path.insert(0, str(ROOT))
from graphglue.adapters.SIF_adapter import to_sif, from_sif  # SIF (Simple Interaction Format)
from conftest import assert_graphs_equal, assert_vertex_attrs_equal, assert_edge_attrs_equal

class TestSIFAdapter:
    """Tests for SIF adapter."""

    def test_lossy_round_trip(self, simple_graph, tmpdir_fixture):
        G = simple_graph
        to_sif(G, tmpdir_fixture / "net.sif", lossless=False)
        G2 = from_sif(tmpdir_fixture / "net.sif")
        assert set(G2.vertices()) == set(G.vertices())
        assert G2.number_of_edges() == G.number_of_edges()

    def test_lossless_round_trip(self, complex_graph, tmpdir_fixture):
        G = complex_graph
        _, manifest = to_sif(
            G,
            tmpdir_fixture / "net.sif",
            lossless=True,
            manifest_path=tmpdir_fixture / "net.manifest.json"
        )
        G2 = from_sif(
            tmpdir_fixture / "net.sif",
            manifest=tmpdir_fixture / "net.manifest.json"
        )
        assert_graphs_equal(G, G2, check_layers=True, check_hyperedges=True)
        assert G2.edge_directed.get("e1") is True
        assert G2.edge_directed.get("e2") is False
        assert_edge_attrs_equal(G, G2, "e1", ignore_private=False)

    def test_nodes_sidecar(self, complex_graph, tmpdir_fixture):
        G = complex_graph
        to_sif(G, tmpdir_fixture / "net.sif", write_nodes=True)
        assert (tmpdir_fixture / "net.sif.nodes").exists()
        G2 = from_sif(tmpdir_fixture / "net.sif", read_nodes_sidecar=True)
        assert_vertex_attrs_equal(G, G2, "A", ignore_none=True)
        assert_vertex_attrs_equal(G, G2, "B", ignore_none=True)

    def test_custom_relation_attr(self, simple_graph, tmpdir_fixture):
        G = simple_graph
        G.set_edge_attrs("e1", interaction_type="phosphorylation")
        to_sif(
            G,
            tmpdir_fixture / "net.sif",
            relation_attr="interaction_type",
            default_relation="unknown"
        )
        G2 = from_sif(
            tmpdir_fixture / "net.sif",
            relation_attr="interaction_type"
        )
        attrs = G2.edge_attributes.filter(
            G2.edge_attributes["edge_id"].is_in(list(G2.edge_to_idx.keys()))
        ).to_dicts()
        assert any(a.get("interaction_type") == "phosphorylation" for a in attrs)

    def test_mixed_directedness(self, tmpdir_fixture):
        from graphglue.core.graph import Graph
        G = Graph(directed=None)
        G.add_vertex("A"); G.add_vertex("B")
        G.add_edge("A", "B", edge_id="e_dir", edge_directed=True)
        G.add_edge("B", "A", edge_id="e_undir", edge_directed=False)
        to_sif(G, tmpdir_fixture / "net.sif", lossless=True,
               manifest_path=tmpdir_fixture / "net.manifest.json")
        G2 = from_sif(tmpdir_fixture / "net.sif",
                      manifest=tmpdir_fixture / "net.manifest.json")
        assert G2.edge_directed.get("e_dir") is True
        assert G2.edge_directed.get("e_undir") is False
