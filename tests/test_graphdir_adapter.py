import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]  # project root
sys.path.insert(0, str(ROOT))
from annnet.adapters.GraphDir_Parquet_adapter import (
    read_parquet_graphdir,
    write_parquet_graphdir,
)  # Parquet (columnar storage)

from .helpers import assert_edge_attrs_equal, assert_graphs_equal, assert_vertex_attrs_equal


class TestGraphDirAdapter:
    """Tests for GraphDir Parquet adapter."""

    def test_simple_round_trip(self, simple_graph, tmpdir_fixture):
        G = simple_graph
        write_parquet_graphdir(G, tmpdir_fixture / "graphdir")
        assert (tmpdir_fixture / "graphdir" / "vertices.parquet").exists()
        assert (tmpdir_fixture / "graphdir" / "edges.parquet").exists()
        assert (tmpdir_fixture / "graphdir" / "manifest.json").exists()
        G2 = read_parquet_graphdir(tmpdir_fixture / "graphdir")
        assert_graphs_equal(G, G2, check_layers=False, check_hyperedges=False)

    def test_complex_round_trip(self, complex_graph, tmpdir_fixture):
        G = complex_graph
        write_parquet_graphdir(G, tmpdir_fixture / "graphdir")
        G2 = read_parquet_graphdir(tmpdir_fixture / "graphdir")
        assert_graphs_equal(G, G2, check_layers=True, check_hyperedges=True)
        assert_vertex_attrs_equal(G, G2, "A")
        assert_edge_attrs_equal(G, G2, "e1", ignore_private=False)

    def test_hyperedge_preservation(self, complex_graph, tmpdir_fixture):
        G = complex_graph
        write_parquet_graphdir(G, tmpdir_fixture / "graphdir")
        G2 = read_parquet_graphdir(tmpdir_fixture / "graphdir")
        assert set(G.hyperedge_definitions.keys()) == set(G2.hyperedge_definitions.keys())
        for eid in G.hyperedge_definitions.keys():
            h1 = G.hyperedge_definitions[eid]
            h2 = G2.hyperedge_definitions[eid]
            assert h1["directed"] == h2["directed"]
            if h1["directed"]:
                assert set(h1["head"]) == set(h2["head"])
                assert set(h1["tail"]) == set(h2["tail"])
            else:
                assert set(h1["members"]) == set(h2["members"])

    def test_layer_preservation(self, complex_graph, tmpdir_fixture):
        G = complex_graph
        write_parquet_graphdir(G, tmpdir_fixture / "graphdir")
        G2 = read_parquet_graphdir(tmpdir_fixture / "graphdir")
        layers1 = set(G.list_layers(include_default=False))
        layers2 = set(G2.list_layers(include_default=False))
        assert layers1 == layers2
        for lid in layers1:
            edges1 = set(G.get_layer_edges(lid))
            edges2 = set(G2.get_layer_edges(lid))
            assert edges1 == edges2

    def test_compression(self, complex_graph, tmpdir_fixture):
        G = complex_graph
        write_parquet_graphdir(G, tmpdir_fixture / "graphdir")
        vertices_size = (tmpdir_fixture / "graphdir" / "vertices.parquet").stat().st_size
        edges_size = (tmpdir_fixture / "graphdir" / "edges.parquet").stat().st_size
        assert vertices_size > 0
        assert edges_size > 0
