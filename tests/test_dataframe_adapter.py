# test_dataframe_adapter.py
import pathlib
import sys

import polars as pl  # PL (Polars)

ROOT = pathlib.Path(__file__).resolve().parents[1]  # project root
sys.path.insert(0, str(ROOT))
from graphglue.adapters.dataframe_adapter import from_dataframes, to_dataframes
from graphglue.core.graph import Graph

from .helpers import assert_edge_attrs_equal, assert_graphs_equal, assert_vertex_attrs_equal


class TestDataFrameAdapter:
    """Tests for Polars DataFrame adapter."""

    def test_simple_round_trip(self, simple_graph):
        G = simple_graph
        dfs = to_dataframes(G, include_layers=False, include_hyperedges=False)
        assert "nodes" in dfs and "edges" in dfs
        assert dfs["nodes"].height == 3
        assert dfs["edges"].height == 2
        G2 = from_dataframes(nodes=dfs["nodes"], edges=dfs["edges"])
        assert_graphs_equal(G, G2, check_layers=False, check_hyperedges=False)

    def test_complex_round_trip(self, complex_graph):
        G = complex_graph
        dfs = to_dataframes(G, include_layers=True, include_hyperedges=True)
        assert all(k in dfs for k in ["nodes", "edges", "hyperedges", "layers", "layer_weights"])
        G2 = from_dataframes(
            nodes=dfs["nodes"],
            edges=dfs["edges"],
            hyperedges=dfs["hyperedges"],
            layers=dfs["layers"],
            layer_weights=dfs["layer_weights"],
            directed=None,
        )
        assert_graphs_equal(G, G2, check_layers=True, check_hyperedges=True)
        assert_vertex_attrs_equal(G, G2, "A")
        assert_edge_attrs_equal(G, G2, "e1")
        assert_edge_attrs_equal(G, G2, "h1")

    def test_exploded_hyperedges(self, complex_graph):
        G = complex_graph
        dfs = to_dataframes(G, explode_hyperedges=True)
        assert dfs["hyperedges"].height > 2
        assert "role" in dfs["hyperedges"].columns
        assert set(dfs["hyperedges"]["role"]) <= {"head", "tail", "member"}
        G2 = from_dataframes(
            nodes=dfs["nodes"],
            edges=dfs["edges"],
            hyperedges=dfs["hyperedges"],
            exploded_hyperedges=True,
        )
        assert "h1" in G2.hyperedge_definitions and "h2" in G2.hyperedge_definitions
        h1 = G2.hyperedge_definitions["h1"]
        assert set(h1["head"]) == {"B", "C"}
        assert set(h1["tail"]) == {"A"}

    def test_public_only_filter(self, complex_graph):
        G = complex_graph
        G.set_vertex_attrs("A", __private="secret")
        G.set_edge_attrs("e1", __internal="hidden")
        dfs = to_dataframes(G, public_only=True)
        node_a = dfs["nodes"].filter(pl.col("vertex_id") == "A").to_dicts()[0]
        assert "__private" not in node_a
        edge_e1 = dfs["edges"].filter(pl.col("edge_id") == "e1").to_dicts()[0]
        assert "__internal" not in edge_e1

    def test_file_io(self, simple_graph, tmpdir_fixture):
        G = simple_graph
        dfs = to_dataframes(G)
        dfs["nodes"].write_parquet(tmpdir_fixture / "nodes.parquet")
        dfs["edges"].write_csv(tmpdir_fixture / "edges.csv")
        nodes_loaded = pl.read_parquet(tmpdir_fixture / "nodes.parquet")
        edges_loaded = pl.read_csv(tmpdir_fixture / "edges.csv")
        G2 = from_dataframes(nodes=nodes_loaded, edges=edges_loaded)
        assert_graphs_equal(G, G2, check_layers=False, check_hyperedges=False)

    def test_empty_graph(self):
        G = Graph()
        dfs = to_dataframes(G)
        assert dfs["nodes"].height == 0
        assert dfs["edges"].height == 0
        assert dfs["hyperedges"].height == 0
        G2 = from_dataframes(**dfs)
        assert G2.number_of_edges() == 0
        assert len(list(G2.vertices())) == 0
