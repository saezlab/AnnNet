import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]  # project root
sys.path.insert(0, str(ROOT))

from graphglue.adapters.GraphDir_Parquet_adapter import (
    read_parquet_graphdir,
    write_parquet_graphdir,
)  # Parquet (columnar storage)
from graphglue.adapters.json_adapter import from_json, to_json  # JSON (JavaScript Object Notation)
from graphglue.adapters.SIF_adapter import from_sif, to_sif  # SIF (Simple Interaction Format)


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_graph_all_adapters(self, tmpdir_fixture):
        from graphglue.core.graph import Graph

        G = Graph()
        to_json(G, tmpdir_fixture / "empty.json")
        G_json = from_json(tmpdir_fixture / "empty.json")
        assert len(list(G_json.vertices())) == 0

        write_parquet_graphdir(G, tmpdir_fixture / "empty_dir")
        G_parquet = read_parquet_graphdir(tmpdir_fixture / "empty_dir")
        assert len(list(G_parquet.vertices())) == 0

        from graphglue.adapters.dataframe_adapter import from_dataframes, to_dataframes

        dfs = to_dataframes(G)
        G_df = from_dataframes(**dfs)
        assert len(list(G_df.vertices())) == 0

    def test_special_characters_in_ids(self, tmpdir_fixture):
        from graphglue.core.graph import Graph

        G = Graph()
        special_ids = [
            "node with spaces",
            "node-with-dashes",
            "node_with_underscores",
            "node.with.dots",
            "α",
            "β",
            "γ",
            "node\twith\ttabs",
        ]
        for vid in special_ids:
            G.add_vertex(vid)
        G.add_edge(special_ids[0], special_ids[1], edge_id="e1")
        G.add_edge("α", "β", edge_id="e2")
        to_json(G, tmpdir_fixture / "special.json")
        G_json = from_json(tmpdir_fixture / "special.json")
        assert set(G.vertices()) == set(G_json.vertices())
        write_parquet_graphdir(G, tmpdir_fixture / "special_dir")
        G_parquet = read_parquet_graphdir(tmpdir_fixture / "special_dir")
        assert set(G.vertices()) == set(G_parquet.vertices())

    def test_large_weights_and_extreme_values(self, tmpdir_fixture):
        from graphglue.core.graph import Graph

        G = Graph()
        G.add_vertex("A")
        G.add_vertex("B")
        G.add_edge("A", "B", edge_id="e1", weight=1e10)
        G.add_edge("A", "B", edge_id="e2", weight=1e-10)
        G.add_edge("A", "B", edge_id="e3", weight=0.0)
        to_json(G, tmpdir_fixture / "extreme.json")
        G_json = from_json(tmpdir_fixture / "extreme.json")
        assert abs(G_json.edge_weights.get("e1", 0) - 1e10) < 1e-6
        assert abs(G_json.edge_weights.get("e2", 0) - 1e-10) < 1e-15
        assert abs(G_json.edge_weights.get("e3", 1) - 0.0) < 1e-10

    def test_self_loops(self, tmpdir_fixture):
        from graphglue.core.graph import Graph

        G = Graph()
        G.add_vertex("A")
        G.add_edge("A", "A", edge_id="loop", weight=2.5)
        to_json(G, tmpdir_fixture / "loop.json")
        G_json = from_json(tmpdir_fixture / "loop.json")
        assert "loop" in G_json.edge_to_idx
        write_parquet_graphdir(G, tmpdir_fixture / "loop_dir")
        G_parquet = read_parquet_graphdir(tmpdir_fixture / "loop_dir")
        assert "loop" in G_parquet.edge_to_idx
        to_sif(
            G,
            tmpdir_fixture / "loop.sif",
            lossless=True,
            manifest_path=tmpdir_fixture / "loop.manifest.json",
        )
        G_sif = from_sif(
            tmpdir_fixture / "loop.sif", manifest=tmpdir_fixture / "loop.manifest.json"
        )
        assert "loop" in G_sif.edge_to_idx

    def test_parallel_edges(self, tmpdir_fixture):
        from graphglue.core.graph import Graph

        G = Graph()
        G.add_vertex("A")
        G.add_vertex("B")
        G.add_edge("A", "B", edge_id="e1", weight=1.0)
        G.add_edge("A", "B", edge_id="e2", weight=2.0)
        G.add_edge("A", "B", edge_id="e3", weight=3.0)
        to_json(G, tmpdir_fixture / "parallel.json")
        G_json = from_json(tmpdir_fixture / "parallel.json")
        assert "e1" in G_json.edge_to_idx
        assert "e2" in G_json.edge_to_idx
        assert "e3" in G_json.edge_to_idx
        assert G_json.number_of_edges() == 3

    def test_null_and_none_handling(self, tmpdir_fixture):
        from graphglue.core.graph import Graph

        G = Graph()
        G.add_vertex("A")
        G.set_vertex_attrs("A", present="value", missing=None, zero=0, empty_string="")
        to_json(G, tmpdir_fixture / "nulls.json")
        G_json = from_json(tmpdir_fixture / "nulls.json")
        attrs = G_json.get_vertex_attrs("A") or {}
        assert attrs.get("present") == "value"
        assert attrs.get("zero") == 0
        assert "missing" not in attrs or attrs.get("missing") is None

    def test_very_large_graph(self, tmpdir_fixture):
        import random

        from graphglue.core.graph import Graph

        G = Graph()
        n_vertices = 1000
        n_edges = 2000
        for i in range(n_vertices):
            G.add_vertex(f"v{i}")
        random.seed(42)
        for i in range(n_edges):
            u = f"v{random.randint(0, n_vertices - 1)}"
            v = f"v{random.randint(0, n_vertices - 1)}"
            G.add_edge(u, v, edge_id=f"e{i}", weight=random.random())
        write_parquet_graphdir(G, tmpdir_fixture / "large_dir")
        G_parquet = read_parquet_graphdir(tmpdir_fixture / "large_dir")
        assert len(list(G_parquet.vertices())) == n_vertices
        assert G_parquet.number_of_edges() == n_edges
