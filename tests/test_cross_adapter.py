# Cross-adapter comparisons in their own file (multi-adapter by definition).
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]  # project root
sys.path.insert(0, str(ROOT))

import polars as pl  # PL (Polars)

from graphglue.adapters.GraphDir_Parquet_adapter import (
    read_parquet_graphdir,
    write_parquet_graphdir,
)
from graphglue.adapters.GraphML_adapter import (
    from_graphml,
    to_graphml,
)  # GraphML (Graph Markup Language)
from graphglue.adapters.json_adapter import from_json, to_json
from graphglue.adapters.SIF_adapter import from_sif, to_sif  # SIF (Simple Interaction Format)


class TestCrossAdapter:
    """Compare different adapters without bundling them in other files."""

    def test_all_adapters_lossless(self, complex_graph, tmpdir_fixture):
        G = complex_graph
        to_json(G, tmpdir_fixture / "graph.json", public_only=False)
        write_parquet_graphdir(G, tmpdir_fixture / "graphdir")
        to_sif(
            G,
            tmpdir_fixture / "graph.sif",
            lossless=True,
            manifest_path=tmpdir_fixture / "graph.sif.manifest.json",
        )
        to_graphml(G, tmpdir_fixture / "graph.graphml", hyperedge_mode="reify")

        G_json = from_json(tmpdir_fixture / "graph.json")
        G_parquet = read_parquet_graphdir(tmpdir_fixture / "graphdir")
        G_sif = from_sif(
            tmpdir_fixture / "graph.sif", manifest=tmpdir_fixture / "graph.sif.manifest.json"
        )
        G_graphml = from_graphml(tmpdir_fixture / "graph.graphml", hyperedge="reified")

        graphs = [G_json, G_parquet, G_sif, G_graphml]
        for i, G_test in enumerate(graphs):
            assert set(G.vertices()) == set(G_test.vertices()), f"Adapter {i} vertices differ"
            assert G.number_of_edges() == G_test.number_of_edges(), (
                f"Adapter {i} edge count differs"
            )
            assert set(G.hyperedge_definitions.keys()) == set(
                G_test.hyperedge_definitions.keys()
            ), f"Adapter {i} hyperedges differ"

    def test_dataframe_to_all_formats(self, complex_graph, tmpdir_fixture):
        from graphglue.adapters.dataframe_adapter import from_dataframes, to_dataframes

        G = complex_graph
        dfs = to_dataframes(G, include_layers=True, include_hyperedges=True)
        dfs["nodes"].write_parquet(tmpdir_fixture / "nodes.parquet")
        dfs["nodes"].write_csv(tmpdir_fixture / "nodes.csv")
        dfs["edges"].write_parquet(tmpdir_fixture / "edges.parquet")
        dfs["edges"].write_csv(tmpdir_fixture / "edges.csv")

        G_parquet = from_dataframes(
            nodes=pl.read_parquet(tmpdir_fixture / "nodes.parquet"),
            edges=pl.read_parquet(tmpdir_fixture / "edges.parquet"),
            hyperedges=dfs["hyperedges"],
            directed=None,
        )

        G_csv = from_dataframes(
            nodes=pl.read_csv(tmpdir_fixture / "nodes.csv"),
            edges=pl.read_csv(tmpdir_fixture / "edges.csv"),
            hyperedges=dfs["hyperedges"],
            directed=None,
        )

        assert set(G.vertices()) == set(G_parquet.vertices())
        assert set(G.vertices()) == set(G_csv.vertices())
        assert G.number_of_edges() == G_parquet.number_of_edges()
        assert G.number_of_edges() == G_csv.number_of_edges()
