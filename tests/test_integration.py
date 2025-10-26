import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]  # project root
sys.path.insert(0, str(ROOT))

from graphglue.adapters.dataframe_adapter import to_dataframes  # DF (DataFrame)
from graphglue.adapters.GraphDir_Parquet_adapter import (
    write_parquet_graphdir,
)  # Parquet (columnar storage)
from graphglue.adapters.SIF_adapter import from_sif, to_sif  # SIF (Simple Interaction Format)


class TestIntegration:
    """End-to-end integration tests."""

    def test_biological_network_workflow(self, tmpdir_fixture):
        from graphglue.core.graph import Graph

        G = Graph(directed=False)
        proteins = ["TP53", "MDM2", "ATM", "CHEK2", "p21"]
        for p in proteins:
            G.add_vertex(p)
            G.set_vertex_attrs(p, type="protein", organism="human")
        interactions = [
            ("TP53", "MDM2", "inhibition"),
            ("MDM2", "TP53", "ubiquitination"),
            ("ATM", "TP53", "phosphorylation"),
            ("TP53", "p21", "activation"),
            ("CHEK2", "TP53", "phosphorylation"),
        ]
        for i, (src, tgt, interaction_type) in enumerate(interactions):
            G.add_edge(src, tgt, edge_id=f"int_{i}", edge_directed=True)
            G.set_edge_attrs(f"int_{i}", interaction_type=interaction_type, confidence=0.9)
        to_sif(G, tmpdir_fixture / "network.sif", relation_attr="interaction_type")
        dfs = to_dataframes(G)
        dfs["edges"].write_csv(tmpdir_fixture / "interactions.csv")
        write_parquet_graphdir(G, tmpdir_fixture / "network_archive")
        assert (tmpdir_fixture / "network.sif").exists()
        assert (tmpdir_fixture / "interactions.csv").exists()
        assert (tmpdir_fixture / "network_archive").exists()
        G_sif = from_sif(tmpdir_fixture / "network.sif")
        assert len(list(G_sif.vertices())) == len(proteins)

    def test_multi_layer_network(self, tmpdir_fixture):
        from graphglue.core.graph import Graph

        G = Graph(directed=True)
        users = ["Alice", "Bob", "Charlie", "David"]
        for u in users:
            G.add_vertex(u)
        G.add_layer("friendship")
        G.add_layer("collaboration")
        G.add_layer("mentorship")
        G.add_edge("Alice", "Bob", edge_id="f1")
        G.add_edge_to_layer("friendship", "f1")
        G.add_edge("Bob", "Charlie", edge_id="f2")
        G.add_edge_to_layer("friendship", "f2")
        G.add_edge("Alice", "Charlie", edge_id="c1")
        G.add_edge_to_layer("collaboration", "c1")
        G.add_edge("Alice", "David", edge_id="m1")
        G.add_edge_to_layer("mentorship", "m1")
        from graphglue.adapters.json_adapter import from_json, to_json

        to_json(G, tmpdir_fixture / "multilayer.json")
        G2 = from_json(tmpdir_fixture / "multilayer.json")
        layers = set(G2.list_layers(include_default=False))
        assert layers == {"friendship", "collaboration", "mentorship"}
        friendship_edges = set(G2.get_layer_edges("friendship"))
        assert "f1" in friendship_edges and "f2" in friendship_edges
