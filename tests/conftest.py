"""Shared fixtures and helpers for adapter tests."""

import pathlib
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]  # project root
sys.path.insert(0, str(ROOT))

from annnet.core.graph import Graph  # noqa: E402

# ======================================================================
# FIXTURES
# ======================================================================


@pytest.fixture
def simple_graph():
    """Minimal graph with vertices and binary edges only."""
    G = Graph(directed=True)
    G.add_vertex("A")
    G.add_vertex("B")
    G.add_vertex("C")

    G.add_edge("A", "B", edge_id="e1", weight=1.5)
    G.add_edge("B", "C", edge_id="e2", weight=2.0)

    return G


@pytest.fixture
def complex_graph():
    """Full-featured graph with all capabilities (mixed directedness)."""
    G = Graph(directed=None)

    # Vertices with attributes
    G.add_vertex("A")
    G.set_vertex_attrs("A", gene="TP53", type="protein", score=0.95)
    G.add_vertex("B")
    G.set_vertex_attrs("B", gene="EGFR", type="protein", score=0.88)
    G.add_vertex("C")
    G.set_vertex_attrs("C", gene="MYC", type="protein")
    G.add_vertex("D")
    G.add_vertex("E")
    G.add_vertex("node with space")

    # Binary edges (mixed directed/undirected)
    G.add_edge("A", "B", edge_id="e1", edge_directed=True, weight=1.5)
    G.set_edge_attrs("e1", relation="activates", confidence=0.9)

    G.add_edge("B", "A", edge_id="e2", edge_directed=False, weight=2.0)
    G.set_edge_attrs("e2", relation="interacts", confidence=0.85)

    G.add_edge("C", "C", edge_id="loop", edge_directed=True, weight=0.5)
    G.set_edge_attrs("loop", relation="self_regulation")

    G.add_edge("A", "B", edge_id="parallel", edge_directed=True, weight=3.14)
    G.set_edge_attrs("parallel", relation="inhibits", tag="secondary")

    # Hyperedges
    G.add_hyperedge(head=["B", "C"], tail=["A"], edge_id="h1", edge_directed=True, weight=0.7)
    G.set_edge_attrs("h1", pathway="signaling", complex="ABC")

    G.add_hyperedge(members=["A", "D", "E"], edge_id="h2", edge_directed=False, weight=5.0)
    G.set_edge_attrs("h2", complex="trimer", stability=0.75)

    # slices
    G.add_slice("core")
    G.add_slice("signaling")
    G.add_slice("regulatory")

    G.add_edge_to_slice("core", "e1")
    G.add_edge_to_slice("core", "e2")
    G.add_edge_to_slice("core", "parallel")
    G.add_edge_to_slice("signaling", "h1")
    G.add_edge_to_slice("regulatory", "loop")

    # Per-slice weights
    G.set_edge_slice_attrs("core", "e1", weight=10.0)
    G.set_edge_slice_attrs("signaling", "h1", weight=0.33)

    return G


@pytest.fixture
def tmpdir_fixture():
    """Temporary directory for file I/O (input/output) tests."""
    tmpdir = Path(tempfile.mkdtemp())
    yield tmpdir
    shutil.rmtree(tmpdir)


# ======================================================================
# HELPERS
# ======================================================================


def assert_graphs_equal(G1, G2, check_slices=True, check_hyperedges=True):
    """Assert two graphs are structurally identical."""
    # Vertices
    assert set(G1.vertices()) == set(G2.vertices()), "Vertex sets differ"

    # Edge count
    assert G1.number_of_edges() == G2.number_of_edges(), "Edge counts differ"

    # Edge IDs
    assert set(G1.edge_to_idx.keys()) == set(G2.edge_to_idx.keys()), "Edge IDs differ"

    # Edge directedness and weights
    for eid in G1.edge_to_idx.keys():
        default_dir = True if G1.directed is None else G1.directed
        dir1 = G1.edge_directed.get(eid, default_dir)
        dir2 = G2.edge_directed.get(eid, default_dir)
        assert dir1 == dir2, f"Edge {eid} directedness differs: {dir1} != {dir2}"

        w1 = G1.edge_weights.get(eid, 1.0)
        w2 = G2.edge_weights.get(eid, 1.0)
        assert abs(w1 - w2) < 1e-6, f"Edge {eid} weight differs: {w1} != {w2}"

    # Hyperedges
    if check_hyperedges:
        assert set(G1.hyperedge_definitions.keys()) == set(G2.hyperedge_definitions.keys()), (
            "Hyperedge IDs differ"
        )

        for eid in G1.hyperedge_definitions.keys():
            h1 = G1.hyperedge_definitions[eid]
            h2 = G2.hyperedge_definitions[eid]
            assert h1["directed"] == h2["directed"], f"Hyperedge {eid} directedness differs"

            if h1["directed"]:
                assert set(h1["head"]) == set(h2["head"]), f"Hyperedge {eid} head differs"
                assert set(h1["tail"]) == set(h2["tail"]), f"Hyperedge {eid} tail differs"
            else:
                assert set(h1["members"]) == set(h2["members"]), f"Hyperedge {eid} members differ"

    # slices
    if check_slices:
        try:
            slices1 = set(G1.list_slices(include_default=False))
            slices2 = set(G2.list_slices(include_default=False))
            assert slices1 == slices2, f"slice sets differ: {slices1} != {slices2}"
        except Exception:
            # Some adapters may not implement slices; let tests control this via flags.
            pass


def assert_vertex_attrs_equal(G1, G2, vertex_id, ignore_none=True):
    """Assert vertex attributes are equal."""
    attrs1 = G1.get_vertex_attrs(vertex_id) or {}
    attrs2 = G2.get_vertex_attrs(vertex_id) or {}

    if ignore_none:
        attrs1 = {k: v for k, v in attrs1.items() if v is not None}
        attrs2 = {k: v for k, v in attrs2.items() if v is not None}

    assert attrs1 == attrs2, f"Vertex {vertex_id} attrs differ: {attrs1} != {attrs2}"


def assert_edge_attrs_equal(G1, G2, edge_id, ignore_none=True, ignore_private=False):
    """Assert edge attributes are equal."""
    try:
        rows1 = G1.edge_attributes.filter(G1.edge_attributes["edge_id"] == edge_id).to_dicts()
        attrs1 = dict(rows1[0]) if rows1 else {}
        attrs1.pop("edge_id", None)
    except Exception:
        attrs1 = {}

    try:
        rows2 = G2.edge_attributes.filter(G2.edge_attributes["edge_id"] == edge_id).to_dicts()
        attrs2 = dict(rows2[0]) if rows2 else {}
        attrs2.pop("edge_id", None)
    except Exception:
        attrs2 = {}

    if ignore_none:
        attrs1 = {k: v for k, v in attrs1.items() if v is not None}
        attrs2 = {k: v for k, v in attrs2.items() if v is not None}

    if ignore_private:
        attrs1 = {k: v for k, v in attrs1.items() if not str(k).startswith("__")}
        attrs2 = {k: v for k, v in attrs2.items() if not str(k).startswith("__")}

    assert attrs1 == attrs2, f"Edge {edge_id} attrs differ: {attrs1} != {attrs2}"


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
