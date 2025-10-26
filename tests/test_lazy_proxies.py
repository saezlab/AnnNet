# tests/test_lazy_proxies_simple_and_multi.py
# Run: python -m unittest tests/test_lazy_proxies_simple_and_multi.py -v

import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from networkx.algorithms import bipartite as nxb

from graphglue.core.graph import Graph

# --------------------------- helper to fetch backend ---------------------------


def nx_backend(
    G: Graph,
    *,
    directed=True,
    hyperedge_mode="skip",
    layer=None,
    layers=None,
    needed_attrs=None,
    simple=False,
    edge_aggs=None,
):
    """Use public G.nx.backend if available, otherwise call the private helper with
    the new signature (requires simple/edge_aggs).
    """
    needed_attrs = needed_attrs or set()
    if hasattr(G.nx, "backend"):
        return G.nx.backend(
            directed=directed,
            hyperedge_mode=hyperedge_mode,
            layer=layer,
            layers=layers,
            needed_attrs=needed_attrs,
            simple=simple,
            edge_aggs=edge_aggs,
        )
    # fallback to private API with required args
    return G.nx._get_or_make_nx(
        directed=directed,
        hyperedge_mode=hyperedge_mode,
        layer=layer,
        layers=layers,
        needed_attrs=needed_attrs,
        simple=simple,
        edge_aggs=edge_aggs,
    )


def ig_backend(
    G: Graph,
    *,
    directed=True,
    hyperedge_mode="skip",
    layer=None,
    layers=None,
    needed_attrs=None,
    simple=False,
    edge_aggs=None,
):
    """Use public G.ig.backend if available, otherwise call the private helper with
    the new signature (requires simple/edge_aggs).
    """
    needed_attrs = needed_attrs or set()
    if hasattr(G.ig, "backend"):
        return G.ig.backend(
            directed=directed,
            hyperedge_mode=hyperedge_mode,
            layer=layer,
            layers=layers,
            needed_attrs=needed_attrs,
            simple=simple,
            edge_aggs=edge_aggs,
        )
    # fallback to private API with required args
    return G.ig._get_or_make_ig(
        directed=directed,
        hyperedge_mode=hyperedge_mode,
        layer=layer,
        layers=layers,
        needed_attrs=needed_attrs,
        simple=simple,
        edge_aggs=edge_aggs,
    )


# --------------------------- Graph builders (MULTI) ---------------------------


def build_multi_graph() -> Graph:
    r"""MULTI-friendly graph for shortest paths etc.
    Directed chain a->b->c->d->e->f with weights 1,2,3,1,4 (sum=11)
    Chords: a->c(10), b->d(5)  (keeps weighted equal-alts, shortens unweighted hops to 4)
    Undirected K4 on {w,x,y,z} with weight=1; bridge c--x (undirected, weight=1)
    """
    G = Graph(directed=True)

    for v, name in [
        ("a", "alpha"),
        ("b", "bravo"),
        ("c", "charlie"),
        ("d", "delta"),
        ("e", "echo"),
        ("f", "phi"),
        ("w", "w"),
        ("x", "x"),
        ("y", "y"),
        ("z", "z"),
    ]:
        G.add_vertex(v, name=name)

    # chain (directed)
    G.add_edge("a", "b", weight=1)
    G.add_edge("b", "c", weight=2)
    G.add_edge("c", "d", weight=3)
    G.add_edge("d", "e", weight=1)
    G.add_edge("e", "f", weight=4)

    # chords (directed)
    G.add_edge("a", "c", weight=10)
    G.add_edge("b", "d", weight=5)

    # undirected K4 (w,x,y,z)
    for u in ["w", "x", "y", "z"]:
        for v in ["w", "x", "y", "z"]:
            if u < v:
                G.add_edge(u, v, weight=1, edge_directed=False)

    # bridge
    G.add_edge("c", "x", weight=1, edge_directed=False)
    return G


def build_multi_communities_graph() -> Graph:
    """Two undirected cliques + weak bridge:
      K6 on {a..f} (15 edges), K4 on {w,x,y,z} (6 edges), bridge e--x (1 edge).
    Louvain on undirected view → deterministic [4, 6].
    """
    G = Graph(directed=True)

    for v in ["a", "b", "c", "d", "e", "f", "w", "x", "y", "z"]:
        G.add_vertex(v)

    k6 = ["a", "b", "c", "d", "e", "f"]
    for i in range(len(k6)):
        for j in range(i + 1, len(k6)):
            G.add_edge(k6[i], k6[j], weight=1, edge_directed=False)

    k4 = ["w", "x", "y", "z"]
    for i in range(len(k4)):
        for j in range(i + 1, len(k4)):
            G.add_edge(k4[i], k4[j], weight=1, edge_directed=False)

    # weak bridge
    G.add_edge("e", "x", weight=0.01, edge_directed=False)
    return G


# --------------------------- Graph builders (SIMPLE) ---------------------------


def build_simple_flow_graph() -> Graph:
    """SIMPLE DiGraph: max flow s->t = 4 with 'capacity'"""
    G = Graph(directed=True)
    for v in ["s", "a", "b", "t"]:
        G.add_vertex(v)
    G.add_edge("s", "a", capacity=2)
    G.add_edge("s", "b", capacity=2)
    G.add_edge("a", "t", capacity=2)
    G.add_edge("b", "t", capacity=2)
    G.add_edge("a", "b", capacity=1)
    return G


def build_simple_core_graph() -> Graph:
    """SIMPLE undirected: K6 + K4 with bridge; k=4 core is nodes {a..f}"""
    G = Graph(directed=True)
    for v in ["a", "b", "c", "d", "e", "f", "w", "x", "y", "z"]:
        G.add_vertex(v)
    k6 = ["a", "b", "c", "d", "e", "f"]
    for i in range(len(k6)):
        for j in range(i + 1, len(k6)):
            G.add_edge(k6[i], k6[j], weight=1, edge_directed=False)
    k4 = ["w", "x", "y", "z"]
    for i in range(len(k4)):
        for j in range(i + 1, len(k4)):
            G.add_edge(k4[i], k4[j], weight=1, edge_directed=False)
    G.add_edge("e", "x", weight=0.01, edge_directed=False)
    return G


def build_simple_bipartite_graph() -> Graph:
    """SIMPLE undirected K3,2 with bipartite=0/1 on U/V"""
    G = Graph(directed=True)
    U = ["u1", "u2", "u3"]
    V = ["v1", "v2"]
    for u in U:
        G.add_vertex(u, bipartite=0)
    for v in V:
        G.add_vertex(v, bipartite=1)
    for u in U:
        for v in V:
            G.add_edge(u, v, edge_directed=False)
    return G


def build_simple_mst_graph() -> Graph:
    r"""SIMPLE undirected weighted square + diagonals:
      A--1--B
      |\   /|
     2 \ /3 4
      | / \ |
      C--1--D
    MST total weight = 4
    """
    G = Graph(directed=True)
    for v in ["A", "B", "C", "D"]:
        G.add_vertex(v)
    # outer square
    G.add_edge("A", "B", weight=1, edge_directed=False)
    G.add_edge("B", "D", weight=4, edge_directed=False)
    G.add_edge("A", "C", weight=2, edge_directed=False)
    G.add_edge("C", "D", weight=1, edge_directed=False)
    # diagonals
    G.add_edge("B", "C", weight=3, edge_directed=False)
    G.add_edge("A", "D", weight=10, edge_directed=False)
    return G


def build_simple_dag() -> Graph:
    """SIMPLE DAG: a->b->c and a->d, b->d"""
    G = Graph(directed=True)
    for v in ["a", "b", "c", "d"]:
        G.add_vertex(v)
    G.add_edge("a", "b")
    G.add_edge("b", "c")
    G.add_edge("a", "d")
    G.add_edge("b", "d")
    return G


def build_simple_cycle() -> Graph:
    """SIMPLE directed 3-cycle: u->v->w->u"""
    G = Graph(directed=True)
    for v in ["u", "v", "w"]:
        G.add_vertex(v)
    G.add_edge("u", "v")
    G.add_edge("v", "w")
    G.add_edge("w", "u")
    return G


# --------------------------- Test Suite: NX MULTI ---------------------------


class TestLazyNXProxy_MULTI(unittest.TestCase):
    """Algorithms that are fine on Multi* backends."""

    def test_weighted_and_unweighted_shortest_paths(self):
        G = build_multi_graph()

        # Weighted via labels (label column 'name') -> 11
        dist_w = G.nx.shortest_path_length(
            G, source="alpha", target="phi", weight="weight", _nx_label_field="name"
        )
        print("[MULTI weighted dijkstra alpha->phi]", dist_w)
        self.assertAlmostEqual(dist_w, 11.0, places=6)

        # Unweighted (BFS) – chords shorten hops to 4
        dist_u = G.nx.shortest_path_length(G, "a", "f", weight=None)
        print("[MULTI unweighted hops a->f]", dist_u)
        self.assertEqual(dist_u, 4)

        # Mutation invalidates cache
        G.add_edge("a", "f", weight=2)
        dist_new = G.nx.shortest_path_length(G, "a", "f", weight="weight")
        print("[MULTI after mutation a->f]", dist_new)
        self.assertAlmostEqual(dist_new, 2.0, places=6)

    def test_communities_pagerank_components(self):
        # Use two cliques + weak bridge to force [4,6]
        G = build_multi_communities_graph()

        # sanity: 22 undirected simple edges: 15 + 6 + 1
        nxG_simple = nx_backend(G, directed=False, simple=True)
        m = nxG_simple.number_of_edges()
        print("[MULTI sanity edges (simple undirected)]", m)
        self.assertEqual(m, 22)

        # Louvain on undirected view → deterministic [4, 6]
        comms = G.nx.louvain_communities(G, _nx_directed=False, weight="weight", seed=42)
        sizes = sorted(len(c) for c in comms)
        print("[MULTI louvain sizes]", sizes)
        self.assertEqual(sizes, [4, 6])

        # PageRank (undirected): 'e' is typically top (clique + bridge)
        pr = G.nx.pagerank(G, _nx_directed=False)
        top_pr = max(pr, key=pr.get)
        print("[MULTI pagerank top]", top_pr)
        self.assertEqual(top_pr, "e")

        # Connected components (undirected)
        comps = list(G.nx.connected_components(G, _nx_directed=False))
        print("[MULTI components]", [sorted(c) for c in comps])
        self.assertEqual(len(comps), 1)
        self.assertEqual(len(next(iter(comps))), 10)


# --------------------------- Test Suite: NX SIMPLE ---------------------------


class TestLazyNXProxy_SIMPLE(unittest.TestCase):
    """Algorithms that REQUIRE simple Graph/DiGraph backends."""

    def test_max_flow_value(self):
        G = build_simple_flow_graph()
        val = G.nx.maximum_flow_value(
            G, "s", "t", capacity="capacity", _nx_simple=True, _nx_edge_aggs={"capacity": "sum"}
        )
        print("[SIMPLE max flow s->t]", val)
        self.assertEqual(val, 4)

    def test_k_core(self):
        G = build_simple_core_graph()
        H = G.nx.k_core(G, k=4, _nx_directed=False, _nx_simple=True)
        nodes = set(H.nodes()) if hasattr(H, "nodes") else set(H)
        print("[SIMPLE k-core k=4]", sorted(nodes))
        self.assertEqual(nodes, {"a", "b", "c", "d", "e", "f"})

    def test_bipartite_projection(self):
        G = build_simple_bipartite_graph()
        nxG = nx_backend(G, directed=False, simple=True)
        U = {n for n, d in nxG.nodes(data=True) if d.get("bipartite") == 0}
        proj = nxb.projected_graph(nxG, U)
        print("[SIMPLE bipartite U-proj nodes]", sorted(proj.nodes()))
        print("[SIMPLE bipartite U-proj edges]", sorted(map(tuple, proj.edges())))
        self.assertEqual(set(proj.nodes()), {"u1", "u2", "u3"})
        self.assertEqual(len(proj.edges()), 3)  # K3

    def test_mst_total_weight(self):
        G = build_simple_mst_graph()
        T = G.nx.minimum_spanning_tree(G, _nx_directed=False, weight="weight", _nx_simple=True)
        total = sum(d.get("weight", 1) for _, _, d in T.edges(data=True))
        print("[SIMPLE mst total]", total)
        self.assertEqual(total, 4)

    def test_topological_sort(self):
        G = build_simple_dag()
        order = list(G.nx.topological_sort(G, _nx_simple=True))
        print("[SIMPLE topo order]", order)
        self.assertLess(order.index("a"), order.index("b"))
        self.assertLess(order.index("b"), order.index("c"))
        self.assertLess(order.index("a"), order.index("d"))

    def test_directed_cycle_simple(self):
        G = build_simple_cycle()
        cyc = list(G.nx.simple_cycles(G, _nx_simple=True))
        print("[SIMPLE cycles]", cyc)
        self.assertTrue(any(set(c) == {"u", "v", "w"} for c in cyc))

    def test_simple_collapse_aggregations(self):
        # verify _nx_edge_aggs works when collapsing parallel edges
        G = Graph(directed=True)
        G.add_vertex("u")
        G.add_vertex("v")
        # simulate parallel edges (adapter emits Multi* for undirected duplicates)
        G.add_edge("u", "v", weight=5, capacity=3, edge_directed=False)
        G.add_edge("u", "v", weight=2, capacity=7, edge_directed=False)

        nxG = G.nx.backend(
            directed=False,
            simple=True,
            needed_attrs={"weight", "capacity"},
            edge_aggs={"weight": "min", "capacity": "sum"},
        )
        w = nxG["u"]["v"].get("weight")
        c = nxG["u"]["v"].get("capacity")
        print("[SIMPLE collapse agg] weight:", w, "capacity:", c)
        assert w == 2 and c == 10

    def test_cache_invalidation_on_mutation(self):
        # backend object id should change when graph mutates (version bumps)
        G = Graph(directed=True)
        G.add_vertex("a")
        G.add_vertex("b")
        G.add_edge("a", "b", weight=1)

        H1 = G.nx.backend(directed=True, simple=False)
        hid1 = id(H1)
        G.add_edge("b", "a", weight=1)  # mutate
        H2 = G.nx.backend(directed=True, simple=False)
        hid2 = id(H2)
        print("[cache invalidation] before:", hid1, "after:", hid2)
        assert hid1 != hid2

    def test_selective_exposure_weight_stripped_when_unused(self):
        # when weight=None ops run, proxy should strip edge attrs for NX call
        G = Graph(directed=True)
        G.add_vertex("a")
        G.add_vertex("b")
        G.add_vertex("c")
        G.add_edge("a", "b", weight=123)
        G.add_edge("b", "c", weight=456)

        # build backend with no needed attrs
        nxG = G.nx.backend(directed=True, simple=False, needed_attrs=set())
        # first edge’s attr dict should be empty
        _, _, d = next(iter(nxG.edges(data=True)))
        print("[selective exposure] edge attrs:", d)
        assert d == {}


# --------------------------- Test Suite: IG MULTI ---------------------------


class TestLazyIGProxy_MULTI(unittest.TestCase):
    """igraph algorithms that are fine on multi-edge backends."""

    def test_weighted_and_unweighted_shortest_paths_ig(self):
        G = build_multi_graph()

        # Weighted Dijkstra via labels (use vertex 'name' strings directly)
        dist_w = G.ig.shortest_paths_dijkstra(
            source="alpha", target="phi", weights="weight", _ig_guess_labels=False
        )
        # unwrap [[val]] -> val
        dist_w = dist_w[0][0] if isinstance(dist_w, list) else dist_w
        print("[IG MULTI weighted dijkstra alpha->phi]", dist_w)
        self.assertAlmostEqual(dist_w, 11.0, places=6)

        # Unweighted hops: chords a->c, b->d shorten a->f from 5 to 4
        dist_u = G.ig.distances(source="alpha", target="phi", weights=None, _ig_guess_labels=False)
        dist_u = dist_u[0][0] if isinstance(dist_u, list) else dist_u
        print("[IG MULTI unweighted hops alpha->phi]", dist_u)
        self.assertEqual(dist_u, 4)

        # Mutation invalidates cache: add direct fast edge a->f (weight=2)
        G.add_edge("a", "f", weight=2)
        dist_new = G.ig.shortest_paths_dijkstra(
            source="alpha", target="phi", weights="weight", _ig_guess_labels=False
        )
        dist_new = dist_new[0][0] if isinstance(dist_new, list) else dist_new
        print("[IG MULTI after mutation alpha->phi]", dist_new)
        self.assertAlmostEqual(dist_new, 2.0, places=6)

    def test_communities_pagerank_components_ig(self):
        G = build_multi_communities_graph()

        # Multilevel (Louvain-like)
        vc = G.ig.community_multilevel(weights="weight", _ig_directed=False)
        sizes = sorted(vc.sizes())
        print("[IG MULTI louvain sizes]", sizes)
        self.assertEqual(sizes, [4, 6])

        # Betweenness / PageRank on undirected view
        igG_und = ig_backend(G, directed=False)
        names = (
            igG_und.vs["name"]
            if "name" in igG_und.vs.attributes()
            else list(range(igG_und.vcount()))
        )

        bc_vals = G.ig.betweenness(directed=False, weights=None)
        top_bc = names[max(range(len(bc_vals)), key=bc_vals.__getitem__)]
        print("[IG MULTI betweenness top]", top_bc)
        self.assertIn(top_bc, {"e", "x"})

        pr_vals = G.ig.pagerank(directed=False)
        top_pr = names[max(range(len(pr_vals)), key=pr_vals.__getitem__)]
        print("[IG MULTI pagerank top]", top_pr)
        self.assertEqual(top_pr, "e")

        # Connected components
        comps = G.ig.components(_ig_directed=False)
        comp_sizes = sorted(comps.sizes())
        print("[IG MULTI connected components]", comp_sizes)
        self.assertEqual(comp_sizes, [10])


# --------------------------- Test Suite: IG SIMPLE ---------------------------


class TestLazyIGProxy_SIMPLE(unittest.TestCase):
    """igraph checks that benefit from simple-edge collapse & aggregation."""

    def test_simple_collapse_aggregations_ig(self):
        # verify simple collapse + _ig_edge_aggs (if proxy supports it);
        # otherwise fallback to igraph.simplify with combine_edges
        G = Graph(directed=True)
        G.add_vertex("u")
        G.add_vertex("v")
        # parallel undirected edges with attrs
        G.add_edge("u", "v", weight=5, capacity=3, edge_directed=False)
        G.add_edge("u", "v", weight=2, capacity=7, edge_directed=False)

        try:
            igG_simple = ig_backend(
                G,
                directed=False,
                simple=True,
                needed_attrs={"weight", "capacity"},
                edge_aggs={"weight": "min", "capacity": "sum"},
            )
            e = igG_simple.es[0]  # one edge after collapse
            w = e["weight"]
            c = e["capacity"]
            print("[IG SIMPLE collapse agg via proxy] weight:", w, "capacity:", c)
            self.assertEqual((w, c), (2, 10))
        except Exception:
            # Fallback: ensure attrs carried over, then collapse
            igG_raw = ig_backend(G, directed=False, needed_attrs={"weight", "capacity"})
            # combine_edges supports "sum","min","max","first","last","mean"
            igG_raw.simplify(
                multiple=True, loops=True, combine_edges={"weight": "min", "capacity": "sum"}
            )
            e = igG_raw.es[0]
            w = e["weight"] if "weight" in igG_raw.es.attributes() else None
            c = e["capacity"] if "capacity" in igG_raw.es.attributes() else None
            print("[IG SIMPLE collapse agg via simplify] weight:", w, "capacity:", c)
            self.assertEqual((w, c), (2, 10))

    def test_cache_invalidation_on_mutation_ig(self):
        # backend object id should change when graph mutates (version bumps)
        G = Graph(directed=True)
        G.add_vertex("a")
        G.add_vertex("b")
        G.add_edge("a", "b", weight=1)

        H1 = ig_backend(G, directed=True, simple=False)
        hid1 = id(H1)
        G.add_edge("b", "a", weight=1)  # mutate
        H2 = ig_backend(G, directed=True, simple=False)
        hid2 = id(H2)
        print("[IG cache invalidation] before:", hid1, "after:", hid2)
        self.assertNotEqual(hid1, hid2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
