import networkx as nx

def to_graphml(graph: "Graph", path, *, directed=True, hyperedge_mode="reify", public_only=False):
    G, _m = to_nx(graph, directed=directed, hyperedge_mode=hyperedge_mode, public_only=public_only)
    nx.write_graphml(G, path)

def from_graphml(path, *, hyperedge="reified") -> "Graph":
    G = nx.read_graphml(path)
    # Prefer reified import when the file encodes hyperedges as nodes
    return from_nx_only(G, hyperedge=("reified" if hyperedge == "reified" else "none"))

def to_gexf(graph: "Graph", path, *, directed=True, hyperedge_mode="reify", public_only=False):
    G, _m = to_nx(graph, directed=directed, hyperedge_mode=hyperedge_mode, public_only=public_only)
    nx.write_gexf(G, path)

def from_gexf(path, *, hyperedge="reified") -> "Graph":
    G = nx.read_gexf(path)
    return from_nx_only(G, hyperedge=("reified" if hyperedge == "reified" else "none"))
