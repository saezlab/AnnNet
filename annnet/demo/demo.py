import annnet as gg


def main():
    G = gg.Graph()

    print("🏗️ Building an extended professional network...")

    # Add people and relationships
    relationships = [
        ("Alice", "Bob", "mentor", 1.0),
        ("Bob", "Charlie", "colleague", 1.2),
        ("Alice", "Dana", "manager", 2.0),
        ("Charlie", "Eve", "colleague", 1.5),
        ("Dana", "Eve", "mentor", 2.2),
        ("Alice", "Eve", "mentor", 2.8),
        ("Eve", "Frank", "colleague", 1.3),
        ("Frank", "Grace", "colleague", 1.1),
        ("Grace", "Heidi", "mentor", 1.7),
        ("Heidi", "Ivan", "colleague", 1.4),
        ("Ivan", "Judy", "colleague", 1.6),
        ("Judy", "Bob", "colleague", 1.8),
        ("Charlie", "Grace", "colleague", 2.0),
    ]

    for src, dst, kind, cost in relationships:
        G.add_edge(src, dst, kind=kind, cost=cost)

    print("🧩 Initial Graph:")
    print(G)

    # --- Lazy conversion: First access triggers creation of nx.Graph ---
    print("\n📈 Degree Centrality (first call triggers conversion):")
    deg_cent = G.nx.degree_centrality()
    for node, score in deg_cent.items():
        print(f"  {node}: {score:.2f}")

    print("\n📏 Closeness Centrality (reuses cached nx.Graph):")
    clos_cent = G.nx.closeness_centrality()
    for node, score in clos_cent.items():
        print(f"  {node}: {score:.2f}")

    print("\n🔀 Betweenness Centrality (still uses cached nx.Graph):")
    betw_cent = G.nx.betweenness_centrality()
    for node, score in betw_cent.items():
        print(f"  {node}: {score:.2f}")

    print("\n📌 PageRank (if supported):")
    try:
        pagerank = G.nx.pagerank()
        for node, score in pagerank.items():
            print(f"  {node}: {score:.4f}")
    except Exception as e:
        print(f"  ⚠️ PageRank failed: {e}")

    print("\n🧭 Shortest paths from Alice:")
    shortest_paths = dict(G.nx.shortest_path_length("Alice"))
    for target, length in shortest_paths.items():
        print(f"  Alice -> {target}: {length}")

    # --- Mutate the graph: should invalidate cache ---
    print("\n✏️ Mutating graph (adding Judy -> Alice)...")
    G.add_edge("Judy", "Alice", kind="feedback", cost=2.5)

    print("📉 Recalculating degree centrality (triggers re-conversion):")
    new_deg_cent = G.nx.degree_centrality()
    for node, score in new_deg_cent.items():
        print(f"  {node}: {score:.2f}")

    # Export and inspect
    print("\n📤 Exporting to NetworkX format:")
    try:
        nxg = G.export("networkx")
        print("  Nodes:", nxg.nodes())
        print("  Edges:", list(nxg.edges()))
    except ModuleNotFoundError:
        print("⚠️  NetworkX not installed.")

    # Plot the graph
    G.plot()


if __name__ == "__main__":
    main()
