## Description

**GraphGlue** is a lightweight, flexible, minimal-dependencies Python package for creating, manipulating, and interoperating with diverse graph data structures. It aims to be the connective tissue ("glue") between multiple graph ecosystems (NetworkX, igraph, graph-tool, etc.) while maintaining a clean and general-purpose internal representation.

---

## 📌 Description

GraphGlue provides a unified interface for:

- Building general-purpose graphs (simple, directed, hyper, signed, etc.)
- Managing node/edge annotations
- Indexing and fast lookups
- Importing from or exporting to various tools and file formats

Its design supports complex workflows in data science, bioinformatics, and network theory — without locking you into a specific graph backend.

---

## ✅ Features

### General Graph Modeling
- Simple graphs, directed graphs, multigraphs
- Hypergraphs (via edge sets or higher-order relationships)
- Signed and weighted edges
- Rich node/edge annotations
- Efficient indexing and lookups

### Import Support
- CSV, JSON, and flat files
- Native import from existing objects:
  - `networkx.Graph`
  - `igraph.Graph`
  - `graph_tool.Graph`
  - `corneto.Graph`

### Export & Interoperability
- NetworkX (`to_networkx()`)
- igraph (`to_igraph()`)
- graph-tool (`to_graphtool()`)
- [CORNETO](https://corneto.org/)
- CSV, JSON, and custom serializations

### Backend-specific Integration (if installed)

If `networkx` is installed, you can call any NetworkX algorithm directly using dot notation:

```python
centrality = G.nx.degree_centrality()
```
**How it works:**

1. GraphGlue converts `G` to a NetworkX object on demand, abstracting away all the non-needed attributes.
2. Runs the algorithm _as-is_.
3. Returns results directly, syncing any graph changes back to the internal representation of the `gg.Graph`.

This allows you to access the full power of NetworkX algorithms with zero boilerplate.

---

## 🛠️ Installation
To install GraphGlue, you can use pip:

```bash
pip install graphglue
```

Optional dependencies for extended functionality:

```bash
pip install graphglue[networkx,igraph]
pip install graphglue[graph-tool]
pip install graphglue[corneto]
```

---

## 🚀 Quick Start

```python
import graphglue as gg

# Create a new graph
G = gg.Graph(directed=True, backend="corneto")

# Add nodes and edges
G.add_node("A", type="gene")
G.add_node("B", type="protein")
G.add_edge("A", "B", sign="+", source="literature")

# Run a NetworkX algorithm (if installed)
path = G.nx.shortest_path(source="A", target="B")

# Export to JSON
G.to_json("graph.json")
```

---

## 📦 Refined Package Structure
The package is organized to separate core functionality, I/O operations, adapters for external libraries, and algorithms. This modular design allows for easy extension and maintenance.

```
graphglue/
│
├── __init__.py                # Public API surface, version, re-exports
├── _version.py                # Single source of truth for __version__
│
├── core/                      # Core, always-installed logic
│   ├── __init__.py
│   ├── _base.py               # BaseGraph class and common interfaces 
│   ├── _graph.py              # Graph class and internal state manager
│   ├── structure.py           # Core data structures (nodes, edges, incidence)
│   ├── metadata.py            # Node/edge attribute store and access helpers
│   ├── views.py               # Subgraph views, shallow/deep copies
│   └── state.py               # Graph history, change tracking, and lazy loading
│
├── io/                        # Loaders and writers for files
│   ├── __init__.py
│   ├── csv.py                 # CSV format (edge list, node table)
│   ├── json.py                # JSON or JSONL parsing (streamable)
│   ├── registry.py            # Entry point registration for loaders/dumpers
│   └── utils.py               # Format detection, path helpers
│
├── adapters/                  # Lazy integration with external libraries
│   ├── __init__.py
│   ├── _base.py               # AbstractAdapter, adapter interface
│   ├── _proxy.py              # Tiny forwarding proxy
│   ├── manager.py             # Adapter registry and lazy bridge logic
│   ├── networkx.py            # If available: convert, cache, sync with NetworkX
│   ├── igraph.py              # If available
│   ├── graphtool.py           # If available
│   └── corneto.py             # If available
│
├── algorithms/                # Pure-python algorithms using core only
│   ├── __init__.py
│   └── traversal.py           # BFS/DFS etc.
│
└── utils/                     # Generic helpers not tied to graph structure
    ├── validation.py
    ├── typing.py
    └── config.py              # Global options, logging, toggles
```

---

## ⚙️ Internal Design
GraphGlue intelligently adapts its internal representation for performance and compatibility:

- Chooses between edge lists, incidence matrices, and adjacency dicts automatically
- Keeps metadata (node/edge attributes) separate from core structure
- Lazy construction of external library representations (e.g., NetworkX) only when needed
- Copy-on-write design for subgraphs and structural mutations

---

## 🧭  Philosophy
GraphGlue vis designed with these principles in mind:

- **Simple**, consistent interface for all graph types
- **Interoperability-first**: integrate, don’t replace
- **Performance-aware**, not performance-obsessed
- **Extendable** and modular, not monolithic

---

## 📜 License
GraphGlue is licensed under the GNU General Public License v3.0 (GPL-3.0). See the [LICENSE](LICENSE) file for details.
