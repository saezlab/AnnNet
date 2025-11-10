import inspect
import math
import time
from collections import defaultdict
from datetime import UTC, datetime
from enum import Enum
from functools import wraps

import numpy as np
import polars as pl
import scipy.sparse as sp


class EdgeType(Enum):
    DIRECTED = "DIRECTED"
    UNDIRECTED = "UNDIRECTED"


class LayerManager:
    """Manager for graph layers and multi-layer operations.

    Provides organized namespace for layer operations by delegating to Graph methods.
    All heavy lifting is done by the Graph class; this is just a clean API surface.

    """

    def __init__(self, graph):
        self._G = graph

    # ==================== Basic Operations (Delegation) ====================

    def add(self, layer_id, **attributes):
        """Create new layer.

        Delegates to Graph.add_layer()
        """
        return self._G.add_layer(layer_id, **attributes)

    def remove(self, layer_id):
        """Remove layer.

        Delegates to Graph.remove_layer()
        """
        return self._G.remove_layer(layer_id)

    def list(self, include_default=False):
        """List layer IDs.

        Delegates to Graph.list_layers()
        """
        return self._G.list_layers(include_default=include_default)

    def exists(self, layer_id):
        """Check if layer exists.

        Delegates to Graph.has_layer()
        """
        return self._G.has_layer(layer_id)

    def info(self, layer_id):
        """Get layer metadata.

        Delegates to Graph.get_layer_info()
        """
        return self._G.get_layer_info(layer_id)

    def count(self):
        """Get number of layers.

        Delegates to Graph.layer_count()
        """
        return self._G.layer_count()

    def vertices(self, layer_id):
        """Get vertices in layer.

        Delegates to Graph.get_layer_vertices()
        """
        return self._G.get_layer_vertices(layer_id)

    def edges(self, layer_id):
        """Get edges in layer.

        Delegates to Graph.get_layer_edges()
        """
        return self._G.get_layer_edges(layer_id)

    # ==================== Active Layer Property ====================

    @property
    def active(self):
        """Get active layer ID.

        Delegates to Graph.get_active_layer()
        """
        return self._G.get_active_layer()

    @active.setter
    def active(self, layer_id):
        """Set active layer ID.

        Delegates to Graph.set_active_layer()
        """
        self._G.set_active_layer(layer_id)

    # ==================== Set Operations (Pure Delegation) ====================

    def union(self, layer_ids):
        """Compute union of layers (returns dict, doesn't create layer).

        Delegates to Graph.layer_union()

        Parameters
        ----------
        layer_ids : list[str]
            Layers to union

        Returns
        -------
        dict
            {"vertices": set[str], "edges": set[str]}

        """
        return self._G.layer_union(layer_ids)

    def intersect(self, layer_ids):
        """Compute intersection of layers (returns dict, doesn't create layer).

        Delegates to Graph.layer_intersection()

        Parameters
        ----------
        layer_ids : list[str]
            Layers to intersect

        Returns
        -------
        dict
            {"vertices": set[str], "edges": set[str]}

        """
        return self._G.layer_intersection(layer_ids)

    def difference(self, layer_a, layer_b):
        """Compute set difference (returns dict, doesn't create layer).

        Delegates to Graph.layer_difference()

        Parameters
        ----------
        layer_a : str
            First layer
        layer_b : str
            Second layer

        Returns
        -------
        dict
            {"vertices": set[str], "edges": set[str]}
            Elements in layer_a but not in layer_b

        """
        return self._G.layer_difference(layer_a, layer_b)

    # ==================== Creation from Operations ====================

    def union_create(self, layer_ids, name, **attributes):
        """Create new layer as union of existing layers.

        Combines Graph.layer_union() + Graph.create_layer_from_operation()

        Parameters
        ----------
        layer_ids : list[str]
            Layers to union
        name : str
            New layer name
        **attributes
            Layer attributes

        Returns
        -------
        str
            Created layer ID

        """
        result = self._G.layer_union(layer_ids)
        return self._G.create_layer_from_operation(name, result, **attributes)

    def intersect_create(self, layer_ids, name, **attributes):
        """Create new layer as intersection of existing layers.

        Combines Graph.layer_intersection() + Graph.create_layer_from_operation()

        Parameters
        ----------
        layer_ids : list[str]
            Layers to intersect
        name : str
            New layer name
        **attributes
            Layer attributes

        Returns
        -------
        str
            Created layer ID

        """
        result = self._G.layer_intersection(layer_ids)
        return self._G.create_layer_from_operation(name, result, **attributes)

    def difference_create(self, layer_a, layer_b, name, **attributes):
        """Create new layer as difference of two layers.

        Combines Graph.layer_difference() + Graph.create_layer_from_operation()

        Parameters
        ----------
        layer_a : str
            First layer
        layer_b : str
            Second layer
        name : str
            New layer name
        **attributes
            Layer attributes

        Returns
        -------
        str
            Created layer ID

        """
        result = self._G.layer_difference(layer_a, layer_b)
        return self._G.create_layer_from_operation(name, result, **attributes)

    def aggregate(
        self, source_layer_ids, target_layer_id, method="union", weight_func=None, **attributes
    ):
        """Create aggregated layer from multiple sources.

        Delegates to Graph.create_aggregated_layer()

        Parameters
        ----------
        source_layer_ids : list[str]
            Source layers
        target_layer_id : str
            Target layer name
        method : {'union', 'intersection'}
            Aggregation method
        weight_func : callable, optional
            Weight merging function (reserved)
        **attributes
            Layer attributes

        Returns
        -------
        str
            Created layer ID

        """
        return self._G.create_aggregated_layer(
            source_layer_ids, target_layer_id, method, weight_func, **attributes
        )

    # ==================== Analysis & Queries ====================

    def stats(self, include_default=False):
        """Get statistics for all layers.

        Delegates to Graph.layer_statistics()

        Returns
        -------
        dict[str, dict]
            {layer_id: {'vertices': int, 'edges': int, 'attributes': dict}}

        """
        return self._G.layer_statistics(include_default=include_default)

    def vertex_presence(self, vertex_id, include_default=False):
        """Find layers containing a vertex.

        Delegates to Graph.vertex_presence_across_layers()

        Parameters
        ----------
        vertex_id : str
            Vertex to search for
        include_default : bool
            Include default layer

        Returns
        -------
        list[str]
            Layer IDs containing the vertex

        """
        return self._G.vertex_presence_across_layers(vertex_id, include_default)

    def edge_presence(
        self, edge_id=None, source=None, target=None, include_default=False, undirected_match=None
    ):
        """Find layers containing an edge.

        Delegates to Graph.edge_presence_across_layers()

        Parameters
        ----------
        edge_id : str, optional
            Edge ID to search for
        source : str, optional
            Source node (with target)
        target : str, optional
            Target node (with source)
        include_default : bool
            Include default layer
        undirected_match : bool, optional
            Allow symmetric matches

        Returns
        -------
        list[str] or dict[str, list[str]]
            If edge_id: list of layer IDs
            If source/target: {layer_id: [edge_ids]}

        """
        return self._G.edge_presence_across_layers(
            edge_id,
            source,
            target,
            include_default=include_default,
            undirected_match=undirected_match,
        )

    def hyperedge_presence(self, members=None, head=None, tail=None, include_default=False):
        """Find layers containing a hyperedge.

        Delegates to Graph.hyperedge_presence_across_layers()

        Parameters
        ----------
        members : Iterable[str], optional
            Undirected hyperedge members
        head : Iterable[str], optional
            Directed hyperedge head
        tail : Iterable[str], optional
            Directed hyperedge tail
        include_default : bool
            Include default layer

        Returns
        -------
        dict[str, list[str]]
            {layer_id: [edge_ids]}

        """
        return self._G.hyperedge_presence_across_layers(
            members=members, head=head, tail=tail, include_default=include_default
        )

    def conserved_edges(self, min_layers=2, include_default=False):
        """Find edges present in multiple layers.

        Delegates to Graph.conserved_edges()

        Parameters
        ----------
        min_layers : int
            Minimum number of layers
        include_default : bool
            Include default layer

        Returns
        -------
        dict[str, int]
            {edge_id: layer_count}

        """
        return self._G.conserved_edges(min_layers, include_default)

    def specific_edges(self, layer_id):
        """Find edges unique to a layer.

        Delegates to Graph.layer_specific_edges()

        Parameters
        ----------
        layer_id : str
            Layer to check

        Returns
        -------
        set[str]
            Edge IDs unique to this layer

        """
        return self._G.layer_specific_edges(layer_id)

    def temporal_dynamics(self, ordered_layers, metric="edge_change"):
        """Analyze temporal changes across layers.

        Delegates to Graph.temporal_dynamics()

        Parameters
        ----------
        ordered_layers : list[str]
            Layers in chronological order
        metric : {'edge_change', 'vertex_change'}
            What to track

        Returns
        -------
        list[dict]
            Per-step changes: [{'added': int, 'removed': int, 'net_change': int}]

        """
        return self._G.temporal_dynamics(ordered_layers, metric)

    # ==================== Convenience Methods ====================

    def summary(self):
        """Get human-readable summary of all layers.

        Returns
        -------
        str
            Formatted summary

        """
        stats = self.stats(include_default=True)
        lines = [f"Layers: {len(stats)}"]

        for i, (layer_id, info) in enumerate(stats.items()):
            prefix = "├─" if i < len(stats) - 1 else "└─"
            lines.append(f"{prefix} {layer_id}: {info['vertices']} vertices, {info['edges']} edges")

        return "\n".join(lines)

    def __repr__(self):
        return f"LayerManager({self.count()} layers)"


class IndexManager:
    """Namespace for index operations.
    Provides clean API over existing dicts.
    """

    def __init__(self, graph):
        self._G = graph

    # ==================== Entity (Node) Indexes ====================

    def entity_to_row(self, entity_id):
        """Map entity ID to matrix row index."""
        if entity_id not in self._G.entity_to_idx:
            raise KeyError(f"Entity '{entity_id}' not found")
        return self._G.entity_to_idx[entity_id]

    def row_to_entity(self, row):
        """Map matrix row index to entity ID."""
        if row not in self._G.idx_to_entity:
            raise KeyError(f"Row {row} not found")
        return self._G.idx_to_entity[row]

    def entities_to_rows(self, entity_ids):
        """Batch convert entity IDs to row indices."""
        return [self._G.entity_to_idx[eid] for eid in entity_ids]

    def rows_to_entities(self, rows):
        """Batch convert row indices to entity IDs."""
        return [self._G.idx_to_entity[r] for r in rows]

    # ==================== Edge Indexes ====================

    def edge_to_col(self, edge_id):
        """Map edge ID to matrix column index."""
        if edge_id not in self._G.edge_to_idx:
            raise KeyError(f"Edge '{edge_id}' not found")
        return self._G.edge_to_idx[edge_id]

    def col_to_edge(self, col):
        """Map matrix column index to edge ID."""
        if col not in self._G.idx_to_edge:
            raise KeyError(f"Column {col} not found")
        return self._G.idx_to_edge[col]

    def edges_to_cols(self, edge_ids):
        """Batch convert edge IDs to column indices."""
        return [self._G.edge_to_idx[eid] for eid in edge_ids]

    def cols_to_edges(self, cols):
        """Batch convert column indices to edge IDs."""
        return [self._G.idx_to_edge[c] for c in cols]

    # ==================== Utilities ====================

    def entity_type(self, entity_id):
        """Get entity type ('vertex' or 'edge')."""
        if entity_id not in self._G.entity_types:
            raise KeyError(f"Entity '{entity_id}' not found")
        return self._G.entity_types[entity_id]

    def is_vertex(self, entity_id):
        """Check if entity is a vertex."""
        return self.entity_type(entity_id) == "vertex"

    def is_edge_entity(self, entity_id):
        """Check if entity is an edge-entity (vertex-edge hybrid)."""
        return self.entity_type(entity_id) == "edge"

    def has_entity(self, entity_id: str) -> bool:
        """True if the ID exists as any entity (vertex or edge-entity)."""
        return entity_id in self._G.entity_to_idx

    def has_vertex(self, vertex_id: str) -> bool:
        """True if the ID exists and is a vertex (not an edge-entity)."""
        return self._G.entity_types.get(vertex_id) == "vertex"

    def has_edge_id(self, edge_id: str) -> bool:
        """True if an edge with this ID exists."""
        return edge_id in self._G.edge_to_idx

    def edge_count(self) -> int:
        """Number of edges (columns in incidence)."""
        return len(self._G.edge_to_idx)

    def entity_count(self) -> int:
        """Number of entities (vertices + edge-entities)."""
        return len(self._G.entity_to_idx)

    def vertex_count(self) -> int:
        """Number of true vertices (excludes edge-entities)."""
        return sum(1 for t in self._G.entity_types.values() if t == "vertex")

    def stats(self):
        """Get index statistics."""
        return {
            "n_entities": len(self._G.entity_to_idx),
            "n_vertices": sum(1 for t in self._G.entity_types.values() if t == "vertex"),
            "n_edge_entities": sum(1 for t in self._G.entity_types.values() if t == "edge"),
            "n_edges": len(self._G.edge_to_idx),
            "max_row": max(self._G.idx_to_entity.keys()) if self._G.idx_to_entity else -1,
            "max_col": max(self._G.idx_to_edge.keys()) if self._G.idx_to_edge else -1,
        }


class CacheManager:
    """Cache manager for materialized views (CSR/CSC)."""

    def __init__(self, graph):
        self._G = graph
        self._csr = None
        self._csc = None
        self._adjacency = None
        self._csr_version = None
        self._csc_version = None
        self._adjacency_version = None

    # ==================== CSR/CSC Properties ====================

    @property
    def csr(self):
        """Get CSR (Compressed Sparse Row) format.
        Builds and caches on first access.
        """
        if self._csr is None or self._csr_version != self._G._version:
            self._csr = self._G._matrix.tocsr()
            self._csr_version = self._G._version
        return self._csr

    @property
    def csc(self):
        """Get CSC (Compressed Sparse Column) format.
        Builds and caches on first access.
        """
        if self._csc is None or self._csc_version != self._G._version:
            self._csc = self._G._matrix.tocsc()
            self._csc_version = self._G._version
        return self._csc

    @property
    def adjacency(self):
        """Get adjacency matrix (computed from incidence).
        For incidence B: adjacency A = B @ B.T
        """
        if self._adjacency is None or self._adjacency_version != self._G._version:
            csr = self.csr
            # Adjacency from incidence: A = B @ B.T
            self._adjacency = csr @ csr.T
            self._adjacency_version = self._G._version
        return self._adjacency

    def has_csr(self) -> bool:
        """True if CSR cache exists and matches current graph version."""
        return self._csr is not None and self._csr_version == self._G._version

    def has_csc(self) -> bool:
        """True if CSC cache exists and matches current graph version."""
        return self._csc is not None and self._csc_version == self._G._version

    def has_adjacency(self) -> bool:
        """True if adjacency cache exists and matches current graph version."""
        return self._adjacency is not None and self._adjacency_version == self._G._version

    def get_csr(self):
        return self.csr

    def get_csc(self):
        return self.csc

    def get_adjacency(self):
        return self.adjacency

    # ==================== Cache Management ====================

    def invalidate(self, formats=None):
        """Invalidate cached formats.

        Parameters
        ----------
        formats : list[str], optional
            Formats to invalidate ('csr', 'csc', 'adjacency').
            If None, invalidate all.

        """
        if formats is None:
            formats = ["csr", "csc", "adjacency"]

        for fmt in formats:
            if fmt == "csr":
                self._csr = None
                self._csr_version = None
            elif fmt == "csc":
                self._csc = None
                self._csc_version = None
            elif fmt == "adjacency":
                self._adjacency = None
                self._adjacency_version = None

    def build(self, formats=None):
        """Pre-build specified formats (eager caching).

        Parameters
        ----------
        formats : list[str], optional
            Formats to build ('csr', 'csc', 'adjacency').
            If None, build all.

        """
        if formats is None:
            formats = ["csr", "csc", "adjacency"]

        for fmt in formats:
            if fmt == "csr":
                _ = self.csr
            elif fmt == "csc":
                _ = self.csc
            elif fmt == "adjacency":
                _ = self.adjacency

    def clear(self):
        """Clear all caches."""
        self.invalidate()

    def info(self):
        """Get cache status and memory usage.

        Returns
        -------
        dict
            Status of each cached format

        """

        def _format_info(matrix, version):
            if matrix is None:
                return {"cached": False}

            # Calculate size
            size_bytes = 0
            if hasattr(matrix, "data"):
                size_bytes += matrix.data.nbytes
            if hasattr(matrix, "indices"):
                size_bytes += matrix.indices.nbytes
            if hasattr(matrix, "indptr"):
                size_bytes += matrix.indptr.nbytes

            return {
                "cached": True,
                "version": version,
                "size_mb": size_bytes / (1024**2),
                "nnz": matrix.nnz if hasattr(matrix, "nnz") else 0,
                "shape": matrix.shape,
            }

        return {
            "csr": _format_info(self._csr, self._csr_version),
            "csc": _format_info(self._csc, self._csc_version),
            "adjacency": _format_info(self._adjacency, self._adjacency_version),
        }


class GraphView:
    """Lazy view into a graph with deferred operations.

    Provides filtered access to graph components without copying the underlying data.
    Views can be materialized into concrete subgraphs when needed.

    Parameters
    ----------
    graph : Graph
        Parent graph instance
    nodes : list[str] | set[str] | callable | None
        Node IDs to include, or predicate function
    edges : list[str] | set[str] | callable | None
        Edge IDs to include, or predicate function
    layers : str | list[str] | None
        Layer ID(s) to include
    predicate : callable | None
        Additional filter: predicate(vertex_id) -> bool

    """

    def __init__(self, graph, nodes=None, edges=None, layers=None, predicate=None):
        self._graph = graph
        self._nodes_filter = nodes
        self._edges_filter = edges
        self._predicate = predicate

        # Normalize layers to list
        if layers is None:
            self._layers = None
        elif isinstance(layers, str):
            self._layers = [layers]
        else:
            self._layers = list(layers)

        # Lazy caches
        self._node_ids_cache = None
        self._edge_ids_cache = None
        self._computed = False

    # ==================== Properties ====================

    @property
    def obs(self):
        """Filtered node attribute table (uses Graph.vertex_attributes)."""
        node_ids = self.node_ids
        if node_ids is None:
            return self._graph.vertex_attributes

        import polars as pl

        return self._graph.vertex_attributes.filter(pl.col("vertex_id").is_in(list(node_ids)))

    @property
    def var(self):
        """Filtered edge attribute table (uses Graph.edge_attributes)."""
        edge_ids = self.edge_ids
        if edge_ids is None:
            return self._graph.edge_attributes

        import polars as pl

        return self._graph.edge_attributes.filter(pl.col("edge_id").is_in(list(edge_ids)))

    @property
    def X(self):
        """Filtered incidence matrix subview."""
        node_ids = self.node_ids
        edge_ids = self.edge_ids

        # Get row and column indices
        if node_ids is not None:
            rows = [
                self._graph.entity_to_idx[nid]
                for nid in node_ids
                if nid in self._graph.entity_to_idx
            ]
        else:
            rows = list(range(self._graph._matrix.shape[0]))

        if edge_ids is not None:
            cols = [
                self._graph.edge_to_idx[eid] for eid in edge_ids if eid in self._graph.edge_to_idx
            ]
        else:
            cols = list(range(self._graph._matrix.shape[1]))

        # Return submatrix slice
        if rows and cols:
            return self._graph._matrix[rows, :][:, cols]
        else:
            import scipy.sparse as sp

            return sp.dok_matrix((len(rows), len(cols)), dtype=self._graph._matrix.dtype)

    @property
    def node_ids(self):
        """Get filtered node IDs (cached)."""
        if not self._computed:
            self._compute_ids()
        return self._node_ids_cache

    @property
    def edge_ids(self):
        """Get filtered edge IDs (cached)."""
        if not self._computed:
            self._compute_ids()
        return self._edge_ids_cache

    @property
    def node_count(self):
        """Number of nodes in this view."""
        node_ids = self.node_ids
        if node_ids is None:
            return sum(1 for t in self._graph.entity_types.values() if t == "vertex")
        return len(node_ids)

    @property
    def edge_count(self):
        """Number of edges in this view."""
        edge_ids = self.edge_ids
        if edge_ids is None:
            return len(self._graph.edge_to_idx)
        return len(edge_ids)

    # ==================== Internal Computation ====================

    def _compute_ids(self):
        """Compute and cache filtered node and edge IDs."""
        node_ids = None
        edge_ids = None

        # Step 1: Apply layer filter (uses Graph._layers)
        if self._layers is not None:
            node_ids = set()
            edge_ids = set()
            for layer_id in self._layers:
                if layer_id in self._graph._layers:
                    node_ids.update(self._graph._layers[layer_id]["vertices"])
                    edge_ids.update(self._graph._layers[layer_id]["edges"])

        # Step 2: Apply node filter
        if self._nodes_filter is not None:
            candidate_nodes = (
                node_ids
                if node_ids is not None
                else set(
                    vid for vid, vtype in self._graph.entity_types.items() if vtype == "vertex"
                )
            )

            if callable(self._nodes_filter):
                filtered_nodes = set()
                for vid in candidate_nodes:
                    try:
                        if self._nodes_filter(vid):
                            filtered_nodes.add(vid)
                    except Exception:
                        pass
                node_ids = filtered_nodes
            else:
                specified = set(self._nodes_filter)
                if node_ids is not None:
                    node_ids &= specified
                else:
                    node_ids = specified & candidate_nodes

        # Step 3: Apply edge filter
        if self._edges_filter is not None:
            candidate_edges = (
                edge_ids if edge_ids is not None else set(self._graph.edge_to_idx.keys())
            )

            if callable(self._edges_filter):
                filtered_edges = set()
                for eid in candidate_edges:
                    try:
                        if self._edges_filter(eid):
                            filtered_edges.add(eid)
                    except Exception:
                        pass
                edge_ids = filtered_edges
            else:
                specified = set(self._edges_filter)
                if edge_ids is not None:
                    edge_ids &= specified
                else:
                    edge_ids = specified & candidate_edges

        # Step 4: Apply additional predicate to nodes
        if self._predicate is not None and node_ids is not None:
            filtered_nodes = set()
            for vid in node_ids:
                try:
                    if self._predicate(vid):
                        filtered_nodes.add(vid)
                except Exception:
                    pass
            node_ids = filtered_nodes

        # Step 5: Filter edges by node connectivity (uses Graph.edge_definitions, hyperedge_definitions)
        if node_ids is not None and edge_ids is not None:
            filtered_edges = set()
            for eid in edge_ids:
                # Binary/vertex-edge edges
                if eid in self._graph.edge_definitions:
                    source, target, _ = self._graph.edge_definitions[eid]
                    if source in node_ids and target in node_ids:
                        filtered_edges.add(eid)
                # Hyperedges
                elif eid in self._graph.hyperedge_definitions:
                    hdef = self._graph.hyperedge_definitions[eid]
                    if hdef.get("directed", False):
                        heads = set(hdef.get("head", []))
                        tails = set(hdef.get("tail", []))
                        if heads.issubset(node_ids) and tails.issubset(node_ids):
                            filtered_edges.add(eid)
                    else:
                        members = set(hdef.get("members", []))
                        if members.issubset(node_ids):
                            filtered_edges.add(eid)
            edge_ids = filtered_edges

        # Cache results
        self._node_ids_cache = node_ids
        self._edge_ids_cache = edge_ids
        self._computed = True

    # ==================== View Methods (use Graph's existing methods) ====================

    def edges_df(self, **kwargs):
        """Get edge DataFrame view with optional filtering.
        Uses Graph.edges_view() and filters by edge IDs.
        """
        # Use Graph's existing edges_view() method
        df = self._graph.edges_view(**kwargs)

        # Filter by edge IDs in this view
        edge_ids = self.edge_ids
        if edge_ids is not None:
            import polars as pl

            df = df.filter(pl.col("edge_id").is_in(list(edge_ids)))

        return df

    def vertices_df(self, **kwargs):
        """Get vertex DataFrame view.
        Uses Graph.vertices_view() and filters by vertex IDs.
        """
        # Use Graph's existing vertices_view() method
        df = self._graph.vertices_view(**kwargs)

        # Filter by vertex IDs in this view
        node_ids = self.node_ids
        if node_ids is not None:
            import polars as pl

            df = df.filter(pl.col("vertex_id").is_in(list(node_ids)))

        return df

    # ==================== Materialization (uses Graph methods) ====================

    def materialize(self, copy_attributes=True):
        """Create a concrete subgraph from this view.
        Uses Graph.add_vertex(), add_edge(), add_hyperedge(), get_*_attrs()
        """
        # Create new Graph instance
        subG = Graph(directed=self._graph.directed)

        node_ids = self.node_ids
        edge_ids = self.edge_ids

        # Determine which nodes to copy
        if node_ids is not None:
            nodes_to_copy = node_ids
        else:
            nodes_to_copy = [
                vid for vid, vtype in self._graph.entity_types.items() if vtype == "vertex"
            ]

        # Copy nodes (uses Graph.add_vertex, get_vertex_attrs)
        for vid in nodes_to_copy:
            if copy_attributes:
                attrs = self._graph.get_vertex_attrs(vid)
                # drop structural keys
                attrs = {k: v for k, v in attrs.items() if k not in self._graph._vertex_RESERVED}
                subG.add_vertex(vid, **attrs)
            else:
                subG.add_vertex(vid)

        # Determine which edges to copy
        if edge_ids is not None:
            edges_to_copy = edge_ids
        else:
            edges_to_copy = self._graph.edge_to_idx.keys()

        # Copy edges (uses Graph methods)
        for eid in edges_to_copy:
            # Binary edges
            if eid in self._graph.edge_definitions:
                source, target, edge_type = self._graph.edge_definitions[eid]

                if source not in nodes_to_copy or target not in nodes_to_copy:
                    continue

                weight = self._graph.edge_weights.get(eid, 1.0)
                directed = self._graph.edge_directed.get(eid, self._graph.directed)

                if copy_attributes:
                    attrs = self._graph.get_edge_attrs(eid)
                    subG.add_edge(source, target, weight=weight, directed=directed, **attrs)
                else:
                    subG.add_edge(source, target, weight=weight, directed=directed)

            # Hyperedges
            elif eid in self._graph.hyperedge_definitions:
                hdef = self._graph.hyperedge_definitions[eid]

                if hdef.get("directed", False):
                    heads = list(hdef.get("head", []))
                    tails = list(hdef.get("tail", []))

                    if not all(h in nodes_to_copy for h in heads):
                        continue
                    if not all(t in nodes_to_copy for t in tails):
                        continue

                    weight = self._graph.edge_weights.get(eid, 1.0)
                    if copy_attributes:
                        attrs = self._graph.get_edge_attrs(eid)
                        subG.add_hyperedge(head=heads, tail=tails, weight=weight, **attrs)
                    else:
                        subG.add_hyperedge(head=heads, tail=tails, weight=weight)
                else:
                    members = list(hdef.get("members", []))

                    if not all(m in nodes_to_copy for m in members):
                        continue

                    weight = self._graph.edge_weights.get(eid, 1.0)
                    if copy_attributes:
                        attrs = self._graph.get_edge_attrs(eid)
                        subG.add_hyperedge(members=members, weight=weight, **attrs)
                    else:
                        subG.add_hyperedge(members=members, weight=weight)

        return subG

    def subview(self, nodes=None, edges=None, layers=None, predicate=None):
        """Create a new GraphView by further restricting this view.

        - nodes/edges: if a list/set is given, intersect with this view's node_ids/edge_ids.
        - layers: defaults to this view's layers if None.
        - predicate: applied in addition to the current filtering (AND).
        """
        # Force compute current filters
        base_nodes = self.node_ids
        base_edges = self.edge_ids

        # Nodes
        if nodes is None:
            new_nodes = base_nodes
            node_pred = None
        elif callable(nodes):
            new_nodes = base_nodes  # keep current set; apply new predicate below
            node_pred = nodes
        else:
            to_set = set(nodes)
            new_nodes = (set(base_nodes) & to_set) if base_nodes is not None else to_set
            node_pred = None

        # Edges
        if edges is None:
            new_edges = base_edges
            edge_pred = None
        elif callable(edges):
            new_edges = base_edges
            edge_pred = edges
        else:
            to_set = set(edges)
            new_edges = (set(base_edges) & to_set) if base_edges is not None else to_set
            edge_pred = None

        # Layers
        new_layers = layers if layers is not None else (self._layers if self._layers else None)

        # Combine predicates (AND) with existing one
        def combined_pred(v):
            ok = True
            if self._predicate:
                try:
                    ok = ok and bool(self._predicate(v))
                except Exception:
                    ok = False
            if predicate:
                try:
                    ok = ok and bool(predicate(v))
                except Exception:
                    ok = False
            if node_pred:
                try:
                    ok = ok and bool(node_pred(v))
                except Exception:
                    ok = False
            return ok

        final_pred = combined_pred if (self._predicate or predicate or node_pred) else None

        # Return a fresh GraphView
        return GraphView(
            self._graph, nodes=new_nodes, edges=new_edges, layers=new_layers, predicate=final_pred
        )

    # ==================== Convenience ====================

    def summary(self):
        """Human-readable summary."""
        lines = [
            "GraphView Summary",
            "─" * 30,
            f"Nodes: {self.node_count}",
            f"Edges: {self.edge_count}",
        ]

        filters = []
        if self._layers:
            filters.append(f"layers={self._layers}")
        if self._nodes_filter:
            if callable(self._nodes_filter):
                filters.append("nodes=<predicate>")
            else:
                filters.append(f"nodes={len(list(self._nodes_filter))} specified")
        if self._edges_filter:
            if callable(self._edges_filter):
                filters.append("edges=<predicate>")
            else:
                filters.append(f"edges={len(list(self._edges_filter))} specified")
        if self._predicate:
            filters.append("predicate=<function>")

        if filters:
            lines.append(f"Filters: {', '.join(filters)}")
        else:
            lines.append("Filters: None (full graph)")

        return "\n".join(lines)

    def __repr__(self):
        return f"GraphView(nodes={self.node_count}, edges={self.edge_count})"

    def __len__(self):
        return self.node_count


class GraphDiff:
    """Represents the difference between two graph states.

    Attributes
    ----------
    vertices_added : set
        Vertices in b but not in a
    vertices_removed : set
        Vertices in a but not in b
    edges_added : set
        Edges in b but not in a
    edges_removed : set
        Edges in a but not in b
    layers_added : set
        Layers in b but not in a
    layers_removed : set
        Layers in a but not in b

    """

    def __init__(self, snapshot_a, snapshot_b):
        self.snapshot_a = snapshot_a
        self.snapshot_b = snapshot_b

        # Compute differences
        self.vertices_added = snapshot_b["vertex_ids"] - snapshot_a["vertex_ids"]
        self.vertices_removed = snapshot_a["vertex_ids"] - snapshot_b["vertex_ids"]
        self.edges_added = snapshot_b["edge_ids"] - snapshot_a["edge_ids"]
        self.edges_removed = snapshot_a["edge_ids"] - snapshot_b["edge_ids"]
        self.layers_added = snapshot_b["layer_ids"] - snapshot_a["layer_ids"]
        self.layers_removed = snapshot_a["layer_ids"] - snapshot_b["layer_ids"]

    def summary(self):
        """Human-readable summary of differences."""
        lines = [
            f"Diff: {self.snapshot_a['label']} → {self.snapshot_b['label']}",
            "",
            f"Vertices: {len(self.vertices_added):+d} added, {len(self.vertices_removed)} removed",
            f"Edges: {len(self.edges_added):+d} added, {len(self.edges_removed)} removed",
            f"Layers: {len(self.layers_added):+d} added, {len(self.layers_removed)} removed",
        ]
        return "\n".join(lines)

    def is_empty(self):
        """Check if there are no differences."""
        return (
            not self.vertices_added
            and not self.vertices_removed
            and not self.edges_added
            and not self.edges_removed
            and not self.layers_added
            and not self.layers_removed
        )

    def __repr__(self):
        return self.summary()

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "snapshot_a": self.snapshot_a["label"],
            "snapshot_b": self.snapshot_b["label"],
            "vertices_added": list(self.vertices_added),
            "vertices_removed": list(self.vertices_removed),
            "edges_added": list(self.edges_added),
            "edges_removed": list(self.edges_removed),
            "layers_added": list(self.layers_added),
            "layers_removed": list(self.layers_removed),
        }


# ===================================


class Graph:
    """Sparse incidence-matrix graph with layers, attributes, parallel edges, and hyperedges.

    The graph is backed by a DOK (Dictionary Of Keys) sparse matrix and exposes
    layered views and attribute tables stored as Polars DF (DataFrame). Supports:
    vertices, binary edges (directed/undirected), edge-entities (vertex-edge hybrids),
    k-ary hyperedges (directed/undirected), per-layer membership and weights,
    and Polars-backed attribute upserts.

    Parameters
    ----------
    directed : bool, optional
        Whether edges are directed by default. Individual edges can override this.

    Notes
    -----
    - Incidence columns encode orientation: +w on source/head, −w on target/tail for
      directed edges; +w on all members for undirected edges/hyperedges.
    - Attributes are **pure**: structural keys are filtered out so attribute tables
      contain only user data.

    See Also
    --------
    add_vertex, add_edge, add_hyperedge, edges_view, vertices_view, layers_view

    """

    # Constants (Attribute helpers)
    _vertex_RESERVED = {"vertex_id"}  # nothing structural for vertices
    _EDGE_RESERVED = {
        "edge_id",
        "source",
        "target",
        "weight",
        "edge_type",
        "directed",
        "layer",
        "layer_weight",
        "kind",
        "members",
        "head",
        "tail",
    }
    _LAYER_RESERVED = {"layer_id"}

    # Construction

    def __init__(self, directed=None, n: int = 0, e: int = 0, **kwargs):
        """Initialize an empty incidence-matrix graph.

        Parameters
        ----------
        directed : bool, optional
            Global default for edge directionality. Individual edges can override this.

        Notes
        -----
        - Stores entities (vertices and edge-entities), edges (including parallels), and
        an incidence matrix in DOK (Dictionary Of Keys) sparse format.
        - Attribute tables are Polars DF (DataFrame) with canonical key columns:
        ``vertex_attributes(vertex_id)``, ``edge_attributes(edge_id)``,
        ``layer_attributes(layer_id)``, and
        ``edge_layer_attributes(layer_id, edge_id, weight)``.
        - A ``'default'`` layer is created and set active.

        """
        self.directed = directed

        # Entity mappings (vertices + vertex-edge hybrids)
        self.entity_to_idx = {}  # entity_id -> row index
        self.idx_to_entity = {}  # row index -> entity_id
        self.entity_types = {}  # entity_id -> 'vertex' or 'edge'

        # Edge mappings (supports parallel edges)
        self.edge_to_idx = {}  # edge_id -> column index
        self.idx_to_edge = {}  # column index -> edge_id
        self.edge_definitions = {}  # edge_id -> (source, target, edge_type)
        self.edge_weights = {}  # edge_id -> weight
        self.edge_directed = {}  # Per-edge directedness; edge_id -> bool  (None = Mixed, True=directed, False=undirected)

        # flexible-direction behavior
        self.edge_direction_policy = {}  # eid -> policy dict
        # ensure 'flexible' isn’t stored as an attribute column
        if not hasattr(self, "_EDGE_RESERVED"):
            self._EDGE_RESERVED = set()
        else:
            self._EDGE_RESERVED = set(self._EDGE_RESERVED)
        self._EDGE_RESERVED.update({"flexible"})

        # Composite vertex key (tuple-of-attrs) support
        self._vertex_key_fields = None            # tuple[str,...] or None
        self._vertex_key_index = {}               # dict[tuple, vertex_id]

        # Sparse incidence matrix
        self._matrix = sp.dok_matrix((0, 0), dtype=np.float32)
        self._num_entities = 0
        self._num_edges = 0

        # Attribute storage using polars DataFrames
        self.vertex_attributes = pl.DataFrame(schema={"vertex_id": pl.Utf8})
        self.edge_attributes = pl.DataFrame(schema={"edge_id": pl.Utf8})
        self.layer_attributes = pl.DataFrame(schema={"layer_id": pl.Utf8})
        self.edge_layer_attributes = pl.DataFrame(
            schema={"layer_id": pl.Utf8, "edge_id": pl.Utf8, "weight": pl.Float64}
        )
        self.edge_kind = {}
        self.hyperedge_definitions = {}
        self.graph_attributes = {}

        # Edge ID counter for parallel edges
        self._next_edge_id = 0

        # Layer management - lightweight dict structure
        self._layers = {}  # layer_id -> {"vertices": set(), "edges": set(), "attributes": {}}
        self._current_layer = None
        self._default_layer = "default"
        self.layer_edge_weights = defaultdict(dict)  # layer_id -> {edge_id: weight}

        # Initialize default layer
        self._layers[self._default_layer] = {"vertices": set(), "edges": set(), "attributes": {}}
        self._current_layer = self._default_layer

        # counts stay logical (start empty)
        self._num_entities = 0
        self._num_edges = 0

        # pre-size the incidence matrix to capacity (no zeros allocated in DOK)
        n = int(n) if n and n > 0 else 0
        e = int(e) if e and e > 0 else 0
        self._matrix = sp.dok_matrix((n, e), dtype=np.float32)

        # grow-only helpers to avoid per-insert exact resizes
        def _grow_rows_to(target: int):
            rows, cols = self._matrix.shape
            if target > rows:
                # geometric bump; keeps behavior, reduces churn
                new_rows = max(target, rows + max(8, rows >> 1))
                self._matrix.resize((new_rows, cols))

        def _grow_cols_to(target: int):
            rows, cols = self._matrix.shape
            if target > cols:
                new_cols = max(target, cols + max(8, cols >> 1))
                self._matrix.resize((rows, new_cols))

        # bind as privates
        self._grow_rows_to = _grow_rows_to
        self._grow_cols_to = _grow_cols_to

        # History and Timeline
        self._history_enabled = True
        self._history = []  # list[dict]
        self._version = 0
        self._history_clock0 = time.perf_counter_ns()
        self._install_history_hooks()  # wrap mutating methods
        self._snapshots = []

    # Layer basics

    def add_layer(self, layer_id, **attributes):
        """Create a new empty layer.

        Parameters
        ----------
        layer_id : str
            New layer identifier (ID).
        **attributes
            Pure layer attributes to store (non-structural).

        Returns
        -------
        str
            The created layer ID.

        Raises
        ------
        ValueError
            If the layer already exists.

        """
        if layer_id in self._layers and layer_id != "default":
            raise ValueError(f"Layer {layer_id} already exists")

        self._layers[layer_id] = {"vertices": set(), "edges": set(), "attributes": attributes}
        # Persist layer metadata to DF (pure attributes, upsert)
        if attributes:
            self.set_layer_attrs(layer_id, **attributes)
        return layer_id

    def set_active_layer(self, layer_id):
        """Set the active layer for subsequent operations.

        Parameters
        ----------
        layer_id : str
            Existing layer ID.

        Raises
        ------
        KeyError
            If the layer does not exist.

        """
        if layer_id not in self._layers:
            raise KeyError(f"Layer {layer_id} not found")
        self._current_layer = layer_id

    def get_active_layer(self):
        """Get the currently active layer ID.

        Returns
        -------
        str
            Active layer ID.

        """
        return self._current_layer

    def get_layers_dict(self, include_default: bool = False):
        """Get a mapping of layer IDs to their metadata.

        Parameters
        ----------
        include_default : bool, optional
            Include the internal ``'default'`` layer if True.

        Returns
        -------
        dict[str, dict]
            ``{layer_id: {"vertices": set, "edges": set, "attributes": dict}}``.

        """
        if include_default:
            return self._layers
        return {k: v for k, v in self._layers.items() if k != self._default_layer}

    def list_layers(self, include_default: bool = False):
        """List layer IDs.

        Parameters
        ----------
        include_default : bool, optional
            Include the internal ``'default'`` layer if True.

        Returns
        -------
        list[str]
            Layer IDs.

        """
        return list(self.get_layers_dict(include_default=include_default).keys())

    def has_layer(self, layer_id):
        """Check whether a layer exists.

        Parameters
        ----------
        layer_id : str

        Returns
        -------
        bool

        """
        return layer_id in self._layers

    def layer_count(self):
        """Get the number of layers (including the internal default).

        Returns
        -------
        int

        """
        return len(self._layers)

    def get_layer_info(self, layer_id):
        """Get a layer's metadata snapshot.

        Parameters
        ----------
        layer_id : str

        Returns
        -------
        dict
            Copy of ``{"vertices": set, "edges": set, "attributes": dict}``.

        Raises
        ------
        KeyError
            If the layer does not exist.

        """
        if layer_id not in self._layers:
            raise KeyError(f"Layer {layer_id} not found")
        return self._layers[layer_id].copy()

    # ID + entity ensure helpers

    def _get_next_edge_id(self) -> str:
        """INTERNAL: Generate a unique edge ID for parallel edges.

        Returns
        -------
        str
            Fresh ``edge_<n>`` identifier (monotonic counter).

        """
        edge_id = f"edge_{self._next_edge_id}"
        self._next_edge_id += 1
        return edge_id

    def _ensure_vertex_table(self) -> None:
        """INTERNAL: Ensure the vertex attribute table exists with a canonical schema.

        Notes
        -----
        - Creates an empty Polars DF [DataFrame] with a single ``Utf8`` ``vertex_id`` column
        if missing or malformed.

        """
        df = getattr(self, "vertex_attributes", None)
        if not isinstance(df, pl.DataFrame) or "vertex_id" not in df.columns:
            self.vertex_attributes = pl.DataFrame({"vertex_id": pl.Series([], dtype=pl.Utf8)})

    def _ensure_vertex_row(self, vertex_id: str) -> None:
        """INTERNAL: Ensure a row for ``vertex_id`` exists in the vertex attribute DF.

        Notes
        -----
        - Appends a new row with ``vertex_id`` and ``None`` for other columns if absent.
        - Preserves existing schema and columns.

        """
        # Intern for cheaper dict ops
        try:
            import sys as _sys

            if isinstance(vertex_id, str):
                vertex_id = _sys.intern(vertex_id)
        except Exception:
            pass

        df = self.vertex_attributes

        # Build/refresh a cached id-set if needed (auto-invalidates on DF object change)
        try:
            cached_ids = getattr(self, "_vertex_attr_ids", None)
            cached_df_id = getattr(self, "_vertex_attr_df_id", None)
            if cached_ids is None or cached_df_id != id(df):
                ids = set()
                if isinstance(df, pl.DataFrame) and df.height > 0 and "vertex_id" in df.columns:
                    # One-time scan to seed cache
                    try:
                        ids = set(df.get_column("vertex_id").to_list())
                    except Exception:
                        # Fallback if column access path changes
                        ids = set(df.select("vertex_id").to_series().to_list())
                self._vertex_attr_ids = ids
                self._vertex_attr_df_id = id(df)
        except Exception:
            # If anything about caching fails, proceed without it
            self._vertex_attr_ids = None
            self._vertex_attr_df_id = None

        # membership check via cache when available
        ids = getattr(self, "_vertex_attr_ids", None)
        if ids is not None and vertex_id in ids:
            return

        # If DF is empty, create the first row with the canonical schema
        if df.is_empty():
            self.vertex_attributes = pl.DataFrame(
                {"vertex_id": [vertex_id]}, schema={"vertex_id": pl.Utf8}
            )
            # keep cache in sync
            try:
                if isinstance(self._vertex_attr_ids, set):
                    self._vertex_attr_ids.add(vertex_id)
                else:
                    self._vertex_attr_ids = {vertex_id}
                self._vertex_attr_df_id = id(self.vertex_attributes)
            except Exception:
                pass
            return

        # Align columns: create a single dict with all columns present
        row = dict.fromkeys(df.columns)
        row["vertex_id"] = vertex_id

        # Append one row efficiently
        try:
            new_df = df.vstack(pl.DataFrame([row]))
        except Exception:
            new_df = pl.concat([df, pl.DataFrame([row])], how="vertical")
        self.vertex_attributes = new_df

        # Update cache after mutation
        try:
            if isinstance(self._vertex_attr_ids, set):
                self._vertex_attr_ids.add(vertex_id)
            else:
                self._vertex_attr_ids = {vertex_id}
            self._vertex_attr_df_id = id(self.vertex_attributes)
        except Exception:
            pass

    def _vertex_key_enabled(self) -> bool:
        return bool(self._vertex_key_fields)

    def _build_key_from_attrs(self, attrs: dict) -> tuple | None:
        """Return tuple of field values in declared order, or None if any missing."""
        if not self._vertex_key_fields:
            return None
        vals = []
        for f in self._vertex_key_fields:
            if f not in attrs or attrs[f] is None:
                return None  # incomplete — not indexable
            vals.append(attrs[f])
        return tuple(vals)

    def _current_key_of_vertex(self, vertex_id) -> tuple | None:
        """Read the current key tuple of a vertex from vertex_attributes (None if incomplete)."""
        if not self._vertex_key_fields:
            return None
        cur = {f: self.get_attr_vertex(vertex_id, f, None) for f in self._vertex_key_fields}
        return self._build_key_from_attrs(cur)

    def _gen_vertex_id_from_key(self, key_tuple: tuple) -> str:
        """Deterministic, human-readable vertex_id from a composite key."""
        parts = [f"{f}={repr(v)}" for f, v in zip(self._vertex_key_fields, key_tuple)]
        return "kv:" + "|".join(parts)

    # Build graph

    def add_vertex(self, vertex_id, layer=None, **attributes):
        """Add (or upsert) a vertex and optionally attach it to a layer.

        Parameters
        ----------
        vertex_id : str
            vertex ID (must be unique across entities).
        layer : str, optional
            Target layer. Defaults to the active layer.
        **attributes
            Pure vertex attributes to store.

        Returns
        -------
        str
            The vertex ID (echoed).

        Notes
        -----
        - Ensures a row exists in the Polars DF [DataFrame] for attributes.
        - Resizes the incidence matrix if needed.

        """
        # Fast normalize to cut hashing/dup costs in dicts.
        try:
            import sys as _sys

            if isinstance(vertex_id, str):
                vertex_id = _sys.intern(vertex_id)
            if layer is None:
                layer = self._current_layer
            elif isinstance(layer, str):
                layer = _sys.intern(layer)
        except Exception:
            layer = layer or self._current_layer

        entity_to_idx = self.entity_to_idx
        idx_to_entity = self.idx_to_entity
        entity_types = self.entity_types
        M = self._matrix  # DOK

        # Add to global superset if new
        if vertex_id not in entity_to_idx:
            idx = self._num_entities
            entity_to_idx[vertex_id] = idx
            idx_to_entity[idx] = vertex_id
            entity_types[vertex_id] = "vertex"
            self._num_entities = idx + 1

            rows, cols = M.shape
            if self._num_entities > rows:
                # geometric growth (≈1.5x), minimum step 8 to avoid frequent resizes
                new_rows = max(self._num_entities, rows + max(8, rows >> 1))
                M.resize((new_rows, cols))

        # Add to specified layer (create if needed)
        layers = self._layers
        if layer not in layers:
            layers[layer] = {"vertices": set(), "edges": set(), "attributes": {}}
        layers[layer]["vertices"].add(vertex_id)

        # Ensure vertex_attributes has a row for this vertex (even with no attrs)
        self._ensure_vertex_table()
        self._ensure_vertex_row(vertex_id)

        # Upsert passed attributes (if any)
        if attributes:
            self.vertex_attributes = self._upsert_row(self.vertex_attributes, vertex_id, attributes)

        return vertex_id

    def add_vertices(self, vertices, layer=None, **attributes):
        # normalize to [(vertex_id, per_attrs), ...]
        it = []
        for item in vertices:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], dict):
                it.append({"vertex_id": item[0], **item[1], **attributes})
            elif isinstance(item, dict):
                d = dict(item)
                d.update(attributes)
                it.append(d)
            else:
                it.append({"vertex_id": item, **attributes})
        self.add_vertices_bulk(
            [(d["vertex_id"], {k: v for k, v in d.items() if k != "vertex_id"}) for d in it],
            layer=layer,
        )
        return [d["vertex_id"] for d in it]

    def add_edge_entity(self, edge_entity_id, layer=None, **attributes):
        """Add an **edge entity** (vertex-edge hybrid) that can connect to vertices/edges.

        Parameters
        ----------
        edge_entity_id : str
            Entity ID to register as type ``'edge'`` in the entity set.
        layer : str, optional
            Target layer. Defaults to the active layer.
        **attributes
            Attributes stored in the vertex attribute DF (treated like vertices).

        Returns
        -------
        str
            The edge-entity ID.

        """
        # Resolve layer default and intern hot strings
        layer = layer or self._current_layer
        try:
            import sys as _sys

            if isinstance(edge_entity_id, str):
                edge_entity_id = _sys.intern(edge_entity_id)
            if isinstance(layer, str):
                layer = _sys.intern(layer)
        except Exception:
            pass

        entity_to_idx = self.entity_to_idx
        layers = self._layers

        # Add to global superset if new (delegate to existing helper)
        if edge_entity_id not in entity_to_idx:
            self._add_edge_entity(edge_entity_id)

        # Add to specified layer
        if layer not in layers:
            layers[layer] = {"vertices": set(), "edges": set(), "attributes": {}}
        layers[layer]["vertices"].add(edge_entity_id)

        # Add attributes (treat edge entities like vertices for attributes)
        if attributes:
            self.set_vertex_attrs(edge_entity_id, **attributes)

        return edge_entity_id

    def _add_edge_entity(self, edge_id):
        """INTERNAL: Register an **edge-entity** so edges can attach to it (vertex-edge mode).

        Parameters
        ----------
        edge_id : str
            Identifier to insert into the entity index as type ``'edge'``.

        Notes
        -----
        - Adds a new entity row and resizes the DOK incidence matrix accordingly.

        """
        try:
            import sys as _sys

            if isinstance(edge_id, str):
                edge_id = _sys.intern(edge_id)
        except Exception:
            pass

        if edge_id not in self.entity_to_idx:
            idx = self._num_entities
            self.entity_to_idx[edge_id] = idx
            self.idx_to_entity[idx] = edge_id
            self.entity_types[edge_id] = "edge"
            self._num_entities = idx + 1

            # Grow-only resize (behavior: matrix >= (num_entities, num_edges))
            M = self._matrix  # DOK
            rows, cols = M.shape
            if self._num_entities > rows:
                # geometric growth to reduce repeated resizes; minimum bump of 8 rows
                new_rows = max(self._num_entities, rows + max(8, rows >> 1))
                M.resize((new_rows, cols))

    def add_edge(
        self,
        source,
        target,
        layer=None,
        weight=1.0,
        edge_id=None,
        edge_type="regular",
        propagate="none",
        layer_weight=None,
        directed=None,
        edge_directed=None,
        **attributes,
    ):
        """Add or update a binary edge between two entities.

        Parameters
        ----------
        source : str
            Source entity ID (vertex or edge-entity for vertex-edge mode).
        target : str
            Target entity ID.
        layer : str, optional
            Layer to place the edge into. Defaults to the active layer.
        weight : float, optional
            Global edge weight stored in the incidence column (default 1.0).
        edge_id : str, optional
            Explicit edge ID. If omitted, a fresh ID is generated.
        edge_type : {'regular', 'vertex_edge'}, optional
            Edge kind. ``'vertex_edge'`` allows connecting to an edge-entity.
        propagate : {'none', 'shared', 'all'}, optional
            Layer propagation:
            - ``'none'`` : only the specified layer
            - ``'shared'`` : all layers that already contain **both** endpoints
            - ``'all'`` : all layers that contain **either** endpoint (and add the other)
        layer_weight : float, optional
            Per-layer weight override for this edge (stored in edge-layer DF).
        edge_directed : bool, optional
            Override default directedness for this edge. If None, uses graph default.
        **attributes
            Pure edge attributes to upsert.

        Returns
        -------
        str
            The edge ID (new or updated).

        Raises
        ------
        ValueError
            If ``propagate`` or ``edge_type`` is invalid.
        TypeError
            If ``weight`` is not numeric.

        Notes
        -----
        - Directed edges write ``+weight`` at source row and ``-weight`` at target row.
        - Undirected edges write ``+weight`` at both endpoints.
        - Updating an existing edge ID overwrites its matrix column and metadata.

        """
        if edge_type is None:
            edge_type = "regular"

        # Resolve dict endpoints via composite key (if enabled)
        if self._vertex_key_enabled():
            if isinstance(source, dict):
                source = self.get_or_create_vertex_by_attrs(layer=layer, **source)
            if isinstance(target, dict):
                target = self.get_or_create_vertex_by_attrs(layer=layer, **target)
        
        flexible = attributes.pop("flexible", None)
        if flexible is not None:
            if not isinstance(flexible, dict) or "var" not in flexible or "threshold" not in flexible:
                raise ValueError("flexible must be a dict with keys {'var','threshold'[,'scope','above','tie']}")
            tie = flexible.get("tie", "keep")
            if tie not in {"keep","undirected","s->t","t->s"}:
                raise ValueError("flexible['tie'] must be one of {'keep','undirected','s->t','t->s'}")

        # normalize endpoints: accept str OR iterable; route hyperedges
        def _to_tuple(x):
            if isinstance(x, (str, bytes)):
                return (x,), False
            try:
                xs = tuple(x)
            except TypeError:
                return (x,), False
            return xs, (len(xs) != 1)

        S, src_multi = _to_tuple(source)
        T, tgt_multi = _to_tuple(target)

        # Hyperedge delegation
        if src_multi or tgt_multi:
            if edge_directed:
                return self.add_hyperedge(
                    head=S,
                    tail=T,
                    edge_directed=True,
                    layer=layer,
                    weight=weight,
                    edge_id=edge_id,
                    **attributes,
                )
            else:
                members = tuple(set(S) | set(T))
                return self.add_hyperedge(
                    members=members,
                    edge_directed=False,
                    layer=layer,
                    weight=weight,
                    edge_id=edge_id,
                    **attributes,
                )

        # Binary case: unwrap singletons to plain IDs
        source, target = S[0], T[0]

        # validate inputs
        if propagate not in {"none", "shared", "all"}:
            raise ValueError(f"propagate must be one of 'none'|'shared'|'all', got {propagate!r}")
        if not isinstance(weight, (int, float)):
            raise TypeError(f"weight must be numeric, got {type(weight).__name__}")
        if edge_type not in {"regular", "vertex_edge"}:
            raise ValueError(f"edge_type must be 'regular' or 'vertex_edge', got {edge_type!r}")

        # resolve layer + whether to touch layering at all
        layer = self._current_layer if layer is None else layer
        touch_layer = layer is not None

        # Intern common strings to speed up dict lookups
        try:
            import sys as _sys

            if isinstance(source, str):
                source = _sys.intern(source)
            if isinstance(target, str):
                target = _sys.intern(target)
            if isinstance(layer, str):
                layer = _sys.intern(layer)
            if isinstance(edge_id, str):
                edge_id = _sys.intern(edge_id)
        except Exception:
            pass

        entity_to_idx = self.entity_to_idx
        idx_to_edge = self.idx_to_edge
        edge_to_idx = self.edge_to_idx
        edge_defs = self.edge_definitions
        edge_w = self.edge_weights
        edge_dir = self.edge_directed
        layers = self._layers
        M = self._matrix  # DOK

        # ensure vertices exist (global)
        def _ensure_vertex_or_edge_entity(x):
            if x in entity_to_idx:
                return
            if edge_type == "vertex_edge" and isinstance(x, str) and x.startswith("edge_"):
                self.add_edge_entity(x, layer=layer)
            else:
                self.add_vertex(x, layer=layer)

        _ensure_vertex_or_edge_entity(source)
        _ensure_vertex_or_edge_entity(target)

        # indices (after potential vertex creation)
        source_idx = entity_to_idx[source]
        target_idx = entity_to_idx[target]

        # edge id
        if edge_id is None:
            edge_id = self._get_next_edge_id()

        # determine direction
        if edge_directed is not None:
            is_dir = bool(edge_directed)
        elif self.directed is not None:
            is_dir = self.directed
        else:
            is_dir = True

        if edge_id in edge_to_idx:
            # UPDATE existing column

            col_idx = edge_to_idx[edge_id]

            # allow explicit direction change; otherwise keep existing
            if edge_directed is None:
                is_dir = edge_dir.get(edge_id, is_dir)
            edge_dir[edge_id] = is_dir

            # keep edge_type attr write
            self.set_edge_attrs(
                edge_id, edge_type=(EdgeType.DIRECTED if is_dir else EdgeType.UNDIRECTED)
            )

            # if source/target changed, update definition
            old_src, old_tgt, old_type = edge_defs[edge_id]
            edge_defs[edge_id] = (source, target, old_type)  # keep old_type by default

            # ensure matrix has enough rows (in case vertices were added since creation)
            self._grow_rows_to(self._num_entities)

            # clear only the cells that were previously set, not the whole column
            try:
                old_src_idx = entity_to_idx[old_src]
                M[old_src_idx, col_idx] = 0
            except KeyError:
                pass
            if old_src != old_tgt:
                try:
                    old_tgt_idx = entity_to_idx[old_tgt]
                    M[old_tgt_idx, col_idx] = 0
                except KeyError:
                    pass

            # write new endpoints
            M[source_idx, col_idx] = weight
            if source != target:
                M[target_idx, col_idx] = -weight if is_dir else weight

            edge_w[edge_id] = weight

        else:
            # CREATE new column

            col_idx = self._num_edges
            edge_to_idx[edge_id] = col_idx
            idx_to_edge[col_idx] = edge_id
            edge_defs[edge_id] = (source, target, edge_type)
            edge_w[edge_id] = weight
            edge_dir[edge_id] = is_dir
            self._num_edges = col_idx + 1

            # grow-only to current logical capacity
            self._grow_rows_to(self._num_entities)
            self._grow_cols_to(self._num_edges)
            M[source_idx, col_idx] = weight
            if source != target:
                M[target_idx, col_idx] = -weight if is_dir else weight

        # layer handling
        if touch_layer:
            if layer not in layers:
                layers[layer] = {"vertices": set(), "edges": set(), "attributes": {}}
            layers[layer]["edges"].add(edge_id)
            layers[layer]["vertices"].update((source, target))

            if layer_weight is not None:
                w = float(layer_weight)
                self.set_edge_layer_attrs(layer, edge_id, weight=w)
                self.layer_edge_weights.setdefault(layer, {})[edge_id] = w

        # propagation
        if propagate == "shared":
            self._propagate_to_shared_layers(edge_id, source, target)
        elif propagate == "all":
            self._propagate_to_all_layers(edge_id, source, target)

        if flexible is not None:
            self.edge_directed[edge_id] = True         # always directed; orientation is controlled
            self.edge_direction_policy[edge_id] = flexible

        # attributes
        if attributes:
            self.set_edge_attrs(edge_id, **attributes)

        if flexible is not None:
            self._apply_flexible_direction(edge_id)

        return edge_id

    def add_parallel_edge(self, source, target, weight=1.0, **attributes):
        """Add a parallel edge (same endpoints, different ID).

        Parameters
        ----------
        source : str
        target : str
        weight : float, optional
        **attributes
            Pure edge attributes.

        Returns
        -------
        str
            The new edge ID.

        """
        try:
            import sys as _sys

            if isinstance(source, str):
                source = _sys.intern(source)
            if isinstance(target, str):
                target = _sys.intern(target)
        except Exception:
            pass

        _add_edge = self.add_edge
        return _add_edge(source, target, weight=weight, edge_id=None, **attributes)

    def add_hyperedge(
        self,
        *,
        members=None,
        head=None,
        tail=None,
        layer=None,
        weight=1.0,
        edge_id=None,
        edge_directed=None,  # bool or None (None -> infer from params)
        **attributes,
    ):
        """Create a k-ary hyperedge as a single incidence column.

        Modes
        -----
        - **Undirected**: pass ``members`` (>=2). Each member gets ``+weight``.
        - **Directed**: pass ``head`` and ``tail`` (both non-empty, disjoint).
        Head gets ``+weight``; tail gets ``-weight``.
        """
        # Map dict endpoints to vertex_id when composite keys are enabled
        if self._vertex_key_enabled():
            def _map(x): 
                return self.get_or_create_vertex_by_attrs(layer=layer, **x) if isinstance(x, dict) else x
            if members is not None:
                members = [_map(u) for u in members]
            else:
                head = [_map(u) for u in head]
                tail = [_map(v) for v in tail]

        # validate form
        if members is None and (head is None or tail is None):
            raise ValueError("Provide members (undirected) OR head+tail (directed).")
        if members is not None and (head is not None or tail is not None):
            raise ValueError("Use either members OR head+tail, not both.")

        if members is not None:
            members = list(members)
            if len(members) < 2:
                raise ValueError("Hyperedge needs >=2 members.")
            directed = False if edge_directed is None else bool(edge_directed)
            if directed:
                raise ValueError("Directed=True requires head+tail, not members.")
        else:
            head = list(head)
            tail = list(tail)
            if not head or not tail:
                raise ValueError("Directed hyperedge needs non-empty head and tail.")
            if set(head) & set(tail):
                raise ValueError("head and tail must be disjoint.")
            directed = True if edge_directed is None else bool(edge_directed)
            if not directed:
                raise ValueError("Undirected=False conflicts with head/tail.")

        # set layer
        layer = self._current_layer if layer is None else layer

        # Intern frequently-used strings for cheaper dict ops
        try:
            import sys as _sys

            if isinstance(layer, str):
                layer = _sys.intern(layer)
            if isinstance(edge_id, str):
                edge_id = _sys.intern(edge_id)
            if members is not None:
                members = [_sys.intern(u) if isinstance(u, str) else u for u in members]
            else:
                head = [_sys.intern(u) if isinstance(u, str) else u for u in head]
                tail = [_sys.intern(v) if isinstance(v, str) else v for v in tail]
        except Exception:
            pass

        # locals for hot paths
        entity_to_idx = self.entity_to_idx
        layers = self._layers
        M = self._matrix  # DOK

        # ensure participants exist globally
        def _ensure_entity(x):
            if x in entity_to_idx:
                return
            if (
                isinstance(x, str)
                and x.startswith("edge_")
                and x in self.entity_types
                and self.entity_types[x] == "edge"
            ):
                return
            self.add_vertex(x, layer=layer)

        if members is not None:
            for u in members:
                _ensure_entity(u)
        else:
            for u in head:
                _ensure_entity(u)
            for v in tail:
                _ensure_entity(v)

        # allocate edge id + column
        if edge_id is None:
            edge_id = self._get_next_edge_id()

        is_new = edge_id not in self.edge_to_idx
        if is_new:
            col_idx = self._num_edges
            self.edge_to_idx[edge_id] = col_idx
            self.idx_to_edge[col_idx] = edge_id
            self._num_edges += 1
            self._grow_rows_to(self._num_entities)
            self._grow_cols_to(self._num_edges)
        else:
            col_idx = self.edge_to_idx[edge_id]
            # clear: delete only previously set cells instead of zeroing whole column
            # handle prior hyperedge or binary edge reuse
            prev_h = self.hyperedge_definitions.get(edge_id)
            if prev_h is not None:
                if prev_h.get("directed", False):
                    rows_to_clear = prev_h["head"] | prev_h["tail"]
                else:
                    rows_to_clear = prev_h["members"]
                for vid in rows_to_clear:
                    try:
                        M[entity_to_idx[vid], col_idx] = 0
                    except KeyError:
                        # vertex may not exist anymore; ignore
                        pass
            else:
                # maybe it was a binary edge before
                prev = self.edge_definitions.get(edge_id)
                if prev is not None:
                    src, tgt, _ = prev
                    if src is not None:
                        try:
                            M[entity_to_idx[src], col_idx] = 0
                        except KeyError:
                            pass
                    if tgt is not None and tgt != src:
                        try:
                            M[entity_to_idx[tgt], col_idx] = 0
                        except KeyError:
                            pass

        self._grow_rows_to(self._num_entities)

        # write column entries
        w = float(weight)
        if members is not None:
            # undirected: +w at each member
            for u in members:
                M[entity_to_idx[u], col_idx] = w
            self.hyperedge_definitions[edge_id] = {
                "directed": False,
                "members": set(members),
            }
        else:
            # directed: +w on head, -w on tail
            for u in head:
                M[entity_to_idx[u], col_idx] = w
            mw = -w
            for v in tail:
                M[entity_to_idx[v], col_idx] = mw
            self.hyperedge_definitions[edge_id] = {
                "directed": True,
                "head": set(head),
                "tail": set(tail),
            }

        # bookkeeping shared with binary edges
        self.edge_weights[edge_id] = w
        self.edge_directed[edge_id] = bool(directed)
        self.edge_kind[edge_id] = "hyper"
        # keep a sentinel in edge_definitions so old code won't crash
        self.edge_definitions[edge_id] = (None, None, "hyper")

        # layer membership + per-layer vertices
        if layer is not None:
            if layer not in layers:
                layers[layer] = {"vertices": set(), "edges": set(), "attributes": {}}
            layers[layer]["edges"].add(edge_id)
            if members is not None:
                layers[layer]["vertices"].update(members)
            else:
                layers[layer]["vertices"].update(self.hyperedge_definitions[edge_id]["head"])
                layers[layer]["vertices"].update(self.hyperedge_definitions[edge_id]["tail"])

        # attributes
        if attributes:
            self.set_edge_attrs(edge_id, **attributes)

        return edge_id

    def add_edge_to_layer(self, lid, eid):
        """Attach an existing edge to a layer (no weight changes).

        Parameters
        ----------
        lid : str
            Layer ID.
        eid : str
            Edge ID.

        Raises
        ------
        KeyError
            If the layer does not exist.

        """
        if lid not in self._layers:
            raise KeyError(f"Layer {lid} does not exist")
        self._layers[lid]["edges"].add(eid)

    def _propagate_to_shared_layers(self, edge_id, source, target):
        """INTERNAL: Add an edge to all layers that already contain **both** endpoints.

        Parameters
        ----------
        edge_id : str
        source : str
        target : str

        """
        for layer_id, layer_data in self._layers.items():
            if source in layer_data["vertices"] and target in layer_data["vertices"]:
                layer_data["edges"].add(edge_id)

    def _propagate_to_all_layers(self, edge_id, source, target):
        """INTERNAL: Add an edge to any layer containing **either** endpoint and
        insert the missing endpoint into that layer.

        Parameters
        ----------
        edge_id : str
        source : str
        target : str

        """
        for layer_id, layer_data in self._layers.items():
            if source in layer_data["vertices"] or target in layer_data["vertices"]:
                layer_data["edges"].add(edge_id)
                # Only add missing endpoint if both vertices should be in layer
                if source in layer_data["vertices"]:
                    layer_data["vertices"].add(target)
                if target in layer_data["vertices"]:
                    layer_data["vertices"].add(source)

    def _normalize_vertices_arg(self, vertices):
        """Normalize a single vertex or an iterable of vertices into a set.

        This internal utility function standardizes input for methods like
        `in_edges()` and `out_edges()` by converting the argument into a set
        of vertex identifiers.

        Parameters
        ----------
        vertices : str | Iterable[str] | None
            - A single vertex ID (string).
            - An iterable of vertex IDs (e.g., list, tuple, set).
            - `None` is allowed and will return an empty set.

        Returns
        -------
        set[str]
            A set of vertex identifiers. If `vertices` is `None`, returns an
            empty set. If a single vertex is provided, returns a one-element set.

        Notes
        -----
        - Strings are treated as **single vertex IDs**, not iterables.
        - If the argument is neither iterable nor a string, it is wrapped in a set.
        - Used internally by API methods that accept flexible vertex arguments.

        """
        if vertices is None:
            return set()
        if isinstance(vertices, (str, bytes)):
            return {vertices}
        try:
            return set(vertices)
        except TypeError:
            return {vertices}

    # Bulk build graph

    def add_vertices_bulk(self, vertices, layer=None):
        """Bulk add vertices (and edge-entities if prefixed externally).
        Accepts: iterable of str  OR  iterable of (vertex_id, attrs_dict)  OR iterable of dicts with keys {'vertex_id', ...attrs}
        Behavior: identical to calling add_vertex() for each, but resizes once and batches attribute inserts.
        """
        import polars as pl

        layer = layer or self._current_layer

        # Normalize items -> [(vid, attrs_dict), ...]
        norm = []
        for it in vertices:
            if isinstance(it, dict):
                vid = it.get("vertex_id") or it.get("id") or it.get("name")
                if vid is None:
                    continue
                a = {k: v for k, v in it.items() if k not in ("vertex_id", "id", "name")}
                norm.append((vid, a))
            elif isinstance(it, (tuple, list)) and it:
                vid = it[0]
                a = it[1] if len(it) > 1 and isinstance(it[1], dict) else {}
                norm.append((vid, a))
            else:
                norm.append((it, {}))

        if not norm:
            return

        # Intern hot strings
        try:
            import sys as _sys

            norm = [
                (_sys.intern(vid) if isinstance(vid, str) else vid, attrs) for vid, attrs in norm
            ]
            if isinstance(layer, str):
                layer = _sys.intern(layer)
        except Exception:
            pass

        # Create missing vertices without per-item resize thrash
        new_rows = 0
        for vid, _ in norm:
            if vid not in self.entity_to_idx:
                idx = self._num_entities
                self.entity_to_idx[vid] = idx
                self.idx_to_entity[idx] = vid
                self.entity_types[vid] = "vertex"
                self._num_entities = idx + 1
                new_rows += 1

        # Grow rows once if needed
        if new_rows:
            self._grow_rows_to(self._num_entities)

        # Layer membership (same semantics as add_vertex)
        if layer not in self._layers:
            self._layers[layer] = {"vertices": set(), "edges": set(), "attributes": {}}
        self._layers[layer]["vertices"].update(vid for vid, _ in norm)

        # Vertex attributes (batch insert for new ones, upsert for existing with attrs)
        self._ensure_vertex_table()
        df = self.vertex_attributes

        # Collect existing ids (if any)
        existing_ids = set()
        try:
            if isinstance(df, pl.DataFrame) and df.height and "vertex_id" in df.columns:
                existing_ids = set(df.get_column("vertex_id").to_list())
        except Exception:
            pass

        # Rows to append for ids missing in DF
        to_append = []
        for vid, attrs in norm:
            if df.is_empty() or vid not in existing_ids:
                row = dict.fromkeys(df.columns) if not df.is_empty() else {"vertex_id": None}
                row["vertex_id"] = vid
                for k, v in attrs.items():
                    row[k] = v
                to_append.append(row)

        if to_append:
            # Ensure df has any new columns first
            need_cols = {k for row in to_append for k in row.keys() if k != "vertex_id"}
            if need_cols:
                df = self._ensure_attr_columns(df, dict.fromkeys(need_cols))

            # Build add_df with full inference over the whole batch to avoid ComputeError
            add_df = pl.DataFrame(
                to_append,
                infer_schema_length=len(to_append),
                nan_to_null=True,
                strict=False,
            )

            # Make sure all df columns exist on add_df
            for c in df.columns:
                if c not in add_df.columns:
                    add_df = add_df.with_columns(pl.lit(None).cast(df.schema[c]).alias(c))

            # Dtype reconciliation (mirror _upsert_row semantics)
            for c in df.columns:
                lc, rc = df.schema[c], add_df.schema[c]
                if lc == pl.Null and rc != pl.Null:
                    df = df.with_columns(pl.col(c).cast(rc))
                elif rc == pl.Null and lc != pl.Null:
                    add_df = add_df.with_columns(pl.col(c).cast(lc).alias(c))
                elif lc != rc:
                    # resolve mismatches by upcasting both to Utf8 (UTF-8 string)
                    df = df.with_columns(pl.col(c).cast(pl.Utf8))
                    add_df = add_df.with_columns(pl.col(c).cast(pl.Utf8).alias(c))

            # Reorder columns EXACTLY to match df before vstack
            add_df = add_df.select(df.columns)

            df = df.vstack(add_df)

        # Upsert attrs for existing ids (vector of updates via helper)
        for vid, attrs in norm:
            if attrs and (df.is_empty() or (vid in existing_ids)):
                df = self._upsert_row(df, vid, attrs)

        self.vertex_attributes = df

    def add_edges_bulk(
        self,
        edges,
        *,
        layer=None,
        default_weight=1.0,
        default_edge_type="regular",
        default_propagate="none",
        default_layer_weight=None,
        default_edge_directed=None,
    ):
        """Bulk add/update *binary* (and vertex-edge) edges.
        Accepts each item as:
        - (src, tgt)
        - (src, tgt, weight)
        - dict with keys: source, target, [weight, edge_id, edge_type, propagate, layer_weight, edge_directed, attributes]
        Behavior: identical to calling add_edge() per item (same propagation/layer/attrs), but grows columns once and avoids full-column wipes.
        """
        layer = self._current_layer if layer is None else layer

        # Normalize into dicts
        norm = []
        for it in edges:
            if isinstance(it, dict):
                d = dict(it)
            elif isinstance(it, (tuple, list)):
                if len(it) == 2:
                    d = {"source": it[0], "target": it[1], "weight": default_weight}
                else:
                    d = {"source": it[0], "target": it[1], "weight": it[2]}
            else:
                continue
            d.setdefault("weight", default_weight)
            d.setdefault("edge_type", default_edge_type)
            d.setdefault("propagate", default_propagate)
            if "layer" not in d:
                d["layer"] = layer
            if "edge_directed" not in d:
                d["edge_directed"] = default_edge_directed
            norm.append(d)

        if not norm:
            return []

        # Intern hot strings & coerce weights
        try:
            import sys as _sys

            for d in norm:
                s, t = d["source"], d["target"]
                if isinstance(s, str):
                    d["source"] = _sys.intern(s)
                if isinstance(t, str):
                    d["target"] = _sys.intern(t)
                lid = d.get("layer")
                if isinstance(lid, str):
                    d["layer"] = _sys.intern(lid)
                eid = d.get("edge_id")
                if isinstance(eid, str):
                    d["edge_id"] = _sys.intern(eid)
                try:
                    d["weight"] = float(d["weight"])
                except Exception:
                    pass
        except Exception:
            pass

        entity_to_idx = self.entity_to_idx
        M = self._matrix
        # 1) Ensure endpoints exist (global); we’ll rely on layer handling below to add membership.
        for d in norm:
            s, t = d["source"], d["target"]
            et = d.get("edge_type", "regular")
            if s not in entity_to_idx:
                # vertex or edge-entity depending on mode?
                if et == "vertex_edge" and isinstance(s, str) and s.startswith("edge_"):
                    self._add_edge_entity(s)
                else:
                    # bare global insert (no layer side-effects; membership handled later)
                    idx = self._num_entities
                    self.entity_to_idx[s] = idx
                    self.idx_to_entity[idx] = s
                    self.entity_types[s] = "vertex"
                    self._num_entities = idx + 1
            if t not in entity_to_idx:
                if et == "vertex_edge" and isinstance(t, str) and t.startswith("edge_"):
                    self._add_edge_entity(t)
                else:
                    idx = self._num_entities
                    self.entity_to_idx[t] = idx
                    self.idx_to_entity[idx] = t
                    self.entity_types[t] = "vertex"
                    self._num_entities = idx + 1

        # Grow rows once if needed
        self._grow_rows_to(self._num_entities)

        # 2) Pre-size columns for new edges
        new_count = sum(1 for d in norm if d.get("edge_id") not in self.edge_to_idx)
        if new_count:
            self._grow_cols_to(self._num_edges + new_count)

        # 3) Create/update columns
        out_ids = []
        for d in norm:
            s, t = d["source"], d["target"]
            w = d["weight"]
            etype = d.get("edge_type", "regular")
            prop = d.get("propagate", default_propagate)
            layer_local = d.get("layer", layer)
            layer_w = d.get("layer_weight", default_layer_weight)
            e_dir = d.get("edge_directed", default_edge_directed)
            edge_id = d.get("edge_id")

            if e_dir is not None:
                is_dir = bool(e_dir)
            elif self.directed is not None:
                is_dir = self.directed
            else:
                is_dir = True
            s_idx = self.entity_to_idx[s]
            t_idx = self.entity_to_idx[t]

            if edge_id is None:
                edge_id = self._get_next_edge_id()

            # update vs create
            if edge_id in self.edge_to_idx:
                col = self.edge_to_idx[edge_id]
                # keep old_type on update (mimic add_edge)
                old_s, old_t, old_type = self.edge_definitions[edge_id]
                # clear only previous cells (no full column wipe)
                try:
                    M[self.entity_to_idx[old_s], col] = 0
                except Exception:
                    pass
                if old_t is not None and old_t != old_s:
                    try:
                        M[self.entity_to_idx[old_t], col] = 0
                    except Exception:
                        pass
                # write new
                M[s_idx, col] = w
                if s != t:
                    M[t_idx, col] = -w if is_dir else w
                self.edge_definitions[edge_id] = (s, t, old_type)
                self.edge_weights[edge_id] = w
                self.edge_directed[edge_id] = is_dir
                # keep attribute side-effect for directedness flag
                self.set_edge_attrs(
                    edge_id, edge_type=(EdgeType.DIRECTED if is_dir else EdgeType.UNDIRECTED)
                )
            else:
                col = self._num_edges
                self.edge_to_idx[edge_id] = col
                self.idx_to_edge[col] = edge_id
                self.edge_definitions[edge_id] = (s, t, etype)
                self.edge_weights[edge_id] = w
                self.edge_directed[edge_id] = is_dir
                self._num_edges = col + 1
                # write cells
                M[s_idx, col] = w
                if s != t:
                    M[t_idx, col] = -w if is_dir else w

            # layer membership + optional per-layer weight
            if layer_local is not None:
                if layer_local not in self._layers:
                    self._layers[layer_local] = {
                        "vertices": set(),
                        "edges": set(),
                        "attributes": {},
                    }
                self._layers[layer_local]["edges"].add(edge_id)
                self._layers[layer_local]["vertices"].update((s, t))
                if layer_w is not None:
                    self.set_edge_layer_attrs(layer_local, edge_id, weight=float(layer_w))
                    self.layer_edge_weights.setdefault(layer_local, {})[edge_id] = float(layer_w)

            # propagation
            if prop == "shared":
                self._propagate_to_shared_layers(edge_id, s, t)
            elif prop == "all":
                self._propagate_to_all_layers(edge_id, s, t)

            # per-edge extra attributes
            attrs = d.get("attributes") or d.get("attrs") or {}
            if attrs:
                self.set_edge_attrs(edge_id, **attrs)

            out_ids.append(edge_id)

        return out_ids

    def add_hyperedges_bulk(
        self,
        hyperedges,
        *,
        layer=None,
        default_weight=1.0,
        default_edge_directed=None,
    ):
        """Bulk add/update hyperedges.
        Each item can be:
        - {'members': [...], 'edge_id': ..., 'weight': ..., 'layer': ..., 'attributes': {...}}
        - {'head': [...], 'tail': [...], ...}
        Behavior: identical to calling add_hyperedge() per item, but grows columns once and avoids full-column wipes.
        """
        layer = self._current_layer if layer is None else layer

        items = []
        for it in hyperedges:
            if not isinstance(it, dict):
                continue
            d = dict(it)
            d.setdefault("weight", default_weight)
            if "layer" not in d:
                d["layer"] = layer
            if "edge_directed" not in d:
                d["edge_directed"] = default_edge_directed
            items.append(d)

        if not items:
            return []

        # Intern + coerce
        try:
            import sys as _sys

            for d in items:
                if "members" in d and d["members"] is not None:
                    d["members"] = [
                        _sys.intern(x) if isinstance(x, str) else x for x in d["members"]
                    ]
                else:
                    d["head"] = [
                        _sys.intern(x) if isinstance(x, str) else x for x in d.get("head", [])
                    ]
                    d["tail"] = [
                        _sys.intern(x) if isinstance(x, str) else x for x in d.get("tail", [])
                    ]
                lid = d.get("layer")
                if isinstance(lid, str):
                    d["layer"] = _sys.intern(lid)
                eid = d.get("edge_id")
                if isinstance(eid, str):
                    d["edge_id"] = _sys.intern(eid)
                try:
                    d["weight"] = float(d["weight"])
                except Exception:
                    pass
        except Exception:
            pass

        # Ensure participants exist (global)
        for d in items:
            if "members" in d and d["members"] is not None:
                for u in d["members"]:
                    if u not in self.entity_to_idx:
                        idx = self._num_entities
                        self.entity_to_idx[u] = idx
                        self.idx_to_entity[idx] = u
                        self.entity_types[u] = "vertex"
                        self._num_entities = idx + 1
            else:
                for u in d.get("head", []):
                    if u not in self.entity_to_idx:
                        idx = self._num_entities
                        self.entity_to_idx[u] = idx
                        self.idx_to_entity[idx] = u
                        self.entity_types[u] = "vertex"
                        self._num_entities = idx + 1
                for v in d.get("tail", []):
                    if v not in self.entity_to_idx:
                        idx = self._num_entities
                        self.entity_to_idx[v] = idx
                        self.entity_types[v] = "vertex"
                        self.idx_to_entity[idx] = v
                        self._num_entities = idx + 1

        # Grow rows once
        self._grow_rows_to(self._num_entities)

        # Pre-size columns
        new_count = sum(1 for d in items if d.get("edge_id") not in self.edge_to_idx)
        if new_count:
            self._grow_cols_to(self._num_edges + new_count)

        M = self._matrix
        out_ids = []

        for d in items:
            members = d.get("members")
            head = d.get("head")
            tail = d.get("tail")
            layer_local = d.get("layer", layer)
            w = float(d.get("weight", default_weight))
            e_id = d.get("edge_id")

            # Decide directedness from form unless forced
            directed = d.get("edge_directed")
            if directed is None:
                directed = members is None

            # allocate/update column
            if e_id is None:
                e_id = self._get_next_edge_id()

            if e_id in self.edge_to_idx:
                col = self.edge_to_idx[e_id]
                # clear old cells (binary or hyper)
                if e_id in self.hyperedge_definitions:
                    h = self.hyperedge_definitions[e_id]
                    if h.get("members"):
                        rows = h["members"]
                    else:
                        rows = set(h.get("head", ())) | set(h.get("tail", ()))
                    for vid in rows:
                        try:
                            M[self.entity_to_idx[vid], col] = 0
                        except Exception:
                            pass
                else:
                    old = self.edge_definitions.get(e_id)
                    if old is not None:
                        os, ot, _ = old
                        try:
                            M[self.entity_to_idx[os], col] = 0
                        except Exception:
                            pass
                        if ot is not None and ot != os:
                            try:
                                M[self.entity_to_idx[ot], col] = 0
                            except Exception:
                                pass
            else:
                col = self._num_edges
                self.edge_to_idx[e_id] = col
                self.idx_to_edge[col] = e_id
                self._num_edges = col + 1

            # write new column values + metadata
            if members is not None:
                for u in members:
                    M[self.entity_to_idx[u], col] = w
                self.hyperedge_definitions[e_id] = {"directed": False, "members": set(members)}
                self.edge_directed[e_id] = False
                self.edge_kind[e_id] = "hyper"
                self.edge_definitions[e_id] = (None, None, "hyper")
            else:
                for u in head:
                    M[self.entity_to_idx[u], col] = w
                for v in tail:
                    M[self.entity_to_idx[v], col] = -w
                self.hyperedge_definitions[e_id] = {
                    "directed": True,
                    "head": set(head),
                    "tail": set(tail),
                }
                self.edge_directed[e_id] = True
                self.edge_kind[e_id] = "hyper"
                self.edge_definitions[e_id] = (None, None, "hyper")

            self.edge_weights[e_id] = w

            # layer membership
            if layer_local is not None:
                if layer_local not in self._layers:
                    self._layers[layer_local] = {
                        "vertices": set(),
                        "edges": set(),
                        "attributes": {},
                    }
                self._layers[layer_local]["edges"].add(e_id)
                if members is not None:
                    self._layers[layer_local]["vertices"].update(members)
                else:
                    self._layers[layer_local]["vertices"].update(head)
                    self._layers[layer_local]["vertices"].update(tail)

            # per-edge attributes (optional)
            attrs = d.get("attributes") or d.get("attrs") or {}
            if attrs:
                self.set_edge_attrs(e_id, **attrs)

            out_ids.append(e_id)

        return out_ids

    def add_edges_to_layer_bulk(self, layer_id, edge_ids):
        """Bulk version of add_edge_to_layer: add many edges to a layer and attach
        all incident vertices. No weights are changed here.
        """
        layer = layer_id if layer_id is not None else self._current_layer
        if layer not in self._layers:
            self._layers[layer] = {"vertices": set(), "edges": set(), "attributes": {}}
        L = self._layers[layer]

        add_edges = {eid for eid in edge_ids if eid in self.edge_to_idx}
        if not add_edges:
            return

        L["edges"].update(add_edges)

        verts = set()
        for eid in add_edges:
            kind = self.edge_kind.get(eid, "binary")
            if kind == "hyper":
                h = self.hyperedge_definitions[eid]
                if h.get("members") is not None:
                    verts.update(h["members"])
                else:
                    verts.update(h.get("head", ()))
                    verts.update(h.get("tail", ()))
            else:
                s, t, _ = self.edge_definitions[eid]
                verts.add(s)
                verts.add(t)

        L["vertices"].update(verts)

    def add_edge_entities_bulk(self, items, layer=None):
        """Bulk add edge-entities (vertex-edge hybrids). Accepts:
        - iterable of str IDs
        - iterable of (edge_entity_id, attrs_dict)
        - iterable of dicts with key 'edge_entity_id' (or 'id')
        Behavior: identical to calling add_edge_entity() for each, but grows rows once
        and batches attribute inserts.
        """
        layer = layer or self._current_layer

        # normalize -> [(eid, attrs)]
        norm = []
        for it in items:
            if isinstance(it, dict):
                eid = it.get("edge_entity_id") or it.get("id")
                if eid is None:
                    continue
                a = {k: v for k, v in it.items() if k not in ("edge_entity_id", "id")}
                norm.append((eid, a))
            elif isinstance(it, (tuple, list)) and it:
                eid = it[0]
                a = it[1] if len(it) > 1 and isinstance(it[1], dict) else {}
                norm.append((eid, a))
            else:
                norm.append((it, {}))
        if not norm:
            return

        # intern hot strings
        try:
            import sys as _sys

            norm = [
                (_sys.intern(eid) if isinstance(eid, str) else eid, attrs) for eid, attrs in norm
            ]
            if isinstance(layer, str):
                layer = _sys.intern(layer)
        except Exception:
            pass

        # create missing rows as type 'edge'
        new_rows = 0
        for eid, _ in norm:
            if eid not in self.entity_to_idx:
                idx = self._num_entities
                self.entity_to_idx[eid] = idx
                self.idx_to_entity[idx] = eid
                self.entity_types[eid] = "edge"
                self._num_entities = idx + 1
                new_rows += 1

        if new_rows:
            self._grow_rows_to(self._num_entities)

        # layer membership
        if layer not in self._layers:
            self._layers[layer] = {"vertices": set(), "edges": set(), "attributes": {}}
        self._layers[layer]["vertices"].update(eid for eid, _ in norm)

        # attributes (edge-entities share vertex_attributes table)
        self._ensure_vertex_table()
        df = self.vertex_attributes
        to_append, existing_ids = [], set()
        try:
            if df.height and "vertex_id" in df.columns:
                existing_ids = set(df.get_column("vertex_id").to_list())
        except Exception:
            pass

        for eid, attrs in norm:
            if df.is_empty() or eid not in existing_ids:
                row = dict.fromkeys(df.columns) if not df.is_empty() else {"vertex_id": None}
                row["vertex_id"] = eid
                for k, v in attrs.items():
                    row[k] = v
                to_append.append(row)

        if to_append:
            need_cols = {k for r in to_append for k in r if k != "vertex_id"}
            if need_cols:
                df = self._ensure_attr_columns(df, dict.fromkeys(need_cols))
            add_df = pl.DataFrame(to_append)
            for c in df.columns:
                if c not in add_df.columns:
                    add_df = add_df.with_columns(pl.lit(None).cast(df.schema[c]).alias(c))
            for c in df.columns:
                lc, rc = df.schema[c], add_df.schema[c]
                if lc == pl.Null and rc != pl.Null:
                    df = df.with_columns(pl.col(c).cast(rc))
                elif rc == pl.Null and lc != pl.Null:
                    add_df = add_df.with_columns(pl.col(c).cast(lc).alias(c))
                elif lc != rc:
                    df = df.with_columns(pl.col(c).cast(pl.Utf8))
                    add_df = add_df.with_columns(pl.col(c).cast(pl.Utf8).alias(c))
                if to_append:
                    need_cols = {k for r in to_append for k in r if k != "vertex_id"}
                    if need_cols:
                        df = self._ensure_attr_columns(df, dict.fromkeys(need_cols))

                    add_df = pl.DataFrame(to_append)

                    # ensure all df columns exist on add_df
                    for c in df.columns:
                        if c not in add_df.columns:
                            add_df = add_df.with_columns(pl.lit(None).cast(df.schema[c]).alias(c))

                    # dtype reconciliation (same as before)
                    for c in df.columns:
                        lc, rc = df.schema[c], add_df.schema[c]
                        if lc == pl.Null and rc != pl.Null:
                            df = df.with_columns(pl.col(c).cast(rc))
                        elif rc == pl.Null and lc != pl.Null:
                            add_df = add_df.with_columns(pl.col(c).cast(lc).alias(c))
                        elif lc != rc:
                            df = df.with_columns(pl.col(c).cast(pl.Utf8))
                            add_df = add_df.with_columns(pl.col(c).cast(pl.Utf8).alias(c))

                    # reorder add_df columns to match df exactly
                    add_df = add_df.select(df.columns)

                    df = df.vstack(add_df)

        for eid, attrs in norm:
            if attrs and (df.is_empty() or (eid in existing_ids)):
                df = self._upsert_row(df, eid, attrs)
        self.vertex_attributes = df

    def set_vertex_key(self, *fields: str):
        """Declare composite key fields (order matters). Rebuilds the uniqueness index.

        - Raises ValueError if duplicates exist among already-populated vertices.
        - Vertices missing some key fields are skipped during indexing.
        """
        if not fields:
            raise ValueError("set_vertex_key requires at least one field")
        self._vertex_key_fields = tuple(str(f) for f in fields)
        self._vertex_key_index.clear()

        df = self.vertex_attributes
        if not isinstance(df, pl.DataFrame) or df.height == 0:
            return  # nothing to index yet

        missing = [f for f in self._vertex_key_fields if f not in df.columns]
        if missing:
            # ok to skip; those rows simply won't be indexable until fields appear
            pass

        # Rebuild index, enforcing uniqueness only for fully-populated tuples
        try:
            for row in df.iter_rows(named=True):
                vid = row.get("vertex_id")
                key = tuple(row.get(f) for f in self._vertex_key_fields)
                if any(v is None for v in key):
                    continue
                owner = self._vertex_key_index.get(key)
                if owner is not None and owner != vid:
                    raise ValueError(f"Composite key conflict for {key}: {owner} vs {vid}")
                self._vertex_key_index[key] = vid
        except Exception:
            # Fallback if iter_rows misbehaves
            for vid in df.get_column("vertex_id").to_list():
                cur = {f: self.get_attr_vertex(vid, f, None) for f in self._vertex_key_fields}
                key = self._build_key_from_attrs(cur)
                if key is None:
                    continue
                owner = self._vertex_key_index.get(key)
                if owner is not None and owner != vid:
                    raise ValueError(f"Composite key conflict for {key}: {owner} vs {vid}")
                self._vertex_key_index[key] = vid

    # Remove / mutate down

    def remove_edge(self, edge_id):
        """Remove an edge (binary or hyperedge) from the graph.

        Parameters
        ----------
        edge_id : str

        Raises
        ------
        KeyError
            If the edge is not found.

        Notes
        -----
        - Physically removes the incidence column (no CSR round-trip).
        - Cleans edge attributes, layer memberships, and per-layer entries.

        """
        if edge_id not in self.edge_to_idx:
            raise KeyError(f"Edge {edge_id} not found")

        col_idx = self.edge_to_idx[edge_id]

        # column removal without CSR (single pass over nonzeros)
        M_old = self._matrix
        rows, cols = M_old.shape
        new_cols = cols - 1
        # Rebuild DOK with columns > col_idx shifted left by 1
        M_new = sp.dok_matrix((rows, new_cols), dtype=M_old.dtype)
        for (r, c), v in M_old.items():
            if c == col_idx:
                continue  # drop this column
            elif c > col_idx:
                M_new[r, c - 1] = v
            else:
                M_new[r, c] = v
        self._matrix = M_new

        # mappings (preserve relative order of remaining edges)
        # Remove the deleted edge id
        del self.edge_to_idx[edge_id]
        # Shift indices for edges after the removed column
        for old_idx in range(col_idx + 1, self._num_edges):
            eid = self.idx_to_edge.pop(old_idx)
            self.idx_to_edge[old_idx - 1] = eid
            self.edge_to_idx[eid] = old_idx - 1
        # Drop the last stale entry (now shifted)
        self.idx_to_edge.pop(self._num_edges - 1, None)
        self._num_edges -= 1

        # Metadata cleanup
        # Edge definitions / weights / directedness
        self.edge_definitions.pop(edge_id, None)
        self.edge_weights.pop(edge_id, None)
        if edge_id in self.edge_directed:
            self.edge_directed.pop(edge_id, None)

        # Remove from edge attributes
        if (
            isinstance(self.edge_attributes, pl.DataFrame)
            and self.edge_attributes.height > 0
            and "edge_id" in self.edge_attributes.columns
        ):
            self.edge_attributes = self.edge_attributes.filter(pl.col("edge_id") != edge_id)

        # Remove from per-layer membership
        for layer_data in self._layers.values():
            layer_data["edges"].discard(edge_id)

        # Remove from edge-layer attributes
        if (
            isinstance(self.edge_layer_attributes, pl.DataFrame)
            and self.edge_layer_attributes.height > 0
            and "edge_id" in self.edge_layer_attributes.columns
        ):
            self.edge_layer_attributes = self.edge_layer_attributes.filter(
                pl.col("edge_id") != edge_id
            )

        # Legacy / auxiliary dicts
        for d in self.layer_edge_weights.values():
            d.pop(edge_id, None)

        self.edge_kind.pop(edge_id, None)
        self.hyperedge_definitions.pop(edge_id, None)

    def remove_vertex(self, vertex_id):
        """Remove a vertex and all incident edges (binary + hyperedges).

        Parameters
        ----------
        vertex_id : str

        Raises
        ------
        KeyError
            If the vertex is not found.

        Notes
        -----
        - Rebuilds entity indexing and shrinks the incidence matrix accordingly.

        """
        if vertex_id not in self.entity_to_idx:
            raise KeyError(f"vertex {vertex_id} not found")

        entity_idx = self.entity_to_idx[vertex_id]

        # Collect incident edges (set to avoid duplicates)
        edges_to_remove = set()

        # Binary edges: edge_definitions {eid: (source, target, ...)}
        for eid, edef in list(self.edge_definitions.items()):
            try:
                source, target = edef[0], edef[1]
            except Exception:
                source, target = edef.get("source"), edef.get("target")
            if source == vertex_id or target == vertex_id:
                edges_to_remove.add(eid)

        # Hyperedges: hyperedge_definitions {eid: {"head":[...], "tail":[...]}} or {"members":[...]}
        def _vertex_in_hyperdef(hdef: dict, vertex: str) -> bool:
            # Common keys first
            for key in ("head", "tail", "members", "vertices", "vertices"):
                seq = hdef.get(key)
                if isinstance(seq, (list, tuple, set)) and vertex in seq:
                    return True
            # Safety net: scan any list/tuple/set values
            for v in hdef.values():
                if isinstance(v, (list, tuple, set)) and vertex in v:
                    return True
            return False

        hdefs = getattr(self, "hyperedge_definitions", {})
        if isinstance(hdefs, dict):
            for heid, hdef in list(hdefs.items()):
                if isinstance(hdef, dict) and _vertex_in_hyperdef(hdef, vertex_id):
                    edges_to_remove.add(heid)

        # Remove all collected edges
        for eid in edges_to_remove:
            self.remove_edge(eid)

        # row removal without CSR: rebuild DOK with rows-1 and shift indices
        M_old = self._matrix
        rows, cols = M_old.shape
        new_rows = rows - 1
        M_new = sp.dok_matrix((new_rows, cols), dtype=M_old.dtype)
        for (r, c), v in M_old.items():
            if r == entity_idx:
                continue  # drop this row
            elif r > entity_idx:
                M_new[r - 1, c] = v
            else:
                M_new[r, c] = v
        self._matrix = M_new

        # Update entity mappings
        del self.entity_to_idx[vertex_id]
        del self.entity_types[vertex_id]

        # Shift indices for entities after the removed row; preserve relative order
        for old_idx in range(entity_idx + 1, self._num_entities):
            ent_id = self.idx_to_entity.pop(old_idx)
            self.idx_to_entity[old_idx - 1] = ent_id
            self.entity_to_idx[ent_id] = old_idx - 1
        # Drop last stale entry and shrink count
        self.idx_to_entity.pop(self._num_entities - 1, None)
        self._num_entities -= 1

        # Remove from vertex attributes
        if isinstance(self.vertex_attributes, pl.DataFrame):
            if self.vertex_attributes.height > 0 and "vertex_id" in self.vertex_attributes.columns:
                self.vertex_attributes = self.vertex_attributes.filter(
                    pl.col("vertex_id") != vertex_id
                )

        # Remove from per-layer membership
        for layer_data in self._layers.values():
            layer_data["vertices"].discard(vertex_id)

    def remove_layer(self, layer_id):
        """Remove a non-default layer and its per-layer attributes.

        Parameters
        ----------
        layer_id : str

        Raises
        ------
        ValueError
            If attempting to remove the internal default layer.
        KeyError
            If the layer does not exist.

        Notes
        -----
        - Does not delete vertices/edges globally; only membership and layer metadata.

        """
        if layer_id == self._default_layer:
            raise ValueError("Cannot remove default layer")
        if layer_id not in self._layers:
            raise KeyError(f"Layer {layer_id} not found")

        # Purge per-layer attributes
        ela = getattr(self, "edge_layer_attributes", None)
        if isinstance(ela, pl.DataFrame) and ela.height > 0 and "layer_id" in ela.columns:
            # Keep everything not matching the layer_id
            self.edge_layer_attributes = ela.filter(pl.col("layer_id") != layer_id)

        # Drop legacy dict slice if present
        if isinstance(getattr(self, "layer_edge_weights", None), dict):
            self.layer_edge_weights.pop(layer_id, None)

        # Remove the layer and reset current if needed
        del self._layers[layer_id]
        if self._current_layer == layer_id:
            self._current_layer = self._default_layer

    # Bulk remove / mutate down

    def remove_edges(self, edge_ids):
        """Remove many edges in one pass (much faster than looping)."""
        to_drop = [eid for eid in edge_ids if eid in self.edge_to_idx]
        if not to_drop:
            return
        self._remove_edges_bulk(to_drop)

    def remove_vertices(self, vertex_ids):
        """Remove many vertices (and all their incident edges) in one pass."""
        to_drop = [vid for vid in vertex_ids if vid in self.entity_to_idx]
        if not to_drop:
            return
        self._remove_vertices_bulk(to_drop)

    def _remove_edges_bulk(self, edge_ids):
        drop = set(edge_ids)
        if not drop:
            return

        # Columns to keep, old->new remap
        keep_pairs = sorted(
            ((idx, eid) for eid, idx in self.edge_to_idx.items() if eid not in drop)
        )
        old_to_new = {
            old: new for new, (old, _eid) in enumerate(((old, eid) for old, eid in keep_pairs))
        }
        new_cols = len(keep_pairs)

        # Rebuild matrix once
        M_old = self._matrix  # DOK
        rows, _cols = M_old.shape
        M_new = sp.dok_matrix((rows, new_cols), dtype=M_old.dtype)
        for (r, c), v in M_old.items():
            if c in old_to_new:
                M_new[r, old_to_new[c]] = v
        self._matrix = M_new

        # Rebuild edge mappings
        self.idx_to_edge.clear()
        self.edge_to_idx.clear()
        for new_idx, (old_idx, eid) in enumerate(keep_pairs):
            self.idx_to_edge[new_idx] = eid
            self.edge_to_idx[eid] = new_idx
        self._num_edges = new_cols

        # Metadata cleanup (vectorized)
        # Dicts
        for eid in drop:
            self.edge_definitions.pop(eid, None)
            self.edge_weights.pop(eid, None)
            self.edge_directed.pop(eid, None)
            self.edge_kind.pop(eid, None)
            self.hyperedge_definitions.pop(eid, None)
        for layer_data in self._layers.values():
            layer_data["edges"].difference_update(drop)
        for d in self.layer_edge_weights.values():
            for eid in drop:
                d.pop(eid, None)

        # DataFrames
        if isinstance(self.edge_attributes, pl.DataFrame) and self.edge_attributes.height:
            if "edge_id" in self.edge_attributes.columns:
                self.edge_attributes = self.edge_attributes.filter(
                    ~pl.col("edge_id").is_in(list(drop))
                )
        if (
            isinstance(self.edge_layer_attributes, pl.DataFrame)
            and self.edge_layer_attributes.height
        ):
            cols = set(self.edge_layer_attributes.columns)
            if {"edge_id"}.issubset(cols):
                self.edge_layer_attributes = self.edge_layer_attributes.filter(
                    ~pl.col("edge_id").is_in(list(drop))
                )

    def _remove_vertices_bulk(self, vertex_ids):
        drop_vs = set(vertex_ids)
        if not drop_vs:
            return

        # 1) Collect incident edges (binary + hyper)
        drop_es = set()
        for eid, (s, t, _typ) in list(self.edge_definitions.items()):
            if s in drop_vs or t in drop_vs:
                drop_es.add(eid)
        for eid, hdef in list(self.hyperedge_definitions.items()):
            if hdef.get("members"):
                if drop_vs & set(hdef["members"]):
                    drop_es.add(eid)
            else:
                if (drop_vs & set(hdef.get("head", ()))) or (
                    drop_vs & set(hdef.get("tail", ()))
                ):  # directed
                    drop_es.add(eid)

        # 2) Drop all those edges in one pass
        if drop_es:
            self._remove_edges_bulk(drop_es)

        # 3) Build row keep list and old->new map
        keep_idx = []
        for idx in range(self._num_entities):
            ent = self.idx_to_entity[idx]
            if ent not in drop_vs:
                keep_idx.append(idx)
        old_to_new = {old: new for new, old in enumerate(keep_idx)}
        new_rows = len(keep_idx)

        # 4) Rebuild matrix rows once
        M_old = self._matrix  # DOK
        _rows, cols = M_old.shape
        M_new = sp.dok_matrix((new_rows, cols), dtype=M_old.dtype)
        for (r, c), v in M_old.items():
            if r in old_to_new:
                M_new[old_to_new[r], c] = v
        self._matrix = M_new

        # 5) Rebuild entity mappings
        new_entity_to_idx = {}
        new_idx_to_entity = {}
        for new_i, old_i in enumerate(keep_idx):
            ent = self.idx_to_entity[old_i]
            new_entity_to_idx[ent] = new_i
            new_idx_to_entity[new_i] = ent
        self.entity_to_idx = new_entity_to_idx
        self.idx_to_entity = new_idx_to_entity
        # types: drop removed
        for vid in drop_vs:
            self.entity_types.pop(vid, None)
        self._num_entities = new_rows

        # 6) Clean vertex attributes and layer memberships
        if isinstance(self.vertex_attributes, pl.DataFrame) and self.vertex_attributes.height:
            if "vertex_id" in self.vertex_attributes.columns:
                self.vertex_attributes = self.vertex_attributes.filter(
                    ~pl.col("vertex_id").is_in(list(drop_vs))
                )
        for layer_data in self._layers.values():
            layer_data["vertices"].difference_update(drop_vs)

    # Attributes & weights

    def set_graph_attribute(self, key, value):
        """Set a graph-level attribute.

        Parameters
        ----------
        key : str
        value : Any

        """
        self.graph_attributes[key] = value

    def get_graph_attribute(self, key, default=None):
        """Get a graph-level attribute.

        Parameters
        ----------
        key : str
        default : Any, optional

        Returns
        -------
        Any

        """
        return self.graph_attributes.get(key, default)

    def set_vertex_attrs(self, vertex_id, **attrs):
        """Upsert pure vertex attributes (non-structural) into the vertex DF [DataFrame]."""
        clean = {k: v for k, v in attrs.items() if k not in self._vertex_RESERVED}
        if not clean:
            return

        # If composite-key is active, validate prospective key BEFORE writing
        if self._vertex_key_enabled():
            old_key = self._current_key_of_vertex(vertex_id)
            # prospective values = old values overridden by incoming clean attrs
            merged = {f: (clean[f] if f in clean else self.get_attr_vertex(vertex_id, f, None))
                    for f in self._vertex_key_fields}
            new_key = self._build_key_from_attrs(merged)
            if new_key is not None:
                owner = self._vertex_key_index.get(new_key)
                if owner is not None and owner != vertex_id:
                    raise ValueError(
                        f"Composite key collision on {self._vertex_key_fields}: {new_key} owned by {owner}"
                    )

        # Write attributes
        self.vertex_attributes = self._upsert_row(self.vertex_attributes, vertex_id, clean)

        watched = self._variables_watched_by_vertices()
        if watched and any(k in watched for k in clean):
            for eid in self._incident_flexible_edges(vertex_id):
                self._apply_flexible_direction(eid)

        # Update index AFTER successful write
        if self._vertex_key_enabled():
            new_key = self._current_key_of_vertex(vertex_id)
            old_key = old_key if 'old_key' in locals() else None
            if old_key != new_key:
                if old_key is not None and self._vertex_key_index.get(old_key) == vertex_id:
                    self._vertex_key_index.pop(old_key, None)
                if new_key is not None:
                    self._vertex_key_index[new_key] = vertex_id

    def get_attr_vertex(self, vertex_id, key, default=None):
        """Get a single vertex attribute (scalar) or default if missing.

        Parameters
        ----------
        vertex_id : str
        key : str
        default : Any, optional

        Returns
        -------
        Any

        """
        df = self.vertex_attributes
        if key not in df.columns:
            return default
        rows = df.filter(pl.col("vertex_id") == vertex_id)
        if rows.height == 0:
            return default
        val = rows.select(pl.col(key)).to_series()[0]
        return default if val is None else val

    def get_vertex_attribute(self, vertex_id, attribute):  # legacy alias
        """(Legacy alias) Get a single vertex attribute from the Polars DF [DataFrame].

        Parameters
        ----------
        vertex_id : str
        attribute : str or enum.Enum
            Column name or Enum with ``.value``.

        Returns
        -------
        Any or None
            Scalar value if present, else ``None``.

        See Also
        --------
        get_attr_vertex

        """
        # allow Attr enums
        attribute = getattr(attribute, "value", attribute)

        df = self.vertex_attributes
        if not isinstance(df, pl.DataFrame):
            return None
        if df.height == 0 or "vertex_id" not in df.columns or attribute not in df.columns:
            return None

        rows = df.filter(pl.col("vertex_id") == vertex_id)
        if rows.height == 0:
            return None

        s = rows.get_column(attribute)
        return s.item(0) if s.len() else None

    def set_edge_attrs(self, edge_id, **attrs):
        """Upsert pure edge attributes (non-structural) into the edge DF.

        Parameters
        ----------
        edge_id : str
        **attrs
            Key/value attributes. Structural keys are ignored.

        """
        # keep attributes table pure: strip structural keys
        clean = {k: v for k, v in attrs.items() if k not in self._EDGE_RESERVED}
        if clean:
            self.edge_attributes = self._upsert_row(self.edge_attributes, edge_id, clean)
        pol = self.edge_direction_policy.get(edge_id)
        if pol and pol.get("scope", "edge") == "edge" and pol["var"] in clean:
            self._apply_flexible_direction(edge_id)

    def get_attr_edge(self, edge_id, key, default=None):
        """Get a single edge attribute (scalar) or default if missing.

        Parameters
        ----------
        edge_id : str
        key : str
        default : Any, optional

        Returns
        -------
        Any

        """
        df = self.edge_attributes
        if key not in df.columns:
            return default
        rows = df.filter(pl.col("edge_id") == edge_id)
        if rows.height == 0:
            return default
        val = rows.select(pl.col(key)).to_series()[0]
        return default if val is None else val

    def get_edge_attribute(self, edge_id, attribute):  # legacy alias
        """(Legacy alias) Get a single edge attribute from the Polars DF [DataFrame].

        Parameters
        ----------
        edge_id : str
        attribute : str or enum.Enum
            Column name or Enum with ``.value``.

        Returns
        -------
        Any or None
            Scalar value if present, else ``None``.

        See Also
        --------
        get_attr_edge

        """
        # allow Attr enums
        attribute = getattr(attribute, "value", attribute)

        df = self.edge_attributes
        if not isinstance(df, pl.DataFrame):
            return None
        if df.height == 0 or "edge_id" not in df.columns or attribute not in df.columns:
            return None

        rows = df.filter(pl.col("edge_id") == edge_id)
        if rows.height == 0:
            return None

        s = rows.get_column(attribute)
        return s.item(0) if s.len() else None

    def set_layer_attrs(self, layer_id, **attrs):
        """Upsert pure layer attributes.

        Parameters
        ----------
        layer_id : str
        **attrs
            Key/value attributes. Structural keys are ignored.

        """
        clean = {k: v for k, v in attrs.items() if k not in self._LAYER_RESERVED}
        if clean:
            self.layer_attributes = self._upsert_row(self.layer_attributes, layer_id, clean)

    def get_layer_attr(self, layer_id, key, default=None):
        """Get a single layer attribute (scalar) or default if missing.

        Parameters
        ----------
        layer_id : str
        key : str
        default : Any, optional

        Returns
        -------
        Any

        """
        df = self.layer_attributes
        if key not in df.columns:
            return default
        rows = df.filter(pl.col("layer_id") == layer_id)
        if rows.height == 0:
            return default
        val = rows.select(pl.col(key)).to_series()[0]
        return default if val is None else val

    def set_edge_layer_attrs(self, layer_id, edge_id, **attrs):
        """Upsert per-layer attributes for a specific edge.

        Parameters
        ----------
        layer_id : str
        edge_id : str
        **attrs
            Pure attributes. Structural keys are ignored (except 'weight', which is allowed here).

        """
        # allow 'weight' through; keep ignoring true structural keys
        clean = {
            k: v for k, v in attrs.items() if (k not in self._EDGE_RESERVED) or (k == "weight")
        }
        if not clean:
            return

        # Normalize hot keys (intern) and avoid float dtype surprises for 'weight'
        try:
            import sys as _sys

            if isinstance(layer_id, str):
                layer_id = _sys.intern(layer_id)
            if isinstance(edge_id, str):
                edge_id = _sys.intern(edge_id)
        except Exception:
            pass
        if "weight" in clean:
            try:
                # cast once to float to reduce dtype mismatch churn inside _upsert_row
                clean["weight"] = float(clean["weight"])
            except Exception:
                # leave as-is if not coercible; behavior stays identical
                pass

        # Ensure edge_layer_attributes compares strings to strings (defensive against prior bad writes),
        # but only cast when actually needed (skip no-op with_columns).
        df = self.edge_layer_attributes
        if isinstance(df, pl.DataFrame) and df.height > 0:
            to_cast = []
            if "layer_id" in df.columns and df.schema["layer_id"] != pl.Utf8:
                to_cast.append(pl.col("layer_id").cast(pl.Utf8))
            if "edge_id" in df.columns and df.schema["edge_id"] != pl.Utf8:
                to_cast.append(pl.col("edge_id").cast(pl.Utf8))
            if to_cast:
                df = df.with_columns(*to_cast)
                self.edge_layer_attributes = df  # reassign only when changed

        # Upsert via central helper (keeps exact behavior, schema handling, and caching)
        self.edge_layer_attributes = self._upsert_row(
            self.edge_layer_attributes, (layer_id, edge_id), clean
        )

    def get_edge_layer_attr(self, layer_id, edge_id, key, default=None):
        """Get a per-layer attribute for an edge.

        Parameters
        ----------
        layer_id : str
        edge_id : str
        key : str
        default : Any, optional

        Returns
        -------
        Any

        """
        df = self.edge_layer_attributes
        if key not in df.columns:
            return default
        rows = df.filter((pl.col("layer_id") == layer_id) & (pl.col("edge_id") == edge_id))
        if rows.height == 0:
            return default
        val = rows.select(pl.col(key)).to_series()[0]
        return default if val is None else val

    def set_layer_edge_weight(self, layer_id, edge_id, weight):  # legacy weight helper
        """Set a legacy per-layer weight override for an edge.

        Parameters
        ----------
        layer_id : str
        edge_id : str
        weight : float

        Raises
        ------
        KeyError
            If the layer or edge does not exist.

        See Also
        --------
        get_effective_edge_weight

        """
        if layer_id not in self._layers:
            raise KeyError(f"Layer {layer_id} not found")
        if edge_id not in self.edge_to_idx:
            raise KeyError(f"Edge {edge_id} not found")
        self.layer_edge_weights[layer_id][edge_id] = float(weight)

    def get_effective_edge_weight(self, edge_id, layer=None):
        """Resolve the effective weight for an edge, optionally within a layer.

        Parameters
        ----------
        edge_id : str
        layer : str, optional
            If provided, return the layer override if present; otherwise global weight.

        Returns
        -------
        float
            Effective weight.

        """
        if layer is not None:
            df = self.edge_layer_attributes
            if (
                isinstance(df, pl.DataFrame)
                and df.height > 0
                and {"layer_id", "edge_id", "weight"} <= set(df.columns)
            ):
                rows = df.filter(
                    (pl.col("layer_id") == layer) & (pl.col("edge_id") == edge_id)
                ).select("weight")
                if rows.height > 0:
                    w = rows.to_series()[0]
                    if w is not None and not (isinstance(w, float) and math.isnan(w)):
                        return float(w)

            # fallback to legacy dict if present
            w2 = self.layer_edge_weights.get(layer, {}).get(edge_id, None)
            if w2 is not None:
                return float(w2)

        return float(self.edge_weights[edge_id])

    def audit_attributes(self):
        """Audit attribute tables for extra/missing rows and invalid edge-layer pairs.

        Returns
        -------
        dict
            {
            'extra_vertex_rows': list[str],
            'extra_edge_rows': list[str],
            'missing_vertex_rows': list[str],
            'missing_edge_rows': list[str],
            'invalid_edge_layer_rows': list[tuple[str, str]],
            }

        """
        vertex_ids = {eid for eid, t in self.entity_types.items() if t == "vertex"}
        edge_ids = set(self.edge_to_idx.keys())

        na = self.vertex_attributes
        ea = self.edge_attributes
        ela = self.edge_layer_attributes

        vertex_attr_ids = (
            set(na.select("vertex_id").to_series().to_list())
            if isinstance(na, pl.DataFrame) and na.height > 0 and "vertex_id" in na.columns
            else set()
        )
        edge_attr_ids = (
            set(ea.select("edge_id").to_series().to_list())
            if isinstance(ea, pl.DataFrame) and ea.height > 0 and "edge_id" in ea.columns
            else set()
        )

        extra_vertex_rows = [i for i in vertex_attr_ids if i not in vertex_ids]
        extra_edge_rows = [i for i in edge_attr_ids if i not in edge_ids]
        missing_vertex_rows = [i for i in vertex_ids if i not in vertex_attr_ids]
        missing_edge_rows = [i for i in edge_ids if i not in edge_attr_ids]

        bad_edge_layer = []
        if (
            isinstance(ela, pl.DataFrame)
            and ela.height > 0
            and {"layer_id", "edge_id"} <= set(ela.columns)
        ):
            for lid, eid in ela.select(["layer_id", "edge_id"]).iter_rows():
                if lid not in self._layers or eid not in edge_ids:
                    bad_edge_layer.append((lid, eid))

        return {
            "extra_vertex_rows": extra_vertex_rows,
            "extra_edge_rows": extra_edge_rows,
            "missing_vertex_rows": missing_vertex_rows,
            "missing_edge_rows": missing_edge_rows,
            "invalid_edge_layer_rows": bad_edge_layer,
        }

    def _pl_dtype_for_value(self, v):
        """INTERNAL: Infer an appropriate Polars dtype for a Python value.

        Parameters
        ----------
        v : Any

        Returns
        -------
        polars.datatypes.DataType
            One of ``pl.Null``, ``pl.Boolean``, ``pl.Int64``, ``pl.Float64``,
            ``pl.Utf8``, ``pl.Binary``, ``pl.Object``, or ``pl.List(inner)``.

        Notes
        -----
        - Enums are mapped to ``pl.Object`` (useful for categorical enums).
        - Lists/tuples infer inner dtype from the first element (defaults to ``Utf8``).

        """
        import enum

        import polars as pl

        if v is None:
            return pl.Null
        if isinstance(v, bool):
            return pl.Boolean
        if isinstance(v, int) and not isinstance(v, bool):
            return pl.Int64
        if isinstance(v, float):
            return pl.Float64
        if isinstance(v, enum.Enum):
            return pl.Object  # important for EdgeType
        if isinstance(v, (bytes, bytearray)):
            return pl.Binary
        if isinstance(v, (list, tuple)):
            inner = self._pl_dtype_for_value(v[0]) if len(v) else pl.Utf8
            return pl.List(pl.Utf8 if inner == pl.Null else inner)
        if isinstance(v, dict):
            return pl.Object
        return pl.Utf8

    def _ensure_attr_columns(self, df: pl.DataFrame, attrs: dict) -> pl.DataFrame:
        """INTERNAL: Create/align attribute columns and dtypes to accept ``attrs``.

        Parameters
        ----------
        df : polars.DataFrame
            Existing attribute table.
        attrs : dict
            Incoming key/value pairs to upsert.

        Returns
        -------
        polars.DataFrame
            DataFrame with columns added/cast so inserts/updates won't hit ``Null`` dtypes.

        Notes
        -----
        - New columns are created with the inferred dtype.
        - If a column is ``Null`` and the incoming value is not, it is cast to the inferred dtype.
        - If dtypes conflict (mixed over time), both sides upcast to ``Utf8`` to avoid schema errors.

        """
        schema = df.schema
        for col, val in attrs.items():
            target = self._pl_dtype_for_value(val)
            if col not in schema:
                df = df.with_columns(pl.lit(None).cast(target).alias(col))
            else:
                cur = schema[col]
                if cur == pl.Null and target != pl.Null:
                    df = df.with_columns(pl.col(col).cast(target))
                # if mixed types are expected over time, upcast to Utf8:
                elif cur != target and target != pl.Null:
                    # upcast both sides to Utf8 to avoid schema conflicts
                    df = df.with_columns(pl.col(col).cast(pl.Utf8))
        return df

    def _upsert_row(self, df: pl.DataFrame, idx, attrs: dict) -> pl.DataFrame:
        """INTERNAL: Upsert a row in a Polars DF [DataFrame] using explicit key columns.

        Keys
        ----
        - ``vertex_attributes``           → key: ``["vertex_id"]``
        - ``edge_attributes``             → key: ``["edge_id"]``
        - ``layer_attributes``            → key: ``["layer_id"]``
        - ``edge_layer_attributes``       → key: ``["layer_id", "edge_id"]``
        """
        if not isinstance(attrs, dict) or not attrs:
            return df

        cols = set(df.columns)

        # Determine key columns + values
        if {"layer_id", "edge_id"} <= cols:
            if not (isinstance(idx, tuple) and len(idx) == 2):
                raise ValueError("idx must be a (layer_id, edge_id) tuple")
            key_cols = ("layer_id", "edge_id")
            key_vals = {"layer_id": idx[0], "edge_id": idx[1]}
            cache_name = "_edge_layer_attr_keys"  # set of (layer_id, edge_id)
            df_id_name = "_edge_layer_attr_df_id"
        elif "vertex_id" in cols:
            key_cols = ("vertex_id",)
            key_vals = {"vertex_id": idx}
            cache_name = "_vertex_attr_ids"  # set of vertex_id
            df_id_name = "_vertex_attr_df_id"
        elif "edge_id" in cols:
            key_cols = ("edge_id",)
            key_vals = {"edge_id": idx}
            cache_name = "_edge_attr_ids"  # set of edge_id
            df_id_name = "_edge_attr_df_id"
        elif "layer_id" in cols:
            key_cols = ("layer_id",)
            key_vals = {"layer_id": idx}
            cache_name = "_layer_attr_ids"  # set of layer_id
            df_id_name = "_layer_attr_df_id"
        else:
            raise ValueError("Cannot infer key columns from DataFrame schema")

        # Ensure attribute columns exist / are cast appropriately
        df = self._ensure_attr_columns(df, attrs)

        # Build the match condition (used later for updates)
        cond = None
        for k in key_cols:
            v = key_vals[k]
            c = pl.col(k) == pl.lit(v)
            cond = c if cond is None else (cond & c)

        # existence check via small per-table caches (no DF scan)
        try:
            key_cache = getattr(self, cache_name, None)
            cached_df_id = getattr(self, df_id_name, None)
            if key_cache is None or cached_df_id != id(df):
                # Rebuild cache lazily for the current df object
                if "vertex_id" in cols and key_cols == ("vertex_id",):
                    series = df.get_column("vertex_id") if df.height else pl.Series([])
                    key_cache = set(series.to_list()) if df.height else set()
                elif (
                    "edge_id" in cols and "layer_id" in cols and key_cols == ("layer_id", "edge_id")
                ):
                    if df.height:
                        key_cache = set(
                            zip(
                                df.get_column("layer_id").to_list(),
                                df.get_column("edge_id").to_list(),
                            )
                        )
                    else:
                        key_cache = set()
                elif "edge_id" in cols and key_cols == ("edge_id",):
                    series = df.get_column("edge_id") if df.height else pl.Series([])
                    key_cache = set(series.to_list()) if df.height else set()
                elif "layer_id" in cols and key_cols == ("layer_id",):
                    series = df.get_column("layer_id") if df.height else pl.Series([])
                    key_cache = set(series.to_list()) if df.height else set()
                else:
                    key_cache = set()
                setattr(self, cache_name, key_cache)
                setattr(self, df_id_name, id(df))
            # Decide existence from cache
            cache_key = (
                key_vals[key_cols[0]]
                if len(key_cols) == 1
                else (key_vals["layer_id"], key_vals["edge_id"])
            )
            exists = cache_key in key_cache
        except Exception:
            # Fallback to original behavior if caching fails
            exists = df.filter(cond).height > 0
            key_cache = None

        if exists:
            # cast literals to column dtypes; keep exact semantics
            schema = df.schema
            upds = []
            for k, v in attrs.items():
                tgt_dtype = schema[k]
                upds.append(
                    pl.when(cond).then(pl.lit(v).cast(tgt_dtype)).otherwise(pl.col(k)).alias(k)
                )
            new_df = df.with_columns(upds)

            # Keep cache pointers in sync with the new df object
            try:
                setattr(self, df_id_name, id(new_df))
                # cache contents unchanged for updates
            except Exception:
                pass

            return new_df

        # build a single row aligned to df schema
        schema = df.schema

        # Start with None for all columns, fill keys and attrs
        new_row = dict.fromkeys(df.columns)
        new_row.update(key_vals)
        new_row.update(attrs)

        to_append = pl.DataFrame([new_row])

        # 1) Ensure to_append has all df columns
        for c in df.columns:
            if c not in to_append.columns:
                to_append = to_append.with_columns(pl.lit(None).cast(schema[c]).alias(c))

        # 2) Resolve dtype mismatches:
        #    - df Null + to_append non-Null -> cast df to right
        #    - to_append Null + df non-Null -> cast to_append to left
        #    - left != right -> upcast both to Utf8
        left_schema = schema
        right_schema = to_append.schema
        df_casts = []
        app_casts = []
        for c in df.columns:
            left = left_schema[c]
            right = right_schema[c]
            if left == pl.Null and right != pl.Null:
                df_casts.append(pl.col(c).cast(right))
            elif right == pl.Null and left != pl.Null:
                app_casts.append(pl.col(c).cast(left).alias(c))
            elif left != right:
                df_casts.append(pl.col(c).cast(pl.Utf8))
                app_casts.append(pl.col(c).cast(pl.Utf8).alias(c))

        if df_casts:
            df = df.with_columns(df_casts)
            left_schema = df.schema  # refresh for correctness
        if app_casts:
            to_append = to_append.with_columns(app_casts)

        new_df = df.vstack(to_append)

        # Update caches after insertion
        try:
            if key_cache is not None:
                if len(key_cols) == 1:
                    key_cache.add(cache_key)
                else:
                    key_cache.add(cache_key)
            setattr(self, df_id_name, id(new_df))
        except Exception:
            pass

        return new_df

    def _variables_watched_by_vertices(self):
        # set of vertex-attribute names used by vertex-scope policies
        return {p["var"] for p in self.edge_direction_policy.values()
                if p.get("scope", "edge") == "vertex"}

    def _incident_flexible_edges(self, v):
        # naive scan; optimize later with an index if needed
        out = []
        for eid, (s, t, _kind) in self.edge_definitions.items():
            if eid in self.edge_direction_policy and (s == v or t == v):
                out.append(eid)
        return out

    def _apply_flexible_direction(self, edge_id):
        pol = self.edge_direction_policy.get(edge_id)
        if not pol: return

        src, tgt, _ = self.edge_definitions[edge_id]
        col = self.edge_to_idx[edge_id]
        w   = float(self.edge_weights.get(edge_id, 1.0))

        var  = pol["var"];  T = float(pol["threshold"])
        scope = pol.get("scope", "edge")   # 'edge'|'vertex'
        above = pol.get("above", "s->t")   # 's->t'|'t->s'
        tie   = pol.get("tie", "keep")     # default behavior

        # decide condition and detect tie
        tie_case = False
        if scope == "edge":
            x = self.get_attr_edge(edge_id, var, None)
            if x is None: return
            if x == T: tie_case = True
            cond = (x > T)
        else:
            xs = self.get_attr_vertex(src, var, None)
            xt = self.get_attr_vertex(tgt, var, None)
            if xs is None or xt is None: return
            if xs == xt: tie_case = True
            cond = (xs - xt) > 0

        M  = self._matrix
        si = self.entity_to_idx[src]; ti = self.entity_to_idx[tgt]

        if tie_case:
            if tie == "keep":
                # do nothing → previous signs remain (default)
                return
            if tie == "undirected":
                # force (+w,+w) while equality holds
                M[(si, col)] = +w
                if src != tgt: M[(ti, col)] = +w
                return
            # force a direction at equality
            cond = True if tie == "s->t" else False

        # rewrite as directed per 'above'
        M[(si, col)] = 0; M[(ti, col)] = 0
        src_to_tgt = cond if above == "s->t" else (not cond)
        if src_to_tgt:
            M[(si, col)] = +w
            if src != tgt: M[(ti, col)] = -w
        else:
            M[(si, col)] = -w
            if src != tgt: M[(ti, col)] = +w

    ## Full attribute dict for a single entity

    def get_edge_attrs(self, edge) -> dict:
        """Return the full attribute dict for a single edge.

        Parameters
        ----------
        edge : int | str
            Edge index (int) or edge id (str).

        Returns
        -------
        dict
            Attribute dictionary for that edge. {} if not found.

        """
        # normalize to edge id
        if isinstance(edge, int):
            eid = self.idx_to_edge[edge]
        else:
            eid = edge

        df = self.edge_attributes
        # Polars-safe: iterate the (at most one) row as a dict
        try:
            import polars as pl

            for row in df.filter(pl.col("edge_id") == eid).iter_rows(named=True):
                return dict(row)
            return {}
        except Exception:
            # Fallback if df is pandas or dict-like
            try:
                row = df[df["edge_id"] == eid].to_dict(orient="records")
                return row[0] if row else {}
            except Exception:
                return {}

    def get_vertex_attrs(self, vertex) -> dict:
        """Return the full attribute dict for a single vertex.

        Parameters
        ----------
        vertex : str
            Vertex id.

        Returns
        -------
        dict
            Attribute dictionary for that vertex. {} if not found.

        """
        df = self.vertex_attributes
        try:
            import polars as pl

            for row in df.filter(pl.col("vertex_id") == vertex).iter_rows(named=True):
                return dict(row)
            return {}
        except Exception:
            try:
                row = df[df["vertex_id"] == vertex].to_dict(orient="records")
                return row[0] if row else {}
            except Exception:
                return {}

    ## Bulk attributes

    def get_attr_edges(self, indexes=None) -> dict:
        """Retrieve edge attributes as a dictionary.

        Parameters
        ----------
        indexes : Iterable[int] | None, optional
            A list or iterable of edge indices to retrieve attributes for.
            - If `None` (default), attributes for **all** edges are returned.
            - If provided, only those edges will be included in the output.

        Returns
        -------
        dict[str, dict]
            A dictionary mapping `edge_id` → `attribute_dict`, where:
            - `edge_id` is the unique string identifier of the edge.
            - `attribute_dict` is a dictionary of attribute names and values.

        Notes
        -----
        - This function reads directly from `self.edge_attributes`, which should be
        a Polars DataFrame where each row corresponds to an edge.
        - Useful for bulk inspection, serialization, or analytics without looping manually.

        """
        df = self.edge_attributes
        if indexes is not None:
            df = df.filter(pl.col("edge_id").is_in([self.idx_to_edge[i] for i in indexes]))
        return {row["edge_id"]: row.as_dict() for row in df.iter_rows(named=True)}

    def get_attr_vertices(self, vertices=None) -> dict:
        """Retrieve vertex (vertex) attributes as a dictionary.

        Parameters
        ----------
        vertices : Iterable[str] | None, optional
            A list or iterable of vertex IDs to retrieve attributes for.
            - If `None` (default), attributes for **all** verices are returned.
            - If provided, only those verices will be included in the output.

        Returns
        -------
        dict[str, dict]
            A dictionary mapping `vertex_id` → `attribute_dict`, where:
            - `vertex_id` is the unique string identifier of the vertex.
            - `attribute_dict` is a dictionary of attribute names and values.

        Notes
        -----
        - This reads from `self.vertex_attributes`, which stores per-vertex metadata.
        - Use this for bulk data extraction instead of repeated single-vertex calls.

        """
        df = self.vertex_attributes
        if vertices is not None:
            df = df.filter(pl.col("vertex_id").is_in(vertices))
        return {row["vertex_id"]: row.as_dict() for row in df.iter_rows(named=True)}

    def get_attr_from_edges(self, key: str, default=None) -> dict:
        """Extract a specific attribute column for all edges.

        Parameters
        ----------
        key : str
            Attribute column name to extract from `self.edge_attributes`.
        default : Any, optional
            Default value to use if the column does not exist or if an edge
            does not have a value. Defaults to `None`.

        Returns
        -------
        dict[str, Any]
            A dictionary mapping `edge_id` → attribute value.

        Notes
        -----
        - If the requested column is missing, all edges return `default`.
        - This is useful for quick property lookups (e.g., weight, label, type).

        """
        df = self.edge_attributes
        if key not in df.columns:
            return {row["edge_id"]: default for row in df.iter_rows(named=True)}
        return {
            row["edge_id"]: row[key] if row[key] is not None else default
            for row in df.iter_rows(named=True)
        }

    def get_edges_by_attr(self, key: str, value) -> list:
        """Retrieve all edges where a given attribute equals a specific value.

        Parameters
        ----------
        key : str
            Attribute column name to filter on.
        value : Any
            Value to match.

        Returns
        -------
        list[str]
            A list of edge IDs where the attribute `key` equals `value`.

        Notes
        -----
        - If the attribute column does not exist, an empty list is returned.
        - Comparison is exact; consider normalizing types before calling.

        """
        df = self.edge_attributes
        if key not in df.columns:
            return []
        return [row["edge_id"] for row in df.iter_rows(named=True) if row[key] == value]

    def get_graph_attributes(self) -> dict:
        """Return a shallow copy of the graph-level attributes dictionary.

        Returns
        -------
        dict
            A dictionary of global metadata describing the graph as a whole.
            Typical keys might include:
            - `"name"` : Graph name or label.
            - `"directed"` : Boolean indicating directedness.
            - `"layers"` : List of layers present in the graph.
            - `"created_at"` : Timestamp of graph creation.

        Notes
        -----
        - Returns a **shallow copy** to prevent external mutation of internal state.
        - Graph-level attributes are meant to store metadata not tied to individual
        verices or edges (e.g., versioning info, provenance, global labels).

        """
        return dict(self.graph_attributes)

    def set_edge_layer_attrs_bulk(self, layer_id, items):
        """items: iterable of (edge_id, attrs_dict) or dict{edge_id: attrs_dict}
        Upserts rows in edge_layer_attributes for one layer in bulk.
        """
        import polars as pl

        # normalize
        rows = []
        if isinstance(items, dict):
            it = items.items()
        else:
            it = items
        for eid, attrs in it:
            if not isinstance(attrs, dict) or not attrs:
                continue
            r = {"layer_id": layer_id, "edge_id": eid}
            r.update(attrs)
            if "weight" in r:
                try:
                    r["weight"] = float(r["weight"])
                except Exception:
                    pass
            rows.append(r)
        if not rows:
            return

        # start from current DF
        df = self.edge_layer_attributes
        add_df = pl.DataFrame(rows)

        # ensure required key cols exist/correct dtype on existing df
        if not isinstance(df, pl.DataFrame) or df.is_empty():
            # create from scratch with canonical dtypes
            self.edge_layer_attributes = add_df
            # legacy mirror
            if "weight" in add_df.columns:
                self.layer_edge_weights.setdefault(layer_id, {})
                for r in add_df.iter_rows(named=True):
                    w = r.get("weight")
                    if w is not None:
                        self.layer_edge_weights[layer_id][r["edge_id"]] = float(w)
            return

        # schema alignment using your _ensure_attr_columns + Utf8 upcast rule
        need_cols = {c: None for c in add_df.columns if c not in df.columns}
        if need_cols:
            df = self._ensure_attr_columns(df, need_cols)  # adds missing columns to df
        # add missing columns to add_df
        for c in df.columns:
            if c not in add_df.columns:
                add_df = add_df.with_columns(pl.lit(None).cast(df.schema[c]).alias(c))
        # reconcile dtype mismatches (Null/Null, mixed -> Utf8), same policy as _upsert_row
        for c in df.columns:
            lc, rc = df.schema[c], add_df.schema[c]
            if lc == pl.Null and rc != pl.Null:
                df = df.with_columns(pl.col(c).cast(rc))
            elif rc == pl.Null and lc != pl.Null:
                add_df = add_df.with_columns(pl.col(c).cast(lc).alias(c))
            elif lc != rc:
                df = df.with_columns(pl.col(c).cast(pl.Utf8))
                add_df = add_df.with_columns(pl.col(c).cast(pl.Utf8).alias(c))

        # drop existing keys for (layer_id, edge_id) we are about to write; then vstack new rows
        mask_keep = ~(
            (pl.col("layer_id") == layer_id) & pl.col("edge_id").is_in(add_df.get_column("edge_id"))
        )
        df = df.filter(mask_keep)
        df = df.vstack(add_df)
        self.edge_layer_attributes = df

        # legacy mirror
        if "weight" in add_df.columns:
            self.layer_edge_weights.setdefault(layer_id, {})
            for r in add_df.iter_rows(named=True):
                w = r.get("weight")
                if w is not None:
                    self.layer_edge_weights[layer_id][r["edge_id"]] = float(w)

    # Basic queries & metrics

    def get_vertex(self, index: int) -> str:
        """Return the vertex ID corresponding to a given internal index.

        Parameters
        ----------
        index : int
            The internal vertex index.

        Returns
        -------
        str
            The vertex ID.

        """
        return self.idx_to_entity[index]

    def get_edge(self, index: int):
        """Return edge endpoints in a canonical form.

        Parameters
        ----------
        index : int
            Internal edge index.

        Returns
        -------
        tuple[frozenset, frozenset]
            (S, T) where S and T are frozensets of vertex IDs.
            - For directed binary edges: ({u}, {v})
            - For undirected binary edges: (M, M)
            - For directed hyperedges: (head_set, tail_set)
            - For undirected hyperedges: (members, members)

        """
        if isinstance(index, str):
            eid = index
            try:
                index = self.edge_to_idx[eid]
            except KeyError:
                raise KeyError(f"Unknown edge id: {eid}") from None
        else:
            eid = self.idx_to_edge[index]

        kind = self.edge_kind.get(eid)

        eid = self.idx_to_edge[index]
        kind = self.edge_kind.get(eid)

        if kind == "hyper":
            meta = self.hyperedge_definitions[eid]
            if meta.get("directed", False):
                return (frozenset(meta["head"]), frozenset(meta["tail"]))
            else:
                M = frozenset(meta["members"])
                return (M, M)
        else:
            u, v, _etype = self.edge_definitions[eid]
            directed = self.edge_directed.get(eid, True if self.directed is None else self.directed)
            if directed:
                return (frozenset([u]), frozenset([v]))
            else:
                M = frozenset([u, v])
                return (M, M)

    def incident_edges(self, vertex_id) -> list[int]:
        """Return all edge indices incident to a given vertex.

        Parameters
        ----------
        vertex_id : str
            vertex identifier.

        Returns
        -------
        list[int]
            List of edge indices incident to the vertex.

        """
        incident = []
        # Fast path: direct matrix row lookup if available
        if vertex_id in self.entity_to_idx:
            row_idx = self.entity_to_idx[vertex_id]
            try:
                incident.extend(self._matrix.tocsr().getrow(row_idx).indices.tolist())
                return incident
            except Exception:
                # fallback if matrix is not in CSR (compressed sparse row) format
                pass

        # Fallback: scan edge definitions
        for j in range(self.number_of_edges()):
            eid = self.idx_to_edge[j]
            kind = self.edge_kind.get(eid)
            if kind == "hyper":
                meta = self.hyperedge_definitions[eid]
                if (
                    meta.get("directed", False)
                    and (vertex_id in meta["head"] or vertex_id in meta["tail"])
                ) or (not meta.get("directed", False) and vertex_id in meta["members"]):
                    incident.append(j)
            else:
                u, v, _etype = self.edge_definitions[eid]
                if vertex_id == u or vertex_id == v:
                    incident.append(j)

        return incident

    def _is_directed_edge(self, edge_id):
        """Check if an edge is directed (per-edge flag overrides graph default).

        Parameters
        ----------
        edge_id : str

        Returns
        -------
        bool

        """
        return bool(self.edge_directed.get(edge_id, self.directed))

    def has_edge(self, source, target, edge_id=None):
        """Test for the existence of an edge.

        Parameters
        ----------
        source : str
        target : str
        edge_id : str, optional
            If provided, check for this specific ID.

        Returns
        -------
        bool

        """
        if edge_id:
            return edge_id in self.edge_to_idx

        # Check any edge between source and target
        for eid, (src, tgt, _) in self.edge_definitions.items():
            if src == source and tgt == target:
                return True
        return False

    def has_vertex(self, vertex_id: str) -> bool:
        """Test for the existence of a vertex.

        Parameters
        ----------
        vertex_id : str

        Returns
        -------
        bool

        """
        return vertex_id in self.entity_to_idx and self.entity_types.get(vertex_id) == "vertex"

    def get_edge_ids(self, source, target):
        """List all edge IDs between two endpoints.

        Parameters
        ----------
        source : str
        target : str

        Returns
        -------
        list[str]
            Edge IDs (may be empty).

        """
        edge_ids = []
        for eid, (src, tgt, _) in self.edge_definitions.items():
            if src == source and tgt == target:
                edge_ids.append(eid)
        return edge_ids

    def degree(self, entity_id):
        """Degree of a vertex or edge-entity (number of incident non-zero entries).

        Parameters
        ----------
        entity_id : str

        Returns
        -------
        int

        """
        if entity_id not in self.entity_to_idx:
            return 0

        entity_idx = self.entity_to_idx[entity_id]
        row = self._matrix.getrow(entity_idx)
        return len(row.nonzero()[1])

    def vertices(self):
        """Get all vertex IDs (excluding edge-entities).

        Returns
        -------
        list[str]

        """
        return [eid for eid, etype in self.entity_types.items() if etype == "vertex"]

    def edges(self):
        """Get all edge IDs.

        Returns
        -------
        list[str]

        """
        return list(self.edge_to_idx.keys())

    def edge_list(self):
        """Materialize (source, target, edge_id, weight) for binary/vertex-edge edges.

        Returns
        -------
        list[tuple[str, str, str, float]]

        """
        edges = []
        for edge_id, (source, target, edge_type) in self.edge_definitions.items():
            weight = self.edge_weights[edge_id]
            edges.append((source, target, edge_id, weight))
        return edges

    def get_directed_edges(self):
        """List IDs of directed edges.

        Returns
        -------
        list[str]

        """
        default_dir = True if self.directed is None else self.directed
        return [eid for eid in self.edge_to_idx.keys() if self.edge_directed.get(eid, default_dir)]

    def get_undirected_edges(self):
        """List IDs of undirected edges.

        Returns
        -------
        list[str]

        """
        default_dir = True if self.directed is None else self.directed
        return [
            eid for eid in self.edge_to_idx.keys() if not self.edge_directed.get(eid, default_dir)
        ]

    def number_of_vertices(self):
        """Count vertices (excluding edge-entities).

        Returns
        -------
        int

        """
        return len([e for e in self.entity_types.values() if e == "vertex"])

    def number_of_edges(self):
        """Count edges (columns in the incidence matrix).

        Returns
        -------
        int

        """
        return self._num_edges

    def global_entity_count(self):
        """Count unique entities present across all layers (union of memberships).

        Returns
        -------
        int

        """
        all_vertices = set()
        for layer_data in self._layers.values():
            all_vertices.update(layer_data["vertices"])
        return len(all_vertices)

    def global_edge_count(self):
        """Count unique edges present across all layers (union of memberships).

        Returns
        -------
        int

        """
        all_edges = set()
        for layer_data in self._layers.values():
            all_edges.update(layer_data["edges"])
        return len(all_edges)

    def in_edges(self, vertices):
        """Iterate over all edges that are **incoming** to one or more vertices.

        Parameters
        ----------
        vertices : str | Iterable[str]
            A single vertex ID or an iterable of vertex IDs. All edges whose
            **target set** intersects with this set will be yielded.

        Yields
        ------
        tuple[int, tuple[frozenset, frozenset]]
            Tuples of the form `(edge_index, (S, T))`, where:
            - `edge_index` : int — internal integer index of the edge.
            - `S` : frozenset[str] — set of source/head verices.
            - `T` : frozenset[str] — set of target/tail verices.

        Behavior
        --------
        - **Directed binary edges**: returned if any vertex is in the target (`T`).
        - **Directed hyperedges**: returned if any vertex is in the tail set.
        - **Undirected edges/hyperedges**: returned if any vertex is in
        the edge's member set (`S ∪ T`).

        Notes
        -----
        - Works with binary and hyperedges.
        - Undirected edges appear in both `in_edges()` and `out_edges()`.
        - The returned `(S, T)` is the canonical form from `get_edge()`.

        """
        V = self._normalize_vertices_arg(vertices)
        if not V:
            return
        for j in range(self.number_of_edges()):
            S, T = self.get_edge(j)
            eid = self.idx_to_edge[j]
            directed = self._is_directed_edge(eid)
            if directed:
                if T & V:
                    yield j, (S, T)
            else:
                if (S | T) & V:
                    yield j, (S, T)

    def out_edges(self, vertices):
        """Iterate over all edges that are **outgoing** from one or more vertices.

        Parameters
        ----------
        vertices : str | Iterable[str]
            A single vertex ID or an iterable of vertex IDs. All edges whose
            **source set** intersects with this set will be yielded.

        Yields
        ------
        tuple[int, tuple[frozenset, frozenset]]
            Tuples of the form `(edge_index, (S, T))`, where:
            - `edge_index` : int — internal integer index of the edge.
            - `S` : frozenset[str] — set of source/head verices.
            - `T` : frozenset[str] — set of target/tail verices.

        Behavior
        --------
        - **Directed binary edges**: returned if any vertex is in the source (`S`).
        - **Directed hyperedges**: returned if any vertex is in the head set.
        - **Undirected edges/hyperedges**: returned if any vertex is in
        the edge's member set (`S ∪ T`).

        Notes
        -----
        - Works with binary and hyperedges.
        - Undirected edges appear in both `out_edges()` and `in_edges()`.
        - The returned `(S, T)` is the canonical form from `get_edge()`.

        """
        V = self._normalize_vertices_arg(vertices)
        if not V:
            return
        for j in range(self.number_of_edges()):
            S, T = self.get_edge(j)
            eid = self.idx_to_edge[j]
            directed = self._is_directed_edge(eid)
            if directed:
                if S & V:
                    yield j, (S, T)
            else:
                if (S | T) & V:
                    yield j, (S, T)

    def get_or_create_vertex_by_attrs(self, layer=None, **attrs) -> str:
        """Return vertex_id for the given composite-key attributes, creating the vertex if needed.

        - Requires set_vertex_key(...) to have been called.
        - All key fields must be present and non-null in attrs.
        """
        if not self._vertex_key_fields:
            raise RuntimeError("Call set_vertex_key(...) before using get_or_create_vertex_by_attrs")

        key = self._build_key_from_attrs(attrs)
        if key is None:
            missing = [f for f in self._vertex_key_fields if f not in attrs or attrs[f] is None]
            raise ValueError(f"Missing composite key fields: {missing}")

        # Existing?
        owner = self._vertex_key_index.get(key)
        if owner is not None:
            return owner

        # Create new vertex
        vid = self._gen_vertex_id_from_key(key)
        # No need to pre-check entity_to_idx here; ids are namespaced by 'kv:' prefix
        self.add_vertex(vid, layer=layer, **attrs)

        # Index ownership
        self._vertex_key_index[key] = vid
        return vid

    def vertex_key_tuple(self, vertex_id) -> tuple | None:
        """Return the composite-key tuple for vertex_id (None if incomplete or no key set)."""
        return self._current_key_of_vertex(vertex_id)

    @property
    def V(self):
        """All vertices as a tuple.

        Returns
        -------
        tuple
            Tuple of all vertex IDs in the graph.

        """
        return tuple(self.vertices())

    @property
    def E(self):
        """All edges as a tuple.

        Returns
        -------
        tuple
            Tuple of all edge identifiers (whatever `self.edges()` yields).

        """
        return tuple(self.edges())

    @property
    def num_vertices(self):
        """Total number of vertices (vertices) in the graph."""
        return self.number_of_vertices()

    @property
    def num_edges(self):
        """Total number of edges in the graph."""
        return self.number_of_edges()

    @property
    def nv(self):
        """Shorthand for num_vertices."""
        return self.num_vertices

    @property
    def ne(self):
        """Shorthand for num_edges."""
        return self.num_edges

    @property
    def shape(self):
        """Graph shape as a tuple: (num_vertices, num_edges).
        Useful for quick inspection.
        """
        return (self.num_vertices, self.num_edges)

    # Materialized views

    def edges_view(
        self,
        layer=None,
        include_directed=True,
        include_weight=True,
        resolved_weight=True,
        copy=True,
    ):
        """Build a Polars DF [DataFrame] view of edges with optional layer join.
        Same columns/semantics as before, but vectorized (no per-edge DF scans).
        """
        # Fast path: no edges
        if not self.edge_to_idx:
            return pl.DataFrame(schema={"edge_id": pl.Utf8, "kind": pl.Utf8})

        eids = list(self.edge_to_idx.keys())
        kinds = [self.edge_kind.get(eid, "binary") for eid in eids]

        # columns we might need
        need_global = include_weight or resolved_weight
        global_w = [self.edge_weights.get(eid, None) for eid in eids] if need_global else None
        dirs = (
            [
                self.edge_directed.get(eid, True if self.directed is None else self.directed)
                for eid in eids
            ]
            if include_directed
            else None
        )

        # endpoints / hyper metadata (one pass; no weight lookups)
        src, tgt, etype = [], [], []
        head, tail, members = [], [], []
        for eid, k in zip(eids, kinds):
            if k == "hyper":
                # hyperedge: store sets in canonical sorted tuples
                h = self.hyperedge_definitions[eid]
                if h.get("directed", False):
                    head.append(tuple(sorted(h.get("head", ()))))
                    tail.append(tuple(sorted(h.get("tail", ()))))
                    members.append(None)
                else:
                    head.append(None)
                    tail.append(None)
                    members.append(tuple(sorted(h.get("members", ()))))
                src.append(None)
                tgt.append(None)
                etype.append(None)
            else:
                s, t, et = self.edge_definitions[eid]
                src.append(s)
                tgt.append(t)
                etype.append(et)
                head.append(None)
                tail.append(None)
                members.append(None)

        # base frame
        cols = {"edge_id": eids, "kind": kinds}
        if include_directed:
            cols["directed"] = dirs
        if include_weight:
            cols["global_weight"] = global_w
        # we still need global weight transiently to compute effective weight even if not displayed
        if resolved_weight and not include_weight:
            cols["_gw_tmp"] = global_w

        base = pl.DataFrame(cols).with_columns(
            pl.Series("source", src, dtype=pl.Utf8),
            pl.Series("target", tgt, dtype=pl.Utf8),
            pl.Series("edge_type", etype, dtype=pl.Utf8),
            pl.Series("head", head, dtype=pl.List(pl.Utf8)),
            pl.Series("tail", tail, dtype=pl.List(pl.Utf8)),
            pl.Series("members", members, dtype=pl.List(pl.Utf8)),
        )

        # join pure edge attributes (left)
        if isinstance(self.edge_attributes, pl.DataFrame) and self.edge_attributes.height > 0:
            out = base.join(self.edge_attributes, on="edge_id", how="left")
        else:
            out = base

        # join layer-specific attributes once, then compute resolved weight vectorized
        if (
            layer is not None
            and isinstance(self.edge_layer_attributes, pl.DataFrame)
            and self.edge_layer_attributes.height > 0
        ):
            layer_slice = self.edge_layer_attributes.filter(pl.col("layer_id") == layer).drop(
                "layer_id"
            )
            if layer_slice.height > 0:
                # prefix non-key columns -> layer_*
                rename_map = {c: f"layer_{c}" for c in layer_slice.columns if c not in {"edge_id"}}
                if rename_map:
                    layer_slice = layer_slice.rename(rename_map)
                out = out.join(layer_slice, on="edge_id", how="left")

        # add effective_weight without per-edge function calls
        if resolved_weight:
            gw_col = "global_weight" if include_weight else "_gw_tmp"
            lw_col = "layer_weight" if ("layer_weight" in out.columns) else None
            if lw_col:
                out = out.with_columns(
                    pl.coalesce([pl.col(lw_col), pl.col(gw_col)]).alias("effective_weight")
                )
            else:
                out = out.with_columns(pl.col(gw_col).alias("effective_weight"))

            # drop temp global if it wasn't requested explicitly
            if not include_weight and "_gw_tmp" in out.columns:
                out = out.drop("_gw_tmp")

        return out.clone() if copy else out

    def vertices_view(self, copy=True):
        """Read-only vertex attribute table.

        Parameters
        ----------
        copy : bool, optional
            Return a cloned DF.

        Returns
        -------
        polars.DataFrame
            Columns: ``vertex_id`` plus pure attributes (may be empty).

        """
        df = self.vertex_attributes
        if df.height == 0:
            return pl.DataFrame(schema={"vertex_id": pl.Utf8})
        return df.clone() if copy else df

    def layers_view(self, copy=True):
        """Read-only layer attribute table.

        Parameters
        ----------
        copy : bool, optional
            Return a cloned DF.

        Returns
        -------
        polars.DataFrame
            Columns: ``layer_id`` plus pure attributes (may be empty).

        """
        df = self.layer_attributes
        if df.height == 0:
            return pl.DataFrame(schema={"layer_id": pl.Utf8})
        return df.clone() if copy else df

    # Layer set-ops & cross-layer analytics

    def get_layer_vertices(self, layer_id):
        """Vertices in a layer.

        Parameters
        ----------
        layer_id : str

        Returns
        -------
        set[str]

        """
        return self._layers[layer_id]["vertices"].copy()

    def get_layer_edges(self, layer_id):
        """Edges in a layer.

        Parameters
        ----------
        layer_id : str

        Returns
        -------
        set[str]

        """
        return self._layers[layer_id]["edges"].copy()

    def layer_union(self, layer_ids):
        """Union of multiple layers.

        Parameters
        ----------
        layer_ids : Iterable[str]

        Returns
        -------
        dict
            ``{"vertices": set[str], "edges": set[str]}``

        """
        if not layer_ids:
            return {"vertices": set(), "edges": set()}

        union_vertices = set()
        union_edges = set()

        for layer_id in layer_ids:
            if layer_id in self._layers:
                union_vertices.update(self._layers[layer_id]["vertices"])
                union_edges.update(self._layers[layer_id]["edges"])

        return {"vertices": union_vertices, "edges": union_edges}

    def layer_intersection(self, layer_ids):
        """Intersection of multiple layers.

        Parameters
        ----------
        layer_ids : Iterable[str]

        Returns
        -------
        dict
            ``{"vertices": set[str], "edges": set[str]}``

        """
        if not layer_ids:
            return {"vertices": set(), "edges": set()}

        if len(layer_ids) == 1:
            layer_id = layer_ids[0]
            return {
                "vertices": self._layers[layer_id]["vertices"].copy(),
                "edges": self._layers[layer_id]["edges"].copy(),
            }

        # Start with first layer
        common_vertices = self._layers[layer_ids[0]]["vertices"].copy()
        common_edges = self._layers[layer_ids[0]]["edges"].copy()

        # Intersect with remaining layers
        for layer_id in layer_ids[1:]:
            if layer_id in self._layers:
                common_vertices &= self._layers[layer_id]["vertices"]
                common_edges &= self._layers[layer_id]["edges"]
            else:
                # Layer doesn't exist, intersection is empty
                return {"vertices": set(), "edges": set()}

        return {"vertices": common_vertices, "edges": common_edges}

    def layer_difference(self, layer1_id, layer2_id):
        """Set difference: elements in ``layer1_id`` not in ``layer2_id``.

        Parameters
        ----------
        layer1_id : str
        layer2_id : str

        Returns
        -------
        dict
            ``{"vertices": set[str], "edges": set[str]}``

        Raises
        ------
        KeyError
            If either layer is missing.

        """
        if layer1_id not in self._layers or layer2_id not in self._layers:
            raise KeyError("One or both layers not found")

        layer1 = self._layers[layer1_id]
        layer2 = self._layers[layer2_id]

        return {
            "vertices": layer1["vertices"] - layer2["vertices"],
            "edges": layer1["edges"] - layer2["edges"],
        }

    def create_layer_from_operation(self, result_layer_id, operation_result, **attributes):
        """Create a new layer from the result of a set operation.

        Parameters
        ----------
        result_layer_id : str
        operation_result : dict
            Output of ``layer_union``/``layer_intersection``/``layer_difference``.
        **attributes
            Pure layer attributes.

        Returns
        -------
        str
            The created layer ID.

        Raises
        ------
        ValueError
            If the target layer already exists.

        """
        if result_layer_id in self._layers:
            raise ValueError(f"Layer {result_layer_id} already exists")

        self._layers[result_layer_id] = {
            "vertices": operation_result["vertices"].copy(),
            "edges": operation_result["edges"].copy(),
            "attributes": attributes,
        }

        return result_layer_id

    def edge_presence_across_layers(
        self,
        edge_id: str | None = None,
        source: str | None = None,
        target: str | None = None,
        *,
        include_default: bool = False,
        undirected_match: bool | None = None,
    ):
        """Locate where an edge exists across layers.

        Parameters
        ----------
        edge_id : str, optional
            If provided, match by ID (any kind: binary/vertex-edge/hyper).
        source : str, optional
            When used with ``target``, match only binary/vertex-edge edges by endpoints.
        target : str, optional
        include_default : bool, optional
            Include the internal default layer in the search.
        undirected_match : bool, optional
            When endpoint matching, allow undirected symmetric matches.

        Returns
        -------
        list[str] or dict[str, list[str]]
            If ``edge_id`` given: list of layer IDs.
            Else: ``{layer_id: [edge_id, ...]}``.

        Raises
        ------
        ValueError
            If both modes (ID and endpoints) are provided or neither is valid.

        """
        has_id = edge_id is not None
        has_pair = (source is not None) and (target is not None)
        if has_id == has_pair:
            raise ValueError("Provide either edge_id OR (source and target), but not both.")

        layers_view = self.get_layers_dict(include_default=include_default)

        if has_id:
            return [lid for lid, ldata in layers_view.items() if edge_id in ldata["edges"]]

        if undirected_match is None:
            undirected_match = False

        out: dict[str, list[str]] = {}
        for lid, ldata in layers_view.items():
            matches = []
            for eid in ldata["edges"]:
                # skip hyper-edges for (source,target) mode
                if self.edge_kind.get(eid) == "hyper":
                    continue
                s, t, _ = self.edge_definitions[eid]
                edge_is_directed = self.edge_directed.get(
                    eid, True if self.directed is None else self.directed
                )
                if s == source and t == target:
                    matches.append(eid)
                elif undirected_match and not edge_is_directed and s == target and t == source:
                    matches.append(eid)
            if matches:
                out[lid] = matches
        return out

    def hyperedge_presence_across_layers(
        self,
        *,
        members=None,
        head=None,
        tail=None,
        include_default: bool = False,
    ):
        """Locate layers containing a hyperedge with exactly these sets.

        Parameters
        ----------
        members : Iterable[str], optional
            Undirected member set (exact match).
        head : Iterable[str], optional
            Directed head set (exact match).
        tail : Iterable[str], optional
            Directed tail set (exact match).
        include_default : bool, optional

        Returns
        -------
        dict[str, list[str]]
            ``{layer_id: [edge_id, ...]}``.

        Raises
        ------
        ValueError
            For invalid combinations or empty sets.

        """
        undirected = members is not None
        if undirected and (head is not None or tail is not None):
            raise ValueError("Use either members OR head+tail, not both.")
        if not undirected and (head is None or tail is None):
            raise ValueError("Directed hyperedge query requires both head and tail.")

        if undirected:
            members = set(members)
            if not members:
                raise ValueError("members must be non-empty.")
        else:
            head = set(head)
            tail = set(tail)
            if not head or not tail:
                raise ValueError("head and tail must be non-empty.")
            if head & tail:
                raise ValueError("head and tail must be disjoint.")

        layers_view = self.get_layers_dict(include_default=include_default)
        out: dict[str, list[str]] = {}

        for lid, ldata in layers_view.items():
            matches = []
            for eid in ldata["edges"]:
                if self.edge_kind.get(eid) != "hyper":
                    continue
                meta = self.hyperedge_definitions.get(eid, {})
                if undirected and (not meta.get("directed", False)):
                    if set(meta.get("members", ())) == members:
                        matches.append(eid)
                elif (not undirected) and meta.get("directed", False):
                    if set(meta.get("head", ())) == head and set(meta.get("tail", ())) == tail:
                        matches.append(eid)
            if matches:
                out[lid] = matches
        return out

    def vertex_presence_across_layers(self, vertex_id, include_default: bool = False):
        """List layers containing a specific vertex.

        Parameters
        ----------
        vertex_id : str
        include_default : bool, optional

        Returns
        -------
        list[str]

        """
        layers_with_vertex = []
        for layer_id, layer_data in self.get_layers_dict(include_default=include_default).items():
            if vertex_id in layer_data["vertices"]:
                layers_with_vertex.append(layer_id)
        return layers_with_vertex

    def conserved_edges(self, min_layers=2, include_default=False):
        """Edges present in at least ``min_layers`` layers.

        Parameters
        ----------
        min_layers : int, optional
        include_default : bool, optional

        Returns
        -------
        dict[str, int]
            ``{edge_id: count}``.

        """
        layers_to_check = self.get_layers_dict(
            include_default=include_default
        )  # hides 'default' by default
        edge_counts = {}
        for _, layer_data in layers_to_check.items():
            for eid in layer_data["edges"]:
                edge_counts[eid] = edge_counts.get(eid, 0) + 1
        return {eid: c for eid, c in edge_counts.items() if c >= min_layers}

    def layer_specific_edges(self, layer_id):
        """Edges that appear **only** in the specified layer.

        Parameters
        ----------
        layer_id : str

        Returns
        -------
        set[str]

        Raises
        ------
        KeyError
            If the layer does not exist.

        """
        if layer_id not in self._layers:
            raise KeyError(f"Layer {layer_id} not found")

        target_edges = self._layers[layer_id]["edges"]
        specific_edges = set()

        for edge_id in target_edges:
            # Count how many layers contain this edge
            count = sum(1 for layer_data in self._layers.values() if edge_id in layer_data["edges"])
            if count == 1:  # Only in target layer
                specific_edges.add(edge_id)

        return specific_edges

    def temporal_dynamics(self, ordered_layers, metric="edge_change"):
        """Compute changes between consecutive layers in a temporal sequence.

        Parameters
        ----------
        ordered_layers : list[str]
            Layer IDs in chronological order.
        metric : {'edge_change', 'vertex_change'}, optional

        Returns
        -------
        list[dict[str, int]]
            Per-step dictionaries with keys: ``'added'``, ``'removed'``, ``'net_change'``.

        Raises
        ------
        ValueError
            If fewer than two layers are provided.
        KeyError
            If a referenced layer does not exist.

        """
        if len(ordered_layers) < 2:
            raise ValueError("Need at least 2 layers for temporal analysis")

        changes = []

        for i in range(len(ordered_layers) - 1):
            current_id = ordered_layers[i]
            next_id = ordered_layers[i + 1]

            if current_id not in self._layers or next_id not in self._layers:
                raise KeyError("One or more layers not found")

            current_data = self._layers[current_id]
            next_data = self._layers[next_id]

            if metric == "edge_change":
                added = len(next_data["edges"] - current_data["edges"])
                removed = len(current_data["edges"] - next_data["edges"])
                changes.append({"added": added, "removed": removed, "net_change": added - removed})

            elif metric == "vertex_change":
                added = len(next_data["vertices"] - current_data["vertices"])
                removed = len(current_data["vertices"] - next_data["vertices"])
                changes.append({"added": added, "removed": removed, "net_change": added - removed})

        return changes

    def create_aggregated_layer(
        self, source_layer_ids, target_layer_id, method="union", weight_func=None, **attributes
    ):
        """Create a new layer by aggregating multiple source layers.

        Parameters
        ----------
        source_layer_ids : list[str]
        target_layer_id : str
        method : {'union', 'intersection'}, optional
        weight_func : callable, optional
            Reserved for future weight merging logic (currently unused).
        **attributes
            Pure layer attributes.

        Returns
        -------
        str
            The created layer ID.

        Raises
        ------
        ValueError
            For unknown methods or missing source layers, or if target exists.

        """
        if not source_layer_ids:
            raise ValueError("Must specify at least one source layer")

        if target_layer_id in self._layers:
            raise ValueError(f"Target layer {target_layer_id} already exists")

        if method == "union":
            result = self.layer_union(source_layer_ids)
        elif method == "intersection":
            result = self.layer_intersection(source_layer_ids)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        return self.create_layer_from_operation(target_layer_id, result, **attributes)

    def layer_statistics(self, include_default: bool = False):
        """Basic per-layer statistics.

        Parameters
        ----------
        include_default : bool, optional

        Returns
        -------
        dict[str, dict]
            ``{layer_id: {'vertices': int, 'edges': int, 'attributes': dict}}``.

        """
        stats = {}
        for layer_id, layer_data in self.get_layers_dict(include_default=include_default).items():
            stats[layer_id] = {
                "vertices": len(layer_data["vertices"]),
                "edges": len(layer_data["edges"]),
                "attributes": layer_data["attributes"],
            }
        return stats

    # Traversal (neighbors)

    def neighbors(self, entity_id):
        """Neighbors of an entity (vertex or edge-entity).

        Parameters
        ----------
        entity_id : str

        Returns
        -------
        list[str]
            Adjacent entities. For hyperedges, uses head/tail orientation.

        """
        if entity_id not in self.entity_to_idx:
            return []
        out = set()
        for eid in self.edge_to_idx.keys():
            kind = self.edge_kind.get(eid, None)
            if kind == "hyper":
                meta = self.hyperedge_definitions[eid]
                if meta["directed"]:
                    if entity_id in meta["head"]:
                        out |= meta["tail"]
                    elif entity_id in meta["tail"]:
                        out |= meta["head"]
                else:
                    if ("members" in meta) and (entity_id in meta["members"]):
                        out |= meta["members"] - {entity_id}
            else:
                # binary / vertex_edge
                s, t, _ = self.edge_definitions[eid]
                edir = self.edge_directed.get(eid, True if self.directed is None else self.directed)
                if s == entity_id:
                    out.add(t)
                elif t == entity_id and (not edir or self.entity_types.get(entity_id) == "edge"):
                    out.add(s)
        return list(out)

    def out_neighbors(self, vertex_id):
        """Out-neighbors of a vertex under directed semantics.

        Parameters
        ----------
        vertex_id : str

        Returns
        -------
        list[str]

        """
        if vertex_id not in self.entity_to_idx:
            return []
        out = set()
        for eid in self.edge_to_idx.keys():
            kind = self.edge_kind.get(eid, None)
            if kind == "hyper":
                meta = self.hyperedge_definitions[eid]
                if meta["directed"]:
                    if vertex_id in meta["head"]:
                        out |= meta["tail"]
                else:
                    if vertex_id in meta.get("members", ()):
                        out |= meta["members"] - {vertex_id}
            else:
                s, t, _ = self.edge_definitions[eid]
                edir = self.edge_directed.get(eid, True if self.directed is None else self.directed)
                if s == vertex_id:
                    out.add(t)
                elif t == vertex_id and not edir:
                    out.add(s)
        return list(out)

    def successors(self, vertex_id):
        """Successors of a vertex under directed semantics.

        Parameters
        ----------
        vertex_id : str

        Returns
        -------
        list[str]

        """
        if vertex_id not in self.entity_to_idx:
            return []
        out = set()
        for eid in self.edge_to_idx.keys():
            kind = self.edge_kind.get(eid, None)
            if kind == "hyper":
                meta = self.hyperedge_definitions[eid]
                if meta["directed"]:
                    if vertex_id in meta["head"]:
                        out |= meta["tail"]
                else:
                    if vertex_id in meta.get("members", ()):
                        out |= meta["members"] - {vertex_id}
            else:
                s, t, _ = self.edge_definitions[eid]
                edir = self.edge_directed.get(eid, True if self.directed is None else self.directed)
                if s == vertex_id:
                    out.add(t)
                elif t == vertex_id and not edir:
                    out.add(s)
        return list(out)

    def in_neighbors(self, vertex_id):
        """In-neighbors of a vertex under directed semantics.

        Parameters
        ----------
        vertex_id : str

        Returns
        -------
        list[str]

        """
        if vertex_id not in self.entity_to_idx:
            return []
        inn = set()
        for eid in self.edge_to_idx.keys():
            kind = self.edge_kind.get(eid, None)
            if kind == "hyper":
                meta = self.hyperedge_definitions[eid]
                if meta["directed"]:
                    if vertex_id in meta["tail"]:
                        inn |= meta["head"]
                else:
                    if vertex_id in meta.get("members", ()):
                        inn |= meta["members"] - {vertex_id}
            else:
                s, t, _ = self.edge_definitions[eid]
                edir = self.edge_directed.get(eid, True if self.directed is None else self.directed)
                if t == vertex_id:
                    inn.add(s)
                elif s == vertex_id and not edir:
                    inn.add(t)
        return list(inn)

    def predecessors(self, vertex_id):
        """In-neighbors of a vertex under directed semantics.

        Parameters
        ----------
        vertex_id : str

        Returns
        -------
        list[str]

        """
        if vertex_id not in self.entity_to_idx:
            return []
        inn = set()
        for eid in self.edge_to_idx.keys():
            kind = self.edge_kind.get(eid, None)
            if kind == "hyper":
                meta = self.hyperedge_definitions[eid]
                if meta["directed"]:
                    if vertex_id in meta["tail"]:
                        inn |= meta["head"]
                else:
                    if vertex_id in meta.get("members", ()):
                        inn |= meta["members"] - {vertex_id}
            else:
                s, t, _ = self.edge_definitions[eid]
                edir = self.edge_directed.get(eid, True if self.directed is None else self.directed)
                if t == vertex_id:
                    inn.add(s)
                elif s == vertex_id and not edir:
                    inn.add(t)
        return list(inn)

    # Slicing / copying / accounting

    def edge_subgraph(self, edges) -> "Graph":
        """Create a new graph containing only a specified subset of edges.

        Parameters
        ----------
        edges : Iterable[str] | Iterable[int]
            Edge identifiers (strings) or edge indices (integers) to retain
            in the subgraph.

        Returns
        -------
        Graph
            A new `Graph` instance containing only the selected edges and the
            vertices incident to them.

        Behavior
        --------
        - Copies the current graph and deletes all edges **not** in the provided set.
        - Optionally, you can prune orphaned vertices (i.e., vertices not incident
        to any remaining edge) — this is generally recommended for consistency.

        Notes
        -----
        - Attributes associated with remaining edges and vertices are preserved.
        - Hyperedges are supported: if a hyperedge is in the provided set, all
        its members are retained.
        - If `edges` is empty, the resulting graph will be empty except for
        any isolated vertices that remain.

        """
        # normalize to edge_id set
        if all(isinstance(e, int) for e in edges):
            E = {self.idx_to_edge[e] for e in edges}
        else:
            E = set(edges)

        # collect incident vertices and partition edges
        V = set()
        bin_payload, hyper_payload = [], []
        for eid in E:
            kind = self.edge_kind.get(eid, "binary")
            if kind == "hyper":
                h = self.hyperedge_definitions[eid]
                if h.get("members"):
                    V.update(h["members"])
                    hyper_payload.append(
                        {
                            "members": list(h["members"]),
                            "edge_id": eid,
                            "weight": self.edge_weights.get(eid, 1.0),
                        }
                    )
                else:
                    V.update(h.get("head", ()))
                    V.update(h.get("tail", ()))
                    hyper_payload.append(
                        {
                            "head": list(h.get("head", ())),
                            "tail": list(h.get("tail", ())),
                            "edge_id": eid,
                            "weight": self.edge_weights.get(eid, 1.0),
                        }
                    )
            else:
                s, t, etype = self.edge_definitions[eid]
                V.add(s)
                V.add(t)
                bin_payload.append(
                    {
                        "source": s,
                        "target": t,
                        "edge_id": eid,
                        "edge_type": etype,
                        "edge_directed": self.edge_directed.get(
                            eid, True if self.directed is None else self.directed
                        ),
                        "weight": self.edge_weights.get(eid, 1.0),
                    }
                )

        # new graph prealloc
        g = Graph(directed=self.directed, n=len(V), e=len(E))
        # vertices with attrs
        v_rows = [
            {"vertex_id": v, **(self._row_attrs(self.vertex_attributes, "vertex_id", v) or {})}
            for v in V
        ]
        g.add_vertices_bulk(v_rows, layer=g._default_layer)

        # edges
        if bin_payload:
            g.add_edges_bulk(bin_payload, layer=g._default_layer)
        if hyper_payload:
            g.add_hyperedges_bulk(hyper_payload, layer=g._default_layer)

        # copy layer memberships for retained edges & incident vertices
        for lid, meta in self._layers.items():
            g.add_layer(lid, **meta["attributes"])
            kept_edges = set(meta["edges"]) & E
            if kept_edges:
                g.add_edges_to_layer_bulk(lid, kept_edges)

        return g

    def subgraph(self, vertices) -> "Graph":
        """Create a vertex-induced subgraph.

        Parameters
        ----------
        vertices : Iterable[str]
            A set or list of vertex identifiers to keep in the subgraph.

        Returns
        -------
        Graph
            A new `Graph` containing only the specified vertices and any edges
            for which **all** endpoints are within this set.

        Behavior
        --------
        - Copies the current graph and removes edges with any endpoint outside
        the provided vertex set.
        - Removes all vertices not listed in `vertices`.

        Notes
        -----
        - For binary edges, both endpoints must be in `vertices` to be retained.
        - For hyperedges, **all** member verices must be included to retain the edge.
        - Attributes for retained verices and edges are preserved.

        """
        V = set(vertices)

        # collect edges fully inside V
        E_bin, E_hyper_members, E_hyper_dir = [], [], []
        for eid, (s, t, et) in self.edge_definitions.items():
            if et == "hyper":
                continue
            if s in V and t in V:
                E_bin.append(eid)
        for eid, h in self.hyperedge_definitions.items():
            if h.get("members"):
                if set(h["members"]).issubset(V):
                    E_hyper_members.append(eid)
            else:
                if set(h.get("head", ())).issubset(V) and set(h.get("tail", ())).issubset(V):
                    E_hyper_dir.append(eid)

        # payloads
        v_rows = [
            {"vertex_id": v, **(self._row_attrs(self.vertex_attributes, "vertex_id", v) or {})}
            for v in V
        ]

        bin_payload = []
        for eid in E_bin:
            s, t, etype = self.edge_definitions[eid]
            bin_payload.append(
                {
                    "source": s,
                    "target": t,
                    "edge_id": eid,
                    "edge_type": etype,
                    "edge_directed": self.edge_directed.get(
                        eid, True if self.directed is None else self.directed
                    ),
                    "weight": self.edge_weights.get(eid, 1.0),
                }
            )

        hyper_payload = []
        for eid in E_hyper_members:
            m = self.hyperedge_definitions[eid]["members"]
            hyper_payload.append(
                {"members": list(m), "edge_id": eid, "weight": self.edge_weights.get(eid, 1.0)}
            )
        for eid in E_hyper_dir:
            h = self.hyperedge_definitions[eid]
            hyper_payload.append(
                {
                    "head": list(h.get("head", ())),
                    "tail": list(h.get("tail", ())),
                    "edge_id": eid,
                    "weight": self.edge_weights.get(eid, 1.0),
                }
            )

        # build new graph
        g = Graph(
            directed=self.directed, n=len(V), e=len(E_bin) + len(E_hyper_members) + len(E_hyper_dir)
        )
        g.add_vertices_bulk(v_rows, layer=g._default_layer)
        if bin_payload:
            g.add_edges_bulk(bin_payload, layer=g._default_layer)
        if hyper_payload:
            g.add_hyperedges_bulk(hyper_payload, layer=g._default_layer)

        # layer memberships restricted to V
        for lid, meta in self._layers.items():
            g.add_layer(lid, **meta["attributes"])
            keep = set()
            for eid in meta["edges"]:
                kind = self.edge_kind.get(eid, "binary")
                if kind == "hyper":
                    h = self.hyperedge_definitions[eid]
                    if h.get("members"):
                        if set(h["members"]).issubset(V):
                            keep.add(eid)
                    else:
                        if set(h.get("head", ())).issubset(V) and set(h.get("tail", ())).issubset(
                            V
                        ):
                            keep.add(eid)
                else:
                    s, t, _ = self.edge_definitions[eid]
                    if s in V and t in V:
                        keep.add(eid)
            if keep:
                g.add_edges_to_layer_bulk(lid, keep)

        return g

    def extract_subgraph(self, vertices=None, edges=None) -> "Graph":
        """Create a subgraph based on a combination of vertex and/or edge filters.

        Parameters
        ----------
        vertices : Iterable[str] | None, optional
            A set of vertex IDs to include. If provided, behaves like `subgraph()`.
            If `None`, no vertex filtering is applied.
        edges : Iterable[str] | Iterable[int] | None, optional
            A set of edge IDs or indices to include. If provided, behaves like
            `edge_subgraph()`. If `None`, no edge filtering is applied.

        Returns
        -------
        Graph
            A new `Graph` filtered according to the provided vertex and/or edge
            sets.

        Behavior
        --------
        - If both `vertices` and `edges` are provided, the resulting subgraph is
        the intersection of the two filters.
        - If only `vertices` is provided, equivalent to `subgraph(vertices)`.
        - If only `edges` is provided, equivalent to `edge_subgraph(edges)`.
        - If neither is provided, a full copy of the graph is returned.

        Notes
        -----
        - This is a convenience method; it delegates to `subgraph()` and
        `edge_subgraph()` internally.

        """
        if vertices is None and edges is None:
            return self.copy()

        if edges is not None:
            if all(isinstance(e, int) for e in edges):
                E = {self.idx_to_edge[e] for e in edges}
            else:
                E = set(edges)
        else:
            E = None

        V = set(vertices) if vertices is not None else None

        # If only one filter, delegate to optimized path
        if V is not None and E is None:
            return self.subgraph(V)
        if V is None and E is not None:
            return self.edge_subgraph(E)

        # Both filters: keep only edges in E whose endpoints (or members) lie in V
        kept_edges = set()
        kept_vertices = set(V)
        for eid in E:
            kind = self.edge_kind.get(eid, "binary")
            if kind == "hyper":
                h = self.hyperedge_definitions[eid]
                if h.get("members"):
                    if set(h["members"]).issubset(V):
                        kept_edges.add(eid)
                else:
                    if set(h.get("head", ())).issubset(V) and set(h.get("tail", ())).issubset(V):
                        kept_edges.add(eid)
            else:
                s, t, _ = self.edge_definitions[eid]
                if s in V and t in V:
                    kept_edges.add(eid)

        return self.edge_subgraph(kept_edges).subgraph(kept_vertices)

    def reverse(self) -> "Graph":
        """Return a new graph with all directed edges reversed.

        Returns
        -------
        Graph
            A new `Graph` instance with reversed directionality where applicable.

        Behavior
        --------
        - **Binary edges:** direction is flipped by swapping source and target.
        - **Directed hyperedges:** `head` and `tail` sets are swapped.
        - **Undirected edges/hyperedges:** unaffected.
        - Edge attributes and metadata are preserved.

        Notes
        -----
        - This operation does not modify the original graph.
        - If the graph is undirected (`self.directed == False`), the result is
        identical to the original.
        - For mixed graphs (directed + undirected edges), only the directed
        ones are reversed.

        """
        g = self.copy()

        for eid, defn in g.edge_definitions.items():
            if not g._is_directed_edge(eid):
                continue
            # Binary edge: swap endpoints
            u, v, etype = defn
            g.edge_definitions[eid] = (v, u, etype)

        for eid, meta in g.hyperedge_definitions.items():
            if not meta.get("directed", False):
                continue
            # Hyperedge: swap head and tail sets
            meta["head"], meta["tail"] = meta["tail"], meta["head"]

        return g

    def subgraph_from_layer(self, layer_id, *, resolve_layer_weights=True):
        if layer_id not in self._layers:
            raise KeyError(f"Layer {layer_id} not found")

        import polars as pl

        layer_meta = self._layers[layer_id]
        V = set(layer_meta["vertices"])
        E = set(layer_meta["edges"])

        g = Graph(directed=self.directed, n=len(V), e=len(E))
        g.add_layer(layer_id, **layer_meta["attributes"])
        g.set_active_layer(layer_id)

        # vertices with attrs (edge-entities share same table)
        v_rows = [
            {"vertex_id": v, **(self._row_attrs(self.vertex_attributes, "vertex_id", v) or {})}
            for v in V
        ]
        g.add_vertices_bulk(v_rows, layer=layer_id)

        # edge attrs
        e_attrs = {}
        if (
            isinstance(self.edge_attributes, pl.DataFrame)
            and self.edge_attributes.height
            and "edge_id" in self.edge_attributes.columns
        ):
            for row in self.edge_attributes.filter(pl.col("edge_id").is_in(list(E))).to_dicts():
                d = dict(row)
                eid = d.pop("edge_id", None)
                if eid is not None:
                    e_attrs[eid] = d

        # weights
        eff_w = {}
        if resolve_layer_weights:
            df = self.edge_layer_attributes
            if (
                isinstance(df, pl.DataFrame)
                and df.height
                and {"layer_id", "edge_id", "weight"}.issubset(df.columns)
            ):
                for r in df.filter(
                    (pl.col("layer_id") == layer_id) & (pl.col("edge_id").is_in(list(E)))
                ).iter_rows(named=True):
                    if r.get("weight") is not None:
                        eff_w[r["edge_id"]] = float(r["weight"])

        # partition edges
        bin_payload, hyper_payload = [], []
        for eid in E:
            w = (
                eff_w.get(eid, self.edge_weights.get(eid, 1.0))
                if resolve_layer_weights
                else self.edge_weights.get(eid, 1.0)
            )
            kind = self.edge_kind.get(eid, "binary")
            attrs = e_attrs.get(eid, {})
            if kind == "hyper":
                h = self.hyperedge_definitions[eid]
                if h.get("members"):
                    hyper_payload.append(
                        {
                            "members": list(h["members"]),
                            "edge_id": eid,
                            "weight": w,
                            "attributes": attrs,
                        }
                    )
                else:
                    hyper_payload.append(
                        {
                            "head": list(h.get("head", ())),
                            "tail": list(h.get("tail", ())),
                            "edge_id": eid,
                            "weight": w,
                            "attributes": attrs,
                        }
                    )
            else:
                s, t, et = self.edge_definitions[eid]
                bin_payload.append(
                    {
                        "source": s,
                        "target": t,
                        "edge_id": eid,
                        "edge_type": et,
                        "edge_directed": self.edge_directed.get(
                            eid, True if self.directed is None else self.directed
                        ),
                        "weight": w,
                        "attributes": attrs,
                    }
                )

        if bin_payload:
            g.add_edges_bulk(bin_payload, layer=layer_id)
        if hyper_payload:
            g.add_hyperedges_bulk(hyper_payload, layer=layer_id)

        return g

    def _row_attrs(self, df, key_col: str, key):
        """INTERNAL: return a dict of attributes for the row in `df` where `key_col == key`,
        excluding the key column itself. If not found or df empty, return {}.
        Caches per (id(df), key_col) for speed; cache auto-refreshes when the df object changes.
        """
        try:
            import polars as pl
        except Exception:
            # If Polars isn't available for some reason, best-effort fallback
            return {}

        # Basic guards
        if not isinstance(df, pl.DataFrame) or df.height == 0 or key_col not in df.columns:
            return {}

        # Cache setup
        cache = getattr(self, "_row_attr_cache", None)
        if cache is None:
            cache = {}
            self._row_attr_cache = cache

        cache_key = (id(df), key_col)
        mapping = cache.get(cache_key)

        # Build the mapping once per df object
        if mapping is None:
            mapping = {}
            # Latest write should win if duplicates exist (matches your upsert semantics)
            for row in df.iter_rows(named=True):
                kval = row.get(key_col)
                if kval is None:
                    continue
                d = dict(row)
                d.pop(key_col, None)
                mapping[kval] = d
            cache[cache_key] = mapping

        return mapping.get(key, {})

    def copy(self):
        """Deep copy the entire graph, including layers, edges, hyperedges, and attributes.
        (Behavior preserved; uses preallocation + vectorized attr extraction.)
        """
        import polars as pl

        # Preallocate with current sizes
        new_graph = Graph(directed=self.directed, n=self._num_entities, e=self._num_edges)

        # Copy layers & their pure attributes ----
        for lid, meta in self._layers.items():
            if lid != new_graph._default_layer:
                new_graph.add_layer(lid, **meta["attributes"])
            else:
                # default layer exists; mirror its attributes too
                if meta["attributes"]:
                    new_graph.set_layer_attrs(lid, **meta["attributes"])

        # Build attribute rows once (no per-row filters)
        if (
            isinstance(self.vertex_attributes, pl.DataFrame)
            and self.vertex_attributes.height
            and "vertex_id" in self.vertex_attributes.columns
        ):
            vmap = {d.pop("vertex_id"): d for d in self.vertex_attributes.to_dicts()}
        else:
            vmap = {}

        # Split entities by type to preserve typing
        vertex_rows = []
        edge_entity_rows = []
        for ent_id, etype in self.entity_types.items():
            row = {"vertex_id": ent_id}
            row.update(vmap.get(ent_id, {}))
            if etype == "vertex":
                vertex_rows.append(row)
            else:
                # entity_types[...] == "edge" → edge-entity
                edge_entity_rows.append({"edge_entity_id": ent_id, **vmap.get(ent_id, {})})

        # Add entities with correct type APIs (bulk)
        if vertex_rows:
            new_graph.add_vertices_bulk(vertex_rows, layer=new_graph._default_layer)
        if edge_entity_rows:
            # attributes for edge-entities live in the same vertex_attributes table
            new_graph.add_edge_entities_bulk(edge_entity_rows, layer=new_graph._default_layer)

        # Binary / vertex-edge edges
        bin_payload = []
        for edge_id, (source, target, edge_type) in self.edge_definitions.items():
            if edge_type == "hyper":
                continue
            bin_payload.append(
                {
                    "source": source,
                    "target": target,
                    "edge_id": edge_id,
                    "edge_type": edge_type,  # 'regular' or 'vertex_edge'
                    "edge_directed": self.edge_directed.get(edge_id, self.directed),
                    "weight": self.edge_weights.get(edge_id, 1.0),
                    "attributes": (self._row_attrs(self.edge_attributes, "edge_id", edge_id) or {}),
                }
            )
        if bin_payload:
            new_graph.add_edges_bulk(bin_payload, layer=new_graph._default_layer)

        # Hyperedges
        hyper_payload = []
        for eid, hdef in self.hyperedge_definitions.items():
            base = {
                "edge_id": eid,
                "weight": self.edge_weights.get(eid, 1.0),
                "attributes": (self._row_attrs(self.edge_attributes, "edge_id", eid) or {}),
            }
            if hdef.get("members"):
                hyper_payload.append({**base, "members": list(hdef["members"])})
            else:
                hyper_payload.append(
                    {**base, "head": list(hdef.get("head", ())), "tail": list(hdef.get("tail", ()))}
                )
        if hyper_payload:
            new_graph.add_hyperedges_bulk(hyper_payload, layer=new_graph._default_layer)

        # Copy layer memberships
        for lid, meta in self._layers.items():
            if lid not in new_graph._layers:
                new_graph.add_layer(lid)
            new_graph._layers[lid]["vertices"] = set(meta["vertices"])
            new_graph._layers[lid]["edges"] = set(meta["edges"])

        # Copy edge-layer attributes + legacy weight dict
        if isinstance(self.edge_layer_attributes, pl.DataFrame):
            new_graph.edge_layer_attributes = self.edge_layer_attributes.clone()
        else:
            new_graph.edge_layer_attributes = self.edge_layer_attributes

        from collections import defaultdict

        new_graph.layer_edge_weights = defaultdict(
            dict, {lid: dict(m) for lid, m in self.layer_edge_weights.items()}
        )

        return new_graph

    def memory_usage(self):
        """Approximate total memory usage in bytes.

        Returns
        -------
        int
            Estimated bytes for the incidence matrix, dictionaries, and attribute DFs.

        """
        # Approximate matrix memory: each non-zero entry stores row, col, and value (4 bytes each)
        matrix_bytes = self._matrix.nnz * (4 + 4 + 4)
        # Estimate dict memory: ~100 bytes per entry
        dict_bytes = (
            len(self.entity_to_idx) + len(self.edge_to_idx) + len(self.edge_weights)
        ) * 100

        df_bytes = 0

        # vertex attributes
        if isinstance(self.vertex_attributes, pl.DataFrame):
            # Polars provides a built-in estimate of total size in bytes
            df_bytes += self.vertex_attributes.estimated_size()

        # Edge attributes
        if isinstance(self.edge_attributes, pl.DataFrame):
            df_bytes += self.edge_attributes.estimated_size()

        return matrix_bytes + dict_bytes + df_bytes

    def get_vertex_incidence_matrix_as_lists(self, values: bool = False) -> dict:
        """Materialize the vertex–edge incidence structure as Python lists.

        Parameters
        ----------
        values : bool, optional (default=False)
            - If `False`, returns edge indices incident to each vertex.
            - If `True`, returns the **matrix values** (usually weights or 1/0) for
            each incident edge instead of the indices.

        Returns
        -------
        dict[str, list]
            A mapping from `vertex_id` → list of incident edges (indices or values),
            where:
            - Keys are vertex IDs.
            - Values are lists of edge indices (if `values=False`) or numeric values
            from the incidence matrix (if `values=True`).

        Notes
        -----
        - Internally uses the sparse incidence matrix `self._matrix`, which is stored
        as a SciPy CSR (compressed sparse row) matrix or similar.
        - The incidence matrix `M` is defined as:
            - Rows: vertices
            - Columns: edges
            - Entry `M[i, j]` non-zero ⇨ vertex `i` is incident to edge `j`.
        - This is a convenient method when you want a native-Python structure for
        downstream use (e.g., exporting, iterating, or visualization).

        """
        result = {}
        csr = self._matrix.tocsr()
        for i in range(csr.shape[0]):
            vertex_id = self.idx_to_entity[i]
            row = csr.getrow(i)
            if values:
                result[vertex_id] = row.data.tolist()
            else:
                result[vertex_id] = row.indices.tolist()
        return result

    def vertex_incidence_matrix(self, values: bool = False, sparse: bool = False):
        """Return the vertex–edge incidence matrix in sparse or dense form.

        Parameters
        ----------
        values : bool, optional (default=False)
            If `True`, include the numeric values stored in the matrix
            (e.g., weights or signed incidence values). If `False`, convert the
            matrix to a binary mask (1 if incident, 0 if not).
        sparse : bool, optional (default=False)
            - If `True`, return the underlying sparse matrix (CSR).
            - If `False`, return a dense NumPy ndarray.

        Returns
        -------
        scipy.sparse.csr_matrix | numpy.ndarray
            The vertex–edge incidence matrix `M`:
            - Rows correspond to vertices.
            - Columns correspond to edges.
            - `M[i, j]` ≠ 0 indicates that vertex `i` is incident to edge `j`.

        Notes
        -----
        - If `values=False`, the returned matrix is binarized before returning.
        - Use `sparse=True` for large graphs to avoid memory blowups.
        - This is the canonical low-level structure that most algorithms (e.g.,
        spectral clustering, Laplacian construction, hypergraph analytics) rely on.

        """
        M = self._matrix.tocsr()

        if not values:
            # Convert to binary mask
            M = M.copy()
            M.data[:] = 1

        if sparse:
            return M
        else:
            return M.toarray()

    def __hash__(self) -> int:
        """Return a stable hash representing the current graph structure and metadata.

        Returns
        -------
        int
            A hash value that uniquely (within high probability) identifies the graph
            based on its topology and attributes.

        Behavior
        --------
        - Includes the set of verices, edges, and directedness in the hash.
        - Includes graph-level attributes (if any) to capture metadata changes.
        - Does **not** depend on memory addresses or internal object IDs, so the same
        graph serialized/deserialized or reconstructed with identical structure
        will produce the same hash.

        Notes
        -----
        - This method enables `Graph` objects to be used in hash-based containers
        (like `set` or `dict` keys).
        - If the graph is **mutated** after hashing (e.g., verices or edges are added
        or removed), the hash will no longer reflect the new state.
        - The method uses a deterministic representation: sorted vertex/edge sets
        ensure that ordering does not affect the hash.

        """
        # Core structural components
        vertex_ids = tuple(sorted(self.verices()))
        edge_defs = []

        for j in range(self.number_of_edges()):
            S, T = self.get_edge(j)
            eid = self.idx_to_edge[j]
            directed = self._is_directed_edge(eid)
            edge_defs.append((eid, tuple(sorted(S)), tuple(sorted(T)), directed))

        edge_defs = tuple(sorted(edge_defs))

        # Include high-level metadata if available
        graph_meta = (
            tuple(sorted(self.graph_attributes.items()))
            if hasattr(self, "graph_attributes")
            else ()
        )

        return hash((vertex_ids, edge_defs, graph_meta))

    # History and Timeline

    def _utcnow_iso(self) -> str:
        return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")

    def _jsonify(self, x):
        # Make args/return JSON-safe & compact.
        import numpy as np

        if x is None or isinstance(x, (bool, int, float, str)):
            return x
        if isinstance(x, (set, frozenset)):
            return sorted(self._jsonify(v) for v in x)
        if isinstance(x, (list, tuple)):
            return [self._jsonify(v) for v in x]
        if isinstance(x, dict):
            return {str(k): self._jsonify(v) for k, v in x.items()}
        # NumPy scalars
        if isinstance(x, (np.generic,)):
            return x.item()
        # Polars, SciPy, or other heavy objects -> just a tag
        t = type(x).__name__
        return f"<<{t}>>"

    def _log_event(self, op: str, **fields):
        if not self._history_enabled:
            return
        self._version += 1
        evt = {
            "version": self._version,
            "ts_utc": self._utcnow_iso(),  # ISO-8601 with Z
            "mono_ns": time.perf_counter_ns() - self._history_clock0,
            "op": op,
        }
        # sanitize
        for k, v in fields.items():
            evt[k] = self._jsonify(v)
        self._history.append(evt)

    def _log_mutation(self, name=None):
        def deco(fn):
            op = name or fn.__name__
            sig = inspect.signature(fn)

            @wraps(fn)
            def wrapper(*args, **kwargs):
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                result = fn(*args, **kwargs)
                payload = {}
                # record all call args except 'self'
                for k, v in bound.arguments.items():
                    if k != "self":
                        payload[k] = v
                payload["result"] = result
                self._log_event(op, **payload)
                return result

            return wrapper

        return deco

    def _install_history_hooks(self):
        # Mutating methods to wrap. Add here if you add new mutators.
        to_wrap = [
            "add_vertex",
            "add_edge_entity",
            "add_edge",
            "add_hyperedge",
            "remove_edge",
            "remove_vertex",
            "set_vertex_attrs",
            "set_edge_attrs",
            "set_layer_attrs",
            "set_edge_layer_attrs",
            "register_layer",
            "unregister_layer",
        ]
        for name in to_wrap:
            if hasattr(self, name):
                fn = getattr(self, name)
                # Avoid double-wrapping
                if getattr(fn, "__wrapped__", None) is None:
                    setattr(self, name, self._log_mutation(name)(fn))

    def history(self, as_df: bool = False):
        """Return the append-only mutation history.

        Parameters
        ----------
        as_df : bool, default False
            If True, return a Polars DF [DataFrame]; otherwise return a list of dicts.

        Returns
        -------
        list[dict] or polars.DataFrame
            Each event includes: 'version', 'ts_utc' (UTC [Coordinated Universal Time]
            ISO-8601 [International Organization for Standardization]), 'mono_ns'
            (monotonic nanoseconds since logger start), 'op', call snapshot fields,
            and 'result' when captured.

        Notes
        -----
        Ordering is guaranteed by 'version' and 'mono_ns'. The log is in-memory until exported.

        """
        return pl.DataFrame(self._history) if as_df else list(self._history)

    def export_history(self, path: str):
        """Write the mutation history to disk.

        Parameters
        ----------
        path : str
            Output path. Supported extensions: '.parquet', '.ndjson' (a.k.a. '.jsonl'),
            '.json', '.csv'. Unknown extensions default to Parquet by appending '.parquet'.

        Returns
        -------
        int
            Number of events written. Returns 0 if the history is empty.

        Raises
        ------
        OSError
            If the file cannot be written.

        """
        if not self._history:
            return 0
        df = pl.DataFrame(self._history)
        p = path.lower()
        if p.endswith(".parquet"):
            df.write_parquet(path)
            return len(df)
        if p.endswith(".ndjson") or p.endswith(".jsonl"):
            with open(path, "w", encoding="utf-8") as f:
                for r in df.iter_rows(named=True):
                    import json

                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            return len(df)
        if p.endswith(".json"):
            import json

            with open(path, "w", encoding="utf-8") as f:
                json.dump(df.to_dicts(), f, ensure_ascii=False)
            return len(df)
        if p.endswith(".csv"):
            df.write_csv(path)
            return len(df)
        # Default to Parquet if unknown
        df.write_parquet(path + ".parquet")
        return len(df)

    def enable_history(self, flag: bool = True):
        """Enable or disable in-memory mutation logging.

        Parameters
        ----------
        flag : bool, default True
            When True, start/continue logging; when False, pause logging.

        Returns
        -------
        None

        """
        self._history_enabled = bool(flag)

    def clear_history(self):
        """Clear the in-memory mutation log.

        Returns
        -------
        None

        Notes
        -----
        This does not delete any files previously exported.

        """
        self._history.clear()

    def mark(self, label: str):
        """Insert a manual marker into the mutation history.

        Parameters
        ----------
        label : str
            Human-readable tag for the marker event.

        Returns
        -------
        None

        Notes
        -----
        The event is recorded with 'op'='mark' alongside standard fields
        ('version', 'ts_utc', 'mono_ns'). Logging must be enabled for the
        marker to be recorded.

        """
        self._log_event("mark", label=label)

    # Lazy proxies
    ## Lazy NX proxy

    @property
    def nx(self):
        """Accessor for the lazy NX proxy.
        Usage: G.nx.algorithm(); e.g: G.nx.louvain_communities(G), G.nx.shortest_path_length(G, weight="weight")
        """
        if not hasattr(self, "_nx_proxy"):
            self._nx_proxy = self._LazyNXProxy(self)
        return self._nx_proxy

    class _LazyNXProxy:
        """Lazy, cached NX (NetworkX) adapter:
        - On-demand backend conversion (no persistent NX graph).
        - Cache keyed by options until Graph._version changes.
        - Selective edge attr exposure (weight/capacity only when needed).
        - Clear warnings when conversion is lossy.
        - Auto label→ID mapping for node arguments (kwargs + positionals).
        - NEW: _nx_simple to collapse Multi* → simple Graph/DiGraph for algos that need it.
        - NEW: _nx_edge_aggs to control parallel-edge aggregation (e.g., {"capacity":"sum"}).
        """

        # ------------------------------ init -----------------------------------
        def __init__(self, owner: "Graph"):
            self._G = owner
            self._cache = {}  # key -> {"nxG": nx.Graph, "version": int}
            self.cache_enabled = True

        # ---------------------------- public API --------------------------------
        def clear(self):
            """Drop all cached NX graphs."""
            self._cache.clear()

        def peek_nodes(self, k: int = 10):
            """Debug helper: return up to k node IDs visible to NX."""
            nxG = self._get_or_make_nx(
                directed=True,
                hyperedge_mode="expand",
                layer=None,
                layers=None,
                needed_attrs=set(),
                simple=False,
                edge_aggs=None,
            )
            out = []
            it = iter(nxG.nodes())
            for _ in range(max(0, int(k))):
                try:
                    out.append(next(it))
                except StopIteration:
                    break
            return out

        # ------------------------- dynamic dispatch -----------------------------
        # Public helper: obtain the cached/backend NX graph directly
        # Usage in tests: nxG = G.nx.backend(directed=False, simple=True)
        def backend(
            self,
            *,
            directed: bool = True,
            hyperedge_mode: str = "expand",
            layer=None,
            layers=None,
            needed_attrs=None,
            simple: bool = False,
            edge_aggs: dict | None = None,
        ):
            """Return the underlying NetworkX graph built with the same lazy/cached
            machinery as normal calls.

            Args:
              directed: build DiGraph (True) or Graph (False) view
              hyperedge_mode: "skip" | "expand"
              layer/layers: layer selection if your Graph is multilayered
              needed_attrs: set of edge attribute names to keep (default empty)
              simple: if True, collapse Multi* -> simple (Di)Graph
              edge_aggs: how to aggregate parallel edge attrs when simple=True,
                         e.g. {"capacity": "sum", "weight": "min"} or callables

            """
            if needed_attrs is None:
                needed_attrs = set()
            return self._get_or_make_nx(
                directed=directed,
                hyperedge_mode=hyperedge_mode,
                layer=layer,
                layers=layers,
                needed_attrs=needed_attrs,
                simple=simple,
                edge_aggs=edge_aggs,
            )

        def __getattr__(self, name: str):
            nx_callable = self._resolve_nx_callable(name)

            def wrapper(*args, **kwargs):
                import inspect

                import networkx as _nx

                # Proxy-only knobs (consumed here; not forwarded to NX)
                directed = bool(kwargs.pop("_nx_directed", getattr(self, "default_directed", True)))
                hyperedge_mode = kwargs.pop(
                    "_nx_hyperedge", getattr(self, "default_hyperedge_mode", "expand")
                )  # "skip" | "expand"
                layer = kwargs.pop("_nx_layer", None)
                layers = kwargs.pop("_nx_layers", None)
                label_field = kwargs.pop("_nx_label_field", None)  # explicit label column
                guess_labels = kwargs.pop(
                    "_nx_guess_labels", True
                )  # try auto-infer when not provided

                # force simple Graph/DiGraph and aggregation policy for parallel edges
                simple = bool(kwargs.pop("_nx_simple", getattr(self, "default_simple", False)))
                edge_aggs = kwargs.pop(
                    "_nx_edge_aggs", None
                )  # e.g. {"weight":"min","capacity":"sum"} or callables

                # Determine required edge attributes (keep graph skinny)
                needed_edge_attrs = self._needed_edge_attrs(nx_callable, kwargs)

                # Do NOT auto-inject G. Only convert/replace if the user passed our Graph.
                args = list(args)
                has_owner_graph = any(a is self._G for a in args) or any(
                    v is self._G for v in kwargs.values()
                )

                # Build backend ONLY if we actually need to replace self._G
                nxG = None
                if has_owner_graph:
                    nxG = self._get_or_make_nx(
                        directed=directed,
                        hyperedge_mode=hyperedge_mode,
                        layer=layer,
                        layers=layers,
                        needed_attrs=needed_edge_attrs,
                        simple=simple,
                        edge_aggs=edge_aggs,
                    )

                # Replace any occurrence of our Graph with the NX backend
                if nxG is not None:
                    for i, v in enumerate(args):
                        if v is self._G:
                            args[i] = nxG
                    for k, v in list(kwargs.items()):
                        if v is self._G:
                            kwargs[k] = nxG

                # Bind to NX signature so we can coerce node args (no defaults!)
                bound = None
                try:
                    sig = inspect.signature(nx_callable)
                    bound = sig.bind_partial(*args, **kwargs)
                except Exception:
                    pass

                # Coerce node args (labels/indices -> vertex IDs)
                try:
                    # Determine default label field if not given
                    if label_field is None and guess_labels:
                        label_field = self._infer_label_field()

                    if bound is not None and nxG is not None:
                        self._coerce_nodes_in_bound(bound, nxG, label_field)
                        # Reconstruct WITHOUT applying defaults (avoid flow_func=None, etc.)
                        pargs = bound.args
                        pkwargs = bound.kwargs
                    else:
                        # Fallback: best-effort coercion on kwargs only
                        if nxG is not None:
                            self._coerce_nodes_in_kwargs(kwargs, nxG, label_field)
                        pargs, pkwargs = tuple(args), kwargs
                except Exception:
                    pargs, pkwargs = tuple(args), kwargs  # best effort; let NX raise if needed

                # Never leak private knobs to NX
                for k in list(pkwargs.keys()):
                    if isinstance(k, str) and k.startswith("_nx_"):
                        pkwargs.pop(k, None)

                try:
                    return nx_callable(*pargs, **pkwargs)
                except _nx.NodeNotFound as e:
                    # Add actionable tip that actually tells how to fix it now.
                    sample = self.peek_nodes(5)
                    tip = (
                        f"{e}. Nodes must be your graph's vertex IDs.\n"
                        f"- If you passed labels, specify _nx_label_field=<vertex label column> "
                        f"or rely on auto-guess (columns like 'name'/'label'/'title').\n"
                        f"- Example: G.nx.shortest_path_length(G, source='a', target='z', weight='weight', _nx_label_field='name')\n"
                        f"- A few node IDs NX sees: {sample}"
                    )
                    raise _nx.NodeNotFound(tip) from e

            return wrapper

        # ------------------------------ internals -------------------------------
        def _resolve_nx_callable(self, name: str):
            import networkx as _nx

            candidates = [
                _nx,
                getattr(_nx, "algorithms", None),
                getattr(_nx.algorithms, "community", None) if hasattr(_nx, "algorithms") else None,
                getattr(_nx.algorithms, "approximation", None)
                if hasattr(_nx, "algorithms")
                else None,
                getattr(_nx.algorithms, "centrality", None) if hasattr(_nx, "algorithms") else None,
                getattr(_nx.algorithms, "shortest_paths", None)
                if hasattr(_nx, "algorithms")
                else None,
                getattr(_nx.algorithms, "flow", None) if hasattr(_nx, "algorithms") else None,
                getattr(_nx.algorithms, "components", None) if hasattr(_nx, "algorithms") else None,
                getattr(_nx.algorithms, "traversal", None) if hasattr(_nx, "algorithms") else None,
                getattr(_nx.algorithms, "bipartite", None) if hasattr(_nx, "algorithms") else None,
                getattr(_nx.algorithms, "link_analysis", None)
                if hasattr(_nx, "algorithms")
                else None,
                getattr(_nx, "classes", None),
                getattr(_nx.classes, "function", None) if hasattr(_nx, "classes") else None,
            ]
            for mod in (m for m in candidates if m is not None):
                attr = getattr(mod, name, None)
                if callable(attr):
                    return attr
            raise AttributeError(f"networkx has no callable '{name}'")

        def _needed_edge_attrs(self, target, kwargs) -> set:
            import inspect

            needed = set()
            # weight
            w_name = kwargs.get("weight", "weight")
            try:
                sig = inspect.signature(target)
                if "weight" in sig.parameters and w_name is not None:
                    needed.add(str(w_name))
            except Exception:
                if "weight" in kwargs and w_name is not None:
                    needed.add(str(w_name))
            # capacity (flows)
            c_name = kwargs.get("capacity", "capacity")
            try:
                sig = inspect.signature(target)
                if "capacity" in sig.parameters and c_name is not None:
                    needed.add(str(c_name))
            except Exception:
                if "capacity" in kwargs and c_name is not None:
                    needed.add(str(c_name))
            return needed

        def _convert_to_nx(
            self,
            *,
            directed: bool,
            hyperedge_mode: str,
            layer,
            layers,
            needed_attrs: set,
            simple: bool,
            edge_aggs: dict | None,
        ):
            from ..adapters import networkx_adapter as _gg_nx  # annnet.adapters.networkx_adapter

            nxG, manifest = _gg_nx.to_nx(
                self._G,
                directed=directed,
                hyperedge_mode=hyperedge_mode,
                layer=layer,
                layers=layers,
                public_only=True,
            )
            # Keep only needed edge attrs
            if needed_attrs:
                for _, _, _, d in nxG.edges(keys=True, data=True):
                    for k in list(d.keys()):
                        if k not in needed_attrs:
                            d.pop(k, None)
            else:
                for _, _, _, d in nxG.edges(keys=True, data=True):
                    d.clear()

            # Collapse Multi* → simple Graph/DiGraph if requested
            if simple and nxG.is_multigraph():
                nxG = self._collapse_multiedges(
                    nxG, directed=directed, aggregations=edge_aggs, needed_attrs=needed_attrs
                )

            self._warn_on_loss(
                hyperedge_mode=hyperedge_mode, layer=layer, layers=layers, manifest=manifest
            )
            return nxG

        def _get_or_make_nx(
            self,
            *,
            directed: bool,
            hyperedge_mode: str,
            layer,
            layers,
            needed_attrs: set,
            simple: bool,
            edge_aggs: dict | None,
        ):
            key = (
                bool(directed),
                str(hyperedge_mode),
                tuple(sorted(layers)) if layers else None,
                str(layer) if layer is not None else None,
                tuple(sorted(needed_attrs)) if needed_attrs else (),
                bool(simple),
                tuple(sorted(edge_aggs.items())) if isinstance(edge_aggs, dict) else None,
            )
            version = getattr(self._G, "_version", None)
            entry = self._cache.get(key)
            if (
                (not self.cache_enabled)
                or (entry is None)
                or (version is not None and entry.get("version") != version)
            ):
                nxG = self._convert_to_nx(
                    directed=directed,
                    hyperedge_mode=hyperedge_mode,
                    layer=layer,
                    layers=layers,
                    needed_attrs=needed_attrs,
                    simple=simple,
                    edge_aggs=edge_aggs,
                )
                if self.cache_enabled:
                    self._cache[key] = {"nxG": nxG, "version": version}
                return nxG
            return entry["nxG"]

        def _warn_on_loss(self, *, hyperedge_mode, layer, layers, manifest):
            import warnings

            has_hyper = False
            try:
                ek = getattr(self._G, "edge_kind", {})  # dict[eid] -> "hyper"/"binary"
                if hasattr(ek, "values"):
                    has_hyper = any(str(v).lower() == "hyper" for v in ek.values())
            except Exception:
                pass
            msgs = []
            if has_hyper and hyperedge_mode != "expand":
                msgs.append("hyperedges dropped (hyperedge_mode='skip')")
            try:
                layers_dict = getattr(self._G, "_layers", None)
                if (
                    isinstance(layers_dict, dict)
                    and len(layers_dict) > 1
                    and (layer is None and not layers)
                ):
                    msgs.append("multiple layers flattened into single NX graph")
            except Exception:
                pass
            if manifest is None:
                msgs.append("no manifest provided; round-trip fidelity not guaranteed")
            if msgs:
                warnings.warn(
                    "Graph→NX conversion is lossy: " + "; ".join(msgs) + ".",
                    category=RuntimeWarning,
                    stacklevel=3,
                )

        # ---------------------- label/ID mapping helpers ------------------------
        def _infer_label_field(self) -> str | None:
            """Heuristic label column if user didn't specify:
            1) Graph.default_label_field if present
            2) first present in ["name","label","title","slug","external_id","string_id"]
            """
            try:
                if hasattr(self._G, "default_label_field") and self._G.default_label_field:
                    return self._G.default_label_field
                va = getattr(self._G, "vertex_attributes", None)
                cols = list(va.columns) if va is not None and hasattr(va, "columns") else []
                for c in ("name", "label", "title", "slug", "external_id", "string_id"):
                    if c in cols:
                        return c
            except Exception:
                pass
            return None

        def _vertex_id_col(self) -> str:
            """Best-effort to determine the vertex ID column name in vertex_attributes."""
            try:
                va = self._G.vertex_attributes
                cols = list(va.columns)
                for k in ("vertex_id", "id", "vid"):
                    if k in cols:
                        return k
            except Exception:
                pass
            return "vertex_id"

        def _lookup_vertex_id_by_label(self, label_field: str, val):
            """Return vertex_id where vertex_attributes[label_field] == val, else None."""
            try:
                va = self._G.vertex_attributes
                if va is None or not hasattr(va, "columns") or label_field not in va.columns:
                    return None
                id_col = self._vertex_id_col()
                # Prefer polars path
                try:
                    import polars as pl  # type: ignore

                    matches = va.filter(pl.col(label_field) == val)
                    if matches.height == 0:
                        return None
                    try:
                        return matches.select(id_col).to_series().to_list()[0]
                    except Exception:
                        return matches.select(id_col).item(0, 0)
                except Exception:
                    # Fallback: convert to dicts (slower; fine for ad-hoc lookups)
                    for row in va.to_dicts():
                        if row.get(label_field) == val:
                            return row.get(id_col)
            except Exception:
                return None
            return None

        def _coerce_node_id(self, x, nxG, label_field: str | None):
            # If already a node ID present in the backend, keep it.
            if x in nxG:
                return x
            # Internal index → vertex_id
            try:
                if isinstance(x, int) and x in getattr(self._G, "idx_to_entity", {}):
                    cand = self._G.idx_to_entity[x]
                    if getattr(self._G, "entity_types", {}).get(cand) == "vertex":
                        return cand
            except Exception:
                pass
            # Label mapping
            if label_field:
                cand = self._lookup_vertex_id_by_label(label_field, x)
                if cand is not None:
                    return cand
            return x  # let NX decide (will raise NodeNotFound if still absent)

        def _coerce_node_or_iter(self, obj, nxG, label_field: str | None):
            if isinstance(obj, (list, tuple, set)):
                coerced = [self._coerce_node_id(v, nxG, label_field) for v in obj]
                return type(obj)(coerced) if not isinstance(obj, set) else set(coerced)
            return self._coerce_node_id(obj, nxG, label_field)

        def _coerce_nodes_in_kwargs(self, kwargs: dict, nxG, label_field: str | None):
            node_keys = {"source", "target", "u", "v", "node", "nodes", "nbunch", "center", "path"}
            for key in list(kwargs.keys()):
                if key in node_keys:
                    kwargs[key] = self._coerce_node_or_iter(kwargs[key], nxG, label_field)

        def _coerce_nodes_in_bound(self, bound, nxG, label_field: str | None):
            """Coerce nodes in a BoundArguments object using common node parameter names."""
            node_keys = {"source", "target", "u", "v", "node", "nodes", "nbunch", "center", "path"}
            for key in list(bound.arguments.keys()):
                if key in node_keys:
                    bound.arguments[key] = self._coerce_node_or_iter(
                        bound.arguments[key], nxG, label_field
                    )

        # ---------------------- Multi* collapse helpers -------------------------
        def _collapse_multiedges(
            self, nxG, *, directed: bool, aggregations: dict | None, needed_attrs: set
        ):
            """Collapse parallel edges into a single edge with aggregated attributes.
            Defaults: weight -> min (good for shortest paths), capacity -> sum (good for max-flow).
            """
            import networkx as _nx

            H = _nx.DiGraph() if directed else _nx.Graph()
            H.add_nodes_from(nxG.nodes(data=True))

            aggregations = aggregations or {}

            def _agg_for(key):
                agg = aggregations.get(key)
                if callable(agg):
                    return agg
                if agg == "sum":
                    return sum
                if agg == "min":
                    return min
                if agg == "max":
                    return max
                # sensible defaults:
                if key == "capacity":
                    return sum
                if key == "weight":
                    return min
                # fallback: first value
                return lambda vals: next(iter(vals))

            # Bucket parallel edges
            bucket = {}  # (u,v) or sorted(u,v) -> {attr: [values]}
            for u, v, _, d in nxG.edges(keys=True, data=True):
                key = (u, v) if directed else tuple(sorted((u, v)))
                entry = bucket.setdefault(key, {})
                for k, val in d.items():
                    if needed_attrs and k not in needed_attrs:
                        continue
                    entry.setdefault(k, []).append(val)

            # Aggregate per (u,v)
            for (u, v), attrs in bucket.items():
                out = {k: _agg_for(k)(vals) for k, vals in attrs.items()}
                H.add_edge(u, v, **out)

            return H

    ## Lazy igraph proxy

    @property
    def ig(self):
        """Accessor for the lazy igraph proxy.
        Usage: G.ig.community_multilevel(G, weights="weight"), G.ig.shortest_paths_dijkstra(G, source="a", target="z", weights="weight")
        (same idea as NX: pass G; proxy swaps it with the backend igraph.Graph lazily)
        """
        if not hasattr(self, "_ig_proxy"):
            self._ig_proxy = self._LazyIGProxy(self)
        return self._ig_proxy

    class _LazyIGProxy:
        """Lazy, cached igraph adapter:
        - On-demand backend conversion (no persistent igraph graph).
        - Cache keyed by options until Graph._version changes.
        - Selective edge-attr exposure (keep only needed weights/capacity).
        - Clear warnings when conversion is lossy.
        - Auto label→ID mapping for node args (kwargs + positionals).
        - _ig_simple=True collapses parallel edges to simple (Di)Graph.
        - _ig_edge_aggs={"weight":"min","capacity":"sum"} for parallel-edge aggregation.
        """

        def __init__(self, owner: "Graph"):
            self._G = owner
            self._cache = {}  # key -> {"igG": ig.Graph, "version": int}
            self.cache_enabled = True

        # ---------------------------- public API --------------------------------
        def clear(self):
            self._cache.clear()

        def peek_vertices(self, k: int = 10):
            igG = self._get_or_make_ig(
                directed=True,
                hyperedge_mode="skip",
                layer=None,
                layers=None,
                needed_attrs=set(),
                simple=True,
                edge_aggs=None,
            )
            out = []
            names = igG.vs["name"] if "name" in igG.vs.attributes() else None
            for i in range(min(max(0, int(k)), igG.vcount())):
                out.append(names[i] if names else i)
            return out

        # public helper so tests don’t touch private API
        def backend(
            self,
            *,
            directed: bool = True,
            hyperedge_mode: str = "skip",
            layer=None,
            layers=None,
            needed_attrs=None,
            simple: bool = False,
            edge_aggs: dict | None = None,
        ):
            needed_attrs = needed_attrs or set()
            return self._get_or_make_ig(
                directed=directed,
                hyperedge_mode=hyperedge_mode,
                layer=layer,
                layers=layers,
                needed_attrs=needed_attrs,
                simple=simple,
                edge_aggs=edge_aggs,
            )

        # ------------------------- dynamic dispatch -----------------------------
        def __getattr__(self, name: str):
            def wrapper(*args, **kwargs):
                import inspect

                import igraph as _ig

                # proxy-only knobs (consumed here)
                directed = bool(kwargs.pop("_ig_directed", True))
                hyperedge_mode = kwargs.pop("_ig_hyperedge", "skip")  # "skip" | "expand"
                layer = kwargs.pop("_ig_layer", None)
                layers = kwargs.pop("_ig_layers", None)
                label_field = kwargs.pop("_ig_label_field", None)
                guess_labels = kwargs.pop("_ig_guess_labels", True)
                simple = bool(kwargs.pop("_ig_simple", False))
                edge_aggs = kwargs.pop(
                    "_ig_edge_aggs", None
                )  # {"weight":"min","capacity":"sum"} or callables

                # keep only attributes actually needed by the called function
                needed_edge_attrs = self._needed_edge_attrs_for_ig(name, kwargs)

                # build/reuse backend
                igG = self._get_or_make_ig(
                    directed=directed,
                    hyperedge_mode=hyperedge_mode,
                    layer=layer,
                    layers=layers,
                    needed_attrs=needed_edge_attrs,
                    simple=simple,
                    edge_aggs=edge_aggs,
                )

                # replace any Graph instance with igG
                args = list(args)
                for i, v in enumerate(args):
                    if v is self._G:
                        args[i] = igG
                for k, v in list(kwargs.items()):
                    if v is self._G:
                        kwargs[k] = igG

                # resolve target callable: prefer bound Graph method, else module-level
                target = getattr(igG, name, None)
                if not callable(target):
                    target = getattr(_ig, name, None)
                if not callable(target):
                    raise AttributeError(
                        f"igraph has no callable '{name}'. "
                        f"Use native igraph names, e.g. community_multilevel, pagerank, shortest_paths_dijkstra, components, etc."
                    )

                # bind to signature (best effort) so we can coerce node args
                try:
                    sig = inspect.signature(target)
                    bound = sig.bind_partial(*args, **kwargs)
                except Exception:
                    bound = None

                try:
                    if label_field is None and guess_labels:
                        label_field = self._infer_label_field()

                    if bound is not None:
                        self._coerce_nodes_in_bound(bound, igG, label_field)
                        bound.apply_defaults()
                        pargs, pkwargs = list(bound.args), dict(bound.kwargs)
                    else:
                        self._coerce_nodes_in_kwargs(kwargs, igG, label_field)
                        pargs, pkwargs = list(args), dict(kwargs)
                except Exception:
                    pargs, pkwargs = list(args), dict(kwargs)  # let igraph raise if invalid

                try:
                    return target(*pargs, **pkwargs)
                except (KeyError, ValueError) as e:
                    sample = self.peek_vertices(5)
                    tip = (
                        f"{e}. Vertices must match this graph's vertex IDs.\n"
                        f"- If you passed labels, set _ig_label_field=<vertex label column> "
                        f"or rely on auto-guess ('name'/'label'/'title').\n"
                        f"- Example: G.ig.shortest_paths_dijkstra(G, source='a', target='z', weights='weight', _ig_label_field='name')\n"
                        f"- A few vertex IDs igraph sees: {sample}"
                    )
                    raise type(e)(tip) from e

            return wrapper

        # ------------------------------ internals -------------------------------
        def _needed_edge_attrs_for_ig(self, func_name: str, kwargs: dict) -> set:
            """Heuristic: igraph uses `weights` (plural) for edge weights in most algos,
            some accept both; flows use 'capacity' if you forward them to adapters.
            """
            needed = set()
            # weight(s)
            w = kwargs.get("weights", kwargs.get("weight", None))
            if w is None:
                # sometimes user passes True to mean default "weight"
                if "weights" in kwargs and kwargs["weights"] is not None:
                    needed.add(str(kwargs["weights"]))
            else:
                needed.add(str(w))
            # capacity (if you forward flow-like algos to ig backends)
            if "capacity" in kwargs and kwargs["capacity"] is not None:
                needed.add(str(kwargs["capacity"]))
            return needed

        def _convert_to_ig(
            self,
            *,
            directed: bool,
            hyperedge_mode: str,
            layer,
            layers,
            needed_attrs: set,
            simple: bool,
            edge_aggs: dict | None,
        ):
            # try both adapter entry points: to_ig / to_igraph

            from ..adapters import igraph_adapter as _gg_ig  # annnet.adapters.igraph_adapter

            conv = None
            for cand in ("to_ig", "to_igraph"):
                conv = getattr(_gg_ig, cand, None) or conv
            if conv is None:
                raise RuntimeError(
                    "igraph adapter missing: expected adapters.igraph_adapter.to_ig(...) or .to_igraph(...)."
                )

            igG, manifest = conv(
                self._G,
                directed=directed,
                hyperedge_mode=hyperedge_mode,
                layer=layer,
                layers=layers,
                public_only=True,
            )

            # keep only requested edge attrs (or none at all)
            igG = self._prune_edge_attributes(igG, needed_attrs)

            # igraph lacks is_multigraph(); always collapse when simple=True
            if simple:
                igG = self._collapse_multiedges(
                    igG, directed=directed, aggregations=edge_aggs, needed_attrs=needed_attrs
                )

            self._warn_on_loss(
                hyperedge_mode=hyperedge_mode, layer=layer, layers=layers, manifest=manifest
            )
            return igG

        def _get_or_make_ig(
            self,
            *,
            directed: bool,
            hyperedge_mode: str,
            layer,
            layers,
            needed_attrs: set,
            simple: bool,
            edge_aggs: dict | None,
        ):
            key = (
                bool(directed),
                str(hyperedge_mode),
                tuple(sorted(layers)) if layers else None,
                str(layer) if layer is not None else None,
                tuple(sorted(needed_attrs)) if needed_attrs else (),
                bool(simple),
                tuple(sorted(edge_aggs.items())) if isinstance(edge_aggs, dict) else None,
            )
            version = getattr(self._G, "_version", None)
            entry = self._cache.get(key)
            if (
                (not self.cache_enabled)
                or (entry is None)
                or (version is not None and entry.get("version") != version)
            ):
                igG = self._convert_to_ig(
                    directed=directed,
                    hyperedge_mode=hyperedge_mode,
                    layer=layer,
                    layers=layers,
                    needed_attrs=needed_attrs,
                    simple=simple,
                    edge_aggs=edge_aggs,
                )
                if self.cache_enabled:
                    self._cache[key] = {"igG": igG, "version": version}
                return igG
            return entry["igG"]

        def _warn_on_loss(self, *, hyperedge_mode, layer, layers, manifest):
            import warnings

            has_hyper = False
            try:
                ek = getattr(self._G, "edge_kind", {})
                if hasattr(ek, "values"):
                    has_hyper = any(str(v).lower() == "hyper" for v in ek.values())
            except Exception:
                pass
            msgs = []
            if has_hyper and hyperedge_mode != "expand":
                msgs.append("hyperedges dropped (hyperedge_mode='skip')")
            try:
                layers_dict = getattr(self._G, "_layers", None)
                if (
                    isinstance(layers_dict, dict)
                    and len(layers_dict) > 1
                    and (layer is None and not layers)
                ):
                    msgs.append("multiple layers flattened into single igraph graph")
            except Exception:
                pass
            if manifest is None:
                msgs.append("no manifest provided; round-trip fidelity not guaranteed")
            if msgs:
                warnings.warn(
                    "Graph→igraph conversion is lossy: " + "; ".join(msgs) + ".",
                    category=RuntimeWarning,
                    stacklevel=3,
                )

        # ---------------------- label/ID mapping helpers ------------------------
        def _infer_label_field(self) -> str | None:
            try:
                if hasattr(self._G, "default_label_field") and self._G.default_label_field:
                    return self._G.default_label_field
                va = getattr(self._G, "vertex_attributes", None)
                cols = list(va.columns) if va is not None and hasattr(va, "columns") else []
                for c in ("name", "label", "title", "slug", "external_id", "string_id"):
                    if c in cols:
                        return c
            except Exception:
                pass
            return None

        def _vertex_id_col(self) -> str:
            try:
                va = self._G.vertex_attributes
                cols = list(va.columns)
                for k in ("vertex_id", "id", "vid"):
                    if k in cols:
                        return k
            except Exception:
                pass
            return "vertex_id"

        def _lookup_vertex_id_by_label(self, label_field: str, val):
            try:
                va = self._G.vertex_attributes
                if va is None or not hasattr(va, "columns") or label_field not in va.columns:
                    return None
                id_col = self._vertex_id_col()
                try:
                    import polars as pl  # type: ignore

                    matches = va.filter(pl.col(label_field) == val)
                    if matches.height == 0:
                        return None
                    try:
                        return matches.select(id_col).to_series().to_list()[0]
                    except Exception:
                        return matches.select(id_col).item(0, 0)
                except Exception:
                    for row in va.to_dicts():
                        if row.get(label_field) == val:
                            return row.get(id_col)
            except Exception:
                return None
            return None

        def _name_to_index_map(self, igG):
            names = igG.vs["name"] if "name" in igG.vs.attributes() else None
            return {n: i for i, n in enumerate(names)} if names is not None else {}

        def _coerce_vertex(self, x, igG, label_field: str | None):
            # already an index?
            if isinstance(x, int) and 0 <= x < igG.vcount():
                return x
            # graph-level mapping (label -> vertex_id)
            if label_field:
                cand = self._lookup_vertex_id_by_label(label_field, x)
                if cand is not None:
                    x = cand
            # igraph name -> index
            name_to_idx = self._name_to_index_map(igG)
            if x in name_to_idx:
                return name_to_idx[x]
            # if user already passed internal vertex_id string, try treating it as name
            if isinstance(x, str) and x in name_to_idx:
                return name_to_idx[x]
            return x  # let igraph validate/raise

        def _coerce_node_or_iter(self, obj, igG, label_field: str | None):
            if isinstance(obj, (list, tuple, set)):
                coerced = [self._coerce_vertex(v, igG, label_field) for v in obj]
                return type(obj)(coerced) if not isinstance(obj, set) else set(coerced)
            return self._coerce_vertex(obj, igG, label_field)

        def _coerce_nodes_in_kwargs(self, kwargs: dict, igG, label_field: str | None):
            node_keys = {
                "source",
                "target",
                "u",
                "v",
                "vertex",
                "vertices",
                "vs",
                "to",
                "fr",
                "root",
                "roots",
                "neighbors",
                "nbunch",
                "path",
                "cut",
            }
            for key in list(kwargs.keys()):
                if key in node_keys:
                    kwargs[key] = self._coerce_node_or_iter(kwargs[key], igG, label_field)

        def _coerce_nodes_in_bound(self, bound, igG, label_field: str | None):
            node_keys = {
                "source",
                "target",
                "u",
                "v",
                "vertex",
                "vertices",
                "vs",
                "to",
                "fr",
                "root",
                "roots",
                "neighbors",
                "nbunch",
                "path",
                "cut",
            }
            for key in list(bound.arguments.keys()):
                if key in node_keys:
                    bound.arguments[key] = self._coerce_node_or_iter(
                        bound.arguments[key], igG, label_field
                    )

        # ---------------------- edge-attr & multiedge helpers -------------------
        def _prune_edge_attributes(self, igG, needed_attrs: set):
            import igraph as _ig

            if not needed_attrs:
                # keep only 'name' on vertices, drop all edge attrs quickly by rebuild
                H = _ig.Graph(directed=igG.is_directed())
                H.add_vertices(igG.vcount())
                if "name" in igG.vs.attributes():
                    H.vs["name"] = igG.vs["name"]
                H.add_edges([e.tuple for e in igG.es])
                return H
            # keep only specific attrs
            H = _ig.Graph(directed=igG.is_directed())
            H.add_vertices(igG.vcount())
            if "name" in igG.vs.attributes():
                H.vs["name"] = igG.vs["name"]
            edges = [e.tuple for e in igG.es]
            H.add_edges(edges)
            have = set(igG.es.attributes())
            for k in needed_attrs:
                if k in have:
                    H.es[k] = igG.es[k]
            return H

        def _collapse_multiedges(
            self, igG, *, directed: bool, aggregations: dict | None, needed_attrs: set
        ):
            import igraph as _ig

            H = _ig.Graph(directed=directed)
            H.add_vertices(igG.vcount())
            if "name" in igG.vs.attributes():
                H.vs["name"] = igG.vs["name"]

            aggregations = aggregations or {}

            def _agg_for(key):
                agg = aggregations.get(key)
                if callable(agg):
                    return agg
                if agg == "sum":
                    return sum
                if agg == "min":
                    return min
                if agg == "max":
                    return max
                if agg == "mean":
                    return lambda vals: (sum(vals) / len(vals)) if vals else None
                if key == "capacity":
                    return sum
                if key == "weight":
                    return min
                return lambda vals: next(iter(vals)) if vals else None

            # bucket edges
            buckets = {}  # (u,v) or sorted(u,v) -> {attr: [vals]}
            for e in igG.es:
                u, v = e.tuple
                key = (u, v) if directed else tuple(sorted((u, v)))
                entry = buckets.setdefault(key, {})
                for k, val in e.attributes().items():
                    if needed_attrs and k not in needed_attrs:
                        continue
                    entry.setdefault(k, []).append(val)

            edges = list(buckets.keys())
            H.add_edges(edges)
            # aggregate per attribute
            all_attrs = set(k for _, attrs in buckets.items() for k in attrs.keys())
            for k in all_attrs:
                agg = _agg_for(k)
                H.es[k] = [agg(buckets[edge].get(k, [])) for edge in edges]
            return H

    # For SBML Stoechiometry

    def set_hyperedge_coeffs(self, edge_id: str, coeffs: dict[str, float]) -> None:
        """Write per-vertex coefficients into the incidence column (DOK [dictionary of keys])."""
        col = self.edge_to_idx[edge_id]
        for vid, coeff in coeffs.items():
            row = self.entity_to_idx[vid]
            self._matrix[row, col] = float(coeff)

    # AnnNet API

    def X(self):
        """Sparse incidence matrix."""
        return self._matrix

    @property
    def obs(self):
        """Node attribute table (observations)."""
        return self.vertex_attributes

    @property
    def var(self):
        """Edge attribute table (variables)."""
        return self.edge_attributes

    @property
    def uns(self):
        """Unstructured metadata."""
        return self.graph_attributes

    @property
    def layers(self):
        """Layer operations (add, remove, union, intersect)."""
        if not hasattr(self, "_layer_manager"):
            self._layer_manager = LayerManager(self)
        return self._layer_manager

    @property
    def idx(self):
        """Index lookups (entity_id↔row, edge_id↔col)."""
        if not hasattr(self, "_index_manager"):
            self._index_manager = IndexManager(self)
        return self._index_manager

    @property
    def cache(self):
        """Cache management (CSR/CSC materialization)."""
        if not hasattr(self, "_cache_manager"):
            self._cache_manager = CacheManager(self)
        return self._cache_manager

    # I/O
    def write(self, path, **kwargs):
        """Save to .annnet format (zero loss)."""
        from ..io.io_annnet import write

        write(self, path, **kwargs)

    @classmethod
    def read(cls, path, **kwargs):
        """Load from .annnet format."""
        from ..io.io_annnet import read

        return read(path, **kwargs)

    # View API
    def view(self, nodes=None, edges=None, layers=None, predicate=None):
        """Create lazy view/subgraph."""
        return GraphView(self, nodes, edges, layers, predicate)

    # Audit
    def snapshot(self, label=None):
        """Create a named snapshot of current graph state.

        Uses existing Graph attributes: entity_types, edge_to_idx, _layers, _version

        Parameters
        ----------
        label : str, optional
            Human-readable label for snapshot (auto-generated if None)

        Returns
        -------
        dict
            Snapshot metadata

        """
        from datetime import datetime

        if label is None:
            label = f"snapshot_{len(self._snapshots)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        snapshot = {
            "label": label,
            "version": self._version,
            "timestamp": datetime.now(UTC).isoformat(),
            "counts": {
                "vertices": self.number_of_vertices(),
                "edges": self.number_of_edges(),
                "layers": len(self._layers),
            },
            # Store minimal state for comparison (uses existing Graph attributes)
            "vertex_ids": set(v for v, t in self.entity_types.items() if t == "vertex"),
            "edge_ids": set(self.edge_to_idx.keys()),
            "layer_ids": set(self._layers.keys()),
        }

        self._snapshots.append(snapshot)
        return snapshot

    def diff(self, a, b=None):
        """Compare two snapshots or compare snapshot with current state.

        Parameters
        ----------
        a : str | dict | Graph
            First snapshot (label, snapshot dict, or Graph instance)
        b : str | dict | Graph | None
            Second snapshot. If None, compare with current state.

        Returns
        -------
        GraphDiff
            Difference object with added/removed entities

        """
        snap_a = self._resolve_snapshot(a)
        snap_b = self._resolve_snapshot(b) if b is not None else self._current_snapshot()

        return GraphDiff(snap_a, snap_b)

    def _resolve_snapshot(self, ref):
        """Resolve snapshot reference (label, dict, or Graph)."""
        if isinstance(ref, dict):
            return ref
        elif isinstance(ref, str):
            # Find by label
            for snap in self._snapshots:
                if snap["label"] == ref:
                    return snap
            raise ValueError(f"Snapshot '{ref}' not found")
        elif isinstance(ref, Graph):
            # Create snapshot from another graph (uses Graph attributes)
            return {
                "label": "external",
                "version": ref._version,
                "vertex_ids": set(v for v, t in ref.entity_types.items() if t == "vertex"),
                "edge_ids": set(ref.edge_to_idx.keys()),
                "layer_ids": set(ref._layers.keys()),
            }
        else:
            raise TypeError(f"Invalid snapshot reference: {type(ref)}")

    def _current_snapshot(self):
        """Create snapshot of current state (uses Graph attributes)."""
        return {
            "label": "current",
            "version": self._version,
            "vertex_ids": set(v for v, t in self.entity_types.items() if t == "vertex"),
            "edge_ids": set(self.edge_to_idx.keys()),
            "layer_ids": set(self._layers.keys()),
        }

    def list_snapshots(self):
        """List all snapshots.

        Returns
        -------
        list[dict]
            Snapshot metadata

        """
        return [
            {
                "label": snap["label"],
                "timestamp": snap["timestamp"],
                "version": snap["version"],
                "counts": snap["counts"],
            }
            for snap in self._snapshots
        ]
