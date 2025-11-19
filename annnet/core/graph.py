import inspect
import math
import time
from collections import defaultdict
from datetime import UTC, datetime
from enum import Enum
from functools import wraps
import itertools
from typing import Optional, List, Tuple, Union

import numpy as np
import polars as pl
import scipy.sparse as sp


class EdgeType(Enum):
    DIRECTED = "DIRECTED"
    UNDIRECTED = "UNDIRECTED"


class SliceManager:
    """Manager for graph slice operations.

    Provides organized namespace for slice operations by delegating to Graph methods.
    All heavy lifting is done by the Graph class; this is just a clean API surface.

    """

    def __init__(self, graph):
        self._G = graph

    # ==================== Basic Operations (Delegation) ====================

    def add(self, slice_id, **attributes):
        """Create new slice.

        Delegates to Graph.add_slice()
        """
        return self._G.add_slice(slice_id, **attributes)

    def remove(self, slice_id):
        """Remove slice.

        Delegates to Graph.remove_slice()
        """
        return self._G.remove_slice(slice_id)

    def list(self, include_default=False):
        """List slice IDs.

        Delegates to Graph.list_slices()
        """
        return self._G.list_slices(include_default=include_default)

    def exists(self, slice_id):
        """Check if slice exists.

        Delegates to Graph.has_slice()
        """
        return self._G.has_slice(slice_id)

    def info(self, slice_id):
        """Get slice metadata.

        Delegates to Graph.get_slice_info()
        """
        return self._G.get_slice_info(slice_id)

    def count(self):
        """Get number of slices.

        Delegates to Graph.slice_count()
        """
        return self._G.slice_count()

    def vertices(self, slice_id):
        """Get vertices in slice.

        Delegates to Graph.get_slice_vertices()
        """
        return self._G.get_slice_vertices(slice_id)

    def edges(self, slice_id):
        """Get edges in slice.

        Delegates to Graph.get_slice_edges()
        """
        return self._G.get_slice_edges(slice_id)

    # ==================== Active slice Property ====================

    @property
    def active(self):
        """Get active slice ID.

        Delegates to Graph.get_active_slice()
        """
        return self._G.get_active_slice()

    @active.setter
    def active(self, slice_id):
        """Set active slice ID.

        Delegates to Graph.set_active_slice()
        """
        self._G.set_active_slice(slice_id)

    # ==================== Set Operations (Pure Delegation) ====================

    def union(self, slice_ids):
        """Compute union of slices (returns dict, doesn't create slice).

        Delegates to Graph.slice_union()

        Parameters
        --
        slice_ids : list[str]
            slices to union

        Returns
        ---
        dict
            {"vertices": set[str], "edges": set[str]}

        """
        return self._G.slice_union(slice_ids)

    def intersect(self, slice_ids):
        """Compute intersection of slices (returns dict, doesn't create slice).

        Delegates to Graph.slice_intersection()

        Parameters
        --
        slice_ids : list[str]
            slices to intersect

        Returns
        ---
        dict
            {"vertices": set[str], "edges": set[str]}

        """
        return self._G.slice_intersection(slice_ids)

    def difference(self, slice_a, slice_b):
        """Compute set difference (returns dict, doesn't create slice).

        Delegates to Graph.slice_difference()

        Parameters
        --
        slice_a : str
            First slice
        slice_b : str
            Second slice

        Returns
        ---
        dict
            {"vertices": set[str], "edges": set[str]}
            Elements in slice_a but not in slice_b

        """
        return self._G.slice_difference(slice_a, slice_b)

    # ==================== Creation from Operations ====================

    def union_create(self, slice_ids, name, **attributes):
        """Create new slice as union of existing slices.

        Combines Graph.slice_union() + Graph.create_slice_from_operation()

        Parameters
        --
        slice_ids : list[str]
            slices to union
        name : str
            New slice name
        **attributes
            slice attributes

        Returns
        ---
        str
            Created slice ID

        """
        result = self._G.slice_union(slice_ids)
        return self._G.create_slice_from_operation(name, result, **attributes)

    def intersect_create(self, slice_ids, name, **attributes):
        """Create new slice as intersection of existing slices.

        Combines Graph.slice_intersection() + Graph.create_slice_from_operation()

        Parameters
        --
        slice_ids : list[str]
            slices to intersect
        name : str
            New slice name
        **attributes
            slice attributes

        Returns
        ---
        str
            Created slice ID

        """
        result = self._G.slice_intersection(slice_ids)
        return self._G.create_slice_from_operation(name, result, **attributes)

    def difference_create(self, slice_a, slice_b, name, **attributes):
        """Create new slice as difference of two slices.

        Combines Graph.slice_difference() + Graph.create_slice_from_operation()

        Parameters
        --
        slice_a : str
            First slice
        slice_b : str
            Second slice
        name : str
            New slice name
        **attributes
            slice attributes

        Returns
        ---
        str
            Created slice ID

        """
        result = self._G.slice_difference(slice_a, slice_b)
        return self._G.create_slice_from_operation(name, result, **attributes)

    def aggregate(
        self, source_slice_ids, target_slice_id, method="union", weight_func=None, **attributes
    ):
        """Create aggregated slice from multiple sources.

        Delegates to Graph.create_aggregated_slice()

        Parameters
        --
        source_slice_ids : list[str]
            Source slices
        target_slice_id : str
            Target slice name
        method : {'union', 'intersection'}
            Aggregation method
        weight_func : callable, optional
            Weight merging function (reserved)
        **attributes
            slice attributes

        Returns
        ---
        str
            Created slice ID

        """
        return self._G.create_aggregated_slice(
            source_slice_ids, target_slice_id, method, weight_func, **attributes
        )

    # ==================== Analysis & Queries ====================

    def stats(self, include_default=False):
        """Get statistics for all slices.

        Delegates to Graph.slice_statistics()

        Returns
        ---
        dict[str, dict]
            {slice_id: {'vertices': int, 'edges': int, 'attributes': dict}}

        """
        return self._G.slice_statistics(include_default=include_default)

    def vertex_presence(self, vertex_id, include_default=False):
        """Find slices containing a vertex.

        Delegates to Graph.vertex_presence_across_slices()

        Parameters
        --
        vertex_id : str
            Vertex to search for
        include_default : bool
            Include default slice

        Returns
        ---
        list[str]
            slice IDs containing the vertex

        """
        return self._G.vertex_presence_across_slices(vertex_id, include_default)

    def edge_presence(
        self, edge_id=None, source=None, target=None, include_default=False, undirected_match=None
    ):
        """Find slices containing an edge.

        Delegates to Graph.edge_presence_across_slices()

        Parameters
        --
        edge_id : str, optional
            Edge ID to search for
        source : str, optional
            Source vertex (with target)
        target : str, optional
            Target vertex (with source)
        include_default : bool
            Include default slice
        undirected_match : bool, optional
            Allow symmetric matches

        Returns
        ---
        list[str] or dict[str, list[str]]
            If edge_id: list of slice IDs
            If source/target: {slice_id: [edge_ids]}

        """
        return self._G.edge_presence_across_slices(
            edge_id,
            source,
            target,
            include_default=include_default,
            undirected_match=undirected_match,
        )

    def hyperedge_presence(self, members=None, head=None, tail=None, include_default=False):
        """Find slices containing a hyperedge.

        Delegates to Graph.hyperedge_presence_across_slices()

        Parameters
        --
        members : Iterable[str], optional
            Undirected hyperedge members
        head : Iterable[str], optional
            Directed hyperedge head
        tail : Iterable[str], optional
            Directed hyperedge tail
        include_default : bool
            Include default slice

        Returns
        ---
        dict[str, list[str]]
            {slice_id: [edge_ids]}

        """
        return self._G.hyperedge_presence_across_slices(
            members=members, head=head, tail=tail, include_default=include_default
        )

    def conserved_edges(self, min_slices=2, include_default=False):
        """Find edges present in multiple slices.

        Delegates to Graph.conserved_edges()

        Parameters
        --
        min_slices : int
            Minimum number of slices
        include_default : bool
            Include default slice

        Returns
        ---
        dict[str, int]
            {edge_id: slice_count}

        """
        return self._G.conserved_edges(min_slices, include_default)

    def specific_edges(self, slice_id):
        """Find edges unique to a slice.

        Delegates to Graph.slice_specific_edges()

        Parameters
        --
        slice_id : str
            slice to check

        Returns
        ---
        set[str]
            Edge IDs unique to this slice

        """
        return self._G.slice_specific_edges(slice_id)

    def temporal_dynamics(self, ordered_slices, metric="edge_change"):
        """Analyze temporal changes across slices.

        Delegates to Graph.temporal_dynamics()

        Parameters
        --
        ordered_slices : list[str]
            slices in chronological order
        metric : {'edge_change', 'vertex_change'}
            What to track

        Returns
        ---
        list[dict]
            Per-step changes: [{'added': int, 'removed': int, 'net_change': int}]

        """
        return self._G.temporal_dynamics(ordered_slices, metric)

    # ==================== Convenience Methods ====================

    def summary(self):
        """Get human-readable summary of all slices.

        Returns
        ---
        str
            Formatted summary

        """
        stats = self.stats(include_default=True)
        lines = [f"slices: {len(stats)}"]

        for i, (slice_id, info) in enumerate(stats.items()):
            prefix = "├─" if i < len(stats) - 1 else "└─"
            lines.append(f"{prefix} {slice_id}: {info['vertices']} vertices, {info['edges']} edges")

        return "\n".join(lines)

    def __repr__(self):
        return f"SliceManager({self.count()} slices)"


class LayerManager:
    """Manager for Kivela multi-layer operations.

    Provides organized namespace for layer operations by delegating to Graph methods.
    All heavy lifting is done by the Graph class; this is just a clean API surface.

    """

    def __init__(self, graph):
        self._G = graph

    # ==================== Multi-aspect awareness ====================

    def aspects(self):
        """List aspect names."""
        return list(self._G.aspects)

    def elementary_layers(self):
        """Dict[aspect -> list of elementary labels]."""
        return {a: list(v) for a, v in self._G.elem_layers.items()}

    def layer_tuples(self):
        """List all aspect-tuples (Cartesian product)."""
        return list(self._G.iter_layers())

    def tuple_id(self, aa):
        """Canonical string id for a layer tuple (matches Graph’s synthetic id)."""
        aa = tuple(aa)
        if len(self._G.aspects) == 1:
            return aa[0]
        return "×".join(aa)

    # ==================== Presence utilities ====================

    def vertex_layers(self, u: str):
        """All layer-tuples where vertex u is present."""
        return list(self._G.iter_vertex_layers(u))

    def has_presence(self, u: str, aa):
        """True if (u, aa) ∈ V_M."""
        return self._G.has_presence(u, tuple(aa))

    # ==================== Aspect / layer / vertex-layer attributes ===========

    def set_aspect_attrs(self, aspect: str, **attrs):
        """Attach metadata to an aspect (delegates to Graph.set_aspect_attrs)."""
        return self._G.set_aspect_attrs(aspect, **attrs)

    def aspect_attrs(self, aspect: str) -> dict:
        """Get metadata dict for an aspect."""
        return self._G.get_aspect_attrs(aspect)

    def set_layer_attrs(self, aa, **attrs):
        """Attach metadata to a Kivela layer aa (aspect tuple)."""
        return self._G.set_layer_attrs(tuple(aa), **attrs)

    def layer_attrs(self, aa) -> dict:
        """Get metadata dict for a Kivela layer aa (aspect tuple)."""
        return self._G.get_layer_attrs(tuple(aa))

    def set_vertex_layer_attrs(self, u: str, aa, **attrs):
        """Attach metadata to a vertex–layer pair (u, aa)."""
        return self._G.set_vertex_layer_attrs(u, tuple(aa), **attrs)

    def vertex_layer_attrs(self, u: str, aa) -> dict:
        """Get metadata dict for a vertex–layer pair (u, aa)."""
        return self._G.get_vertex_layer_attrs(u, tuple(aa))

    # ==================== Elementary layer attributes ===========

    def elem_layer_id(self, aspect: str, label: str) -> str:
        """Canonical '{aspect}_{label}' id used in Graph.layer_attributes."""
        return self._G._elem_layer_id(aspect, label)

    def set_elem_layer_attrs(self, aspect: str, label: str, **attrs):
        """
        Upsert attributes for elementary Kivela layer (aspect, label).

        Writes into Graph.layer_attributes with layer_id = "{aspect}_{label}".
        """
        return self._G.set_elementary_layer_attrs(aspect, label, **attrs)

    def elem_layer_attrs(self, aspect: str, label: str) -> dict:
        """
        Read attributes for elementary Kivela layer (aspect, label) as dict.

        Reads from Graph.layer_attributes.
        """
        return self._G.get_elementary_layer_attrs(aspect, label)

    # ==================== Algebra on Kivela layers =========================

    def vertex_set(self, aa):
        """Vertices present in Kivela layer aa (tuple)."""
        return self._G.layer_vertex_set(tuple(aa))

    def edge_set(self, aa, include_inter=False, include_coupling=False):
        """Edges associated with Kivela layer aa."""
        return self._G.layer_edge_set(
            tuple(aa),
            include_inter=include_inter,
            include_coupling=include_coupling,
        )

    def union(self, layer_tuples, include_inter=False, include_coupling=False):
        """Set-union over Kivela layers; returns {'vertices', 'edges'}."""
        return self._G.layer_union(
            [tuple(a) for a in layer_tuples],
            include_inter=include_inter,
            include_coupling=include_coupling,
        )

    def intersection(self, layer_tuples, include_inter=False, include_coupling=False):
        """Set-intersection over Kivela layers; returns {'vertices', 'edges'}."""
        return self._G.layer_intersection(
            [tuple(a) for a in layer_tuples],
            include_inter=include_inter,
            include_coupling=include_coupling,
        )

    def difference(self, layer_a, layer_b, include_inter=False, include_coupling=False):
        """Set-difference layer_a layer_b; returns {'vertices', 'edges'}."""
        return self._G.layer_difference(
            tuple(layer_a),
            tuple(layer_b),
            include_inter=include_inter,
            include_coupling=include_coupling,
        )

    # ==================== Layer-slice bridge ===============================

    def to_slice(self, aa, slice_id=None, include_inter=False, include_coupling=False, **attrs):
        """
        Create a slice from a single Kivela layer aa and return its slice_id.
        """
        aa = tuple(aa)
        sid = slice_id or self.tuple_id(aa)
        return self._G.create_slice_from_layer(
            sid,
            aa,
            include_inter=include_inter,
            include_coupling=include_coupling,
            **attrs,
        )

    def union_to_slice(self, layer_tuples, slice_id, include_inter=False, include_coupling=False, **attrs):
        """
        Create slice from union of several Kivela layers.
        """
        return self._G.create_slice_from_layer_union(
            slice_id,
            [tuple(a) for a in layer_tuples],
            include_inter=include_inter,
            include_coupling=include_coupling,
            **attrs,
        )

    def intersection_to_slice(self, layer_tuples, slice_id, include_inter=False, include_coupling=False, **attrs):
        """
        Create slice from intersection of several Kivela layers.
        """
        return self._G.create_slice_from_layer_intersection(
            slice_id,
            [tuple(a) for a in layer_tuples],
            include_inter=include_inter,
            include_coupling=include_coupling,
            **attrs,
        )

    def difference_to_slice(self, layer_a, layer_b, slice_id, include_inter=False, include_coupling=False, **attrs):
        """
        Create slice from set-difference layer_a layer_b.
        """
        return self._G.create_slice_from_layer_difference(
            slice_id,
            tuple(layer_a),
            tuple(layer_b),
            include_inter=include_inter,
            include_coupling=include_coupling,
            **attrs,
        )

    # ==================== Subgraphs =======================================

    def subgraph(self, aa, include_inter=False, include_coupling=False):
        """Concrete subgraph induced by Kivela layer aa."""
        return self._G.subgraph_from_layer_tuple(
            tuple(aa),
            include_inter=include_inter,
            include_coupling=include_coupling,
        )

    def subgraph_union(self, layer_tuples, include_inter=False, include_coupling=False):
        """Subgraph induced by union of several Kivela layers."""
        return self._G.subgraph_from_layer_union(
            [tuple(a) for a in layer_tuples],
            include_inter=include_inter,
            include_coupling=include_coupling,
        )

    def subgraph_intersection(self, layer_tuples, include_inter=False, include_coupling=False):
        """Subgraph induced by intersection of several Kivela layers."""
        return self._G.subgraph_from_layer_intersection(
            [tuple(a) for a in layer_tuples],
            include_inter=include_inter,
            include_coupling=include_coupling,
        )

    def subgraph_difference(self, layer_a, layer_b, include_inter=False, include_coupling=False):
        """Subgraph induced by layer_a layer_b."""
        return self._G.subgraph_from_layer_difference(
            tuple(layer_a),
            tuple(layer_b),
            include_inter=include_inter,
            include_coupling=include_coupling,
        )

    # ==================== Intra/inter/coupling surfacing ====================

    def intra_edges_tuple(self, aa):
        """Edge IDs of intra edges inside tuple-layer aa."""
        aa = tuple(aa)
        # intra appear in Graph.edge_kind with edge_layers[eid] == aa
        return {eid for eid, k in self._G.edge_kind.items() if k == "intra" and self._G.edge_layers[eid] == aa}

    def inter_edges_between(self, aa, β):
        """Edge IDs of inter edges between tuple-layers aa and β."""
        aa = tuple(aa); β = tuple(β)
        return {eid for eid, k in self._G.edge_kind.items() if k == "inter" and self._G.edge_layers[eid] == (aa, β)}

    def coupling_edges_between(self, aa, β):
        """Edge IDs of coupling edges connecting same-vertex (aa)↔(β)."""
        aa = tuple(aa); β = tuple(β)
        return {eid for eid, k in self._G.edge_kind.items() if k == "coupling" and self._G.edge_layers[eid] == (aa, β)}

    # ==================== Supra / blocks ====================

    def supra_adjacency(self, layers=None):
        """Proxy to full supra A over selected layer-tuples."""
        return self._G.supra_adjacency(layers)

    def blocks(self, layers=None):
        """Return dict of diagonal/off-diagonal blocks."""
        return {
            "intra": self._G.build_intra_block(layers),
            "inter": self._G.build_inter_block(layers),
            "coupling": self._G.build_coupling_block(layers),
        }


class IndexManager:
    """Namespace for index operations.
    Provides clean API over existing dicts.
    """

    def __init__(self, graph):
        self._G = graph

    # ==================== Entity (vertex) Indexes ====================

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
        --
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
        --
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
        ---
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
    --
    graph : Graph
        Parent graph instance
    vertices : list[str] | set[str] | callable | None
        vertex IDs to include, or predicate function
    edges : list[str] | set[str] | callable | None
        Edge IDs to include, or predicate function
    layers : str | list[str] | None
        Layer ID(s) to include
    predicate : callable | None
        Additional filter: predicate(vertex_id) -> bool

    """

    def __init__(self, graph, vertices=None, edges=None, layers=None, predicate=None):
        self._graph = graph
        self._vertices_filter = vertices
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
        self._vertex_ids_cache = None
        self._edge_ids_cache = None
        self._computed = False

    # ==================== Properties ====================

    @property
    def obs(self):
        """Filtered vertex attribute table (uses Graph.vertex_attributes)."""
        vertex_ids = self.vertex_ids
        if vertex_ids is None:
            return self._graph.vertex_attributes

        

        return self._graph.vertex_attributes.filter(pl.col("vertex_id").is_in(list(vertex_ids)))

    @property
    def var(self):
        """Filtered edge attribute table (uses Graph.edge_attributes)."""
        edge_ids = self.edge_ids
        if edge_ids is None:
            return self._graph.edge_attributes

        

        return self._graph.edge_attributes.filter(pl.col("edge_id").is_in(list(edge_ids)))

    @property
    def X(self):
        """Filtered incidence matrix subview."""
        vertex_ids = self.vertex_ids
        edge_ids = self.edge_ids

        # Get row and column indices
        if vertex_ids is not None:
            rows = [
                self._graph.entity_to_idx[nid]
                for nid in vertex_ids
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
            

            return sp.dok_matrix((len(rows), len(cols)), dtype=self._graph._matrix.dtype)

    @property
    def vertex_ids(self):
        """Get filtered vertex IDs (cached)."""
        if not self._computed:
            self._compute_ids()
        return self._vertex_ids_cache

    @property
    def edge_ids(self):
        """Get filtered edge IDs (cached)."""
        if not self._computed:
            self._compute_ids()
        return self._edge_ids_cache

    @property
    def vertex_count(self):
        """Number of vertices in this view."""
        vertex_ids = self.vertex_ids
        if vertex_ids is None:
            return sum(1 for t in self._graph.entity_types.values() if t == "vertex")
        return len(vertex_ids)

    @property
    def edge_count(self):
        """Number of edges in this view."""
        edge_ids = self.edge_ids
        if edge_ids is None:
            return len(self._graph.edge_to_idx)
        return len(edge_ids)

    # ==================== Internal Computation ====================

    def _compute_ids(self):
        """Compute and cache filtered vertex and edge IDs."""
        vertex_ids = None
        edge_ids = None

        # Step 1: Apply layer filter (uses Graph._layers)
        if self._layers is not None:
            vertex_ids = set()
            edge_ids = set()
            for layer_id in self._layers:
                if layer_id in self._graph._layers:
                    vertex_ids.update(self._graph._layers[layer_id]["vertices"])
                    edge_ids.update(self._graph._layers[layer_id]["edges"])

        # Step 2: Apply vertex filter
        if self._vertices_filter is not None:
            candidate_vertices = (
                vertex_ids
                if vertex_ids is not None
                else set(
                    vid for vid, vtype in self._graph.entity_types.items() if vtype == "vertex"
                )
            )

            if callable(self._vertices_filter):
                filtered_vertices = set()
                for vid in candidate_vertices:
                    try:
                        if self._vertices_filter(vid):
                            filtered_vertices.add(vid)
                    except Exception:
                        pass
                vertex_ids = filtered_vertices
            else:
                specified = set(self._vertices_filter)
                if vertex_ids is not None:
                    vertex_ids &= specified
                else:
                    vertex_ids = specified & candidate_vertices

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

        # Step 4: Apply additional predicate to vertices
        if self._predicate is not None and vertex_ids is not None:
            filtered_vertices = set()
            for vid in vertex_ids:
                try:
                    if self._predicate(vid):
                        filtered_vertices.add(vid)
                except Exception:
                    pass
            vertex_ids = filtered_vertices

        # Step 5: Filter edges by vertex connectivity (uses Graph.edge_definitions, hyperedge_definitions)
        if vertex_ids is not None and edge_ids is not None:
            filtered_edges = set()
            for eid in edge_ids:
                # Binary/vertex-edge edges
                if eid in self._graph.edge_definitions:
                    source, target, _ = self._graph.edge_definitions[eid]
                    if source in vertex_ids and target in vertex_ids:
                        filtered_edges.add(eid)
                # Hyperedges
                elif eid in self._graph.hyperedge_definitions:
                    hdef = self._graph.hyperedge_definitions[eid]
                    if hdef.get("directed", False):
                        heads = set(hdef.get("head", []))
                        tails = set(hdef.get("tail", []))
                        if heads.issubset(vertex_ids) and tails.issubset(vertex_ids):
                            filtered_edges.add(eid)
                    else:
                        members = set(hdef.get("members", []))
                        if members.issubset(vertex_ids):
                            filtered_edges.add(eid)
            edge_ids = filtered_edges

        # Cache results
        self._vertex_ids_cache = vertex_ids
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
            

            df = df.filter(pl.col("edge_id").is_in(list(edge_ids)))

        return df

    def vertices_df(self, **kwargs):
        """Get vertex DataFrame view.
        Uses Graph.vertices_view() and filters by vertex IDs.
        """
        # Use Graph's existing vertices_view() method
        df = self._graph.vertices_view(**kwargs)

        # Filter by vertex IDs in this view
        vertex_ids = self.vertex_ids
        if vertex_ids is not None:
            

            df = df.filter(pl.col("vertex_id").is_in(list(vertex_ids)))

        return df

    # ==================== Materialization (uses Graph methods) ====================

    def materialize(self, copy_attributes=True):
        """Create a concrete subgraph from this view.
        Uses Graph.add_vertex(), add_edge(), add_hyperedge(), get_*_attrs()
        """
        # Create new Graph instance
        subG = Graph(directed=self._graph.directed)

        vertex_ids = self.vertex_ids
        edge_ids = self.edge_ids

        # Determine which vertices to copy
        if vertex_ids is not None:
            vertices_to_copy = vertex_ids
        else:
            vertices_to_copy = [
                vid for vid, vtype in self._graph.entity_types.items() if vtype == "vertex"
            ]

        # Copy vertices (uses Graph.add_vertex, get_vertex_attrs)
        for vid in vertices_to_copy:
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

                if source not in vertices_to_copy or target not in vertices_to_copy:
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

                    if not all(h in vertices_to_copy for h in heads):
                        continue
                    if not all(t in vertices_to_copy for t in tails):
                        continue

                    weight = self._graph.edge_weights.get(eid, 1.0)
                    if copy_attributes:
                        attrs = self._graph.get_edge_attrs(eid)
                        subG.add_hyperedge(head=heads, tail=tails, weight=weight, **attrs)
                    else:
                        subG.add_hyperedge(head=heads, tail=tails, weight=weight)
                else:
                    members = list(hdef.get("members", []))

                    if not all(m in vertices_to_copy for m in members):
                        continue

                    weight = self._graph.edge_weights.get(eid, 1.0)
                    if copy_attributes:
                        attrs = self._graph.get_edge_attrs(eid)
                        subG.add_hyperedge(members=members, weight=weight, **attrs)
                    else:
                        subG.add_hyperedge(members=members, weight=weight)

        return subG

    def subview(self, vertices=None, edges=None, layers=None, predicate=None):
        """Create a new GraphView by further restricting this view.

        - vertices/edges: if a list/set is given, intersect with this view's vertex_ids/edge_ids.
        - layers: defaults to this view's layers if None.
        - predicate: applied in addition to the current filtering (AND).
        """
        # Force compute current filters
        base_vertices = self.vertex_ids
        base_edges = self.edge_ids

        # vertices
        if vertices is None:
            new_vertices = base_vertices
            vertex_pred = None
        elif callable(vertices):
            new_vertices = base_vertices  # keep current set; apply new predicate below
            vertex_pred = vertices
        else:
            to_set = set(vertices)
            new_vertices = (set(base_vertices) & to_set) if base_vertices is not None else to_set
            vertex_pred = None

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
            if vertex_pred:
                try:
                    ok = ok and bool(vertex_pred(v))
                except Exception:
                    ok = False
            return ok

        final_pred = combined_pred if (self._predicate or predicate or vertex_pred) else None

        # Return a fresh GraphView
        return GraphView(
            self._graph, vertices=new_vertices, edges=new_edges, layers=new_layers, predicate=final_pred
        )

    # ==================== Convenience ====================

    def summary(self):
        """Human-readable summary."""
        lines = [
            "GraphView Summary",
            "─" * 30,
            f"vertices: {self.vertex_count}",
            f"Edges: {self.edge_count}",
        ]

        filters = []
        if self._layers:
            filters.append(f"layers={self._layers}")
        if self._vertices_filter:
            if callable(self._vertices_filter):
                filters.append("vertices=<predicate>")
            else:
                filters.append(f"vertices={len(list(self._vertices_filter))} specified")
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
        return f"GraphView(vertices={self.vertex_count}, edges={self.edge_count})"

    def __len__(self):
        return self.vertex_count


class GraphDiff:
    """Represents the difference between two graph states.

    Attributes
    --
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
            f"Diff: {self.snapshot_a['label']} - {self.snapshot_b['label']}",
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
    """Sparse incidence-matrix graph with slices, attributes, parallel edges, and hyperedges.

    The graph is backed by a DOK (Dictionary Of Keys) sparse matrix and exposes
    sliceed views and attribute tables stored as Polars DF (DataFrame). Supports:
    vertices, binary edges (directed/undirected), edge-entities (vertex-edge hybrids),
    k-ary hyperedges (directed/undirected), per-slice membership and weights,
    and Polars-backed attribute upserts.

    Parameters
    --
    directed : bool, optional
        Whether edges are directed by default. Individual edges can override this.

    Notes
    -
    - Incidence columns encode orientation: +w on source/head, −w on target/tail for
      directed edges; +w on all members for undirected edges/hyperedges.
    - Attributes are **pure**: structural keys are filtered out so attribute tables
      contain only user data.

    See Also
    
    add_vertex, add_edge, add_hyperedge, edges_view, vertices_view, slices_view

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
        "slice",
        "slice_weight",
        "kind",
        "members",
        "head",
        "tail",
    }
    _slice_RESERVED = {"slice_id"}

    # Construction

    def __init__(self, directed=None, n: int = 0, e: int = 0, **kwargs):
        """Initialize an empty incidence-matrix graph.

        Parameters
        --
        directed : bool, optional
            Global default for edge directionality. Individual edges can override this.

        Notes
        -
        - Stores entities (vertices and edge-entities), edges (including parallels), and
        an incidence matrix in DOK (Dictionary Of Keys) sparse format.
        - Attribute tables are Polars DF (DataFrame) with canonical key columns:
        ``vertex_attributes(vertex_id)``, ``edge_attributes(edge_id)``,
        ``slice_attributes(slice_id)``, and
        ``edge_slice_attributes(slice_id, edge_id, weight)``.
        - A ``'default'`` slice is created and set active.

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
        self.slice_attributes = pl.DataFrame(schema={"slice_id": pl.Utf8})
        self.edge_slice_attributes = pl.DataFrame(
            schema={"slice_id": pl.Utf8, "edge_id": pl.Utf8, "weight": pl.Float64}
        )
        self.edge_kind = {}
        self.hyperedge_definitions = {}
        self.graph_attributes = {}
        self.layer_attributes = pl.DataFrame(schema={"layer_id": pl.Utf8})

        # Edge ID counter for parallel edges
        self._next_edge_id = 0

        # slice management - lightweight dict structure
        self._slices = {}  # slice_id -> {"vertices": set(), "edges": set(), "attributes": {}}
        self._current_slice = None
        self._default_slice = "default"
        self.slice_edge_weights = defaultdict(dict)  # slice_id -> {edge_id: weight}

        # Initialize default slice
        self._slices[self._default_slice] = {"vertices": set(), "edges": set(), "attributes": {}}
        self._current_slice = self._default_slice

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

        # Multi-aspect definition
        self.aspects: list[str] = []                         # e.g., ["time", "relation"]
        self.elem_layers: dict[str, list[str]] = {}          # aspect -> elementary labels
        self._all_layers: tuple[tuple[str, ...], ...] = ()   # cartesian product cache

        # vertex and vertex–layer presence
        self._V: set[str] = set()                             # vertex ids (entities of type 'vertex')
        self._VM: set[tuple[str, tuple[str, ...]]] = set()    # {(u, aa_tuple)}
        self.vertex_aligned: bool = False                      # if True, VM == V × all_layers

        # Stable indexing for supra rows
        self._nl_to_row: dict[tuple[str, tuple[str, ...]], int] = {}
        self._row_to_nl: list[tuple[str, tuple[str, ...]]] = []

        # Legacy 1-aspect shim: when aspects==1 we map layer_id "L" -> ("L",)
        self._legacy_single_aspect_enabled: bool = True
        
        # Multilayer edge bookkeeping (used by supra_adjacency)
        self.edge_kind = {}     # eid -> {"intra","inter","coupling"}
        self.edge_layers = {}   # eid -> aa   or -> (aa,β) for inter/coupling

        # Aspect / layer / vertex–layer attribute tables (Kivela metadata)
        # All of this is annotation on top of the structural incidence.
        self._aspect_attrs = {}        # aspect_name -> {attr_name: value}
        self._layer_attrs = {}         # aa (tuple[str,...]) -> {attr_name: value}
        self._vertex_layer_attrs = {}    # (u, aa) -> {attr_name: value}

    # slice basics

    def add_slice(self, slice_id, **attributes):
        """Create a new empty slice.

        Parameters
        --
        slice_id : str
            New slice identifier (ID).
        **attributes
            Pure slice attributes to store (non-structural).

        Returns
        ---
        str
            The created slice ID.

        Raises
        --
        ValueError
            If the slice already exists.

        """
        if slice_id in self._slices and slice_id != "default":
            raise ValueError(f"slice {slice_id} already exists")

        self._slices[slice_id] = {"vertices": set(), "edges": set(), "attributes": attributes}
        # Persist slice metadata to DF (pure attributes, upsert)
        if attributes:
            self.set_slice_attrs(slice_id, **attributes)
        # slice_id as an elementary slice of that aspect
        if len(self.aspects) == 1:
            a = self.aspects[0]
            if a in self.elem_layers:
                if slice_id not in self.elem_layers[a]:
                    self.elem_layers[a].append(slice_id)
        return slice_id

    def set_active_slice(self, slice_id):
        """Set the active slice for subsequent operations.

        Parameters
        --
        slice_id : str
            Existing slice ID.

        Raises
        --
        KeyError
            If the slice does not exist.

        """
        if slice_id not in self._slices:
            raise KeyError(f"slice {slice_id} not found")
        self._current_slice = slice_id

    def get_active_slice(self):
        """Get the currently active slice ID.

        Returns
        ---
        str
            Active slice ID.

        """
        return self._current_slice

    def get_slices_dict(self, include_default: bool = False):
        """Get a mapping of slice IDs to their metadata.

        Parameters
        --
        include_default : bool, optional
            Include the internal ``'default'`` slice if True.

        Returns
        ---
        dict[str, dict]
            ``{slice_id: {"vertices": set, "edges": set, "attributes": dict}}``.

        """
        if include_default:
            return self._slices
        return {k: v for k, v in self._slices.items() if k != self._default_slice}

    def list_slices(self, include_default: bool = False):
        """List slice IDs.

        Parameters
        --
        include_default : bool, optional
            Include the internal ``'default'`` slice if True.

        Returns
        ---
        list[str]
            slice IDs.

        """
        return list(self.get_slices_dict(include_default=include_default).keys())

    def has_slice(self, slice_id):
        """Check whether a slice exists.

        Parameters
        --
        slice_id : str

        Returns
        ---
        bool

        """
        return slice_id in self._slices

    def slice_count(self):
        """Get the number of slices (including the internal default).

        Returns
        ---
        int

        """
        return len(self._slices)

    def get_slice_info(self, slice_id):
        """Get a slice's metadata snapshot.

        Parameters
        --
        slice_id : str

        Returns
        ---
        dict
            Copy of ``{"vertices": set, "edges": set, "attributes": dict}``.

        Raises
        --
        KeyError
            If the slice does not exist.

        """
        if slice_id not in self._slices:
            raise KeyError(f"slice {slice_id} not found")
        return self._slices[slice_id].copy()
    
    # Multilayers

    ## Aspects & layers

    def set_aspects(self, aspects: list[str], elem_layers: dict[str, list[str]]):
        """
        Define multi-aspect structure.
        aspects: e.g., ["time","relation"]
        elem_layers: {"time": ["t1","t2"], "relation": ["F","A"], ...}
        """
        if not aspects:
            raise ValueError("aspects must be a non-empty list")
        # normalize & copy
        self.aspects = list(aspects)
        self.elem_layers = {a: list(elem_layers.get(a, [])) for a in self.aspects}
        self._rebuild_all_layers_cache()

    def _rebuild_all_layers_cache(self):
        if not self.aspects:
            self._all_layers = ()
            return
        # build cartesian product (tuple of tuples)
        try:
            spaces = [self.elem_layers[a] for a in self.aspects]
        except KeyError as e:
            raise KeyError(f"elem_layers missing for aspect {e!s}")
        self._all_layers = tuple(itertools.product(*spaces)) if all(spaces) else ()

    ## Presence (V_M)
    
    def add_presence(self, u: str, layer_tuple: tuple[str, ...]):
        """Declare that vertex u is present in the given multi-aspect layer."""
        self._validate_layer_tuple(layer_tuple)
        if self.entity_types.get(u) != "vertex":
            raise KeyError(f"'{u}' is not a vertex")
        self._V.add(u)
        self._VM.add((u, tuple(layer_tuple)))

    def remove_presence(self, u: str, layer_tuple: tuple[str, ...]):
        """Remove presence (u, aa)."""
        self._validate_layer_tuple(layer_tuple)
        try:
            self._VM.remove((u, tuple(layer_tuple)))
        except KeyError:
            pass

    def has_presence(self, u: str, layer_tuple: tuple[str, ...]) -> bool:
        self._validate_layer_tuple(layer_tuple)
        return (u, tuple(layer_tuple)) in self._VM

    def iter_layers(self):
        """Iterate over all aspect-tuples (Cartesian product)."""
        return iter(self._all_layers)

    def iter_vertex_layers(self, u: str):
        """Iterate aa where (u, aa) ∈ V_M."""
        for (uu, aa) in self._VM:
            if uu == u:
                yield aa

    ## Index for supra rows 

    def ensure_vertex_layer_index(self, restrict_layers: list[tuple[str, ...]] | None = None):
        """
        Build stable mapping between vertex–layer tuples and row indices.
        Sorting: by vertex_id (lexicographic), then by layer tuple.
        """
        if restrict_layers is not None:
            R = {tuple(x) for x in restrict_layers}
            vm = [(u, aa) for (u, aa) in self._VM if aa in R]
        else:
            vm = list(self._VM)
        vm.sort(key=lambda x: (x[0], x[1]))
        self._row_to_nl = vm
        self._nl_to_row = {nl: i for i, nl in enumerate(vm)}
        return len(vm)

    def nl_to_row(self, u: str, layer_tuple: tuple[str, ...]) -> int:
        """Map (u, aa) -> row index (after ensure_vertex_layer_index())."""
        key = (u, tuple(layer_tuple))
        if key not in self._nl_to_row:
            raise KeyError(f"vertex–layer {key!r} not indexed; call ensure_vertex_layer_index()")
        return self._nl_to_row[key]

    def row_to_nl(self, row: int) -> tuple[str, tuple[str, ...]]:
        """Map row index -> (u, aa)."""
        try:
            return self._row_to_nl[row]
        except Exception:
            raise KeyError(f"row {row} not in vertex–layer index")

    ## Validation helpers 
    
    def _validate_layer_tuple(self, aa: tuple[str, ...]):
        if not self.aspects:
            raise ValueError("no aspects are configured; call set_aspects(...) first")
        if len(aa) != len(self.aspects):
            raise ValueError(f"layer tuple rank mismatch: expected {len(self.aspects)}, got {len(aa)}")
        for i, a in enumerate(self.aspects):
            allowed = self.elem_layers.get(a, [])
            if aa[i] not in allowed:
                raise KeyError(f"unknown elementary layer {aa[i]!r} for aspect {a!r}")

    def layer_id_to_tuple(self, layer_id: str) -> tuple[str, ...]:
        """
        Map legacy string layer_id to aspect tuple.
        Only defined for single-aspect setups.
        """
        if len(self.aspects) != 1:
            raise ValueError("layer_id_to_tuple is only valid when exactly 1 aspect is configured")
        return (layer_id,)

    def layer_tuple_to_id(self, aa: tuple[str, ...]) -> str:
        """
        Canonical string id for a layer tuple.
        For single-aspect setups this is just the single element,
        otherwise aspects are joined by '×' (matches LayerManager.tuple_id).
        """
        aa = tuple(aa)
        if len(self.aspects) == 1:
            return aa[0]
        return "×".join(aa)

    ## Aspect / layer / vertex–layer attributes

    def _elem_layer_id(self, aspect: str, label: str) -> str:
        """
        Canonical id for an *elementary* Kivela layer (aspect, label).

        This is the key used in `layer_attributes.layer_id`:
            layer_id = "{aspect}_{label}"
        """
        if aspect not in self.aspects:
            raise KeyError(f"unknown aspect {aspect!r}; known: {self.aspects!r}")
        allowed = self.elem_layers.get(aspect, [])
        if label not in allowed:
            raise KeyError(
                f"unknown elementary layer {label!r} for aspect {aspect!r}; "
                f"known: {allowed!r}"
            )
        return f"{aspect}_{label}"

    def _upsert_layer_attribute_row(self, layer_id: str, attrs: dict):
        """
        Upsert a row in `self.layer_attributes` for `layer_id`.

        Strategy (simple & robust):
          - convert current DF to list[dict]
          - find existing row for this layer_id (if any)
          - merge attrs into that row (override keys)
          - rebuild DataFrame from the updated list of rows

        This avoids all schema/dtype headaches (Polars infers them).
        """
        import polars as pl

        df = self.layer_attributes

        # Convert existing DF to list of dict rows
        rows = df.to_dicts() if df.height > 0 else []

        # Find if we already have a row for this layer_id
        existing = None
        new_rows = []
        for r in rows:
            if r.get("layer_id") == layer_id:
                existing = r
                # don't append the old version
            else:
                new_rows.append(r)

        if existing is None:
            base = {"layer_id": layer_id}
        else:
            base = dict(existing)  # copy

        # Merge new attrs (override old keys)
        base.update(attrs)

        # Append updated row
        new_rows.append(base)

        # Rebuild DF; Polars will infer schema and fill missing values with nulls
        self.layer_attributes = pl.DataFrame(new_rows)

    def set_elementary_layer_attrs(self, aspect: str, label: str, **attrs):
        """
        Attach attributes to an *elementary* Kivela layer (aspect, label).

        Stored in `self.layer_attributes` with:
            layer_id = "{aspect}_{label}"
        """
        lid = self._elem_layer_id(aspect, label)
        self._upsert_layer_attribute_row(lid, attrs)

    def get_elementary_layer_attrs(self, aspect: str, label: str) -> dict:
        """
        Get attributes for an *elementary* Kivela layer (aspect, label) as a dict.

        Returns {} if no row exists in `layer_attributes`.
        """
        lid = self._elem_layer_id(aspect, label)
        df = self.layer_attributes
        if df.height == 0 or "layer_id" not in df.columns:
            return {}

        rows = df.filter(pl.col("layer_id") == lid)
        if rows.height == 0:
            return {}

        # single row: drop 'layer_id' and convert to dict
        row = rows.drop("layer_id").to_dicts()[0]
        return row

    def set_aspect_attrs(self, aspect: str, **attrs):
        """
        Attach metadata to a Kivela aspect.
        Example: set_aspect_attrs("time", order="temporal", unit="year")
        """
        if aspect not in self.aspects:
            raise KeyError(f"unknown aspect {aspect!r}; known: {self.aspects!r}")
        d = self._aspect_attrs.setdefault(aspect, {})
        d.update(attrs)

    def get_aspect_attrs(self, aspect: str) -> dict:
        """Return a shallow copy of metadata for a Kivela aspect."""
        if aspect not in self.aspects:
            raise KeyError(f"unknown aspect {aspect!r}")
        return dict(self._aspect_attrs.get(aspect, {}))

    def set_layer_attrs(self, layer_tuple: tuple[str, ...], **attrs):
        """
        Attach metadata to a Kivela layer (aspect-tuple aa).
        Example: set_layer_attrs(("t1", "F"), time=1, relation="friendship")
        """
        aa = tuple(layer_tuple)
        self._validate_layer_tuple(aa)
        d = self._layer_attrs.setdefault(aa, {})
        d.update(attrs)

    def get_layer_attrs(self, layer_tuple: tuple[str, ...]) -> dict:
        """
        Get metadata dict for a Kivela layer (aspect-tuple aa).
        Returns a shallow copy; empty dict if not set.
        """
        aa = tuple(layer_tuple)
        self._validate_layer_tuple(aa)
        return dict(self._layer_attrs.get(aa, {}))

    def set_vertex_layer_attrs(self, u: str, layer_tuple: tuple[str, ...], **attrs):
        """
        Attach metadata to a vertex–layer pair (u, aa).
        Requires that (u, aa) ∈ V_M.
        Example: set_vertex_layer_attrs("u1", ("t1",), activity=3.7)
        """
        aa = tuple(layer_tuple)
        self._assert_presence(u, aa)  # enforce that (u,aa) exists in V_M
        key = (u, aa)
        d = self._vertex_layer_attrs.setdefault(key, {})
        d.update(attrs)

    def get_vertex_layer_attrs(self, u: str, layer_tuple: tuple[str, ...]) -> dict:
        """
        Get metadata dict for a vertex–layer pair (u, aa).
        Returns a shallow copy; empty dict if not set.
        """
        aa = tuple(layer_tuple)
        key = (u, aa)
        return dict(self._vertex_layer_attrs.get(key, {}))

    ## explicit vertex–layer edge APIs

    def set_edge_kivela_role(self, eid: str, role: str, layers):
        """
        Annotate an existing structural edge with Kivela semantics.

        role:
          - "intra"    -> layers = aa tuple
          - "inter"    -> layers = (aa, β)
          - "coupling" -> layers = (aa, β)

        This does *not* create edges or touch incidence; it only sets metadata.
        """
        # Sanity: edge must exist in the structural registry
        if eid not in getattr(self, "edge_definitions", {}) and \
           eid not in getattr(self, "hyperedge_definitions", {}):
            raise KeyError(
                f"Kivela annotation for unknown edge {eid!r}; "
                "create the edge via add_edge/add_hyperedge first."
            )

        role = str(role)
        if role == "intra":
            aa = tuple(layers)
            self._validate_layer_tuple(aa)
            self.edge_kind[eid] = "intra"
            self.edge_layers[eid] = aa
        elif role in {"inter", "coupling"}:
            La, Lb = layers
            La = tuple(La); Lb = tuple(Lb)
            self._validate_layer_tuple(La)
            self._validate_layer_tuple(Lb)
            self.edge_kind[eid] = role
            self.edge_layers[eid] = (La, Lb)
        else:
            raise ValueError(f"unknown Kivela role {role!r}")

    def add_intra_edge(self, u: str, v: str, layer: str, *, weight: float = 1.0, eid: str | None = None):
        """
        Legacy single-axis convenience: (u,v) in 'layer' (string).
        If exactly 1 aspect is configured, this delegates to add_intra_edge_nl with (layer,) tuple.
        """
        if len(getattr(self, "aspects", [])) == 1 and getattr(self, "_legacy_single_aspect_enabled", True):
            aa = self.layer_id_to_tuple(layer)
            return self.add_intra_edge_nl(u, v, aa, weight=weight, eid=eid)
        # Fallback when multi-aspect is not configured: let add_edge handle bookkeeping.
        eid = eid or f"{u}--{v}@{layer}"
        self.add_edge(u, v, layer=layer, weight=weight, edge_id=eid)
        # In legacy single-layer mode we don't have a full aspect tuple; store as ("layer",)
        if self.aspects:
            aa = self.layer_id_to_tuple(layer) if len(self.aspects) == 1 else (layer,)
            self.set_edge_kivela_role(eid, "intra", aa)
        else:
            # no aspects configured -> treat as plain edge; leave edge_kind/edge_layers unset
            pass
        return eid

    def add_intra_edge_nl(self, u: str, v: str, layer_tuple: tuple[str, ...], *,
                          weight: float = 1.0, eid: str | None = None):
        """
        Add (u,v) inside a multi-aspect layer aa (tuple). Requires presence (u,aa),(v,aa) in V_M.
        """
        self._validate_layer_tuple(layer_tuple)
        aa = tuple(layer_tuple)
        self._assert_presence(u, aa)
        self._assert_presence(v, aa)
        eid = eid or f"{u}--{v}@{'.'.join(aa)}"
        # Use a synthetic layer id for intra edges so existing slice bookkeeping runs.
        Lid = aa[0] if len(self.aspects) == 1 else "×".join(aa)
        self.add_edge(u, v, layer=Lid, weight=weight, edge_id=eid)
        # Pure Kivela annotation:
        self.set_edge_kivela_role(eid, "intra", aa)
        return eid

    def add_inter_edge_nl(self, u: str, layer_a: tuple[str, ...], v: str, layer_b: tuple[str, ...], *,
                          weight: float = 1.0, eid: str | None = None):
        """
        Add an inter-layer edge between (u, aa) and (v, β). Requires presence (u,aa),(v,β) in V_M.
        """
        self._validate_layer_tuple(layer_a); self._validate_layer_tuple(layer_b)
        aa, β = tuple(layer_a), tuple(layer_b)
        self._assert_presence(u, aa)
        self._assert_presence(v, β)
        eid = eid or f"{u}--{v}=={'.'.join(aa)}~{'.'.join(β)}"
        # No single slice applies; just register the edge structurally.
        self.add_edge(u, v, weight=weight, edge_id=eid)
        self.set_edge_kivela_role(eid, "inter", (aa, β))
        return eid

    def add_coupling_edge_nl(self, u: str, layer_a: tuple[str, ...], layer_b: tuple[str, ...], *,
                             weight: float = 1.0, eid: str | None = None):
        """
        Add a diagonal coupling (u, aa) <-> (u, β). Requires presence (u,aa),(u,β).
        """
        eid2 = self.add_inter_edge_nl(u, layer_a, u, layer_b, weight=weight, eid=eid)
        # Re-label as coupling so supra_adjacency treats it as off-diagonal coupling
        aa = tuple(layer_a); β = tuple(layer_b)
        self.set_edge_kivela_role(eid2, "coupling", (aa, β))
        return eid2
    
    def layer_vertex_set(self, layer_tuple):
        """
        Vertices present in Kivela layer aa (aspect tuple).

        Uses V_M as SSoT for node-layer presence.
        """
        aa = tuple(layer_tuple)
        return {u for (u, L) in self._VM if L == aa}

    def layer_edge_set(
        self,
        layer_tuple,
        *,
        include_inter: bool = False,
        include_coupling: bool = False,
    ):
        """
        Edges associated with Kivela layer aa.

        By default returns only intra-layer edges:
          { eid | edge_kind[eid] == 'intra' and edge_layers[eid] == aa }

        If include_inter=True / include_coupling=True, also includes inter/coupling
        edges where aa participates in the layer pair (aa, β) or (β, aa).
        """
        aa = tuple(layer_tuple)
        E = set()
        for eid, kind in self.edge_kind.items():
            layers = self.edge_layers.get(eid)

            if kind == "intra":
                if layers == aa:
                    E.add(eid)

            elif kind == "inter" and include_inter:
                # layers expected to be (La, Lb)
                if isinstance(layers, tuple) and len(layers) == 2 and aa in layers:
                    E.add(eid)

            elif kind == "coupling" and include_coupling:
                if isinstance(layers, tuple) and len(layers) == 2 and aa in layers:
                    E.add(eid)

        return E

    ### Legacy inter-layer convenience (string layers) delegates when 1 aspect:

    def add_inter_edge(self, u: str, v: str, layer_a: str, layer_b: str, *,
                       weight: float = 1.0, eid: str | None = None):
        if len(getattr(self, "aspects", [])) == 1 and getattr(self, "_legacy_single_aspect_enabled", True):
            aa, β = self.layer_id_to_tuple(layer_a), self.layer_id_to_tuple(layer_b)
            return self.add_inter_edge_nl(u, aa, v, β, weight=weight, eid=eid)
        # Fallback in non-multi-aspect mode
        eid = eid or f"{u}--{v}=={layer_a}~{layer_b}"
        self.add_edge(u, v, weight=weight, edge_id=eid)
        self.edge_kind[eid] = "inter"
        self.edge_layers[eid] = (layer_a, layer_b)
        return eid

    ## Layer algebra

    def layer_union(
        self,
        layer_tuples,
        *,
        include_inter: bool = False,
        include_coupling: bool = False,
    ):
        """
        Union of several Kivela layers.

        Returns dict:
          {
            "vertices": set[str],
            "edges": set[str],
          }
        """
        Vs = []
        Es = []
        for aa in layer_tuples:
            Vs.append(self.layer_vertex_set(aa))
            Es.append(
                self.layer_edge_set(
                    aa,
                    include_inter=include_inter,
                    include_coupling=include_coupling,
                )
            )
        if not Vs:
            return {"vertices": set(), "edges": set()}
        V = set().union(*Vs)
        E = set().union(*Es)
        return {"vertices": V, "edges": E}

    def layer_intersection(
        self,
        layer_tuples,
        *,
        include_inter: bool = False,
        include_coupling: bool = False,
    ):
        """
        Intersection of several Kivela layers.

        Empty input - empty sets.
        """
        layer_tuples = list(layer_tuples)
        if not layer_tuples:
            return {"vertices": set(), "edges": set()}

        # start with first layer
        V = self.layer_vertex_set(layer_tuples[0])
        E = self.layer_edge_set(
            layer_tuples[0],
            include_inter=include_inter,
            include_coupling=include_coupling,
        )

        for aa in layer_tuples[1:]:
            V &= self.layer_vertex_set(aa)
            E &= self.layer_edge_set(
                aa,
                include_inter=include_inter,
                include_coupling=include_coupling,
            )

        return {"vertices": V, "edges": E}

    def layer_difference(
        self,
        layer_a,
        layer_b,
        *,
        include_inter: bool = False,
        include_coupling: bool = False,
    ):
        """
        Set difference: elements in layer_a but not in layer_b.

        Returns dict {"vertices": set[str], "edges": set[str]}.
        """
        Va = self.layer_vertex_set(layer_a)
        Ea = self.layer_edge_set(
            layer_a,
            include_inter=include_inter,
            include_coupling=include_coupling,
        )
        Vb = self.layer_vertex_set(layer_b)
        Eb = self.layer_edge_set(
            layer_b,
            include_inter=include_inter,
            include_coupling=include_coupling,
        )
        return {
            "vertices": Va - Vb,
            "edges": Ea - Eb,
        }

    ## Layer X Slice

    def create_slice_from_layer(
        self,
        slice_id: str,
        layer_tuple,
        *,
        include_inter: bool = False,
        include_coupling: bool = False,
        **attributes,
    ):
        """
        Create a slice whose vertices/edges are induced by a single Kivela layer.

        Attributes are attached to the slice; you can include the layer tuple for traceability.
        """
        result = self.layer_union(
            [layer_tuple],
            include_inter=include_inter,
            include_coupling=include_coupling,
        )
        attributes.setdefault("source", "kivela_layer")
        attributes.setdefault("layer_tuple", tuple(layer_tuple))
        return self.create_slice_from_operation(slice_id, result, **attributes)

    def create_slice_from_layer_union(
        self,
        slice_id: str,
        layer_tuples,
        *,
        include_inter: bool = False,
        include_coupling: bool = False,
        **attributes,
    ):
        """
        Create a slice as union of several Kivela layers (using layer_union).
        """
        result = self.layer_union(
            layer_tuples,
            include_inter=include_inter,
            include_coupling=include_coupling,
        )
        attributes.setdefault("source", "kivela_layer_union")
        attributes.setdefault("layers", [tuple(a) for a in layer_tuples])
        return self.create_slice_from_operation(slice_id, result, **attributes)

    def create_slice_from_layer_intersection(
        self,
        slice_id: str,
        layer_tuples,
        *,
        include_inter: bool = False,
        include_coupling: bool = False,
        **attributes,
    ):
        """
        Create a slice as intersection of several Kivela layers (using layer_intersection).
        """
        result = self.layer_intersection(
            layer_tuples,
            include_inter=include_inter,
            include_coupling=include_coupling,
        )
        attributes.setdefault("source", "kivela_layer_intersection")
        attributes.setdefault("layers", [tuple(a) for a in layer_tuples])
        return self.create_slice_from_operation(slice_id, result, **attributes)

    def create_slice_from_layer_difference(
        self,
        slice_id: str,
        layer_a,
        layer_b,
        *,
        include_inter: bool = False,
        include_coupling: bool = False,
        **attributes,
    ):
        """
        Create a slice as difference layer_a layer_b.
        """
        result = self.layer_difference(
            layer_a,
            layer_b,
            include_inter=include_inter,
            include_coupling=include_coupling,
        )
        attributes.setdefault("source", "kivela_layer_difference")
        attributes.setdefault("layer_a", tuple(layer_a))
        attributes.setdefault("layer_b", tuple(layer_b))
        return self.create_slice_from_operation(slice_id, result, **attributes)

    ## Subgraph

    def subgraph_from_layer_tuple(
        self,
        layer_tuple,
        *,
        include_inter: bool = False,
        include_coupling: bool = False,
    ) -> "Graph":
        """
        Concrete subgraph induced by a single Kivela layer.

        - Vertices = layer_vertex_set(aa)
        - Edges    = layer_edge_set(aa, ...)

        Uses extract_subgraph(vertices, edges) to avoid duplicating logic.
        """
        aa = tuple(layer_tuple)
        V = self.layer_vertex_set(aa)
        E = self.layer_edge_set(
            aa,
            include_inter=include_inter,
            include_coupling=include_coupling,
        )
        return self.extract_subgraph(vertices=V, edges=E)

    def subgraph_from_layer_union(
        self,
        layer_tuples,
        *,
        include_inter: bool = False,
        include_coupling: bool = False,
    ) -> "Graph":
        """
        Concrete subgraph induced by the union of several Kivela layers.
        """
        res = self.layer_union(
            layer_tuples,
            include_inter=include_inter,
            include_coupling=include_coupling,
        )
        return self.extract_subgraph(vertices=res["vertices"], edges=res["edges"])

    def subgraph_from_layer_intersection(
        self,
        layer_tuples,
        *,
        include_inter: bool = False,
        include_coupling: bool = False,
    ) -> "Graph":
        """
        Concrete subgraph induced by intersection of several Kivela layers.
        """
        res = self.layer_intersection(
            layer_tuples,
            include_inter=include_inter,
            include_coupling=include_coupling,
        )
        return self.extract_subgraph(vertices=res["vertices"], edges=res["edges"])

    def subgraph_from_layer_difference(
        self,
        layer_a,
        layer_b,
        *,
        include_inter: bool = False,
        include_coupling: bool = False,
    ) -> "Graph":
        """
        Concrete subgraph induced by layer_a layer_b.
        """
        res = self.layer_difference(
            layer_a,
            layer_b,
            include_inter=include_inter,
            include_coupling=include_coupling,
        )
        return self.extract_subgraph(vertices=res["vertices"], edges=res["edges"])

    ## helper

    def _assert_presence(self, u: str, aa: tuple[str, ...]):
        if (u, aa) not in self._VM:
            raise KeyError(f"presence missing: {(u, aa)} not in V_M; call add_presence(u, aa) first")

    ## Supra_Adjacency

    def supra_adjacency(self, layers: list[str] | None = None):
        # Map optional legacy 'layers' (strings) to aspect tuples if needed
        if layers is not None and len(getattr(self, "aspects", [])) == 1 and getattr(self, "_legacy_single_aspect_enabled", True):
            layers_t = [self.layer_id_to_tuple(L) for L in layers]
        else:
            layers_t = None if layers is None else [tuple(L) for L in layers]
        self.ensure_vertex_layer_index(layers_t)
 
        n = len(self._row_to_nl)
        A = sp.dok_matrix((n, n), dtype=float)
 
        # Fill diagonal blocks from intra-layer edges
        for eid, kind in self.edge_kind.items():
            if kind != "intra":
                continue
            L = self.edge_layers[eid]
            # normalize L to tuple
            if not isinstance(L, tuple):
                if len(getattr(self, "aspects", [])) == 1 and getattr(self, "_legacy_single_aspect_enabled", True):
                   L = self.layer_id_to_tuple(L)
                else:
                   raise ValueError("intra edge layer is not a tuple; configure aspects or use add_intra_edge_nl")
            if layers_t is not None and L not in layers_t:
                continue
            # resolve endpoints (u,v) in layer L
            try:
                u, v, _etype = self.edge_definitions[eid]
            except KeyError:
                continue
            ru = self._nl_to_row.get((u, L))
            rv = self._nl_to_row.get((v, L))
            if ru is None or rv is None:
                continue
            w = self.edge_weights.get(eid, 1)
            A[ru, rv] = A.get((ru, rv), 0.0) + w
            A[rv, ru] = A.get((rv, ru), 0.0) + w  # undirected; adapt if directed
 
        # Fill off-diagonal blocks from inter-layer/coupling edges
        for eid, kind in self.edge_kind.items():
            if kind not in {"inter", "coupling"}:
                continue
            La, Lb = self.edge_layers[eid]
            # normalize La/Lb to tuples
            if not isinstance(La, tuple):
                if len(getattr(self, "aspects", [])) == 1 and getattr(self, "_legacy_single_aspect_enabled", True):
                    La = self.layer_id_to_tuple(La)
                else:
                    raise ValueError("inter edge layer_a is not a tuple; configure aspects or use add_inter_edge_nl")
            if not isinstance(Lb, tuple):
                if len(getattr(self, "aspects", [])) == 1 and getattr(self, "_legacy_single_aspect_enabled", True):
                    Lb = self.layer_id_to_tuple(Lb)
                else:
                    raise ValueError("inter edge layer_b is not a tuple; configure aspects or use add_inter_edge_nl")
            if layers_t is not None and (La not in layers_t or Lb not in layers_t):
                 continue
            # endpoints
            try:
                u, v, _etype = self.edge_definitions[eid]
            except KeyError:
                continue
            ru = self._nl_to_row.get((u, La))
            rv = self._nl_to_row.get((v, Lb))
            if ru is None or rv is None:
                continue
            w = self.edge_weights.get(eid, 1)
            A[ru, rv] = A.get((ru, rv), 0.0) + w
            A[rv, ru] = A.get((rv, ru), 0.0) + w  # undirected; adapt if directed
        return A.tocsr()

    ##  Block partitions & Laplacians
    
    def _normalize_layers_arg(self, layers):
        """
        Normalize 'layers' argument to a list of aspect tuples or None.
        Accepts legacy single-aspect string IDs.
        """
        if layers is None:
            return None
        if len(getattr(self, "aspects", [])) == 1 and getattr(self, "_legacy_single_aspect_enabled", True):
            return [self.layer_id_to_tuple(L) for L in layers]
        return [tuple(L) for L in layers]

    def _build_block(self, include_kinds: set[str], layers: list[str] | list[tuple] | None = None):
        """
        Internal builder for a supra block that includes only edges with kinds in include_kinds.
        Kinds: {"intra","inter","coupling"}.
        """
        
        layers_t = self._normalize_layers_arg(layers)
        self.ensure_vertex_layer_index(layers_t)
        n = len(self._row_to_nl)
        A = sp.dok_matrix((n, n), dtype=float)

        # Intra-layer edges (diagonal blocks)
        if "intra" in include_kinds:
            for eid, kind in self.edge_kind.items():
                if kind != "intra":
                    continue
                L = self.edge_layers[eid]
                if not isinstance(L, tuple):
                    if len(getattr(self, "aspects", [])) == 1 and getattr(self, "_legacy_single_aspect_enabled", True):
                        L = self.layer_id_to_tuple(L)
                    else:
                        continue
                if layers_t is not None and L not in layers_t:
                    continue
                try:
                    u, v, _etype = self.edge_definitions[eid]
                except KeyError:
                    continue
                ru = self._nl_to_row.get((u, L)); rv = self._nl_to_row.get((v, L))
                if ru is None or rv is None:
                    continue
                w = self.edge_weights.get(eid, 1.0)
                A[ru, rv] = A.get((ru, rv), 0.0) + w
                A[rv, ru] = A.get((rv, ru), 0.0) + w

        # Inter/coupling edges (off-diagonal blocks)
        if include_kinds & {"inter", "coupling"}:
            for eid, kind in self.edge_kind.items():
                if kind not in include_kinds:
                    continue
                La, Lb = self.edge_layers[eid]
                if not isinstance(La, tuple):
                    if len(getattr(self, "aspects", [])) == 1 and getattr(self, "_legacy_single_aspect_enabled", True):
                        La = self.layer_id_to_tuple(La)
                    else:
                        continue
                if not isinstance(Lb, tuple):
                    if len(getattr(self, "aspects", [])) == 1 and getattr(self, "_legacy_single_aspect_enabled", True):
                        Lb = self.layer_id_to_tuple(Lb)
                    else:
                        continue
                if layers_t is not None and (La not in layers_t or Lb not in layers_t):
                    continue
                try:
                    u, v, _etype = self.edge_definitions[eid]
                except KeyError:
                    continue
                ru = self._nl_to_row.get((u, La)); rv = self._nl_to_row.get((v, Lb))
                if ru is None or rv is None:
                    continue
                w = self.edge_weights.get(eid, 1.0)
                A[ru, rv] = A.get((ru, rv), 0.0) + w
                A[rv, ru] = A.get((rv, ru), 0.0) + w

        return A.tocsr()

    def build_intra_block(self, layers: list[str] | list[tuple] | None = None):
        """Supra matrix containing only intra-layer edges (diagonal blocks)."""
        return self._build_block({"intra"}, layers)

    def build_inter_block(self, layers: list[str] | list[tuple] | None = None):
        """Supra matrix containing only inter-layer (non-diagonal) edges (excludes coupling)."""
        return self._build_block({"inter"}, layers)

    def build_coupling_block(self, layers: list[str] | list[tuple] | None = None):
        """Supra matrix containing only coupling (diagonal same-vertex cross-layer) edges."""
        return self._build_block({"coupling"}, layers)

    def supra_degree(self, layers: list[str] | list[tuple] | None = None):
        """
        Degree vector over the supra-graph (sum of row of supra adjacency).
        """
        
        A = self.supra_adjacency(layers)
        # sum over columns per row -> shape (n,1); flatten to 1D
        deg = np.asarray(A.sum(axis=1)).ravel()
        return deg

    def supra_laplacian(self, kind: str = "comb", layers: list[str] | list[tuple] | None = None):
        """
        Build supra-Laplacian.
        kind="comb" -> combinatorial L = D - A
        kind="norm" -> normalized L_sym = I - D^{-1/2} A D^{-1/2}
        """
        
        
        A = self.supra_adjacency(layers)
        n = A.shape[0]
        deg = self.supra_degree(layers)
        if kind == "comb":
            D = sp.diags(deg, format="csr")
            return D - A
        elif kind == "norm":
            # D^{-1/2}; zero where deg==0
            invsqrt = np.zeros_like(deg, dtype=float)
            nz = deg > 0
            invsqrt[nz] = 1.0 / np.sqrt(deg[nz])
            Dm12 = sp.diags(invsqrt, format="csr")
            I = sp.eye(n, format="csr")
            return I - (Dm12 @ A @ Dm12)
        else:
            raise ValueError("kind must be 'comb' or 'norm'")

    ## Coupling generators (vertex-independent)
    
    def _aspect_index(self, aspect: str) -> int:
        if aspect not in self.aspects:
            raise KeyError(f"unknown aspect {aspect!r}; known: {self.aspects!r}")
        return self.aspects.index(aspect)

    def _layer_matches_filter(self, aa: tuple[str, ...], layer_filter: dict[str, set]) -> bool:
        """
        layer_filter: {aspect_name: {elem1, elem2, ...}}; a layer matches if aa[a] ∈ set for all keys.
        """
        if not layer_filter:
            return True
        for a_name, allowed in layer_filter.items():
            i = self._aspect_index(a_name)
            if aa[i] not in allowed:
                return False
        return True

    def add_layer_coupling_pairs(self, layer_pairs: list[tuple[tuple[str, ...], tuple[str, ...]]],
                                 *, weight: float = 1.0) -> int:
        """
        Generic: for each pair (aa, β) in layer_pairs, add diagonal couplings (u,aa)<->(u,β) for all u
        that are present in both layers. Returns number of edges added.
        """
        added = 0
        # normalize to tuples, validate once
        norm_pairs = []
        for (La, Lb) in layer_pairs:
            La = tuple(La); Lb = tuple(Lb)
            self._validate_layer_tuple(La); self._validate_layer_tuple(Lb)
            norm_pairs.append((La, Lb))
        # Build per-layer presence index to avoid O(|V_M|^2)
        layer_to_vertices = {}
        for (u, aa) in self._VM:
            layer_to_vertices.setdefault(aa, set()).add(u)
        for (La, Lb) in norm_pairs:
            Ua = layer_to_vertices.get(La, set())
            Ub = layer_to_vertices.get(Lb, set())
            for u in Ua & Ub:
                self.add_coupling_edge_nl(u, La, Lb, weight=weight)
                added += 1
        return added

    def add_categorical_coupling(self, aspect: str, groups: list[list[str]], *,
                                 weight: float = 1.0) -> int:
        """
        Categorical couplings along one aspect:
        For each vertex u and each fixed assignment of the other aspects, fully connect u across
        the elementary layers in each group on `aspect`.
        Example: aspect="time", groups=[["t1","t2","t3"]] connects (u,t1,⋅)-(u,t2,⋅)-(u,t3,⋅) per (other aspects).
        Returns number of edges added.
        """
        ai = self._aspect_index(aspect)
        added = 0
        # Map: (u, other_aspects_tuple) -> {elem_on_aspect: full_layer_tuple}
        buckets = {}
        for (u, aa) in self._VM:
            other = aa[:ai] + aa[ai+1:]
            buckets.setdefault((u, other), {}).setdefault(aa[ai], aa)
        for grp in groups:
            gset = set(grp)
            for (u, other), mapping in buckets.items():
                # pick only those aa whose aspect element is in this group
                layers = [mapping[e] for e in mapping.keys() if e in gset]
                if len(layers) < 2:
                    continue
                for La, Lb in itertools.combinations(sorted(layers), 2):
                    self.add_coupling_edge_nl(u, La, Lb, weight=weight)
                    added += 1
        return added

    def add_diagonal_coupling_filter(self, layer_filter: dict[str, set], *,
                                     weight: float = 1.0) -> int:
        """
        Diagonal couplings inside a filtered slice of the layer space:
        For each vertex u, connect all (u,aa) pairs where aa matches `layer_filter`.
        layer_filter example: {"time": {"t1","t2"}, "rel": {"F"}}.
        Returns number of edges added.
        """
        added = 0
        # collect per vertex the matching layers actually present
        per_u = {}
        for (u, aa) in self._VM:
            if self._layer_matches_filter(aa, layer_filter):
                per_u.setdefault(u, []).append(aa)
        for u, layers in per_u.items():
            if len(layers) < 2:
                continue
            for La, Lb in itertools.combinations(sorted(layers), 2):
                self.add_coupling_edge_nl(u, La, Lb, weight=weight)
                added += 1
        return added

    ## Tensor view & flattening map
    
    def tensor_index(self, layers: list[str] | list[tuple] | None = None):
        """
        Build consistent indices for vertices and layers used in the tensor view.
        Uses the current vertex–layer index ordering to ensure round-trip with supra.
        Returns:
          vertices: list[str]
          layers_t: list[tuple[str,...]]
          vertex_to_i: dict[vertex->int]
          layer_to_i: dict[layer_tuple->int]
        """
        layers_t = self._normalize_layers_arg(layers)
        self.ensure_vertex_layer_index(layers_t)
        # collect in the order they appear in the vertex–layer index: (vertex major, then layer)
        vertices = []
        layers_list = []
        seen_vertices = set()
        seen_layers = set()
        for (u, aa) in self._row_to_nl:
            if u not in seen_vertices:
                vertices.append(u); seen_vertices.add(u)
            if aa not in seen_layers:
                layers_list.append(aa); seen_layers.add(aa)
        vertex_to_i = {u: i for i, u in enumerate(vertices)}
        layer_to_i = {aa: i for i, aa in enumerate(layers_list)}
        return vertices, layers_list, vertex_to_i, layer_to_i

    def adjacency_tensor_view(self, layers: list[str] | list[tuple] | None = None):
        """
       Sparse 4-index adjacency view: triplets (ui, ai, vi, bi, w).
        Returns a dict:
          {
            "vertices": list[str],
            "layers": list[tuple[str,...]],
            "vertex_to_i": {...},
            "layer_to_i": {...},
            "ui": np.ndarray[int], "ai": np.ndarray[int],
            "vi": np.ndarray[int], "bi": np.ndarray[int],
            "w":  np.ndarray[float]
          }
        Symmetric entries are emitted once (ui,ai,vi,bi) and once swapped (vi,bi,ui,ai).
        """
        
        vertices, layers_t, vertex_to_i, layer_to_i = self.tensor_index(layers)
        ui = []; ai = []; vi = []; bi = []; wv = []

        # Intra edges -> (u,aa)↔(v,aa)
        for eid, kind in self.edge_kind.items():
            if kind != "intra":
                continue
            L = self.edge_layers[eid]
            if not isinstance(L, tuple):
                if len(getattr(self, "aspects", [])) == 1 and getattr(self, "_legacy_single_aspect_enabled", True):
                    L = self.layer_id_to_tuple(L)
                else:
                    continue
            if layers is not None and L not in set(layers_t):
                continue
            try:
                u, v, _etype = self.edge_definitions[eid]
            except KeyError:
                continue
            if (u, L) not in self._nl_to_row or (v, L) not in self._nl_to_row:
                continue
            w = self.edge_weights.get(eid, 1.0)
            ui.extend((vertex_to_i[u], vertex_to_i[v]))
            vi.extend((vertex_to_i[v], vertex_to_i[u]))
            a = layer_to_i[L]
            ai.extend((a, a)); bi.extend((a, a))
            wv.extend((w, w))

        # Inter / coupling -> (u,aa)↔(v,β)
        for eid, kind in self.edge_kind.items():
            if kind not in {"inter", "coupling"}:
                continue
            La, Lb = self.edge_layers[eid]
            if not isinstance(La, tuple):
                if len(getattr(self, "aspects", [])) == 1 and getattr(self, "_legacy_single_aspect_enabled", True):
                    La = self.layer_id_to_tuple(La)
                else:
                    continue
            if not isinstance(Lb, tuple):
                if len(getattr(self, "aspects", [])) == 1 and getattr(self, "_legacy_single_aspect_enabled", True):
                   Lb = self.layer_id_to_tuple(Lb)
                else:
                    continue
            if layers is not None:
                S = set(layers_t)
                if La not in S or Lb not in S:
                    continue
            try:
                u, v, _etype = self.edge_definitions[eid]
            except KeyError:
                continue
            if (u, La) not in self._nl_to_row or (v, Lb) not in self._nl_to_row:
                continue
            w = self.edge_weights.get(eid, 1.0)
            ui.extend((vertex_to_i[u], vertex_to_i[v]))
            vi.extend((vertex_to_i[v], vertex_to_i[u]))
            ai.extend((layer_to_i[La], layer_to_i[Lb]))
            bi.extend((layer_to_i[Lb], layer_to_i[La]))
            wv.extend((w, w))

        return {
            "vertices": vertices,
            "layers": layers_t,
           "vertex_to_i": vertex_to_i,
            "layer_to_i": layer_to_i,
            "ui": np.asarray(ui, dtype=int),
           "ai": np.asarray(ai, dtype=int),
            "vi": np.asarray(vi, dtype=int),
            "bi": np.asarray(bi, dtype=int),
            "w":  np.asarray(wv, dtype=float),
        }

    def flatten_to_supra(self, tensor_view: dict):
        """
        f: 4-index -> supra (CSR). Uses current vertex–layer index mapping (_nl_to_row).
        tensor_view: output of adjacency_tensor_view or unflatten_from_supra.
        """
        
        # Ensure index reflects current layers subset (tensor_view["layers"])
        layers_t = tensor_view["layers"] if tensor_view.get("layers", None) else None
        self.ensure_vertex_layer_index(layers_t)
        n = len(self._row_to_nl)
        A = sp.dok_matrix((n, n), dtype=float)
        vertices = tensor_view["vertices"]; layers = tensor_view["layers"]
        ui, ai, vi, bi, w = (tensor_view["ui"], tensor_view["ai"],
                             tensor_view["vi"], tensor_view["bi"], tensor_view["w"])
        # Map back from indices to (u,aa) rows using current _nl_to_row
        for k in range(len(w)):
            u = vertices[int(ui[k])]; aa = layers[int(ai[k])]
            v = vertices[int(vi[k])]; β = layers[int(bi[k])]
            ru = self._nl_to_row.get((u, aa)); rv = self._nl_to_row.get((v, β))
            if ru is None or rv is None:
                continue
            A[ru, rv] = A.get((ru, rv), 0.0) + float(w[k])
        return A.tocsr()

    def unflatten_from_supra(self, A, layers: list[str] | list[tuple] | None = None):
        """
        f^{-1}: supra -> 4-index tensor triplets (same schema as adjacency_tensor_view).
        """
        
        A = A.tocsr()
        vertices, layers_t, vertex_to_i, layer_to_i = self.tensor_index(layers)
        rows, cols = A.nonzero()
        data = A.data
        ui = np.empty_like(rows); vi = np.empty_like(cols)
        ai = np.empty_like(rows); bi = np.empty_like(cols)
        for k in range(len(rows)):
            (u, aa) = self._row_to_nl[int(rows[k])]
            (v, β) = self._row_to_nl[int(cols[k])]
            ui[k] = vertex_to_i[u]; vi[k] = vertex_to_i[v]
            ai[k] = layer_to_i[aa]; bi[k] = layer_to_i[β]
        return {
            "vertices": vertices, "layers": layers_t,
            "vertex_to_i": vertex_to_i, "layer_to_i": layer_to_i,
            "ui": ui, "ai": ai, "vi": vi, "bi": bi, "w": data.astype(float, copy=False),
        }

    ##  Dynamics & spectral probes

    def supra_adjacency_scaled(self, *, coupling_scale: float = 1.0,
                               include_inter: bool = True,
                               layers: list[str] | list[tuple] | None = None):
        """
        Build supra adjacency with optional scaling of coupling edges and optional
        inclusion of inter-layer (non-diagonal) edges.
        A = A_intra + (include_inter ? A_inter : 0) + coupling_scale * A_coupling
        """
        
        A_intra = self.build_intra_block(layers)
        A_coup  = self.build_coupling_block(layers)
        A_inter = self.build_inter_block(layers) if include_inter else None
        A = A_intra.copy()
        if include_inter:
            A = A + A_inter
        if coupling_scale != 1.0:
            A = A + (A_coup * (coupling_scale - 1.0))
        else:
            A = A + A_coup
        return A.tocsr()

    def transition_matrix(self, layers: list[str] | list[tuple] | None = None):
        """
        Row-stochastic transition matrix P = D^{-1} A (random walk on supra-graph).
        Rows with zero degree remain zero.
        """
        
        
        A = self.supra_adjacency(layers).tocsr()
        deg = self.supra_degree(layers)
        invdeg = np.zeros_like(deg, dtype=float)
        nz = deg > 0
        invdeg[nz] = 1.0 / deg[nz]
        Dinv = sp.diags(invdeg, format="csr")
        return Dinv @ A

    def random_walk_step(self, p, layers: list[str] | list[tuple] | None = None):
        """
        One step of random walk distribution: p' = p P. Accepts dense 1D array-like (len = |V_M|).
        """
        
        P = self.transition_matrix(layers)
        p = np.asarray(p, dtype=float).reshape(1, -1)
        if p.shape[1] != P.shape[0]:
            raise ValueError(f"p has length {p.shape[1]} but supra has size {P.shape[0]}")
        return (p @ P).ravel()

    def diffusion_step(self, x, tau: float = 1.0, kind: str = "comb",
                       layers: list[str] | list[tuple] | None = None):
        """
        One explicit Euler step of diffusion on the supra-graph:
          x' = x - tau * L x   where L is combinatorial (kind='comb') or normalized (kind='norm')
        """
        
        L = self.supra_laplacian(kind=kind, layers=layers)
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != L.shape[0]:
            raise ValueError(f"x has length {x.shape[0]} but supra has size {L.shape[0]}")
        return x - tau * (L @ x)

    def algebraic_connectivity(self, layers: list[str] | list[tuple] | None = None):
        """
        Second-smallest eigenvalue (Fiedler value) of the combinatorial supra-Laplacian.
        Returns (lambda_2, fiedler_vector) or (0.0, None) if |V_M| < 2.
        """
        
        from scipy.sparse.linalg import eigsh
        L = self.supra_laplacian(kind="comb", layers=layers).astype(float)
        n = L.shape[0]
        if n < 2:
            return 0.0, None
        # Compute k=2 smallest eigenvalues of symmetric PSD L
        vals, vecs = eigsh(L, k=2, which="SM", return_eigenvectors=True)
        # Sort just in case
        order = np.argsort(vals)
        vals = vals[order]; vecs = vecs[:, order]
        # lambda_0 ~ 0 (within numerical eps); lambda_1 is algebraic connectivity
        return float(vals[1]), vecs[:, 1]

    def k_smallest_laplacian_eigs(self, k: int = 6, kind: str = "comb",
                                  layers: list[str] | list[tuple] | None = None):
        """
        Convenience: return k smallest eigenvalues/eigenvectors of supra-Laplacian.
        """
        
        from scipy.sparse.linalg import eigsh
        if k < 1:
            raise ValueError("k must be >= 1")
        L = self.supra_laplacian(kind=kind, layers=layers).astype(float)
        k = min(k, max(1, L.shape[0]-1))
        vals, vecs = eigsh(L, k=k, which="SM", return_eigenvectors=True)
        order = np.argsort(vals)
        return vals[order], vecs[:, order]

    def dominant_rw_eigenpair(self, layers: list[str] | list[tuple] | None = None):
        """
        Dominant eigenpair of the random-walk operator P (right eigenvector). Uses eigsh on (P+P^T)/2 fallback if needed.
        Returns (lambda_max, v). For an irreducible RW, lambda_max≈1.
        """
        
        from scipy.sparse.linalg import eigsh
        P = self.transition_matrix(layers).tocsr().astype(float)
        n = P.shape[0]
        if n == 0:
            return 0.0, None
        # Symmetrize for stable eigensolve; still informative about spectral radius.
        S = (P + P.T) * 0.5
        vals, vecs = eigsh(S, k=1, which="LA")
        return float(vals[0]), vecs[:, 0]

    def sweep_coupling_regime(self, scales, metric="algebraic_connectivity",
                               layers: list[str] | list[tuple] | None = None):
        """
        Scan a list/array of coupling scales (omega) and evaluate a metric on the scaled supra graph.
        metric options:
          - "algebraic_connectivity"  -> lambda_2(L_comb(A_intra + A_inter + omega A_coup))
          - callable(A) -> float      -> custom metric that takes a CSR supra matrix
        Returns list of floats aligned with 'scales'.
        """
        results = []
        if isinstance(metric, str):
            metric = metric.strip().lower()
        for ω in scales:
            Aω = self.supra_adjacency_scaled(coupling_scale=float(ω), include_inter=True, layers=layers)
            if metric == "algebraic_connectivity":
                # Compute λ2 of L = D - Aω
                
                from scipy.sparse import diags
                deg = Aω.sum(axis=1).A.ravel()
                L = diags(deg) - Aω
                from scipy.sparse.linalg import eigsh
                if L.shape[0] < 2:
                    results.append(0.0)
                    continue
                vals, _ = eigsh(L.astype(float), k=2, which="SM")
                vals.sort()
                results.append(float(vals[1]))
            elif callable(metric):
                results.append(float(metric(Aω)))
            else:
                raise ValueError("Unknown metric; use 'algebraic_connectivity' or provide a callable(A)->float)")
        return results

    ## Layer-aware descriptors

    def _rows_for_layer(self, L):
        """Return row indices in the supra index that belong to aspect-tuple layer L."""
        if not isinstance(L, tuple):
            if len(getattr(self, "aspects", [])) == 1 and getattr(self, "_legacy_single_aspect_enabled", True):
                L = self.layer_id_to_tuple(L)
            else:
                raise ValueError("Layer id must be an aspect tuple")
        rows = []
        for i, (u, aa) in enumerate(self._row_to_nl):
            if aa == L:
                rows.append(i)
        return rows

    def layer_degree_vectors(self, layers: list[str] | list[tuple] | None = None):
        """
        Per-layer degree vectors (intra-layer only). Returns:
          { layer_tuple: (rows_idx_list, deg_vector_np) }
        """
        
        A_intra = self.build_intra_block(layers).tocsr()
        out = {}
        # choose layers subset from current index
        chosen_layers = self._normalize_layers_arg(layers)
        if chosen_layers is None:
            # infer from current index
            chosen_layers = []
            seen = set()
            for _, aa in self._row_to_nl:
                if aa not in seen:
                    chosen_layers.append(aa); seen.add(aa)
        for L in chosen_layers:
            rows = self._rows_for_layer(L)
            if not rows:
                continue
            sub = A_intra[rows][:, rows]
            deg = np.asarray(sub.sum(axis=1)).ravel()
            out[L] = (rows, deg)
        return out

    def participation_coefficient(self, layers: list[str] | list[tuple] | None = None):
        """
        Participation coefficient per vertex (Guimerà & Amaral style on multiplex):
          P_u = 1 - sum_L (k_u^L / k_u)^2, using intra-layer degrees only.
        Returns dict[vertex -> float].
        """
        
        # build per-layer deg vectors and aggregate per vertex
        layer_deg = self.layer_degree_vectors(layers)
        # aggregate k_u over layers
        per_vertex_total = {}
        per_vertex_by_layer = {}
        for L, (rows, deg) in layer_deg.items():
            for i, r in enumerate(rows):
                u, _ = self._row_to_nl[r]
                per_vertex_total[u] = per_vertex_total.get(u, 0.0) + float(deg[i])
                per_vertex_by_layer.setdefault(u, {})[L] = float(deg[i])
        P = {}
        for u, k in per_vertex_total.items():
            if k <= 0:
                P[u] = 0.0
                continue
            s = 0.0
            for L, kL in per_vertex_by_layer[u].items():
                x = kL / k
                s += x * x
            P[u] = 1.0 - s
        return P

    def versatility(self, layers: list[str] | list[tuple] | None = None):
        """
        Simple versatility proxy: dominant eigenvector of supra adjacency, summed over
        vertex's layer-copies. Returns dict[vertex -> float] normalized to unit max.
        """
        
        from scipy.sparse.linalg import eigsh
        A = self.supra_adjacency(layers).astype(float)
        n = A.shape[0]
        if n == 0:
            return {}
        # largest eigenpair of symmetric A
        vals, vecs = eigsh(A, k=1, which="LA")
        v = vecs[:, 0]
        per_vertex = {}
        for i, (u, _) in enumerate(self._row_to_nl):
            per_vertex[u] = per_vertex.get(u, 0.0) + float(abs(v[i]))
        # normalize
        m = max(per_vertex.values()) if per_vertex else 1.0
        if m > 0:
            for u in per_vertex:
                per_vertex[u] /= m
        return per_vertex

    ## Multislice modularity (scorer)

    def multislice_modularity(self, partition, *, gamma: float = 1.0, omega: float = 1.0,
                              include_inter: bool = False,
                              layers: list[str] | list[tuple] | None = None):
        """
        Mucha et al. multislice modularity (scorer only).
        Uses: intra-layer A and configuration null model per layer; coupling edges scaled by 'omega'.
        Optionally includes inter-layer (non-diagonal) edges if include_inter=True.
        Args:
          partition: list/array of community ids, length = |V_M| in current index.
          gamma: resolution parameter applied uniformly to all layers.
          omega: coupling strength multiplying the coupling block (ignores stored coupling weights).
          include_inter: whether to include inter-layer (non-diagonal) edges in A (default False).
          layers: optional subset of layers to score on (will rebuild the supra index).
        Returns:
          Q (float)
        """
        
        
        # Ensure index over the right layers
        layers_t = self._normalize_layers_arg(layers)
        self.ensure_vertex_layer_index(layers_t)
        n = len(self._row_to_nl)
        part = np.asarray(partition)
        if part.shape[0] != n:
            raise ValueError(f"partition length {part.shape[0]} != |V_M| {n}")
        # Build A = A_intra + (include_inter ? A_inter : 0) + omega * (binary coupling structure)
        A_intra = self.build_intra_block(layers_t).tocsr()
        A_coup  = self.build_coupling_block(layers_t).tocsr()
        # binarize coupling structure then scale by omega
        if A_coup.nnz:
            A_coup = (A_coup > 0).astype(float) * float(omega)
        A = A_intra.copy()
        if include_inter:
            A = A + self.build_inter_block(layers_t).tocsr()
        A = A + A_coup
        # 2μ = total edge weight in supra (sum of all entries)
        two_mu = float(A.sum())
        if two_mu <= 0:
            return 0.0
        # Build expected term (configuration null model) per layer block
        B = A.tolil()
        layer_deg = self.layer_degree_vectors(layers_t)  # degrees from intra only
        # map row -> layer to avoid repeated lookups
        row_layer = [aa for (_, aa) in self._row_to_nl]
        # For each layer L: subtract gamma * (k_i^L k_j^L)/(2 m_L) within its block
        for L, (rows, deg) in layer_deg.items():
            mL2 = float(deg.sum())  # = 2 m_L
            if mL2 <= 0:
                continue
            # outer product deg_i deg_j / (2 m_L)
            # Use sparse rank-1: for each i in rows, for each j in rows
            # But we only need to subtract for pairs that share a community later; to keep simple and general,
            # build a dense block if small, else sparse updates.
            if len(rows) <= 2048:
                
                expected = gamma * (np.outer(deg, deg) / mL2)
                # subtract into B
                for ii, ri in enumerate(rows):
                    # vectorized subtract on row
                    cols = rows
                    B[ri, cols] = (B[ri, cols]).toarray().ravel() - expected[ii, :]
            else:
                # large layer: fall back to sparse updates
                scale = gamma / mL2
                for a, ri in enumerate(rows):
                    if deg[a] == 0:
                        continue
                    valrow = (deg[a] * deg) * scale  # numpy broadcasting
                    B[ri, rows] = (B[ri, rows]).toarray().ravel() - valrow
        B = B.tocsr()
        # Q = (1 / (2μ)) * sum_{i,j} B_{ij} δ(g_i, g_j)
        # Compute by iterating over nonzeros in B and summing those within same community
        rows, cols = B.nonzero()
        data = B.data
        mask = (part[rows] == part[cols])
        Q = float(data[mask].sum()) / two_mu
        return Q


    # ID + entity ensure helpers

    def _get_next_edge_id(self) -> str:
        """INTERNAL: Generate a unique edge ID for parallel edges.

        Returns
        ---
        str
            Fresh ``edge_<n>`` identifier (monotonic counter).

        """
        edge_id = f"edge_{self._next_edge_id}"
        self._next_edge_id += 1
        return edge_id

    def _ensure_vertex_table(self) -> None:
        """INTERNAL: Ensure the vertex attribute table exists with a canonical schema.

        Notes
        -
        - Creates an empty Polars DF [DataFrame] with a single ``Utf8`` ``vertex_id`` column
        if missing or malformed.

        """
        df = getattr(self, "vertex_attributes", None)
        if not isinstance(df, pl.DataFrame) or "vertex_id" not in df.columns:
            self.vertex_attributes = pl.DataFrame({"vertex_id": pl.Series([], dtype=pl.Utf8)})

    def _ensure_vertex_row(self, vertex_id: str) -> None:
        """INTERNAL: Ensure a row for ``vertex_id`` exists in the vertex attribute DF.

        Notes
        -
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

    def add_vertex(self, vertex_id, slice=None, **attributes):
        """Add (or upsert) a vertex and optionally attach it to a slice.

        Parameters
        --
        vertex_id : str
            vertex ID (must be unique across entities).
        slice : str, optional
            Target slice. Defaults to the active slice.
        **attributes
            Pure vertex attributes to store.

        Returns
        ---
        str
            The vertex ID (echoed).

        Notes
        -
        - Ensures a row exists in the Polars DF [DataFrame] for attributes.
        - Resizes the incidence matrix if needed.

        """
        # Fast normalize to cut hashing/dup costs in dicts.
        try:
            import sys as _sys

            if isinstance(vertex_id, str):
                vertex_id = _sys.intern(vertex_id)
            if slice is None:
                slice = self._current_slice
            elif isinstance(slice, str):
                slice = _sys.intern(slice)
        except Exception:
            slice = slice or self._current_slice

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

        # Add to specified slice (create if needed)
        slices = self._slices
        if slice not in slices:
            slices[slice] = {"vertices": set(), "edges": set(), "attributes": {}}
        slices[slice]["vertices"].add(vertex_id)

        # Multislice presence sync:
        # - Track V (true vertices only)
        # - If vertex_aligned: ensure presence across all slices
        # - Else, if we are in 1-aspect shim mode and a slice was given, add (u, (slice,))
        if self.entity_types.get(vertex_id) == "vertex":
            self._V.add(vertex_id)
            if self.vertex_aligned and self._all_slices:
                for aa in self._all_slices:
                    self._VM.add((vertex_id, aa))
            elif slice is not None and len(self.aspects) == 1 and self._legacy_single_aspect_enabled:
                self._VM.add((vertex_id, (slice,)))

        # Ensure vertex_attributes has a row for this vertex (even with no attrs)
        self._ensure_vertex_table()
        self._ensure_vertex_row(vertex_id)

        # Upsert passed attributes (if any)
        if attributes:
            self.vertex_attributes = self._upsert_row(self.vertex_attributes, vertex_id, attributes)

        return vertex_id

    def add_vertices(self, vertices, slice=None, **attributes):
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
            slice=slice,
        )
        return [d["vertex_id"] for d in it]

    def add_edge_entity(self, edge_entity_id, slice=None, **attributes):
        """Add an **edge entity** (vertex-edge hybrid) that can connect to vertices/edges.

        Parameters
        --
        edge_entity_id : str
            Entity ID to register as type ``'edge'`` in the entity set.
        slice : str, optional
            Target slice. Defaults to the active slice.
        **attributes
            Attributes stored in the vertex attribute DF (treated like vertices).

        Returns
        ---
        str
            The edge-entity ID.

        """
        # Resolve slice default and intern hot strings
        slice = slice or self._current_slice
        try:
            import sys as _sys

            if isinstance(edge_entity_id, str):
                edge_entity_id = _sys.intern(edge_entity_id)
            if isinstance(slice, str):
                slice = _sys.intern(slice)
        except Exception:
            pass

        entity_to_idx = self.entity_to_idx
        slices = self._slices

        # Add to global superset if new (delegate to existing helper)
        if edge_entity_id not in entity_to_idx:
            self._add_edge_entity(edge_entity_id)

        # Add to specified slice
        if slice not in slices:
            slices[slice] = {"vertices": set(), "edges": set(), "attributes": {}}
        slices[slice]["vertices"].add(edge_entity_id)

        # Add attributes (treat edge entities like vertices for attributes)
        if attributes:
            self.set_vertex_attrs(edge_entity_id, **attributes)

        return edge_entity_id

    def _add_edge_entity(self, edge_id):
        """INTERNAL: Register an **edge-entity** so edges can attach to it (vertex-edge mode).

        Parameters
        --
        edge_id : str
            Identifier to insert into the entity index as type ``'edge'``.

        Notes
        -
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
        slice=None,
        weight=1.0,
        edge_id=None,
        edge_type="regular",
        propagate="none",
        slice_weight=None,
        directed=None,
        edge_directed=None,
        **attributes,
    ):
        """Add or update a binary edge between two entities.

        Parameters
        --
        source : str
            Source entity ID (vertex or edge-entity for vertex-edge mode).
        target : str
            Target entity ID.
        slice : str, optional
            slice to place the edge into. Defaults to the active slice.
        weight : float, optional
            Global edge weight stored in the incidence column (default 1.0).
        edge_id : str, optional
            Explicit edge ID. If omitted, a fresh ID is generated.
        edge_type : {'regular', 'vertex_edge'}, optional
            Edge kind. ``'vertex_edge'`` allows connecting to an edge-entity.
        propagate : {'none', 'shared', 'all'}, optional
            slice propagation:
            - ``'none'`` : only the specified slice
            - ``'shared'`` : all slices that already contain **both** endpoints
            - ``'all'`` : all slices that contain **either** endpoint (and add the other)
        slice_weight : float, optional
            Per-slice weight override for this edge (stored in edge-slice DF).
        edge_directed : bool, optional
            Override default directedness for this edge. If None, uses graph default.
        **attributes
            Pure edge attributes to upsert.

        Returns
        ---
        str
            The edge ID (new or updated).

        Raises
        --
        ValueError
            If ``propagate`` or ``edge_type`` is invalid.
        TypeError
            If ``weight`` is not numeric.

        Notes
        -
        - Directed edges write ``+weight`` at source row and ``-weight`` at target row.
        - Undirected edges write ``+weight`` at both endpoints.
        - Updating an existing edge ID overwrites its matrix column and metadata.

        """
        if edge_type is None:
            edge_type = "regular"

        # Resolve dict endpoints via composite key (if enabled)
        if self._vertex_key_enabled():
            if isinstance(source, dict):
                source = self.get_or_create_vertex_by_attrs(slice=slice, **source)
            if isinstance(target, dict):
                target = self.get_or_create_vertex_by_attrs(slice=slice, **target)
        
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
                    slice=slice,
                    weight=weight,
                    edge_id=edge_id,
                    **attributes,
                )
            else:
                members = tuple(set(S) | set(T))
                return self.add_hyperedge(
                    members=members,
                    edge_directed=False,
                    slice=slice,
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

        # resolve slice + whether to touch sliceing at all
        slice = self._current_slice if slice is None else slice
        touch_slice = slice is not None

        # Intern common strings to speed up dict lookups
        try:
            import sys as _sys

            if isinstance(source, str):
                source = _sys.intern(source)
            if isinstance(target, str):
                target = _sys.intern(target)
            if isinstance(slice, str):
                slice = _sys.intern(slice)
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
        slices = self._slices
        M = self._matrix  # DOK

        # ensure vertices exist (global)
        def _ensure_vertex_or_edge_entity(x):
            if x in entity_to_idx:
                return
            if edge_type == "vertex_edge" and isinstance(x, str) and x.startswith("edge_"):
                self.add_edge_entity(x, slice=slice)
            else:
                self.add_vertex(x, slice=slice)

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

        # slice handling
        if touch_slice:
            if slice not in slices:
                slices[slice] = {"vertices": set(), "edges": set(), "attributes": {}}
            slices[slice]["edges"].add(edge_id)
            slices[slice]["vertices"].update((source, target))

            if slice_weight is not None:
                w = float(slice_weight)
                self.set_edge_slice_attrs(slice, edge_id, weight=w)
                self.slice_edge_weights.setdefault(slice, {})[edge_id] = w

        # propagation
        if propagate == "shared":
            self._propagate_to_shared_slices(edge_id, source, target)
        elif propagate == "all":
            self._propagate_to_all_slices(edge_id, source, target)

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
        --
        source : str
        target : str
        weight : float, optional
        **attributes
            Pure edge attributes.

        Returns
        ---
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
        slice=None,
        weight=1.0,
        edge_id=None,
        edge_directed=None,  # bool or None (None -> infer from params)
        **attributes,
    ):
        """Create a k-ary hyperedge as a single incidence column.

        Modes
        -
        - **Undirected**: pass ``members`` (>=2). Each member gets ``+weight``.
        - **Directed**: pass ``head`` and ``tail`` (both non-empty, disjoint).
        Head gets ``+weight``; tail gets ``-weight``.
        """
        # Map dict endpoints to vertex_id when composite keys are enabled
        if self._vertex_key_enabled():
            def _map(x): 
                return self.get_or_create_vertex_by_attrs(slice=slice, **x) if isinstance(x, dict) else x
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

        # set slice
        slice = self._current_slice if slice is None else slice

        # Intern frequently-used strings for cheaper dict ops
        try:
            import sys as _sys

            if isinstance(slice, str):
                slice = _sys.intern(slice)
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
        slices = self._slices
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
            self.add_vertex(x, slice=slice)

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

        # slice membership + per-slice vertices
        if slice is not None:
            if slice not in slices:
                slices[slice] = {"vertices": set(), "edges": set(), "attributes": {}}
            slices[slice]["edges"].add(edge_id)
            if members is not None:
                slices[slice]["vertices"].update(members)
            else:
                slices[slice]["vertices"].update(self.hyperedge_definitions[edge_id]["head"])
                slices[slice]["vertices"].update(self.hyperedge_definitions[edge_id]["tail"])

        # attributes
        if attributes:
            self.set_edge_attrs(edge_id, **attributes)

        return edge_id

    def add_edge_to_slice(self, lid, eid):
        """Attach an existing edge to a slice (no weight changes).

        Parameters
        --
        lid : str
            slice ID.
        eid : str
            Edge ID.

        Raises
        --
        KeyError
            If the slice does not exist.

        """
        if lid not in self._slices:
            raise KeyError(f"slice {lid} does not exist")
        self._slices[lid]["edges"].add(eid)

    def _propagate_to_shared_slices(self, edge_id, source, target):
        """INTERNAL: Add an edge to all slices that already contain **both** endpoints.

        Parameters
        --
        edge_id : str
        source : str
        target : str

        """
        for slice_id, slice_data in self._slices.items():
            if source in slice_data["vertices"] and target in slice_data["vertices"]:
                slice_data["edges"].add(edge_id)

    def _propagate_to_all_slices(self, edge_id, source, target):
        """INTERNAL: Add an edge to any slice containing **either** endpoint and
        insert the missing endpoint into that slice.

        Parameters
        --
        edge_id : str
        source : str
        target : str

        """
        for slice_id, slice_data in self._slices.items():
            if source in slice_data["vertices"] or target in slice_data["vertices"]:
                slice_data["edges"].add(edge_id)
                # Only add missing endpoint if both vertices should be in slice
                if source in slice_data["vertices"]:
                    slice_data["vertices"].add(target)
                if target in slice_data["vertices"]:
                    slice_data["vertices"].add(source)

    def _normalize_vertices_arg(self, vertices):
        """Normalize a single vertex or an iterable of vertices into a set.

        This internal utility function standardizes input for methods like
        `in_edges()` and `out_edges()` by converting the argument into a set
        of vertex identifiers.

        Parameters
        --
        vertices : str | Iterable[str] | None
            - A single vertex ID (string).
            - An iterable of vertex IDs (e.g., list, tuple, set).
            - `None` is allowed and will return an empty set.

        Returns
        ---
        set[str]
            A set of vertex identifiers. If `vertices` is `None`, returns an
            empty set. If a single vertex is provided, returns a one-element set.

        Notes
        -
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

    def add_vertices_bulk(self, vertices, slice=None):
        """Bulk add vertices (and edge-entities if prefixed externally).
        Accepts: iterable of str  OR  iterable of (vertex_id, attrs_dict)  OR iterable of dicts with keys {'vertex_id', ...attrs}
        Behavior: identical to calling add_vertex() for each, but resizes once and batches attribute inserts.
        """
        

        slice = slice or self._current_slice

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
            if isinstance(slice, str):
                slice = _sys.intern(slice)
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

        # slice membership (same semantics as add_vertex)
        if slice not in self._slices:
            self._slices[slice] = {"vertices": set(), "edges": set(), "attributes": {}}
        self._slices[slice]["vertices"].update(vid for vid, _ in norm)

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
        slice=None,
        default_weight=1.0,
        default_edge_type="regular",
        default_propagate="none",
        default_slice_weight=None,
        default_edge_directed=None,
    ):
        """Bulk add/update *binary* (and vertex-edge) edges.
        Accepts each item as:
        - (src, tgt)
        - (src, tgt, weight)
        - dict with keys: source, target, [weight, edge_id, edge_type, propagate, slice_weight, edge_directed, attributes]
        Behavior: identical to calling add_edge() per item (same propagation/slice/attrs), but grows columns once and avoids full-column wipes.
        """
        slice = self._current_slice if slice is None else slice

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
            if "slice" not in d:
                d["slice"] = slice
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
                lid = d.get("slice")
                if isinstance(lid, str):
                    d["slice"] = _sys.intern(lid)
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
        # 1) Ensure endpoints exist (global); we’ll rely on slice handling below to add membership.
        for d in norm:
            s, t = d["source"], d["target"]
            et = d.get("edge_type", "regular")
            if s not in entity_to_idx:
                # vertex or edge-entity depending on mode?
                if et == "vertex_edge" and isinstance(s, str) and s.startswith("edge_"):
                    self._add_edge_entity(s)
                else:
                    # bare global insert (no slice side-effects; membership handled later)
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
            slice_local = d.get("slice", slice)
            slice_w = d.get("slice_weight", default_slice_weight)
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

            # slice membership + optional per-slice weight
            if slice_local is not None:
                if slice_local not in self._slices:
                    self._slices[slice_local] = {
                        "vertices": set(),
                        "edges": set(),
                        "attributes": {},
                    }
                self._slices[slice_local]["edges"].add(edge_id)
                self._slices[slice_local]["vertices"].update((s, t))
                if slice_w is not None:
                    self.set_edge_slice_attrs(slice_local, edge_id, weight=float(slice_w))
                    self.slice_edge_weights.setdefault(slice_local, {})[edge_id] = float(slice_w)

            # propagation
            if prop == "shared":
                self._propagate_to_shared_slices(edge_id, s, t)
            elif prop == "all":
                self._propagate_to_all_slices(edge_id, s, t)

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
        slice=None,
        default_weight=1.0,
        default_edge_directed=None,
    ):
        """Bulk add/update hyperedges.
        Each item can be:
        - {'members': [...], 'edge_id': ..., 'weight': ..., 'slice': ..., 'attributes': {...}}
        - {'head': [...], 'tail': [...], ...}
        Behavior: identical to calling add_hyperedge() per item, but grows columns once and avoids full-column wipes.
        """
        slice = self._current_slice if slice is None else slice

        items = []
        for it in hyperedges:
            if not isinstance(it, dict):
                continue
            d = dict(it)
            d.setdefault("weight", default_weight)
            if "slice" not in d:
                d["slice"] = slice
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
                lid = d.get("slice")
                if isinstance(lid, str):
                    d["slice"] = _sys.intern(lid)
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
            slice_local = d.get("slice", slice)
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

            # slice membership
            if slice_local is not None:
                if slice_local not in self._slices:
                    self._slices[slice_local] = {
                        "vertices": set(),
                        "edges": set(),
                        "attributes": {},
                    }
                self._slices[slice_local]["edges"].add(e_id)
                if members is not None:
                    self._slices[slice_local]["vertices"].update(members)
                else:
                    self._slices[slice_local]["vertices"].update(head)
                    self._slices[slice_local]["vertices"].update(tail)

            # per-edge attributes (optional)
            attrs = d.get("attributes") or d.get("attrs") or {}
            if attrs:
                self.set_edge_attrs(e_id, **attrs)

            out_ids.append(e_id)

        return out_ids

    def add_edges_to_slice_bulk(self, slice_id, edge_ids):
        """Bulk version of add_edge_to_slice: add many edges to a slice and attach
        all incident vertices. No weights are changed here.
        """
        slice = slice_id if slice_id is not None else self._current_slice
        if slice not in self._slices:
            self._slices[slice] = {"vertices": set(), "edges": set(), "attributes": {}}
        L = self._slices[slice]

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

    def add_edge_entities_bulk(self, items, slice=None):
        """Bulk add edge-entities (vertex-edge hybrids). Accepts:
        - iterable of str IDs
        - iterable of (edge_entity_id, attrs_dict)
        - iterable of dicts with key 'edge_entity_id' (or 'id')
        Behavior: identical to calling add_edge_entity() for each, but grows rows once
        and batches attribute inserts.
        """
        slice = slice or self._current_slice

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
            if isinstance(slice, str):
                slice = _sys.intern(slice)
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

        # slice membership
        if slice not in self._slices:
            self._slices[slice] = {"vertices": set(), "edges": set(), "attributes": {}}
        self._slices[slice]["vertices"].update(eid for eid, _ in norm)

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
        --
        edge_id : str

        Raises
        --
        KeyError
            If the edge is not found.

        Notes
        -
        - Physically removes the incidence column (no CSR round-trip).
        - Cleans edge attributes, slice memberships, and per-slice entries.

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

        # Remove from per-slice membership
        for slice_data in self._slices.values():
            slice_data["edges"].discard(edge_id)

        # Remove from edge-slice attributes
        if (
            isinstance(self.edge_slice_attributes, pl.DataFrame)
            and self.edge_slice_attributes.height > 0
            and "edge_id" in self.edge_slice_attributes.columns
        ):
            self.edge_slice_attributes = self.edge_slice_attributes.filter(
                pl.col("edge_id") != edge_id
            )

        # Legacy / auxiliary dicts
        for d in self.slice_edge_weights.values():
            d.pop(edge_id, None)

        self.edge_kind.pop(edge_id, None)
        self.hyperedge_definitions.pop(edge_id, None)

    def remove_vertex(self, vertex_id):
        """Remove a vertex and all incident edges (binary + hyperedges).

        Parameters
        --
        vertex_id : str

        Raises
        --
        KeyError
            If the vertex is not found.

        Notes
        -
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

        # Remove from per-slice membership
        for slice_data in self._slices.values():
            slice_data["vertices"].discard(vertex_id)

    def remove_slice(self, slice_id):
        """Remove a non-default slice and its per-slice attributes.

        Parameters
        --
        slice_id : str

        Raises
        --
        ValueError
            If attempting to remove the internal default slice.
        KeyError
            If the slice does not exist.

        Notes
        -
        - Does not delete vertices/edges globally; only membership and slice metadata.

        """
        if slice_id == self._default_slice:
            raise ValueError("Cannot remove default slice")
        if slice_id not in self._slices:
            raise KeyError(f"slice {slice_id} not found")

        # Purge per-slice attributes
        ela = getattr(self, "edge_slice_attributes", None)
        if isinstance(ela, pl.DataFrame) and ela.height > 0 and "slice_id" in ela.columns:
            # Keep everything not matching the slice_id
            self.edge_slice_attributes = ela.filter(pl.col("slice_id") != slice_id)

        # Drop legacy dict slice if present
        if isinstance(getattr(self, "slice_edge_weights", None), dict):
            self.slice_edge_weights.pop(slice_id, None)

        # Remove the slice and reset current if needed
        del self._slices[slice_id]
        if self._current_slice == slice_id:
            self._current_slice = self._default_slice

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
        for slice_data in self._slices.values():
            slice_data["edges"].difference_update(drop)
        for d in self.slice_edge_weights.values():
            for eid in drop:
                d.pop(eid, None)

        # DataFrames
        if isinstance(self.edge_attributes, pl.DataFrame) and self.edge_attributes.height:
            if "edge_id" in self.edge_attributes.columns:
                self.edge_attributes = self.edge_attributes.filter(
                    ~pl.col("edge_id").is_in(list(drop))
                )
        if (
            isinstance(self.edge_slice_attributes, pl.DataFrame)
            and self.edge_slice_attributes.height
        ):
            cols = set(self.edge_slice_attributes.columns)
            if {"edge_id"}.issubset(cols):
                self.edge_slice_attributes = self.edge_slice_attributes.filter(
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

        # 6) Clean vertex attributes and slice memberships
        if isinstance(self.vertex_attributes, pl.DataFrame) and self.vertex_attributes.height:
            if "vertex_id" in self.vertex_attributes.columns:
                self.vertex_attributes = self.vertex_attributes.filter(
                    ~pl.col("vertex_id").is_in(list(drop_vs))
                )
        for slice_data in self._slices.values():
            slice_data["vertices"].difference_update(drop_vs)

    # Attributes & weights

    def set_graph_attribute(self, key, value):
        """Set a graph-level attribute.

        Parameters
        --
        key : str
        value : Any

        """
        self.graph_attributes[key] = value

    def get_graph_attribute(self, key, default=None):
        """Get a graph-level attribute.

        Parameters
        --
        key : str
        default : Any, optional

        Returns
        ---
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
        --
        vertex_id : str
        key : str
        default : Any, optional

        Returns
        ---
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
        --
        vertex_id : str
        attribute : str or enum.Enum
            Column name or Enum with ``.value``.

        Returns
        ---
        Any or None
            Scalar value if present, else ``None``.

        See Also
        
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
        --
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
        --
        edge_id : str
        key : str
        default : Any, optional

        Returns
        ---
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
        --
        edge_id : str
        attribute : str or enum.Enum
            Column name or Enum with ``.value``.

        Returns
        ---
        Any or None
            Scalar value if present, else ``None``.

        See Also
        
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

    def set_slice_attrs(self, slice_id, **attrs):
        """Upsert pure slice attributes.

        Parameters
        --
        slice_id : str
        **attrs
            Key/value attributes. Structural keys are ignored.

        """
        clean = {k: v for k, v in attrs.items() if k not in self._slice_RESERVED}
        if clean:
            self.slice_attributes = self._upsert_row(self.slice_attributes, slice_id, clean)

    def get_slice_attr(self, slice_id, key, default=None):
        """Get a single slice attribute (scalar) or default if missing.

        Parameters
        --
        slice_id : str
        key : str
        default : Any, optional

        Returns
        ---
        Any

        """
        df = self.slice_attributes
        if key not in df.columns:
            return default
        rows = df.filter(pl.col("slice_id") == slice_id)
        if rows.height == 0:
            return default
        val = rows.select(pl.col(key)).to_series()[0]
        return default if val is None else val

    def set_edge_slice_attrs(self, slice_id, edge_id, **attrs):
        """Upsert per-slice attributes for a specific edge.

        Parameters
        --
        slice_id : str
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

            if isinstance(slice_id, str):
                slice_id = _sys.intern(slice_id)
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

        # Ensure edge_slice_attributes compares strings to strings (defensive against prior bad writes),
        # but only cast when actually needed (skip no-op with_columns).
        df = self.edge_slice_attributes
        if isinstance(df, pl.DataFrame) and df.height > 0:
            to_cast = []
            if "slice_id" in df.columns and df.schema["slice_id"] != pl.Utf8:
                to_cast.append(pl.col("slice_id").cast(pl.Utf8))
            if "edge_id" in df.columns and df.schema["edge_id"] != pl.Utf8:
                to_cast.append(pl.col("edge_id").cast(pl.Utf8))
            if to_cast:
                df = df.with_columns(*to_cast)
                self.edge_slice_attributes = df  # reassign only when changed

        # Upsert via central helper (keeps exact behavior, schema handling, and caching)
        self.edge_slice_attributes = self._upsert_row(
            self.edge_slice_attributes, (slice_id, edge_id), clean
        )

    def get_edge_slice_attr(self, slice_id, edge_id, key, default=None):
        """Get a per-slice attribute for an edge.

        Parameters
        --
        slice_id : str
        edge_id : str
        key : str
        default : Any, optional

        Returns
        ---
        Any

        """
        df = self.edge_slice_attributes
        if key not in df.columns:
            return default
        rows = df.filter((pl.col("slice_id") == slice_id) & (pl.col("edge_id") == edge_id))
        if rows.height == 0:
            return default
        val = rows.select(pl.col(key)).to_series()[0]
        return default if val is None else val

    def set_slice_edge_weight(self, slice_id, edge_id, weight):  # legacy weight helper
        """Set a legacy per-slice weight override for an edge.

        Parameters
        --
        slice_id : str
        edge_id : str
        weight : float

        Raises
        --
        KeyError
            If the slice or edge does not exist.

        See Also
        
        get_effective_edge_weight

        """
        if slice_id not in self._slices:
            raise KeyError(f"slice {slice_id} not found")
        if edge_id not in self.edge_to_idx:
            raise KeyError(f"Edge {edge_id} not found")
        self.slice_edge_weights[slice_id][edge_id] = float(weight)

    def get_effective_edge_weight(self, edge_id, slice=None):
        """Resolve the effective weight for an edge, optionally within a slice.

        Parameters
        --
        edge_id : str
        slice : str, optional
            If provided, return the slice override if present; otherwise global weight.

        Returns
        ---
        float
            Effective weight.

        """
        if slice is not None:
            df = self.edge_slice_attributes
            if (
                isinstance(df, pl.DataFrame)
                and df.height > 0
                and {"slice_id", "edge_id", "weight"} <= set(df.columns)
            ):
                rows = df.filter(
                    (pl.col("slice_id") == slice) & (pl.col("edge_id") == edge_id)
                ).select("weight")
                if rows.height > 0:
                    w = rows.to_series()[0]
                    if w is not None and not (isinstance(w, float) and math.isnan(w)):
                        return float(w)

            # fallback to legacy dict if present
            w2 = self.slice_edge_weights.get(slice, {}).get(edge_id, None)
            if w2 is not None:
                return float(w2)

        return float(self.edge_weights[edge_id])

    def audit_attributes(self):
        """Audit attribute tables for extra/missing rows and invalid edge-slice pairs.

        Returns
        ---
        dict
            {
            'extra_vertex_rows': list[str],
            'extra_edge_rows': list[str],
            'missing_vertex_rows': list[str],
            'missing_edge_rows': list[str],
            'invalid_edge_slice_rows': list[tuple[str, str]],
            }

        """
        vertex_ids = {eid for eid, t in self.entity_types.items() if t == "vertex"}
        edge_ids = set(self.edge_to_idx.keys())

        na = self.vertex_attributes
        ea = self.edge_attributes
        ela = self.edge_slice_attributes

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

        bad_edge_slice = []
        if (
            isinstance(ela, pl.DataFrame)
            and ela.height > 0
            and {"slice_id", "edge_id"} <= set(ela.columns)
        ):
            for lid, eid in ela.select(["slice_id", "edge_id"]).iter_rows():
                if lid not in self._slices or eid not in edge_ids:
                    bad_edge_slice.append((lid, eid))

        return {
            "extra_vertex_rows": extra_vertex_rows,
            "extra_edge_rows": extra_edge_rows,
            "missing_vertex_rows": missing_vertex_rows,
            "missing_edge_rows": missing_edge_rows,
            "invalid_edge_slice_rows": bad_edge_slice,
        }

    def _pl_dtype_for_value(self, v):
        """INTERNAL: Infer an appropriate Polars dtype for a Python value.

        Parameters
        --
        v : Any

        Returns
        ---
        polars.datatypes.DataType
            One of ``pl.Null``, ``pl.Boolean``, ``pl.Int64``, ``pl.Float64``,
            ``pl.Utf8``, ``pl.Binary``, ``pl.Object``, or ``pl.List(inner)``.

        Notes
        -
        - Enums are mapped to ``pl.Object`` (useful for categorical enums).
        - Lists/tuples infer inner dtype from the first element (defaults to ``Utf8``).

        """
        import enum

        

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
        --
        df : polars.DataFrame
            Existing attribute table.
        attrs : dict
            Incoming key/value pairs to upsert.

        Returns
        ---
        polars.DataFrame
            DataFrame with columns added/cast so inserts/updates won't hit ``Null`` dtypes.

        Notes
        -
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
        
        - ``vertex_attributes``           - key: ``["vertex_id"]``
        - ``edge_attributes``             - key: ``["edge_id"]``
        - ``slice_attributes``            - key: ``["slice_id"]``
        - ``edge_slice_attributes``       - key: ``["slice_id", "edge_id"]``
        """
        if not isinstance(attrs, dict) or not attrs:
            return df

        cols = set(df.columns)

        # Determine key columns + values
        if {"slice_id", "edge_id"} <= cols:
            if not (isinstance(idx, tuple) and len(idx) == 2):
                raise ValueError("idx must be a (slice_id, edge_id) tuple")
            key_cols = ("slice_id", "edge_id")
            key_vals = {"slice_id": idx[0], "edge_id": idx[1]}
            cache_name = "_edge_slice_attr_keys"  # set of (slice_id, edge_id)
            df_id_name = "_edge_slice_attr_df_id"
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
        elif "slice_id" in cols:
            key_cols = ("slice_id",)
            key_vals = {"slice_id": idx}
            cache_name = "_slice_attr_ids"  # set of slice_id
            df_id_name = "_slice_attr_df_id"
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
                    "edge_id" in cols and "slice_id" in cols and key_cols == ("slice_id", "edge_id")
                ):
                    if df.height:
                        key_cache = set(
                            zip(
                                df.get_column("slice_id").to_list(),
                                df.get_column("edge_id").to_list(),
                            )
                        )
                    else:
                        key_cache = set()
                elif "edge_id" in cols and key_cols == ("edge_id",):
                    series = df.get_column("edge_id") if df.height else pl.Series([])
                    key_cache = set(series.to_list()) if df.height else set()
                elif "slice_id" in cols and key_cols == ("slice_id",):
                    series = df.get_column("slice_id") if df.height else pl.Series([])
                    key_cache = set(series.to_list()) if df.height else set()
                else:
                    key_cache = set()
                setattr(self, cache_name, key_cache)
                setattr(self, df_id_name, id(df))
            # Decide existence from cache
            cache_key = (
                key_vals[key_cols[0]]
                if len(key_cols) == 1
                else (key_vals["slice_id"], key_vals["edge_id"])
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
                # do nothing - previous signs remain (default)
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
        --
        edge : int | str
            Edge index (int) or edge id (str).

        Returns
        ---
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
        --
        vertex : str
            Vertex id.

        Returns
        ---
        dict
            Attribute dictionary for that vertex. {} if not found.

        """
        df = self.vertex_attributes
        try:
            

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
        --
        indexes : Iterable[int] | None, optional
            A list or iterable of edge indices to retrieve attributes for.
            - If `None` (default), attributes for **all** edges are returned.
            - If provided, only those edges will be included in the output.

        Returns
        ---
        dict[str, dict]
            A dictionary mapping `edge_id` - `attribute_dict`, where:
            - `edge_id` is the unique string identifier of the edge.
            - `attribute_dict` is a dictionary of attribute names and values.

        Notes
        -
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
        --
        vertices : Iterable[str] | None, optional
            A list or iterable of vertex IDs to retrieve attributes for.
            - If `None` (default), attributes for **all** verices are returned.
            - If provided, only those verices will be included in the output.

        Returns
        ---
        dict[str, dict]
            A dictionary mapping `vertex_id` - `attribute_dict`, where:
            - `vertex_id` is the unique string identifier of the vertex.
            - `attribute_dict` is a dictionary of attribute names and values.

        Notes
        -
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
        --
        key : str
            Attribute column name to extract from `self.edge_attributes`.
        default : Any, optional
            Default value to use if the column does not exist or if an edge
            does not have a value. Defaults to `None`.

        Returns
        ---
        dict[str, Any]
            A dictionary mapping `edge_id` - attribute value.

        Notes
        -
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
        --
        key : str
            Attribute column name to filter on.
        value : Any
            Value to match.

        Returns
        ---
        list[str]
            A list of edge IDs where the attribute `key` equals `value`.

        Notes
        -
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
        ---
        dict
            A dictionary of global metadata describing the graph as a whole.
            Typical keys might include:
            - `"name"` : Graph name or label.
            - `"directed"` : Boolean indicating directedness.
            - `"slices"` : List of slices present in the graph.
            - `"created_at"` : Timestamp of graph creation.

        Notes
        -
        - Returns a **shallow copy** to prevent external mutation of internal state.
        - Graph-level attributes are meant to store metadata not tied to individual
        verices or edges (e.g., versioning info, provenance, global labels).

        """
        return dict(self.graph_attributes)

    def set_edge_slice_attrs_bulk(self, slice_id, items):
        """items: iterable of (edge_id, attrs_dict) or dict{edge_id: attrs_dict}
        Upserts rows in edge_slice_attributes for one slice in bulk.
        """
        

        # normalize
        rows = []
        if isinstance(items, dict):
            it = items.items()
        else:
            it = items
        for eid, attrs in it:
            if not isinstance(attrs, dict) or not attrs:
                continue
            r = {"slice_id": slice_id, "edge_id": eid}
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
        df = self.edge_slice_attributes
        add_df = pl.DataFrame(rows)

        # ensure required key cols exist/correct dtype on existing df
        if not isinstance(df, pl.DataFrame) or df.is_empty():
            # create from scratch with canonical dtypes
            self.edge_slice_attributes = add_df
            # legacy mirror
            if "weight" in add_df.columns:
                self.slice_edge_weights.setdefault(slice_id, {})
                for r in add_df.iter_rows(named=True):
                    w = r.get("weight")
                    if w is not None:
                        self.slice_edge_weights[slice_id][r["edge_id"]] = float(w)
            return

        # schema alignment using _ensure_attr_columns + Utf8 upcast rule
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

        # drop existing keys for (slice_id, edge_id) we are about to write; then vstack new rows
        mask_keep = ~(
            (pl.col("slice_id") == slice_id) & pl.col("edge_id").is_in(add_df.get_column("edge_id"))
        )
        df = df.filter(mask_keep)
        df = df.vstack(add_df)
        self.edge_slice_attributes = df

        # legacy mirror
        if "weight" in add_df.columns:
            self.slice_edge_weights.setdefault(slice_id, {})
            for r in add_df.iter_rows(named=True):
                w = r.get("weight")
                if w is not None:
                    self.slice_edge_weights[slice_id][r["edge_id"]] = float(w)

    # Basic queries & metrics

    def get_vertex(self, index: int) -> str:
        """Return the vertex ID corresponding to a given internal index.

        Parameters
        --
        index : int
            The internal vertex index.

        Returns
        ---
        str
            The vertex ID.

        """
        return self.idx_to_entity[index]

    def get_edge(self, index: int):
        """Return edge endpoints in a canonical form.

        Parameters
        --
        index : int
            Internal edge index.

        Returns
        ---
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
        --
        vertex_id : str
            vertex identifier.

        Returns
        ---
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
        --
        edge_id : str

        Returns
        ---
        bool

        """
        return bool(self.edge_directed.get(edge_id, self.directed))

    def has_edge(
        self,
        source: Optional[str] = None,
        target: Optional[str] = None,
        edge_id: Optional[str] = None,
    ) -> Union[bool, Tuple[bool, List[str]]]:
        """
        Edge existence check with three modes.

        Modes
        -----
        1) edge_id only:
            has_edge(edge_id="e1")
            -> bool    (True if e1 exists anywhere)

        2) source + target only:
            has_edge(source="u", target="v")
            -> (bool, [edge_ids...])
               bool = True if at least one edge u->v exists

        3) source + target + edge_id:
            has_edge(source="u", target="v", edge_id="e1")
            -> bool    (True only if e1 exists AND is u->v)

        Any other combination is invalid and raises ValueError.
        """

        # ---- Mode 1: edge_id only ----
        if edge_id is not None and source is None and target is None:
            return edge_id in self.edge_definitions

        # ---- Mode 2: source + target only ----
        if edge_id is None and source is not None and target is not None:
            eids: List[str] = []
            for eid, (src, tgt, _) in self.edge_definitions.items():
                if src == source and tgt == target:
                    eids.append(eid)
            return (len(eids) > 0, eids)

        # ---- Mode 3: edge_id + source + target ----
        if edge_id is not None and source is not None and target is not None:
            data = self.edge_definitions.get(edge_id)
            if data is None:
                return False
            src, tgt, _ = data
            return src == source and tgt == target

        # ---- Anything else is ambiguous / invalid ----
        raise ValueError(
            "Invalid argument combination: use either "
            "(edge_id), (source,target), or (source,target,edge_id)."
        )

    def has_vertex(self, vertex_id: str) -> bool:
        """Test for the existence of a vertex.

        Parameters
        --
        vertex_id : str

        Returns
        ---
        bool

        """
        return vertex_id in self.entity_to_idx and self.entity_types.get(vertex_id) == "vertex"

    def get_edge_ids(self, source, target):
        """List all edge IDs between two endpoints.

        Parameters
        --
        source : str
        target : str

        Returns
        ---
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
        --
        entity_id : str

        Returns
        ---
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
        ---
        list[str]

        """
        return [eid for eid, etype in self.entity_types.items() if etype == "vertex"]

    def edges(self):
        """Get all edge IDs.

        Returns
        ---
        list[str]

        """
        return list(self.edge_to_idx.keys())

    def edge_list(self):
        """Materialize (source, target, edge_id, weight) for binary/vertex-edge edges.

        Returns
        ---
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
        ---
        list[str]

        """
        default_dir = True if self.directed is None else self.directed
        return [eid for eid in self.edge_to_idx.keys() if self.edge_directed.get(eid, default_dir)]

    def get_undirected_edges(self):
        """List IDs of undirected edges.

        Returns
        ---
        list[str]

        """
        default_dir = True if self.directed is None else self.directed
        return [
            eid for eid in self.edge_to_idx.keys() if not self.edge_directed.get(eid, default_dir)
        ]

    def number_of_vertices(self):
        """Count vertices (excluding edge-entities).

        Returns
        ---
        int

        """
        return len([e for e in self.entity_types.values() if e == "vertex"])

    def number_of_edges(self):
        """Count edges (columns in the incidence matrix).

        Returns
        ---
        int

        """
        return self._num_edges

    def global_entity_count(self):
        """Count unique entities present across all slices (union of memberships).

        Returns
        ---
        int

        """
        all_vertices = set()
        for slice_data in self._slices.values():
            all_vertices.update(slice_data["vertices"])
        return len(all_vertices)

    def global_edge_count(self):
        """Count unique edges present across all slices (union of memberships).

        Returns
        ---
        int

        """
        all_edges = set()
        for slice_data in self._slices.values():
            all_edges.update(slice_data["edges"])
        return len(all_edges)

    def in_edges(self, vertices):
        """Iterate over all edges that are **incoming** to one or more vertices.

        Parameters
        --
        vertices : str | Iterable[str]
            A single vertex ID or an iterable of vertex IDs. All edges whose
            **target set** intersects with this set will be yielded.

        Yields
        --
        tuple[int, tuple[frozenset, frozenset]]
            Tuples of the form `(edge_index, (S, T))`, where:
            - `edge_index` : int — internal integer index of the edge.
            - `S` : frozenset[str] — set of source/head verices.
            - `T` : frozenset[str] — set of target/tail verices.

        Behavior
        
        - **Directed binary edges**: returned if any vertex is in the target (`T`).
        - **Directed hyperedges**: returned if any vertex is in the tail set.
        - **Undirected edges/hyperedges**: returned if any vertex is in
        the edge's member set (`S ∪ T`).

        Notes
        -
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
        --
        vertices : str | Iterable[str]
            A single vertex ID or an iterable of vertex IDs. All edges whose
            **source set** intersects with this set will be yielded.

        Yields
        --
        tuple[int, tuple[frozenset, frozenset]]
            Tuples of the form `(edge_index, (S, T))`, where:
            - `edge_index` : int — internal integer index of the edge.
            - `S` : frozenset[str] — set of source/head verices.
            - `T` : frozenset[str] — set of target/tail verices.

        Behavior
        
        - **Directed binary edges**: returned if any vertex is in the source (`S`).
        - **Directed hyperedges**: returned if any vertex is in the head set.
        - **Undirected edges/hyperedges**: returned if any vertex is in
        the edge's member set (`S ∪ T`).

        Notes
        -
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

    def get_or_create_vertex_by_attrs(self, slice=None, **attrs) -> str:
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
        self.add_vertex(vid, slice=slice, **attrs)

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
        ---
        tuple
            Tuple of all vertex IDs in the graph.

        """
        return tuple(self.vertices())

    @property
    def E(self):
        """All edges as a tuple.

        Returns
        ---
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
        slice=None,
        include_directed=True,
        include_weight=True,
        resolved_weight=True,
        copy=True,
    ):
        """Build a Polars DF [DataFrame] view of edges with optional slice join.
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

        # join slice-specific attributes once, then compute resolved weight vectorized
        if (
            slice is not None
            and isinstance(self.edge_slice_attributes, pl.DataFrame)
            and self.edge_slice_attributes.height > 0
        ):
            slice_slice = self.edge_slice_attributes.filter(pl.col("slice_id") == slice).drop(
                "slice_id"
            )
            if slice_slice.height > 0:
                # prefix non-key columns -> slice_*
                rename_map = {c: f"slice_{c}" for c in slice_slice.columns if c not in {"edge_id"}}
                if rename_map:
                    slice_slice = slice_slice.rename(rename_map)
                out = out.join(slice_slice, on="edge_id", how="left")

        # add effective_weight without per-edge function calls
        if resolved_weight:
            gw_col = "global_weight" if include_weight else "_gw_tmp"
            lw_col = "slice_weight" if ("slice_weight" in out.columns) else None
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
        --
        copy : bool, optional
            Return a cloned DF.

        Returns
        ---
        polars.DataFrame
            Columns: ``vertex_id`` plus pure attributes (may be empty).

        """
        df = self.vertex_attributes
        if df.height == 0:
            return pl.DataFrame(schema={"vertex_id": pl.Utf8})
        return df.clone() if copy else df

    def slices_view(self, copy=True):
        """Read-only slice attribute table.

        Parameters
        --
        copy : bool, optional
            Return a cloned DF.

        Returns
        ---
        polars.DataFrame
            Columns: ``slice_id`` plus pure attributes (may be empty).

        """
        df = self.slice_attributes
        if df.height == 0:
            return pl.DataFrame(schema={"slice_id": pl.Utf8})
        return df.clone() if copy else df

    # slice set-ops & cross-slice analytics

    def get_slice_vertices(self, slice_id):
        """Vertices in a slice.

        Parameters
        --
        slice_id : str

        Returns
        ---
        set[str]

        """
        return self._slices[slice_id]["vertices"].copy()

    def get_slice_edges(self, slice_id):
        """Edges in a slice.

        Parameters
        --
        slice_id : str

        Returns
        ---
        set[str]

        """
        return self._slices[slice_id]["edges"].copy()

    def slice_union(self, slice_ids):
        """Union of multiple slices.

        Parameters
        --
        slice_ids : Iterable[str]

        Returns
        ---
        dict
            ``{"vertices": set[str], "edges": set[str]}``

        """
        if not slice_ids:
            return {"vertices": set(), "edges": set()}

        union_vertices = set()
        union_edges = set()

        for slice_id in slice_ids:
            if slice_id in self._slices:
                union_vertices.update(self._slices[slice_id]["vertices"])
                union_edges.update(self._slices[slice_id]["edges"])

        return {"vertices": union_vertices, "edges": union_edges}

    def slice_intersection(self, slice_ids):
        """Intersection of multiple slices.

        Parameters
        --
        slice_ids : Iterable[str]

        Returns
        ---
        dict
            ``{"vertices": set[str], "edges": set[str]}``

        """
        if not slice_ids:
            return {"vertices": set(), "edges": set()}

        if len(slice_ids) == 1:
            slice_id = slice_ids[0]
            return {
                "vertices": self._slices[slice_id]["vertices"].copy(),
                "edges": self._slices[slice_id]["edges"].copy(),
            }

        # Start with first slice
        common_vertices = self._slices[slice_ids[0]]["vertices"].copy()
        common_edges = self._slices[slice_ids[0]]["edges"].copy()

        # Intersect with remaining slices
        for slice_id in slice_ids[1:]:
            if slice_id in self._slices:
                common_vertices &= self._slices[slice_id]["vertices"]
                common_edges &= self._slices[slice_id]["edges"]
            else:
                # slice doesn't exist, intersection is empty
                return {"vertices": set(), "edges": set()}

        return {"vertices": common_vertices, "edges": common_edges}

    def slice_difference(self, slice1_id, slice2_id):
        """Set difference: elements in ``slice1_id`` not in ``slice2_id``.

        Parameters
        --
        slice1_id : str
        slice2_id : str

        Returns
        ---
        dict
            ``{"vertices": set[str], "edges": set[str]}``

        Raises
        --
        KeyError
            If either slice is missing.

        """
        if slice1_id not in self._slices or slice2_id not in self._slices:
            raise KeyError("One or both slices not found")

        slice1 = self._slices[slice1_id]
        slice2 = self._slices[slice2_id]

        return {
            "vertices": slice1["vertices"] - slice2["vertices"],
            "edges": slice1["edges"] - slice2["edges"],
        }

    def create_slice_from_operation(self, result_slice_id, operation_result, **attributes):
        """Create a new slice from the result of a set operation.

        Parameters
        --
        result_slice_id : str
        operation_result : dict
            Output of ``slice_union``/``slice_intersection``/``slice_difference``.
        **attributes
            Pure slice attributes.

        Returns
        ---
        str
            The created slice ID.

        Raises
        --
        ValueError
            If the target slice already exists.

        """
        if result_slice_id in self._slices:
            raise ValueError(f"slice {result_slice_id} already exists")

        self._slices[result_slice_id] = {
            "vertices": operation_result["vertices"].copy(),
            "edges": operation_result["edges"].copy(),
            "attributes": attributes,
        }

        return result_slice_id

    def edge_presence_across_slices(
        self,
        edge_id: str | None = None,
        source: str | None = None,
        target: str | None = None,
        *,
        include_default: bool = False,
        undirected_match: bool | None = None,
    ):
        """Locate where an edge exists across slices.

        Parameters
        --
        edge_id : str, optional
            If provided, match by ID (any kind: binary/vertex-edge/hyper).
        source : str, optional
            When used with ``target``, match only binary/vertex-edge edges by endpoints.
        target : str, optional
        include_default : bool, optional
            Include the internal default slice in the search.
        undirected_match : bool, optional
            When endpoint matching, allow undirected symmetric matches.

        Returns
        ---
        list[str] or dict[str, list[str]]
            If ``edge_id`` given: list of slice IDs.
            Else: ``{slice_id: [edge_id, ...]}``.

        Raises
        --
        ValueError
            If both modes (ID and endpoints) are provided or neither is valid.

        """
        has_id = edge_id is not None
        has_pair = (source is not None) and (target is not None)
        if has_id == has_pair:
            raise ValueError("Provide either edge_id OR (source and target), but not both.")

        slices_view = self.get_slices_dict(include_default=include_default)

        if has_id:
            return [lid for lid, ldata in slices_view.items() if edge_id in ldata["edges"]]

        if undirected_match is None:
            undirected_match = False

        out: dict[str, list[str]] = {}
        for lid, ldata in slices_view.items():
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

    def hyperedge_presence_across_slices(
        self,
        *,
        members=None,
        head=None,
        tail=None,
        include_default: bool = False,
    ):
        """Locate slices containing a hyperedge with exactly these sets.

        Parameters
        --
        members : Iterable[str], optional
            Undirected member set (exact match).
        head : Iterable[str], optional
            Directed head set (exact match).
        tail : Iterable[str], optional
            Directed tail set (exact match).
        include_default : bool, optional

        Returns
        ---
        dict[str, list[str]]
            ``{slice_id: [edge_id, ...]}``.

        Raises
        --
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

        slices_view = self.get_slices_dict(include_default=include_default)
        out: dict[str, list[str]] = {}

        for lid, ldata in slices_view.items():
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

    def vertex_presence_across_slices(self, vertex_id, include_default: bool = False):
        """List slices containing a specific vertex.

        Parameters
        --
        vertex_id : str
        include_default : bool, optional

        Returns
        ---
        list[str]

        """
        slices_with_vertex = []
        for slice_id, slice_data in self.get_slices_dict(include_default=include_default).items():
            if vertex_id in slice_data["vertices"]:
                slices_with_vertex.append(slice_id)
        return slices_with_vertex

    def conserved_edges(self, min_slices=2, include_default=False):
        """Edges present in at least ``min_slices`` slices.

        Parameters
        --
        min_slices : int, optional
        include_default : bool, optional

        Returns
        ---
        dict[str, int]
            ``{edge_id: count}``.

        """
        slices_to_check = self.get_slices_dict(
            include_default=include_default
        )  # hides 'default' by default
        edge_counts = {}
        for _, slice_data in slices_to_check.items():
            for eid in slice_data["edges"]:
                edge_counts[eid] = edge_counts.get(eid, 0) + 1
        return {eid: c for eid, c in edge_counts.items() if c >= min_slices}

    def slice_specific_edges(self, slice_id):
        """Edges that appear **only** in the specified slice.

        Parameters
        --
        slice_id : str

        Returns
        ---
        set[str]

        Raises
        --
        KeyError
            If the slice does not exist.

        """
        if slice_id not in self._slices:
            raise KeyError(f"slice {slice_id} not found")

        target_edges = self._slices[slice_id]["edges"]
        specific_edges = set()

        for edge_id in target_edges:
            # Count how many slices contain this edge
            count = sum(1 for slice_data in self._slices.values() if edge_id in slice_data["edges"])
            if count == 1:  # Only in target slice
                specific_edges.add(edge_id)

        return specific_edges

    def temporal_dynamics(self, ordered_slices, metric="edge_change"):
        """Compute changes between consecutive slices in a temporal sequence.

        Parameters
        --
        ordered_slices : list[str]
            slice IDs in chronological order.
        metric : {'edge_change', 'vertex_change'}, optional

        Returns
        ---
        list[dict[str, int]]
            Per-step dictionaries with keys: ``'added'``, ``'removed'``, ``'net_change'``.

        Raises
        --
        ValueError
            If fewer than two slices are provided.
        KeyError
            If a referenced slice does not exist.

        """
        if len(ordered_slices) < 2:
            raise ValueError("Need at least 2 slices for temporal analysis")

        changes = []

        for i in range(len(ordered_slices) - 1):
            current_id = ordered_slices[i]
            next_id = ordered_slices[i + 1]

            if current_id not in self._slices or next_id not in self._slices:
                raise KeyError("One or more slices not found")

            current_data = self._slices[current_id]
            next_data = self._slices[next_id]

            if metric == "edge_change":
                added = len(next_data["edges"] - current_data["edges"])
                removed = len(current_data["edges"] - next_data["edges"])
                changes.append({"added": added, "removed": removed, "net_change": added - removed})

            elif metric == "vertex_change":
                added = len(next_data["vertices"] - current_data["vertices"])
                removed = len(current_data["vertices"] - next_data["vertices"])
                changes.append({"added": added, "removed": removed, "net_change": added - removed})

        return changes

    def create_aggregated_slice(
        self, source_slice_ids, target_slice_id, method="union", weight_func=None, **attributes
    ):
        """Create a new slice by aggregating multiple source slices.

        Parameters
        --
        source_slice_ids : list[str]
        target_slice_id : str
        method : {'union', 'intersection'}, optional
        weight_func : callable, optional
            Reserved for future weight merging logic (currently unused).
        **attributes
            Pure slice attributes.

        Returns
        ---
        str
            The created slice ID.

        Raises
        --
        ValueError
            For unknown methods or missing source slices, or if target exists.

        """
        if not source_slice_ids:
            raise ValueError("Must specify at least one source slice")

        if target_slice_id in self._slices:
            raise ValueError(f"Target slice {target_slice_id} already exists")

        if method == "union":
            result = self.slice_union(source_slice_ids)
        elif method == "intersection":
            result = self.slice_intersection(source_slice_ids)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        return self.create_slice_from_operation(target_slice_id, result, **attributes)

    def slice_statistics(self, include_default: bool = False):
        """Basic per-slice statistics.

        Parameters
        --
        include_default : bool, optional

        Returns
        ---
        dict[str, dict]
            ``{slice_id: {'vertices': int, 'edges': int, 'attributes': dict}}``.

        """
        stats = {}
        for slice_id, slice_data in self.get_slices_dict(include_default=include_default).items():
            stats[slice_id] = {
                "vertices": len(slice_data["vertices"]),
                "edges": len(slice_data["edges"]),
                "attributes": slice_data["attributes"],
            }
        return stats

    # Traversal (neighbors)

    def neighbors(self, entity_id):
        """Neighbors of an entity (vertex or edge-entity).

        Parameters
        --
        entity_id : str

        Returns
        ---
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
        --
        vertex_id : str

        Returns
        ---
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
        --
        vertex_id : str

        Returns
        ---
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
        --
        vertex_id : str

        Returns
        ---
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
        --
        vertex_id : str

        Returns
        ---
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
        --
        edges : Iterable[str] | Iterable[int]
            Edge identifiers (strings) or edge indices (integers) to retain
            in the subgraph.

        Returns
        ---
        Graph
            A new `Graph` instance containing only the selected edges and the
            vertices incident to them.

        Behavior
        
        - Copies the current graph and deletes all edges **not** in the provided set.
        - Optionally, you can prune orphaned vertices (i.e., vertices not incident
        to any remaining edge) — this is generally recommended for consistency.

        Notes
        -
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
        g.add_vertices_bulk(v_rows, slice=g._default_slice)

        # edges
        if bin_payload:
            g.add_edges_bulk(bin_payload, slice=g._default_slice)
        if hyper_payload:
            g.add_hyperedges_bulk(hyper_payload, slice=g._default_slice)

        # copy slice memberships for retained edges & incident vertices
        for lid, meta in self._slices.items():
            g.add_slice(lid, **meta["attributes"])
            kept_edges = set(meta["edges"]) & E
            if kept_edges:
                g.add_edges_to_slice_bulk(lid, kept_edges)

        return g

    def subgraph(self, vertices) -> "Graph":
        """Create a vertex-induced subgraph.

        Parameters
        --
        vertices : Iterable[str]
            A set or list of vertex identifiers to keep in the subgraph.

        Returns
        ---
        Graph
            A new `Graph` containing only the specified vertices and any edges
            for which **all** endpoints are within this set.

        Behavior
        
        - Copies the current graph and removes edges with any endpoint outside
        the provided vertex set.
        - Removes all vertices not listed in `vertices`.

        Notes
        -
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
        g.add_vertices_bulk(v_rows, slice=g._default_slice)
        if bin_payload:
            g.add_edges_bulk(bin_payload, slice=g._default_slice)
        if hyper_payload:
            g.add_hyperedges_bulk(hyper_payload, slice=g._default_slice)

        # slice memberships restricted to V
        for lid, meta in self._slices.items():
            g.add_slice(lid, **meta["attributes"])
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
                g.add_edges_to_slice_bulk(lid, keep)

        return g

    def extract_subgraph(self, vertices=None, edges=None) -> "Graph":
        """Create a subgraph based on a combination of vertex and/or edge filters.

        Parameters
        --
        vertices : Iterable[str] | None, optional
            A set of vertex IDs to include. If provided, behaves like `subgraph()`.
            If `None`, no vertex filtering is applied.
        edges : Iterable[str] | Iterable[int] | None, optional
            A set of edge IDs or indices to include. If provided, behaves like
            `edge_subgraph()`. If `None`, no edge filtering is applied.

        Returns
        ---
        Graph
            A new `Graph` filtered according to the provided vertex and/or edge
            sets.

        Behavior
        
        - If both `vertices` and `edges` are provided, the resulting subgraph is
        the intersection of the two filters.
        - If only `vertices` is provided, equivalent to `subgraph(vertices)`.
        - If only `edges` is provided, equivalent to `edge_subgraph(edges)`.
        - If neither is provided, a full copy of the graph is returned.

        Notes
        -
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
        ---
        Graph
            A new `Graph` instance with reversed directionality where applicable.

        Behavior
        
        - **Binary edges:** direction is flipped by swapping source and target.
        - **Directed hyperedges:** `head` and `tail` sets are swapped.
        - **Undirected edges/hyperedges:** unaffected.
        - Edge attributes and metadata are preserved.

        Notes
        -
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

    def subgraph_from_slice(self, slice_id, *, resolve_slice_weights=True):
        if slice_id not in self._slices:
            raise KeyError(f"slice {slice_id} not found")

        

        slice_meta = self._slices[slice_id]
        V = set(slice_meta["vertices"])
        E = set(slice_meta["edges"])

        g = Graph(directed=self.directed, n=len(V), e=len(E))
        g.add_slice(slice_id, **slice_meta["attributes"])
        g.set_active_slice(slice_id)

        # vertices with attrs (edge-entities share same table)
        v_rows = [
            {"vertex_id": v, **(self._row_attrs(self.vertex_attributes, "vertex_id", v) or {})}
            for v in V
        ]
        g.add_vertices_bulk(v_rows, slice=slice_id)

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
        if resolve_slice_weights:
            df = self.edge_slice_attributes
            if (
                isinstance(df, pl.DataFrame)
                and df.height
                and {"slice_id", "edge_id", "weight"}.issubset(df.columns)
            ):
                for r in df.filter(
                    (pl.col("slice_id") == slice_id) & (pl.col("edge_id").is_in(list(E)))
                ).iter_rows(named=True):
                    if r.get("weight") is not None:
                        eff_w[r["edge_id"]] = float(r["weight"])

        # partition edges
        bin_payload, hyper_payload = [], []
        for eid in E:
            w = (
                eff_w.get(eid, self.edge_weights.get(eid, 1.0))
                if resolve_slice_weights
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
            g.add_edges_bulk(bin_payload, slice=slice_id)
        if hyper_payload:
            g.add_hyperedges_bulk(hyper_payload, slice=slice_id)

        return g

    def _row_attrs(self, df, key_col: str, key):
        """INTERNAL: return a dict of attributes for the row in `df` where `key_col == key`,
        excluding the key column itself. If not found or df empty, return {}.
        Caches per (id(df), key_col) for speed; cache auto-refreshes when the df object changes.
        """

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
            # Latest write should win if duplicates exist (matches upsert semantics)
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
        """Deep copy the entire graph, including slices, edges, hyperedges, and attributes.
        (Behavior preserved; uses preallocation + vectorized attr extraction.)
        """
        

        # Preallocate with current sizes
        new_graph = Graph(directed=self.directed, n=self._num_entities, e=self._num_edges)

        # Copy slices & their pure attributes 
        for lid, meta in self._slices.items():
            if lid != new_graph._default_slice:
                new_graph.add_slice(lid, **meta["attributes"])
            else:
                # default slice exists; mirror its attributes too
                if meta["attributes"]:
                    new_graph.set_slice_attrs(lid, **meta["attributes"])

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
                # entity_types[...] == "edge" - edge-entity
                edge_entity_rows.append({"edge_entity_id": ent_id, **vmap.get(ent_id, {})})

        # Add entities with correct type APIs (bulk)
        if vertex_rows:
            new_graph.add_vertices_bulk(vertex_rows, slice=new_graph._default_slice)
        if edge_entity_rows:
            # attributes for edge-entities live in the same vertex_attributes table
            new_graph.add_edge_entities_bulk(edge_entity_rows, slice=new_graph._default_slice)

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
            new_graph.add_edges_bulk(bin_payload, slice=new_graph._default_slice)

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
            new_graph.add_hyperedges_bulk(hyper_payload, slice=new_graph._default_slice)

        # Copy slice memberships
        for lid, meta in self._slices.items():
            if lid not in new_graph._slices:
                new_graph.add_slice(lid)
            new_graph._slices[lid]["vertices"] = set(meta["vertices"])
            new_graph._slices[lid]["edges"] = set(meta["edges"])

        # Copy edge-slice attributes + legacy weight dict
        if isinstance(self.edge_slice_attributes, pl.DataFrame):
            new_graph.edge_slice_attributes = self.edge_slice_attributes.clone()
        else:
            new_graph.edge_slice_attributes = self.edge_slice_attributes

        new_graph.slice_edge_weights = defaultdict(
            dict, {lid: dict(m) for lid, m in self.slice_edge_weights.items()}
        )

        return new_graph

    def memory_usage(self):
        """Approximate total memory usage in bytes.

        Returns
        ---
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
        --
        values : bool, optional (default=False)
            - If `False`, returns edge indices incident to each vertex.
            - If `True`, returns the **matrix values** (usually weights or 1/0) for
            each incident edge instead of the indices.

        Returns
        ---
        dict[str, list]
            A mapping from `vertex_id` - list of incident edges (indices or values),
            where:
            - Keys are vertex IDs.
            - Values are lists of edge indices (if `values=False`) or numeric values
            from the incidence matrix (if `values=True`).

        Notes
        -
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
        --
        values : bool, optional (default=False)
            If `True`, include the numeric values stored in the matrix
            (e.g., weights or signed incidence values). If `False`, convert the
            matrix to a binary mask (1 if incident, 0 if not).
        sparse : bool, optional (default=False)
            - If `True`, return the underlying sparse matrix (CSR).
            - If `False`, return a dense NumPy ndarray.

        Returns
        ---
        scipy.sparse.csr_matrix | numpy.ndarray
            The vertex–edge incidence matrix `M`:
            - Rows correspond to vertices.
            - Columns correspond to edges.
            - `M[i, j]` ≠ 0 indicates that vertex `i` is incident to edge `j`.

        Notes
        -
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
        ---
        int
            A hash value that uniquely (within high probability) identifies the graph
            based on its topology and attributes.

        Behavior
        
        - Includes the set of verices, edges, and directedness in the hash.
        - Includes graph-level attributes (if any) to capture metadata changes.
        - Does **not** depend on memory addresses or internal object IDs, so the same
        graph serialized/deserialized or reconstructed with identical structure
        will produce the same hash.

        Notes
        -
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
            "set_slice_attrs",
            "set_edge_slice_attrs",
            "register_slice",
            "unregister_slice",
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
        --
        as_df : bool, default False
            If True, return a Polars DF [DataFrame]; otherwise return a list of dicts.

        Returns
        ---
        list[dict] or polars.DataFrame
            Each event includes: 'version', 'ts_utc' (UTC [Coordinated Universal Time]
            ISO-8601 [International Organization for Standardization]), 'mono_ns'
            (monotonic nanoseconds since logger start), 'op', call snapshot fields,
            and 'result' when captured.

        Notes
        -
        Ordering is guaranteed by 'version' and 'mono_ns'. The log is in-memory until exported.

        """
        return pl.DataFrame(self._history) if as_df else list(self._history)

    def export_history(self, path: str):
        """Write the mutation history to disk.

        Parameters
        --
        path : str
            Output path. Supported extensions: '.parquet', '.ndjson' (a.k.a. '.jsonl'),
            '.json', '.csv'. Unknown extensions default to Parquet by appending '.parquet'.

        Returns
        ---
        int
            Number of events written. Returns 0 if the history is empty.

        Raises
        --
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
        --
        flag : bool, default True
            When True, start/continue logging; when False, pause logging.

        Returns
        ---
        None

        """
        self._history_enabled = bool(flag)

    def clear_history(self):
        """Clear the in-memory mutation log.

        Returns
        ---
        None

        Notes
        -
        This does not delete any files previously exported.

        """
        self._history.clear()

    def mark(self, label: str):
        """Insert a manual marker into the mutation history.

        Parameters
        --
        label : str
            Human-readable tag for the marker event.

        Returns
        ---
        None

        Notes
        -
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
        - Auto label-ID mapping for vertex arguments (kwargs + positionals).
        - NEW: _nx_simple to collapse Multi* - simple Graph/DiGraph for algos that need it.
        - NEW: _nx_edge_aggs to control parallel-edge aggregation (e.g., {"capacity":"sum"}).
        """

        # -- init ---
        def __init__(self, owner: "Graph"):
            self._G = owner
            self._cache = {}  # key -> {"nxG": nx.Graph, "version": int}
            self.cache_enabled = True

        #  public API 
        def clear(self):
            """Drop all cached NX graphs."""
            self._cache.clear()

        def peek_vertices(self, k: int = 10):
            """Debug helper: return up to k vertex IDs visible to NX."""
            nxG = self._get_or_make_nx(
                directed=True,
                hyperedge_mode="expand",
                slice=None,
                slices=None,
                needed_attrs=set(),
                simple=False,
                edge_aggs=None,
            )
            out = []
            it = iter(nxG.vertices())
            for _ in range(max(0, int(k))):
                try:
                    out.append(next(it))
                except StopIteration:
                    break
            return out

        # - dynamic dispatch -
        # Public helper: obtain the cached/backend NX graph directly
        # Usage in tests: nxG = G.nx.backend(directed=False, simple=True)
        def backend(
            self,
            *,
            directed: bool = True,
            hyperedge_mode: str = "expand",
            slice=None,
            slices=None,
            needed_attrs=None,
            simple: bool = False,
            edge_aggs: dict | None = None,
        ):
            """Return the underlying NetworkX graph built with the same lazy/cached
            machinery as normal calls.

            Args:
              directed: build DiGraph (True) or Graph (False) view
              hyperedge_mode: "skip" | "expand"
              slice/slices: slice selection if Graph is multisliceed
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
                slice=slice,
                slices=slices,
                needed_attrs=needed_attrs,
                simple=simple,
                edge_aggs=edge_aggs,
            )

        def __getattr__(self, name: str):
            nx_callable = self._resolve_nx_callable(name)

            def wrapper(*args, **kwargs):
                

                import networkx as _nx

                # Proxy-only knobs (consumed here; not forwarded to NX)
                directed = bool(kwargs.pop("_nx_directed", getattr(self, "default_directed", True)))
                hyperedge_mode = kwargs.pop(
                    "_nx_hyperedge", getattr(self, "default_hyperedge_mode", "expand")
                )  # "skip" | "expand"
                slice = kwargs.pop("_nx_slice", None)
                slices = kwargs.pop("_nx_slices", None)
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
                        slice=slice,
                        slices=slices,
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

                # Bind to NX signature so we can coerce vertex args (no defaults!)
                bound = None
                try:
                    sig = inspect.signature(nx_callable)
                    bound = sig.bind_partial(*args, **kwargs)
                except Exception:
                    pass

                # Coerce vertex args (labels/indices -> vertex IDs)
                try:
                    # Determine default label field if not given
                    if label_field is None and guess_labels:
                        label_field = self._infer_label_field()

                    if bound is not None and nxG is not None:
                        self._coerce_vertices_in_bound(bound, nxG, label_field)
                        # Reconstruct WITHOUT applying defaults (avoid flow_func=None, etc.)
                        pargs = bound.args
                        pkwargs = bound.kwargs
                    else:
                        # Fallback: best-effort coercion on kwargs only
                        if nxG is not None:
                            self._coerce_vertices_in_kwargs(kwargs, nxG, label_field)
                        pargs, pkwargs = tuple(args), kwargs
                except Exception:
                    pargs, pkwargs = tuple(args), kwargs  # best effort; let NX raise if needed

                # Never leak private knobs to NX
                for k in list(pkwargs.keys()):
                    if isinstance(k, str) and k.startswith("_nx_"):
                        pkwargs.pop(k, None)

                try:
                    return nx_callable(*pargs, **pkwargs)
                except _nx.vertexNotFound as e:
                    # Add actionable tip that actually tells how to fix it now.
                    sample = self.peek_vertices(5)
                    tip = (
                        f"{e}. vertices must be graph's vertex IDs.\n"
                        f"- If you passed labels, specify _nx_label_field=<vertex label column> "
                        f"or rely on auto-guess (columns like 'name'/'label'/'title').\n"
                        f"- Example: G.nx.shortest_path_length(G, source='a', target='z', weight='weight', _nx_label_field='name')\n"
                        f"- A few vertex IDs NX sees: {sample}"
                    )
                    raise _nx.vertexNotFound(tip) from e

            return wrapper

        # -- internals ---
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
            slice,
            slices,
            needed_attrs: set,
            simple: bool,
            edge_aggs: dict | None,
        ):
            from ..adapters import networkx_adapter as _gg_nx  # annnet.adapters.networkx_adapter

            nxG, manifest = _gg_nx.to_nx(
                self._G,
                directed=directed,
                hyperedge_mode=hyperedge_mode,
                slice=slice,
                slices=slices,
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

            # Collapse Multi* - simple Graph/DiGraph if requested
            if simple and nxG.is_multigraph():
                nxG = self._collapse_multiedges(
                    nxG, directed=directed, aggregations=edge_aggs, needed_attrs=needed_attrs
                )

            self._warn_on_loss(
                hyperedge_mode=hyperedge_mode, slice=slice, slices=slices, manifest=manifest
            )
            return nxG

        def _get_or_make_nx(
            self,
            *,
            directed: bool,
            hyperedge_mode: str,
            slice,
            slices,
            needed_attrs: set,
            simple: bool,
            edge_aggs: dict | None,
        ):
            key = (
                bool(directed),
                str(hyperedge_mode),
                tuple(sorted(slices)) if slices else None,
                str(slice) if slice is not None else None,
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
                    slice=slice,
                    slices=slices,
                    needed_attrs=needed_attrs,
                    simple=simple,
                    edge_aggs=edge_aggs,
                )
                if self.cache_enabled:
                    self._cache[key] = {"nxG": nxG, "version": version}
                return nxG
            return entry["nxG"]

        def _warn_on_loss(self, *, hyperedge_mode, slice, slices, manifest):
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
                slices_dict = getattr(self._G, "_slices", None)
                if (
                    isinstance(slices_dict, dict)
                    and len(slices_dict) > 1
                    and (slice is None and not slices)
                ):
                    msgs.append("multiple slices flattened into single NX graph")
            except Exception:
                pass
            if manifest is None:
                msgs.append("no manifest provided; round-trip fidelity not guaranteed")
            if msgs:
                warnings.warn(
                    "Graph-NX conversion is lossy: " + "; ".join(msgs) + ".",
                    category=RuntimeWarning,
                    stacklevel=3,
                )

        # -- label/ID mapping helpers 
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
                      # type: ignore

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

        def _coerce_vertex_id(self, x, nxG, label_field: str | None):
            # If already a vertex ID present in the backend, keep it.
            if x in nxG:
                return x
            # Internal index - vertex_id
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
            return x  # let NX decide (will raise vertexNotFound if still absent)

        def _coerce_vertex_or_iter(self, obj, nxG, label_field: str | None):
            if isinstance(obj, (list, tuple, set)):
                coerced = [self._coerce_vertex_id(v, nxG, label_field) for v in obj]
                return type(obj)(coerced) if not isinstance(obj, set) else set(coerced)
            return self._coerce_vertex_id(obj, nxG, label_field)

        def _coerce_vertices_in_kwargs(self, kwargs: dict, nxG, label_field: str | None):
            vertex_keys = {"source", "target", "u", "v", "vertex", "vertices", "nbunch", "center", "path"}
            for key in list(kwargs.keys()):
                if key in vertex_keys:
                    kwargs[key] = self._coerce_vertex_or_iter(kwargs[key], nxG, label_field)

        def _coerce_vertices_in_bound(self, bound, nxG, label_field: str | None):
            """Coerce vertices in a BoundArguments object using common vertex parameter names."""
            vertex_keys = {"source", "target", "u", "v", "vertex", "vertices", "nbunch", "center", "path"}
            for key in list(bound.arguments.keys()):
                if key in vertex_keys:
                    bound.arguments[key] = self._coerce_vertex_or_iter(
                        bound.arguments[key], nxG, label_field
                    )

        # -- Multi* collapse helpers -
        def _collapse_multiedges(
            self, nxG, *, directed: bool, aggregations: dict | None, needed_attrs: set
        ):
            """Collapse parallel edges into a single edge with aggregated attributes.
            Defaults: weight -> min (good for shortest paths), capacity -> sum (good for max-flow).
            """
            import networkx as _nx

            H = _nx.DiGraph() if directed else _nx.Graph()
            H.add_vertices_from(nxG.vertices(data=True))

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
        - Auto label-ID mapping for vertex args (kwargs + positionals).
        - _ig_simple=True collapses parallel edges to simple (Di)Graph.
        - _ig_edge_aggs={"weight":"min","capacity":"sum"} for parallel-edge aggregation.
        """

        def __init__(self, owner: "Graph"):
            self._G = owner
            self._cache = {}  # key -> {"igG": ig.Graph, "version": int}
            self.cache_enabled = True

        #  public API 
        def clear(self):
            self._cache.clear()

        def peek_vertices(self, k: int = 10):
            igG = self._get_or_make_ig(
                directed=True,
                hyperedge_mode="skip",
                slice=None,
                slices=None,
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
            slice=None,
            slices=None,
            needed_attrs=None,
            simple: bool = False,
            edge_aggs: dict | None = None,
        ):
            needed_attrs = needed_attrs or set()
            return self._get_or_make_ig(
                directed=directed,
                hyperedge_mode=hyperedge_mode,
                slice=slice,
                slices=slices,
                needed_attrs=needed_attrs,
                simple=simple,
                edge_aggs=edge_aggs,
            )

        # - dynamic dispatch -
        def __getattr__(self, name: str):
            def wrapper(*args, **kwargs):
                

                import igraph as _ig

                # proxy-only knobs (consumed here)
                directed = bool(kwargs.pop("_ig_directed", True))
                hyperedge_mode = kwargs.pop("_ig_hyperedge", "skip")  # "skip" | "expand"
                slice = kwargs.pop("_ig_slice", None)
                slices = kwargs.pop("_ig_slices", None)
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
                    slice=slice,
                    slices=slices,
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

                # bind to signature (best effort) so we can coerce vertex args
                try:
                    sig = inspect.signature(target)
                    bound = sig.bind_partial(*args, **kwargs)
                except Exception:
                    bound = None

                try:
                    if label_field is None and guess_labels:
                        label_field = self._infer_label_field()

                    if bound is not None:
                        self._coerce_vertices_in_bound(bound, igG, label_field)
                        bound.apply_defaults()
                        pargs, pkwargs = list(bound.args), dict(bound.kwargs)
                    else:
                        self._coerce_vertices_in_kwargs(kwargs, igG, label_field)
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

        # -- internals ---
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
            slice,
            slices,
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
                slice=slice,
                slices=slices,
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
                hyperedge_mode=hyperedge_mode, slice=slice, slices=slices, manifest=manifest
            )
            return igG

        def _get_or_make_ig(
            self,
            *,
            directed: bool,
            hyperedge_mode: str,
            slice,
            slices,
            needed_attrs: set,
            simple: bool,
            edge_aggs: dict | None,
        ):
            key = (
                bool(directed),
                str(hyperedge_mode),
                tuple(sorted(slices)) if slices else None,
                str(slice) if slice is not None else None,
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
                    slice=slice,
                    slices=slices,
                    needed_attrs=needed_attrs,
                    simple=simple,
                    edge_aggs=edge_aggs,
                )
                if self.cache_enabled:
                    self._cache[key] = {"igG": igG, "version": version}
                return igG
            return entry["igG"]

        def _warn_on_loss(self, *, hyperedge_mode, slice, slices, manifest):
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
                slices_dict = getattr(self._G, "_slices", None)
                if (
                    isinstance(slices_dict, dict)
                    and len(slices_dict) > 1
                    and (slice is None and not slices)
                ):
                    msgs.append("multiple slices flattened into single igraph graph")
            except Exception:
                pass
            if manifest is None:
                msgs.append("no manifest provided; round-trip fidelity not guaranteed")
            if msgs:
                warnings.warn(
                    "Graph-igraph conversion is lossy: " + "; ".join(msgs) + ".",
                    category=RuntimeWarning,
                    stacklevel=3,
                )

        # -- label/ID mapping helpers 
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
                      # type: ignore

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

        def _coerce_vertex_or_iter(self, obj, igG, label_field: str | None):
            if isinstance(obj, (list, tuple, set)):
                coerced = [self._coerce_vertex(v, igG, label_field) for v in obj]
                return type(obj)(coerced) if not isinstance(obj, set) else set(coerced)
            return self._coerce_vertex(obj, igG, label_field)

        def _coerce_vertices_in_kwargs(self, kwargs: dict, igG, label_field: str | None):
            vertex_keys = {
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
                if key in vertex_keys:
                    kwargs[key] = self._coerce_vertex_or_iter(kwargs[key], igG, label_field)

        def _coerce_vertices_in_bound(self, bound, igG, label_field: str | None):
            vertex_keys = {
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
                if key in vertex_keys:
                    bound.arguments[key] = self._coerce_vertex_or_iter(
                        bound.arguments[key], igG, label_field
                    )

        # -- edge-attr & multiedge helpers ---
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
        """vertex attribute table (observations)."""
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
    def slices(self):
        """slice operations (add, remove, union, intersect)."""
        if not hasattr(self, "_slice_manager"):
            self._slice_manager = SliceManager(self)
        return self._slice_manager

    @property
    def layers(self):
        """Layer operations."""
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
    def view(self, vertices=None, edges=None, slices=None, predicate=None):
        """Create lazy view/subgraph."""
        return GraphView(self, vertices, edges, slices, predicate)

    # Audit
    def snapshot(self, label=None):
        """Create a named snapshot of current graph state.

        Uses existing Graph attributes: entity_types, edge_to_idx, _slices, _version

        Parameters
        --
        label : str, optional
            Human-readable label for snapshot (auto-generated if None)

        Returns
        ---
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
                "slices": len(self._slices),
            },
            # Store minimal state for comparison (uses existing Graph attributes)
            "vertex_ids": set(v for v, t in self.entity_types.items() if t == "vertex"),
            "edge_ids": set(self.edge_to_idx.keys()),
            "slice_ids": set(self._slices.keys()),
        }

        self._snapshots.append(snapshot)
        return snapshot

    def diff(self, a, b=None):
        """Compare two snapshots or compare snapshot with current state.

        Parameters
        --
        a : str | dict | Graph
            First snapshot (label, snapshot dict, or Graph instance)
        b : str | dict | Graph | None
            Second snapshot. If None, compare with current state.

        Returns
        ---
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
                "slice_ids": set(ref._slices.keys()),
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
            "slice_ids": set(self._slices.keys()),
        }

    def list_snapshots(self):
        """List all snapshots.

        Returns
        ---
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
