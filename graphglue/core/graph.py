import numpy as np
import scipy.sparse as sp
import polars as pl
from collections import defaultdict
import math
from functools import wraps
from datetime import datetime, timezone
import inspect
import time
from enum import Enum

class EdgeType(Enum):
    DIRECTED = "DIRECTED"
    UNDIRECTED = "UNDIRECTED"

class Graph:
    """
    Sparse incidence-matrix graph with layers, attributes, parallel edges, and hyperedges.

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
    _vertex_RESERVED = {"vertex_id"}               # nothing structural for vertices
    _EDGE_RESERVED = {"edge_id","source","target","weight","edge_type","directed","layer","layer_weight","kind","members","head","tail"}    
    _LAYER_RESERVED = {"layer_id"}

    # Construction

    def __init__(self, directed=True):
        """
        Initialize an empty incidence-matrix graph.

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
        self.entity_types = {}   # entity_id -> 'vertex' or 'edge'
        
        # Edge mappings (supports parallel edges)
        self.edge_to_idx = {}    # edge_id -> column index
        self.idx_to_edge = {}    # column index -> edge_id
        self.edge_definitions = {}  # edge_id -> (source, target, edge_type)
        self.edge_weights = {}   # edge_id -> weight
        self.edge_directed = {} # edge_id -> bool  (True=directed, False=undirected)

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
        self._default_layer = 'default'
        self.layer_edge_weights = defaultdict(dict)  # layer_id -> {edge_id: weight}

        # Initialize default layer
        self._layers[self._default_layer] = {
            "vertices": set(),
            "edges": set(), 
            "attributes": {}
        }
        self._current_layer = self._default_layer

        # History and Timeline
        self._history_enabled = True
        self._history = []           # list[dict]
        self._version = 0
        self._history_clock0 = time.perf_counter_ns()
        self._install_history_hooks()  # wrap mutating methods

    # Layer basics

    def add_layer(self, layer_id, **attributes):
        """
        Create a new empty layer.

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
        if layer_id in self._layers:
            raise ValueError(f"Layer {layer_id} already exists")
        
        self._layers[layer_id] = {
            "vertices": set(),
            "edges": set(),
            "attributes": attributes
        }
        # Persist layer metadata to DF (pure attributes, upsert)
        if attributes:
            self.set_layer_attrs(layer_id, **attributes)
        return layer_id

    def set_active_layer(self, layer_id):
        """
        Set the active layer for subsequent operations.

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
        """
        Get the currently active layer ID.

        Returns
        -------
        str
            Active layer ID.
        """
        return self._current_layer
    
    def layers(self, include_default: bool = False):
        """
        Get a mapping of layer IDs to their metadata.

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
        """
        List layer IDs.

        Parameters
        ----------
        include_default : bool, optional
            Include the internal ``'default'`` layer if True.

        Returns
        -------
        list[str]
            Layer IDs.
        """
        return list(self.layers(include_default=include_default).keys())

    def has_layer(self, layer_id):
        """
        Check whether a layer exists.

        Parameters
        ----------
        layer_id : str

        Returns
        -------
        bool
        """
        return layer_id in self._layers
    
    def layer_count(self):
        """
        Get the number of layers (including the internal default).

        Returns
        -------
        int
        """
        return len(self._layers)

    def get_layer_info(self, layer_id):
        """
        Get a layer's metadata snapshot.

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
        """
        INTERNAL: Generate a unique edge ID for parallel edges.

        Returns
        -------
        str
            Fresh ``edge_<n>`` identifier (monotonic counter).
        """
        edge_id = f"edge_{self._next_edge_id}"
        self._next_edge_id += 1
        return edge_id
    
    def _ensure_vertex_table(self) -> None:
        """
        INTERNAL: Ensure the vertex attribute table exists with a canonical schema.

        Notes
        -----
        - Creates an empty Polars DF [DataFrame] with a single ``Utf8`` ``vertex_id`` column
        if missing or malformed.
        """        
        df = getattr(self, "vertex_attributes", None)
        if not isinstance(df, pl.DataFrame) or "vertex_id" not in df.columns:
            self.vertex_attributes = pl.DataFrame({"vertex_id": pl.Series([], dtype=pl.Utf8)})

    def _ensure_vertex_row(self, vertex_id: str) -> None:
        """
        INTERNAL: Ensure a row for ``vertex_id`` exists in the vertex attribute DF.

        Parameters
        ----------
        vertex_id : str

        Notes
        -----
        - Appends a new row with ``vertex_id`` and ``None`` for other columns if absent.
        - Preserves existing schema and columns.
        """    
        df = self.vertex_attributes
        # if row already exists, nothing to do
        if df.height and df.filter(pl.col("vertex_id") == vertex_id).height > 0:
            return
        if df.is_empty():
            # Create first row with the canonical vertex schema
            self.vertex_attributes = pl.DataFrame({"vertex_id": [vertex_id]}, schema={"vertex_id": pl.Utf8})
            return
        # Align columns: create a single dict with all columns present
        row = {c: None for c in df.columns}
        row["vertex_id"] = vertex_id
        self.vertex_attributes = pl.concat([df, pl.DataFrame([row])], how="vertical")

    # Build graph

    def add_vertex(self, vertex_id, layer=None, **attributes):
        """
        Add (or upsert) a vertex and optionally attach it to a layer.

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
        layer = layer or self._current_layer

        # Add to global superset if new
        if vertex_id not in self.entity_to_idx:
            idx = self._num_entities
            self.entity_to_idx[vertex_id] = idx
            self.idx_to_entity[idx] = vertex_id
            self.entity_types[vertex_id] = "vertex"
            self._num_entities += 1
            # Resize incidence matrix
            self._matrix.resize((self._num_entities, self._num_edges))

        # Add to specified layer
        if layer not in self._layers:
            self._layers[layer] = {"vertices": set(), "edges": set(), "attributes": {}}
        self._layers[layer]["vertices"].add(vertex_id)

        # Ensure vertex_attributes has a row for this vertex (even with no attrs)
        self._ensure_vertex_table()
        self._ensure_vertex_row(vertex_id)

        # Upsert passed attributes (if any)
        if attributes:
            self.vertex_attributes = self._upsert_row(self.vertex_attributes, vertex_id, attributes)

        return vertex_id

    def add_vertices(self, vertices, layer=None, **attributes):
        """
        Add (or upsert) multiple vertices and optionally attach them to a layer.

        Parameters
        ----------
        vertices : Iterable[str] | Mapping[str, dict] | Iterable[tuple[str, dict]]
            - Iterable of vertex IDs -> the same `attributes` apply to each.
            - Mapping vertex_id -> per-vertex attributes dict (merged with `attributes`).
            - Iterable of (vertex_id, per_vertex_attributes) tuples.
        layer : str, optional
            Target layer for all vertices. Defaults to the active layer.
        **attributes
            Attributes applied to every vertex (overridden by per-vertex attributes when provided).

        Returns
        -------
        list[str]
            The list of vertex IDs (echoed in insertion order).
        """
        # Normalize input into an iterator of (vertex_id, per_vertex_attrs)
        if vertices is None:
            return []

        if isinstance(vertices, dict):
            iterator = vertices.items()
        else:
            iterator = []
            for item in vertices:
                # Allow (vertex_id, {attr: ...}) tuples
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], dict):
                    vertex_id, per_attrs = item
                    iterator.append((vertex_id, per_attrs))
                else:
                    # Treat plain values as vertex IDs with no per-vertex attrs
                    iterator.append((item, {}))

        added = []
        for vertex_id, per_attrs in iterator:
            merged_attrs = {**attributes, **per_attrs} if attributes or per_attrs else {}
            added.append(self.add_vertex(vertex_id, layer=layer, **merged_attrs))
        return added

    def add_edge_entity(self, edge_entity_id, layer=None, **attributes):
        """
        Add an **edge entity** (vertex-edge hybrid) that can connect to vertices/edges.

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
        layer = layer or self._current_layer
        
        # Add to global superset if new
        if edge_entity_id not in self.entity_to_idx:
            self._add_edge_entity(edge_entity_id)
        
        # Add to specified layer
        if layer not in self._layers:
            self._layers[layer] = {"vertices": set(), "edges": set(), "attributes": {}}
        
        self._layers[layer]["vertices"].add(edge_entity_id)
        
        # Add attributes (treat edge entities like vertices for attributes)
        if attributes:
            self.set_vertex_attrs(edge_entity_id, **attributes)

        return edge_entity_id
  
    def _add_edge_entity(self, edge_id):
        """
        INTERNAL: Register an **edge-entity** so edges can attach to it (vertex-edge mode).

        Parameters
        ----------
        edge_id : str
            Identifier to insert into the entity index as type ``'edge'``.

        Notes
        -----
        - Adds a new entity row and resizes the DOK incidence matrix accordingly.
        """
        if edge_id not in self.entity_to_idx:
            idx = self._num_entities
            self.entity_to_idx[edge_id] = idx
            self.idx_to_entity[idx] = edge_id
            self.entity_types[edge_id] = 'edge'
            self._num_entities += 1
            
            # Resize matrix
            self._matrix.resize((self._num_entities, self._num_edges))
 
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
        edge_directed=None,
        **attributes,
    ):
            """
            Add or update a binary edge between two entities.

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
            # ---- normalize endpoints: accept str OR iterable; route hyperedges ----
            def _to_tuple(x):
                # str/bytes -> (x,), not iterable of chars
                if isinstance(x, (str, bytes)):
                    return (x,), False
                try:
                    xs = tuple(x)
                except TypeError:
                    # non-iterable -> treat as single vertex id
                    return (x,), False
                return xs, (len(xs) != 1)  # (sequence, is_multi)

            S, src_multi = _to_tuple(source)
            T, tgt_multi = _to_tuple(target)

            # If any endpoint has >1 members, this is a hyperedge
            if src_multi or tgt_multi:
                if edge_directed:
                    return self.add_hyperedge(
                        head=S, tail=T, edge_directed=True,
                        layer=layer, weight=weight, edge_id=edge_id, **attributes
                    )
                else:
                    members = tuple(set(S) | set(T))
                    return self.add_hyperedge(
                        members=members, edge_directed=False,
                        layer=layer, weight=weight, edge_id=edge_id, **attributes
                    )

            # Binary case: unwrap singletons to plain vertex IDs (str)
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

            # ensure vertices exist (global)
            def _ensure_vertex_or_edge_entity(x):
                if x in self.entity_to_idx:
                    return
                if edge_type == "vertex_edge" and isinstance(x, str) and x.startswith("edge_"):
                    self.add_edge_entity(x, layer=layer)
                else:
                    self.add_vertex(x, layer=layer)

            _ensure_vertex_or_edge_entity(source)
            _ensure_vertex_or_edge_entity(target)

            # indices (after potential vertex creation)
            source_idx = self.entity_to_idx[source]
            target_idx = self.entity_to_idx[target]

            # edge id
            if edge_id is None:
                edge_id = self._get_next_edge_id()

            # determine direction
            is_dir = self.directed if edge_directed is None else bool(edge_directed)

            # create or update bookkeeping
            if edge_id in self.edge_to_idx:
                # update
                col_idx = self.edge_to_idx[edge_id]

                # allow explicit direction change; otherwise keep existing
                if edge_directed is None:
                    is_dir = self.edge_directed.get(edge_id, is_dir)
                self.edge_directed[edge_id] = is_dir
                self.set_edge_attrs(edge_id, edge_type=(EdgeType.DIRECTED if is_dir else EdgeType.UNDIRECTED))


                # if source/target changed, update definition
                old_src, old_tgt, old_type = self.edge_definitions[edge_id]
                self.edge_definitions[edge_id] = (source, target, old_type)  # keep old_type by default

                # ensure matrix has enough rows (in case vertices were added since creation)
                if self._matrix.shape[0] < self._num_entities:
                    self._matrix.resize((self._num_entities, self._matrix.shape[1]))

                # rewrite column
                self._matrix[:, col_idx] = 0
                self._matrix[source_idx, col_idx] = weight
                if source != target:
                    self._matrix[target_idx, col_idx] = -weight if is_dir else weight

                self.edge_weights[edge_id] = weight

            else:
                # create
                col_idx = self._num_edges
                self.edge_to_idx[edge_id] = col_idx
                self.idx_to_edge[col_idx] = edge_id
                self.edge_definitions[edge_id] = (source, target, edge_type)
                self.edge_weights[edge_id] = weight
                self.edge_directed[edge_id] = is_dir
                self._num_edges += 1

                # grow matrix to fit
                self._matrix.resize((self._num_entities, self._num_edges))
                self._matrix[source_idx, col_idx] = weight
                if source != target:
                    self._matrix[target_idx, col_idx] = -weight if is_dir else weight

            # layer handling
            if touch_layer:
                if layer not in self._layers:
                    self._layers[layer] = {"vertices": set(), "edges": set(), "attributes": {}}
                self._layers[layer]["edges"].add(edge_id)
                self._layers[layer]["vertices"].update((source, target))

                if layer_weight is not None:
                    w = float(layer_weight)
                    self.set_edge_layer_attrs(layer, edge_id, weight=w)
                    self.layer_edge_weights.setdefault(layer, {})[edge_id] = w

            # propagation
            if propagate == "shared":
                self._propagate_to_shared_layers(edge_id, source, target)
            elif propagate == "all":
                self._propagate_to_all_layers(edge_id, source, target)

            # attributes
            if attributes:
                self.set_edge_attrs(edge_id, **attributes)

            return edge_id

    def add_parallel_edge(self, source, target, weight=1.0, **attributes):
        """
        Add a parallel edge (same endpoints, different ID).

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
        return self.add_edge(source, target, weight=weight, edge_id=None, **attributes)
   
    def add_hyperedge(
        self,
        *,
        members=None,
        head=None,
        tail=None,
        layer=None,
        weight=1.0,
        edge_id=None,
        edge_directed=None,   # bool or None (None -> infer from params)
        **attributes,
    ):
        """
        Create a k-ary hyperedge as a single incidence column.

        Modes
        -----
        - **Undirected**: pass ``members`` (>=2). Each member gets ``+weight``.
        - **Directed**: pass ``head`` and ``tail`` (both non-empty, disjoint).
        Head gets ``+weight``; tail gets ``-weight``.

        Parameters
        ----------
        members : Iterable[str], optional
            Undirected member set (size ≥ 2).
        head : Iterable[str], optional
            Directed head set (non-empty).
        tail : Iterable[str], optional
            Directed tail set (non-empty, disjoint from head).
        layer : str, optional
            Layer to place the hyperedge into. Defaults to the active layer.
        weight : float, optional
            Global weight stored in the column.
        edge_id : str, optional
            Explicit ID; generated if omitted.
        edge_directed : bool, optional
            Force directed/undirected; inferred from parameters if None.
        **attributes
            Pure edge attributes.

        Returns
        -------
        str
            The hyperedge ID.

        Raises
        ------
        ValueError
            For invalid argument combinations or empty/disjointness violations.
        """
        # ---- validate form ----
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

        # ensure participants exist globally (vertices or edge-entities already supported)
        def _ensure_entity(x):
            if x in self.entity_to_idx:
                return
            # hyperedges connect to vertices/edge-entities similarly to binary edges
            if isinstance(x, str) and x.startswith("edge_") and x in self.entity_types and self.entity_types[x] == "edge":
                # already an edge-entity
                return
            # default: treat as vertex
            self.add_vertex(x, layer=layer)

        if members is not None:
            for u in members:
                _ensure_entity(u)
        else:
            for u in head + tail:
                _ensure_entity(u)

        # allocate edge id + column
        if edge_id is None:
            edge_id = self._get_next_edge_id()
        is_new = edge_id not in self.edge_to_idx
        if is_new:
            col_idx = self._num_edges
            self.edge_to_idx[edge_id] = col_idx
            self.idx_to_edge[col_idx] = edge_id
            self._num_edges += 1
            # grow matrix for new column
            self._matrix.resize((self._num_entities, self._num_edges))
        else:
            col_idx = self.edge_to_idx[edge_id]
            # zero out old column if reusing id
            self._matrix[:, col_idx] = 0

        # write column entries
        if members is not None:
            # undirected: put +weight at each member
            for u in members:
                self._matrix[self.entity_to_idx[u], col_idx] = float(weight)
            self.hyperedge_definitions[edge_id] = {
                "directed": False,
                "members": set(members),
            }
        else:
            # directed: +weight on head, -weight on tail (orientation)
            for u in head:
                self._matrix[self.entity_to_idx[u], col_idx] = float(weight)
            for v in tail:
                self._matrix[self.entity_to_idx[v], col_idx] = -float(weight)
            self.hyperedge_definitions[edge_id] = {
                "directed": True,
                "head": set(head),
                "tail": set(tail),
            }

        # bookkeeping shared with binary edges
        self.edge_weights[edge_id] = float(weight)
        self.edge_directed[edge_id] = bool(directed)
        self.edge_kind[edge_id] = "hyper"

        # keep a sentinel in edge_definitions so old code won't crash
        self.edge_definitions[edge_id] = (None, None, "hyper")

        # layer membership + per-layer weights
        if layer is not None:
            if layer not in self._layers:
                self._layers[layer] = {"vertices": set(), "edges": set(), "attributes": {}}
            self._layers[layer]["edges"].add(edge_id)
            if members is not None:
                self._layers[layer]["vertices"].update(members)
            else:
                self._layers[layer]["vertices"].update(self.hyperedge_definitions[edge_id]["head"])
                self._layers[layer]["vertices"].update(self.hyperedge_definitions[edge_id]["tail"])

        # attributes
        if attributes:
            self.set_edge_attrs(edge_id, **attributes)

        return edge_id

    def add_edge_to_layer(self, lid, eid):
        """
        Attach an existing edge to a layer (no weight changes).

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
        """
        INTERNAL: Add an edge to all layers that already contain **both** endpoints.

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
        """
        INTERNAL: Add an edge to any layer containing **either** endpoint and
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
        """
        Normalize a single vertex or an iterable of vertices into a set.

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

    # Remove / mutate down

    def remove_edge(self, edge_id):
        """
        Remove an edge (binary or hyperedge) from the graph.

        Parameters
        ----------
        edge_id : str

        Raises
        ------
        KeyError
            If the edge is not found.

        Notes
        -----
        - Physically removes the incidence column (CSR [Compressed Sparse Row] slice).
        - Cleans edge attributes, layer memberships, and per-layer entries.
        """
        if edge_id not in self.edge_to_idx:
            raise KeyError(f"Edge {edge_id} not found")

        col_idx = self.edge_to_idx[edge_id]

        # Convert to CSR for efficient column removal
        csr_matrix = self._matrix.tocsr()

        # Create mask to remove column
        mask = np.ones(self._num_edges, dtype=bool)
        mask[col_idx] = False

        # Remove column
        csr_matrix = csr_matrix[:, mask]
        self._matrix = csr_matrix.todok()

        # Update mappings
        del self.edge_to_idx[edge_id]
        del self.edge_definitions[edge_id]
        del self.edge_weights[edge_id]

        # Update directionality metadata
        if edge_id in self.edge_directed:
            del self.edge_directed[edge_id]

        # Reindex remaining edges
        new_edge_to_idx = {}
        new_idx_to_edge = {}

        new_idx = 0
        for old_idx in range(self._num_edges):
            if old_idx != col_idx:
                edge_id_old = self.idx_to_edge[old_idx]
                new_edge_to_idx[edge_id_old] = new_idx
                new_idx_to_edge[new_idx] = edge_id_old
                new_idx += 1

        self.edge_to_idx = new_edge_to_idx
        self.idx_to_edge = new_idx_to_edge
        self._num_edges -= 1

        # Remove from edge attributes
        if isinstance(self.edge_attributes, pl.DataFrame) and self.edge_attributes.height > 0 and "edge_id" in self.edge_attributes.columns:
            self.edge_attributes = self.edge_attributes.filter(pl.col("edge_id") != edge_id)

        # Remove from per-layer membership
        for layer_data in self._layers.values():
            layer_data["edges"].discard(edge_id)

        # Remove from edge-layer attributes
        if isinstance(self.edge_layer_attributes, pl.DataFrame) and self.edge_layer_attributes.height > 0 and "edge_id" in self.edge_layer_attributes.columns:
            self.edge_layer_attributes = self.edge_layer_attributes.filter(pl.col("edge_id") != edge_id)

        # also clear in legacy dict
        for d in self.layer_edge_weights.values():
            d.pop(edge_id, None)

        self.edge_kind.pop(edge_id, None)
        self.hyperedge_definitions.pop(edge_id, None)

    def remove_vertex(self, vertex_id):
        """
        Remove a vertex and all incident edges (binary + hyperedges).

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

        # Convert to CSR for efficient row removal
        csr_matrix = self._matrix.tocsr()

        # Remove entity row from matrix
        mask = np.ones(self._num_entities, dtype=bool)
        mask[entity_idx] = False
        csr_matrix = csr_matrix[mask, :]
        self._matrix = csr_matrix.todok()

        # Update entity mappings
        del self.entity_to_idx[vertex_id]
        del self.entity_types[vertex_id]

        # Reindex remaining entities
        new_entity_to_idx = {}
        new_idx_to_entity = {}
        new_idx = 0
        for old_idx in range(self._num_entities):
            if old_idx != entity_idx:
                entity_id = self.idx_to_entity[old_idx]
                new_entity_to_idx[entity_id] = new_idx
                new_idx_to_entity[new_idx] = entity_id
                new_idx += 1

        self.entity_to_idx = new_entity_to_idx
        self.idx_to_entity = new_idx_to_entity
        self._num_entities -= 1

        # Remove from vertex attributes
        if isinstance(self.vertex_attributes, pl.DataFrame):
            if self.vertex_attributes.height > 0 and "vertex_id" in self.vertex_attributes.columns:
                self.vertex_attributes = self.vertex_attributes.filter(pl.col("vertex_id") != vertex_id)

        # Remove from per-layer membership
        for layer_data in self._layers.values():
            layer_data["vertices"].discard(vertex_id)

    def remove_layer(self, layer_id):
        """
        Remove a non-default layer and its per-layer attributes.

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

    # Attributes & weights

    def set_graph_attribute(self, key, value):
        """
        Set a graph-level attribute.

        Parameters
        ----------
        key : str
        value : Any
        """
        self.graph_attributes[key] = value
    
    def get_graph_attribute(self, key, default=None):
        """
        Get a graph-level attribute.

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
        """
        Upsert pure vertex attributes (non-structural) into the vertex DF.

        Parameters
        ----------
        vertex_id : str
        **attrs
            Key/value attributes. Structural keys are ignored.
        """
        # keep attributes table pure
        clean = {k: v for k, v in attrs.items() if k not in self._vertex_RESERVED}
        if clean:
            self.vertex_attributes = self._upsert_row(self.vertex_attributes, vertex_id, clean)

    def get_attr_vertex(self, vertex_id, key, default=None):
        """
        Get a single vertex attribute (scalar) or default if missing.

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

    def get_vertex_attribute(self, vertex_id, attribute): #legacy alias
        """
        (Legacy alias) Get a single vertex attribute from the Polars DF [DataFrame].

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
        """
        Upsert pure edge attributes (non-structural) into the edge DF.

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

    def get_attr_edge(self, edge_id, key, default=None):
        """
        Get a single edge attribute (scalar) or default if missing.

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

    def get_edge_attribute(self, edge_id, attribute): #legacy alias
        """
        (Legacy alias) Get a single edge attribute from the Polars DF [DataFrame].

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
        """
        Upsert pure layer attributes.

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
        """
        Get a single layer attribute (scalar) or default if missing.

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
        """
        Upsert per-layer attributes for a specific edge.

        Parameters
        ----------
        layer_id : str
        edge_id : str
        **attrs
            Pure attributes. Structural keys are ignored (except 'weight', which is allowed here).
        """
        # allow 'weight' through; keep ignoring true structural keys
        clean = {k: v for k, v in attrs.items() if (k not in self._EDGE_RESERVED) or (k == "weight")}
        if not clean:
            return

        # Ensure edge_layer_attributes compares strings to strings (defensive against prior bad writes)
        if isinstance(self.edge_layer_attributes, pl.DataFrame) and self.edge_layer_attributes.height > 0:
            # Cast only if columns exist to avoid raising in early init states
            to_cast = []
            if "layer_id" in self.edge_layer_attributes.columns:
                to_cast.append(pl.col("layer_id").cast(pl.Utf8))
            if "edge_id" in self.edge_layer_attributes.columns:
                to_cast.append(pl.col("edge_id").cast(pl.Utf8))
            if to_cast:
                self.edge_layer_attributes = self.edge_layer_attributes.with_columns(*to_cast)

        # edge_layer_attributes is a pl.DataFrame with columns: layer_id, edge_id, ...

        self.edge_layer_attributes = self._upsert_row(
            self.edge_layer_attributes, (layer_id, edge_id), clean
        )

    def get_edge_layer_attr(self, layer_id, edge_id, key, default=None):
        """
        Get a per-layer attribute for an edge.

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

    def set_layer_edge_weight(self, layer_id, edge_id, weight): #legacy weight helper
        """
        Set a legacy per-layer weight override for an edge.

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
        """
        Resolve the effective weight for an edge, optionally within a layer.

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
        """
        Audit attribute tables for extra/missing rows and invalid edge-layer pairs.

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
        if isinstance(ela, pl.DataFrame) and ela.height > 0 and {"layer_id", "edge_id"} <= set(ela.columns):
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
        """
        INTERNAL: Infer an appropriate Polars dtype for a Python value.

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
        import enum, polars as pl
        if v is None: return pl.Null
        if isinstance(v, bool): return pl.Boolean
        if isinstance(v, int) and not isinstance(v, bool): return pl.Int64
        if isinstance(v, float): return pl.Float64
        if isinstance(v, enum.Enum): return pl.Object     # important for EdgeType
        if isinstance(v, (bytes, bytearray)): return pl.Binary
        if isinstance(v, (list, tuple)):
            inner = self._pl_dtype_for_value(v[0]) if len(v) else pl.Utf8
            return pl.List(pl.Utf8 if inner == pl.Null else inner)
        if isinstance(v, dict): return pl.Object
        return pl.Utf8

    def _ensure_attr_columns(self, df: pl.DataFrame, attrs: dict) -> pl.DataFrame:
        """
        INTERNAL: Create/align attribute columns and dtypes to accept ``attrs``.

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
        """
        INTERNAL: Upsert a row in a Polars DF [DataFrame] using explicit key columns.

        Keys
        ----
        - ``vertex_attributes``           → key: ``["vertex_id"]``
        - ``edge_attributes``           → key: ``["edge_id"]``
        - ``layer_attributes``          → key: ``["layer_id"]``
        - ``edge_layer_attributes``     → key: ``["layer_id", "edge_id"]``

        Parameters
        ----------
        df : polars.DataFrame
            Target attribute table.
        idx : str | tuple[str, str]
            Key value(s). For edge-layer, pass ``(layer_id, edge_id)``.
        attrs : dict
            Columns to insert/update.

        Returns
        -------
        polars.DataFrame
            A **new** DataFrame with the row inserted/updated (caller must reassign).

        Raises
        ------
        ValueError
            If key columns cannot be inferred from ``df`` schema or a composite key is malformed.

        Notes
        -----
        - Ensures necessary columns/dtypes first (via ``_ensure_attr_columns``).
        - Updates cast literals to existing column dtypes; inserts align schemas before ``vstack``.
        - Resolves dtype mismatches by upcasting both sides to ``Utf8``.
        """
        if not isinstance(attrs, dict) or not attrs:
            return df

        cols = set(df.columns)

        # determine key columns + values
        if {"layer_id", "edge_id"} <= cols:
            if not (isinstance(idx, tuple) and len(idx) == 2):
                raise ValueError("idx must be a (layer_id, edge_id) tuple")
            key_vals = {"layer_id": idx[0], "edge_id": idx[1]}
            key_cols = ["layer_id", "edge_id"]
        elif "vertex_id" in cols:
            key_vals = {"vertex_id": idx}
            key_cols = ["vertex_id"]
        elif "edge_id" in cols:
            key_vals = {"edge_id": idx}
            key_cols = ["edge_id"]
        elif "layer_id" in cols:
            key_vals = {"layer_id": idx}
            key_cols = ["layer_id"]
        else:
            raise ValueError("Cannot infer key columns from DataFrame schema")

        # ensure attribute columns exist/cast appropriately
        df = self._ensure_attr_columns(df, attrs)

        # match condition
        cond = None
        for k, v in key_vals.items():
            c = (pl.col(k) == pl.lit(v))
            cond = c if cond is None else (cond & c)

        exists = df.filter(cond).height > 0

        if exists:
            # cast literal to column dtype to avoid type conflicts
            upds = []
            for k, v in attrs.items():
                col_dtype = df.schema[k]
                upds.append(
                    pl.when(cond)
                    .then(pl.lit(v).cast(col_dtype))
                    .otherwise(pl.col(k))
                    .alias(k)
                )
            return df.with_columns(upds)

        # INSERT
        new_row = {c: None for c in df.columns}
        new_row.update(key_vals)
        new_row.update(attrs)

        to_append = pl.DataFrame([new_row])

        # align schemas before vstack
        # 1) ensure to_append has all df columns
        for c in df.columns:
            if c not in to_append.columns:
                to_append = to_append.with_columns(pl.lit(None).cast(df.schema[c]).alias(c))

        # 2) cast Null columns in df to incoming dtypes and resolve mismatches by upcasting to Utf8
        for c in df.columns:
            if c not in to_append.columns:
                continue
            left = df.schema[c]
            right = to_append.schema[c]
            if left == pl.Null and right != pl.Null:
                df = df.with_columns(pl.col(c).cast(right))
            elif right == pl.Null and left != pl.Null:
                to_append = to_append.with_columns(pl.col(c).cast(left))
            elif left != right:
                # upcast both to Utf8 to avoid SchemaError from mixed types
                df = df.with_columns(pl.col(c).cast(pl.Utf8))
                to_append = to_append.with_columns(pl.col(c).cast(pl.Utf8))

        return df.vstack(to_append)

    ## Full attribute dict for a single entity
    
    def get_edge_attrs(self, edge) -> dict:
        """
        Return the full attribute dict for a single edge.

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
            import polars as pl  # noqa: F401
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
        """
        Return the full attribute dict for a single vertex.

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
            import polars as pl  # noqa: F401
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
        """
        Retrieve edge attributes as a dictionary.

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
        """
        Retrieve vertex (vertex) attributes as a dictionary.

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
        """
        Extract a specific attribute column for all edges.

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
        return {row["edge_id"]: row[key] if row[key] is not None else default for row in df.iter_rows(named=True)}

    def get_edges_by_attr(self, key: str, value) -> list:
        """
        Retrieve all edges where a given attribute equals a specific value.

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
        """
        Return a shallow copy of the graph-level attributes dictionary.

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

    # Basic queries & metrics

    def get_vertex(self, index: int) -> str:
        """
        Return the vertex ID corresponding to a given internal index.

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
        """
        Return edge endpoints in a canonical form.

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
            directed = self.edge_directed.get(eid, self.directed)
            if directed:
                return (frozenset([u]), frozenset([v]))
            else:
                M = frozenset([u, v])
                return (M, M)

    def incident_edges(self, vertex_id) -> list[int]:
        """
        Return all edge indices incident to a given vertex.

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
                if (meta.get("directed", False) and (vertex_id in meta["head"] or vertex_id in meta["tail"])) \
                or (not meta.get("directed", False) and vertex_id in meta["members"]):
                    incident.append(j)
            else:
                u, v, _etype = self.edge_definitions[eid]
                if vertex_id == u or vertex_id == v:
                    incident.append(j)

        return incident

    def _is_directed_edge(self, edge_id):
        """
        Check if an edge is directed (per-edge flag overrides graph default).

        Parameters
        ----------
        edge_id : str

        Returns
        -------
        bool
        """
        return bool(self.edge_directed.get(edge_id, self.directed))

    def has_edge(self, source, target, edge_id=None):
        """
        Test for the existence of an edge.

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
            
    def get_edge_ids(self, source, target):
        """
        List all edge IDs between two endpoints.

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
        """
        Degree of a vertex or edge-entity (number of incident non-zero entries).

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
        """
        Get all vertex IDs (excluding edge-entities).

        Returns
        -------
        list[str]
        """
        return [eid for eid, etype in self.entity_types.items() if etype == 'vertex']
    
    def edges(self):
        """
        Get all edge IDs.

        Returns
        -------
        list[str]
        """
        return list(self.edge_to_idx.keys())
    
    def edge_list(self):
        """
        Materialize (source, target, edge_id, weight) for binary/vertex-edge edges.

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
        """
        List IDs of directed edges.

        Returns
        -------
        list[str]
        """
        return [eid for eid in self.edge_to_idx.keys() 
                if self.edge_directed.get(eid, self.directed)]

    def get_undirected_edges(self):
        """
        List IDs of undirected edges.

        Returns
        -------
        list[str]
        """
        return [eid for eid in self.edge_to_idx.keys() 
                if not self.edge_directed.get(eid, self.directed)]

    def number_of_vertices(self):
        """
        Count vertices (excluding edge-entities).

        Returns
        -------
        int
        """
        return len([e for e in self.entity_types.values() if e == 'vertex'])
    
    def number_of_edges(self):
        """
        Count edges (columns in the incidence matrix).

        Returns
        -------
        int
        """
        return self._num_edges

    def global_entity_count(self):
        """
        Count unique entities present across all layers (union of memberships).

        Returns
        -------
        int
        """
        all_vertices = set()
        for layer_data in self._layers.values():
            all_vertices.update(layer_data["vertices"])
        return len(all_vertices)

    def global_edge_count(self):
        """
        Count unique edges present across all layers (union of memberships).

        Returns
        -------
        int
        """
        all_edges = set()
        for layer_data in self._layers.values():
            all_edges.update(layer_data["edges"])
        return len(all_edges)

    def in_edges(self, vertices):
        """
        Iterate over all edges that are **incoming** to one or more vertices.

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
        """
        Iterate over all edges that are **outgoing** from one or more vertices.

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

    @property
    def V(self):
        """
        All vertices as a tuple.

        Returns
        -------
        tuple
            Tuple of all vertex IDs in the graph.
        """
        return tuple(self.vertices())

    @property
    def E(self):
        """
        All edges as a tuple.

        Returns
        -------
        tuple
            Tuple of all edge identifiers (whatever `self.edges()` yields).
        """
        return tuple(self.edges())

    @property
    def num_vertices(self):
        """
        Total number of vertices (vertices) in the graph.
        """
        return self.number_of_vertices()

    @property
    def num_edges(self):
        """
        Total number of edges in the graph.
        """
        return self.number_of_edges()

    @property
    def nv(self):
        """
        Shorthand for num_vertices.
        """
        return self.num_vertices

    @property
    def ne(self):
        """
        Shorthand for num_edges.
        """
        return self.num_edges

    @property
    def shape(self):
        """
        Graph shape as a tuple: (num_vertices, num_edges).
        Useful for quick inspection.
        """
        return (self.num_vertices, self.num_edges)

    # Materialized views

    def edges_view(self, layer=None, include_directed=True, include_weight=True, resolved_weight=True, copy=True):
        """
        Build a Polars DF view of edges with optional layer join.

        Parameters
        ----------
        layer : str, optional
            If provided, join per-layer edge attributes (prefixed with ``layer_``).
        include_directed : bool, optional
            Include a ``directed`` column.
        include_weight : bool, optional
            Include a ``global_weight`` column.
        resolved_weight : bool, optional
            Include an ``effective_weight`` column.
        copy : bool, optional
            Return a cloned DF to keep it read-only.

        Returns
        -------
        polars.DataFrame
            Columns include: ``edge_id``, ``kind`` ('binary'/'hyper'), endpoint metadata,
            and optional weight/directedness and attribute columns.
        """    
        # build base rows from in-memory graph state
        rows = []
        for eid in self.edge_to_idx.keys():
            kind = self.edge_kind.get(eid, None)
            row = {"edge_id": eid, "kind": kind or "binary"}
            if include_directed:
                row["directed"] = self.edge_directed.get(eid, self.directed)
            if include_weight:
                row["global_weight"] = self.edge_weights.get(eid, None)
            if resolved_weight:
                row["effective_weight"] = self.get_effective_edge_weight(eid, layer=layer)

            if kind == "hyper":
                meta = self.hyperedge_definitions[eid]
                if meta["directed"]:
                    row["head"] = tuple(sorted(meta["head"]))
                    row["tail"] = tuple(sorted(meta["tail"]))
                else:
                    row["members"] = tuple(sorted(meta["members"]))
            else:
                s, t, etype = self.edge_definitions[eid]
                row["source"] = s
                row["target"] = t
                row["edge_type"] = etype  # 'regular' | 'vertex_edge' | None

            rows.append(row)

        base = pl.DataFrame(rows) if rows else pl.DataFrame(schema={"edge_id": pl.Utf8, "kind": pl.Utf8})

        # join with pure edge attributes (on explicit key column)
        if hasattr(self, "edge_attributes") and isinstance(self.edge_attributes, pl.DataFrame) and self.edge_attributes.height > 0:
            out = base.join(self.edge_attributes, on="edge_id", how="left")
        else:
            out = base

        # add layer-specific attributes (prefixed), if requested layer exists
        if layer is not None and hasattr(self, "edge_layer_attributes") and isinstance(self.edge_layer_attributes, pl.DataFrame) and self.edge_layer_attributes.height > 0:
            layer_slice = (
                self.edge_layer_attributes
                .filter(pl.col("layer_id") == layer)
                .drop("layer_id")
            )
            if layer_slice.height > 0:
                # prefix non-key columns
                rename_map = {c: f"layer_{c}" for c in layer_slice.columns if c not in {"edge_id"}}
                if rename_map:
                    layer_slice = layer_slice.rename(rename_map)
                out = out.join(layer_slice, on="edge_id", how="left")

        return out.clone() if copy else out

    def vertices_view(self, copy=True):
        """
        Read-only vertex attribute table.

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
        """
        Read-only layer attribute table.

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
        """
        vertices in a layer.

        Parameters
        ----------
        layer_id : str

        Returns
        -------
        set[str]
        """
        return self._layers[layer_id]["vertices"].copy()

    def get_layer_edges(self, layer_id):
        """
        Edges in a layer.

        Parameters
        ----------
        layer_id : str

        Returns
        -------
        set[str]
        """
        return self._layers[layer_id]["edges"].copy()

    def layer_union(self, layer_ids):
        """
        Union of multiple layers.

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
        """
        Intersection of multiple layers.

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
                "edges": self._layers[layer_id]["edges"].copy()
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
        """
        Set difference: elements in ``layer1_id`` not in ``layer2_id``.

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
            "edges": layer1["edges"] - layer2["edges"]
        }

    def create_layer_from_operation(self, result_layer_id, operation_result, **attributes):
        """
        Create a new layer from the result of a set operation.

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
            "attributes": attributes
        }
        
        return result_layer_id

    def edge_presence_across_layers(
        self,
        edge_id: str | None = None,
        source: str | None = None,
        target: str | None = None,
        *,
        include_default: bool = False,
        undirected_match: bool | None = None
    ):
        """
        Locate where an edge exists across layers.

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

        layers_view = self.layers(include_default=include_default)

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
                edge_is_directed = self.edge_directed.get(eid, self.directed)
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
        """
        Locate layers containing a hyperedge with exactly these sets.

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

        layers_view = self.layers(include_default=include_default)
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
        """
        List layers containing a specific vertex.

        Parameters
        ----------
        vertex_id : str
        include_default : bool, optional

        Returns
        -------
        list[str]
        """
        layers_with_vertex = []
        for layer_id, layer_data in self.layers(include_default=include_default).items():
            if vertex_id in layer_data["vertices"]:
                layers_with_vertex.append(layer_id)
        return layers_with_vertex

    def conserved_edges(self, min_layers=2, include_default=False):
        """
        Edges present in at least ``min_layers`` layers.

        Parameters
        ----------
        min_layers : int, optional
        include_default : bool, optional

        Returns
        -------
        dict[str, int]
            ``{edge_id: count}``.
        """
        layers_to_check = self.layers(include_default=include_default)  # hides 'default' by default
        edge_counts = {}
        for _, layer_data in layers_to_check.items():
            for eid in layer_data["edges"]:
                edge_counts[eid] = edge_counts.get(eid, 0) + 1
        return {eid: c for eid, c in edge_counts.items() if c >= min_layers}

    def layer_specific_edges(self, layer_id):
        """
        Edges that appear **only** in the specified layer.

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
            count = sum(1 for layer_data in self._layers.values() 
                    if edge_id in layer_data["edges"])
            if count == 1:  # Only in target layer
                specific_edges.add(edge_id)
        
        return specific_edges

    def temporal_dynamics(self, ordered_layers, metric='edge_change'):
        """
        Compute changes between consecutive layers in a temporal sequence.

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
            
            if metric == 'edge_change':
                added = len(next_data["edges"] - current_data["edges"])
                removed = len(current_data["edges"] - next_data["edges"])
                changes.append({'added': added, 'removed': removed, 'net_change': added - removed})
            
            elif metric == 'vertex_change':
                added = len(next_data["vertices"] - current_data["vertices"])
                removed = len(current_data["vertices"] - next_data["vertices"])
                changes.append({'added': added, 'removed': removed, 'net_change': added - removed})
        
        return changes

    def create_aggregated_layer(self, source_layer_ids, target_layer_id, method='union', 
                            weight_func=None, **attributes):
        """
        Create a new layer by aggregating multiple source layers.

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
        
        if method == 'union':
            result = self.layer_union(source_layer_ids)
        elif method == 'intersection':
            result = self.layer_intersection(source_layer_ids)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        return self.create_layer_from_operation(target_layer_id, result, **attributes)

    def layer_statistics(self, include_default: bool = False):
        """
        Basic per-layer statistics.

        Parameters
        ----------
        include_default : bool, optional

        Returns
        -------
        dict[str, dict]
            ``{layer_id: {'vertices': int, 'edges': int, 'attributes': dict}}``.
        """
        stats = {}
        for layer_id, layer_data in self.layers(include_default=include_default).items():
            stats[layer_id] = {
                'vertices': len(layer_data["vertices"]),
                'edges': len(layer_data["edges"]),
                'attributes': layer_data["attributes"]
            }
        return stats

    # Traversal (neighbors)

    def neighbors(self, entity_id):
        """
        Neighbors of an entity (vertex or edge-entity).

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
                        out |= (meta["tail"])
                    elif entity_id in meta["tail"]:
                        out |= (meta["head"])
                else:
                    if (("members" in meta) and (entity_id in meta["members"])):
                        out |= (meta["members"] - {entity_id})
            else:
                # binary / vertex_edge
                s, t, _ = self.edge_definitions[eid]
                edir = self.edge_directed.get(eid, self.directed)
                if s == entity_id:
                    out.add(t)
                elif t == entity_id and (not edir or self.entity_types.get(entity_id) == 'edge'):
                    out.add(s)
        return list(out)

    def out_neighbors(self, vertex_id):
        """
        Out-neighbors of a vertex under directed semantics.

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
                        out |= (meta["tail"])
                else:
                    if vertex_id in meta.get("members", ()):
                        out |= (meta["members"] - {vertex_id})
            else:
                s, t, _ = self.edge_definitions[eid]
                edir = self.edge_directed.get(eid, self.directed)
                if s == vertex_id:
                    out.add(t)
                elif t == vertex_id and not edir:
                    out.add(s)
        return list(out)

    def successors(self, vertex_id):
        """
        successors of a vertex under directed semantics.

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
                        out |= (meta["tail"])
                else:
                    if vertex_id in meta.get("members", ()):
                        out |= (meta["members"] - {vertex_id})
            else:
                s, t, _ = self.edge_definitions[eid]
                edir = self.edge_directed.get(eid, self.directed)
                if s == vertex_id:
                    out.add(t)
                elif t == vertex_id and not edir:
                    out.add(s)
        return list(out)

    def in_neighbors(self, vertex_id):
        """
        In-neighbors of a vertex under directed semantics.

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
                        inn |= (meta["head"])
                else:
                    if vertex_id in meta.get("members", ()):
                        inn |= (meta["members"] - {vertex_id})
            else:
                s, t, _ = self.edge_definitions[eid]
                edir = self.edge_directed.get(eid, self.directed)
                if t == vertex_id:
                    inn.add(s)
                elif s == vertex_id and not edir:
                    inn.add(t)
        return list(inn)

    def predecessors(self, vertex_id):
        """
        In-neighbors of a vertex under directed semantics.

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
                        inn |= (meta["head"])
                else:
                    if vertex_id in meta.get("members", ()):
                        inn |= (meta["members"] - {vertex_id})
            else:
                s, t, _ = self.edge_definitions[eid]
                edir = self.edge_directed.get(eid, self.directed)
                if t == vertex_id:
                    inn.add(s)
                elif s == vertex_id and not edir:
                    inn.add(t)
        return list(inn)

    # Slicing / copying / accounting

    def edge_subgraph(self, edges) -> "Graph":
        """
        Create a new graph containing only a specified subset of edges.

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
        g = self.copy()
        # Normalize edge IDs
        if all(isinstance(e, int) for e in edges):
            edges = {self.idx_to_edge[e] for e in edges}
        else:
            edges = set(edges)

        # Drop edges not in selection
        for eid in list(g.edge_definitions.keys()):
            if eid not in edges:
                g.remove_edge(eid)

        # Optional: prune isolated vertices
        to_remove = [v for v in g.vertices() if not g.incident_edges(v)]
        for v in to_remove:
            g.remove_vertex(v)

        return g

    def subgraph(self, vertices) -> "Graph":
        """
        Create a vertex-induced subgraph.

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
        g = self.copy()
        V = set(vertices)

        # Drop edges that touch any vertex outside the set
        for j in range(g.number_of_edges()):
            S, T = g.get_edge(j)
            if not (S | T).issubset(V):
                eid = g.idx_to_edge[j]
                g.remove_edge(eid)

        # Drop all vertices not in the set
        for v in list(g.vertices()):
            if v not in V:
                g.remove_vertex(v)

        return g

    def extract_subgraph(self, vertices=None, edges=None) -> "Graph":
        """
        Create a subgraph based on a combination of vertex and/or edge filters.

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
        g = self.copy()
        if vertices is not None:
            g = g.subgraph(vertices)
        if edges is not None:
            g = g.edge_subgraph(edges)
        return g

    def reverse(self) -> "Graph":
        """
        Return a new graph with all directed edges reversed.

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
        """
        Build a new graph restricted to a single layer.

        Parameters
        ----------
        layer_id : str
        resolve_layer_weights : bool, optional
            If True, materialize each edge with its effective per-layer weight.

        Returns
        -------
        Graph
            New graph containing only entities/edges of the layer.

        Raises
        ------
        KeyError
            If the layer does not exist.
        """
        if layer_id not in self._layers:
            raise KeyError(f"Layer {layer_id} not found")

        lg = Graph(directed=self.directed)
        # Create the destination layer and make it active
        lg.add_layer(layer_id, **self.get_layer_info(layer_id)["attributes"])
        lg.set_active_layer(layer_id)

        layer_vertices = self._layers[layer_id]["vertices"]
        layer_edges = self._layers[layer_id]["edges"]

        def _row_attrs(df: pl.DataFrame, key_col: str, key_val, drop_key: str):
            if not isinstance(df, pl.DataFrame) or df.height == 0 or key_col not in df.columns:
                return {}
            rows = df.filter(pl.col(key_col) == key_val)
            if rows.height == 0:
                return {}
            d = rows.to_dicts()[0]
            d.pop(drop_key, None)
            return d

        # 1) bring over entities (vertices + edge-entities) that participate in this layer
        for ent_id in layer_vertices:
            if self.entity_types.get(ent_id) == "vertex":
                attrs = _row_attrs(self.vertex_attributes, "vertex_id", ent_id, "vertex_id")
                lg.add_vertex(ent_id, layer=layer_id, **attrs)
            else:
                # edge-entity (attributes stored in vertex_attributes as well)
                attrs = _row_attrs(self.vertex_attributes, "vertex_id", ent_id, "vertex_id")
                lg.add_edge_entity(ent_id, layer=layer_id, **attrs)

        # 2) bring over edges in this layer (preserve directedness + attrs + effective weight)
        for eid in list(layer_edges):
            # resolve weight for this layer, or fallback to global
            w = (
                self.get_effective_edge_weight(eid, layer=layer_id)
                if resolve_layer_weights
                else self.edge_weights.get(eid, 1.0)
            )
            ed_attrs = _row_attrs(self.edge_attributes, "edge_id", eid, "edge_id")

            kind = self.edge_kind.get(eid, "binary")
            if kind == "hyper":
                hdef = self.hyperedge_definitions[eid]
                # undirected hyperedge uses 'members'; directed uses 'head'/'tail'
                if hdef.get("members"):
                    lg.add_hyperedge(
                        members=set(hdef["members"]),
                        layer=layer_id,
                        weight=float(w),
                        edge_id=eid,
                        **ed_attrs
                    )
                else:
                    head = set(hdef.get("head", []))
                    tail = set(hdef.get("tail", []))
                    lg.add_hyperedge(
                        head=head,
                        tail=tail,
                        layer=layer_id,
                        weight=float(w),
                        edge_id=eid,
                        **ed_attrs
                    )
                continue

            # binary or vertex-edge
            src, tgt, etype = self.edge_definitions[eid]  # etype in {'regular','vertex_edge'}
            edir = self.edge_directed.get(eid, self.directed)
            lg.add_edge(
                src,
                tgt,
                weight=float(w),
                edge_id=eid,         # preserve original id
                edge_type=etype,     # 'regular' or 'vertex_edge'
                edge_directed=edir,
                layer=layer_id,
                **ed_attrs
            )

        # 3) the layer's attributes were set on creation above
        return lg

    def copy(self):
        """
        Deep copy the entire graph, including layers, edges, hyperedges, and attributes.

        Returns
        -------
        Graph
        """
        # local helper (do NOT use self._row_attrs here)
        def _row_attrs(df, key_col: str, key_val, drop_key: str):
            import polars as pl  # safe even if already imported at module level
            if not isinstance(df, pl.DataFrame) or df.height == 0 or key_col not in df.columns:
                return {}
            rows = df.filter(pl.col(key_col) == key_val)
            if rows.height == 0:
                return {}
            d = rows.to_dicts()[0]
            d.pop(drop_key, None)
            return d

        import polars as pl  # ensure available in this scope

        new_graph = Graph(directed=self.directed)

        # ---- Copy layers and their attributes first ----
        for lid in self._layers:
            if lid != new_graph._default_layer:  # 'default' already exists in new_graph
                new_graph.add_layer(lid)
            la = self.get_layer_info(lid)["attributes"]
            if la:
                new_graph.set_layer_attrs(lid, **la)

        # ---- Copy entities (vertices + edge-entities) ----
        for ent_id, etype in self.entity_types.items():
            attrs = _row_attrs(self.vertex_attributes, "vertex_id", ent_id, "vertex_id")
            if etype == "vertex":
                new_graph.add_vertex(ent_id, layer=new_graph._default_layer, **attrs)
            else:
                new_graph.add_edge_entity(ent_id, layer=new_graph._default_layer, **attrs)

        # ---- Copy *binary / vertex-edge* edges (skip hyperedges here) ----
        for edge_id, (source, target, edge_type) in self.edge_definitions.items():
            if edge_type == "hyper":
                continue  # will be handled below

            weight = self.edge_weights.get(edge_id, 1.0)
            edge_attrs = _row_attrs(self.edge_attributes, "edge_id", edge_id, "edge_id")
            edge_dir = self.edge_directed.get(edge_id, self.directed)

            new_graph.add_edge(
                source,
                target,
                weight=weight,
                edge_id=edge_id,
                edge_type=edge_type,    # 'regular' or 'vertex_edge'
                edge_directed=edge_dir,
                **edge_attrs,
            )

        # ---- Copy hyperedges ----
        for eid, hdef in self.hyperedge_definitions.items():
            weight = self.edge_weights.get(eid, 1.0)
            edge_attrs = _row_attrs(self.edge_attributes, "edge_id", eid, "edge_id")

            if hdef.get("members"):  # undirected hyperedge
                new_graph.add_hyperedge(
                    members=set(hdef["members"]),
                    weight=weight,
                    edge_id=eid,
                    **edge_attrs,
                )
            else:  # directed hyperedge
                head = set(hdef.get("head", []))
                tail = set(hdef.get("tail", []))
                new_graph.add_hyperedge(
                    head=head,
                    tail=tail,
                    weight=weight,
                    edge_id=eid,
                    **edge_attrs,
                )

        # ---- Copy layer memberships (vertices & edges) EXACTLY ----
        for lid, meta in self._layers.items():
            # ensure layer exists
            if lid not in new_graph._layers:
                new_graph.add_layer(lid)

            # overwrite (not update) to match exactly
            new_graph._layers[lid]["vertices"] = set(meta["vertices"])
            new_graph._layers[lid]["edges"] = set(meta["edges"])


        # ---- Copy per-layer edge attributes (DF) and legacy dict overrides ----
        if isinstance(self.edge_layer_attributes, pl.DataFrame) and self.edge_layer_attributes.height > 0:
            for row in self.edge_layer_attributes.to_dicts():
                lid = row.get("layer_id")
                eid = row.get("edge_id")
                if lid is not None and eid is not None:
                    r = dict(row)
                    r.pop("layer_id", None); r.pop("edge_id", None)
                    if r:
                        new_graph.set_edge_layer_attrs(lid, eid, **r)

        for lid, m in self.layer_edge_weights.items():
            for eid, w in m.items():
                new_graph.set_layer_edge_weight(lid, eid, w)

        # ---- Directness flags ----
        new_graph.edge_directed.update(self.edge_directed)

        return new_graph

    def memory_usage(self):
        """
        Approximate total memory usage in bytes.

        Returns
        -------
        int
            Estimated bytes for the incidence matrix, dictionaries, and attribute DFs.
        """        
        # Approximate matrix memory: each non-zero entry stores row, col, and value (4 bytes each)
        matrix_bytes = self._matrix.nnz * (4 + 4 + 4)
        # Estimate dict memory: ~100 bytes per entry
        dict_bytes = (len(self.entity_to_idx) + len(self.edge_to_idx) + len(self.edge_weights)) * 100
        
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
        """
        Materialize the vertex–edge incidence structure as Python lists.

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
        """
        Return the vertex–edge incidence matrix in sparse or dense form.

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
        """
        Return a stable hash representing the current graph structure and metadata.

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
        graph_meta = tuple(sorted(self.graph_attributes.items())) if hasattr(self, "graph_attributes") else ()

        return hash((vertex_ids, edge_defs, graph_meta))

    # History and Timeline

    def _utcnow_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")

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
            "ts_utc": self._utcnow_iso(),                    # ISO-8601 with Z
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
            "add_vertex", "add_edge_entity", "add_edge", "add_hyperedge",
            "remove_edge", "remove_vertex",
            "set_vertex_attrs", "set_edge_attrs", "set_layer_attrs", "set_edge_layer_attrs",
            "register_layer", "unregister_layer"
        ]
        for name in to_wrap:
            if hasattr(self, name):
                fn = getattr(self, name)
                # Avoid double-wrapping
                if getattr(fn, "__wrapped__", None) is None:
                    setattr(self, name, self._log_mutation(name)(fn))

    def history(self, as_df: bool = False):
        """
        Return the append-only mutation history.

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
        """
        Write the mutation history to disk.

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
            df.write_parquet(path);  return len(df)
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
            df.write_csv(path); return len(df)
        # Default to Parquet if unknown
        df.write_parquet(path + ".parquet"); return len(df)

    def enable_history(self, flag: bool = True):
        """
        Enable or disable in-memory mutation logging.

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
        """
        Clear the in-memory mutation log.

        Returns
        -------
        None

        Notes
        -----
        This does not delete any files previously exported.
        """
        self._history.clear()

    def mark(self, label: str):
        """
        Insert a manual marker into the mutation history.

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
        """
        Accessor for the lazy NX proxy.
        Usage: G.nx.algorithm(); e.g: G.nx.louvain_communities(G), G.nx.shortest_path_length(G, weight="weight")
        """
        if not hasattr(self, "_nx_proxy"):
            self._nx_proxy = self._LazyNXProxy(self)
        return self._nx_proxy

    class _LazyNXProxy:
        """
        Lazy, cached NX (NetworkX) adapter:
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
                directed=True, hyperedge_mode="skip", layer=None, layers=None, needed_attrs=set(),
                simple=False, edge_aggs=None
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
            hyperedge_mode: str = "skip",
            layer=None,
            layers=None,
            needed_attrs=None,
            simple: bool = False,
            edge_aggs: dict | None = None,
        ):
            """
            Return the underlying NetworkX graph built with the same lazy/cached
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
                directed = bool(kwargs.pop("_nx_directed", True))
                hyperedge_mode = kwargs.pop("_nx_hyperedge", "skip")  # "skip" | "expand"
                layer = kwargs.pop("_nx_layer", None)
                layers = kwargs.pop("_nx_layers", None)
                label_field = kwargs.pop("_nx_label_field", None)      # explicit label column
                guess_labels = kwargs.pop("_nx_guess_labels", True)    # try auto-infer when not provided

                # force simple Graph/DiGraph and aggregation policy for parallel edges
                simple = bool(kwargs.pop("_nx_simple", False))
                edge_aggs = kwargs.pop("_nx_edge_aggs", None)  # e.g. {"weight":"min","capacity":"sum"} or callables

                # Determine required edge attributes (keep graph skinny)
                needed_edge_attrs = self._needed_edge_attrs(nx_callable, kwargs)

                # Acquire (or build) backend NX graph for this config
                nxG = self._get_or_make_nx(
                    directed=directed,
                    hyperedge_mode=hyperedge_mode,
                    layer=layer,
                    layers=layers,
                    needed_attrs=needed_edge_attrs,
                    simple=simple,
                    edge_aggs=edge_aggs,
                )

                # Replace any Graph instance with nxG
                args = list(args)
                for i, v in enumerate(args):
                    if v is self._G:
                        args[i] = nxG
                for k, v in list(kwargs.items()):
                    if v is self._G:
                        kwargs[k] = nxG

                # Bind to NX signature so we can coerce node args reliably
                try:
                    sig = inspect.signature(nx_callable)
                    bound = sig.bind_partial(*args, **kwargs)
                except Exception:
                    bound = None

                # Coerce node args (labels/indices -> vertex IDs)
                try:
                    # Determine default label field if not given
                    if label_field is None and guess_labels:
                        label_field = self._infer_label_field()

                    if bound is not None:
                        self._coerce_nodes_in_bound(bound, nxG, label_field)
                        # Reconstruct call
                        bound.apply_defaults()
                        pargs = bound.args
                        pkwargs = bound.kwargs
                    else:
                        # Fallback: best-effort coercion on kwargs only
                        self._coerce_nodes_in_kwargs(kwargs, nxG, label_field)
                        pargs, pkwargs = tuple(args), kwargs
                except Exception:
                    pargs, pkwargs = tuple(args), kwargs  # best effort; let NX raise if needed

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
                getattr(_nx.algorithms, "approximation", None) if hasattr(_nx, "algorithms") else None,
                getattr(_nx.algorithms, "centrality", None) if hasattr(_nx, "algorithms") else None,
                getattr(_nx.algorithms, "shortest_paths", None) if hasattr(_nx, "algorithms") else None,
                getattr(_nx.algorithms, "flow", None) if hasattr(_nx, "algorithms") else None,
                getattr(_nx.algorithms, "components", None) if hasattr(_nx, "algorithms") else None,
                getattr(_nx.algorithms, "traversal", None) if hasattr(_nx, "algorithms") else None,
                getattr(_nx.algorithms, "bipartite", None) if hasattr(_nx, "algorithms") else None,
                getattr(_nx.algorithms, "link_analysis", None) if hasattr(_nx, "algorithms") else None,
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

        def _convert_to_nx(self, *, directed: bool, hyperedge_mode: str, layer, layers,
                           needed_attrs: set, simple: bool, edge_aggs: dict | None):
            from ..adapters import networkx as _gg_nx  # graphglue.adapters.networkx
            import networkx as _nx

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
                nxG = self._collapse_multiedges(nxG, directed=directed,
                                                aggregations=edge_aggs, needed_attrs=needed_attrs)

            self._warn_on_loss(hyperedge_mode=hyperedge_mode, layer=layer, layers=layers, manifest=manifest)
            return nxG

        def _get_or_make_nx(self, *, directed: bool, hyperedge_mode: str, layer, layers,
                            needed_attrs: set, simple: bool, edge_aggs: dict | None):
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
            if (not self.cache_enabled) or (entry is None) or (version is not None and entry.get("version") != version):
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
                if isinstance(layers_dict, dict) and len(layers_dict) > 1 and (layer is None and not layers):
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
            """
            Heuristic label column if user didn't specify:
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
                    bound.arguments[key] = self._coerce_node_or_iter(bound.arguments[key], nxG, label_field)

        # ---------------------- Multi* collapse helpers -------------------------
        def _collapse_multiedges(self, nxG, *, directed: bool,
                                 aggregations: dict | None, needed_attrs: set):
            """
            Collapse parallel edges into a single edge with aggregated attributes.
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
        """
        Accessor for the lazy igraph proxy.
        Usage: G.ig.community_multilevel(G, weights="weight"), G.ig.shortest_paths_dijkstra(G, source="a", target="z", weights="weight")
        (same idea as NX: pass G; proxy swaps it with the backend igraph.Graph lazily)
        """
        if not hasattr(self, "_ig_proxy"):
            self._ig_proxy = self._LazyIGProxy(self)
        return self._ig_proxy

    class _LazyIGProxy:
        """
        Lazy, cached igraph adapter:
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
            self._cache = {}   # key -> {"igG": ig.Graph, "version": int}
            self.cache_enabled = True

        # ---------------------------- public API --------------------------------
        def clear(self):
            self._cache.clear()

        def peek_vertices(self, k: int = 10):
            igG = self._get_or_make_ig(directed=True, hyperedge_mode="skip",
                                       layer=None, layers=None, needed_attrs=set(),
                                       simple=True, edge_aggs=None)
            out = []
            names = igG.vs["name"] if "name" in igG.vs.attributes() else None
            for i in range(min(max(0, int(k)), igG.vcount())):
                out.append(names[i] if names else i)
            return out

        # public helper so tests don’t touch private API
        def backend(self, *,
                    directed: bool = True,
                    hyperedge_mode: str = "skip",
                    layer=None,
                    layers=None,
                    needed_attrs=None,
                    simple: bool = False,
                    edge_aggs: dict | None = None):
            needed_attrs = needed_attrs or set()
            return self._get_or_make_ig(directed=directed, hyperedge_mode=hyperedge_mode,
                                        layer=layer, layers=layers, needed_attrs=needed_attrs,
                                        simple=simple, edge_aggs=edge_aggs)

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
                edge_aggs = kwargs.pop("_ig_edge_aggs", None)  # {"weight":"min","capacity":"sum"} or callables

                # keep only attributes actually needed by the called function
                needed_edge_attrs = self._needed_edge_attrs_for_ig(name, kwargs)

                # build/reuse backend
                igG = self._get_or_make_ig(directed=directed, hyperedge_mode=hyperedge_mode,
                                           layer=layer, layers=layers, needed_attrs=needed_edge_attrs,
                                           simple=simple, edge_aggs=edge_aggs)

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
                    raise AttributeError(f"igraph has no callable '{name}'. "
                                         f"Use native igraph names, e.g. community_multilevel, pagerank, shortest_paths_dijkstra, components, etc.")

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
            """
            Heuristic: igraph uses `weights` (plural) for edge weights in most algos,
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

        def _convert_to_ig(self, *, directed: bool, hyperedge_mode: str, layer, layers,
                           needed_attrs: set, simple: bool, edge_aggs: dict | None):
            # try both adapter entry points: to_ig / to_igraph
            from ..adapters import igraph as _gg_ig  # graphglue.adapters.igraph
            import igraph as _ig

            conv = None
            for cand in ("to_ig", "to_igraph"):
                conv = getattr(_gg_ig, cand, None) or conv
            if conv is None:
                raise RuntimeError("igraph adapter missing: expected adapters.igraph.to_ig(...) or .to_igraph(...).")

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
                igG = self._collapse_multiedges(igG, directed=directed,
                                                aggregations=edge_aggs, needed_attrs=needed_attrs)


            self._warn_on_loss(hyperedge_mode=hyperedge_mode, layer=layer, layers=layers, manifest=manifest)
            return igG

        def _get_or_make_ig(self, *, directed: bool, hyperedge_mode: str, layer, layers,
                            needed_attrs: set, simple: bool, edge_aggs: dict | None):
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
            if (not self.cache_enabled) or (entry is None) or (version is not None and entry.get("version") != version):
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
                if isinstance(layers_dict, dict) and len(layers_dict) > 1 and (layer is None and not layers):
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
                "source", "target", "u", "v", "vertex", "vertices", "vs", "to", "fr",
                "root", "roots", "neighbors", "nbunch", "path", "cut",
            }
            for key in list(kwargs.keys()):
                if key in node_keys:
                    kwargs[key] = self._coerce_node_or_iter(kwargs[key], igG, label_field)

        def _coerce_nodes_in_bound(self, bound, igG, label_field: str | None):
            node_keys = {
                "source", "target", "u", "v", "vertex", "vertices", "vs", "to", "fr",
                "root", "roots", "neighbors", "nbunch", "path", "cut",
            }
            for key in list(bound.arguments.keys()):
                if key in node_keys:
                    bound.arguments[key] = self._coerce_node_or_iter(bound.arguments[key], igG, label_field)

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

        def _collapse_multiedges(self, igG, *, directed: bool,
                                 aggregations: dict | None, needed_attrs: set):
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
                if agg == "sum": return sum
                if agg == "min": return min
                if agg == "max": return max
                if agg == "mean": 
                    return lambda vals: (sum(vals) / len(vals)) if vals else None
                if key == "capacity": return sum
                if key == "weight": return min
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