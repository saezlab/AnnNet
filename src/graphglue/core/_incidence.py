import numpy as np
import scipy.sparse as sp
import pandas as pd
from collections import defaultdict

class IncidenceGraph:
    """
    Graph implementation using sparse incidence matrix representation.
    Rows = nodes + node-edge hybrid entities
    Columns = edges (including parallel edges)
    
    Matrix[i,j] = +weight if node i is source of edge j
    Matrix[i,j] = -weight if node i is target of edge j  
    Matrix[i,j] = 0 if node i not incident to edge j
    """
    
    def __init__(self, directed=True):
        self.directed = directed
        
        # Entity mappings (nodes + node-edge hybrids)
        self.entity_to_idx = {}  # entity_id -> row index
        self.idx_to_entity = {}  # row index -> entity_id
        self.entity_types = {}   # entity_id -> 'node' or 'edge'
        
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
        self.node_attributes = pd.DataFrame()  # Using pandas for now, can switch to polars
        self.edge_attributes = pd.DataFrame()
        self.layer_attributes = pd.DataFrame()
        self.graph_attributes = {}
        self.edge_layer_attributes = (pd.DataFrame(columns=["weight"]).set_index(pd.MultiIndex.from_tuples([], names=["layer_id", "edge_id"])))
        self.edge_layer_attributes.index = pd.MultiIndex.from_tuples([], names=["layer_id", "edge_id"])
        self.edge_kind = {}
        self.hyperedge_definitions = {}

        # Make attribute tables indexed by ID (no duplicates)
        self.node_attributes.index.name = "node_id"
        self.edge_attributes.index.name = "edge_id"
        self.layer_attributes.index.name = "layer_id"
        
        # Edge ID counter for parallel edges
        self._next_edge_id = 0

        # Layer management - lightweight dict structure
        self._layers = {}  # layer_id -> {"nodes": set(), "edges": set(), "attributes": {}}
        self._current_layer = None
        self._default_layer = 'default'
        self.layer_edge_weights = defaultdict(dict)  # layer_id -> {edge_id: weight}

        # Initialize default layer
        self._layers[self._default_layer] = {
            "nodes": set(),
            "edges": set(), 
            "attributes": {}
        }
        self._current_layer = self._default_layer
 
    def _get_next_edge_id(self) -> str:
        """Generate unique edge ID for parallel edges."""
        edge_id = f"edge_{self._next_edge_id}"
        self._next_edge_id += 1
        return edge_id
    
    def add_node(self, node_id, layer=None, **attributes):
        layer = layer or self._current_layer
        
        # Add to global superset if new
        if node_id not in self.entity_to_idx:
            idx = self._num_entities
            self.entity_to_idx[node_id] = idx
            self.idx_to_entity[idx] = node_id
            self.entity_types[node_id] = 'node'
            self._num_entities += 1
            
            # Resize incidence matrix
            self._matrix.resize((self._num_entities, self._num_edges))
        
        # Add to specified layer
        if layer not in self._layers:
            self._layers[layer] = {"nodes": set(), "edges": set(), "attributes": {}}
        
        self._layers[layer]["nodes"].add(node_id)
        
        # Add attributes
        if attributes:
            self.set_node_attrs(node_id, **attributes)

    
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
            # validate inputs
            if propagate not in {"none", "shared", "all"}:
                raise ValueError(f"propagate must be one of 'none'|'shared'|'all', got {propagate!r}")
            if not isinstance(weight, (int, float)):
                raise TypeError(f"weight must be numeric, got {type(weight).__name__}")
            if edge_type not in {"regular", "node_edge"}:
                raise ValueError(f"edge_type must be 'regular' or 'node_edge', got {edge_type!r}")

            # resolve layer + whether to touch layering at all
            layer = self._current_layer if layer is None else layer
            touch_layer = layer is not None

            # ensure nodes exist (global)
            def _ensure_node_or_edge_entity(x):
                if x in self.entity_to_idx:
                    return
                if edge_type == "node_edge" and isinstance(x, str) and x.startswith("edge_"):
                    self.add_edge_entity(x, layer=layer)
                else:
                    self.add_node(x, layer=layer)

            _ensure_node_or_edge_entity(source)
            _ensure_node_or_edge_entity(target)

            # indices (after potential node creation)
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

                # if source/target changed, update definition
                old_src, old_tgt, old_type = self.edge_definitions[edge_id]
                self.edge_definitions[edge_id] = (source, target, old_type)  # keep old_type by design

                # ensure matrix has enough rows (in case nodes were added since creation)
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
                    self._layers[layer] = {"nodes": set(), "edges": set(), "attributes": {}}
                self._layers[layer]["edges"].add(edge_id)
                self._layers[layer]["nodes"].update((source, target))

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

    def add_edge_entity(self, edge_entity_id, layer=None, **attributes):
        """
        Explicitly add an edge as an entity that can be connected to other nodes/edges.
        
        Args:
            edge_entity_id: ID for the edge entity
            layer: layer to add it to
            **attributes: attributes for the edge entity
        """
        layer = layer or self._current_layer
        
        # Add to global superset if new
        if edge_entity_id not in self.entity_to_idx:
            self._add_edge_entity(edge_entity_id)
        
        # Add to specified layer
        if layer not in self._layers:
            self._layers[layer] = {"nodes": set(), "edges": set(), "attributes": {}}
        
        self._layers[layer]["nodes"].add(edge_entity_id)
        
        # Add attributes (treat edge entities like nodes for attributes)
        if attributes:
            self.set_node_attrs(edge_entity_id, **attributes)

        return edge_entity_id
    
    def _add_edge_entity(self, edge_id):
        """Add an edge as an entity (for node-edge hybrid connections)."""
        if edge_id not in self.entity_to_idx:
            idx = self._num_entities
            self.entity_to_idx[edge_id] = idx
            self.idx_to_entity[idx] = edge_id
            self.entity_types[edge_id] = 'edge'
            self._num_entities += 1
            
            # Resize matrix
            self._matrix.resize((self._num_entities, self._num_edges))
    
    def add_parallel_edge(self, source, target, weight=1.0, **attributes):
        """Add a parallel edge (same nodes, different edge ID)."""
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
        Create a k-ary hyperedge as a single incidence-matrix column.
        - Undirected: pass members (>=2).
        - Directed: pass head and tail (both non-empty, disjoint).
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

        # ---- set layer ----
        layer = self._current_layer if layer is None else layer

        # ---- ensure participants exist globally (nodes or edge-entities already supported) ----
        def _ensure_entity(x):
            if x in self.entity_to_idx:
                return
            # hyperedges connect to nodes/edge-entities similarly to binary edges
            if isinstance(x, str) and x.startswith("edge_") and x in self.entity_types and self.entity_types[x] == "edge":
                # already an edge-entity
                return
            # default: treat as node
            self.add_node(x, layer=layer)

        if members is not None:
            for u in members:
                _ensure_entity(u)
        else:
            for u in head + tail:
                _ensure_entity(u)

        # ---- allocate edge id + column ----
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

        # ---- write column entries ----
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

        # ---- bookkeeping shared with binary edges ----
        self.edge_weights[edge_id] = float(weight)
        self.edge_directed[edge_id] = bool(directed)
        self.edge_kind[edge_id] = "hyper"

        # keep a sentinel in edge_definitions so old code won't crash
        self.edge_definitions[edge_id] = (None, None, "hyper")

        # layer membership + per-layer weights
        if layer is not None:
            if layer not in self._layers:
                self._layers[layer] = {"nodes": set(), "edges": set(), "attributes": {}}
            self._layers[layer]["edges"].add(edge_id)
            if members is not None:
                self._layers[layer]["nodes"].update(members)
            else:
                self._layers[layer]["nodes"].update(self.hyperedge_definitions[edge_id]["head"])
                self._layers[layer]["nodes"].update(self.hyperedge_definitions[edge_id]["tail"])

        # attributes (pure, non-structural)
        if attributes:
            self.set_edge_attrs(edge_id, **attributes)

        return edge_id

    def remove_edge(self, edge_id):
        """Remove an edge from the graph."""
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
        if not self.edge_attributes.empty and edge_id in self.edge_attributes.index:
                self.edge_attributes.drop(index=edge_id, inplace=True, errors="ignore")
        for layer_data in self._layers.values():
            layer_data["edges"].discard(edge_id)

        if not self.edge_layer_attributes.empty:
            mask = self.edge_layer_attributes.index.get_level_values("edge_id") == edge_id
            self.edge_layer_attributes = self.edge_layer_attributes.loc[~mask]
        # also clear in legacy dict
        for d in self.layer_edge_weights.values():
            d.pop(edge_id, None)
        self.edge_kind.pop(edge_id, None)
        self.hyperedge_definitions.pop(edge_id, None)

    def remove_node(self, node_id):
        """Remove a node and all incident edges."""
        if node_id not in self.entity_to_idx:
            raise KeyError(f"Node {node_id} not found")
        
        entity_idx = self.entity_to_idx[node_id]
        
        # Find edges incident to this entity
        edges_to_remove = []
        for edge_id, (source, target, _) in self.edge_definitions.items():
            if source == node_id or target == node_id:
                edges_to_remove.append(edge_id)
        
        # Remove edges
        for edge_id in edges_to_remove:
            self.remove_edge(edge_id)
        
        # Convert to CSR for efficient row removal
        csr_matrix = self._matrix.tocsr()
        
        # Remove entity row from matrix
        mask = np.ones(self._num_entities, dtype=bool)
        mask[entity_idx] = False
        csr_matrix = csr_matrix[mask, :]
        self._matrix = csr_matrix.todok()
        
        # Update entity mappings
        del self.entity_to_idx[node_id]
        del self.entity_types[node_id]
        
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
        
        # Remove from node attributes
        if not self.node_attributes.empty and node_id in self.node_attributes.index:
            self.node_attributes.drop(index=node_id, inplace=True, errors="ignore")
    
        for layer_data in self._layers.values():
            layer_data["nodes"].discard(node_id)
      
    def set_active_layer(self, layer_id):
        """Set the active layer for operations."""
        if layer_id not in self._layers:
            raise KeyError(f"Layer {layer_id} not found")
        self._current_layer = layer_id

    def get_active_layer(self):
        """Get the currently active layer ID."""
        return self._current_layer
    
    def layers(self, include_default: bool = False):
        """
        Return dict of layers. Excludes the internal 'default' layer unless flagged.
        """
        if include_default:
            return self._layers
        return {k: v for k, v in self._layers.items() if k != self._default_layer}

    def list_layers(self, include_default: bool = False):
        """List layer IDs, excluding 'default' unless flagged."""
        return list(self.layers(include_default=include_default).keys())

    
    def has_layer(self, layer_id):
        """Check if layer exists."""
        return layer_id in self._layers
    
    def layer_count(self):
        """Get number of layers."""
        return len(self._layers)

    def has_edge(self, source, target, edge_id=None):
        """Check if edge exists. If edge_id specified, check that specific edge."""
        if edge_id:
            return edge_id in self.edge_to_idx
        
        # Check any edge between source and target
        for eid, (src, tgt, _) in self.edge_definitions.items():
            if src == source and tgt == target:
                return True
        return False
    
    def get_edge_ids(self, source, target):
        """Get all edge IDs between two nodes (for parallel edges)."""
        edge_ids = []
        for eid, (src, tgt, _) in self.edge_definitions.items():
            if src == source and tgt == target:
                edge_ids.append(eid)
        return edge_ids
    
    def degree(self, entity_id):
        """Get degree of a node or edge entity."""
        if entity_id not in self.entity_to_idx:
            return 0
        
        entity_idx = self.entity_to_idx[entity_id]
        row = self._matrix.getrow(entity_idx)
        return len(row.nonzero()[1])
    
    def nodes(self):
        """Get all node IDs (excluding edge entities)."""
        return [eid for eid, etype in self.entity_types.items() if etype == 'node']
    
    def edges(self):
        """Get all edge IDs."""
        return list(self.edge_to_idx.keys())
    
    def edge_list(self):
        """Get list of (source, target, edge_id, weight) tuples."""
        edges = []
        for edge_id, (source, target, edge_type) in self.edge_definitions.items():
            weight = self.edge_weights[edge_id]
            edges.append((source, target, edge_id, weight))
        return edges
    
    def number_of_nodes(self):
        """Get number of nodes (excluding edge entities)."""
        return len([e for e in self.entity_types.values() if e == 'node'])
    
    def number_of_edges(self):
        """Get number of edges."""
        return self._num_edges

    def copy(self):
        """Create a deep copy of the graph."""
        new_graph = IncidenceGraph(directed=self.directed)
        
        # Copy graph attributes
        new_graph.graph_attributes = self.graph_attributes.copy()
        
        # Copy all entities (nodes + edge entities) with attributes
        for entity_id, entity_type in self.entity_types.items():
            if entity_type == 'node':
                node_attrs = {}
                if (not self.node_attributes.empty) and (entity_id in self.node_attributes.index):
                    node_attrs = self.node_attributes.loc[entity_id].to_dict()
                new_graph.add_node(entity_id, **node_attrs)
            elif entity_type == 'edge':
                # Copy edge entities
                new_graph.add_edge_entity(entity_id)
        
        # Copy all edges with attributes
        for edge_id, (source, target, edge_type) in self.edge_definitions.items():
            weight = self.edge_weights[edge_id]
            edge_attrs = {}
            if (not self.edge_attributes.empty) and (edge_id in self.edge_attributes.index):
                edge_attrs = self.edge_attributes.loc[edge_id].to_dict()
            edge_dir = self.edge_directed.get(edge_id, self.directed)
            new_graph.add_edge(source, target, weight=weight, edge_id=edge_id, 
                            edge_type=edge_type, edge_directed=edge_dir, **edge_attrs)
        
        for eid, meta in self.hyperedge_definitions.items():
            w = self.edge_weights[eid]
            edir = self.edge_directed.get(eid, self.directed)
            ed_attrs = {}
            if (not self.edge_attributes.empty) and (eid in self.edge_attributes.index):
                ed_attrs = self.edge_attributes.loc[eid].to_dict()
            if meta["directed"]:
                new_graph.add_hyperedge(head=meta["head"], tail=meta["tail"],
                                        weight=w, edge_id=eid, edge_directed=edir, **ed_attrs)
            else:
                new_graph.add_hyperedge(members=meta["members"],
                                        weight=w, edge_id=eid, edge_directed=False, **ed_attrs)

        # Copy layer structure
        for layer_id, layer_data in self._layers.items():
            if layer_id != self._default_layer:  # Skip default layer (already exists)
                new_graph.add_layer(layer_id, **layer_data["attributes"])
            
            # Copy layer membership
            for node_id in layer_data["nodes"]:
                new_graph._layers[layer_id]["nodes"].add(node_id)
            for edge_id in layer_data["edges"]:
                new_graph._layers[layer_id]["edges"].add(edge_id)
        
        new_graph.layer_attributes = self.layer_attributes.copy(deep=True)
        new_graph.edge_layer_attributes = self.edge_layer_attributes.copy(deep=True)
        # legacy dict (until fully removed)
        new_graph.layer_edge_weights = defaultdict(dict, {
            k: dict(v) for k, v in self.layer_edge_weights.items()
        })

        # Set current layer
        new_graph._current_layer = self._current_layer
        return new_graph

    def memory_usage(self):
        """Estimate memory usage in bytes."""
        matrix_bytes = self._matrix.nnz * (4 + 4 + 4)  # data + row_ind + col_ind
        dict_bytes = (len(self.entity_to_idx) + len(self.edge_to_idx) + len(self.edge_weights)) * 100
        df_bytes = self.node_attributes.memory_usage(deep=True).sum() if not self.node_attributes.empty else 0
        df_bytes += self.edge_attributes.memory_usage(deep=True).sum() if not self.edge_attributes.empty else 0
        return matrix_bytes + dict_bytes + df_bytes
    
    def get_node_attribute(self, node_id, attribute):
        """Get specific node attribute."""
        try:
            return self.node_attributes.loc[node_id, attribute]
        except Exception:
            return None
    
    def get_edge_attribute(self, edge_id, attribute):
        """Get specific edge attribute."""
        try:
            return self.edge_attributes.loc[edge_id, attribute]
        except Exception:
            return None

    def set_graph_attribute(self, key, value):
        """Set graph-level attribute."""
        self.graph_attributes[key] = value
    
    def get_graph_attribute(self, key, default=None):
        """Get graph-level attribute."""
        return self.graph_attributes.get(key, default)

    def _propagate_to_shared_layers(self, edge_id, source, target):
        """Add edge to layers where both source and target exist."""
        for layer_id, layer_data in self._layers.items():
            if source in layer_data["nodes"] and target in layer_data["nodes"]:
                layer_data["edges"].add(edge_id)

    def _propagate_to_all_layers(self, edge_id, source, target):
        """Add edge to all layers containing either source or target."""
        for layer_id, layer_data in self._layers.items():
            if source in layer_data["nodes"] or target in layer_data["nodes"]:
                layer_data["edges"].add(edge_id)
                # Only add missing endpoint if both nodes should be in layer
                if source in layer_data["nodes"]:
                    layer_data["nodes"].add(target)
                if target in layer_data["nodes"]:
                    layer_data["nodes"].add(source)

    def add_layer(self, layer_id, **attributes):
        """Create new empty layer."""
        if layer_id in self._layers:
            raise ValueError(f"Layer {layer_id} already exists")
        
        self._layers[layer_id] = {
            "nodes": set(),
            "edges": set(),
            "attributes": attributes
        }
        # Persist layer metadata to DF (pure attributes, upsert)
        if attributes:
            self.set_layer_attrs(layer_id, **attributes)
        return layer_id

    def get_layer_nodes(self, layer_id):
        """Get nodes in specified layer."""
        return self._layers[layer_id]["nodes"].copy()

    def get_layer_edges(self, layer_id):
        """Get edges in specified layer."""
        return self._layers[layer_id]["edges"].copy()

    def layer_union(self, layer_ids):
        """
        Union of multiple layers - returns dict with combined nodes/edges.
        Returns: {"nodes": set(), "edges": set()}
        """
        if not layer_ids:
            return {"nodes": set(), "edges": set()}
        
        union_nodes = set()
        union_edges = set()
        
        for layer_id in layer_ids:
            if layer_id in self._layers:
                union_nodes.update(self._layers[layer_id]["nodes"])
                union_edges.update(self._layers[layer_id]["edges"])
        
        return {"nodes": union_nodes, "edges": union_edges}

    def layer_intersection(self, layer_ids):
        """
        Intersection of multiple layers - returns dict with common nodes/edges.
        Returns: {"nodes": set(), "edges": set()}
        """
        if not layer_ids:
            return {"nodes": set(), "edges": set()}
        
        if len(layer_ids) == 1:
            layer_id = layer_ids[0]
            return {
                "nodes": self._layers[layer_id]["nodes"].copy(),
                "edges": self._layers[layer_id]["edges"].copy()
            }
        
        # Start with first layer
        common_nodes = self._layers[layer_ids[0]]["nodes"].copy()
        common_edges = self._layers[layer_ids[0]]["edges"].copy()
        
        # Intersect with remaining layers
        for layer_id in layer_ids[1:]:
            if layer_id in self._layers:
                common_nodes &= self._layers[layer_id]["nodes"]
                common_edges &= self._layers[layer_id]["edges"]
            else:
                # Layer doesn't exist, intersection is empty
                return {"nodes": set(), "edges": set()}
        
        return {"nodes": common_nodes, "edges": common_edges}

    def layer_difference(self, layer1_id, layer2_id):
        """
        Difference: elements in layer1 but not in layer2.
        Returns: {"nodes": set(), "edges": set()}
        """
        if layer1_id not in self._layers or layer2_id not in self._layers:
            raise KeyError("One or both layers not found")
        
        layer1 = self._layers[layer1_id]
        layer2 = self._layers[layer2_id]
        
        return {
            "nodes": layer1["nodes"] - layer2["nodes"],
            "edges": layer1["edges"] - layer2["edges"]
        }

    def create_layer_from_operation(self, result_layer_id, operation_result, **attributes):
        """
        Create new layer from operation result.
        
        Args:
            result_layer_id: ID for new layer
            operation_result: dict from layer_union/intersection/difference
            attributes: layer attributes
        """
        if result_layer_id in self._layers:
            raise ValueError(f"Layer {result_layer_id} already exists")
        
        self._layers[result_layer_id] = {
            "nodes": operation_result["nodes"].copy(),
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
        Find where an edge exists across layers.

        Modes:
        - If edge_id is given: return [layer_id,...] containing that edge (any kind).
        - If (source,target) is given: match ONLY binary/node_edge edges with exactly those endpoints.
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
        Return {layer_id: [edge_id,...]} where a hyperedge with exactly these sets exists.
        - Undirected: pass members (set equality).
        - Directed: pass head and tail (set equality on each).
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


    def node_presence_across_layers(self, node_id, include_default: bool = False):
        """
        Check which layers contain a specific node.
        Returns: list of layer_ids containing the node
        """
        layers_with_node = []
        for layer_id, layer_data in self.layers(include_default=include_default).items():
            if node_id in layer_data["nodes"]:
                layers_with_node.append(layer_id)
        return layers_with_node

    def conserved_edges(self, min_layers=2, include_default=False):
        """
        Edges present in at least `min_layers` *real* layers.
        Returns: dict {edge_id: count}. Excludes 'default' unless include_default=True.
        """
        layers_to_check = self.layers(include_default=include_default)  # hides 'default' by default
        edge_counts = {}
        for _, layer_data in layers_to_check.items():
            for eid in layer_data["edges"]:
                edge_counts[eid] = edge_counts.get(eid, 0) + 1
        return {eid: c for eid, c in edge_counts.items() if c >= min_layers}


    def layer_specific_edges(self, layer_id):
        """
        Find edges that exist only in the specified layer.
        Returns: set of edge_ids
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
        Analyze changes across ordered layers (e.g., time series).
        
        Args:
            ordered_layers: list of layer_ids in temporal order
            metric: 'edge_change', 'node_change'
        
        Returns: list of change metrics between consecutive layers
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
            
            elif metric == 'node_change':
                added = len(next_data["nodes"] - current_data["nodes"])
                removed = len(current_data["nodes"] - next_data["nodes"])
                changes.append({'added': added, 'removed': removed, 'net_change': added - removed})
        
        return changes

    def create_aggregated_layer(self, source_layer_ids, target_layer_id, method='union', 
                            weight_func=None, **attributes):
        """
        Create new layer by aggregating multiple source layers.
        
        Args:
            source_layer_ids: list of layer IDs to aggregate
            target_layer_id: ID for new aggregated layer
            method: 'union', 'intersection'
            weight_func: function to combine edge weights (future use)
            attributes: attributes for new layer
        
        Returns: target_layer_id
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
        """Get statistics for each layer."""
        stats = {}
        for layer_id, layer_data in self.layers(include_default=include_default).items():
            stats[layer_id] = {
                'nodes': len(layer_data["nodes"]),
                'edges': len(layer_data["edges"]),
                'attributes': layer_data["attributes"]
            }
        return stats

    def global_entity_count(self):
        """Count unique entities across all layers."""
        all_nodes = set()
        for layer_data in self._layers.values():
            all_nodes.update(layer_data["nodes"])
        return len(all_nodes)

    def global_edge_count(self):
        """Count unique edges across all layers."""
        all_edges = set()
        for layer_data in self._layers.values():
            all_edges.update(layer_data["edges"])
        return len(all_edges)

    def remove_layer(self, layer_id):
        if layer_id == self._default_layer:
            raise ValueError("Cannot remove default layer")
        if layer_id not in self._layers:
            raise KeyError(f"Layer {layer_id} not found")
        # purge per-layer attrs first
        if not self.edge_layer_attributes.empty:
            mask = self.edge_layer_attributes.index.get_level_values("layer_id") == layer_id
            self.edge_layer_attributes = self.edge_layer_attributes.loc[~mask]
        # drop legacy dict slice
        self.layer_edge_weights.pop(layer_id, None)
        # now remove the layer and reset current if needed
        del self._layers[layer_id]
        if self._current_layer == layer_id:
            self._current_layer = self._default_layer


    def get_layer_info(self, layer_id):
        """Get layer information."""
        if layer_id not in self._layers:
            raise KeyError(f"Layer {layer_id} not found")
        return self._layers[layer_id].copy()

    def get_effective_edge_weight(self, edge_id, layer=None):
        """
        If layer is None: return the global (edge_id) weight.
        If layer is given: return the layer-specific override if present,
                       else the global weight.
        """
        if layer is not None:
            try:
                w = self.edge_layer_attributes.loc[(layer, edge_id), "weight"]
                if pd.notna(w):
                    return float(w)
            except Exception:
                # fallback to legacy dict if present
                w2 = self.layer_edge_weights.get(layer, {}).get(edge_id, None)
                if w2 is not None:
                    return float(w2)
        return float(self.edge_weights[edge_id])

    def set_layer_edge_weight(self, layer_id, edge_id, weight):
        if layer_id not in self._layers:
            raise KeyError(f"Layer {layer_id} not found")
        if edge_id not in self.edge_to_idx:
            raise KeyError(f"Edge {edge_id} not found")
        self.layer_edge_weights[layer_id][edge_id] = float(weight)

    def is_edge_directed(self, edge_id):
        return bool(self.edge_directed.get(edge_id, self.directed))

    # ---- Attribute helpers ----
    _NODE_RESERVED = {"node_id"}               # nothing structural for nodes (for now)
    _EDGE_RESERVED = {"edge_id","source","target","weight","edge_type","directed","layer","layer_weight","kind","members","head","tail"}    
    _LAYER_RESERVED = {"layer_id"}

    def _upsert_row(self, df: pd.DataFrame, idx, attrs: dict) -> None:
        """
        In-place upsert into a DataFrame by index (supports MultiIndex).
        - Creates missing columns on the fly
        - Validates index shape
        - No-op on empty attrs
        """
        if not isinstance(attrs, dict) or not attrs:
            return

        # Validate index shape for MultiIndex
        if isinstance(df.index, pd.MultiIndex):
            if not isinstance(idx, tuple) or len(idx) != df.index.nlevels:
                raise ValueError(
                    f"Index tuple {idx!r} must match MultiIndex nlevels={df.index.nlevels} "
                    f"with names={df.index.names}"
                )

        # Ensure columns exist before assignment
        missing_cols = [k for k in attrs.keys() if k not in df.columns]
        if missing_cols:
            # create missing columns with NaN
            for col in missing_cols:
                df[col] = pd.NA

        # Upsert
        if idx in df.index:
            for k, v in attrs.items():
                df.loc[idx, k] = v
        else:
            # build a full row dict with existing columns set to NA, then overlay attrs
            row = {c: pd.NA for c in df.columns}
            row.update(attrs)
            df.loc[idx] = row


    def set_node_attrs(self, node_id, **attrs):
        clean = {k: v for k, v in attrs.items() if k not in self._NODE_RESERVED}
        if clean:
            self._upsert_row(self.node_attributes, node_id, clean)

    def set_edge_attrs(self, edge_id, **attrs):
        # keep attributes table pure: strip structural keys
        clean = {k: v for k, v in attrs.items() if k not in self._EDGE_RESERVED}
        if clean:
            self._upsert_row(self.edge_attributes, edge_id, clean)

    def set_layer_attrs(self, layer_id, **attrs):
        clean = {k: v for k, v in attrs.items() if k not in self._LAYER_RESERVED}
        if clean:
            self._upsert_row(self.layer_attributes, layer_id, clean)

    def get_node_attr(self, node_id, key, default=None):
        try:
            return self.node_attributes.loc[node_id, key]
        except Exception:
            return default

    def get_edge_attr(self, edge_id, key, default=None):
        try:
            return self.edge_attributes.loc[edge_id, key]
        except Exception:
            return default

    def get_layer_attr(self, layer_id, key, default=None):
        try:
            return self.layer_attributes.loc[layer_id, key]
        except Exception:
            return default 
        
    # attribute views
    def edges_view(self, layer=None, include_directed=True, include_weight=True, resolved_weight=True, copy=True):
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
                row["edge_type"] = etype  # 'regular' | 'node_edge' | None

            rows.append(row)

        base = pd.DataFrame(rows).set_index("edge_id")

        out = base.join(self.edge_attributes, how="left") if not self.edge_attributes.empty else base

        if layer is not None and not self.edge_layer_attributes.empty:
            try:
                layer_slice = self.edge_layer_attributes.xs(layer, level="layer_id", drop_level=True).add_prefix("layer_")
                out = out.join(layer_slice, how="left")
            except KeyError:
                pass

        return out.copy() if copy else out

    def nodes_view(self, copy=True):
        """
        Read-only table: node_id + pure attributes.
        """
        if self.node_attributes.empty:
            return pd.DataFrame(index=pd.Index([], name="node_id"))
        return self.node_attributes.copy() if copy else self.node_attributes

    def layers_view(self, copy=True):
        """
        Read-only table: layer_id + pure attributes.
        """
        if self.layer_attributes.empty:
            return pd.DataFrame(index=pd.Index([], name="layer_id"))
        return self.layer_attributes.copy() if copy else self.layer_attributes

    def set_edge_layer_attrs(self, layer_id, edge_id, **attrs):
        # keep attributes table pure: strip structural keys
        clean = {k: v for k, v in attrs.items() if k not in self._EDGE_RESERVED}
        if not clean:
            return
        # ensure the table exists with MultiIndex
        if not isinstance(self.edge_layer_attributes.index, pd.MultiIndex):
            self.edge_layer_attributes = pd.DataFrame(
                columns=["weight", "directed"]
            ).set_index(pd.MultiIndex.from_tuples([], names=["layer_id", "edge_id"]))
        # upsert
        self._upsert_row(self.edge_layer_attributes, (layer_id, edge_id), clean)



    def get_edge_layer_attr(self, layer_id, edge_id, key, default=None):
        if key not in self.edge_layer_attributes.columns:
            return default
        try:
            return self.edge_layer_attributes.loc[(layer_id, edge_id), key]
        except KeyError:
            return default
    
    def audit_attributes(self):
        node_ids = {eid for eid, t in self.entity_types.items() if t == 'node'}
        edge_ids = set(self.edge_to_idx.keys())

        extra_node_rows = [i for i in self.node_attributes.index if i not in node_ids]
        extra_edge_rows = [i for i in self.edge_attributes.index if i not in edge_ids]
        missing_node_rows = [i for i in node_ids if i not in self.node_attributes.index]
        missing_edge_rows = [i for i in edge_ids if i not in self.edge_attributes.index]

        bad_edge_layer = []
        if not self.edge_layer_attributes.empty:
            for lid, eid in self.edge_layer_attributes.index:
                if lid not in self._layers or eid not in edge_ids:
                    bad_edge_layer.append((lid, eid))

        return {
            "extra_node_rows": extra_node_rows,
            "extra_edge_rows": extra_edge_rows,
            "missing_node_rows": missing_node_rows,
            "missing_edge_rows": missing_edge_rows,
            "invalid_edge_layer_rows": bad_edge_layer
        }
    
    def neighbors(self, entity_id):
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
                # binary / node_edge (existing behavior)
                s, t, _ = self.edge_definitions[eid]
                edir = self.edge_directed.get(eid, self.directed)
                if s == entity_id:
                    out.add(t)
                elif t == entity_id and (not edir or self.entity_types.get(entity_id) == 'edge'):
                    out.add(s)
        return list(out)

    def out_neighbors(self, node_id):
        if node_id not in self.entity_to_idx:
            return []
        out = set()
        for eid in self.edge_to_idx.keys():
            kind = self.edge_kind.get(eid, None)
            if kind == "hyper":
                meta = self.hyperedge_definitions[eid]
                if meta["directed"]:
                    if node_id in meta["head"]:
                        out |= (meta["tail"])
                else:
                    if node_id in meta.get("members", ()):
                        out |= (meta["members"] - {node_id})
            else:
                s, t, _ = self.edge_definitions[eid]
                edir = self.edge_directed.get(eid, self.directed)
                if s == node_id:
                    out.add(t)
                elif t == node_id and not edir:
                    out.add(s)
        return list(out)

    def in_neighbors(self, node_id):
        if node_id not in self.entity_to_idx:
            return []
        inn = set()
        for eid in self.edge_to_idx.keys():
            kind = self.edge_kind.get(eid, None)
            if kind == "hyper":
                meta = self.hyperedge_definitions[eid]
                if meta["directed"]:
                    if node_id in meta["tail"]:
                        inn |= (meta["head"])
                else:
                    if node_id in meta.get("members", ()):
                        inn |= (meta["members"] - {node_id})
            else:
                s, t, _ = self.edge_definitions[eid]
                edir = self.edge_directed.get(eid, self.directed)
                if t == node_id:
                    inn.add(s)
                elif s == node_id and not edir:
                    inn.add(t)
        return list(inn)

    def get_directed_edges(self):
        """Return all directed edges."""
        return [eid for eid in self.edge_to_idx.keys() 
                if self.edge_directed.get(eid, self.directed)]

    def get_undirected_edges(self):
        """Return all undirected edges."""
        return [eid for eid in self.edge_to_idx.keys() 
                if not self.edge_directed.get(eid, self.directed)]
    def subgraph_from_layer(self, layer_id, *, resolve_layer_weights=True):
        """Return a new IncidenceGraph restricted to one layer."""
        if layer_id not in self._layers:
            raise KeyError(f"Layer {layer_id} not found")
        lg = IncidenceGraph(directed=self.directed)

        # 1) bring over entities (nodes + edge-entities) that participate in this layer
        layer_nodes = self._layers[layer_id]["nodes"]
        layer_edges = self._layers[layer_id]["edges"]

        for ent_id in layer_nodes:
            if self.entity_types.get(ent_id) == "node":
                attrs = self.node_attributes.loc[ent_id].to_dict() if (not self.node_attributes.empty and ent_id in self.node_attributes.index) else {}
                lg.add_node(ent_id, layer=lg._default_layer, **attrs)
            else:
                # edge-entity
                attrs = self.node_attributes.loc[ent_id].to_dict() if (not self.node_attributes.empty and ent_id in self.node_attributes.index) else {}
                lg.add_edge_entity(ent_id, layer=lg._default_layer, **attrs)

        # 2) bring over edges in this layer (preserve directedness + attrs)
        for eid in layer_edges:
            src, tgt, etype = self.edge_definitions[eid]
            w = (self.get_effective_edge_weight(eid, layer=layer_id)
                if resolve_layer_weights else self.edge_weights[eid])
            ed_attrs = self.edge_attributes.loc[eid].to_dict() if (not self.edge_attributes.empty and eid in self.edge_attributes.index) else {}
            edir = self.edge_directed.get(eid, self.directed)
            lg.add_edge(src, tgt, weight=float(w), edge_id=eid,
                        edge_type=etype, edge_directed=edir, **ed_attrs)

        # 3) set layer metadata (copy only this layer as 'default')
        lg.set_layer_attrs(layer_id, **self.get_layer_info(layer_id)["attributes"])
        return lg

