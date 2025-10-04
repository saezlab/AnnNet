import numpy as np
import scipy.sparse as sp
import polars as pl
from collections import defaultdict
from ._base import EdgeType
import math

class IncidenceGraph:

    # Constants (Attribute helpers)
    _NODE_RESERVED = {"node_id"}               # nothing structural for nodes
    _EDGE_RESERVED = {"edge_id","source","target","weight","edge_type","directed","layer","layer_weight","kind","members","head","tail"}    
    _LAYER_RESERVED = {"layer_id"}

    # Construction

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
        self.node_attributes = pl.DataFrame(schema={"node_id": pl.Utf8})
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

    # Layer basics

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

    def get_layer_info(self, layer_id):
        """Get layer information."""
        if layer_id not in self._layers:
            raise KeyError(f"Layer {layer_id} not found")
        return self._layers[layer_id].copy()

    # ID + entity ensure helpers

    def _get_next_edge_id(self) -> str:
        """Generate unique edge ID for parallel edges."""
        edge_id = f"edge_{self._next_edge_id}"
        self._next_edge_id += 1
        return edge_id
    
    def _ensure_node_table(self) -> None:
        df = getattr(self, "node_attributes", None)
        if not isinstance(df, pl.DataFrame) or "node_id" not in df.columns:
            self.node_attributes = pl.DataFrame({"node_id": pl.Series([], dtype=pl.Utf8)})

    def _ensure_node_row(self, node_id: str) -> None:
        df = self.node_attributes
        # if row already exists, nothing to do
        if df.height and df.filter(pl.col("node_id") == node_id).height > 0:
            return
        if df.is_empty():
            # Create first row with the canonical node schema
            self.node_attributes = pl.DataFrame({"node_id": [node_id]}, schema={"node_id": pl.Utf8})
            return
        # Align columns: create a single dict with all columns present
        row = {c: None for c in df.columns}
        row["node_id"] = node_id
        self.node_attributes = pl.concat([df, pl.DataFrame([row])], how="vertical")

    # Build graph

    def add_node(self, node_id, layer=None, **attributes):
        layer = layer or self._current_layer

        # Add to global superset if new
        if node_id not in self.entity_to_idx:
            idx = self._num_entities
            self.entity_to_idx[node_id] = idx
            self.idx_to_entity[idx] = node_id
            self.entity_types[node_id] = "node"
            self._num_entities += 1
            # Resize incidence matrix
            self._matrix.resize((self._num_entities, self._num_edges))

        # Add to specified layer
        if layer not in self._layers:
            self._layers[layer] = {"nodes": set(), "edges": set(), "attributes": {}}
        self._layers[layer]["nodes"].add(node_id)

        # Ensure node_attributes has a row for this node (even with no attrs)
        self._ensure_node_table()
        self._ensure_node_row(node_id)

        # Upsert passed attributes (if any)
        if attributes:
            self.node_attributes = self._upsert_row(self.node_attributes, node_id, attributes)

        return node_id

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
                self.set_edge_attrs(edge_id, edge_type=(EdgeType.DIRECTED if is_dir else EdgeType.UNDIRECTED))


                # if source/target changed, update definition
                old_src, old_tgt, old_type = self.edge_definitions[edge_id]
                self.edge_definitions[edge_id] = (source, target, old_type)  # keep old_type by default

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

        # set layer
        layer = self._current_layer if layer is None else layer

        # ensure participants exist globally (nodes or edge-entities already supported)
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
                self._layers[layer] = {"nodes": set(), "edges": set(), "attributes": {}}
            self._layers[layer]["edges"].add(edge_id)
            if members is not None:
                self._layers[layer]["nodes"].update(members)
            else:
                self._layers[layer]["nodes"].update(self.hyperedge_definitions[edge_id]["head"])
                self._layers[layer]["nodes"].update(self.hyperedge_definitions[edge_id]["tail"])

        # attributes
        if attributes:
            self.set_edge_attrs(edge_id, **attributes)

        return edge_id

    def add_edge_to_layer(self, lid, eid):
        if lid not in self._layers:
            raise KeyError(f"Layer {lid} does not exist")
        self._layers[lid]["edges"].add(eid)

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

    # Remove / mutate down

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

    def remove_node(self, node_id):
        """Remove a node and all incident edges (binary + hyperedges)."""
        if node_id not in self.entity_to_idx:
            raise KeyError(f"Node {node_id} not found")

        entity_idx = self.entity_to_idx[node_id]

        # Collect incident edges (set to avoid duplicates)
        edges_to_remove = set()

        # Binary edges: edge_definitions {eid: (source, target, ...)}
        for eid, edef in list(self.edge_definitions.items()):
            try:
                source, target = edef[0], edef[1]
            except Exception:
                source, target = edef.get("source"), edef.get("target")
            if source == node_id or target == node_id:
                edges_to_remove.add(eid)

        # Hyperedges: hyperedge_definitions {eid: {"head":[...], "tail":[...]}} or {"members":[...]}
        def _node_in_hyperdef(hdef: dict, node: str) -> bool:
            # Common keys first
            for key in ("head", "tail", "members", "nodes", "vertices"):
                seq = hdef.get(key)
                if isinstance(seq, (list, tuple, set)) and node in seq:
                    return True
            # Safety net: scan any list/tuple/set values
            for v in hdef.values():
                if isinstance(v, (list, tuple, set)) and node in v:
                    return True
            return False

        hdefs = getattr(self, "hyperedge_definitions", {})
        if isinstance(hdefs, dict):
            for heid, hdef in list(hdefs.items()):
                if isinstance(hdef, dict) and _node_in_hyperdef(hdef, node_id):
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
        if isinstance(self.node_attributes, pl.DataFrame):
            if self.node_attributes.height > 0 and "node_id" in self.node_attributes.columns:
                self.node_attributes = self.node_attributes.filter(pl.col("node_id") != node_id)

        # Remove from per-layer membership
        for layer_data in self._layers.values():
            layer_data["nodes"].discard(node_id)

    def remove_layer(self, layer_id):
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
        """Set graph-level attribute."""
        self.graph_attributes[key] = value
    
    def get_graph_attribute(self, key, default=None):
        """Get graph-level attribute."""
        return self.graph_attributes.get(key, default)

    def set_node_attrs(self, node_id, **attrs):
        # keep attributes table pure
        clean = {k: v for k, v in attrs.items() if k not in self._NODE_RESERVED}
        if clean:
            self.node_attributes = self._upsert_row(self.node_attributes, node_id, clean)

    def get_node_attr(self, node_id, key, default=None):
        df = self.node_attributes
        if key not in df.columns:
            return default
        rows = df.filter(pl.col("node_id") == node_id)
        if rows.height == 0:
            return default
        val = rows.select(pl.col(key)).to_series()[0]
        return default if val is None else val

    def get_node_attribute(self, node_id, attribute): #legacy alias
        """Polars-only: returns scalar or None."""
        # allow Attr enums
        attribute = getattr(attribute, "value", attribute)

        df = self.node_attributes
        if not isinstance(df, pl.DataFrame):
            return None
        if df.height == 0 or "node_id" not in df.columns or attribute not in df.columns:
            return None

        rows = df.filter(pl.col("node_id") == node_id)
        if rows.height == 0:
            return None

        s = rows.get_column(attribute)
        return s.item(0) if s.len() else None

    def set_edge_attrs(self, edge_id, **attrs):
        # keep attributes table pure: strip structural keys
        clean = {k: v for k, v in attrs.items() if k not in self._EDGE_RESERVED}
        if clean:
            self.edge_attributes = self._upsert_row(self.edge_attributes, edge_id, clean)

    def get_edge_attr(self, edge_id, key, default=None):
        df = self.edge_attributes
        if key not in df.columns:
            return default
        rows = df.filter(pl.col("edge_id") == edge_id)
        if rows.height == 0:
            return default
        val = rows.select(pl.col(key)).to_series()[0]
        return default if val is None else val

    def get_edge_attribute(self, edge_id, attribute): #legacy alias
        """Polars-only: returns scalar or None."""
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
        clean = {k: v for k, v in attrs.items() if k not in self._LAYER_RESERVED}
        if clean:
            self.layer_attributes = self._upsert_row(self.layer_attributes, layer_id, clean)

    def get_layer_attr(self, layer_id, key, default=None):
        df = self.layer_attributes
        if key not in df.columns:
            return default
        rows = df.filter(pl.col("layer_id") == layer_id)
        if rows.height == 0:
            return default
        val = rows.select(pl.col(key)).to_series()[0]
        return default if val is None else val
 
    def set_edge_layer_attrs(self, layer_id, edge_id, **attrs):
        # keep attributes table pure: strip structural keys
        clean = {k: v for k, v in attrs.items() if k not in self._EDGE_RESERVED}
        if not clean:
            return
        # edge_layer_attributes is a pl.DataFrame with columns: layer_id, edge_id, ...
        self.edge_layer_attributes = self._upsert_row(
            self.edge_layer_attributes, (layer_id, edge_id), clean
        )

    def get_edge_layer_attr(self, layer_id, edge_id, key, default=None):
        df = self.edge_layer_attributes
        if key not in df.columns:
            return default
        rows = df.filter((pl.col("layer_id") == layer_id) & (pl.col("edge_id") == edge_id))
        if rows.height == 0:
            return default
        val = rows.select(pl.col(key)).to_series()[0]
        return default if val is None else val

    def set_layer_edge_weight(self, layer_id, edge_id, weight): #legacy weight helper
        if layer_id not in self._layers:
            raise KeyError(f"Layer {layer_id} not found")
        if edge_id not in self.edge_to_idx:
            raise KeyError(f"Edge {edge_id} not found")
        self.layer_edge_weights[layer_id][edge_id] = float(weight)

    def get_effective_edge_weight(self, edge_id, layer=None):
        """
        If layer is None: return the global (edge_id) weight.
        If layer is given: return the layer-specific override if present,
                    else the global weight.
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
        node_ids = {eid for eid, t in self.entity_types.items() if t == "node"}
        edge_ids = set(self.edge_to_idx.keys())

        na = self.node_attributes
        ea = self.edge_attributes
        ela = self.edge_layer_attributes

        node_attr_ids = (
            set(na.select("node_id").to_series().to_list())
            if isinstance(na, pl.DataFrame) and na.height > 0 and "node_id" in na.columns
            else set()
        )
        edge_attr_ids = (
            set(ea.select("edge_id").to_series().to_list())
            if isinstance(ea, pl.DataFrame) and ea.height > 0 and "edge_id" in ea.columns
            else set()
        )

        extra_node_rows = [i for i in node_attr_ids if i not in node_ids]
        extra_edge_rows = [i for i in edge_attr_ids if i not in edge_ids]
        missing_node_rows = [i for i in node_ids if i not in node_attr_ids]
        missing_edge_rows = [i for i in edge_ids if i not in edge_attr_ids]

        bad_edge_layer = []
        if isinstance(ela, pl.DataFrame) and ela.height > 0 and {"layer_id", "edge_id"} <= set(ela.columns):
            for lid, eid in ela.select(["layer_id", "edge_id"]).iter_rows():
                if lid not in self._layers or eid not in edge_ids:
                    bad_edge_layer.append((lid, eid))

        return {
            "extra_node_rows": extra_node_rows,
            "extra_edge_rows": extra_edge_rows,
            "missing_node_rows": missing_node_rows,
            "missing_edge_rows": missing_edge_rows,
            "invalid_edge_layer_rows": bad_edge_layer,
        }
    
    def _pl_dtype_for_value(self, v):
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
        """Create/cast attribute columns so updates/inserts won't hit Null dtypes."""
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
        Polars upsert using explicit key columns (no index).
        Keys:
        - node_attributes           -> ["node_id"]
        - edge_attributes           -> ["edge_id"]
        - layer_attributes          -> ["layer_id"]
        - edge_layer_attributes     -> ["layer_id", "edge_id"]
        Returns a NEW DataFrame; caller must reassign.
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
        elif "node_id" in cols:
            key_vals = {"node_id": idx}
            key_cols = ["node_id"]
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

    # Basic queries & metrics

    def is_edge_directed(self, edge_id):
        return bool(self.edge_directed.get(edge_id, self.directed))

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
    
    def get_directed_edges(self):
        """Return all directed edges."""
        return [eid for eid in self.edge_to_idx.keys() 
                if self.edge_directed.get(eid, self.directed)]

    def get_undirected_edges(self):
        """Return all undirected edges."""
        return [eid for eid in self.edge_to_idx.keys() 
                if not self.edge_directed.get(eid, self.directed)]

    def number_of_nodes(self):
        """Get number of nodes (excluding edge entities)."""
        return len([e for e in self.entity_types.values() if e == 'node'])
    
    def number_of_edges(self):
        """Get number of edges."""
        return self._num_edges

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

    # Materialized views

    def edges_view(self, layer=None, include_directed=True, include_weight=True, resolved_weight=True, copy=True):
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
                row["edge_type"] = etype  # 'regular' | 'node_edge' | None

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

    def nodes_view(self, copy=True):
        """
        Read-only table: node_id + pure attributes.
        """
        df = self.node_attributes
        if df.height == 0:
            return pl.DataFrame(schema={"node_id": pl.Utf8})
        return df.clone() if copy else df

    def layers_view(self, copy=True):
        """
        Read-only table: layer_id + pure attributes.
        """
        df = self.layer_attributes
        if df.height == 0:
            return pl.DataFrame(schema={"layer_id": pl.Utf8})
        return df.clone() if copy else df

    # Layer set-ops & cross-layer analytics

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

    # Traversal (neighbors)

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
                # binary / node_edge
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

    # Slicing / copying / accounting

    def subgraph_from_layer(self, layer_id, *, resolve_layer_weights=True):
        """Return a new IncidenceGraph restricted to one layer."""
        if layer_id not in self._layers:
            raise KeyError(f"Layer {layer_id} not found")

        lg = IncidenceGraph(directed=self.directed)
        # Create the destination layer and make it active
        lg.add_layer(layer_id, **self.get_layer_info(layer_id)["attributes"])
        lg.set_active_layer(layer_id)

        layer_nodes = self._layers[layer_id]["nodes"]
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

        # 1) bring over entities (nodes + edge-entities) that participate in this layer
        for ent_id in layer_nodes:
            if self.entity_types.get(ent_id) == "node":
                attrs = _row_attrs(self.node_attributes, "node_id", ent_id, "node_id")
                lg.add_node(ent_id, layer=layer_id, **attrs)
            else:
                # edge-entity (attributes stored in node_attributes as well)
                attrs = _row_attrs(self.node_attributes, "node_id", ent_id, "node_id")
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

            # binary or node-edge
            src, tgt, etype = self.edge_definitions[eid]  # etype in {'regular','node_edge'}
            edir = self.edge_directed.get(eid, self.directed)
            lg.add_edge(
                src,
                tgt,
                weight=float(w),
                edge_id=eid,         # preserve original id
                edge_type=etype,     # 'regular' or 'node_edge'
                edge_directed=edir,
                layer=layer_id,
                **ed_attrs
            )

        # 3) the layer's attributes were set on creation above
        return lg

    def copy(self):
        """Deep copy the entire graph (entities, edges, hyperedges, attributes, layers)."""
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

        new_graph = IncidenceGraph(directed=self.directed)

        # ---- Copy layers and their attributes first ----
        for lid in self._layers:
            if lid != new_graph._default_layer:  # 'default' already exists in new_graph
                new_graph.add_layer(lid)
            la = self.get_layer_info(lid)["attributes"]
            if la:
                new_graph.set_layer_attrs(lid, **la)

        # ---- Copy entities (nodes + edge-entities) ----
        for ent_id, etype in self.entity_types.items():
            attrs = _row_attrs(self.node_attributes, "node_id", ent_id, "node_id")
            if etype == "node":
                new_graph.add_node(ent_id, layer=new_graph._default_layer, **attrs)
            else:
                new_graph.add_edge_entity(ent_id, layer=new_graph._default_layer, **attrs)

        # ---- Copy *binary / node-edge* edges (skip hyperedges here) ----
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
                edge_type=edge_type,    # 'regular' or 'node_edge'
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

        # ---- Copy layer memberships (nodes & edges) EXACTLY ----
        for lid, meta in self._layers.items():
            # ensure layer exists
            if lid not in new_graph._layers:
                new_graph.add_layer(lid)

            # overwrite (not update) to match exactly
            new_graph._layers[lid]["nodes"] = set(meta["nodes"])
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
        # Approximate matrix memory: each non-zero entry stores row, col, and value (4 bytes each)
        matrix_bytes = self._matrix.nnz * (4 + 4 + 4)
        # Estimate dict memory: ~100 bytes per entry
        dict_bytes = (len(self.entity_to_idx) + len(self.edge_to_idx) + len(self.edge_weights)) * 100
        
        df_bytes = 0

        # Node attributes
        if isinstance(self.node_attributes, pl.DataFrame):
            # Polars provides a built-in estimate of total size in bytes
            df_bytes += self.node_attributes.estimated_size()

        # Edge attributes
        if isinstance(self.edge_attributes, pl.DataFrame):
            df_bytes += self.edge_attributes.estimated_size()

        return matrix_bytes + dict_bytes + df_bytes
