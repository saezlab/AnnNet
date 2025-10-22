from __future__ import annotations
from typing import Dict, Optional, Literal, Union
import polars as pl

if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

try:
    from graphglue.core.graph import Graph
except ImportError:
    from ..core.graph import Graph


def to_dataframes(
    graph: "Graph",
    *,
    include_layers: bool = True,
    include_hyperedges: bool = True,
    explode_hyperedges: bool = False,
    public_only: bool = True,
) -> Dict[str, pl.DataFrame]:
    """
    Export graph to Polars DataFrames.
    
    Returns a dictionary of DataFrames representing different aspects of the graph:
    - 'nodes': Vertex IDs and attributes
    - 'edges': Binary edges with source, target, weight, directed, attributes
    - 'hyperedges': Hyperedges with head/tail sets (if include_hyperedges=True)
    - 'layers': Layer membership (if include_layers=True)
    - 'layer_weights': Per-layer edge weights (if include_layers=True)
    
    Args:
        graph: Graph instance to export
        include_layers: Include layer membership tables
        include_hyperedges: Include hyperedge table
        explode_hyperedges: If True, explode hyperedges to one row per endpoint
        public_only: If True, filter out attributes starting with '__'
    
    Returns:
        Dictionary mapping table names to Polars DataFrames
    """
    result = {}
    
    # 1. Nodes table
    nodes_data = []
    for vid in graph.vertices():
        row = {"vertex_id": vid}
        attrs = graph.vertex_attributes.filter(
            pl.col("vertex_id") == vid
        ).to_dicts()
        if attrs:
            attr_dict = dict(attrs[0])
            attr_dict.pop("vertex_id", None)
            if public_only:
                attr_dict = {k: v for k, v in attr_dict.items() 
                           if not str(k).startswith("__")}
            row.update(attr_dict)
        nodes_data.append(row)
    
    result['nodes'] = pl.DataFrame(nodes_data) if nodes_data else pl.DataFrame(
        schema={"vertex_id": pl.Utf8}
    )
    
    # 2. Binary edges table
    edges_data = []
    for eid, (src, tgt, etype) in graph.edge_definitions.items():
        if etype == "hyper":
            continue
        
        row = {
            "edge_id": eid,
            "source": src,
            "target": tgt,
            "weight": graph.edge_weights.get(eid, 1.0),
            "directed": graph.edge_directed.get(
                eid, 
                True if graph.directed is None else graph.directed
            ),
            "edge_type": etype,
        }
        
        # Edge attributes
        attrs = graph.edge_attributes.filter(
            pl.col("edge_id") == eid
        ).to_dicts()
        if attrs:
            attr_dict = dict(attrs[0])
            attr_dict.pop("edge_id", None)
            if public_only:
                attr_dict = {k: v for k, v in attr_dict.items() 
                           if not str(k).startswith("__")}
            row.update(attr_dict)
        
        edges_data.append(row)
    
    result['edges'] = pl.DataFrame(edges_data) if edges_data else pl.DataFrame(
        schema={
            "edge_id": pl.Utf8,
            "source": pl.Utf8,
            "target": pl.Utf8,
            "weight": pl.Float64,
            "directed": pl.Boolean,
            "edge_type": pl.Utf8,
        }
    )
    
    # 3. Hyperedges table
    if include_hyperedges:
        hyperedges_data = []
        
        if explode_hyperedges:
            # Exploded format: one row per endpoint
            for eid, meta in graph.hyperedge_definitions.items():
                directed = meta.get("directed", False)
                weight = graph.edge_weights.get(eid, 1.0)
                
                # Get edge attributes once
                attrs = graph.edge_attributes.filter(
                    pl.col("edge_id") == eid
                ).to_dicts()
                attr_dict = {}
                if attrs:
                    attr_dict = dict(attrs[0])
                    attr_dict.pop("edge_id", None)
                    if public_only:
                        attr_dict = {k: v for k, v in attr_dict.items() 
                                   if not str(k).startswith("__")}
                
                if directed:
                    # Head vertices (sources)
                    for v in meta.get("head", []):
                        row = {
                            "edge_id": eid,
                            "vertex_id": v,
                            "role": "head",
                            "weight": weight,
                            "directed": True,
                        }
                        row.update(attr_dict)
                        hyperedges_data.append(row)
                    
                    # Tail vertices (targets)
                    for v in meta.get("tail", []):
                        row = {
                            "edge_id": eid,
                            "vertex_id": v,
                            "role": "tail",
                            "weight": weight,
                            "directed": True,
                        }
                        row.update(attr_dict)
                        hyperedges_data.append(row)
                else:
                    # Undirected: all members
                    for v in meta.get("members", []):
                        row = {
                            "edge_id": eid,
                            "vertex_id": v,
                            "role": "member",
                            "weight": weight,
                            "directed": False,
                        }
                        row.update(attr_dict)
                        hyperedges_data.append(row)
        else:
            # Compact format: lists in cells
            for eid, meta in graph.hyperedge_definitions.items():
                directed = meta.get("directed", False)
                weight = graph.edge_weights.get(eid, 1.0)
                
                row = {
                    "edge_id": eid,
                    "directed": directed,
                    "weight": weight,
                }
                
                if directed:
                    row["head"] = list(meta.get("head", []))
                    row["tail"] = list(meta.get("tail", []))
                    row["members"] = None
                else:
                    row["head"] = None
                    row["tail"] = None
                    row["members"] = list(meta.get("members", []))
                
                # Edge attributes
                attrs = graph.edge_attributes.filter(
                    pl.col("edge_id") == eid
                ).to_dicts()
                if attrs:
                    attr_dict = dict(attrs[0])
                    attr_dict.pop("edge_id", None)
                    if public_only:
                        attr_dict = {k: v for k, v in attr_dict.items() 
                                   if not str(k).startswith("__")}
                    row.update(attr_dict)
                
                hyperedges_data.append(row)
        
        if hyperedges_data:
            result['hyperedges'] = pl.DataFrame(hyperedges_data)
        else:
            if explode_hyperedges:
                result['hyperedges'] = pl.DataFrame(
                    schema={
                        "edge_id": pl.Utf8,
                        "vertex_id": pl.Utf8,
                        "role": pl.Utf8,
                        "weight": pl.Float64,
                        "directed": pl.Boolean,
                    }
                )
            else:
                result['hyperedges'] = pl.DataFrame(
                    schema={
                        "edge_id": pl.Utf8,
                        "directed": pl.Boolean,
                        "weight": pl.Float64,
                        "head": pl.List(pl.Utf8),
                        "tail": pl.List(pl.Utf8),
                        "members": pl.List(pl.Utf8),
                    }
                )
    
    # 4. Layer membership
    if include_layers:
        layers_data = []
        try:
            for lid in graph.list_layers(include_default=True):
                layer_meta = graph._layers.get(lid, {})
                for eid in layer_meta.get("edges", []):
                    layers_data.append({
                        "layer_id": lid,
                        "edge_id": eid,
                    })
        except Exception:
            pass
        
        result['layers'] = pl.DataFrame(layers_data) if layers_data else pl.DataFrame(
            schema={"layer_id": pl.Utf8, "edge_id": pl.Utf8}
        )
        
        # 5. Per-layer weights
        layer_weights_data = []
        try:
            df = graph.edge_layer_attributes
            if isinstance(df, pl.DataFrame) and df.height > 0:
                if {"layer_id", "edge_id", "weight"}.issubset(df.columns):
                    layer_weights_data = df.select(
                        ["layer_id", "edge_id", "weight"]
                    ).to_dicts()
        except Exception:
            pass
        
        result['layer_weights'] = pl.DataFrame(
            layer_weights_data
        ) if layer_weights_data else pl.DataFrame(
            schema={"layer_id": pl.Utf8, "edge_id": pl.Utf8, "weight": pl.Float64}
        )
    
    return result


def from_dataframes(
    nodes: Optional[pl.DataFrame] = None,
    edges: Optional[pl.DataFrame] = None,
    hyperedges: Optional[pl.DataFrame] = None,
    layers: Optional[pl.DataFrame] = None,
    layer_weights: Optional[pl.DataFrame] = None,
    *,
    directed: Optional[bool] = None,
    exploded_hyperedges: bool = False,
) -> "Graph":
    """
    Import graph from Polars DataFrames.
    
    Accepts DataFrames in the format produced by to_dataframes():
    
    Nodes DataFrame (optional):
        - Required: vertex_id
        - Optional: any attribute columns
    
    Edges DataFrame (optional):
        - Required: source, target
        - Optional: edge_id, weight, directed, edge_type, attribute columns
    
    Hyperedges DataFrame (optional):
        - Compact format: edge_id, directed, weight, head (list), tail (list), members (list)
        - Exploded format: edge_id, vertex_id, role, weight, directed
    
    Layers DataFrame (optional):
        - Required: layer_id, edge_id
    
    Layer_weights DataFrame (optional):
        - Required: layer_id, edge_id, weight
    
    Args:
        nodes: DataFrame with vertex_id and attributes
        edges: DataFrame with binary edges
        hyperedges: DataFrame with hyperedges
        layers: DataFrame with layer membership
        layer_weights: DataFrame with per-layer edge weights
        directed: Default directedness (None = mixed graph)
        exploded_hyperedges: If True, hyperedges DataFrame is in exploded format
    
    Returns:
        Graph instance
    """
    G = Graph(directed=directed)
    
    # 1. Add vertices
    if nodes is not None and nodes.height > 0:
        if "vertex_id" not in nodes.columns:
            raise ValueError("nodes DataFrame must have 'vertex_id' column")
        
        vertex_rows = []
        for row in nodes.to_dicts():
            vid = row.pop("vertex_id")
            vertex_rows.append({"vertex_id": vid, "attributes": row})
        
        for vrow in vertex_rows:
            G.add_vertex(vrow["vertex_id"])
            if vrow["attributes"]:
                G.set_vertex_attrs(vrow["vertex_id"], **vrow["attributes"])
    
    # 2. Add binary edges
    if edges is not None and edges.height > 0:
        if "source" not in edges.columns or "target" not in edges.columns:
            raise ValueError("edges DataFrame must have 'source' and 'target' columns")
        
        edge_rows = []
        for row in edges.to_dicts():
            src = row.pop("source")
            tgt = row.pop("target")
            eid = row.pop("edge_id", None)
            weight = row.pop("weight", 1.0)
            edge_directed = row.pop("directed", directed)
            etype = row.pop("edge_type", "regular")
            
            edge_rows.append({
                "source": src,
                "target": tgt,
                "edge_id": eid,
                "weight": weight,
                "edge_directed": edge_directed,
                "edge_type": etype,
                "attributes": row,
            })
        
        G.add_edges_bulk(edge_rows)
    
    # 3. Add hyperedges
    if hyperedges is not None and hyperedges.height > 0:
        if exploded_hyperedges:
            # Exploded format: group by edge_id
            if "edge_id" not in hyperedges.columns or "vertex_id" not in hyperedges.columns:
                raise ValueError("Exploded hyperedges must have 'edge_id' and 'vertex_id' columns")
            
            grouped = hyperedges.group_by("edge_id").agg(pl.all())
            
            for row in grouped.to_dicts():
                eid = row["edge_id"][0] if isinstance(row["edge_id"], list) else row["edge_id"]
                vertices = row["vertex_id"]
                roles = row.get("role", [])
                directed_vals = row.get("directed", [])
                weights = row.get("weight", [])
                
                directed = directed_vals[0] if directed_vals else False
                weight = weights[0] if weights else 1.0
                
                if directed:
                    head = [v for v, r in zip(vertices, roles) if r == "head"]
                    tail = [v for v, r in zip(vertices, roles) if r == "tail"]
                    G.add_hyperedge(head=head, tail=tail, edge_id=eid, 
                                  edge_directed=True, weight=weight)
                else:
                    members = vertices
                    G.add_hyperedge(members=members, edge_id=eid, 
                                  edge_directed=False, weight=weight)
        else:
            # Compact format
            if "edge_id" not in hyperedges.columns:
                raise ValueError("hyperedges DataFrame must have 'edge_id' column")
            
            for row in hyperedges.to_dicts():
                eid = row.pop("edge_id")
                directed_he = row.pop("directed", False)
                weight = row.pop("weight", 1.0)
                head = row.pop("head", None)
                tail = row.pop("tail", None)
                members = row.pop("members", None)
                
                if directed_he:
                    G.add_hyperedge(head=head or [], tail=tail or [], 
                                  edge_id=eid, edge_directed=True, weight=weight)
                else:
                    G.add_hyperedge(members=members or [], edge_id=eid, 
                                  edge_directed=False, weight=weight)
                
                # Attributes
                if row:
                    G.set_edge_attrs(eid, **row)
    
    # 4. Add layer memberships
    if layers is not None and layers.height > 0:
        if "layer_id" not in layers.columns or "edge_id" not in layers.columns:
            raise ValueError("layers DataFrame must have 'layer_id' and 'edge_id' columns")
        
        for row in layers.to_dicts():
            lid = row["layer_id"]
            eid = row["edge_id"]
            
            try:
                if lid not in set(G.list_layers(include_default=True)):
                    G.add_layer(lid)
            except Exception:
                G.add_layer(lid)
            
            try:
                G.add_edge_to_layer(lid, eid)
            except Exception:
                pass
    
    # 5. Add per-layer weights
    if layer_weights is not None and layer_weights.height > 0:
        if {"layer_id", "edge_id", "weight"}.issubset(layer_weights.columns):
            for row in layer_weights.to_dicts():
                lid = row["layer_id"]
                eid = row["edge_id"]
                weight = row["weight"]
                
                try:
                    G.set_edge_layer_attrs(lid, eid, weight=weight)
                except Exception:
                    pass
    
    return G