try:
    import igraph as ig
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional dependency 'python-igraph' is not installed. "
        "Install with: pip install graphglue[igraph]"
    ) from e

from typing import Any
from enum import Enum
import json

def _collect_layers_and_weights(graph) -> tuple[dict, dict]:
    """
    Returns:
      layers_section: {layer_id: [edge_id, ...]}
      layer_weights:  {layer_id: {edge_id: weight}}
    Backends supported: Polars-like, .to_dicts()-like, dict.
    """
    layers_section: dict = {}
    layer_weights: dict = {}

    # --- Source A: edge_layer_attributes table (Polars-like)
    t = getattr(graph, "edge_layer_attributes", None)
    if t is not None and hasattr(t, "filter"):
        try:
            # Attempt Polars path without hard dependency
            import polars as pl  # optional; if missing, we'll fall through
            # Get all rows then group by layer in Python (keeps us backend-agnostic)
            if hasattr(t, "to_dicts"):
                rows = t.to_dicts()
            else:
                # last-ditch: try turning the entire table into a list of dicts
                # many DataFrame-likes support .rows(named=True)
                rows = getattr(t, "rows", lambda named=False: [])(named=True)  # type: ignore
            for r in rows:
                lid = r.get("layer")
                if lid is None:
                    continue
                eid = r.get("edge_id", r.get("edge"))
                if eid is None:
                    continue
                layers_section.setdefault(lid, []).append(eid)
                w = r.get("weight")
                if w is not None:
                    layer_weights.setdefault(lid, {})[eid] = float(w)
        except Exception:
            pass  # fall through to other sources

    # --- Source B: edge_layer_attributes with .to_dicts() but no Polars
    if not layers_section and t is not None and hasattr(t, "to_dicts"):
        try:
            for r in t.to_dicts():
                lid = r.get("layer")
                if lid is None:
                    continue
                eid = r.get("edge_id", r.get("edge"))
                if eid is None:
                    continue
                layers_section.setdefault(lid, []).append(eid)
                w = r.get("weight")
                if w is not None:
                    layer_weights.setdefault(lid, {})[eid] = float(w)
        except Exception:
            pass

    # --- Source C: dict mapping layer -> {edge_id: attrs}
    if not layers_section and isinstance(t, dict):
        for lid, ed in t.items():
            if isinstance(ed, dict):
                eids = list(ed.keys())
                layers_section.setdefault(lid, []).extend(eids)
                # pick weights if present
                for eid, attrs in ed.items():
                    if isinstance(attrs, dict) and "weight" in attrs and attrs["weight"] is not None:
                        layer_weights.setdefault(lid, {})[eid] = float(attrs["weight"])

    # --- Fallback D: per-edge scan (if graph exposes edge iteration + get_edge_layers)
    if not layers_section:
        edges_iter = None
        for attr in ("edges", "iter_edges", "edge_ids"):
            if hasattr(graph, attr):
                try:
                    edges_iter = list(getattr(graph, attr)())
                    break
                except Exception:
                    pass
        if edges_iter:
            for eid in edges_iter:
                lids = []
                for probe in ("get_edge_layers", "edge_layers"):
                    if hasattr(graph, probe):
                        try:
                            lids = list(getattr(graph, probe)(eid))
                            break
                        except Exception:
                            pass
                for lid in lids or []:
                    layers_section.setdefault(lid, []).append(eid)

    # --- Collect per-layer weight overrides using canonical accessor
    if hasattr(graph, "get_edge_layer_attr"):
        for lid, eids in list(layers_section.items()):
            for eid in eids:
                w = None
                try:
                    w = graph.get_edge_layer_attr(lid, eid, "weight", default=None)
                except Exception:
                    try:
                        # some implementations don't support default=
                        w = graph.get_edge_layer_attr(lid, eid, "weight")
                    except Exception:
                        w = None
                if w is not None:
                    layer_weights.setdefault(lid, {})[eid] = float(w)

    # Ensure deterministic and non-empty lists in manifest
    for lid, eids in layers_section.items():
        # unique, stable order
        seen = set()
        uniq = []
        for e in eids:
            if e not in seen:
                seen.add(e)
                uniq.append(e)
        layers_section[lid] = uniq

    return layers_section, layer_weights


def _serialize_value(v: Any) -> Any:
    if isinstance(v, Enum):
        return v.name
    if hasattr(v, "items"):
        return dict(v)
    return v


def _attrs_to_dict(attrs_dict: dict) -> dict:
    out = {}
    for k, v in attrs_dict.items():
        if isinstance(v, Enum):
            out[k] = v.name
        elif hasattr(v, "items"):
            out[k] = {kk: (vv.name if isinstance(vv, Enum) else vv) for kk, vv in dict(v).items()}
        else:
            out[k] = v
    return out


def _is_directed_eid(graph: "Graph", eid: str) -> bool:
    kind = graph.edge_kind.get(eid)
    if kind == "hyper":
        return bool(graph.hyperedge_definitions[eid].get("directed", False))
    return bool(graph.edge_directed.get(eid, graph.directed))


def _export_legacy(graph: "Graph", *, directed: bool = True,
                   skip_hyperedges: bool = True, public_only: bool = False):
    """
    Export Graph to igraph.Graph without manifest.
    
    igraph requires integer vertex indices; external vertex IDs are preserved
    in vertex attribute 'name'. Edge IDs stored in edge attribute 'eid'.
    
    Parameters
    ----------
    graph : Graph
        Source graph instance.
    directed : bool
        If True, export as directed igraph.Graph; else undirected.
        Undirected edges in directed export are emitted bidirectionally.
    skip_hyperedges : bool
        If True, drop hyperedges. If False:
          - directed hyperedges expand head×tail (cartesian product)
          - undirected hyperedges expand to clique
    public_only : bool
        If True, strip private attrs starting with "__".
    
    Returns
    -------
    igraph.Graph
    """
    import igraph as ig

    # Build the vertex universe robustly
    # Start with declared vertices
    base_vertices = set(graph.vertices())

    # Ensure endpoints that appear in edges are also included
    endpoints = set()
    for eidx in range(graph.number_of_edges()):
        eid = graph.idx_to_edge[eidx]
        # graph.get_edge(eidx) returns (S, T) as sets for both binary and hyper encodings
        try:
            S, T = graph.get_edge(eidx)
        except Exception:
            S, T = set(), set()
        endpoints.update(S)
        endpoints.update(T)

    # If we are going to expand hyperedges, include their members/head/tail too
    if not skip_hyperedges:
        for eid, hdef in getattr(graph, "hyperedge_definitions", {}).items():
            if isinstance(hdef, dict):
                if hdef.get("members") is not None:
                    endpoints.update(hdef.get("members", []))
                else:
                    endpoints.update(hdef.get("head", []))
                    endpoints.update(hdef.get("tail", []))

    vertices = list(dict.fromkeys(list(base_vertices) + [v for v in endpoints]))  # stable order
    vidx = {v: i for i, v in enumerate(vertices)}

    # Create igraph graph and set vertex 'name'
    G = ig.Graph(directed=bool(directed))
    G.add_vertices(len(vertices))
    G.vs["name"] = vertices

    # Attach vertex attributes (works for both vertices and edge-entities)
    # Polars-safe extraction; ignore if table missing/empty.
    vtab = getattr(graph, "vertex_attributes", None)
    # Pre-scan to a dict for O(1) lookup
    vattr_map = {}
    try:
        if vtab is not None and hasattr(vtab, "to_dicts") and vtab.height > 0 and "vertex_id" in vtab.columns:
            for row in vtab.to_dicts():
                d = dict(row)
                vid = d.pop("vertex_id", None)
                if vid is not None:
                    vattr_map[vid] = d
    except Exception:
        vattr_map = {}

    def _serialize_value(x):
        # reuse your serializer if it exists; otherwise passthrough
        try:
            return globals()["_serialize_value"](x)
        except KeyError:
            return x

    for v in vertices:
        v_attr = dict(vattr_map.get(v, {}))
        if public_only:
            v_attr = {k: _serialize_value(val) for k, val in v_attr.items()
                      if not str(k).startswith("__")}
        else:
            v_attr = {k: _serialize_value(val) for k, val in v_attr.items()}

        # Ensure attribute columns exist and then write row value
        for k, val in v_attr.items():
            if k not in G.vs.attributes():
                G.vs[k] = [None] * G.vcount()
            G.vs[vidx[v]][k] = val

    # Helper: directedness per edge-id (fallback if helper missing)
    def _is_dir_eid(g, eid):
        try:
            return _is_directed_eid(g, eid)  # use existing helper if present
        except NameError:
            return bool(getattr(g, "edge_directed", {}).get(eid, getattr(g, "directed", True)))

    # Add edges (binary & vertex-edge). Hyperedges: skip or expand
    etab = getattr(graph, "edge_attributes", None)
    eattr_map = {}
    try:
        if etab is not None and hasattr(etab, "to_dicts") and etab.height > 0 and "edge_id" in etab.columns:
            for row in etab.to_dicts():
                d = dict(row)
                eid = d.pop("edge_id", None)
                if eid is not None:
                    eattr_map[eid] = d
    except Exception:
        eattr_map = {}

    for eidx in range(graph.number_of_edges()):
        eid = graph.idx_to_edge[eidx]
        S, T = graph.get_edge(eidx)

        e_attr = dict(eattr_map.get(eid, {}))
        if public_only:
            e_attr = {k: _serialize_value(val) for k, val in e_attr.items()
                      if not str(k).startswith("__")}
        else:
            e_attr = {k: _serialize_value(val) for k, val in e_attr.items()}

        weight = graph.edge_weights.get(eid, 1.0)
        if public_only:
            e_attr["weight"] = weight
        else:
            e_attr["__weight"] = weight

        is_hyper = (graph.edge_kind.get(eid) == "hyper")
        is_dir = _is_dir_eid(graph, eid)
        members = S | T

        # Binary / vertex-edge path ----------
        if not is_hyper and len(members) <= 2:
            if len(members) == 1:
                # self-loop
                u = next(iter(members))
                # Guard: endpoint must be indexed (should be by construction)
                if u not in vidx:
                    continue
                G.add_edge(vidx[u], vidx[u])
                e = G.es[-1]
                e["eid"] = eid
                for k, val in e_attr.items():
                    e[k] = val
            else:
                if is_dir:
                    # directed: source in S, target in T
                    uu = next(iter(S))
                    vv = next(iter(T))
                    if uu in vidx and vv in vidx:
                        G.add_edge(vidx[uu], vidx[vv])
                        e = G.es[-1]
                        e["eid"] = eid
                        for k, val in e_attr.items():
                            e[k] = val
                else:
                    # undirected edge; emit bidirectionally if `directed=True`
                    u, v = tuple(members)
                    if (u in vidx) and (v in vidx):
                        if directed:
                            G.add_edge(vidx[u], vidx[v])
                            e = G.es[-1]
                            e["eid"] = eid
                            for k, val in e_attr.items():
                                e[k] = val
                            G.add_edge(vidx[v], vidx[u])
                            e = G.es[-1]
                            e["eid"] = eid
                            for k, val in e_attr.items():
                                e[k] = val
                        else:
                            G.add_edge(vidx[u], vidx[v])
                            e = G.es[-1]
                            e["eid"] = eid
                            for k, val in e_attr.items():
                                e[k] = val
            continue  # done with this edge

        # ---------- Hyperedge path ----------
        if skip_hyperedges:
            continue

        if is_dir:
            # expand head × tail
            for u in S:
                for v in T:
                    if u not in vidx or v not in vidx:
                        continue
                    G.add_edge(vidx[u], vidx[v])
                    e = G.es[-1]
                    e["eid"] = eid
                    if not directed:
                        e["directed"] = True  # mark orientation in an undirected export
                    for k, val in e_attr.items():
                        e[k] = val
        else:
            # undirected hyperedge: clique (or bidir clique if directed=True)
            mem = [m for m in members if m in vidx]
            n = len(mem)
            if directed:
                for a in range(n):
                    for b in range(n):
                        if a == b:
                            continue
                        G.add_edge(vidx[mem[a]], vidx[mem[b]])
                        e = G.es[-1]
                        e["eid"] = eid
                        for k, val in e_attr.items():
                            e[k] = val
            else:
                for a in range(n):
                    for b in range(a + 1, n):
                        G.add_edge(vidx[mem[a]], vidx[mem[b]])
                        e = G.es[-1]
                        e["eid"] = eid
                        for k, val in e_attr.items():
                            e[k] = val

    return G


def _coeff_from_obj(obj) -> float:
    if isinstance(obj, (int, float)):
        return float(obj)
    if hasattr(obj, "items"):
        v = obj.get("__value", 1)
        if hasattr(v, "items"):
            v = v.get("__value", 1)
        try:
            return float(v)
        except Exception:
            return 1.0
    return 1.0


def _endpoint_coeff_map(edge_attrs: dict, key: str, vertices: set) -> dict:
    out = {}
    side = edge_attrs.get(key, {})
    for v in vertices:
        val = side.get(v, {})
        out[v] = _coeff_from_obj(val)
    return out


def to_igraph(graph: "Graph", directed=True, hyperedge_mode="skip", 
              layer=None, layers=None, public_only=False):
    """
    Export Graph → (igraph.Graph, manifest).
    
    Manifest preserves hyperedges with per-endpoint coefficients, layers,
    vertex/edge attrs, and stable edge IDs. igraph uses integer vertex
    indices; external IDs are stored in vertex attribute 'name'.
    
    Parameters
    ----------
    graph : Graph
        Source Graph instance.
    directed : bool, default True
        Export as directed igraph.Graph (True) or undirected (False).
    hyperedge_mode : {"skip", "expand"}, default "skip"
        How to handle hyperedges:
        - "skip": drop hyperedges entirely from igraph.Graph
        - "expand": cartesian product (directed) or clique (undirected)
    layer : str, optional
        Export single layer only.
    layers : list[str], optional
        Export union of specified layers.
    public_only : bool, default False
        If True, strip attributes starting with "__".
    
    Returns
    -------
    tuple[igraph.Graph, dict]
        (igG, manifest) where:
        - igG: igraph.Graph with integer vertex indices and 'name' attribute
          for external vertex IDs. Edge IDs stored in 'eid' attribute.
        - manifest: dict preserving full structure (hyperedges, layers,
          attributes, weights) for exact round-tripping.
    
    Notes
    -----
    igraph cannot represent hyperedges natively. The manifest is the SSOT
    (Single Source Of Truth) for reconstruction. The igraph.Graph is lossy:
    hyperedges are either dropped or expanded into multiple binary edges.
    """
    igG = _export_legacy(
        graph,
        directed=directed,
        skip_hyperedges=(hyperedge_mode == "skip"),
        public_only=public_only
    )

    vertex_attrs = {}
    for v in graph.vertices():
        v_attrs = graph.vertex_attributes.filter(
            graph.vertex_attributes["vertex_id"] == v
        ).to_dicts()
        if v_attrs:
            attrs = dict(v_attrs[0])
            attrs.pop("vertex_id", None)
            vertex_attrs[v] = _attrs_to_dict(attrs)
        else:
            vertex_attrs[v] = {}

    edge_attrs = {}
    for eidx in range(graph.number_of_edges()):
        eid = graph.idx_to_edge[eidx]
        e_attrs = graph.edge_attributes.filter(
            graph.edge_attributes["edge_id"] == eid
        ).to_dicts()
        if e_attrs:
            attrs = dict(e_attrs[0])
            attrs.pop("edge_id", None)
            edge_attrs[eid] = _attrs_to_dict(attrs)
        else:
            edge_attrs[eid] = {}

    manifest_edges = {}
    for eidx in range(graph.number_of_edges()):
        S, T = graph.get_edge(eidx)
        eid = graph.idx_to_edge[eidx]

        is_hyper = (graph.edge_kind.get(eid) == "hyper")

        if not is_hyper:
            members = S | T
            if len(members) == 1:
                u = next(iter(members))
                manifest_edges[eid] = (u, u, "regular")
            elif len(members) == 2:
                u, v = sorted(members)
                manifest_edges[eid] = (u, v, "regular")
            else:
                eattr = edge_attrs.get(eid, {})
                head_map = _endpoint_coeff_map(eattr, "__source_attr", S)
                tail_map = _endpoint_coeff_map(eattr, "__target_attr", T)
                manifest_edges[eid] = (head_map, tail_map, "hyper")
        else:
            eattr = edge_attrs.get(eid, {})
            head_map = _endpoint_coeff_map(eattr, "__source_attr", S)
            tail_map = _endpoint_coeff_map(eattr, "__target_attr", T)
            manifest_edges[eid] = (head_map, tail_map, "hyper")

    # ---------- BEGIN robust layer discovery + weights ----------
    def _rows_from_table(t):
        """Return a list[dict] from common table backends: Polars."""
        # Polars
        if hasattr(t, "to_dicts"):
            try:
                return list(t.to_dicts())
            except Exception:
                pass
        return []

    # Gather all edge ids we exported
    try:
        all_eids = [e for e in manifest_edges.keys()]
    except Exception:
        all_eids = []

    # Start with whatever the graph reports natively
    layers_section: dict = {}
    try:
        lids_native = list(graph.list_layers(include_default=True))
    except Exception:
        try:
            lids_native = list(graph.list_layers())
        except Exception:
            lids_native = []

    for lid in lids_native:
        try:
            eids = list(graph.get_layer_edges(lid))
        except Exception:
            eids = []
        if eids:
            seen = set()
            uniq = []
            for e in eids:
                if e not in seen:
                    seen.add(e)
                    uniq.append(e)
            layers_section[lid] = uniq

    # Now inspect edge_layer_attributes to find missing layers (e.g., "Lw")
    table_layers = {}
    t = getattr(graph, "edge_layer_attributes", None)
    if t is not None:
        # Case A: mapping {layer: {edge_id: attrs}}
        if isinstance(t, dict):
            for lid, mapping in t.items():
                if isinstance(mapping, dict):
                    table_layers.setdefault(lid, []).extend(list(mapping.keys()))
        # Case B: any table-like (Polars)
        rows = _rows_from_table(t)
        if rows:
            for r in rows:
                lid = r.get("layer") or r.get("layer_id") or r.get("lid")
                if lid is None:
                    continue
                eid = r.get("edge_id", r.get("edge"))
                if eid is not None:
                    table_layers.setdefault(lid, []).append(eid)

    # Merge table-discovered layers into layers_section
    for lid, eids in table_layers.items():
        if not eids:
            continue
        if lid in layers_section:
            seen = set(layers_section[lid])
            layers_section[lid].extend([e for e in eids if e not in seen])
        else:
            # stable, deduped
            layers_section[lid] = list(dict.fromkeys(eids))

    # If still missing layers, check common internal maps
    etl = getattr(graph, "edge_to_layers", None)  # {edge_id: [layer,...]}
    if etl:
        for eid, lids in etl.items():
            for lid in (lids or []):
                layers_section.setdefault(lid, [])
                if eid not in layers_section[lid]:
                    layers_section[lid].append(eid)

    le = getattr(graph, "layer_edges", None)  # {layer: [edge_id,...]}
    if le:
        for lid, eids in le.items():
            layers_section.setdefault(lid, [])
            for eid in list(eids):
                if eid not in layers_section[lid]:
                    layers_section[lid].append(eid)

    # As a last resort, probe per-edge membership if an API (Application Programming Interface) exists
    if hasattr(graph, "is_edge_in_layer") and all_eids:
        # also synthesize layer IDs from anything present
        known_lids = set(layers_section.keys())
        # if your graph can list non-default layers separately, try that too
        try:
            known_lids.update(list(graph.list_layers()))
        except Exception:
            pass
        for lid in known_lids:
            arr = layers_section.setdefault(lid, [])
            seen = set(arr)
            for eid in all_eids:
                try:
                    if graph.is_edge_in_layer(lid, eid) and eid not in seen:
                        arr.append(eid); seen.add(eid)
                except Exception:
                    pass

    # Remove empty layers to avoid `{}` vs only "default"
    layers_section = {lid: lst for lid, lst in layers_section.items() if lst}

    # Respect `layer` / `layers` parameters by post-filtering
    if layer is not None or layers is not None:
        requested = []
        if layer is not None:
            requested.extend([layer] if isinstance(layer, str) else list(layer))
        if layers is not None:
            requested.extend(list(layers))
        requested = {str(x) for x in requested}
        layers_section = {lid: eids for lid, eids in layers_section.items() if str(lid) in requested}

    # Per-layer weight overrides (only if present)
    layer_weights: dict = {}
    if hasattr(graph, "get_edge_layer_attr"):
        for lid, eids in layers_section.items():
            for eid in eids:
                w = None
                try:
                    w = graph.get_edge_layer_attr(lid, eid, "weight", default=None)
                except Exception:
                    try:
                        w = graph.get_edge_layer_attr(lid, eid, "weight")
                    except Exception:
                        w = None
                if w is not None:
                    layer_weights.setdefault(lid, {})[eid] = float(w)
    # ---------- END robust layer discovery + weights ----------
    base_weights = dict(graph.edge_weights)  # baseline/global weights
    manifest = {
         "edges": manifest_edges,
         "weights": base_weights,
         "layers": layers_section,
         "vertex_attrs": vertex_attrs,
         "edge_attrs": edge_attrs,
         "layer_weights": layer_weights,
         "edge_directed": {eid: bool(_is_directed_eid(graph, eid)) for eid in all_eids},
         "manifest_version": 1,
    }

    return igG, manifest


def save_manifest(manifest: dict, path: str):
    """
    Write manifest to JSON file.
    
    Parameters
    ----------
    manifest : dict
        Manifest dictionary from to_igraph().
    path : str
        Output file path (typically .json extension).
    
    Returns
    -------
    None
    
    Raises
    ------
    OSError
        If file cannot be written.
    """
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def load_manifest(path: str) -> dict:
    """
    Load manifest from JSON file.
    
    Parameters
    ----------
    path : str
        Path to manifest JSON file created by save_manifest().
    
    Returns
    -------
    dict
        Manifest dictionary suitable for from_igraph().
    
    Raises
    ------
    OSError
        If file cannot be read.
    json.JSONDecodeError
        If file contains invalid JSON.
    """
    with open(path) as f:
        return json.load(f)


def from_igraph(igG: ig.Graph, manifest: dict) -> "Graph":
    """
    Reconstruct a Graph from igraph.Graph + manifest.
    
    Parameters
    ----------
    igG : igraph.Graph
        igraph.Graph (largely ignored for structural data - used only
        for fallback vertex extraction if manifest is incomplete).
        Vertex 'name' attribute should contain external vertex IDs.
    manifest : dict
        Manifest dictionary created by to_igraph(). Must contain:
        - "edges" : dict[str, tuple]
            {edge_id: (u, v, "regular") | (head_map, tail_map, "hyper")}
        - "weights" : dict[str, float]
            {edge_id: weight}
        - "layers" : dict[str, list[str]]
            {layer_id: [edge_id, ...]}
        - "vertex_attrs" : dict[str, dict]
            {vertex_id: {attr: value, ...}}
        - "edge_attrs" : dict[str, dict]
            {edge_id: {attr: value, ...}}
    
    Returns
    -------
    Graph
        Reconstructed Graph instance with full hyperedge structure,
        layers, weights, and attributes restored from manifest.
    
    Notes
    -----
    The manifest is the single source of truth (SSOT). The igraph.Graph
    is only consulted for vertex names if the manifest is incomplete.
    This ensures exact round-trip fidelity for hyperedges, layers,
    and per-endpoint coefficients that igraph cannot represent natively.
    Vertex IDs are read from the igraph 'name' attribute if present,
    otherwise integer indices are used as fallback.
    """
    from ..core.graph import Graph
    
    H = Graph()

    names = igG.vs["name"] if "name" in igG.vs.attributes() else list(range(igG.vcount()))
    for v in names:
        try:
            H.add_vertex(v)
        except Exception:
            pass

    edges_def = manifest.get("edges", {})
    for eid, defn in edges_def.items():
        kind = defn[-1]
        if kind == "regular":
            u, v = defn[0], defn[1]
            is_dir = bool(manifest.get("edge_directed", {}).get(eid, True))
            H.add_edge(u, v, edge_id=eid, edge_directed=is_dir)
        elif kind == "hyper":
            head_map, tail_map = defn[0], defn[1]
            if isinstance(head_map, dict) and isinstance(tail_map, dict):
                head = list(head_map.keys())
                tail = list(tail_map.keys())
                is_dir = bool(manifest.get("edge_directed", {}).get(eid, True))
                is_dir = bool(manifest.get("edge_directed", {}).get(eid, True))
                H.add_hyperedge(head=head, tail=tail, edge_id=eid, edge_directed=is_dir)
                try:
                    existing_src = H.get_edge_attribute(eid, "__source_attr") or {}
                except Exception:
                    existing_src = {}
                try:
                    existing_tgt = H.get_edge_attribute(eid, "__target_attr") or {}
                except Exception:
                    existing_tgt = {}

                src_map = {u: {"__value": float(c)} for u, c in (head_map or {}).items()}
                tgt_map = {v: {"__value": float(c)} for v, c in (tail_map or {}).items()}

                # merge and write in one go
                merged_src = {**existing_src, **src_map}
                merged_tgt = {**existing_tgt, **tgt_map}
                H.set_edge_attrs(eid, __source_attr=merged_src, __target_attr=merged_tgt)


            else:
                try:
                    u, v = defn[0], defn[1]
                    is_dir = bool(manifest.get("edge_directed", {}).get(eid, True))
                    H.add_edge(u, v, edge_id=eid, edge_directed=is_dir)
                except Exception:
                    pass
        else:
            try:
                u, v = defn[0], defn[1]
                is_dir = bool(manifest.get("edge_directed", {}).get(eid, True))
                H.add_edge(u, v, edge_id=eid, edge_directed=is_dir)
            except Exception:
                pass

    for eid, w in manifest.get("weights", {}).items():
        try:
            H.edge_weights[eid] = w
        except Exception:
            pass

    for lid, eids in manifest.get("layers", {}).items():
        if lid not in H.list_layers(include_default=True):
            H.add_layer(lid)
        for eid in eids:
            try:
                H.add_edge_to_layer(lid, eid)
            except Exception:
                pass

    # Reapply per-layer weight overrides captured in the manifest
    lw = {}
    try:
        lw = manifest.get("layer_weights", {}) or {}
    except Exception:
        lw = {}
    for lid, per_edge in lw.items():
        # ensure layer exists
        try:
            if lid not in H.list_layers(include_default=True):
                H.add_layer(lid)
        except Exception:
            pass
        for eid, w in per_edge.items():
            try:
                H.set_edge_layer_attrs(lid, eid, weight=float(w))
            except Exception:
                pass

    return H


def to_backend(graph, **kwargs):
    """
    Export Graph to igraph without manifest (legacy compatibility).
    
    Parameters
    ----------
    graph : Graph
        Source Graph instance to export.
    **kwargs
        Forwarded to _export_legacy(). Supported:
        - directed : bool, default True
            Export as directed igraph.Graph (True) or undirected (False).
        - skip_hyperedges : bool, default True
            If True, drop hyperedges. If False, expand them
            (cartesian product for directed, clique for undirected).
        - public_only : bool, default False
            Strip attributes starting with "__" if True.
    
    Returns
    -------
    igraph.Graph
        igraph.Graph with integer vertex indices. External vertex IDs
        are stored in vertex attribute 'name'. Edge IDs stored in edge
        attribute 'eid'. Hyperedges are either dropped or expanded into
        multiple binary edges. No manifest is returned, so round-tripping
        will lose hyperedge structure, layers, and precise edge IDs.
    
    Notes
    -----
    This is a lossy export. Use to_igraph() with manifest for full fidelity.
    igraph requires integer vertex indices internally; the 'name' attribute
    preserves your original string IDs.
    """
    return _export_legacy(graph, **kwargs)

def from_ig_only(igG):
    """
    Build a Graph from a *plain* igraph.Graph (no manifest).
    Uses vertex 'name' when present; otherwise igraph indices (ints).
    Preserves all vertex/edge attributes.
    """
    from ..core.graph import Graph
    import igraph as ig

    H = Graph()

    # Vertex IDs: prefer 'name'
    names = igG.vs["name"] if "name" in igG.vs.attributes() else list(range(igG.vcount()))
    # Copy vertices and their attributes (verbatim)
    for i, vid in enumerate(names):
        try:
            H.add_vertex(vid)
        except Exception:
            pass
        vattrs = {}
        for k in igG.vs.attributes():
            if k == "name":  # ID already used; keep as attr too
                vattrs[k] = names[i]
            else:
                vattrs[k] = igG.vs[i][k]
        try:
            if vattrs:
                H.set_vertex_attrs(vid, **vattrs)
        except Exception:
            pass

    # Edges: copy everything
    is_dir = igG.is_directed()
    seen_auto = 0
    for e in igG.es:
        src = names[e.source]
        dst = names[e.target]
        d = {k: e[k] for k in igG.es.attributes()}

        eid = d.get("eid")
        if eid is None:
            seen_auto += 1
            eid = f"ig::e#{seen_auto}"

        e_directed = bool(d.get("directed", is_dir))
        w = d.get("weight", d.get("__weight", 1.0))

        try:
            H.add_vertex(src); H.add_vertex(dst)
        except Exception:
            pass
        try:
            H.add_edge(src, dst, edge_id=eid, edge_directed=e_directed)
        except Exception:
            H.add_edge(src, dst, edge_id=eid, edge_directed=True)

        try:
            H.edge_weights[eid] = float(w)
        except Exception:
            pass

        try:
            if d:
                H.set_edge_attrs(eid, **d)
        except Exception:
            pass

    return H


class IGraphAdapter:
    """
    Legacy adapter class for backward compatibility.
    
    Methods
    -------
    export(graph, **kwargs)
        Export Graph to igraph.Graph without manifest (lossy).
    """
    def export(self, graph, **kwargs):
        """
        Export Graph to igraph.Graph without manifest.
        
        Parameters
        ----------
        graph : Graph
        **kwargs
            See to_backend() for supported parameters.
        
        Returns
        -------
        igraph.Graph
        """
        return _export_legacy(graph, **kwargs)