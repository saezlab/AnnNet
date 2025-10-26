from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional, Tuple, Union

try:
    from ..core.graph import Graph
except Exception:
    from graphglue.core.graph import Graph


def _split_sif_line(line: str, delimiter: Optional[str]) -> List[str]:
    if delimiter is not None:
        return [t for t in line.rstrip("\n\r").split(delimiter) if t != ""]
    if "\t" in line:
        return [t for t in line.rstrip("\n\r").split("\t") if t != ""]
    return line.strip().split()


def _safe_vertex_attr_table(graph: 'Graph'):
    va = getattr(graph, "vertex_attributes", None)
    if va is None:
        return None
    return va if hasattr(va, "columns") and hasattr(va, "to_dicts") else None


def _get_all_edge_attrs(graph: 'Graph', edge_id: str):
    ea = getattr(graph, "edge_attributes", None)
    if (
        ea is not None
        and hasattr(ea, "columns")
        and hasattr(ea, "filter")
        and hasattr(ea, "to_dicts")
    ):
        try:
            if "edge_id" in ea.columns:
                rows = ea.filter(ea["edge_id"] == edge_id).to_dicts()
                if rows:
                    attrs = dict(rows[0])
                    attrs.pop("edge_id", None)
                    return {k: v for k, v in attrs.items() if v is not None}
        except Exception:
            pass
    return {}


def _get_edge_weight(graph: 'Graph', edge_id: str, default=1.0):
    ew = getattr(graph, "edge_weights", None)
    if ew is not None and hasattr(ew, "get"):
        try:
            w = ew.get(edge_id, default)
            return float(w) if w is not None else default
        except Exception:
            pass
    return default


def to_sif(
    graph: 'Graph',
    path: Optional[str] = None,
    *,
    relation_attr: str = "relation",
    default_relation: str = "interacts_with",
    write_nodes: bool = True,
    nodes_path: Optional[str] = None,
    lossless: bool = False,
    manifest_path: Optional[str] = None,
) -> Union[None, Tuple[None, Dict]]:
    """Export graph to SIF format.

    Standard mode (lossless=False):
        - Writes only binary edges to SIF file
        - Hyperedges, weights, attrs, IDs are lost
        - Returns None

    Lossless mode (lossless=True):
        - Writes binary edges to SIF file
        - Returns (None, manifest) where manifest contains all lost info
        - Manifest can be saved separately and used with from_sif()

    Args:
        path: Output SIF file path (if None, only manifest is returned in lossless mode)
        relation_attr: Edge attribute key for relation type
        default_relation: Default relation if attr missing
        write_nodes: Whether to write .nodes sidecar with vertex attrs
        nodes_path: Custom path for nodes sidecar (default: path + ".nodes")
        lossless: If True, return manifest with all non-SIF data
        manifest_path: If provided, write manifest to this path (only when lossless=True)

    Returns:
        None (standard mode) or (None, manifest_dict) (lossless mode)

    """

    manifest = (
        {
            "version": "1.0",
            "binary_edges": {},
            "hyperedges": {},
            "vertex_attrs": {},
            "edge_metadata": {},
            "layers": {},
        }
        if lossless
        else None
    )

    if path:
        with open(path, "w", encoding="utf-8") as f:
            edge_defs = getattr(graph, "edge_definitions", {}) or {}

            for eid, (src, tgt, _etype) in edge_defs.items():
                if src is None or tgt is None:
                    continue
                src_str = str(src).strip()
                tgt_str = str(tgt).strip()
                if (
                    not src_str
                    or not tgt_str
                    or src_str.lower() == "none"
                    or tgt_str.lower() == "none"
                ):
                    continue

                all_attrs = _get_all_edge_attrs(graph, eid)
                rel = all_attrs.get(relation_attr, default_relation)

                f.write(f"{src_str}\t{rel}\t{tgt_str}\n")

                if lossless:
                    directed = getattr(graph, "edge_directed", {}).get(eid, True)
                    weight = _get_edge_weight(graph, eid, 1.0)

                    manifest["binary_edges"][eid] = {
                        "source": src_str,
                        "target": tgt_str,
                        "directed": directed,
                    }

                    if weight != 1.0 or all_attrs or eid != f'edge_{len(manifest["binary_edges"])}':
                        manifest["edge_metadata"][eid] = {
                            "weight": weight,
                            "attrs": all_attrs,
                        }

        if write_nodes:
            sidecar = nodes_path if nodes_path is not None else (str(path) + ".nodes")
            with open(sidecar, "w", encoding="utf-8") as nf:
                nf.write("# nodes sidecar for SIF; format: <vertex_id>\tkey=value ...\n")

                vtable = _safe_vertex_attr_table(graph)
                vmap: Dict[str, Dict[str, object]] = {}

                if vtable is not None:
                    try:
                        for row in vtable.to_dicts():
                            vid_raw = row.get("vertex_id", None)
                            if vid_raw is None:
                                continue
                            vid = str(vid_raw).strip()
                            if not vid or vid.lower() == "none":
                                continue
                            attrs = {
                                k: v for k, v in row.items() if k != "vertex_id" and v is not None
                            }
                            vmap[vid] = attrs
                    except Exception:
                        vmap = {}

                if not vmap:
                    getter = getattr(graph, "get_vertex_attrs", None)
                    if callable(getter):
                        try:
                            for vid in graph.vertices():
                                if vid is None:
                                    continue
                                svid = str(vid).strip()
                                if not svid or svid.lower() == "none":
                                    continue
                                attrs = getter(vid) or {}
                                vmap[svid] = {k: v for k, v in attrs.items() if v is not None}
                        except Exception:
                            pass

                for vid in graph.vertices():
                    if vid is None:
                        continue
                    svid = str(vid).strip()
                    if not svid or svid.lower() == "none":
                        continue
                    attrs = vmap.get(svid, {})
                    if attrs:
                        kv = "\t".join(f"{k}={v}" for k, v in attrs.items())
                        nf.write(f"{svid}\t{kv}\n")
                    else:
                        nf.write(f"{svid}\n")

                    if lossless and attrs:
                        manifest["vertex_attrs"][svid] = attrs

    if lossless:
        edge_kind = getattr(graph, "edge_kind", {}) or {}
        hyp_defs = getattr(graph, "hyperedge_definitions", {}) or {}

        for eid, kind in edge_kind.items():
            if kind != "hyper":
                continue
            meta = hyp_defs.get(eid)
            if not meta:
                continue

            directed = bool(meta.get("directed", False))
            head = list(meta.get("head", [])) if directed else []
            tail = list(meta.get("tail", [])) if directed else []
            members = list(meta.get("members", [])) if not directed else []

            weight = _get_edge_weight(graph, eid, 1.0)
            attrs = _get_all_edge_attrs(graph, eid)

            manifest["hyperedges"][eid] = {
                "directed": directed,
                "head": head,
                "tail": tail,
                "members": members,
                "weight": weight,
                "attrs": attrs,
            }

        try:
            layer_ids = list(graph.list_layers(include_default=True))
            for lid in layer_ids:
                try:
                    edge_ids = list(graph.get_layer_edges(lid))
                    if not edge_ids:
                        continue

                    layer_info = {"edges": edge_ids, "weights": {}}

                    for eid in edge_ids:
                        try:
                            w = graph.get_edge_layer_attr(lid, eid, "weight", default=None)
                            if w is not None:
                                layer_info["weights"][eid] = float(w)
                        except Exception:
                            pass

                    manifest["layers"][str(lid)] = layer_info
                except Exception:
                    pass
        except Exception:
            pass

        if manifest_path:
            with open(manifest_path, "w", encoding="utf-8") as mf:
                json.dump(manifest, mf, indent=2)

        return None, manifest

    return None


def from_sif(
    path: str,
    *,
    manifest: Optional[Union[str, Dict]] = None,
    directed: bool = True,
    relation_attr: str = "relation",
    default_relation: str = "interacts_with",
    read_nodes_sidecar: bool = True,
    nodes_path: Optional[str] = None,
    encoding: str = "utf-8",
    delimiter: Optional[str] = None,
    comment_prefixes: Iterable[str] = ("#", "!"),
) -> 'Graph':
    """Import graph from SIF (Simple Interaction Format).

    Standard mode (manifest=None):
        - Reads binary edges from SIF file (source, relation, target)
        - Auto-generates edge IDs (edge_0, edge_1, ...)
        - All edges inherit the `directed` parameter
        - Vertex attributes loaded from optional .nodes sidecar
        - Hyperedges, per-edge directedness, and complex metadata are lost

    Lossless mode (manifest provided):
        - Reads binary edges from SIF file
        - Restores original edge IDs, weights, and attributes from manifest
        - Reconstructs hyperedges from manifest
        - Restores per-edge directedness from manifest
        - Restores layer memberships and layer-specific weights from manifest
        - Full round-trip fidelity when paired with to_sif(lossless=True)

    SIF Format:
        - Three columns: source<TAB>relation<TAB>target
        - Lines starting with comment_prefixes are ignored
        - Vertices referenced in edges are created automatically

    Sidecar .nodes file format (optional):
        - One vertex per line: vertex_id<TAB>key=value<TAB>key=value...
        - Boolean values: true/false (case-insensitive)
        - Numeric values: auto-detected floats
        - String values: everything else

    Args:
        path: Input SIF file path
        manifest: Manifest dict or path to manifest JSON (for lossless reconstruction)
        directed: Default directedness for edges (overridden by manifest if provided)
        relation_attr: Edge attribute key for storing relation type
        default_relation: Default relation if not specified in file
        read_nodes_sidecar: Whether to read .nodes sidecar file with vertex attributes
        nodes_path: Custom path for nodes sidecar (default: path + ".nodes")
        encoding: File encoding (default: utf-8)
        delimiter: Custom delimiter (default: auto-detect TAB or whitespace)
        comment_prefixes: Line prefixes to skip (default: # and !)

    Returns:
        Graph: Reconstructed graph object

    Notes:
        - SIF format only supports binary edges natively
        - For full graph reconstruction (hyperedges, layers, metadata), use manifest
        - Manifest files are created by to_sif(lossless=True)
        - Edge IDs are auto-generated in standard mode, preserved in lossless mode
        - Vertex attributes require .nodes sidecar file or manifest

    """

    if manifest is not None and not isinstance(manifest, dict):
        with open(str(manifest), "r", encoding="utf-8") as mf:
            manifest = json.load(mf)

    if manifest and "binary_edges" in manifest:
        H = Graph(directed=None)
    else:
        H = Graph(directed=directed)

    def _parse_node_kv(tok: str):
        if "=" not in tok:
            return None, None
        k, v = tok.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k == "":
            return None, None

        lv = v.lower()
        if lv == "true":
            return k, True
        if lv == "false":
            return k, False
        if lv in ("nan", "inf", "-inf"):
            return k, v

        try:
            return k, float(v)
        except Exception:
            return k, v

    if read_nodes_sidecar:
        sidecar = nodes_path if nodes_path is not None else (str(path) + ".nodes")
        import os

        if os.path.exists(sidecar):
            with open(sidecar, "r", encoding=encoding) as nf:
                for raw in nf:
                    s = raw.rstrip("\n\r")
                    if not s or any(s.lstrip().startswith(pfx) for pfx in comment_prefixes):
                        continue

                    toks = s.split("\t") if "\t" in s else [s]
                    vid_raw = toks[0]
                    if vid_raw is None:
                        continue
                    vid = str(vid_raw).strip()
                    if vid == "" or vid.lower() == "none":
                        continue

                    H.add_vertex(vid)

                    if len(toks) > 1:
                        kvs: Dict[str, object] = {}
                        for t in toks[1:]:
                            k, v = _parse_node_kv(t)
                            if k is not None:
                                kvs[k] = v
                        if kvs:
                            H.set_vertex_attrs(vid, **kvs)

    if manifest and "vertex_attrs" in manifest:
        for vid, attrs in manifest["vertex_attrs"].items():
            H.add_vertex(vid)
            H.set_vertex_attrs(vid, **attrs)

    edge_mapping = {}

    with open(path, "r", encoding=encoding) as f:
        for raw in f:
            if not raw:
                continue
            if any(raw.lstrip().startswith(pfx) for pfx in comment_prefixes):
                continue

            toks = _split_sif_line(raw, delimiter)
            if not toks:
                continue

            if len(toks) < 3:
                continue

            src = toks[0].strip()
            rel = toks[1].strip()
            tgt = toks[2].strip()

            if src == "" or src.lower() == "none" or tgt == "" or tgt.lower() == "none":
                continue

            H.add_vertex(src)
            H.add_vertex(tgt)

            edge_key = (src, tgt, rel)

            if manifest and "binary_edges" in manifest:
                orig_eid = None
                edge_directed_val = directed

                for eid, info in manifest["binary_edges"].items():
                    if info["source"] == src and info["target"] == tgt:
                        meta = manifest.get("edge_metadata", {}).get(eid, {})
                        edge_rel = meta.get("attrs", {}).get(relation_attr, default_relation)

                        if edge_rel == rel:
                            orig_eid = eid
                            edge_directed_val = info.get("directed", directed)
                            break

                if orig_eid:
                    eid = H.add_edge(src, tgt, edge_id=orig_eid, edge_directed=edge_directed_val)
                    edge_mapping[edge_key] = eid

                    if orig_eid in manifest.get("edge_metadata", {}):
                        meta = manifest["edge_metadata"][orig_eid]
                        weight = meta.get("weight", 1.0)
                        attrs = meta.get("attrs", {})

                        if weight != 1.0:
                            H.edge_weights[eid] = weight

                        if attrs:
                            H.set_edge_attrs(eid, **attrs)
                        # REMOVED the else clause - don't add relation if no attrs in manifest
                else:
                    eid = H.add_edge(src, tgt, edge_directed=edge_directed_val)
                    H.set_edge_attrs(eid, **{relation_attr: rel})
                    edge_mapping[edge_key] = eid
            else:
                eid = H.add_edge(src, tgt, edge_directed=directed)
                H.set_edge_attrs(eid, **{relation_attr: rel})
                edge_mapping[edge_key] = eid

    if manifest and "hyperedges" in manifest:
        for eid, info in manifest["hyperedges"].items():
            directed_he = info.get("directed", False)
            head = info.get("head", [])
            tail = info.get("tail", [])
            members = info.get("members", [])
            weight = info.get("weight", 1.0)
            attrs = info.get("attrs", {})

            for v in head + tail + members:
                H.add_vertex(v)

            if directed_he:
                he_id = H.add_hyperedge(head=head, tail=tail, edge_id=eid, edge_directed=True)
            else:
                he_id = H.add_hyperedge(members=members, edge_id=eid, edge_directed=False)

            if weight != 1.0:
                H.edge_weights[he_id] = weight

            if attrs:
                H.set_edge_attrs(he_id, **attrs)

    if manifest and "layers" in manifest:
        for lid, layer_info in manifest["layers"].items():
            try:
                if lid not in set(H.list_layers(include_default=True)):
                    H.add_layer(lid)
            except Exception:
                H.add_layer(lid)

            for eid in layer_info.get("edges", []):
                try:
                    H.add_edge_to_layer(lid, eid)
                except Exception:
                    pass

            for eid, weight in layer_info.get("weights", {}).items():
                try:
                    H.set_edge_layer_attrs(lid, eid, weight=weight)
                except Exception:
                    pass

    return H
