from __future__ import annotations
from typing import Iterable, Tuple, List, Dict, Optional


def to_sif(graph: "Graph", path, *, relation_attr="relation",
           expand_hyperedges: bool = False, default_relation="interacts_with"):
    """
    Write SIF: 'source<tab>relation<tab>target' per line.
    - Binary edges: relation from attr or default.
    - Hyperedges: expand to pairwise if expand_hyperedges=True, else skip.
    """
    with open(path, "w", encoding="utf-8") as f:
        # binary edges
        for eidx in range(graph.number_of_edges()):
            eid = graph.idx_to_edge[eidx]
            if graph.edge_kind.get(eid) == "hyper":
                continue
            S, T = graph.get_edge(eidx)
            members = (S | T)
            if len(members) == 1:
                u = next(iter(members)); v = u
            else:
                u, v = sorted(members)
            rel = (graph.get_edge_attribute(eid, relation_attr)
                   if hasattr(graph, "get_edge_attribute") else None) or default_relation
            f.write(f"{u}\t{rel}\t{v}\n")

        # hyperedges (optional expansion)
        if expand_hyperedges:
            for eidx in range(graph.number_of_edges()):
                eid = graph.idx_to_edge[eidx]
                if graph.edge_kind.get(eid) != "hyper":
                    continue
                S, T = graph.get_edge(eidx)
                members = sorted((S | T))
                for i in range(len(members)):
                    for j in range(i+1, len(members)):
                        f.write(f"{members[i]}\thyper_{eid}\t{members[j]}\n")

def from_sif(path: str,
             *,
             directed: bool = True,
             relation_attr: str = "relation",
             delimiter: Optional[str] = None,
             comment_prefixes: Tuple[str, ...] = ("#", "!", "//"),
             encoding: str = "utf-8") -> "Graph":
    """
    Load a SIF (Simple Interaction Format) file into a Graph.

    SIF lines:
      - Classic:   source <sep> relation <sep> target1 [target2 ...] [kv kv ...]
      - Strict-3:  source <sep> relation <sep> target [kv kv ...]

    Extras after targets:
      - key=value  -> edge attribute
      - lone number-> weight
      - anything   -> appended to 'sif_extra' (list[str]) for zero-loss ingestion.

    Parameters
    ----------
    path : str
    directed : bool
        If True, create directed edges source→target; else undirected.
    relation_attr : str
        Attribute name to store the relation label under.
    delimiter : Optional[str]
        If None, auto-detect: prefer tab if present, else split on whitespace.
    comment_prefixes : tuple[str, ...]
        Lines starting with any of these are ignored.
    encoding : str

    Returns
    -------
    Graph
    """
    from ..core.graph import Graph

    H = Graph()

    def _split(line: str) -> List[str]:
        if delimiter is not None:
            return [t for t in line.rstrip("\n\r").split(delimiter) if t != ""]
        # auto: use TAB if present, else whitespace
        if "\t" in line:
            return [t for t in line.rstrip("\n\r").split("\t") if t != ""]
        return line.strip().split()

    def _parse_kv(tok: str):
        if "=" not in tok:
            return None, None
        k, v = tok.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k == "":
            return None, None
        # try numeric
        try:
            if v.lower() in ("nan", "inf", "-inf"):
                return k, v
            fv = float(v)
            return k, fv
        except Exception:
            return k, v

    with open(path, "r", encoding=encoding) as f:
        line_no = 0
        for raw in f:
            line_no += 1
            s = raw.strip()
            if not s:
                continue
            if any(s.startswith(pfx) for pfx in comment_prefixes):
                continue

            toks = _split(raw)
            if len(toks) < 3:
                # not enough tokens -> stash as graph-level note
                # but don't abort; keep zero-loss by storing in graph attr once
                try:
                    notes = H.get_graph_attr("sif_ignored_lines") or []
                except Exception:
                    notes = []
                notes.append({"line": line_no, "content": raw.rstrip("\n\r")})
                try:
                    H.set_graph_attrs(sif_ignored_lines=notes)
                except Exception:
                    pass
                continue

            src, rel = toks[0], toks[1]
            # everything after the 2nd token are targets and/or extras
            tail = toks[2:]

            # extract trailing key=val pairs and numeric weight(s)
            extras: List[str] = []
            kvs: Dict[str, object] = {}
            numeric_weights: List[float] = []

            # Identify target tokens: up to the first token that looks like key=val.
            # If none are key=val, assume at least one target.
            first_kv_idx = None
            for i, t in enumerate(tail):
                if "=" in t:
                    first_kv_idx = i
                    break
            if first_kv_idx is None:
                # entire tail are targets/weight/extras
                targets = tail[:]
                trailing = []
            else:
                targets = tail[:first_kv_idx]
                trailing = tail[first_kv_idx:]

            # If strict-3 form, targets has exactly one item; ok.
            # Parse trailing key=val, and collect tokens that weren't k=v
            for t in trailing:
                k, v = _parse_kv(t)
                if k is not None:
                    kvs[k] = v
                else:
                    # maybe a lone numeric token intended as weight
                    try:
                        numeric_weights.append(float(t))
                    except Exception:
                        extras.append(t)

            # If there are still leftover tokens after we use the first target,
            # keep them in extras too (zero-loss), unless they were intended extra targets.
            if not targets:
                # malformed: no target; treat the first non-kv token as pseudo-target if exists
                if extras:
                    targets = [extras.pop(0)]
                else:
                    # give up on this line cleanly
                    try:
                        notes = H.get_graph_attr("sif_malformed") or []
                    except Exception:
                        notes = []
                    notes.append({"line": line_no, "content": raw.rstrip("\n\r")})
                    try:
                        H.set_graph_attrs(sif_malformed=notes)
                    except Exception:
                        pass
                    continue

            # numeric weight policy:
            # - if 'weight' key present in kvs -> that wins
            # - else if exactly one numeric token -> weight
            # - else ignore (but extras preserved)
            explicit_weight = kvs.get("weight", None)
            lone_weight = numeric_weights[0] if len(numeric_weights) == 1 else None

            # Build edges
            for i, tgt in enumerate(targets):
                # create vertices
                try:
                    H.add_vertex(src)
                except Exception:
                    pass
                try:
                    H.add_vertex(tgt)
                except Exception:
                    pass

                eid = str(kvs.get("id")) if ("id" in kvs) else f"sif::{line_no}#{i}"
                try:
                    H.add_edge(src, tgt, edge_id=eid, edge_directed=bool(directed))
                except Exception:
                    # fallback to directed True if API insists
                    H.add_edge(src, tgt, edge_id=eid, edge_directed=True)

                # weight
                w = explicit_weight if explicit_weight is not None else lone_weight
                if w is not None:
                    try:
                        H.edge_weights[eid] = float(w)
                    except Exception:
                        pass

                # attributes (relation + kvs + residues)
                attrs = {relation_attr: rel}
                # keep all kv pairs (minus 'id' if we used it as eid)
                for k, v in kvs.items():
                    if k == "id":
                        continue
                    attrs[k] = v
                if extras:
                    attrs["sif_extra"] = list(extras)
                try:
                    H.set_edge_attrs(eid, **attrs)
                except Exception:
                    pass

    return H
