from typing import Iterator, Optional
from ._base import BaseGraph, Edge, EdgeType, Attributes
from .layered_incidence_graph_pl import IncidenceGraph
import polars as pl
from ..utils.typing import Attr, Attributes
from .structure import EdgeType

def _safe_edge_attrs_dict(G: IncidenceGraph, eid: str) -> dict:
    try:
        if isinstance(G.edge_attributes, pl.DataFrame):
            rows = G.edge_attributes.filter(pl.col("edge_id") == eid)
            row = rows.to_dicts()[0] if rows.height else {}
            row.pop("edge_id", None)
    except Exception:
        row = {}
    return {k: v for k, v in row.items() if v is not None and v == v}



def _edge_layers(G: IncidenceGraph, eid: str) -> list[str]:
    return [lid for lid, ldata in G.layers(include_default=True).items() if eid in ldata["edges"]]


def _edge_endpoints(G: IncidenceGraph, eid: str) -> tuple:
    kind = G.edge_kind.get(eid)
    if kind == "hyper":
        meta = G.hyperedge_definitions[eid]
        if meta.get("directed", False):
            return tuple(meta["head"] | meta["tail"])
        return tuple(meta["members"])
    else:
        u, v, _ = G.edge_definitions[eid]
        return (u, v)


class IncidenceAdapter(BaseGraph):
    def __init__(self, include_edge_entities: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.G = IncidenceGraph()
        self._include_edge_entities = include_edge_entities
        self._v2i_cache = None

    # -------------------------
    # Core vertex/edge add
    # -------------------------

    def _add_vertex(self, v, **attr) -> int:
        self.G.add_node(v, **attr)
        self._invalidate_vertex_index()
        return self._vertex_index()[v]

    def _add_edge(
        self,
        source,
        target,
        type: EdgeType,
        edge_source_attr: Optional[Attributes] = None,  # ignored for DF writes (kept scalar-only)
        edge_target_attr: Optional[Attributes] = None,  # ignored for DF writes (kept scalar-only)
        **kwargs,
    ) -> int:
        directed = (type == EdgeType.DIRECTED)
        s = tuple(source)
        t = tuple(target)

        # Create edge / hyperedge in backend
        if len(s) <= 1 and len(t) <= 1:
            u = next(iter(s)) if s else None
            v = next(iter(t)) if t else None
            eid = self.G.add_edge(source=u, target=v, edge_directed=directed, **kwargs)
        else:
            if directed:
                eid = self.G.add_hyperedge(head=s, tail=t, edge_directed=True, **kwargs)
            else:
                members = tuple(set(s) | set(t))
                eid = self.G.add_hyperedge(members=members, edge_directed=False, **kwargs)

        # Persist attributes
        payload = {}

        # Store edge type as a scalar (string is safest); coerce to enum on read.
        payload["__edge_type"] = type.name

        # Optional weight (float) and label (str)
        if "weight" in kwargs and kwargs["weight"] is not None:
            payload["__weight"] = float(kwargs["weight"])
        if "label" in kwargs and kwargs["label"] is not None:
            payload["label"] = str(kwargs["label"])

        if payload:
            self.G.set_edge_attrs(eid, **payload)

        return self._eid_to_public_index(eid)


    # -------------------------
    # Index helpers
    # -------------------------

    def _vertex_index(self):
        if self._v2i_cache is None:
            self._v2i_cache = {v: i for i, v in enumerate(self._get_vertices())}
        return self._v2i_cache

    def _invalidate_vertex_index(self):
        self._v2i_cache = None

    def _eid_to_public_index(self, eid: str) -> int:
        return self.G.edge_to_idx[eid]

    def _index_to_eid(self, j: int) -> str:
        return self.G.idx_to_edge[j]

    # -------------------------
    # Required abstract methods
    # -------------------------

    def get_edge(self, index: int) -> Edge:
        G = self.G
        eid = G.idx_to_edge[index]
        kind = G.edge_kind.get(eid)
        if kind == "hyper":
            meta = G.hyperedge_definitions[eid]
            if meta.get("directed", False):
                return (frozenset(meta["head"]), frozenset(meta["tail"]))
            else:
                M = frozenset(meta["members"])
                return (M, M)
        else:
            u, v, _et = G.edge_definitions[eid]
            if G.edge_directed.get(eid, True):
                return (frozenset([u]), frozenset([v]))
            else:
                return (frozenset([u, v]), frozenset([u, v]))

    def _get_vertices(self):
        if self._include_edge_entities:
            return tuple(self.G.entity_to_idx.keys())
        return tuple(self.G.nodes())

    def _get_incident_edges(self, vertex) -> Iterator[int]:
        G = self.G
        try:
            ridx = G.entity_to_idx[vertex]
            row = G._matrix.tocsr().getrow(ridx)
            for j in row.indices:
                yield j
            return
        except Exception:
            pass
        for j in range(G.number_of_edges()):
            eid = G.idx_to_edge[j]
            kind = G.edge_kind.get(eid)
            if kind == "hyper":
                meta = G.hyperedge_definitions[eid]
                if (meta.get("directed", False) and (vertex in meta["head"] or vertex in meta["tail"])) \
                   or (not meta.get("directed", False) and vertex in meta["members"]):
                    yield j
            else:
                u, v, _ = G.edge_definitions[eid]
                if vertex == u or vertex == v:
                    yield j

    def _get_edge_attributes(self, e):
        eid = self._index_to_eid(e)
        return Attributes(_safe_edge_attrs_dict(self.G, eid))

    def _get_vertex_attributes(self, v):
        try:
            if isinstance(self.G.node_attributes, pl.DataFrame):
                rows = self.G.node_attributes.filter(pl.col("node_id") == v)
                row = rows.to_dicts()[0] if rows.height else {}
                row.pop("node_id", None)
            else:
                row = self.G.node_attributes.loc[v].to_dict()
        except Exception:
            row = {}
        clean = {k: val for k, val in row.items() if val is not None and val == val}
        return Attributes(clean)


    def get_graph_attributes(self):
        return Attributes(self.G.graph_attributes)

    def _num_vertices(self) -> int:
        return self.G.number_of_nodes()

    def _num_edges(self) -> int:
        return self.G.number_of_edges()

    def extract_subgraph(self, vertices=None, edges=None):
        H = self.__class__(include_edge_entities=self._include_edge_entities)
        G = self.G
        vs = set(vertices) if vertices is not None else set(G.nodes())
        for v in vs:
            H.G.add_node(v)
        ecols = edges if edges is not None else range(G.number_of_edges())
        for j in ecols:
            eid = G.idx_to_edge[j]
            kind = G.edge_kind.get(eid)
            w = G.edge_weights[eid]
            is_dir = G.edge_directed.get(eid, True)
            attrs = _safe_edge_attrs_dict(G, eid)
            member_layers = _edge_layers(G, eid)
            if kind == "hyper":
                meta = G.hyperedge_definitions[eid]
                if meta.get("directed", False):
                    head = tuple(meta["head"] & vs)
                    tail = tuple(meta["tail"] & vs)
                    if head and tail:
                        H.G.add_hyperedge(head=head, tail=tail, weight=w,
                                          edge_id=eid, edge_directed=True, **attrs)
                else:
                    members = tuple(meta["members"] & vs)
                    if len(members) >= 2:
                        H.G.add_hyperedge(members=members, weight=w,
                                          edge_id=eid, edge_directed=False, **attrs)
            else:
                u, v, _et = G.edge_definitions[eid]
                if u in vs and v in vs:
                    H.G.add_edge(u, v, weight=w, edge_id=eid,
                                 edge_directed=is_dir, **attrs)
            for L in member_layers:
                H.G._layers.setdefault(L, {"nodes": set(), "edges": set(), "attributes": {}})
                H.G._layers[L]["edges"].add(eid)
                H.G._layers[L]["nodes"].update(_edge_endpoints(G, eid))
                try:
                    lw = H.G.get_effective_edge_weight(eid, layer=L)
                    if lw != w:
                        H.G.set_layer_edge_weight(L, eid, lw)
                except Exception:
                    pass
        return H

    def reverse(self) -> "BaseGraph":
        R = self.__class__(include_edge_entities=self._include_edge_entities)
        G = self.G
        for v in G.nodes():
            R.G.add_node(v)
        for j in range(G.number_of_edges()):
            eid = G.idx_to_edge[j]
            kind = G.edge_kind.get(eid)
            w = G.edge_weights[eid]
            attrs = _safe_edge_attrs_dict(G, eid)
            if kind == "hyper":
                meta = G.hyperedge_definitions[eid]
                if meta.get("directed", False):
                    R.G.add_hyperedge(head=tuple(meta["tail"]), tail=tuple(meta["head"]),
                                      weight=w, edge_id=eid, edge_directed=True, **attrs)
                else:
                    R.G.add_hyperedge(members=tuple(meta["members"]), weight=w,
                                      edge_id=eid, edge_directed=False, **attrs)
            else:
                u, v, _et = G.edge_definitions[eid]
                is_dir = G.edge_directed.get(eid, True)
                if is_dir:
                    R.G.add_edge(v, u, weight=w, edge_id=eid, edge_directed=True, **attrs)
                else:
                    R.G.add_edge(u, v, weight=w, edge_id=eid, edge_directed=False, **attrs)
        for L, data in G.layers(include_default=True).items():
            R.G._layers.setdefault(L, {"nodes": set(), "edges": set(),
                                       "attributes": dict(data["attributes"])})
            R.G._layers[L]["nodes"].update(data["nodes"])
            R.G._layers[L]["edges"].update(data["edges"])
            for eid in data["edges"]:
                try:
                    lw = G.get_effective_edge_weight(eid, layer=L)
                    if lw != G.edge_weights[eid]:
                        R.G.set_layer_edge_weight(L, eid, lw)
                except Exception:
                    pass
        return R

    # -------------------------
    # Optional layer helpers
    # -------------------------

    def layers_containing_edge(self, edge_index: int, include_default: bool = False):
        eid = self._index_to_eid(edge_index)
        return self.G.edge_presence_across_layers(edge_id=eid, include_default=include_default)

    def effective_edge_weight(self, edge_index: int, layer: Optional[str] = None) -> float:
        eid = self._index_to_eid(edge_index)
        return self.G.get_effective_edge_weight(eid, layer=layer)
 
    @property
    def deep(self) -> IncidenceGraph:
        """Full access to the underlying layered incidence engine."""
        return self.G
 
    def neighbors_exclusive(self, v): #Current neighbors includes the vertex itself
        return [u for u in self.neighbors(v) if u != v]
 
    def _ensure_edge_type_attr(self, eid, etype: EdgeType = EdgeType.DIRECTED) -> None:
        """
        Ensure edge `eid` has the __edge_type attribute set in the Polars DF (dataframe).
        Stored as string (etype.name) for Polars, converted back to enum on read.
        """
        df = self.G.edge_attributes
        etype_str = etype.name  # e.g., "DIRECTED"

        # Make sure the column exists
        if "__edge_type" not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("__edge_type"))

        # Update if row exists, else append
        if df.filter(pl.col("edge_id") == eid).height:
            df = df.with_columns(
                pl.when(pl.col("edge_id") == eid)
                .then(pl.lit(etype_str))
                .otherwise(pl.col("__edge_type"))
                .alias("__edge_type")
            )
        else:
            row = pl.DataFrame({"edge_id": [eid], "__edge_type": [etype_str]})
            df = pl.concat([df, row], how="vertical_relaxed")

        self.G.edge_attributes = df