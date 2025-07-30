from __future__ import annotations
from typing import Optional, Any

import polars as pl
from .structure import CORE_SCHEMA, SRC, DST, VERTICES, VID
from ._state import _State
from ..adapters import manager as _backend_manager

__all__ = [
    'Graph',
]


class Graph:
    """
    Core Graph class (Polars-backed) for structure and metadata.
    """

    def __init__(
        self,
        *,
        vertices: pl.DataFrame,
        edges: pl.DataFrame,
        directed: bool = False,
    ):
        self.vertices = vertices
        self.edges = edges
        self.directed = directed
        self._state = _State()

        if (self.edges[VERTICES].list.len() < 1).any():
            raise ValueError("Each edge must reference at least one vertex.")

    def __repr__(self) -> str:
        return f"<Graph | V={len(self.vertices)} · E={len(self.edges)} · directed={self.directed}>"

    def add_edge(self, src: VID, dst: VID, **kwargs) -> None:
        """
        Add a new edge to the graph.

        Parameters:
            src (VID): Source vertex ID.
            dst (VID): Destination vertex ID.
            **kwargs: Additional edge attributes.

        Raises:
            ValueError: If either vertex does not exist in the graph.
        """
        if src not in self.vertices[VID].to_list():
            raise ValueError(f"Source vertex {src} does not exist.")
        if dst not in self.vertices[VID].to_list():
            raise ValueError(f"Destination vertex {dst} does not exist.")

        # Build a row that matches the existing edge schema
        row = {col: None for col in self.edges.columns}
        row[VERTICES] = [src, dst]
        row[SRC] = src
        row[DST] = dst
        row.update(kwargs)

        new_edge = pl.DataFrame([row], schema=self.edges.schema)
        self.edges = pl.concat([self.edges, new_edge], how="vertical")
        self._state.version += 1

    def export(self, fmt: str = "networkx", **kwargs) -> Any:
        """
        Export the graph using the specified adapter.

        Parameters:
            fmt (str): Name of the adapter to use, e.g., "networkx".
            **kwargs: Additional arguments passed to the adapter.

        Returns:
            Any: Library-specific graph object.
        """
        adapter = _backend_manager.get_adapter(fmt)
        return adapter.export(self.vertices, self.edges, directed=self.directed, **kwargs)

    @property
    def nx(self) -> "BackendProxy":  # type: ignore
        """On-demand accessor for NetworkX algorithms.

        Examples
        --------
        >>> G.nx.degree_centrality()
        """
        return _backend_manager.get_proxy("networkx", self)
