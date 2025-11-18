"""SBML (Systems Biology Markup Language) → Graph adapter
------------------------------------------------------
Targets the provided `Graph` API.

Two entry points:
  - from_sbml(path, graph=None, slice="default", preserve_stoichiometry=True)
  - from_cobra_model(model, graph=None, slice="default", preserve_stoichiometry=True)

If `Graph.set_hyperedge_coeffs(edge_id, coeffs: dict[str, float])` is not available,
stoichiometric coefficients are stored under an edge attribute `stoich` (lossy but usable).
"""

from __future__ import annotations

import types
import warnings
from collections.abc import Iterable, Sequence

import numpy as np

warnings.filterwarnings("ignore", message="Signature .*numpy.longdouble.*")

from ..core.graph import Graph

# ----------------------- utilities -----------------------


def _monkeypatch_set_hyperedge_coeffs(G) -> bool:
    """Add `set_hyperedge_coeffs(edge_id, coeffs)` to Graph instance if missing.
    Writes per-vertex coefficients into the incidence column (DOK [Dictionary Of Keys]).
    Returns True if patch was applied, False if already available.
    """
    if hasattr(G, "set_hyperedge_coeffs"):
        return False  # already there

    def set_hyperedge_coeffs(self, edge_id: str, coeffs: dict[str, float]) -> None:
        col = self.edge_to_idx[edge_id]
        for vid, coeff in coeffs.items():
            row = self.entity_to_idx[vid]
            self._matrix[row, col] = float(coeff)

    G.set_hyperedge_coeffs = types.MethodType(set_hyperedge_coeffs, G)  # type: ignore
    return True


def _ensure_vertices(G, vertices: Iterable[str], slice: str | None) -> None:
    # `add_vertices_bulk` exists and handles missing vertices efficiently.
    G.add_vertices_bulk(list(vertices), slice=slice)


BOUNDARY_SOURCE = "__BOUNDARY_SOURCE__"
BOUNDARY_SINK = "__BOUNDARY_SINK__"


def _ensure_boundary_vertices(G, slice: str):
    # idempotent – Graph.add_vertices_bulk ignores existing ids
    G.add_vertices_bulk([BOUNDARY_SOURCE, BOUNDARY_SINK], slice=slice)


def _graph_from_stoich(
    S: np.ndarray,
    metabolite_ids: Sequence[str],
    reaction_ids: Sequence[str],
    graph: Graph | None = None,
    *,
    slice: str = "default",
    preserve_stoichiometry: bool = True,
) -> Graph:
    if graph is None:
        if Graph is None:
            raise RuntimeError("Graph class not importable; pass `graph=` explicitly.")
        G = Graph(directed=True)
    else:
        G = graph

    # Ensure all species + boundary placeholders exist
    G.add_vertices_bulk(list(metabolite_ids), slice=slice)
    _ensure_boundary_vertices(G, slice)

    m, n = S.shape
    assert m == len(metabolite_ids)
    assert n == len(reaction_ids)

    # Optional: enable per-vertex coefficients
    if preserve_stoichiometry and not hasattr(G, "set_hyperedge_coeffs"):
        # fallback = store stoich dict as attribute later
        pass

    for j, eid in enumerate(reaction_ids):
        col = S[:, j]
        head = [metabolite_ids[i] for i, v in enumerate(col) if v > 0]  # products
        tail = [metabolite_ids[i] for i, v in enumerate(col) if v < 0]  # reactants

        if not head and not tail:
            # Truly empty column; ignore
            continue

        boundary = None
        coeffs = {metabolite_ids[i]: float(v) for i, v in enumerate(col) if v != 0.0}

        if not head:
            # sink: products empty → route to SINK on head side
            head = [BOUNDARY_SINK]
            boundary = ("sink", BOUNDARY_SINK)
            # keep column balanced if we write per-vertex coefficients
            sink_coeff = float(sum(-v for v in col if v < 0))  # sum of absolute reactants
            coeffs[BOUNDARY_SINK] = sink_coeff

        elif not tail:
            # source: reactants empty → route from SOURCE on tail side
            tail = [BOUNDARY_SOURCE]
            boundary = ("source", BOUNDARY_SOURCE)
            source_coeff = float(-sum(v for v in col if v > 0))  # negative sum of products
            coeffs[BOUNDARY_SOURCE] = source_coeff

        eid_added = G.add_hyperedge(
            head=head,
            tail=tail,
            slice=slice,
            edge_id=eid,
            edge_directed=True,
            weight=1.0,
        )

        # write exact coefficients if supported; else stash as attribute
        if preserve_stoichiometry and hasattr(G, "set_hyperedge_coeffs"):
            G.set_hyperedge_coeffs(eid_added, coeffs)  # you said you added this
        else:
            G.set_edge_attrs(eid_added, stoich=coeffs)

        # mark boundary reactions for easy filtering
        if boundary:
            kind, bnode = boundary
            G.set_edge_attrs(eid_added, is_boundary=True, boundary_kind=kind, boundary_node=bnode)

    return G


# ---------------- COBRA-based import ----------------


def from_cobra_model(
    model,
    graph: Graph | None = None,
    *,
    slice: str = "default",
    preserve_stoichiometry: bool = True,
) -> Graph:
    """Convert a COBRApy model to Graph. Requires cobra.util.array.create_stoichiometric_matrix.
    Edge attributes added: name, default_lb, default_ub, gpr (Gene-Protein-Reaction rule [GPR]).
    """
    try:
        from cobra.util.array import create_stoichiometric_matrix  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("COBRApy not installed (needed for stoichiometric matrix).") from e

    S = create_stoichiometric_matrix(model)
    rxn_ids = [rxn.id for rxn in model.reactions]
    met_ids = [met.id for met in model.metabolites]

    G = _graph_from_stoich(
        S, met_ids, rxn_ids, graph=graph, slice=slice, preserve_stoichiometry=preserve_stoichiometry
    )

    # Attach per-reaction metadata via set_edge_attrs (Graph API)
    for rxn in model.reactions:
        eid = rxn.id
        attrs = {
            "name": getattr(rxn, "name", None),
            "default_lb": getattr(rxn, "lower_bound", None),
            "default_ub": getattr(rxn, "upper_bound", None),
            "gpr": getattr(rxn, "gene_reaction_rule", None),
        }
        # drop Nones
        clean = {k: v for k, v in attrs.items() if v is not None}
        if clean:
            G.set_edge_attrs(eid, **clean)

    return G


def from_sbml(
    path: str,
    graph: Graph | None = None,
    *,
    slice: str = "default",
    preserve_stoichiometry: bool = True,
    quiet: bool = True,
) -> Graph:
    """Read SBML using COBRApy if available; falls back to python-libsbml (if you extend this file)."""
    try:
        from cobra.io import read_sbml_model  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("COBRApy not installed; install cobra to read SBML.") from e

    model = read_sbml_model(path)
    return from_cobra_model(
        model, graph=graph, slice=slice, preserve_stoichiometry=preserve_stoichiometry
    )
