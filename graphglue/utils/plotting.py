from __future__ import annotations

import contextlib
import io
import math
from collections.abc import Iterable
from typing import Any, Literal

import numpy as np

# Small helpers


def _normalize(
    values: Iterable[float], lo: float | None = None, hi: float | None = None, eps: float = 1e-12
):
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return arr
    if lo is None:
        lo = np.nanmin(arr)
    if hi is None:
        hi = np.nanmax(arr)
    if not math.isfinite(lo):
        lo = 0.0
    if not math.isfinite(hi):
        hi = 1.0
    denom = max(hi - lo, eps)
    return (arr - lo) / denom


def _greyscale(v: float) -> str:
    v = float(np.clip(v, 0.0, 1.0))
    c = int(round(v * 255))
    return f"#{c:02x}{c:02x}{c:02x}"


def _suppress_repr_warnings(g: Any) -> None:
    """Monkey-patch _repr_* methods to hide stderr noise from visualization libs."""
    repr_methods = [m for m in dir(g) if m.startswith("_repr_") and callable(getattr(g, m))]
    for method_name in repr_methods:
        original = getattr(g, method_name)

        def make_wrapper(orig_func):
            def wrapper(*args, **kwargs):
                with contextlib.redirect_stderr(io.StringIO()):
                    return orig_func(*args, **kwargs)

            return wrapper

        setattr(g, method_name, make_wrapper(original))


# Label builders


def build_vertex_labels(graph, key: str | None = None) -> dict[str, str]:
    labels: dict[str, str] = {}
    for vid in graph.vertices():
        if key is None:
            labels[vid] = str(vid)
        else:
            labels[vid] = str(graph.get_attr_vertex(vid, key, default=vid))
    return labels


def build_edge_labels(
    graph,
    *,
    use_weight: bool = True,
    extra_keys: list[str] | None = None,
    layer: str | None = None,
) -> dict[int, str]:
    extra_keys = extra_keys or []
    labels: dict[int, str] = {}
    for j in range(graph.number_of_edges()):
        eid = graph.idx_to_edge[j]
        parts: list[str] = []
        if use_weight:
            try:
                w = graph.get_effective_edge_weight(eid, layer=layer)
                parts.append(f"w={w:.3g}")
            except Exception:
                pass
        for k in extra_keys:
            v = graph.get_attr_edge(eid, k, default=None)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                parts.append(f"{k}={v}")
        if parts:
            labels[j] = "\\n".join(parts)
    return labels


# Edge style from weights


def edge_style_from_weights(
    graph,
    *,
    layer: str | None = None,
    min_width: float = 0.5,
    max_width: float = 5.0,
    color_mode: Literal["greys", "signed"] = "greys",
) -> dict[int, dict[str, str]]:
    """Compute visual edge styles (pen width and color) from effective weights.

    Parameters
    ----------
    graph : object
        Graph-like object exposing `number_of_edges()`, `idx_to_edge`, and
        `get_effective_edge_weight(eid, layer)` methods.
    layer : str, optional
        Layer name for retrieving edge weights. Defaults to `None`, which uses global weights.
    min_width : float, optional
        Minimum line width for edges. Default is 0.5.
    max_width : float, optional
        Maximum line width for edges. Default is 5.0.
    color_mode : {'greys', 'signed'}, optional
        Edge coloring mode:
        - ``'greys'`` : map absolute weight to grayscale (darker = heavier)
        - ``'signed'`` : use red for positive, blue for negative, black for zero

    Returns
    -------
    dict[int, dict[str, str]]
        A mapping from edge index to a style dict with keys:
        - ``penwidth`` : stroke width (stringified float)
        - ``color`` : color name or hex code

    Notes
    -----
    - Invalid or missing weights default to 1.0.
    - Normalization is performed across all edges in the graph.

    """
    eidxs = list(range(graph.number_of_edges()))
    raw_vals: list[float] = []
    for j in eidxs:
        eid = graph.idx_to_edge[j]
        try:
            raw_vals.append(abs(float(graph.get_effective_edge_weight(eid, layer=layer))))
        except Exception:
            raw_vals.append(1.0)

    x = _normalize(raw_vals)
    styles: dict[int, dict[str, str]] = {}
    for j, xv in zip(eidxs, x):
        pen = min_width + float(xv) * (max_width - min_width)
        if color_mode == "signed":
            eid = graph.idx_to_edge[j]
            try:
                w = float(graph.get_effective_edge_weight(eid, layer=layer))
            except Exception:
                w = 0.0
            color = "firebrick4" if w > 0 else ("dodgerblue4" if w < 0 else "black")
        else:
            color = _greyscale(1.0 - float(xv))  # heavier => darker
        styles[j] = {"penwidth": f"{pen:.3f}", "color": color}
    return styles


# Backends


def _add_nodes_graphviz(
    Gv, node_names: Iterable[str], custom_vertex_attr: dict[str, dict[str, str]] | None = None
):
    custom_vertex_attr = custom_vertex_attr or {}
    for v in node_names:
        attrs = {"shape": "circle"}
        attrs.update(custom_vertex_attr.get(v, {}))
        Gv.node(v, **attrs)


def _add_nodes_pydot(
    Gd, node_names: Iterable[str], custom_vertex_attr: dict[str, dict[str, str]] | None = None
):
    import pydot

    custom_vertex_attr = custom_vertex_attr or {}
    for v in node_names:
        attrs = {"shape": "circle"}
        attrs.update(custom_vertex_attr.get(v, {}))
        Gd.add_node(pydot.Node(v, **attrs))


def to_graphviz(
    graph,
    *,
    layout: str = "dot",
    graph_attr: dict[str, str] | None = None,
    node_attr: dict[str, str] | None = None,
    edge_attr: dict[str, str] | None = None,
    custom_edge_attr: dict[int, dict[str, str]] | None = None,
    custom_vertex_attr: dict[str, dict[str, str]] | None = None,
    edge_indexes: list[int] | None = None,
    orphan_edges: bool = True,
    suppress_warnings: bool = True,
):
    import graphviz

    Gv = graphviz.Digraph(
        engine=layout, graph_attr=graph_attr, node_attr=node_attr, edge_attr=edge_attr
    )

    # vertices to materialize (union of all endpoints)
    all_nodes: set[str] = set()
    edges_iter = range(graph.number_of_edges()) if edge_indexes is None else edge_indexes

    # First pass: collect nodes
    for j in edges_iter:
        S, T = graph.get_edge(j)
        if not orphan_edges and (len(S) == 0 or len(T) == 0):
            continue
        all_nodes.update(map(str, S | T))

    _add_nodes_graphviz(Gv, sorted(all_nodes), custom_vertex_attr)

    # Second pass: add edges
    for j in range(graph.number_of_edges()):
        if edge_indexes is not None and j not in edge_indexes:
            continue
        S, T = graph.get_edge(j)
        if not orphan_edges and (len(S) == 0 or len(T) == 0):
            continue

        # pick styling overrides
        e_attr = dict()
        if custom_edge_attr and j in custom_edge_attr:
            e_attr.update(custom_edge_attr[j])

        # Hyperedge if |S|>1 or |T|>1
        if len(S) > 1 or len(T) > 1:
            center = f"e_{j}_center"
            Gv.node(center, shape="square", width="0.1", height="0.1", label="")
            for u in S:
                a = {"arrowtail": "none", "arrowhead": "none", "dir": "both"}
                a.update(e_attr)
                Gv.edge(str(u), center, **a)
            for v in T:
                Gv.edge(center, str(v), **e_attr)
        else:
            # binary edge
            u = next(iter(S))
            v = next(iter(T))
            # Directed if S != T
            if S != T:
                head = "normal"
                # if a sign attribute exists, tee for negative interaction (optional)
                inter = graph.get_attr_edge(graph.idx_to_edge[j], "interaction", default=None)
                if isinstance(inter, (int, float)) and inter < 0:
                    head = "tee"
                a = {"arrowhead": head}
                a.update(e_attr)
                Gv.edge(str(u), str(v), **a)
            else:
                a = {"arrowhead": "none", "dir": "none"}
                a.update(e_attr)
                uu, vv = list(S)[0], list(T)[0]
                Gv.edge(str(uu), str(vv), **a)

    if suppress_warnings:
        _suppress_repr_warnings(Gv)
    if (
        len(
            [
                1
                for j in range(graph.number_of_edges())
                if len(graph.get_edge(j)[0]) > 1 or len(graph.get_edge(j)[1]) > 1
            ]
        )
        > 0
    ) and graph_attr is None:
        Gv.graph_attr["splines"] = "true"
    return Gv


def to_pydot(
    graph,
    *,
    layout: str = "dot",  # kept for API parity; pydot doesn't use engine here
    graph_attr: dict[str, str] | None = None,
    node_attr: dict[str, str] | None = None,
    edge_attr: dict[str, str] | None = None,
    custom_edge_attr: dict[int, dict[str, str]] | None = None,
    custom_vertex_attr: dict[str, dict[str, str]] | None = None,
    edge_indexes: list[int] | None = None,
    orphan_edges: bool = True,
):
    import pydot

    Gd = pydot.Dot(graph_type="digraph", **(graph_attr or {}))
    if node_attr:
        Gd.set_node_defaults(**node_attr)
    if edge_attr:
        Gd.set_edge_defaults(**edge_attr)

    all_nodes: set[str] = set()
    edges_iter = range(graph.number_of_edges()) if edge_indexes is None else edge_indexes
    for j in edges_iter:
        S, T = graph.get_edge(j)
        if not orphan_edges and (len(S) == 0 or len(T) == 0):
            continue
        all_nodes.update(map(str, S | T))

    _add_nodes_pydot(Gd, sorted(all_nodes), custom_vertex_attr)

    for j in range(graph.number_of_edges()):
        if edge_indexes is not None and j not in edge_indexes:
            continue
        S, T = graph.get_edge(j)
        if not orphan_edges and (len(S) == 0 or len(T) == 0):
            continue

        e_attr = dict()
        if custom_edge_attr and j in custom_edge_attr:
            e_attr.update(custom_edge_attr[j])

        if len(S) > 1 or len(T) > 1:
            center = f"e_{j}_center"
            Gd.add_node(pydot.Node(center, shape="square", width="0.1", height="0.1", label=""))
            for u in S:
                a = {"arrowtail": "none", "arrowhead": "none", "dir": "both"}
                a.update(e_attr)
                Gd.add_edge(pydot.Edge(str(u), center, **a))
            for v in T:
                Gd.add_edge(pydot.Edge(center, str(v), **e_attr))
        else:
            u = next(iter(S))
            v = next(iter(T))
            if S != T:
                head = "normal"
                inter = graph.get_attr_edge(graph.idx_to_edge[j], "interaction", default=None)
                if isinstance(inter, (int, float)) and inter < 0:
                    head = "tee"
                a = {"arrowhead": head}
                a.update(e_attr)
                Gd.add_edge(pydot.Edge(str(u), str(v), **a))
            else:
                a = {"arrowhead": "none", "dir": "none"}
                a.update(e_attr)
                uu, vv = list(S)[0], list(T)[0]
                Gd.add_edge(pydot.Edge(str(uu), str(vv), **a))

    if (
        len(
            [
                1
                for j in range(graph.number_of_edges())
                if len(graph.get_edge(j)[0]) > 1 or len(graph.get_edge(j)[1]) > 1
            ]
        )
        > 0
    ) and graph_attr is None:
        Gd.set_splines("true")
    return Gd


# One-call plotting API


def plot(
    graph,
    *,
    backend: Literal["graphviz", "pydot"] = "graphviz",
    layout: str = "dot",
    layer: str | None = None,
    show_edge_labels: bool = False,
    edge_label_keys: list[str] | None = None,
    show_vertex_labels: bool = True,
    vertex_label_key: str | None = None,
    use_weight_style: bool = True,
    orphan_edges: bool = True,
    suppress_warnings: bool = True,
    **kwargs,
):
    """Build a fully styled graph object ready for rendering with Graphviz or Pydot.

    Parameters
    ----------
    graph : object
        Graph-like object with `vertices()`, `get_edge()`, `get_attr_edge()`, etc.
    backend : {'graphviz', 'pydot'}, optional
        Visualization backend to use. Default is ``'graphviz'``.
    layout : str, optional
        Layout engine (e.g. ``'dot'``, ``'neato'``). Default is ``'dot'``.
    layer : str, optional
        Layer name for weight or label extraction. Default is `None`.
    show_edge_labels : bool, optional
        Whether to include weight and attribute labels on edges. Default is ``False``.
    edge_label_keys : list of str, optional
        Extra edge attribute keys to display if `show_edge_labels=True`.
    show_vertex_labels : bool, optional
        Whether to label vertices with IDs or attributes. Default is ``True``.
    vertex_label_key : str, optional
        Attribute key for vertex labels. If `None`, uses vertex IDs.
    use_weight_style : bool, optional
        Whether to style edges based on weights. Default is ``True``.
    orphan_edges : bool, optional
        Whether to render edges with missing endpoints. Default is ``True``.
    suppress_warnings : bool, optional
        Suppress backend rendering warnings (stderr). Default is ``True``.
    **kwargs
        Additional keyword arguments forwarded to `to_graphviz()` or `to_pydot()`.

    Returns
    -------
    graphviz.Digraph or pydot.Dot
        A styled, backend-specific graph object suitable for rendering or exporting.

    Raises
    ------
    ValueError
        If an invalid `backend` name is provided.

    Notes
    -----
    - Edge styling and labels are applied before backend construction.
    - If `show_edge_labels=True`, edges are regenerated with label overrides.

    """
    # edge styles
    custom_edge_attr: dict[int, dict[str, str]] = {}
    if use_weight_style:
        custom_edge_attr = edge_style_from_weights(graph, layer=layer)

    # vertex labels (set via custom_vertex_attr)
    custom_vertex_attr: dict[str, dict[str, str]] | None = None
    if show_vertex_labels:
        vlabels = build_vertex_labels(graph, key=vertex_label_key)
        custom_vertex_attr = {k: {"label": v} for k, v in vlabels.items()}

    if backend == "graphviz":
        G = to_graphviz(
            graph,
            layout=layout,
            graph_attr=kwargs.get("graph_attr"),
            node_attr=kwargs.get("node_attr", dict(fixedsize="true")),
            edge_attr=kwargs.get("edge_attr"),
            custom_edge_attr=custom_edge_attr,
            custom_vertex_attr=custom_vertex_attr,
            edge_indexes=kwargs.get("edge_indexes"),
            orphan_edges=orphan_edges,
            suppress_warnings=suppress_warnings,
        )
        if show_edge_labels:
            elabels = build_edge_labels(
                graph, use_weight=True, extra_keys=edge_label_keys, layer=layer
            )
            # reapply labels by regenerating with label overrides
            for j, txt in elabels.items():
                custom_edge_attr.setdefault(j, {})["label"] = txt
            G = to_graphviz(
                graph,
                layout=layout,
                graph_attr=kwargs.get("graph_attr"),
                node_attr=kwargs.get("node_attr", dict(fixedsize="true")),
                edge_attr=kwargs.get("edge_attr"),
                custom_edge_attr=custom_edge_attr,
                custom_vertex_attr=custom_vertex_attr,
                edge_indexes=kwargs.get("edge_indexes"),
                orphan_edges=orphan_edges,
                suppress_warnings=suppress_warnings,
            )
        return G

    elif backend == "pydot":
        G = to_pydot(
            graph,
            layout=layout,
            graph_attr=kwargs.get("graph_attr"),
            node_attr=kwargs.get("node_attr"),
            edge_attr=kwargs.get("edge_attr"),
            custom_edge_attr=custom_edge_attr,
            custom_vertex_attr=custom_vertex_attr,
            edge_indexes=kwargs.get("edge_indexes"),
            orphan_edges=orphan_edges,
        )
        if show_edge_labels:
            # mutate by adding parallel labeled edges
            import pydot

            elabels = build_edge_labels(
                graph, use_weight=True, extra_keys=edge_label_keys, layer=layer
            )
            for j, txt in elabels.items():
                S, T = graph.get_edge(j)
                sv = next(iter(S)) if len(S) else f"e_{j}_source"
                tv = next(iter(T)) if len(T) else f"e_{j}_target"
                G.add_edge(pydot.Edge(str(sv), str(tv), label=txt))
        return G

    else:
        raise ValueError("backend must be 'graphviz' or 'pydot'")


# Renderer


def render(obj: Any, path: str, format: str = "svg") -> str:
    """Render a Graphviz or Pydot graph object to disk and return the output path.

    Parameters
    ----------
    obj : graphviz.Digraph or pydot.Dot
        The graph object returned by `plot()`.
    path : str
        Destination file path (without extension if `format` is specified).
    format : {'svg', 'png', 'raw'}, optional
        Output format. Default is ``'svg'``. ``'png'`` and ``'raw'`` are supported for Pydot.

    Returns
    -------
    str
        Full path to the written output file.

    Raises
    ------
    TypeError
        If `obj` is not a supported graph type.

    Notes
    -----
    - Graphviz objects use the built-in `.render()` API.
    - Pydot objects write directly via `.write_svg()`, `.write_png()`, or `.write_raw()`.
    - The file extension is appended automatically if not present.

    """
    kind = obj.__class__.__module__
    fmt = format.lower()
    if "graphviz" in kind:
        return obj.render(path, format=fmt, cleanup=True)
    elif "pydot" in kind:
        if fmt == "png":
            obj.write_png(path if path.lower().endswith(".png") else f"{path}.png")
            return path if path.lower().endswith(".png") else f"{path}.png"
        elif fmt in ("svg",):
            obj.write_svg(path if path.lower().endswith(".svg") else f"{path}.svg")
            return path if path.lower().endswith(".svg") else f"{path}.svg"
        else:
            obj.write_raw(path)
            return path
    else:
        raise TypeError("Unknown graph object; expected graphviz.Digraph or pydot.Dot")
