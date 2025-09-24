import contextlib
import io
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np

from ..core._graph import Attr, BaseGraph, EdgeType

# tiny colormap & normalize utilities
def _normalize(values, lo=None, hi=None, eps=1e-12):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    if lo is None: lo = np.nanmin(arr)
    if hi is None: hi = np.nanmax(arr)
    if not np.isfinite(lo): lo = 0.0
    if not np.isfinite(hi): hi = 1.0
    denom = max(hi - lo, eps)
    return (arr - lo) / denom

def _greyscale(v):
    # v in [0,1] -> hex grey
    v = float(np.clip(v, 0.0, 1.0))
    c = int(round(v * 255))
    return f"#{c:02x}{c:02x}{c:02x}"


def suppress_repr_warnings(g):
    """Monkey-patch the _repr_* methods of an object instance (e.g. a graphviz.Digraph)
    so that any output written to stderr during their execution is suppressed.

    Parameters:
        g : object
            The instance whose _repr_* methods should have their warnings suppressed.
    """
    # Identify all representation methods (names starting with '_repr_') on the instance.
    repr_methods = [m for m in dir(g) if m.startswith("_repr_") and callable(getattr(g, m))]
    for method_name in repr_methods:
        original = getattr(g, method_name)

        def make_wrapper(orig_func):
            def wrapper(*args, **kwargs):
                with contextlib.redirect_stderr(io.StringIO()):
                    return orig_func(*args, **kwargs)

            return wrapper

        # Directly set the wrapped function on the instance
        setattr(g, method_name, make_wrapper(original))


def clip_quantiles(arr, q):
    if q < 0 or q > 1:
        raise ValueError(f"Clipping value must be between 0 and 1, got {q}")
    # compute the quantiles at clipping and 1-clipping and clip the flow
    q = np.quantile(arr, [q, 1 - q])
    return np.clip(arr, q[0], q[1])


def vertex_style(
    P,
    G: Optional[BaseGraph] = None,  # TODO: G should not be required
    vertex_var: str = "vertex_values",
    negative_color: str = "dodgerblue4",
    positive_color: str = "firebrick4",
    sample: Optional[int] = None,
):
    v_values = np.array(P.expr[vertex_var].value)
    if v_values is None:
        raise ValueError(
            f"""Variable {vertex_var} in the problem, but values are None.
            Has the problem been solved?"""
        )
    if sample is not None:
        if len(v_values.shape) < 2:
            raise ValueError(
                f"""Variable {vertex_var} in the problem is not a matrix.
                Cannot select sample {sample}"""
            )
        v_values = v_values[:, sample]
    if G is None:
        vertices = list(range(len(v_values)))
    else:
        vertices = G.V
    return create_graphviz_vertex_attributes(
        vertices, v_values, negative_color=negative_color, positive_color=positive_color
    )


def edge_style(
    P,
    max_edge_width: float = 5,
    min_edge_width: float = 0.25,
    edge_var: str = "edge_values",
    negative_color: str = "dodgerblue4",
    positive_color: str = "firebrick4",
    sample: Optional[int] = None,
):
    e_values = P.expr[edge_var].value
    if e_values is None:
        raise ValueError(
            f"""Variable {edge_var} in the problem, but values are None.
            Has the problem been solved?"""
        )
    if sample is not None:
        if len(e_values.shape) < 2:
            raise ValueError(
                f"""Variable {edge_var} in the problem is not a matrix.
                Cannot select sample {sample}"""
            )
        e_values = e_values[:, sample]
    return create_graphviz_edge_attributes(
        e_values,
        max_edge_width=max_edge_width,
        min_edge_width=min_edge_width,
        negative_color=negative_color,
        positive_color=positive_color,
    )


def create_graphviz_edge_attributes(
    edge_values: Union[List, np.ndarray],
    max_edge_width: float = 5,
    min_edge_width: float = 0.25,
    negative_color: str = "dodgerblue4",
    positive_color: str = "firebrick4",
):
    e_values = np.array(edge_values)
    edge_attrs = dict()
    for i, v in enumerate(e_values):
        if abs(v) > 0:
            edge_width = max_edge_width
        else:
            edge_width = min_edge_width
        edge_attrs[i] = {"penwidth": str(edge_width)}
        if e_values[i] > 0:
            edge_attrs[i]["color"] = positive_color
        elif e_values[i] < 0:
            edge_attrs[i]["color"] = negative_color
        else:
            edge_attrs[i]["color"] = "black"
    return edge_attrs


def create_graphviz_vertex_attributes(
    graph_vertices: List,
    vertex_values: Union[List, np.ndarray],
    negative_color: str = "dodgerblue4",
    positive_color: str = "firebrick4",
):
    v_values = np.array(vertex_values)
    if len(v_values) != len(graph_vertices):
        raise ValueError(
            f"""Length of vertex_values ({len(v_values)}) does not match the number
            of vertices ({len(graph_vertices)})"""
        )
    vertex_attrs = dict()
    for vn, v in zip(graph_vertices, v_values):
        vertex_attrs[vn] = dict()
        if v > 0:
            vertex_attrs[vn]["color"] = positive_color
            vertex_attrs[vn]["penwidth"] = "2"
        elif v < 0:
            vertex_attrs[vn]["color"] = negative_color
            vertex_attrs[vn]["penwidth"] = "2"
    return vertex_attrs


def flow_style(
    P,
    max_edge_width: float = 5,
    min_edge_width: float = 0.25,
    flow_name: str = "flow",
    positive_color: str = "dodgerblue4",
    negative_color: str = "firebrick4",
    zero_flow_threshold: float = 1e-6,
    scale: Optional[Literal["log", "std"]] = "log",
    clip_quantil: Optional[float] = 0.05,
):
    flow = np.array(P.expr[flow_name].value)
    flow[np.abs(flow) < zero_flow_threshold] = 0
    if scale is not None:
        if scale == "log":
            flow = np.log10(np.abs(flow) + 1e-6) * np.sign(flow)
        elif scale == "std":
            flow = flow / np.std(flow)
        else:
            raise ValueError(f"Unknown normalization method: {scale}")
    if clip_quantil is not None:
        flow = clip_quantiles(flow, clip_quantil)
    max_flow = max(np.max(np.abs(flow)), 1e-6)
    edge_attrs = dict()
    for i, v in enumerate(flow):
        # Apply threshold edge width
        if abs(v) > 0:
            edge_width = max_edge_width
        else:
            edge_width = min_edge_width
        if scale is not None:
            edge_width = min_edge_width + (max_edge_width - min_edge_width) * abs(v / max_flow)
        edge_attrs[i] = {"penwidth": str(edge_width)}
        if flow[i] > 0:
            edge_attrs[i]["color"] = positive_color
        elif flow[i] < 0:
            edge_attrs[i]["color"] = negative_color
        else:
            edge_attrs[i]["color"] = "black"
    return edge_attrs


def _create_vertices(g, e, vertex_props=None):
    if vertex_props is None:
        vertex_props = {}
    v_s, v_t = [], []
    i, (s, t) = e

    # Function to update node properties with user-provided properties
    def update_node_props(v_name, default_shape):
        # Merge default shape with user provided properties, if any
        props = {"shape": default_shape}
        if v_name in vertex_props:
            props.update(vertex_props[v_name])
        g.node(v_name, **props)

    if len(s) == 0:
        v_name = f"e_{i}_source"
        update_node_props(v_name, "point")  # Updated to use helper function
        v_s.append(v_name)

    if len(t) == 0:
        v_name = f"e_{i}_target"
        update_node_props(v_name, "point")  # Updated to use helper function
        v_t.append(v_name)

    for v in s:
        v_name = str(v)
        v_s.append(v_name)
        update_node_props(v_name, "circle")  # Updated to use helper function

    for v in t:
        v_name = str(v)
        v_t.append(v_name)
        update_node_props(v_name, "circle")  # Updated to use helper function

    return v_s, v_t


def to_python_graphviz(
    graph: BaseGraph,
    graph_attr: Optional[Dict[str, str]] = None,
    node_attr: Optional[Dict[str, str]] = None,
    edge_attr: Optional[Dict[str, str]] = None,
    custom_edge_attr: Optional[Dict[int, Dict[str, str]]] = None,
    custom_vertex_attr: Optional[Dict[Union[int, str], Dict[str, str]]] = None,
    edge_indexes: Optional[List[int]] = None,
    layout: str = "dot",
    orphan_edges: bool = True,
    supress_warnings: bool = True,
) -> Any:
    import graphviz  # type: ignore

    is_hypergraph = False
    if custom_edge_attr is None:
        custom_edge_attr = {}
    if custom_vertex_attr is None:
        custom_vertex_attr = {}
    if len(custom_vertex_attr) > 0:
        keys = list(custom_vertex_attr.keys())
        if all(isinstance(k, int) for k in keys):
            vertices = graph.V
            custom_vertex_attr = {str(v): custom_vertex_attr[i] for i, v in enumerate(vertices)}
    if node_attr is None:
        node_attr = dict(fixedsize="true")
    g = graphviz.Digraph(engine=layout, graph_attr=graph_attr, node_attr=node_attr, edge_attr=edge_attr)
    for i, e in enumerate(graph.edges()):
        if edge_indexes is not None and i not in edge_indexes:
            continue
        i, (s, t) = e
        if not orphan_edges and (len(s) == 0 or len(t) == 0):
            continue
        v_s, v_t = _create_vertices(g, e, vertex_props=custom_vertex_attr)
        if len(s) > 1 or len(t) > 1:
            is_hypergraph = True
            edge_center = f"e_{i}_center"
            g.node(edge_center, shape="square", width="0.1", height="0.1", label="")
            for v in v_s:
                e_attr = dict(arrowtail="none", arrowhead="none", dir="both")
                e_attr.update(custom_edge_attr.get(i, {}))
                g.edge(v, edge_center, **e_attr)
            for v in v_t:
                e_attr = custom_edge_attr.get(i, {})
                g.edge(edge_center, v, **e_attr)
        else:
            if graph.get_attr_edge(i).get_attr(Attr.EDGE_TYPE, "") == EdgeType.UNDIRECTED.value:
                e_attr = dict(arrowhead="none", dir="none")
                e_attr.update(custom_edge_attr.get(i, {}))
                g.edge(v_s[0], v_t[0], **e_attr)
            else:
                head = "normal"
                if graph.get_attr_edge(i).get("interaction", 0) < 0:
                    head = "tee"
                e_attr = dict(arrowhead=head)
                e_attr.update(custom_edge_attr.get(i, {}))
                g.edge(v_s[0], v_t[0], **e_attr)
    if is_hypergraph and graph_attr is None:
        g.graph_attr["splines"] = "true"
    if supress_warnings:
        suppress_repr_warnings(g)
    return g


def _pydot_create_vertices(g, e, vertex_props=None):
    import pydot

    if vertex_props is None:
        vertex_props = {}
    v_s, v_t = [], []
    i, (s, t) = e

    def update_node_props(v_name, default_shape):
        props = {"shape": default_shape}
        if v_name in vertex_props:
            props.update(vertex_props[v_name])
        node = pydot.Node(v_name, **props)
        g.add_node(node)

    if len(s) == 0:
        v_name = f"e_{i}_source"
        update_node_props(v_name, "point")
        v_s.append(v_name)
    if len(t) == 0:
        v_name = f"e_{i}_target"
        update_node_props(v_name, "point")
        v_t.append(v_name)
    for v in s:
        v_name = str(v)
        v_s.append(v_name)
        update_node_props(v_name, "circle")
    for v in t:
        v_name = str(v)
        v_t.append(v_name)
        update_node_props(v_name, "circle")
    return v_s, v_t


def to_pydot(
    graph: BaseGraph,
    graph_attr: Optional[Dict[str, str]] = None,
    node_attr: Optional[Dict[str, str]] = None,
    edge_attr: Optional[Dict[str, str]] = None,
    custom_edge_attr: Optional[Dict[int, Dict[str, str]]] = None,
    custom_vertex_attr: Optional[Dict[Union[int, str], Dict[str, str]]] = None,
    edge_indexes: Optional[List[int]] = None,
    layout: str = "dot",
    orphan_edges: bool = True,
) -> Any:
    import pydot

    is_hypergraph = False
    if custom_edge_attr is None:
        custom_edge_attr = {}
    if custom_vertex_attr is None:
        custom_vertex_attr = {}
    if len(custom_vertex_attr) > 0:
        keys = list(custom_vertex_attr.keys())
        if all(isinstance(k, int) for k in keys):
            vertices = graph.V
            custom_vertex_attr = {str(v): custom_vertex_attr[i] for i, v in enumerate(vertices)}
    # Create a pydot.Dot graph
    g = pydot.Dot(graph_type="digraph", **(graph_attr if graph_attr else {}))
    if node_attr is not None:
        g.set_node_defaults(**node_attr)
    if edge_attr is not None:
        g.set_edge_defaults(**edge_attr)
    for i, e in enumerate(graph.edges()):
        if edge_indexes is not None and i not in edge_indexes:
            continue
        i, (s, t) = e
        if not orphan_edges and (len(s) == 0 or len(t) == 0):
            continue
        v_s, v_t = _pydot_create_vertices(g, e, vertex_props=custom_vertex_attr)
        if len(s) > 1 or len(t) > 1:
            is_hypergraph = True
            edge_center = f"e_{i}_center"
            center_node = pydot.Node(edge_center, shape="square", width="0.1", height="0.1", label="")
            g.add_node(center_node)
            for v in v_s:
                e_attr = dict(arrowtail="none", arrowhead="none", dir="both")
                e_attr.update(custom_edge_attr.get(i, {}))
                edge = pydot.Edge(v, edge_center, **e_attr)
                g.add_edge(edge)
            for v in v_t:
                e_attr = custom_edge_attr.get(i, {})
                edge = pydot.Edge(edge_center, v, **e_attr)
                g.add_edge(edge)
        else:
            if graph.get_attr_edge(i).get_attr(Attr.EDGE_TYPE, "") == EdgeType.UNDIRECTED.value:
                e_attr = dict(arrowhead="none", dir="none")
                e_attr.update(custom_edge_attr.get(i, {}))
                edge = pydot.Edge(v_s[0], v_t[0], **e_attr)
                g.add_edge(edge)
            else:
                head = "normal"
                if graph.get_attr_edge(i).get("interaction", 0) < 0:
                    head = "tee"
                e_attr = dict(arrowhead=head)
                e_attr.update(custom_edge_attr.get(i, {}))
                edge = pydot.Edge(v_s[0], v_t[0], **e_attr)
                g.add_edge(edge)
    if is_hypergraph and graph_attr is None:
        g.set_splines("true")
    return g


def to_graphviz(
    graph: BaseGraph,
    graph_attr: Optional[Dict[str, str]] = None,
    node_attr: Optional[Dict[str, str]] = None,
    edge_attr: Optional[Dict[str, str]] = None,
    custom_edge_attr: Optional[Dict[int, Dict[str, str]]] = None,
    custom_vertex_attr: Optional[Dict[Union[int, str], Dict[str, str]]] = None,
    edge_indexes: Optional[List[int]] = None,
    layout: str = "dot",
    orphan_edges: bool = True,
    supress_warnings: bool = True,
    backend: Literal["graphviz", "pydot"] = "graphviz",
) -> Any:
    """Generate and return a graph using the selected backend.
    backend:
       - 'graphviz' (default) returns a graphviz.Digraph object.
       - 'pydot' returns a pydot.Dot graph.
    """
    if backend == "graphviz":
        return to_python_graphviz(
            graph,
            graph_attr,
            node_attr,
            edge_attr,
            custom_edge_attr,
            custom_vertex_attr,
            edge_indexes,
            layout,
            orphan_edges,
            supress_warnings,
        )
    elif backend == "pydot":
        return to_pydot(
            graph,
            graph_attr,
            node_attr,
            edge_attr,
            custom_edge_attr,
            custom_vertex_attr,
            edge_indexes,
            layout,
            orphan_edges,
        )
    else:
        raise ValueError("Unknown backend specified. Must be 'graphviz' or 'pydot'.")
    
# pull effective (layer-aware) weights from an IncidenceAdapter
def edge_weights_for_layer(graph: BaseGraph, layer: Optional[str] = None):
    """
    Returns a dict {edge_index -> weight} using deep weights if available.
    If 'layer' is provided and backend supports layer overrides, use the effective per-layer weight.
    Fallback: 1.0 for edges missing weights.
    """
    w = {}
    deep = getattr(graph, "deep", None)
    for eidx, _ in graph.edges():
        if deep is None:
            w[eidx] = 1.0
            continue
        try:
            # prefer stable eid, then compute effective weight if layer provided
            eid = deep.idx_to_edge[eidx]
            if layer:
                # backend method should exist in your incidence graph
                w[eidx] = float(deep.get_effective_edge_weight(eid, layer=layer))
            else:
                w[eidx] = float(deep.edge_weights.get(eid, 1.0))
        except Exception:
            w[eidx] = 1.0
    return w

# label helpers (opt-in)
def build_edge_labels(graph: BaseGraph, use_weight=True, extra_keys=None, layer=None):
    """
    Returns {edge_index: 'label'} optionally including weight and selected attributes.
    """
    extra_keys = extra_keys or []
    labels = {}
    w = edge_weights_for_layer(graph, layer=layer) if use_weight else {}
    for eidx, _ in graph.edges():
        parts = []
        if use_weight and eidx in w:
            parts.append(f"w={w[eidx]:.3g}")
        if extra_keys:
            attrs = graph.get_attr_edge(eidx)
            for k in extra_keys:
                if k in attrs and attrs[k] is not None and not (isinstance(attrs[k], float) and np.isnan(attrs[k])):
                    parts.append(f"{k}={attrs[k]}")
        if parts:
            labels[eidx] = "\\n".join(parts)  # graphviz-friendly
    return labels

def build_vertex_labels(graph: BaseGraph, key=None):
    """
    If key is provided, use vertex attribute 'key' as label; else use vertex id.
    """
    labels = {}
    for v in graph.V:
        if key is None:
            labels[str(v)] = str(v)
        else:
            attrs = graph.get_attr_vertex(v)
            labels[str(v)] = str(attrs.get(key, v))
    return labels

# weight-driven edge styling (overrides create_graphviz_edge_attributes if you like)
def edge_style_from_weights(graph: BaseGraph,
                            layer: Optional[str] = None,
                            min_width: float = 0.5,
                            max_width: float = 5.0,
                            color_mode: str = "greys"):  # or "signed"
    """
    Returns {edge_index: {penwidth, color}} using (layer-aware) weights.
    'greys' maps |w| to grey intensity; 'signed' uses red/blue for +/-.
    """
    weights = edge_weights_for_layer(graph, layer=layer)
    if not weights:
        return {}
    eidxs = sorted(weights.keys())
    vals = np.array([abs(weights[i]) for i in eidxs], dtype=float)
    x = _normalize(vals)
    styles = {}
    for i, v, xv in zip(eidxs, vals, x):
        pen = min_width + xv * (max_width - min_width)
        if color_mode == "signed":
            raw = float(weights[i])
            color = "firebrick4" if raw > 0 else ("dodgerblue4" if raw < 0 else "black")
        else:
            color = _greyscale(1.0 - xv)  # heavier => darker
        styles[i] = {"penwidth": f"{pen:.3f}", "color": color}
    return styles

# one-call plotting API
def plot(graph: BaseGraph,
         backend: Literal["graphviz","pydot"] = "graphviz",
         layout: str = "dot",
         layer: Optional[str] = None,
         show_edge_labels: bool = False,
         edge_label_keys: Optional[List[str]] = None,
         show_vertex_labels: bool = True,
         vertex_label_key: Optional[str] = None,
         use_weight_style: bool = True,
         orphan_edges: bool = True,
         suppress_warnings: bool = True,
         **kwargs):
    """
    Return a graph object (graphviz.Digraph or pydot.Dot) ready to render().
    - layer: use layer-aware weights for styling if backend supports it.
    - show_edge_labels: overlay weight and/or selected attributes.
    - show_vertex_labels: render vertex ids or a vertex attribute as label.
    """
    # Build base styling
    custom_edge_attr = {}
    if use_weight_style:
        custom_edge_attr = edge_style_from_weights(graph, layer=layer)

    # Vertex labels (for graphviz: we set node labels via custom_vertex_attr)
    custom_vertex_attr = None
    if show_vertex_labels:
        vlabels = build_vertex_labels(graph, key=vertex_label_key)
        custom_vertex_attr = {k: {"label": v} for k, v in vlabels.items()}

    # Choose backend
    if backend == "graphviz":
        g = to_python_graphviz(
            graph,
            graph_attr=kwargs.get("graph_attr"),
            node_attr=kwargs.get("node_attr"),
            edge_attr=kwargs.get("edge_attr"),
            custom_edge_attr=custom_edge_attr,
            custom_vertex_attr=custom_vertex_attr,
            edge_indexes=kwargs.get("edge_indexes"),
            layout=layout,
            orphan_edges=orphan_edges,
            supress_warnings=suppress_warnings,  # preserve your param name
        )
        # Edge labels (graphviz attaches labels on edges directly)
        if show_edge_labels:
            elabels = build_edge_labels(graph, use_weight=True, extra_keys=edge_label_keys, layer=layer)
            # reapply labels to edges by creating parallel unlabeled edges is awkward;
            # here we add 'label' to custom_edge_attr and re-render quickly:
            for eidx, (s, t) in graph.edges():
                if eidx in elabels:
                    if eidx not in custom_edge_attr:
                        custom_edge_attr[eidx] = {}
                    custom_edge_attr[eidx]["label"] = elabels[eidx]
            # rebuild with labels (cheap for small graphs)
            g = to_python_graphviz(
                graph,
                graph_attr=kwargs.get("graph_attr"),
                node_attr=kwargs.get("node_attr"),
                edge_attr=kwargs.get("edge_attr"),
                custom_edge_attr=custom_edge_attr,
                custom_vertex_attr=custom_vertex_attr,
                edge_indexes=kwargs.get("edge_indexes"),
                layout=layout,
                orphan_edges=orphan_edges,
                supress_warnings=suppress_warnings,
            )
        return g

    elif backend == "pydot":
        g = to_pydot(
            graph,
            graph_attr=kwargs.get("graph_attr"),
            node_attr=kwargs.get("node_attr"),
            edge_attr=kwargs.get("edge_attr"),
            custom_edge_attr=custom_edge_attr,
            custom_vertex_attr=custom_vertex_attr,
            edge_indexes=kwargs.get("edge_indexes"),
            layout=layout,
            orphan_edges=orphan_edges,
        )
        # pydot edge labels
        if show_edge_labels:
            elabels = build_edge_labels(graph, use_weight=True, extra_keys=edge_label_keys, layer=layer)
            # mutate edges to add labels
            for eidx, (s, t) in graph.edges():
                if eidx in elabels:
                    # pydot uses string node names
                    sv = next(iter(s)) if len(s) else f"e_{eidx}_source"
                    tv = next(iter(t)) if len(t) else f"e_{eidx}_target"
                    # There may be multiple edges; simplest is to add another edge with label
                    import pydot
                    g.add_edge(pydot.Edge(str(sv), str(tv), label=elabels[eidx]))
        return g

    else:
        raise ValueError("backend must be 'graphviz' or 'pydot'")

# convenience renderer
def render(obj, path: str, format: str = "svg"):
    """
    Render a graphviz.Digraph or pydot.Dot to disk.
    Returns the output path.
    """
    kind = obj.__class__.__module__
    if "graphviz" in kind:
        # graphviz.Digraph
        return obj.render(path, format=format, cleanup=True)
    elif "pydot" in kind:
        # pydot requires write_* by format
        if format.lower() == "png":
            obj.write_png(path if path.lower().endswith(".png") else f"{path}.png")
            return path if path.lower().endswith(".png") else f"{path}.png"
        elif format.lower() in ("svg",):
            obj.write_svg(path if path.lower().endswith(".svg") else f"{path}.svg")
            return path if path.lower().endswith(".svg") else f"{path}.svg"
        else:
            obj.write_raw(path)
            return path
    else:
        raise TypeError("Unknown graph object; expected graphviz.Digraph or pydot.Dot")












